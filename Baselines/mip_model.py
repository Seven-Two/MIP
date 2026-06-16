import copy

import numpy as np
import dgl
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import ModuleList
from torch_geometric.nn.inits import glorot

from Backbones.gnns import SGC_Agg
from Baselines.grace import LogReg, ModelGrace, traingrace


class SimplePrompt(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.global_emb = nn.Parameter(torch.Tensor(1, in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.global_emb)

    def add(self, x: Tensor):
        prompt = self.global_emb.expand(x.size(0), -1)
        return x + prompt, x, prompt


class BilinearPrompt(nn.Module):
    """Low-rank bilinear cosine prompt matching."""

    def __init__(self, in_channels: int, p_num: int, rank: int):
        super().__init__()
        self.p_list = nn.Parameter(torch.Tensor(p_num, in_channels))
        self.w_p = nn.Parameter(torch.Tensor(in_channels, rank))
        self.w_q = nn.Parameter(torch.Tensor(in_channels, rank))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.p_list)
        glorot(self.w_p)
        glorot(self.w_q)

    def add(self, x: Tensor):
        x_proj = F.normalize(x @ self.w_p, dim=1)
        prompt_proj = F.normalize(self.p_list @ self.w_q, dim=1)
        score = x_proj @ prompt_proj.t()
        weight = F.softmax(score, dim=1)
        prompt = weight @ self.p_list
        return x + prompt, x, prompt


class NET(torch.nn.Module):
    def __init__(self, model, task_manager, args):
        super().__init__()
        self.task_manager = task_manager
        self.n_tasks = args.n_tasks
        self.model = model
        self.drop_edge = args.mip_args["pe"]
        self.drop_feature = args.mip_args["pf"]
        self.temp = args.mip_args["temp"]
        self.device = f"cuda:{args.gpu}"
        self.lamb = args.mip_args["lamb"]
        self.lamb2 = args.mip_args["reg"]
        self.smoothing_hops = int(args.mip_args.get("smooth_k", 3))
        self._infomax_graph = None
        self._infomax_graph_num_nodes = None

        num_prompt = int(args.mip_args["prompts"])
        prompt_rank = int(args.mip_args.get("rank", min(args.d_data, 32)))
        if num_prompt < 2:
            prompt = SimplePrompt(args.d_data).cuda()
        else:
            prompt = BilinearPrompt(args.d_data, num_prompt, prompt_rank).cuda()

        cls_head = LogReg(args.hidden, args.n_cls_per_task).cuda()
        self.classifications = ModuleList([copy.deepcopy(cls_head) for _ in range(args.n_tasks)])
        self.prompts = ModuleList([copy.deepcopy(prompt) for _ in range(args.n_tasks - 1)])
        self.optimizers = []
        for taskid in range(args.n_tasks):
            param_groups = [{"params": self.classifications[taskid].parameters()}]
            if taskid > 0:
                param_groups.append({"params": self.prompts[taskid - 1].parameters()})
            self.optimizers.append(torch.optim.Adam(param_groups, lr=args.lr, weight_decay=args.weight_decay))

        self.ce = torch.nn.functional.cross_entropy

    def getprototype(self, g, features, train_ids, k=3):
        g = addedges(g)
        smoothed = SGC_Agg(k=k)(g, features)
        degs = g.in_degrees().float().clamp(min=1)
        norm = torch.pow(degs, -0.5).to(smoothed.device).unsqueeze(1)
        smoothed = smoothed * norm
        return torch.mean(smoothed[train_ids], dim=0)

    def gettaskid(self, prototypes, g, features, task, test_ids, k=3):
        g = addedges(g)
        smoothed = SGC_Agg(k=k)(g, features)
        degs = g.in_degrees().float().clamp(min=1)
        norm = torch.pow(degs, -0.5).to(smoothed.device).unsqueeze(1)
        smoothed = smoothed * norm
        test_prototype = torch.mean(smoothed[test_ids], dim=0).cpu()
        dist = torch.norm(prototypes[0:task] - test_prototype, dim=1)
        _, taskid = torch.min(dist, dim=0)
        return taskid.numpy()

    def pretrain(self, args, g, features, batch_size=None):
        num_hidden = args.hidden
        num_proj_hidden = 2 * num_hidden
        gracemodel = ModelGrace(self.model, num_hidden, num_proj_hidden, tau=0.5).cuda()
        traingrace(
            gracemodel,
            g,
            features,
            batch_size,
            drop_edge_prob=self.drop_edge,
            drop_feature_prob=self.drop_feature,
        )

    def _laplacian_features(self, g, features):
        return SGC_Agg(k=self.smoothing_hops)(addedges(g), features)

    def _get_infomax_graph(self, num_nodes: int, device):
        if (
            self._infomax_graph is None
            or self._infomax_graph_num_nodes != num_nodes
            or self._infomax_graph.device != device
        ):
            node_ids = torch.arange(num_nodes, device=device)
            self._infomax_graph = dgl.graph((node_ids, node_ids), num_nodes=num_nodes, device=device)
            self._infomax_graph_num_nodes = num_nodes
        return self._infomax_graph

    def _infomax_loss(self, h: Tensor, prompt: Tensor):
        graph = self._get_infomax_graph(h.size(0), h.device).local_var()
        graph.ndata["h"] = F.normalize(h, dim=1)
        graph.ndata["prompt_pos"] = F.normalize(prompt, dim=1)
        # Follow the old tcpp/mip style: negatives are built by shuffling the
        # prompt assignments within the current batch, then reusing DGL edge ops.
        shuffled_prompt = prompt[torch.randperm(prompt.size(0), device=prompt.device)]
        graph.ndata["prompt_neg"] = F.normalize(shuffled_prompt, dim=1)

        def compute_similarity(edges):
            h_src = edges.src["h"]
            prompt_pos = edges.dst["prompt_pos"]
            prompt_neg = edges.dst["prompt_neg"]
            pos_sim = (h_src * prompt_pos).sum(dim=1) / self.temp
            neg_sim = (h_src * prompt_neg).sum(dim=1) / self.temp
            return {"pos_sim": pos_sim, "neg_sim": neg_sim}

        graph.apply_edges(compute_similarity)
        pos_logits = graph.edata["pos_sim"]
        neg_logits = graph.edata["neg_sim"]
        return -F.logsigmoid(pos_logits).mean() - F.logsigmoid(-neg_logits).mean()

    def _prompt_reg_loss(self, prompt: Tensor):
        return torch.mean(torch.sum(prompt.pow(2), dim=1))

    def observe_il(self, *args):
        if len(args) == 8:
            g, features, labels, t, train_ids, ids_per_cls, offset1, _ = args
        elif len(args) == 11:
            _, g, features, labels, t, train_ids, _, _, ids_per_cls, offset1, _ = args
        else:
            raise TypeError(f"unexpected observe_il args length: {len(args)}")

        if t > 0:
            self.model.eval()

        labels = labels - offset1
        cls_head = self.classifications[t]
        cls_head.train()
        cls_head.zero_grad()
        optimizer_t = self.optimizers[t]

        if t > 0:
            prompt_t = self.prompts[t - 1]
            prompt_t.train()
            prompt_t.zero_grad()
            features_aug, _, prompt_vectors = prompt_t.add(features)
            output = self.model(g, features_aug)
            smooth_features = self._laplacian_features(g, features)
            im_loss = self._infomax_loss(smooth_features[train_ids], prompt_vectors[train_ids])
            reg_loss = self._prompt_reg_loss(prompt_vectors[train_ids])
        else:
            output = self.model(g, features)
            im_loss = 0.0
            reg_loss = 0.0

        output = cls_head(output)
        loss = self.ce(output[train_ids], labels[train_ids])
        if t > 0:
            loss = loss + self.lamb * im_loss + self.lamb2 * reg_loss

        loss.backward()
        optimizer_t.step()

    def getpred(self, g, features, taskid, test_ids, labels):
        self.model.eval()
        if taskid == 0:
            output = self.model(g, features)
        else:
            prompt_t = self.prompts[taskid - 1]
            features, _, _ = prompt_t.add(features)
            output = self.model(g, features)
        cls_head = self.classifications[taskid]
        return cls_head(output)


def addedges(subgraph):
    subgraph = copy.deepcopy(subgraph)
    nodedegree = subgraph.in_degrees().cpu()
    isolated_nodes = torch.where(nodedegree == 1)[0]
    connected_nodes = torch.where(nodedegree != 1)[0]
    isolated_nodes = isolated_nodes.numpy()
    connected_nodes = connected_nodes.numpy()
    randomnode = np.random.choice(connected_nodes, isolated_nodes.shape[0])
    srcs = np.concatenate([isolated_nodes, randomnode])
    dsts = np.concatenate([randomnode, isolated_nodes])
    subgraph.add_edges(srcs, dsts)
    return subgraph
