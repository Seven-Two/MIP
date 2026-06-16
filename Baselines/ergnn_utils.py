import random
import torch
import torch.nn as nn
import copy

class MF_sampler(nn.Module):
    # sampler for ERGNN MF and MF*
    def __init__(self, plus):
        super().__init__()
        self.plus = plus

    def forward(self, ids_per_cls_train, budget, feats, reps, d):
        if self.plus:
            return self.sampling(ids_per_cls_train, budget, reps)
        else:
            return self.sampling(ids_per_cls_train, budget, feats)

    def sampling(self,ids_per_cls_train, budget, vecs):
        centers = [vecs[ids].mean(0) for ids in ids_per_cls_train]
        sim = [centers[i].view(1,-1).mm(vecs[ids_per_cls_train[i]].permute(1,0)).squeeze() for i in range(len(centers))]
        rank = [s.sort()[1].tolist() for s in sim]
        ids_selected = []
        for i,ids in enumerate(ids_per_cls_train):
            nearest = rank[i][0:min(budget, len(ids_per_cls_train[i]))]
            ids_selected.extend([ids[i] for i in nearest])
        return ids_selected


class CM_sampler(nn.Module):
    # sampler for ERGNN CM and CM*
    def __init__(self, plus):
        super().__init__()
        self.plus = plus

    def forward(self, ids_per_cls_train, budget, feats, reps, d, using_half=True):
        if self.plus:
            return self.sampling(ids_per_cls_train, budget, reps, d, using_half=using_half)
        else:
            return self.sampling(ids_per_cls_train, budget, feats, d, using_half=using_half)

    def sampling(self,ids_per_cls_train, budget, vecs, d, using_half=True):
        budget_dist_compute = 1000
        '''
        if using_half:
            vecs = vecs.half()
        '''
        vecs = vecs.half()
        ids_selected = []
        for i,ids in enumerate(ids_per_cls_train):
            other_cls_ids = list(range(len(ids_per_cls_train)))
            other_cls_ids.pop(i)
            ids_selected0 = ids_per_cls_train[i] if len(ids_per_cls_train[i]) < budget_dist_compute else random.choices(ids_per_cls_train[i], k=budget_dist_compute)

            dist = []
            vecs_0 = vecs[ids_selected0]
            for j in other_cls_ids:
                chosen_ids = random.choices(ids_per_cls_train[j], k=min(budget_dist_compute,len(ids_per_cls_train[j])))
                vecs_1 = vecs[chosen_ids]
                if len(chosen_ids) < 26 or len(ids_selected0) < 26:
                    # torch.cdist throws error for tensor smaller than 26
                    dist.append(torch.cdist(vecs_0.float(), vecs_1.float()).half())
                else:
                    dist.append(torch.cdist(vecs_0,vecs_1))

            #dist = [torch.cdist(vecs[ids_selected0], vecs[random.choices(ids_per_cls_train[j], k=min(budget_dist_compute,len(ids_per_cls_train[j])))]) for j in other_cls_ids]
            dist_ = torch.cat(dist,dim=-1) # include distance to all the other classes
            n_selected = (dist_<d).sum(dim=-1)
            rank = n_selected.sort()[1].tolist()
            current_ids_selected = rank[:budget]
            ids_selected.extend([ids_per_cls_train[i][j] for j in current_ids_selected])
        return ids_selected


class random_sampler(nn.Module):
    # sampler for ERGNN CM and CM*
    def __init__(self, plus):
        super().__init__()
        self.plus = plus

    def forward(self, ids_per_cls_train, budget, feats, reps, d):
        if self.plus:
            return self.sampling(ids_per_cls_train, budget, reps, d)
        else:
            return self.sampling(ids_per_cls_train, budget, feats, d)

    def sampling(self,ids_per_cls_train, budget, vecs, d):
        ids_selected = []
        for i,ids in enumerate(ids_per_cls_train):
            ids_selected.extend(random.sample(ids,min(budget,len(ids))))
        return ids_selected
class degree_based_sampler(nn.Module):
    # based on random walk sasmpler01, sample a subgraph based on the degrees of the neighbors
    def __init__(self, args):
        super().__init__()

    def forward(self, graph, center_node_budget, nei_budget, gnn, ids_per_cls, restart=0.0):
        center_nodes_selected = self.node_sampler(ids_per_cls, graph, center_node_budget)
        all_nodes_selected = self.nei_sampler(center_nodes_selected, graph, nei_budget)
        return center_nodes_selected, all_nodes_selected


    def node_sampler(self,ids_per_cls_train, graph, budget, max_ratio_per_cls = 1.0):
        store_ids = []
        for i, ids in enumerate(ids_per_cls_train):
            budget_ = min(budget, int(max_ratio_per_cls * len(ids))) if isinstance(budget, int) else int(
                budget * len(ids))
            store_ids.extend(random.sample(ids, budget_))
        return store_ids


    def nei_sampler(self, center_nodes_selected, graph, nei_budget):
        probs = graph.in_degrees().float()
        nodes_selected_current_hop = copy.deepcopy(center_nodes_selected)
        retained_nodes = copy.deepcopy(center_nodes_selected)
        for b in nei_budget:
            if b==0:
                continue
            # from 1-hop to len(nei_budget)-hop neighbors
            neighbors = list(set(graph.in_edges(nodes_selected_current_hop)[0].tolist()))
            # remove selected nodes
            for n in retained_nodes:
                neighbors.remove(n)
            if len(neighbors)==0:
                continue
            prob = probs[neighbors]
            sampled_neibs_ = torch.multinomial(prob, min(b, len(neighbors)), replacement=False).tolist()
            sampled_neibs = torch.tensor(neighbors)[sampled_neibs_] # map the ids to the original ones
            retained_nodes.extend(sampled_neibs.tolist())
        return list(set(retained_nodes))
