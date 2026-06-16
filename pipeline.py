import os
import pickle
import numpy as np
import torch
from Backbones.model_factory import get_model
from Backbones.utils import evaluatewp, NodeLevelDataset, evaluate, evaluate_batch, evaluate_taskIL
from training.utils import mkdir_if_missing
from dataset.utils import semi_task_manager
import importlib
import copy
import dgl
import time

PROMPT_METHODS = {"tpp", "tcpp", "mip"}

def get_pipeline(args):
    # choose the pipeline for the chosen setting
    if args.minibatch:
        if args.ILmode == 'classIL':
            # if args.inter_task_edges:
            #     if args.method in joint_alias:
            #         return pipeline_class_IL_inter_edge_minibatch_joint
            #     else:
            #         return pipeline_class_IL_inter_edge_minibatch
            # else:
            #     if args.method in joint_alias:
            #         return pipeline_class_IL_no_inter_edge_minibatch_joint
            #     else:
            return pipeline_class_IL_no_inter_edge_minibatch
        elif args.ILmode == 'taskIL':
            # if args.inter_task_edges:
            #     if args.method in joint_alias:
            #         return pipeline_task_IL_inter_edge_minibatch_joint
            #     else:
            #         return pipeline_task_IL_inter_edge_minibatch
            # else:
            #     if args.method in joint_alias:
            #         return pipeline_task_IL_no_inter_edge_minibatch_joint
            #     else:
            # return pipeline_task_IL_no_inter_edge_minibatch
            pass
    else:
        if args.ILmode == 'classIL':
            # if args.inter_task_edges:
            #     if args.method in joint_alias:
            #         return pipeline_class_IL_inter_edge_joint
            #     else:
            #         return pipeline_class_IL_inter_edge
            # else:
            #     if args.method in joint_alias:
            #         return pipeline_class_IL_no_inter_edge_joint
            #     else:
            return pipeline_class_IL_no_inter_edge
        elif args.ILmode == 'taskIL':
            # if args.inter_task_edges:
            #     if args.method in joint_alias:
            #         return pipeline_task_IL_inter_edge_joint
            #     else:
            #         return pipeline_task_IL_inter_edge
            # else:
            #     if args.method in joint_alias:
            #         return pipeline_task_IL_no_inter_edge_joint
            #     else:
            return pipeline_task_IL_no_inter_edge




def data_prepare(args, dataset):
    torch.cuda.set_device(args.gpu)
    n_cls_so_far = 0
    str_int_tsk = 'inter_tsk_edge' if args.inter_task_edges else 'no_inter_tsk_edge'
    # 从task_seq从后往前取

    for task, task_cls in enumerate(args.task_seq[::-1]):
        n_cls_so_far += len(task_cls)
        try:
            if args.load_check:
                subgraph, ids_per_cls, [train_ids, valid_ids, test_ids] = pickle.load(open(f'{args.data_path}/{str_int_tsk}/{args.dataset}_{task_cls}.pkl', 'rb'))
            else:
                if f'{args.dataset}_{task_cls}.pkl' not in os.listdir(f'{args.data_path}/{str_int_tsk}'):
                    subgraph, ids_per_cls, [train_ids, valid_ids, test_ids] = pickle.load(open(f'{args.data_path}/{str_int_tsk}/{args.dataset}_{task_cls}.pkl', 'rb'))
        except:
            print(f'preparing data for task {task}')
            if args.inter_task_edges:
                mkdir_if_missing(f'{args.data_path}/inter_tsk_edge')
                cls_retain = []
                for clss in args.task_seq[0:task + 1]:
                    cls_retain.extend(clss)
                subgraph, ids_per_cls_all, [train_ids, valid_ids, test_ids] = dataset.get_graph(tasks_to_retain=cls_retain)
                with open(f'{args.data_path}/inter_tsk_edge/{args.dataset}_{task_cls}.pkl', 'wb') as f:
                    pickle.dump([subgraph, ids_per_cls_all, [train_ids, valid_ids, test_ids]], f)
            else:
                mkdir_if_missing(f'{args.data_path}/no_inter_tsk_edge')
                subgraph, ids_per_cls, [train_ids, valid_ids, test_ids] = dataset.get_graph(tasks_to_retain=task_cls)
                with open(f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{task_cls}.pkl', 'wb') as f:
                    pickle.dump([subgraph, ids_per_cls, [train_ids, valid_ids, test_ids]], f)


def pipeline_task_IL_no_inter_edge(args, valid=False):
    epochs = args.epochs if valid else 0  # training epochs is zero for testing mode
    torch.cuda.set_device(args.gpu) if args.device == 'cuda' else None
    dataset = NodeLevelDataset(args.dataset, ratio_valid_test=args.ratio_valid_test, args=args)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls - 1,
                                                                  args.n_cls_per_task)]  # this line will remove the final task if only one class included
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)
    task_manager = semi_task_manager()
    model = get_model(dataset, args).cuda(args.gpu) if args.device == 'cuda' else get_model(dataset, args).cpu()
    life_model = importlib.import_module(f'Baselines.{args.method}_model')
    life_model_ins = life_model.NET(model, task_manager, args) if valid else None
    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    if args.method in PROMPT_METHODS:
        prototypes = torch.zeros(args.n_tasks, args.d_data)
    meanas = []
    prev_model = None
    data_prepare(args, dataset)
    n_cls_so_far = 0
    for task, task_cls in enumerate(args.task_seq):
        name, ite = args.current_model_save_path
        config_name = name.split('/')[-1]
        subfolder_c = name.split(config_name)[-2]
        save_model_name = f'{config_name}_{ite}_{task_cls}'
        save_model_path = f'{args.result_path}/{subfolder_c}val_models/{save_model_name}.pkl'
        if args.method in PROMPT_METHODS:
            save_proto_name = save_model_name + '_prototypes'
            save_proto_path = f'{args.result_path}/{subfolder_c}val_models/{save_proto_name}.pkl'

        if not valid:

            model = pickle.load(open(save_model_path, 'rb')).cuda(args.gpu)
            if args.method in PROMPT_METHODS:
                life_model_ins = pickle.load(open(save_model_path, 'rb')).cuda(args.gpu)
                prototypes = pickle.load(open(save_proto_path, 'rb'))


        n_cls_so_far += len(task_cls)
        subgraph, ids_per_cls, [train_ids, valid_ids, test_ids] = pickle.load(open(
            f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{task_cls}.pkl', 'rb'))

        subgraph = subgraph.to(device='cuda:{}'.format(args.gpu)) if args.device == 'cuda' else subgraph.to(
            device='cpu')
        features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
        task_manager.add_task(task, n_cls_so_far)

        label_offset1 = task_manager.get_label_offset(task-1)[1]

        for epoch in range(epochs):
            if args.method in PROMPT_METHODS:
                life_model_ins.observe_il(subgraph, features, labels, task, train_ids, ids_per_cls, label_offset1,
                                          dataset)
            elif args.method == 'lwf':
                life_model_ins.observe_task_IL(args, subgraph, features, labels, task, prev_model, train_ids,
                                               ids_per_cls, dataset)
            elif args.method == 'puma':
                life_model_ins.observe(args, subgraph, features, labels, task, train_ids, test_ids,
                                          ids_per_cls, dataset)
            else:
                life_model_ins.observe_task_IL(args, subgraph, features, labels, task, train_ids, ids_per_cls, dataset)
        if not valid:
            try:
                model = pickle.load(open(save_model_path, 'rb')).cuda(
                    args.gpu) if args.device == 'cuda' else pickle.load(open(save_model_path, 'rb')).cpu()
            except:
                model.load_state_dict(torch.load(save_model_path.replace('.pkl', '.pt')))

        if valid and (args.method in PROMPT_METHODS):
            prototypes[task] = life_model_ins.getprototype(subgraph, features, train_ids)
        acc_mean = []
        for t in range(task + 1):
            subgraph, ids_per_cls, [train_ids, valid_ids_, test_ids_] = pickle.load(open(
                f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{args.task_seq[t]}.pkl', 'rb'))
            test_ids = valid_ids_ if valid else test_ids_  # whether use validation or test set
            subgraph = subgraph.to(device='cuda:{}'.format(args.gpu)) if args.device == 'cuda' else subgraph.to(
                device='cpu')
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls]
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            if args.method in PROMPT_METHODS:
                taskid = t
                label_offset1, label_offset2 = task_manager.get_label_offset(int(taskid) - 1)[1], task_manager.get_label_offset(int(taskid))[1]
                labels = labels - label_offset1
                output = life_model_ins.getpred(subgraph, features, taskid, test_ids, labels)
                acc = evaluatewp(output, labels, test_ids, cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            else:

                label_offset1, label_offset2 = task_manager.get_label_offset(t - 1)[1], task_manager.get_label_offset(t)[1]
                labels = labels - label_offset1

                if args.classifier_increase:
                    acc = evaluate_taskIL(model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                                   cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
                else:
                    # deprecated
                    acc = evaluate_taskIL(model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                                   cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            acc_matrix[task][t] = round(acc * 100, 2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc * 100:.2f}|", end="")

        accs = acc_mean[:task + 1]
        meana = round(np.mean(accs) * 100, 2)
        meanas.append(meana)

        acc_mean = round(np.mean(acc_mean) * 100, 2)
        print(f"acc_mean: {acc_mean}", end="")
        print()
        if valid:
            mkdir_if_missing(f'{args.result_path}/{subfolder_c}/val_models')
            with open(save_model_path, 'wb') as f:
                if args.method in PROMPT_METHODS:
                    pickle.dump(life_model_ins, f)
                else:
                    pickle.dump(model, f)
            if args.method in PROMPT_METHODS:
                with open(save_proto_path, 'wb') as f:
                    pickle.dump(prototypes, f)
        prev_model = copy.deepcopy(model).cuda(args.gpu) if args.method not in PROMPT_METHODS else None

    print('AP: ', acc_mean)
    backward = []
    for t in range(args.n_tasks - 1):
        b = acc_matrix[args.n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward), 2)
    print('AF: ', mean_backward)
    print('\n')
    return acc_mean, mean_backward, acc_matrix

def pipeline_class_IL_no_inter_edge(args, valid=False):
    epochs = args.epochs if valid else 0
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test,args=args)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)
    task_manager = semi_task_manager()
    data_prepare(args, dataset)

    # model = get_model(dataset, args).cuda(args.gpu) if valid else None
    model = get_model(dataset, args).cuda(args.gpu) if valid else None
    life_model = importlib.import_module(f'Baselines.{args.method}_model')
    life_model_ins = life_model.NET(model, task_manager, args) if valid else None

    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    if args.method in PROMPT_METHODS:
        prototypes = torch.zeros(args.n_tasks, args.d_data)

    n_cls_so_far = 0
    for task, task_cls in enumerate(args.task_seq):

        name, ite = args.current_model_save_path
        config_name = name.split('/')[-1]
        subfolder_c = name.split(config_name)[-2]
        save_model_name = f'{config_name}_{ite}_{task_cls}'
        save_model_path = f'{args.result_path}/{subfolder_c}val_models/{save_model_name}.pkl'

        if args.method in PROMPT_METHODS:
            save_proto_name = save_model_name + '_prototypes'
            save_proto_path = f'{args.result_path}/{subfolder_c}val_models/{save_proto_name}.pkl'

        if not valid:

            model = pickle.load(open(save_model_path, 'rb')).cuda(args.gpu)
            if args.method in PROMPT_METHODS:
                life_model_ins = pickle.load(open(save_model_path, 'rb')).cuda(args.gpu)
                prototypes = pickle.load(open(save_proto_path, 'rb'))

        n_cls_so_far+=len(task_cls)
        subgraph, ids_per_cls, [train_ids, valid_ids, test_ids] = pickle.load(open(f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{task_cls}.pkl', 'rb'))
        subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
        features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
        task_manager.add_task(task, n_cls_so_far)
        label_offset1 = task_manager.get_label_offset(task-1)[1]

        if task == 0 and valid and (args.method in PROMPT_METHODS):
            print(args.dataset)
            if 'Products' in args.dataset:
                pass
                # life_model_ins.pretrain(args, subgraph, features, batch_size=1000)
            else:
                life_model_ins.pretrain(args, subgraph, features)

        for epoch in range(epochs):
            if args.method == 'tpp':
                life_model_ins.observe_il(subgraph, features, labels, task, train_ids, ids_per_cls, label_offset1,
                                          dataset)
            elif args.method in {'tcpp', 'mip'}:
                life_model_ins.observe_il(epoch, subgraph, features, labels, task, train_ids, valid_ids, test_ids, ids_per_cls, label_offset1,
                                          dataset)
            elif args.method == 'puma':
                life_model_ins.observe(args, subgraph, features, labels, task, train_ids, test_ids,
                                          ids_per_cls, dataset)
            else:
                life_model_ins.observe(args, subgraph, features, labels, task, train_ids, ids_per_cls, dataset)


        if valid and (args.method in PROMPT_METHODS):
            prototypes[task] = life_model_ins.getprototype(subgraph, features, train_ids)

        acc_mean = []
        for t in range(task+1):

            subgraph, ids_per_cls, [train_ids, valid_ids_, test_ids_] = pickle.load(open(f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{args.task_seq[t]}.pkl', 'rb'))
            subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            test_ids = valid_ids_ if valid else test_ids_
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls]

            if args.method in PROMPT_METHODS:
                if task > 0:
                    taskid = life_model_ins.gettaskid(prototypes, subgraph, features, task+1, test_ids)
                else:
                    taskid = 0
                # taskid = t
                # print(t, taskid)
                label_offset1, label_offset2 = task_manager.get_label_offset(int(taskid) - 1)[1], task_manager.get_label_offset(int(taskid))[1]
                labels = labels - label_offset1
                output = life_model_ins.getpred(subgraph, features, taskid, test_ids, labels)
                acc = evaluatewp(output, labels, test_ids, cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            else:

                if args.classifier_increase:
                    label_offset1, label_offset2 = task_manager.get_label_offset(int(t) - 1)[1], \
                                                   task_manager.get_label_offset(int(t))[1]

                    acc = evaluate(model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                                   cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
                else:
                    acc = evaluate(model, subgraph, features, labels, test_ids, label_offset1, args.n_cls,
                                   cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            acc_matrix[task][t] = round(acc*100,2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc*100:.2f}|", end="")

        acc_mean = round(np.mean(acc_mean)*100,2)
        print(f"acc_mean(ID acc): {acc_mean})", end="")
        print()

        if valid:
            mkdir_if_missing(f'{args.result_path}/{subfolder_c}/val_models')
            with open(save_model_path, 'wb') as f:
                if args.method in PROMPT_METHODS:
                    pickle.dump(life_model_ins, f)
                else:
                    pickle.dump(model, f)
            if args.method in PROMPT_METHODS:
                with open(save_proto_path, 'wb') as f:
                    pickle.dump(prototypes, f)


    print('AP: ', acc_mean)
    backward = []
    for t in range(args.n_tasks-1):
        b = acc_matrix[args.n_tasks-1][t]-acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward),2)
    print('AF: ', mean_backward)
    print('\n')
    return acc_mean, mean_backward, acc_matrix


def pipeline_class_IL_no_inter_edge_minibatch(args, valid=False):
    epochs = args.epochs if valid else 0
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test,args=args)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)
    task_manager = semi_task_manager()
    data_prepare(args, dataset)

    model = get_model(dataset, args).cuda(args.gpu)
    life_model = importlib.import_module(f'Baselines.{args.method}_model')
    life_model_ins = life_model.NET(model, task_manager, args) if valid else None

    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])

    if args.method in PROMPT_METHODS:
        prototypes = torch.zeros(args.n_tasks, args.d_data)

    name, ite = args.current_model_save_path
    config_name = name.split('/')[-1]
    subfolder_c = name.split(config_name)[-2]
    save_model_name = f'{config_name}_{ite}'
    save_model_path = f'{args.result_path}/{subfolder_c}val_models/{save_model_name}.pkl'

    if args.method in PROMPT_METHODS:
        save_proto_name = save_model_name + '_prototypes'
        save_proto_path = f'{args.result_path}/{subfolder_c}val_models/{save_proto_name}.pkl'
    if not valid:
        life_model_ins = pickle.load(open(save_model_path,'rb')).cuda(args.gpu)
        if args.method in PROMPT_METHODS:
            prototypes = pickle.load(open(save_proto_path,'rb'))

    n_cls_so_far = 0
    for task, task_cls in enumerate(args.task_seq):
        n_cls_so_far += len(task_cls)
        subgraph, ids_per_cls, [train_ids, valid_ids, test_ids] = pickle.load(open(f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{task_cls}.pkl', 'rb'))
        # subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
        features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
        task_manager.add_task(task, n_cls_so_far)
        label_offset1 = task_manager.get_label_offset(task - 1)[1]

        if task == 0 and valid and args.method in PROMPT_METHODS:
            life_model_ins.pretrain(args, subgraph, features, batch_size = args.batch_size)

        dataloader = dgl.dataloading.DataLoader(subgraph, train_ids, args.nb_sampler,
                                                batch_size=args.batch_size, shuffle=args.batch_shuffle,
                                                drop_last=False)

        for epoch in range(epochs):
            if args.method == 'tpp':
                life_model_ins.observe_il(subgraph, features, labels, task, train_ids, ids_per_cls, label_offset1,
                                          dataset)
            elif args.method in {'tcpp', 'mip'}:
                life_model_ins.observe_il(epoch, subgraph, features, labels, task, train_ids, valid_ids, test_ids, ids_per_cls, label_offset1,
                                          dataset)
            elif args.method == 'puma':
                life_model_ins.observe(args, subgraph, features, labels, task, train_ids, test_ids,
                                          ids_per_cls, dataset)
            else:
                life_model_ins.observe_class_IL_batch(args, subgraph, dataloader, features, labels, task, train_ids, ids_per_cls, dataset)

            torch.cuda.empty_cache()

        if valid and args.method in PROMPT_METHODS:
            prototypes[task] = life_model_ins.getprototype(subgraph, features, train_ids)

        acc_mean = []
        for t in range(task + 1):

            subgraph, ids_per_cls, [train_ids, valid_ids_, test_ids_] = pickle.load(open(f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{args.task_seq[t]}.pkl', 'rb'))
            subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
            test_ids = valid_ids_ if valid else test_ids_
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls]
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            if args.method in PROMPT_METHODS:
                if task == 0:
                    taskid = 0
                else:
                    taskid = life_model_ins.gettaskid(prototypes, subgraph, features, task+1, test_ids)
                    print(t, taskid)
                label_offset1 = task_manager.get_label_offset(int(taskid) - 1)[1]
                labels = labels - label_offset1
            output = life_model_ins(subgraph, features, taskid)
            acc = evaluatewp(output, labels, test_ids, cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            acc_matrix[task][t] = round(acc * 100, 2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc * 100:.2f}|", end="")

        acc_mean = round(np.mean(acc_mean) * 100, 2)
        print(f"acc_mean(ID acc): {acc_mean})", end="")
        print()

    if valid:
        mkdir_if_missing(f'{args.result_path}/{subfolder_c}/val_models')
        with open(save_model_path, 'wb') as f:
            pickle.dump(life_model_ins, f)
        if args.method in PROMPT_METHODS:
            with open(save_proto_path, 'wb') as f:
                pickle.dump(prototypes, f)

    print('AP: ', acc_mean)
    backward = []
    for t in range(args.n_tasks - 1):
        b = acc_matrix[args.n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward), 2)
    print('AF: ', mean_backward)
    print('\n')
    return acc_mean, mean_backward, acc_matrix
