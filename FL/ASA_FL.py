# ASA_FL.py
# proposed by: ll
# @Author  : ll
# @Date    : 2024-6-1
import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
import copy
import sys
import os
import numpy as np
# import syft as sy
from FL.learning_tasks import svmLoss
import utils
import FLLocalSupport as FLSup


def get_cross_rounders(clients_est_round_T_train, max_round_interval):
    cross_rounder_ids = []
    for c_id in range(len(clients_est_round_T_train)):
        if clients_est_round_T_train[c_id] > max_round_interval:
            cross_rounder_ids.append(c_id)
    return cross_rounder_ids


def sort_ids_by_atime_asc(id_list, atime_list, quota):
    """
    Sort a list of client ids according to their performance, in an descending ordered 根据客户端 ID 的性能按降序对列表进行排序
    :param id_list: a list of client ids to sort
    :param perf_list: full list of all clients' arrival time
    :return: sorted id_list
    """
    # make use of a map
    cp_map = {}  # client-perf-map
    for id in id_list:
        cp_map[id] = atime_list[id]  # build the map with corresponding perf
    # sort by perf
    sorted_map = sorted(cp_map.items(), key=lambda x: x[1])  # a sorted list of tuples
    quota = int(quota) - 1
    max_faster_T = sorted_map[quota][1]  # the id of the fastest client
    print("max_faster_T:", max_faster_T)
    sorted_id_list = [sorted_map[i][0] for i in range(len(id_list))]  # extract the ids into a list
    return sorted_id_list, max_faster_T


def select_clients_ACFM(make_ids, clients_arrival_T, T_threshold, quota):
    """
    Select clients to aggregate their models according to Compensatory First-Come-First-Merge principle.
    :param make_ids: ids of clients start their training this round
    :param undrafted_ids: ids of clients unpicked previous rounds
    :param clients_arrival_T: clients' arrival time at rd round
    :param quota: number of clients to draw this round
    :return:
    picks: ids of selected clients
    clients_arrival_T: clients' arrival time for next  round
    undrafted_ids: ids of undrafted clients
    """
    max_slow_T = max(clients_arrival_T)
    # print("max_slow_arrival_T:", max_slow_T)
    if len(make_ids) <= quota:
        last_quota = len(make_ids)
        sorted_ids, max_faster_T = sort_ids_by_atime_asc(make_ids, clients_arrival_T, last_quota)
        F_S_interval = 0  # 两类客户端完成本地训练的时间差
        max_slow_T = 0
        fast_ids = sorted_ids[0:quota]
        slow_ids = []
    else:
        last_quota = quota + 1
        sorted_ids, max_faster_T = sort_ids_by_atime_asc(make_ids, clients_arrival_T, quota)
        F_S_interval = max_slow_T - max_faster_T  # 两类客户端完成本地训练的时间差
        fast_ids = sorted_ids[0:quota]
        slow_ids = []
        for i in make_ids:
            if i not in fast_ids:
                slow_ids.append(i)

    return fast_ids, slow_ids, max_faster_T, max_slow_T, F_S_interval, last_quota


def train(models, picked_ids, env_cfg, cm_map, fdl, last_loss_rep, mode, verbose=True):
    """
    Execute one EPOCH of training process of any machine learning model on all clients
    :param models: a list of model prototypes corresponding to clients
    :param picked_ids: participating client indices for local training
    :param env_cfg: environment configurations
    :param cm_map: the client-model map, as a dict
    :param fdl: an instance of FederatedDataLoader
    :param last_loss_rep: loss reports in last run
    :param mode : need sum of loss or sum of loss**2
    :param verbose: display batch progress or not.
    :return: epoch training loss of each client, batch-summed
    """
    dev = env_cfg.device
    if len(picked_ids) == 0:  # no training happens
        return last_loss_rep
    # extract settings
    n_models = env_cfg.n_clients  # # of clients
    # initialize loss report, keep loss tracks for idlers, clear those for participants
    client_train_loss_vec = last_loss_rep
    for id in picked_ids:
        client_train_loss_vec[id] = 0.0
    # Disable printing
    if not verbose:
        sys.stdout = open(os.devnull, 'w')
    # initialize training mode
    for m in range(n_models):
        models[m].train()

    # Define loss based on task
    if env_cfg.loss == 'mse':  # regression task
        loss_func = nn.MSELoss(reduction='mean')  # cannot back-propagate with 'reduction=sum'
    elif env_cfg.loss == 'svmLoss':  # SVM task
        loss_func = svmLoss(reduction='mean')  # self-defined loss, have to use default reduction 'mean'
    elif env_cfg.loss == 'nllLoss':  # CNN mnist task
        loss_func = nn.NLLLoss()

    # one optimizer for each model (re-instantiate optimizers to clear any possible momentum
    optimizers = []
    for i in range(n_models):
        if env_cfg.optimizer == 'SGD':
            optimizers.append(optim.SGD(models[i].parameters(), lr=env_cfg.lr))
        elif env_cfg.optimizer == 'Adam':
            optimizers.append(optim.Adam(models[i].parameters(), lr=env_cfg.lr))
        else:
            print('Err> Invalid optimizer %s specified' % env_cfg.optimizer)

    # begin an epoch of training
    for batch_id, (inputs, labels, client) in enumerate(fdl):
        inputs, labels = inputs.to(dev), labels.to(dev)  # data to device
        model_id = cm_map[client.id]  # locate the right model index
        # neglect non-participants
        if model_id not in picked_ids:
            continue
        # mini-batch GD
        if mode == 0:
            print('\n> Batch #', batch_id, 'on', client.id)
            print('>   model_id = ', model_id)

        # ts = time.time_ns() / 1000000.0  # ms
        model = models[model_id]
        optimizer = optimizers[model_id]
        # gradient descent procedure
        optimizer.zero_grad()
        y_hat = model(inputs)
        # loss
        loss = loss_func(y_hat, labels)
        loss.backward()
        # weights
        optimizer.step()
        # te = time.time_ns() / 1000000.0   # ms
        # print('> T_batch = ', te-ts)
        # display
        if mode == 0:
            print('>   batch loss = ', loss.item())  # avg. batch loss
            client_train_loss_vec[model_id] += loss.detach().item() * len(inputs)  # sum up
        elif mode == 1:
            client_train_loss_vec[model_id] += (loss.detach().item() * len(inputs)) ** 2
    # Restore printing
    if not verbose:
        sys.stdout = sys.__stdout__
    # end an epoch-training - all clients have traversed their own local data once
    return client_train_loss_vec


def local_test(models, picked_ids, env_cfg, cm_map, fdl, last_loss_rep):
    """
    Evaluate client models locally and return a list of loss/error
    :param models: a list of model prototypes corresponding to clients
    :param env_cfg: environment configurations
    :param picked_ids: selected client indices for local training
    :param cm_map: the client-model map, as a dict
    :param fdl: an instance of FederatedDataLoader
    :param last_loss_rep: loss reports in last run
    :return: epoch test loss of each client, batch-summed
    """
    if not picked_ids:  # no training happens
        return last_loss_rep
    dev = env_cfg.device
    # initialize loss report, keep loss tracks for idlers, clear those for participants
    client_test_loss_vec = last_loss_rep
    for id in picked_ids:
        client_test_loss_vec[id] = 0.0
    # Define loss based on task
    if env_cfg.loss == 'mse':  # regression task
        loss_func = nn.MSELoss(reduction='sum')
    elif env_cfg.loss == 'svmLoss':  # SVM task
        loss_func = svmLoss(reduction='sum')
    elif env_cfg.loss == 'nllLoss':  # CNN mnist task
        loss_func = nn.NLLLoss(reduction='sum')
        is_cnn = True

    # initialize evaluation mode
    for m in range(env_cfg.n_clients):
        models[m].eval()
    # local evaluation, batch-wise
    acc = 0.0
    count = 0.0
    with torch.no_grad():
        for batch_id, (inputs, labels, client) in enumerate(fdl):
            inputs, labels = inputs.to(dev), labels.to(dev)  # data to device
            model_id = cm_map[client.id]  # locate the right model index
            # neglect non-participants
            if model_id not in picked_ids:
                continue
            model = models[model_id]
            # inference
            y_hat = model(inputs)
            # loss
            loss = loss_func(y_hat, labels)
            client_test_loss_vec[model_id] += loss.detach().item()
            # accuracy
            b_acc, b_cnt = utils.batch_sum_accuracy(y_hat, labels, env_cfg.loss)
            # if len(labels)==0:
            #     print(model_id,y_hat,labels)
            acc += b_acc
            count += b_cnt

    print('> acc = %.6f' % (acc / count))
    return client_test_loss_vec


def global_test(model, env_cfg, cm_map, fdl):
    """
    Testing the aggregated global model by averaging its error on each local data
    :param model: the global model
    :param env_cfg: environment configurations
    :param cm_map: the client-model map, as a dict
    :param fdl: FederatedDataLoader
    :return: global model's loss on each client (as a vector), accuracy
    """
    dev = env_cfg.device
    test_sum_loss_vec = [0 for i in range(env_cfg.n_clients)]
    # Define loss based on task
    if env_cfg.loss == 'mse':  # regression task
        loss_func = nn.MSELoss(reduction='sum')
    elif env_cfg.loss == 'svmLoss':  # SVM task
        loss_func = svmLoss(reduction='sum')
    elif env_cfg.loss == 'nllLoss':  # CNN mnist task
        loss_func = nn.NLLLoss(reduction='sum')
        is_cnn = True

    # initialize evaluation mode
    print('> global test')
    model.eval()
    # local evaluation, batch-wise
    acc = 0.0
    count = 0
    for batch_id, (inputs, labels, client) in enumerate(fdl):
        inputs, labels = inputs.to(dev), labels.to(dev)  # data to device
        model_id = cm_map[client.id]
        # inference
        y_hat = model(inputs)
        # loss
        loss = loss_func(y_hat, labels)
        test_sum_loss_vec[model_id] += loss.detach().item()
        # compute accuracy
        b_acc, b_cnt = utils.batch_sum_accuracy(y_hat, labels, env_cfg.loss)
        acc += b_acc
        count += b_cnt

    print('>   acc = %.6f' % (acc / count))
    return test_sum_loss_vec, acc / count


def init_cache(glob_model, env_cfg):
    """
    Initiate cloud cache with the global model
    :param glob_model:  initial global model
    :param env_cfg:  env config
    :return: the cloud cache
    """
    cache = []
    for i in range(env_cfg.n_clients):
        cache.append(copy.deepcopy(glob_model))
    return cache


def update_cloud_cache(cache, models, the_ids):
    """
    Update the model cache residing on the cloud, it contains the latest non-aggregated models
    :param cache: the model cache
    :param models: latest local model set containing picked, undrafted and deprecated models
    :param the_ids: ids of clients to update cache
    :return:
    """
    # use deepcopy to decouple cloud cache and local models
    for id in the_ids:
        cache[id] = copy.deepcopy(models[id])


def update_cloud_cache_deprecated(cache, global_model, slow_ids):
    """
    Update entries of those clients lagging too much behind with the latest global model 更新那些客户的条目，这些客户端在最新的全局模型中落后太多
    :param cache: the model cache
    :param global_model: the aggregated global model
    :param deprecated_ids: ids of clients to update cache
    :return:
    """
    # use deepcopy to decouple cloud cache and local models
    for id in slow_ids:
        cache[id] = copy.deepcopy(global_model)


def get_versions(ids, versions):
    """
    Show versions of specified clients, as a dict
    :param ids: clients ids
    :param versions: versions vector of all clients
    :return:
    """
    cv_map = {}
    for id in ids:
        cv_map[id] = versions[id]

    return cv_map


def update_versions(versions, make_ids, rd):
    """
    Update versions of local models that successfully perform training in the current round
    :param versions: version vector
    :param make_ids: well-progressed clients ids
    :param rd: round number
    :return: na
    """
    for id in make_ids:
        versions[id] = rd


def version_filter(versions, the_ids, base_v, lag_tolerant=1):
    """
    good_ids, deprecated_ids = version_filter(versions, client_ids, rd - 1, lag_tolerant=lag_t)  # find deprecated 查找已弃用
    ood_ids, deprecated_ids = version_filter(versions, client_ids, rd - 1, lag_tolerant=lag_t)  # find deprecated
    Apply a filter to client ids by checking their model versions. If the version is lagged behind the latest version 通过检查客户端 ID 的模型版本，将筛选器应用于客户端 ID。如果版本落后于最新版本
    (i.e., round number) by a number > lag_tolarant, then it will be filtered out.
    :param versions: client versions vector
    :param the_ids: client ids to check version
    :param base_v: latest base version
    :param lag_tolerant: maximum tolerance of version lag
    :return: non-straggler ids, deprecated ids
    """
    good_ids = []
    deprecated_ids = []
    for id in the_ids:
        if base_v - versions[id] <= lag_tolerant:
            good_ids.append(id)
        else:  # stragglers
            deprecated_ids.append(id)

    return good_ids, deprecated_ids


def distribute_models(global_model, models, make_ids):
    """
    distribute_models(fast_model, models, fast_ids)
    Distribute the global model
    :param global_model: aggregated global model
    :param models: local models
    :param make_ids: ids of clients that will replace their local models with the global one
    :return:
    """
    for id in make_ids:
        models[id] = copy.deepcopy(global_model)


def extract_weights(model):
    weights = []
    for name, weight in model.named_parameters():
        weights.append((name, weight.data))

    return weights


def extract_client_updates(global_model, models, picked_ids):
    baseline_weights = extract_weights(global_model)
    recieve_buffer = []
    for m in picked_ids:
        recieve_buffer.append((m, extract_weights(models[m])))
    # Calculate updates from weights
    updates = []
    for m, weight in recieve_buffer:
        update = []
        for i, (name, weight) in enumerate(weight):
            bl_name, baseline = baseline_weights[i]

            # Ensure correct weight is being updated
            assert name == bl_name

            # Calculate update
            delta = weight - baseline
            update.append((name, delta))
        updates.append(update)

    return updates


def safa_aggregate(models, picked_ids, local_shards_sizes, data_size):
    """
    The function implements aggregation step (Semi-Async. FedAvg), allowing cross-round com
    :param models: a list of local models
    :param picked_ids: selected client indices for local training
    :param local_shards_sizes: a list of local data sizes, aligned with the orders of local models, say clients.
    :param data_size: total data size
    :return: a global model
    """
    print('>   Aggregating (SAAR)...')
    global_model = copy.deepcopy(models[0])
    global_model_params = global_model.state_dict()
    for pname, param in global_model_params.items():
        global_model_params[pname] = 0.0
    round_data_size = 0
    for id in picked_ids:
        round_data_size += local_shards_sizes[id]
    client_weights_vec = np.array(local_shards_sizes) / round_data_size  # client weights (i.e., n_k / n)
    for m in picked_ids:  # for each local model
        for pname, param in models[m].state_dict().items():
            global_model_params[pname] += param.data * client_weights_vec[m]  # sum up the corresponding param
    # load state dict back
    global_model.load_state_dict(global_model_params)
    return global_model


def run_ASA_FL(env_cfg, glob_model, cm_map, data_size, fed_loader_train, fed_loader_test, client_shard_sizes,
                clients_perf_vec, clients_crash_prob_vec, crash_trace, progress_trace, lag_t=1, wait_n=5):
    """
    Run FL with SAAR algorithm
    :param env_cfg: environment config
    :param glob_model: the global model
    :param cm_map: client-model mapping
    :param data_size: total data size
    :param fed_loader_train: federated training set
    :param fed_loader_test: federated test set
    :param client_shard_sizes: sizes of clients' shards
    :param clients_perf_vec: batch overhead values of clients
    :param clients_crash_prob_vec: crash probs of clients
    :param crash_trace: simulated crash trace
    :param progress_trace: simulated progress trace
    :param lag_t: tolerance of lag
    :return:
    """
    # init
    global_model = glob_model  # the global model
    # fast_model = None  # the fast model
    slow_model = None  # the slow model`
    models = [None for _ in range(env_cfg.n_clients)]  # local models

    client_ids = list(range(env_cfg.n_clients))
    distribute_models(global_model, models, client_ids)  # init local models
    # global cache, storing models to merge before aggregation and latest models after aggregation.
    cache = copy.deepcopy(models)
    # cache = None  # cache will be initiated in the very first epoch
    T_max = 3000
    # traces
    reporting_train_loss_vec = [0.0 for _ in range(env_cfg.n_clients)]
    clients_loss_sum = [0.0 for _ in range(env_cfg.n_clients)]
    reporting_test_loss_vec = [0.0 for _ in range(env_cfg.n_clients)]
    versions = np.array([-1 for _ in range(env_cfg.n_clients)])
    epoch_train_trace = []
    epoch_test_trace = []
    pick_trace = []
    make_trace = []
    undrafted_trace = []
    deprecated_trace = []
    F_S_interval_trace = []
    T_threshold_trace = []
    round_trace = []
    acc_trace = []
    time_trace = []
    fast_ids_shard_size = {}
    slow_ids_shard_size = {}
    fast_ids_perf_vec = {}
    slow_ids_perf_vec = {}

    # Global event handler
    event_handler = FLSup.EventHandler(['time', 'T_dist', 'obj_acc_time'])
    # Local counters
    # 1. Local timers - record work time of each client
    client_timers = [0.01 for _ in range(env_cfg.n_clients)]  # totally
    client_comm_timers = [0.0 for _ in range(env_cfg.n_clients)]  # comm. totally
    # 2. Futile counters - progression (i,e, work time) in vain caused by denial
    clients_est_round_T_train = np.array(client_shard_sizes) / env_cfg.batch_size * env_cfg.n_epochs / np.array(
        clients_perf_vec)
    clients_est_round_T_comm = env_cfg.model_size / env_cfg.bw_s_set

    # 第一轮时间包括初始分发模型时间，训练时间和上传时间
    clients_arrival_T = np.array(clients_est_round_T_train) + clients_est_round_T_comm
    print('clients_arrival_T', clients_arrival_T)
    T_threshold = sum(clients_arrival_T) / len(clients_arrival_T)  # 时间阈值的选取(时间平均值) # 保留乘以一个参数（0.5、0.7）
    # print('avg_clients_arrival_T', avg_clients_arrival_T)
    T_threshold_trace.append(T_threshold)
    time_trace.append(T_threshold)
    print('T_threshold', T_threshold)
    picked_ids = []
    undrafted_ids = []
    client_futile_timers = [0.0 for _ in range(env_cfg.n_clients)]  # totally
    eu_count = 0.0  # effective updates count
    sync_count = 0.0  # synchronization count
    version_var = 0.0
    # Best loss (global)
    best_rd = -1
    best_loss = float('inf')
    best_acc = -1.0
    best_model = None
    fast_round = 0
    slow_round = 0
    global_round = 0
    last_fast_ids_arrival_t = {}
    last_slow_ids_arrival_t = {}
    # wait_total_t = 0
    rd = 0
    quota = math.ceil(env_cfg.n_clients * env_cfg.pick_C)  # the quota
    crash_ids = crash_trace[rd]
    # print('crash_ids: ', crash_ids)
    # 去除掉掉线的客户端
    available_ids = [c_id for c_id in range(env_cfg.n_clients) if c_id not in crash_ids]
    # quota = math.ceil(len(available_ids) * env_cfg.pick_C)
    # print(len(crash_ids))
    print(len(available_ids))
    fast_ids, slow_ids, max_faster_T, max_slow_T, F_S_interval, last_quota = select_clients_ACFM(available_ids,
                                                                                                 clients_arrival_T,
                                                                                                 T_threshold,
                                                                                                 quota)
    print("last_quota", last_quota)
    print("max_faster_T", max_faster_T)
    time_trace.append(max_slow_T)
    print("max_slow_T", max_slow_T)
    print("F_S_interval", F_S_interval)
    distribute_models(global_model, models, available_ids)
    for id_ in fast_ids:
        index = id_
        if index < len(client_shard_sizes):  # 确保索引在client_sharp_size列表的范围内
            fast_ids_shard_size[id_] = client_shard_sizes[index]  # 字典
            fast_ids_perf_vec[id_] = clients_perf_vec[id_]
    # print('>   @Cloud> fast_ids_shard_size:', fast_ids_shard_size)
    # print('>   @Cloud> fast_ids_perf_vec:', fast_ids_perf_vec)
    # undrafted_ids = [c_id for c_id in make_ids if c_id not in picked_ids]
    for id_ in slow_ids:
        index = id_
        if index < len(client_shard_sizes):  # 确保索引在client_sharp_size列表的范围内
            slow_ids_shard_size[id_] = client_shard_sizes[index]
            slow_ids_perf_vec[id_] = clients_perf_vec[id_]

    # begin training: global rounds
    for rd in range(env_cfg.n_rounds):
        print('\n> ----------------------Round #%d-------------------------------' % rd)
        m_syn = 0
        # reset timers
        client_round_timers = [0.0 for _ in range(env_cfg.n_clients)]  # local time in current round
        client_round_comm_timers = [0.0 for _ in range(env_cfg.n_clients)]  # local comm. time in current round
        picked_client_round_timers = [0.0 for _ in range(env_cfg.n_clients)]  # the picked clients to wait
        # randomly pick a specified fraction of clients to launch training
        # quota = math.ceil(env_cfg.n_clients * env_cfg.pick_pct)  # the quota
        # simulate device or network failure
        m_syn = len(fast_ids)  # count sync. overheads
        sync_count += m_syn
        # fast_ids_shard_size[id_] # 字典
        # fast_ids_perf_vec[id_], interval(间隔)
        # case 1 The fast_ids has completed the local training while the slow_ids has not
        if F_S_interval > T_threshold or last_quota <= quota:
            fast_round += 1
            print('> F_S_interval01:', F_S_interval)
            # fast_ids Local training step
            # print('\n> Round（fast_ids） #%d' % rd)
            # wait_total_t += T_threshold
            # print('> distributing fast_ids model')
            # 随机打乱选取的客户端性能
            keys = list(fast_ids_perf_vec.keys())
            values = list(fast_ids_perf_vec.values())
            # 模拟客户端性能波动
            # random.shuffle(values)
            fast_ids_perf_vec = dict(zip(keys, values))
            # print('fast_ids_per:', fast_ids_perf_vec)
            for epo in range(env_cfg.n_epochs):  # local epochs (same # of epochs for each client)
                print('\n> @Devices> fast local epoch #%d' % epo)
                # if fast_round == 0:
                reporting_train_loss_vec = train(models, available_ids, env_cfg, cm_map, fed_loader_train,
                                                 reporting_train_loss_vec, mode=0, verbose=False)
                # else:
                #     reporting_train_loss_vec = train(models, fast_ids, env_cfg, cm_map, fed_loader_train,
                #                                      reporting_train_loss_vec, mode=0, verbose=False)
                # add to trace
                epoch_train_trace.append(
                    np.array(reporting_train_loss_vec) / (np.array(client_shard_sizes) * env_cfg.train_pct))
                # local test reports
                reporting_test_loss_vec = local_test(models, available_ids, env_cfg, cm_map, fed_loader_test,
                                                     reporting_test_loss_vec)
                # add to trace
                epoch_test_trace.append(
                    np.array(reporting_test_loss_vec) / (np.array(client_shard_sizes) * env_cfg.test_pct))
            # Aggregation step
            # discriminative update of cloud cache and aggregate
            # pre-aggregation: update cache from picked clients
            update_cloud_cache(cache, models, fast_ids)
            print('\n> -------------------fast Aggregation step (Round #%d)----------------------------' % rd)
            fast_model = safa_aggregate(cache, fast_ids, client_shard_sizes, data_size)  # aggregation
            distribute_models(fast_model, models, fast_ids)
            # Reporting phase: distributed test of the global model
            post_aggre_loss_vec, acc = global_test(fast_model, env_cfg, cm_map, fed_loader_test)
            update_cloud_cache_deprecated(cache, fast_model, fast_ids)
            # print('>   @Devices> post-aggregation loss reports  = ',
            #       np.array(post_aggre_loss_vec) / ((np.array(client_shard_sizes)) * env_cfg.test_pct))
            # overall loss, i.e., objective (1) in McMahan's paper
            overall_loss = np.array(post_aggre_loss_vec).sum() / (data_size * env_cfg.test_pct)
            # update so-far best
            if overall_loss < best_loss:
                best_loss = overall_loss
                best_acc = acc
                best_model = global_model
                best_rd = rd
            print('>   @Cloud> post-aggregation loss avg = ', overall_loss)
            round_trace.append(overall_loss)
            acc_trace.append(acc)
            # # post-aggregation: update cache from undrafted clients
            # update_cloud_cache(cache, fast_model, fast_ids)
            # fast_ids_round_T_train = np.array(fast_ids_shard_size) / env_cfg.batch_size * env_cfg.n_epochs / np.array(
            #     fast_ids_perf_vec)
            #######fast_ids__perf_vec需要保留回传
            keys = set(fast_ids_shard_size.keys()) & set(fast_ids_perf_vec.keys())
            results = []
            # 对每个键进行计算
            for key in keys:
                shard_size = fast_ids_shard_size[key]
                perf_vec = fast_ids_perf_vec[key]
                result = shard_size / env_cfg.batch_size * env_cfg.n_epochs / perf_vec + env_cfg.model_size / env_cfg.bw_s_set
                last_fast_ids_arrival_t[key] = result
                results.append(result)
            fast_ids_arrival_T = np.array(results)

            # clients_est_round_T_comm = env_cfg.model_size / env_cfg.bw_s_set
            # 包括初始分发模型时间，训练时间和上传时间
            # fast_ids_arrival_T = np.array(fast_ids_round_T_train) + clients_est_round_T_comm
            # print('> fast_ids_arrival_T:', fast_ids_arrival_T)
            max_faster_T = max(fast_ids_arrival_T)
            # print(max_faster_T)
            # print("last_fast_ids_arrival_t", last_fast_ids_arrival_t)
            if last_quota <= quota:
                F_S_interval = 0
                T_threshold = 0
                T_threshold_trace.append(T_threshold)
            else:
                F_S_interval = F_S_interval - max_faster_T - T_threshold
                F_S_interval_trace.append(F_S_interval)
                print('> F_S_interval02:', F_S_interval)
                T_threshold += sum(fast_ids_arrival_T) / len(fast_ids_arrival_T)  # 保留乘以一个参数（0.2、0.5、0.7）
                # T_threshold *= 1.3
                if T_threshold >= 0:
                    time_trace.append(T_threshold)
                print('> T_threshold:', T_threshold)
                T_threshold_trace.append(T_threshold)
            time_trace.append(T_threshold)
        # case2 Missed each other within the time threshold
        if F_S_interval < 0:
            print('> F_S_interval01:', F_S_interval)
            # slow_ids Local training step
            # print('\n> Round（slow_ids） #%d' % rd)
            # wait_total_t += T_threshold
            slow_round += 1  # 记录整个过程中slow轮数
            # 随机打乱选取的客户端性能
            keys = list(slow_ids_perf_vec.keys())
            values = list(slow_ids_perf_vec.values())
            random.shuffle(values)
            slow_ids_perf_vec = dict(zip(keys, values))
            for epo in range(env_cfg.n_epochs):  # local epochs (same # of epochs for each client)
                print('\n> @Devices> slow local epoch #%d' % epo)
                reporting_train_loss_vec = train(models, available_ids, env_cfg, cm_map, fed_loader_train,
                                                 reporting_train_loss_vec, mode=0, verbose=False)
                # reporting_train_loss_vec = train(models, slow_ids, env_cfg, cm_map, fed_loader_train,
                #                                  reporting_train_loss_vec, mode=0, verbose=False)
                # add to trace
                epoch_train_trace.append(
                    np.array(reporting_train_loss_vec) / (np.array(client_shard_sizes) * env_cfg.train_pct))
                # local test reports
                reporting_test_loss_vec = local_test(models, available_ids, env_cfg, cm_map, fed_loader_test,
                                                     reporting_test_loss_vec)
                # add to trace
                epoch_test_trace.append(
                    np.array(reporting_test_loss_vec) / (np.array(client_shard_sizes) * env_cfg.test_pct))
            # Aggregation step
            # discriminative update of cloud cache and aggregate
            # pre-aggregation: update cache from picked clients
            update_cloud_cache(cache, models, slow_ids)
            print('\n> ------------slow Aggregation step (Round #%d)---------------' % rd)
            slow_model = safa_aggregate(cache, slow_ids, client_shard_sizes, data_size)  # aggregation
            # print('> distributing slow_ids model')
            distribute_models(slow_model, models, slow_ids)  # 模型分布
            update_cloud_cache_deprecated(cache, slow_model, slow_ids)
            # Reporting phase: distributed test of the global model
            post_aggre_loss_vec, acc = global_test(slow_model, env_cfg, cm_map, fed_loader_test)
            # print('>   @Devices> post-aggregation loss reports  = ',
            #       np.array(post_aggre_loss_vec) / ((np.array(client_shard_sizes)) * env_cfg.test_pct))
            # overall loss, i.e., objective (1) in McMahan's paper
            overall_loss = np.array(post_aggre_loss_vec).sum() / (data_size * env_cfg.test_pct)
            if overall_loss < best_loss:
                best_loss = overall_loss
                best_acc = acc
                best_model = global_model
                best_rd = rd
            print('>   @Cloud> post-aggregation loss avg = ', overall_loss)
            round_trace.append(overall_loss)
            acc_trace.append(acc)
            # # post-aggregation: update cache from undrafted clients
            # update_cloud_cache(cache, models, undrafted_ids)
            keys = set(slow_ids_shard_size.keys()) & set(slow_ids_perf_vec.keys())
            results = []
            # 对每个键进行计算
            for key in keys:
                shard_size = slow_ids_shard_size[key]
                perf_vec = slow_ids_perf_vec[key]
                result = shard_size / env_cfg.batch_size * env_cfg.n_epochs / perf_vec + env_cfg.model_size / env_cfg.bw_s_set
                last_slow_ids_arrival_t[key] = result
                results.append(result)
            slow_ids_arrival_t = np.array(results)
            max_slow_T = max(slow_ids_arrival_t)
            print(max_slow_T)
            time_trace.append(max_slow_T)
            # print('last_fast_ids_arrival_t', last_slow_ids_arrival_t)
            F_S_interval = max_slow_T + F_S_interval
            F_S_interval_trace.append(F_S_interval)
            print('last F_S_interval02', F_S_interval)
            T_threshold += sum(slow_ids_arrival_t) / len(slow_ids_arrival_t)*0.2  # 保留乘以一个参数（0.2、0.5、0.7）
            # T_threshold /= 1.3
            # print('T_threshold:', T_threshold)
            if T_threshold >= 0:
                time_trace.append(T_threshold)
                T_threshold_trace.append(T_threshold)
        # case3 The fast_ids has completed the local training and the slow_ids yet to complete
        if 0 <= F_S_interval <= T_threshold or rd == env_cfg.n_rounds - 1 or rd == env_cfg.n_rounds:  # 所有的可用客户端参与聚合
            global_round += 1
            print('\n> ------------Global Aggregation step---------------')
            # 随机打乱选取的客户端性能
            keys = list(slow_ids_perf_vec.keys())
            values = list(slow_ids_perf_vec.values())
            random.shuffle(values)
            slow_ids_perf_vec = dict(zip(keys, values))
            for epo in range(env_cfg.n_epochs):  # local epochs (same # of epochs for each client)
                # print('\n> @Devices> slow local epoch #%d' % epo)
                reporting_train_loss_vec = train(models, available_ids, env_cfg, cm_map, fed_loader_train,
                                                 reporting_train_loss_vec, mode=0, verbose=False)
                # reporting_train_loss_vec = train(models, slow_ids, env_cfg, cm_map, fed_loader_train,
                #                                  reporting_train_loss_vec, mode=0, verbose=False)
                # add to trace
                epoch_train_trace.append(
                    np.array(reporting_train_loss_vec) / (np.array(client_shard_sizes) * env_cfg.train_pct))
                # local test reports
                reporting_test_loss_vec = local_test(models, available_ids, env_cfg, cm_map, fed_loader_test,
                                                     reporting_test_loss_vec)
                # add to trace
                epoch_test_trace.append(
                    np.array(reporting_test_loss_vec) / (np.array(client_shard_sizes) * env_cfg.test_pct))
            # Aggregation step
            # discriminative update of cloud cache and aggregate
            # pre-aggregation: update cache from picked clients
            update_cloud_cache(cache, models, slow_ids)
            # print('\n> Aggregation step (Round #%d)' % rd)
            # slow_model = safa_aggregate(cache, slow_ids, client_shard_sizes, data_size)  # aggregation
            # # post-aggregation: update cache from undrafted clients
            # update_cloud_cache(cache, models, undrafted_ids)
            keys = set(slow_ids_shard_size.keys()) & set(slow_ids_perf_vec.keys())
            results = []
            # 对每个键进行计算
            for key in keys:
                shard_size = slow_ids_shard_size[key]
                perf_vec = slow_ids_perf_vec[key]
                result = shard_size / env_cfg.batch_size * env_cfg.n_epochs / perf_vec + env_cfg.model_size / env_cfg.bw_s_set
                last_slow_ids_arrival_t[key] = result
                results.append(result)
            slow_ids_arrival_t = np.array(results)
            # max_slow_T = max(slow_ids_arrival_t)
            # print(max_slow_T)
            # print('last_fast_ids_arrival_t', last_slow_ids_arrival_t)
            # T_threshold -= sum(slow_ids_arrival_t) / len(slow_ids_arrival_t)  # 保留乘以一个参数（0.2、0.5、0.7）
            # T_threshold -= T_threshold
            # print('T_threshold:', T_threshold)
            # T_threshold_trace.append(T_threshold)
            # print('> distributing slow_ids model')
            # distribute_models(slow_model, models, slow_ids)  # 模型分布
            # global aggregations
            pick_ids = fast_ids + slow_ids
            # print('> pick_ids:', pick_ids)
            # update_cloud_cache(cache, models, pick_ids)
            # print('\n> ------------------Global model Aggregation------------------------')
            global_model = safa_aggregate(cache, pick_ids, client_shard_sizes, data_size)
            update_cloud_cache_deprecated(cache, global_model, client_ids)
            # print('> distributing global model')
            distribute_models(global_model, models, client_ids)  # up to all clients
            post_aggre_loss_vec, acc = global_test(global_model, env_cfg, cm_map, fed_loader_test)
            # print('>   @Devices> post-aggregation loss reports  = ',
            #       np.array(post_aggre_loss_vec) / ((np.array(client_shard_sizes)) * env_cfg.test_pct))
            # overall loss, i.e., objective (1) in McMahan's paper
            overall_loss = np.array(post_aggre_loss_vec).sum() / (data_size * env_cfg.test_pct)
            # update so-far best
            if overall_loss < best_loss:
                best_loss = overall_loss
                best_acc = acc
                best_model = global_model
                best_rd = rd
            if env_cfg.keep_best:  # if to keep best
                global_model = best_model
                overall_loss = best_loss
                acc = best_acc
            print('>   @Cloud> post-aggregation loss avg = ', overall_loss)
            round_trace.append(overall_loss)
            acc_trace.append(acc)
            # update available_ids timers
            fast_keys = list(last_fast_ids_arrival_t.keys())
            slow_keys = list(last_slow_ids_arrival_t.keys())
            for key in fast_keys:
                clients_arrival_T[key] = last_fast_ids_arrival_t[key]
            for key in slow_keys:
                clients_arrival_T[key] = last_slow_ids_arrival_t[key]
            # update crash_ids timers
            for c_id in range(env_cfg.n_clients):
                if c_id in crash_ids:
                    T_comm = env_cfg.model_size / env_cfg.bw_s_set
                    T_train = client_shard_sizes[c_id] / env_cfg.batch_size * env_cfg.n_epochs / clients_perf_vec[c_id]
                    clients_arrival_T[c_id] = T_comm + T_train
            crash_ids = crash_trace[rd]
            # print(len(crash_ids))
            # print('crash_ids: ', crash_ids)
            # 去除掉掉线的客户端
            available_ids = [c_id for c_id in range(env_cfg.n_clients) if c_id not in crash_ids]
            # print(len(available_ids))
            # quota = math.ceil(len(available_ids) * env_cfg.pick_C)
            fast_ids, slow_ids, max_faster_T, max_slow_T, F_S_interval, last_quota = select_clients_ACFM(available_ids,
                                                                                                         clients_arrival_T,
                                                                                                         T_threshold,
                                                                                                         quota)

            time_trace.append(max_slow_T)
            T_threshold -= sum(clients_arrival_T) / len(clients_arrival_T)  # 时间阈值的选取(时间平均值) # 保留乘以一个参数（0.5、0.7）
            T_threshold_trace.append(T_threshold)
            F_S_interval_trace.append(F_S_interval)
            if T_threshold >= 0:
                time_trace.append(T_threshold)
            distribute_models(global_model, models, available_ids)
            # print('>fast_ids', fast_ids)
            # print('>slow_ids', slow_ids)
            # print('>max_faster_T', max_faster_T)
            # print('>max_slow_T', max_slow_T)
            # print('>F_S_interval', F_S_interval)
            for id_ in fast_ids:
                index = id_
                if index < len(client_shard_sizes):  # 确保索引在client_sharp_size列表的范围内
                    fast_ids_shard_size[id_] = client_shard_sizes[index]  # 字典
                    fast_ids_perf_vec[id_] = clients_perf_vec[id_]
            # print('>   @Cloud> fast_ids_shard_size:', fast_ids_shard_size)
            # print('>   @Cloud> fast_ids_perf_vec:', fast_ids_perf_vec)
            # undrafted_ids = [c_id for c_id in make_ids if c_id not in picked_ids]
            for id_ in slow_ids:
                index = id_
                if index < len(client_shard_sizes):  # 确保索引在client_sharp_size列表的范围内
                    slow_ids_shard_size[id_] = client_shard_sizes[index]
                    slow_ids_perf_vec[id_] = clients_perf_vec[id_]
            # print('>   @Cloud> slow_ids_shard_size:', slow_ids_shard_size)
            # print('>   @Cloud> slow_ids_perf_vec:', slow_ids_perf_vec)
            # tracing
            make_trace.append(available_ids)
            pick_trace.append(fast_ids)
            undrafted_trace.append(slow_ids)
            print('> Clients crashed: ', crash_ids)
    # Stats
    global_timer = event_handler.get_state('time')
    global_T_dist_timer = event_handler.get_state('T_dist')
    # Traces
    # print('> Train trace:')
    # utils.show_epoch_trace(epoch_train_trace, env_cfg.n_clients, plotting=False, cols=1)  # training trace
    # print('> Test trace:')
    # utils.show_epoch_trace(epoch_test_trace, env_cfg.n_clients, plotting=False, cols=1)
    # print('> Round trace:')
    # utils.show_round_trace(round_trace, plotting=env_cfg.showplot, title_='SAFA')

    # display timers
    print('\n> Experiment stats')
    print('> Clients round time:', client_timers)
    print('> Clients futile run time:', client_futile_timers)
    futile_pcts = (np.array(client_futile_timers) / np.array(client_timers)).tolist()
    print('> Clients futile percent (avg.=%.3f):' % np.mean(futile_pcts), futile_pcts)
    eu_ratio = eu_count / env_cfg.n_rounds / env_cfg.n_clients
    print('> EUR:', eu_ratio)
    sync_ratio = sync_count / env_cfg.n_rounds / env_cfg.n_clients
    print('> SR:', sync_ratio)
    version_var = version_var / env_cfg.n_rounds
    print('> VV:', version_var)
    print('> Total time consumption:', global_timer)
    print('> Total distribution time (T_dist):', global_T_dist_timer)
    print('> Loss = %.6f/at Round %d:' % (best_loss, best_rd))
    print('>round_trace', round_trace)
    print('> F_S_interval_trace:', F_S_interval_trace)
    print('> T_threshold_trace:', T_threshold_trace)
    print('> All categories round:fast, slow, global', fast_round, slow_round, global_round)
    print('time_trace:', time_trace)
    print('> Average time_trace:', sum(time_trace) / 100)

    # Logging
    detail_env = (client_shard_sizes, clients_perf_vec, clients_crash_prob_vec)
    # utils.log_stats('stats/exp_log.txt', env_cfg, detail_env, epoch_test_trace,
    #                 round_trace, acc_trace, make_trace, pick_trace, crash_trace, deprecated_trace, T_threshold_trace,
    #                 client_timers, client_futile_timers, global_timer, global_T_dist_timer, eu_ratio, sync_ratio,
    #                 version_var, best_rd, best_loss, extra_args={'lag_tolerance': lag_t}, log_loss_traces=False)
    utils.log_stats('../stats/T_threshold_analysis.txt', env_cfg, detail_env, time_trace, epoch_train_trace,
                    epoch_test_trace,
                    round_trace, acc_trace, make_trace, pick_trace, crash_trace, deprecated_trace,
                    client_timers, client_futile_timers, global_timer, global_T_dist_timer, eu_ratio, sync_ratio,
                    version_var, best_rd, best_loss, extra_args={'lag_tolerance': lag_t}, log_loss_traces=False)

    return best_model, best_rd, best_loss