import random

import torch


class args_parser():
    def __init__(self, load_dict=None):
        # -----------------FL parameter---------------

        # mode: Primal FedAvg, Semi-Asynchronous FedAvg， ASA-FL, FedCS
        # self.mode = None
        self.mode = 'Semi-Asynchronous FedAvg'
        # float(sys.argv[1])  # E(cr)
        self.cr_prob = 0.1
        # int(sys.argv[2])  # lag tolerance, for SAFA
        self.lag_tol = 5
        # define 最大能容忍的陈旧度
        self.lag_t_max = 3
        # number of client for waiting at one round
        self.wait_num = 5

        # number of global round
        self.n_rounds = 50  # 100
        # Total latency limit

        # 训练集数据占比
        self.train_pct = 0.7
        # 测试集数据占比
        self.test_pct = 1 - self.train_pct
        # 是否随机扰乱数据
        self.shuffle = False
        # #选择客户端比例
        self.pick_pct = 0.3  # 0.02, 0.3

        '''
        data_dist: local data size distribution, valid options include:
            ('E',None): equal-size partition, local size = total_size / n_clients
            ('N',rlt_sigma): partition with local sizes following normal distribution, mu = total_size/n_clients,
                sigma = rlt_sigma * mu
            ('X',None): partition with local sizes following exponential distribution, lambda = n_clients/total_size
        '''
        self.data_dist = ('N', 0.3)  # data (size) distribution
        '''
        perf_dist: client performance distribution (unit: virtual time per batch), valid options include:
            ('E',None): equal performance, perf = 1 unit
            ('N',rlt_sigma): performances follow normal distribution, mu = 1, sigma = rlt_sigma * mu
            ('X',None): performances follow exponential distribution, lambda = 1/1
        '''
        # self.perf_dist = ('X', None)  # client performance distribution
        self.perf_dist = ('N', 0.3)  # client performance distribution
        '''
        crash_dist: client crash prob. distribution, valid options include:
            ('E',prob): equal probability to crash, crash_prob = prob
            ('U',(low, high)): uniform distribution between low and high
        '''

        self.crash_dist = ('E', self.cr_prob)  # client crash probability distribution 客户端崩溃概率分布
        # if to keep best
        self.keep_best = True
        # if to show plot
        self.showplot = False

        # runtime
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        # task type can be 'Reg','CNN','SVM'
        self.task_type = 'Reg'

        if self.task_type == 'Reg':
            # number of clients,can be 5，100
            self.n_clients = 50
            # float(sys.argv[3])  # select percent
            self.pick_C = 0.3
            self.dataset = 'Boston'
            self.path = 'data/housing.csv'
            self.in_dim = 8  # 12
            self.out_dim = 1
            self.lr = 1e-4
            # number of local iter
            self.n_epochs = 5  # 3
            # mini-batch size
            self.batch_size = 64  # 5 batch_size = -1 means full local data set as a mini-batch
            self.loss = 'mse'
            self.T_max = 2000
            self.alpha = 0.1
            self.beta = 0.5
        elif self.task_type == 'CNN':
            # number of clients,can be 5，100
            self.n_clients = 100
            # float(sys.argv[3])  # select percent
            self.pick_C = 0.3
            self.dataset = 'data/fashion-mnist'
            self.lr = 1e-3
            # number of local iter
            self.n_epochs = 5
            self.in_dim = 28
            self.out_dim = 10
            # mini-batch size
            self.batch_size = 40  # batch_size = -1 means full local data set as a mini-batch
            self.loss = 'nllLoss'
            self.T_max = 3000
            self.alpha = 0.07
            self.beta = 0.09
        elif self.task_type == 'SVM':
            # number of clients,can be 5，100
            self.n_clients = 600  # 500
            # float(sys.argv[3])  # select percent
            self.pick_C = 0.3
            self.dataset = 'tcpdump99'
            self.path = 'data/kddcup_10'
            self.in_dim = 35
            self.out_dim = 1
            self.lr = 1e-2
            # number of local iter
            self.n_epochs = 5
            # mini-batch size
            self.batch_size = 100  # batch_size = -1 means full local data set as a mini-batch
            self.loss = 'svmLoss'
            self.T_max = 3000  # 5000
            self.alpha = 0.01
            self.beta = 0.9

        # bandwidth vector 带宽向量
        self.bw_set = (0.175, 1250)  # (client throughput, bandwidth_server) in MB/s
        self.bw_s_set = 1250
        self.bw_c_set = [round(random.uniform(0.125, 0.3), 3) for _ in range(self.n_clients)]
        self.optimizer = 'SGD'

        self.lr_decay = 1.0
        self.model_size = 10.0  # 10MB
        # factor control the discount of time

        self.varsigma = 0.9

        # result store path
        self.RL_path = './result/RL_{}_{}.npz'.format(self.data_dist, self.task_type)
        self.reward_path = './result/RL_reward_{}_{}_.npz'.format(self.data_dist, self.task_type)
        self.actor_path = './actor_model_{}_{}'.format(self.data_dist, self.task_type)

        # parameter for Oort
        self.T_limit = self.T_max / 50

        # parameter for test
        self.load = False
        self.wait_num_fixed = False
        self.lag_t_fixed = False
        if self.wait_num_fixed:
            self.wait_num = 25
        if self.lag_t_fixed:
            self.lag_tol = 2
        if self.load:
            self.actor_path = './actor_model{}{}_ep999_runs1'.format(self.task_type, self.data_dist)
