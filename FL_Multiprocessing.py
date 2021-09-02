# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 14:11:57 2020

@author: ysxh1998
"""
from tqdm import tqdm#spyder要用的进度条就只用tqdm下的tqdm即可，jupyter的话需要tqdm_notebook
import time
# import torch.multiprocessing as mp
import multiprocessing as mp
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import random
import os.path
import csv
import copy
import joblib
from torchvision import datasets
import torchvision
import os
# import seaborn as sns; sns.set(color_codes=True)
# sns.set_style("white")
# sns.__version__

# def job2(q):
#     res = 0
#     for i in range(1000):
#         res += i+i**2+i**3
#     q.put(res)
    
# def job1(q,**kwargs):
#     if isinstance(kwargs,dict):
#         print(True)
#     q.put(kwargs)
    

# if __name__ == '__main__':
#     dict = torch.load('D:/ML/meta_learning_FL/fed_1.pth')
#     for i in range(5):
#         q = mp.Queue()
#         p1 = mp.Process(target=job1,args=(q,),kwargs=dict)
#         # p2 = mp.Process(target=job2,args=(q,))
#         p1.start()
#         # p2.start()
#         p1.join()
#         # p2.join()
#         res1 = q.get()
#         # res2 = q.get()
#         print(res1)
#         # print(res2)
#         # print(res1+res2)

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0")
def w(v):
    if USE_CUDA:
        return v.to(device)
    return v

def detach_var(v):
    var = w(Variable(v.data, requires_grad=True)) #w()是转cuda用的
    var.retain_grad()#保留该节点的梯度
    return var

def do_fit(dataset_begin, dataset_end, opt_net, meta_opt, \
            target_cls, target_to_opt, unroll, optim_it, n_epochs, out_mul, should_train=True):
    '''
    opt_net (the optimizer)
    meta_opt (the optimizer of the optimizer, ie Adam)
    target_cls (the optimizee's target) 损失函数
    optim_it(optimizee_iteration) 
    target_to_opt 要优化的optimizee
    '''
    if should_train:
        opt_net.train()
    else:
        opt_net.eval()
        unroll = 1
    # 初始化损失函数
    target = target_cls(training=should_train, dataset_begin=dataset_begin, dataset_end=dataset_end) 
    # 初始化optimizee
    optimizee = w(target_to_opt())
    n_params = 0
    for p in optimizee.parameters():
        #np.prod()通过元素乘积计算optimizee的总参数个数
        n_params += int(np.prod(p.size()))
    # 生成两个这样的Variable装在list里
    # hidden_sz是hidden_size,这两行是要输入给lstm的hidden_state and cell_state
    hidden_states = [w(torch.zeros(n_params, opt_net.hidden_sz)) for _ in range(2)] 
    cell_states = [w(torch.zeros(n_params, opt_net.hidden_sz)) for _ in range(2)] 
    all_losses_ever = []
    if should_train:
        # 网络梯度清零
        meta_opt.zero_grad()
    all_losses = None
    for iteration in range(1, optim_it + 1):
        # 求出loss
        loss = optimizee(target)
        if all_losses is None:
            all_losses = loss
        else:
            all_losses += loss
        # Get the optimizee's gradients. We retain the graph because we'll need to run
        # backprop again when we optimize the optimizer.
        all_losses_ever.append(loss.data.cpu().numpy())

        # 反向传播，保留计算图（默认会释放计算图）
        loss.backward(retain_graph=should_train)

        # Update each parameters and the cell and hidden states by iterating through the
        # optimizee's "all_named_parameters".
        offset = 0
        result_params = {}
        hidden_states2 = [w(torch.zeros(n_params, opt_net.hidden_sz)) for _ in range(2)]
        cell_states2 = [w(torch.zeros(n_params, opt_net.hidden_sz)) for _ in range(2)]
        for name, p in optimizee.all_named_parameters():
            cur_sz = int(np.prod(p.size()))
            # We do this so the gradients are disconnected from the graph but we still get
            # gradients from the rest
            # 得到梯度 cur_sz -> (cur_sz, 1)
            gradients = detach_var(p.grad.view(cur_sz, 1))
            # 当前的gradients对应h[offset:offset+cur_sz]
            updates, new_hidden, new_cell = opt_net(
                gradients,
                [h[offset:offset+cur_sz] for h in hidden_states],
                [c[offset:offset+cur_sz] for c in cell_states]
            )
            for i in range(len(new_hidden)):
                hidden_states2[i][offset:offset+cur_sz] = new_hidden[i]
                cell_states2[i][offset:offset+cur_sz] = new_cell[i]#分批更新hidden和cell参数
            
            result_params[name] = p + updates.view(*p.size()) * out_mul
            result_params[name].retain_grad()

            offset += cur_sz

        if iteration % unroll == 0:
            '''
              如果should_train=True，则更新optimizer的参数
            '''
            if should_train:
                meta_opt.zero_grad()
                all_losses.backward()
                meta_opt.step()

            all_losses = None

            optimizee = w(target_to_opt(**{k: detach_var(v) for k, v in result_params.items()}))
            hidden_states = [detach_var(v) for v in hidden_states2]
            cell_states = [detach_var(v) for v in cell_states2]

        else:
            optimizee = w(target_to_opt(**result_params))
            assert len(list(optimizee.all_named_parameters()))
            hidden_states = hidden_states2
            cell_states = cell_states2

    return all_losses_ever

# @cache.cache
def fit_optimizer(dataset_begin, dataset_end, target_cls, target_to_opt, global_model=None, preproc=False, unroll=20, \
                  optim_it=40, n_epochs=20, n_tests=100, lr=0.001, out_mul=1.0):
    if global_model == None:
        opt_net = w(Optimizer(preproc=preproc))
    else:
        opt_net = w(Optimizer(preproc=preproc))
        opt_net.load_state_dict(global_model)
    meta_opt = optim.Adam(opt_net.parameters(), lr=lr)

    best_net = None
    best_loss = 100000000000000000

    for _ in tqdm(range(n_epochs), 'epochs'):
        for _ in tqdm(range(20), 'iterations'):
            # 进行20个iterations的训练
            # 在每个iteration中，会对optimizee进行optim_it(100)次训练，每(unroll)20次更新一下optimizer
            do_fit(dataset_begin, dataset_end, opt_net, meta_opt, target_cls, target_to_opt, \
                    unroll, optim_it, n_epochs, out_mul, should_train=True)
        # 然后进行n_test(100)次测试
        loss = (np.mean([
            np.sum(do_fit(dataset_begin, dataset_end, opt_net, meta_opt, target_cls, target_to_opt, \
                          unroll, optim_it, n_epochs, out_mul, should_train=False))
            for _ in tqdm(range(n_tests), 'tests')
        ]))
        # 把测试结果输出
        print(loss)
        if loss < best_loss:
            print(best_loss, loss)
            best_loss = loss
            best_net = copy.deepcopy(opt_net.state_dict())

    return best_loss, best_net

class Optimizer(nn.Module):
    def __init__(self, preproc=False, hidden_sz=20, preproc_factor=10.0):
        super().__init__()
        self.hidden_sz = hidden_sz
        if preproc:
            self.recurs = nn.LSTMCell(2, hidden_sz)
        else:
            self.recurs = nn.LSTMCell(1, hidden_sz)
        self.recurs2 = nn.LSTMCell(hidden_sz, hidden_sz)
        self.output = nn.Linear(hidden_sz, 1)
        self.preproc = preproc
        self.preproc_factor = preproc_factor
        self.preproc_threshold = np.exp(-preproc_factor)

    def forward(self, inp, hidden, cell):
        if self.preproc:
            # Implement preproc described in Appendix A
            # Note: we do all this work on tensors, which means
            # the gradients won't propagate through inp. This
            # should be ok because the algorithm involves
            # making sure that inp is already detached.
            inp = inp.data
            inp2 = w(torch.zeros(inp.size()[0], 2))
            keep_grads = (torch.abs(inp) >= self.preproc_threshold).squeeze()
            inp2[:, 0][keep_grads] = (torch.log(torch.abs(inp[keep_grads]) + 1e-8) / self.preproc_factor).squeeze()
            inp2[:, 1][keep_grads] = torch.sign(inp[keep_grads]).squeeze()

            inp2[:, 0][~keep_grads] = -1
            inp2[:, 1][~keep_grads] = (float(np.exp(self.preproc_factor)) * inp[~keep_grads]).squeeze()
            inp = w(Variable(inp2))
        hidden0, cell0 = self.recurs(inp, (hidden[0], cell[0]))
        hidden1, cell1 = self.recurs2(hidden0, (hidden[1], cell[1]))
        return self.output(hidden1), (hidden0, hidden1), (cell0, cell1)

class MNISTLoss:
    def __init__(self, training=True, dataset_begin=0, dataset_end=30000):
        dataset = datasets.MNIST(
            root= 'E:/mnist', train=True, download=False,
            transform=torchvision.transforms.ToTensor()
        )
        indices = list(range(dataset_begin, dataset_end))
        np.random.RandomState(10).shuffle(indices)
        if training:
            # print(f'训练集range为({dataset_begin}, {dataset_end})')
            pass
        else:
            indices = list(range(30000, 33000))
        self.loader = torch.utils.data.DataLoader(
            dataset, batch_size=128,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))
        self.batches = []
        self.cur_batch = 0
        
    def sample(self):
        if self.cur_batch >= len(self.batches):
            self.batches = []
            self.cur_batch = 0
            for b in self.loader:
                self.batches.append(b)
        batch = self.batches[self.cur_batch]
        self.cur_batch += 1
        return batch

class MNISTNet(nn.Module):
    def __init__(self, layer_size=5, n_layers=1,**kwargs):#layer_size=20
        super().__init__()
        # Sadly this network needs to be implemented without using the convenient pytorch
        # abstractions such as nn.Linear, because afaik there is no way to load parameters
        # in those in a way that preserves gradients.

        if kwargs != {}:
            self.params = kwargs
        else:
            inp_size = 28*28
            self.params = {}
            for i in range(n_layers):
                self.params[f'mat_{i}'] = nn.init.xavier_normal_(\
                                                                  nn.Parameter(torch.randn(inp_size, layer_size) * 0.001)) # TODO: use a better initialization
                self.params[f'bias_{i}'] = nn.Parameter(torch.zeros(layer_size))
                inp_size = layer_size

            self.params['final_mat'] = nn.init.xavier_normal_(\
                                                              nn.Parameter(torch.randn(inp_size, 10) * 0.001)) # TODO:init #I have found a better way to initialize
            self.params['final_bias'] = nn.Parameter(torch.zeros(10))
            
            # Put it all in a module list so that ordinary optimizers can find them.
            self.mods = nn.ParameterList()
            for v in self.params.values():
                self.mods.append(v)
          
        self.activation = nn.Sigmoid()
        self.loss = nn.NLLLoss()
                
    def all_named_parameters(self):
        return [(k, v) for k, v in self.params.items()]
    
    def forward(self, loss):
        inp, out = loss.sample()
        inp = w(Variable(inp.view(inp.size()[0], 28*28)))
        out = w(Variable(out))
        
        cur_layer = 0
        while f'mat_{cur_layer}' in self.params:
            inp = self.activation(torch.matmul(inp, self.params[f'mat_{cur_layer}']) + self.params[f'bias_{cur_layer}'])
            cur_layer += 1
                    
        inp = F.log_softmax(torch.matmul(inp, self.params['final_mat']) + self.params['final_bias'], dim=1)
        l = self.loss(inp, out)
        return l
# global_model = torch.load(os.path.join(model_dir,'fed_1.pth'))
loss, mnist_optimizer = fit_optimizer(24000, 27000, MNISTLoss, MNISTNet, lr=0.01, n_epochs=5, n_tests=15, out_mul=0.1, preproc=True)

def average_weights(w):
  """
  Return the average of the weights.
  """
  w_avg = copy.deepcopy(w[0])#为什么用这句会有'function' object has no attribute 'deepcopy'?
  #w_avg = w[0]
  for key in w_avg.keys():
    for i in range(1,len(w)):
      w_avg[key] += w[i][key]
    w_avg[key] = torch.div(w_avg[key],len(w))
  return w_avg

def process(q):
    print('process',os.getpid(), os.getppid())
    # _,weight = fit_optimizer(0,3000,MNISTLoss, MNISTNet,lr=0.01, n_epochs=5, n_tests=15, out_mul=0.1, preproc=True)
    opt_net = w(Optimizer(preproc=True))
    meta_opt = optim.Adam(opt_net.parameters(), lr=0.001)
    all_losses = do_fit(0, 3000, opt_net, meta_opt, MNISTLoss, MNISTNet, \
                    20, 40, 20, 1, should_train=True)
    print(all_losses)
    # print(weight)
    # q.put(weight)
    
# def use_gpu(q):
#     t = []
#     print(os.getpid())
#     a = torch.randn(10,10).cuda(0)
#     print(a)
#     # t.append(a)
#     # q.put(a)

mp.set_start_method('spawn',force=True)

if __name__ == '__main__':
    # global_weights = torch.load(os.path.join(model_dir,'fed_1.pth'))



    for i in range(5):
        print('main',os.getpid(), os.getppid())
        q = mp.Queue()
        p = mp.Process(target=process,args=(q,)) 
#         p1 = mp.Process(target=process1,args=(q,),kwargs=global_weights)
#         p2 = mp.Process(target=process2,args=(q,global_weights))
#         p3 = mp.Process(target=process3,args=(q,global_weights))
        p.start()
#         p1.start()
#         p2.start()
#         p3.start()
        p.join()
#         p1.join()
#         p2.join()
#         p3.join()
        # weights_of_silos = q.get()
#         weights_of_silo2 = q.get()
#         weights_of_silo3 = q.get()
        # print(weights_of_silos)
#         for weight in weights_of_silo2:
#             weights_of_silos.append(weight)
#         for weight in weights_of_silo3:
#             weights_of_silos.append(weight)
#         global_weights = average_weights(weights_of_silos)
#         federated_model = os.path.join(model_dir,f'fed_{i+2}.pth')
#         torch.save(global_weights,federated_model)









































