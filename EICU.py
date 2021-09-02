# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 09:30:58 2020

@author: ysxh1998
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets
import torchvision
from sklearn.metrics import f1_score
from torch import tensor
USE_CUDA = torch.cuda.is_available()

csv_data = pd.read_csv(r"D:\ML\meta_learning_FL\Data_table_cleaned.csv")
csv_data = np.array(csv_data)

#取前三千行处理
hospid = csv_data[:,0]
y_true = csv_data[:,1]
x_true = csv_data[:,2:]

print(np.argmax(np.bincount(delete_338)))

where_73 = np.argwhere(hospid==73)#6724

where_167 = np.argwhere(hospid==167)#5544

where_264 = np.argwhere(hospid==264)#5091

where_420 = np.argwhere(hospid==420)#4601

where_338 = np.argwhere(hospid==338)#4216

where_243 = np.argwhere(hospid==243)#4139

#
s_1 = where_73[:4000]
x_1 = x_true[s_1]
y_1 = y_true[s_1]
x_1 = np.squeeze(x_1,1)
y_1 = np.squeeze(y_1,1)

s_2 = where_167[:4000]
x_2 = x_true[s_2]
y_2 = y_true[s_2]
x_2 = np.squeeze(x_2,1)
y_2 = np.squeeze(y_2,1)

s_3 = where_264[:4000]
x_3 = x_true[s_3]
y_3 = y_true[s_3]
x_3 = np.squeeze(x_3,1)
y_3 = np.squeeze(y_3,1)

s_4 = where_420[:4000]
x_4 = x_true[s_4]
y_4 = y_true[s_4]
x_4 = np.squeeze(x_4,1)
y_4 = np.squeeze(y_4,1)

s_5 = where_338[:4000]
x_5 = x_true[s_5]
y_5 = y_true[s_5]
x_5 = np.squeeze(x_5,1)
y_5 = np.squeeze(y_5,1)


#

#一个silo6063个
# silo_1 = np.vstack((where_73[:519],where_167))

# silo_2 = np.vstack((where_73[519:1491],where_264))

# silo_3 = np.vstack((where_73[1491:2953],where_420))

# silo_4 = np.vstack((where_73[2953:4800],where_338))

# silo_5 = np.vstack((where_73[4800:],where_243))

#
# x_1 = x_true[silo_1]
# y_1 = y_true[silo_1]
# x_1 = np.squeeze(x_1,1)
# y_1 = np.squeeze(y_1,1)

# x_2 = x_true[silo_2]
# y_2 = y_true[silo_2]
# x_2 = np.squeeze(x_2,1)
# y_2 = np.squeeze(y_2,1)

# x_3 = x_true[silo_3]
# y_3 = y_true[silo_3]
# x_3 = np.squeeze(x_3,1)
# y_3 = np.squeeze(y_3,1)

# x_4 = x_true[silo_4]
# y_4 = y_true[silo_4]
# x_4 = np.squeeze(x_4,1)
# y_4 = np.squeeze(y_4,1)

# x_5 = x_true[silo_5]
# y_5 = y_true[silo_5]
# x_5 = np.squeeze(x_5,1)
# y_5 = np.squeeze(y_5,1)

np.save(r"D:\ML\meta_learning_FL\eicu_data\x_1.npy",x_1)
np.save(r"D:\ML\meta_learning_FL\eicu_data\y_1.npy",y_1)

np.save(r"D:\ML\meta_learning_FL\eicu_data\x_2.npy",x_2)
np.save(r"D:\ML\meta_learning_FL\eicu_data\y_2.npy",y_2)

np.save(r"D:\ML\meta_learning_FL\eicu_data\x_3.npy",x_3)
np.save(r"D:\ML\meta_learning_FL\eicu_data\y_3.npy",y_3)

np.save(r"D:\ML\meta_learning_FL\eicu_data\x_4.npy",x_4)
np.save(r"D:\ML\meta_learning_FL\eicu_data\y_4.npy",y_4)

np.save(r"D:\ML\meta_learning_FL\eicu_data\x_5.npy",x_5)
np.save(r"D:\ML\meta_learning_FL\eicu_data\y_5.npy",y_5)

#################
condition = (hospid==73)|(hospid==167)|(hospid==264)|(hospid==420)|(hospid==338)|(hospid==243)
where_eicu = np.argwhere(condition)

y_eicu = y_true[where_eicu]
x_eicu = x_true[where_eicu]
hospid_eicu = hospid[where_eicu]

np.save(r"D:\ML\meta_learning_FL\Elabel.npy",y_eicu)
np.save(r"D:\ML\meta_learning_FL\Epara.npy",x_eicu)
#测试一下原来MNISTLoss是怎样产生样本的
def w(v):
    if USE_CUDA:
        return v.cuda(0)
    return v


class MNISTLoss:
    def __init__(self, training=True, dataset_begin=0, dataset_end=30000):
        dataset = datasets.MNIST(
            root=r'E:\mnist', train=True, download=True,
            transform=torchvision.transforms.ToTensor()
        )
        indices = list(range(dataset_begin, dataset_end))
        np.random.RandomState(10).shuffle(indices)
        if training:
            # print(f'训练集range为({dataset_begin}, {dataset_end})')
            pass
        else:
            indices = list(range(30000, 60000))#原来30000~33000曲线波动太大了，会有阶梯状
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

inp, out = MNISTLoss().sample()
inp = Variable(inp.view(inp.size()[0], 28*28))

class MNISTNet(nn.Module):
    def __init__(self, layer_size=20, n_layers=1,**kwargs):#layer_size=20
        super().__init__()
        # Sadly this network needs to be implemented without using the convenient pytorch
        # abstractions such as nn.Linear, because afaik there is no way to load parameters
        # in those in a way that preserves gradients.


#        print('layer_size:', layer_size, 'n_layers:', n_layers)

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
        _,predicted = torch.max(inp,1)
        acc = (predicted==out).float().mean()#.float()很重要
        f1 = f1_score(out.cpu(), predicted.cpu(), average='macro')
        l = self.loss(inp, out)
        return l,acc,f1

#嘿嘿，看起来MNISTNet最后会在经过网络前把数据打包为(128,28*28)的格式，我现在只需要把格式换为(20,1400)就好，而且MNISTLoss中直接去调用保存下来的npy文件
#，按原来的形式设置对train和test的判断，并且依照原有的dataloader自己写一个iteration就好。
#tensor不能像numpy一样可以用另一个numpy数组做索引

class EICULoss:
    def __init__(self,training=True, dataset_begin=0,dataset_end=1500):
        X = np.load(r"D:\ML\meta_learning_FL\Epara.npy")#这里dataset是提取出来的npy
        y = np.load(r"D:\ML\meta_learning_FL\Elabel.npy")
        indices = np.array(range(dataset_begin,dataset_end))
        np.random.RandomState(10).shuffle(indices)
        if training:
            # print(f'训练集range为({dataset_begin}, {dataset_end})')
            pass
        else:
            indices = list(range(1500, 3000))
        self.X = tensor(X[indices])
        self.y = tensor(y[indices])#同步打乱X与y，并且转成tensor
        self.Xbatches = []
        self.ybatches = []
        self.cur_batch = 0
        
    def sample(self):
        if self.cur_batch >= len(self.Xbatches):
            self.Xbatches = []
            self.ybatches = []
            self.cur_batch = 0
            for b in range(0,75):
                temp = b*20
                self.Xbatches.append(self.X[temp:temp+20])
                self.ybatches.append(self.y[temp:temp+20])
        Xbatch = self.Xbatches[self.cur_batch]
        ybatch = self.ybatches[self.cur_batch]
        self.cur_batch += 1
        return Xbatch, ybatch
    
class EICUNet:
    def __init__(self, layer_size=20, n_layers=1,**kwargs):#layer_size=20
        super().__init__()
        # Sadly this network needs to be implemented without using the convenient pytorch
        # abstractions such as nn.Linear, because afaik there is no way to load parameters
        # in those in a way that preserves gradients.


#        print('layer_size:', layer_size, 'n_layers:', n_layers)

        if kwargs != {}:
            self.params = kwargs
        else:
            inp_size = 1400
            self.params = {}
            for i in range(n_layers):
                self.params[f'mat_{i}'] = nn.init.xavier_normal_(\
                                                                 nn.Parameter(torch.randn(inp_size, layer_size) * 0.001)) # TODO: use a better initialization
                self.params[f'bias_{i}'] = nn.Parameter(torch.zeros(layer_size))
                inp_size = layer_size

            self.params['final_mat'] = nn.init.xavier_normal_(\
                                                              nn.Parameter(torch.randn(inp_size, 2) * 0.001)) # TODO:init #I have found a better way to initialize
            self.params['final_bias'] = nn.Parameter(torch.zeros(2))

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
        inp = w(Variable(inp.view(inp.size()[0], 1400)))
        out = w(Variable(out))

        cur_layer = 0
        while f'mat_{cur_layer}' in self.params:
            inp = self.activation(torch.matmul(inp, self.params[f'mat_{cur_layer}']) + self.params[f'bias_{cur_layer}'])
            cur_layer += 1

        inp = F.log_softmax(torch.matmul(inp, self.params['final_mat']) + self.params['final_bias'], dim=1)
        _,predicted = torch.max(inp,1)
        acc = (predicted==out).float().mean()#.float()很重要
        f1 = f1_score(out.cpu(), predicted.cpu(), average='macro')
        l = self.loss(inp, out)
        return l,acc,f1




































































