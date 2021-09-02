import numpy as np
import torch
import torch.nn as nn
from torch import tensor
import torch.optim as optim
from visdom import Visdom


class EICULoss:
    def __init__(self, training=True, dataset_begin=0, dataset_end=6000):
        X = np.load("./eicu_x.npy")  # 这里dataset是提取出来的npy
        y = np.load("./eicu_y.npy")
        indices = np.array(range(dataset_begin, dataset_end))
        np.random.RandomState(10).shuffle(indices)
        if training:
            # print(f'训练集range为({dataset_begin}, {dataset_end})')
            pass
        else:
            indices = list(range(6000, 12000))
        self.X = tensor(X[indices], dtype=torch.float32)
        self.y = tensor(y[indices], dtype=torch.float32)  # 同步打乱X与y，并且转成tensor
        self.Xbatches = []
        self.ybatches = []
        self.cur_batch = 0

    def sample(self):
        if self.cur_batch >= len(self.Xbatches):
            self.Xbatches = []
            self.ybatches = []
            self.cur_batch = 0
            for b in range(0, 100):
                temp = b * 30
                self.Xbatches.append(self.X[temp:temp + 30])
                self.ybatches.append(self.y[temp:temp + 30])
        Xbatch = self.Xbatches[self.cur_batch]
        ybatch = self.ybatches[self.cur_batch]
        self.cur_batch += 1
        return Xbatch, ybatch


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.m1 = nn.Sequential(
            nn.Linear(1400, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3))

        self.m2 = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid())

    def forward(self, x):
        x = self.m1(x)
        x = self.m2(x)
        return x


class loss_visualize(object):
    def __init__(self, title='loss', env='loss'):
        self.viz = Visdom(env=env)
        epoch = 0
        self.title = title
        self.loss = self.viz.line(X=np.array([epoch]),
                                  Y=np.zeros([1, 1]),  # 2 stand train and valid
                                  opts=dict(legend=['train loss'],
                                            showlegend=True, title=self.title))

    def plot_loss(self, epoch, epoch_loss):
        train_loss = epoch_loss['train']
        loss = [train_loss]
        self.viz.line(
            X=np.array([epoch]),
            Y=loss,
            win=self.loss,  # win要保持一致
            update='append')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dataset = EICULoss()
    epochs = 100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net()
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    loss_func = nn.BCELoss()
    show_loss = loss_visualize()
    show_acc = loss_visualize(title='acc')
    for epoch in range(epochs):
        batch_loss = 0
        batch_acc = 0
        for i in range(int(12000/30)):
            data, label = dataset.sample()
            data, label = data.to(device), label.to(device)
            prediction = net(data).squeeze(1)
            train_loss = loss_func(prediction, label)
            batch_loss += train_loss

            prediction = prediction.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            prediction[np.where(prediction >= 0.5)] = 1
            prediction[np.where(prediction != 1)] = 0
            acc = 1 - np.mean(np.abs(prediction - label))
            batch_acc += acc

            optimizer.zero_grad()  # 梯度归零
            train_loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
        batch_loss, batch_acc = batch_loss.detach().cpu().numpy(), batch_acc
        epoch_loss = {'train': batch_loss/(i + 1)}
        epoch_acc = {'train': batch_acc/(i + 1)}
        show_loss.plot_loss(epoch, epoch_loss)
        show_acc.plot_loss(epoch, epoch_acc)
        stop = 1

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
