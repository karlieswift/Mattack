"""
@Env: /anaconda3/python3.10
@Time: 2023/10/17-23:50
@Auth: karlieswift
@File: lossfunctions.py
@Desc:
"""

import torch.nn as nn
import torch


class NewLoss(nn.Module):
    def __init__(self,a=1,b=0.005):
        super(NewLoss, self).__init__()
        self.a=a
        self.b=b
        self.cross_entropy = nn.CrossEntropyLoss()
    def forward(self, pred, labels):
        ce = self.cross_entropy(pred, labels)
        labels_index = []
        # 取出非目标索引
        for i in labels:
            labels_index.append([False if x_i == i else True for x_i in range(pred.shape[1])])
        # 取出非目标索引对应的值
        neg_pre=pred[torch.tensor(labels_index)].reshape(-1, pred.shape[1] - 1)
        # 非目标索引对应的值-均值
        # neg_pre=F.log_softmax(neg_pre,dim=1) #这一行代码加不加一样，可以化简（减少幂指数运算）
        neg_pre_mean=neg_pre.mean(dim=1).reshape(neg_pre.shape[0],-1).repeat(1, neg_pre.shape[1])
        # 平方求和
        neg_loss=(neg_pre-neg_pre_mean)**2
        loss = self.a * ce + self.b * neg_loss.sum()
        return loss

class DRSL(nn.Module):
    def __init__(self, a=1, b=0.001):
        super(DRSL, self).__init__()
        self.a = a
        self.b = b
        self.cross_entropy = nn.CrossEntropyLoss()
    def forward(self, pred, labels):
        no_labels_one_hot = 1 - nn.functional.one_hot(labels, num_classes=pred.shape[1])
        ce = self.cross_entropy(pred, labels)
        neg_pre_mean = torch.mul(no_labels_one_hot, pred).mean(dim=1).reshape(-1, 1)
        neg_pre = pred[no_labels_one_hot == 1].reshape(-1, pred.shape[1] - 1)
        neg_loss = (neg_pre - neg_pre_mean) ** 2
        loss = self.a * ce + self.b * neg_loss.sum()
        return loss

class PRSL(nn.Module):
    def __init__(self, a=1, b=0.001, start=0, end=4 ,ignore_index=-100):
        super(PRSL, self).__init__()
        self.a = a
        self.b = b
        self.start = start
        self.end = end
        self.ignore_index = ignore_index
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

    def forward(self, pred, labels):
        ce = self.cross_entropy(pred, labels)
        pred = pred[(labels.reshape(-1, 1) != self.ignore_index).repeat(1, pred.shape[1])].reshape(-1, pred.shape[-1])
        no_pred = pred[~(
                torch.arange(pred.shape[1]).reshape(1, -1).repeat(pred.shape[0], 1).to(pred.device)
                == labels[labels != self.ignore_index].reshape(-1, 1)
        )].reshape(pred.shape[0], -1)
        x_mean = torch.sort(no_pred, descending=True)[0][:, self.start:self.end].mean(dim=1).reshape(no_pred.shape[0],                                                                                       -1)
        x = -nn.functional.log_softmax(x_mean, dim=0).sum()
        loss = self.a * ce + self.b * x
        return loss
