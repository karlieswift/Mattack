"""
@Env: /anaconda3/python3.10
@Time: 2023/8/1-18:54
@Auth: karlieswift
@File: loss.py
@Desc: Loss function performance testing.
"""

import torch
from torch import nn


# 原始for循环
class DRSLX(nn.Module):
    def __init__(self, a=1, b=0.001):
        super(DRSLX, self).__init__()
        self.a = a
        self.b = b
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        print(1)
        ce = self.cross_entropy(pred, labels)
        labels_index = []
        # 取出非目标索引
        for i in labels:
            labels_index.append([False if x_i == i else True for x_i in range(pred.shape[1])])
        print(2)
        # 取出非目标索引对应的值
        neg_pre = pred[torch.tensor(labels_index)].reshape(-1, pred.shape[1] - 1)
        print(3)
        # 非目标索引对应的值-均值
        neg_pre_mean = neg_pre.mean(dim=1).reshape(neg_pre.shape[0], -1).repeat(1, neg_pre.shape[1])
        # 平方求和
        neg_loss = (neg_pre - neg_pre_mean) ** 2
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

# 排序取出
class DRSL1(nn.Module):
    def __init__(self, a=1, b=0.001,top_n=4):
        super(DRSL1, self).__init__()
        self.a = a
        self.b = b
        self.top_n = top_n
        self.cross_entropy = nn.CrossEntropyLoss()
    def forward(self, pred, labels):
        ce = self.cross_entropy(pred, labels)
        x_mean = torch.sort(pred)[0][:, -self.top_n:-1].mean(dim=1).reshape(pred.shape[0], -1)
        x = -nn.functional.log_softmax(x_mean, dim=0).sum()
        loss = self.a * ce + self.b * x
        return loss



# 改进的DRSL1
class DRSL2(nn.Module):
    def __init__(self, a=1, b=0.001,top_n=4):
        super(DRSL2, self).__init__()
        self.a = a
        self.b = b
        self.top_n = top_n
        self.cross_entropy = nn.CrossEntropyLoss()
    def forward(self, pred, labels):
        ce = self.cross_entropy(pred, labels)
        no_labels_one_hot = 1 - nn.functional.one_hot(labels, num_classes=pred.shape[1])
        no_pred = pred[no_labels_one_hot == 1].reshape(pred.shape[0], -1)#区别于DRSL1的代码
        x_mean = torch.sort(no_pred)[0][:, -self.top_n:-1].mean(dim=1).reshape(pred.shape[0], -1)
        x = -nn.functional.log_softmax(x_mean, dim=0).sum()
        loss = self.a * ce + self.b * x
        return loss
class DRSL3(nn.Module):
    def __init__(self, a=1, b=0.001, start=0, end=4):
        super(DRSL3, self).__init__()
        self.a = a
        self.b = b
        self.start = start
        self.end = end
        self.cross_entropy = nn.CrossEntropyLoss()
    def forward(self, pred, labels):
        ce = self.cross_entropy(pred, labels)
        no_labels_one_hot = 1 - nn.functional.one_hot(labels, num_classes=pred.shape[1])
        no_pred = pred[no_labels_one_hot == 1].reshape(pred.shape[0], -1)#区别于DRSL1的代码
        x_mean = torch.sort(no_pred,descending=True)[0][:,self.start:self.end].mean(dim=1).reshape(pred.shape[0], -1)
        x = -nn.functional.log_softmax(x_mean, dim=0).sum()
        loss = self.a * ce + self.b * x
        return loss


#
class DRSL31(nn.Module):
    def __init__(self, a=1, b=0.001, start=1, end=10):
        super(DRSL31, self).__init__()
        self.a = a
        self.b = b
        self.start = start
        self.end = end

        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        ce = self.cross_entropy(pred, labels)
        no_labels_one_hot = 1 - nn.functional.one_hot(labels, num_classes=pred.shape[1])
        no_pred = pred[no_labels_one_hot == 1].reshape(pred.shape[0], -1)  # 区别于DRSL1的代码
        no_pred_softmax = torch.tensor([0.] * self.top_n)
        # no_pred[no_pred< 0.01].reshape(-1, pred.shape[1] - 1)

        for i in range(no_pred.shape[0]):
            temp = no_pred[i]
            temp_softmax = torch.softmax(temp, dim=-1)
            index = torch.argwhere(temp_softmax < 0.001)
            a = torch.sort(temp[index])[0][:self.top_n]
            no_pred_softmax = torch.concat((no_pred_softmax, a.reshape(-1)))
        no_pred_softmax = no_pred_softmax.reshape(pred.shape[0], -1)
        x_mean = no_pred_softmax.mean(dim=1).reshape(pred.shape[0], -1)
        x = -nn.functional.log_softmax(x_mean, dim=0).sum()
        loss = self.a * ce + self.b * x
        return loss




eps = 1e-7
import torch.nn.functional as F
class SCELoss(nn.Module):
    def __init__(self, num_classes=50000, a=1, b=1):
        super(SCELoss, self).__init__()
        self.num_classes = num_classes
        self.a = a
        self.b = b
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        pred=pred[...,:-1]
        ce = self.cross_entropy(pred, labels)
        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=eps, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))
        loss = self.a * ce + self.b * rce.mean()
        return loss

if __name__ == '__main__':
    logits = torch.randn(size=(1, 30, 50001))
    labels = torch.randint(1, 4, size=(1, 30))
    logits = logits.contiguous().view(-1, logits.size(-1))
    labels = labels.contiguous().view(-1)

    import time
    loss_fct = DRSL1()
    t3 = time.time()
    loss = loss_fct(logits, labels)
    t4 = time.time()

    loss_fct=nn.CrossEntropyLoss()
    t1 = time.time()
    loss = loss_fct(logits, labels)
    t2 = time.time()


    loss_fct = SCELoss()
    t5 = time.time()
    loss = loss_fct(logits, labels)
    t6 = time.time()


    print("CE", t2 - t1)
    print("DRSL2", t4 - t3)
    print("DRSL1", t6 - t5)
    # print( (t2 - t1)/(t4 - t3))

