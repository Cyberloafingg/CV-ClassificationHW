
# 对于CB_loss的测试，构造了一个简单的数据集，数据集中有三个类别，使用CB_loss进行训练，以及交叉熵进行训练，对比两者的效果
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random
# 取消plt警告
import warnings
warnings.filterwarnings("ignore")
import tqdm

class My_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return F.binary_cross_entropy(x, y)

class CB_Loss(nn.Module):
    def __init__(self, samples_per_cls, no_of_classes, loss_type, beta, gamma):
        super(CB_Loss, self).__init__()
        self.samples_per_cls = samples_per_cls
        self.no_of_classes = no_of_classes
        self.loss_type = loss_type
        self.beta = beta
        self.gamma = gamma

    def forward(self, logits, labels):
        effective_num = 1.0 - torch.pow(self.beta, self.samples_per_cls)# 0.001,0.003,0.006
        weights = (1.0 - self.beta) / torch.tensor(effective_num)# 0.1,0.0334,0.0167
        weights = weights / torch.sum(weights) * self.no_of_classes # 1.999,0.667,0.334
        # q：下面这行在干啥
        # a：把每个类别的权重乘以该类别的样本数，这样就可以把每个类别的样本数考虑进去了
        lll = labels.unsqueeze(1)
        labels_one_hot = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), 1)
        weights = weights.unsqueeze(0).repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1).repeat(1, self.no_of_classes)
        if self.loss_type == "sigmoid":
            cb_loss = F.binary_cross_entropy_with_logits(input=logits,target=labels_one_hot, weight=weights)
        elif self.loss_type == "softmax":
            pred = logits.softmax(dim=1)
            cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)
        return cb_loss

def generate_data(num,per_cls=[0.05,0.4,0.55]):
    x_train = []
    y_train = []
    for i in range(num):
        x_ = []
        if i < num * 0.05:
            for i in range(3):
                x_.append(1)
            for i in range(7):
                x_.append(0)
            y_ = 0
        elif i < num * 0.4:
            for i in range(3):
                x_.append(0)
            for i in range(4):
                x_.append(1)
            for i in range(3):
                x_.append(0)
            y_ = 1
        else :
            for i in range(7):
                x_.append(0)
            for i in range(3):
                x_.append(1)
            y_ = 2
        x_train.append(x_)
        y_train.append(y_)
    x_train = np.array(x_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int64)
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)
    # 随机打乱
    index = [i for i in range(len(inputs))]
    random.shuffle(index)
    inputs = inputs[index]
    targets = targets[index]
    return inputs, targets

num = 1000
per_cls=[0.05,0.4,0.55]
inputs, targets = generate_data(num,per_cls)
input_size = 10
output_size = 3
num_epochs = 5000
learning_rate = 0.01
model = nn.Sequential(
    nn.Linear(input_size, 10),
    nn.ReLU(),
    nn.Linear(10, output_size)
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
inputs = inputs.to(device)
targets = targets.to(device)
model = model.to(device)

class_per_cls = np.array([num * i for i in per_cls])
class_per_cls = torch.from_numpy(class_per_cls)
class_per_cls = class_per_cls.to(device)
criterion = CB_Loss(class_per_cls, 3, "softmax", 0.0, 2.0)
# criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
loss_history = []
for epoch in tqdm.tqdm(range(num_epochs)):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss_history.append(loss.to('cpu').detach().numpy())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
plt.plot(loss_history)
plt.show()

model.eval()
model = model.to('cpu')
# 测试
inputs, targets = generate_data(100,per_cls)
outputs = model(inputs)
_, predicted = torch.max(outputs.data, 1)
targets = targets.resize(100,)
total = targets.size(0)
correct = (predicted == targets).sum().item()
print('Accuracy of the network on the 100 check  {} %'.format(100 * correct / total))
