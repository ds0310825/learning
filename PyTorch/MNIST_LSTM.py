import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataloader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')  # ignore test_data & test_labels warning

train_set = MNIST(
    root=r'C:\Users\user\PycharmProjects\PyTorch\Datasets',
    train=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,))  # normalization
    ])
)

test_set = MNIST(
    root=r'C:\Users\user\PycharmProjects\PyTorch\Datasets',
    train=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,))
    ])
)

batch_size = 32

train_gen = dataloader.DataLoader(
    train_set,
    shuffle=True,
    batch_size=batch_size
)

test_data = {
    'x': test_set.test_data[:100].view((-1, 28, 28)).type(torch.float32) / 255.,
    'y': test_set.test_labels[:100]
}


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.lstm = nn.LSTM(
            input_size=28,
            hidden_size=84,
            num_layers=1,
            batch_first=True
        )

        self.out = nn.Linear(84, 10)

    def forward(self, x):
        x, (hidden_state, cell_state) = self.lstm(x, None)

        # x.size(): [batch_size, time, hidden_size]
        # x[:, -1, :] == hidden_state[-1, :, :]

        x = self.out(x[:, -1, :])
        return x


net = Net()

print(net)

optimizer = torch.optim.Adam(net.parameters(), lr=10e-3)
loss_func = nn.CrossEntropyLoss()


def compute_accuracy(predict, target):
    _, predict = torch.max(predict, 1)
    count = np.count_nonzero((predict == target).squeeze() == True)
    accuracy = count / predict.size()[0]

    return accuracy


train_accs = []
test_accs = []

for epoch in range(3):
    for step, (train_x, train_y) in enumerate(train_gen):
        train_x = train_x.view((-1, 28, 28)).type(torch.float32)

        predict = net(train_x)

        loss = loss_func(predict, train_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 25 == 0:
            train_accs.append(compute_accuracy(predict, train_y))
            # record train accuracy

            predict = net(test_data['x'])
            accuracy = compute_accuracy(predict, test_data['y'])
            test_accs.append(accuracy)
            # record test accuracy

            plt.cla()  # clear last state of chart

            plt.plot(range(len(train_accs)), train_accs, label='train')
            plt.plot(range(len(test_accs)), test_accs, label='test')

            plt.legend()

            plt.pause(0.05)

