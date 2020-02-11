import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

# data pre-processing
train_set = torchvision.datasets.MNIST(
    root=r'C:\Users\user\PycharmProjects\PyTorch\Datasets',
    download=True,
    train=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,))
    ])
)

test_set = torchvision.datasets.MNIST(
    root=r'C:\Users\user\PycharmProjects\PyTorch\Datasets',
    train=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,))
    ])
)

# data generator
batch_size = 50

train_gen = DataLoader(
    train_set,
    shuffle=True,
    batch_size=batch_size
)

# test data
test_batch_size = 32
test_data = {
    'x': test_set.test_data[:test_batch_size].view((test_batch_size, 1, 28, 28)).type(torch.float32) / 255.,
    'y': test_set.test_labels[:test_batch_size]
}


# set the CNN (Convolution Neural Network)
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()  # inheritance (父類別)

        '''
        @:info:
            input_size  : 1, 28, 28
            output_size : 10, 14, 14
        '''
        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, 10,  # channel: 1 -> 10
                      kernel_size=5,  # 5x5 convolutinal kernel
                      stride=1,  # move 1 pixel every time
                      padding=2,  # extend 2 zero-value pixels in image's edge 0,0,image,0,0
                      ),
            nn.ReLU(),  # activation function change all negative values to 0
            nn.MaxPool2d(2),  # pooling: 28x28 -> 14x14
        )

        '''
        @:info:
            input_size  : 10, 14, 14
            output_size : 32, 7, 7
        '''
        self.conv_2 = nn.Sequential(
            nn.Conv2d(10, 32,
                      kernel_size=5,
                      stride=1,
                      padding=2,
                      ),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.hidden_layer_1 = nn.Linear(1568, 64)  # 32x7x7 = 1568
        self.hidden_layer_2 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, 10)  # classification 10 class

    def forward(self, feature, batch_size):
        # print(feature.size())

        output = self.conv_1(feature)
        # print(output.size())

        output = self.conv_2(output)
        # print(output.size())

        output = output.view((batch_size, -1))  # flatten covoluted images
        # print(output.size())

        output = self.hidden_layer_1(output)
        output = self.hidden_layer_2(output)
        output = self.output_layer(output)
        # print(output.size())

        return output


net = Net()
loss_func = nn.CrossEntropyLoss()  # 交叉熵
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)  # Adam optimizer (recommend)


def compute_accuracy(predict, target):
    _, predict = torch.max(predict, 1)
    count = np.count_nonzero((predict == target).squeeze() == True)
    accuracy = count / target.size()[0]
    return accuracy


# record accuracies for chart
train_accuracys = []
test_accuracys = []

for epoch in range(1, 3):
    for step, (train_x, train_y) in enumerate(train_gen):  # generate data
        # print(train_x.size())
        # print(train_y.size())
        predict = net(train_x, batch_size)

        loss = loss_func(predict, train_y)  # compute loss
        optimizer.zero_grad()  # zeroing gradient last step
        loss.backward()  # backward propagation
        optimizer.step()  # update the parameters

        train_accuracy = compute_accuracy(predict, train_y)
        # print(train_accuracy)

        if (step + 1) % 10 == 0:  # plot every 10 steps
            train_accuracys.append(train_accuracy)
            predict = net(test_data['x'], test_batch_size)  # take a test
            test_accracy = compute_accuracy(predict, test_data['y'])  # compute the score
            test_accuracys.append(test_accracy)
            print(test_accracy)

            # paint the chart
            plt.cla()
            plt.plot(range(train_accuracys.__len__()), train_accuracys, 'r', label='train')
            plt.plot(range(test_accuracys.__len__()), test_accuracys, 'g', label='test')
            plt.legend()

            plt.pause(5e-4)
