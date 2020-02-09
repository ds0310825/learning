import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

train_set = torchvision.datasets. \
    MNIST(root=r'C:\Users\user\PycharmProjects\PyTorch\Datasets',
          train=True,
          download=True,
          transform=transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize((0.1,), (0.3,))
          ])
          )

test_set = torchvision.datasets. \
    MNIST(root=r'C:\Users\user\PycharmProjects\PyTorch\Datasets',
          train=False,
          transform=transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize((0.1,), (0.3,))
          ])
          )

batch_size = 128

train_gen = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
)

test_gen = DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=True,
)

print('data preprocessing done')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.input_layer = nn.Linear(784, 128)
        self.hidden_layer_1 = nn.Linear(128, 32)
        self.hidden_layer_2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layer_1(x)
        x = self.hidden_layer_2(x)
        x = F.softmax(x)
        return x


def compute_accuracy(predict, target):
    _, predict = torch.max(predict, 1)
    matched = (predict == target).squeeze()
    quantity = np.count_nonzero(matched.data.numpy() == True)
    accuracy = quantity / predict.__len__()

    return accuracy


net = Net()

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

print('start training')

acc_array = []

for epoch in range(3):
    for step, (train_x, train_y) in enumerate(train_gen):
        print('----------------------------------------------------')
        print('epoch:{}, step:{}   '.format(epoch, step), end='')
        train_x = torch.reshape(train_x, (-1, 784))
        # train_y = F.one_hot(train_y, 10)
        # train_y = torch.tensor(train_y, dtype=torch.float32)

        predict = net(train_x)

        optimizer.zero_grad()

        # print(predict.size())
        # print(train_y.size())

        loss = loss_func(predict, train_y)
        # print(loss.data)
        loss.backward()
        optimizer.step()
        # train_y = F.one_hot(train_y, 10)
        accuracy = compute_accuracy(predict, train_y)
        print('train_acc:', accuracy, end='')

        with torch.no_grad():
            for test_data in test_gen:
                test_x, test_y = test_data
                test_x = torch.reshape(test_x, (-1, 784))
                # test_y = F.one_hot(test_y, 10)

                test_predict = net(test_x)

                test_accuracy = compute_accuracy(test_predict, test_y)
                print('  valid_acc:', test_accuracy, )
                acc_array.append(test_accuracy)

                plt.cla()
                plt.plot(range(acc_array.__len__()), acc_array)
                plt.pause(0.05)

                break

        # print('Epoch:', i)
        # print('Step:', step)
        # print('x:', batch_x[0])
        # print('y:', batch_y[0])
print('done')
