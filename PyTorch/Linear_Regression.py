import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

x_data = torch.unsqueeze(torch.linspace(-2, 2, 50), dim=1)

# x_data = torch.linspace(-1, 1, dsize, requires_grad=True).reshape(-1, 1)
# print(x_data)
y_data = x_data.pow(3) + 0.5 * torch.rand(x_data.size())


# print(y_data)

# plt.scatter(x_path, y_data)
# plt.show()


class Linear_Regression(nn.Module):
    def __init__(self, n_hidden_unit):
        super(Linear_Regression, self).__init__()
        self.hidden_1 = nn.Linear(1, n_hidden_unit)
        self.hidden_2 = nn.Linear(n_hidden_unit, n_hidden_unit*2)
        self.linear = nn.Linear(n_hidden_unit*2, 1)

    def forward(self, x):
        x = F.selu(self.hidden_1(x))
        x = F.selu(self.hidden_2(x))
        x = self.linear(x)
        return x


linear_regression = Linear_Regression(5)
print(linear_regression)

optimizer = torch.optim.Adam(linear_regression.parameters(),
                             lr=1e-3)
loss_func = torch.nn.MSELoss()


for i in range(10000):
    # inputs = Variable(x_data)
    predict = linear_regression(x_data)

    # print(x_data.view(-1))
    # print(predict)
    # print(y_data.view(-1))

    loss = loss_func(predict, y_data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i + 1) % 100 == 0:
        # predict = predict

        plt.cla()
        plt.scatter(x_data.view(-1).data.numpy(), y_data.view(-1).data.numpy())
        # print(predict)
        # print(predict.reshape(dsize))
        plt.scatter(x_data.view(-1).data.numpy(), predict.view(-1).data.numpy().reshape(50))
        plt.text(0.5, 0, 'loss=%.5f' % loss.data)
        plt.pause(0.01)
