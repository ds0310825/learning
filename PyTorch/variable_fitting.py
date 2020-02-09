import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.predict = nn.Linear(1, 1)

    def forward(self, x):
        out = self.predict(x) * line_data
        return out


net = Net()

loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=3e-3)

x_constant = torch.ones(1)

line_data = torch.unsqueeze(torch.linspace(-1, 1, 50), dim=1)

y_goal = 5.2 * line_data + 1 * (0.5 - torch.rand(line_data.size()))
# print(torch.rand(line_data.size()))
# print(y_goal.size())

plt.ion()

for i in range(10000):
    predict = net(x_constant)

    # print(y_predict.size())

    loss = loss_func(y_goal, predict)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i + 1) % 50 == 0:
        print(net)

        plt.cla()
        plt.title('loss:%.3f' % loss.data.numpy())
        plt.scatter(line_data.data.numpy(), y_goal.data.numpy())
        plt.scatter(line_data.data.numpy(), predict.data.numpy())
        plt.pause(0.001)
