import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

x = torch.tensor(
    [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]], dtype=torch.float32)
y = torch.tensor(
    [0, 1, 1, 0], dtype=torch.float32
)


# plt.scatter(x[:, 0], x[:, 1], c=y)
# plt.pause(2)


class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.output_layer = nn.Linear(n_feature, n_output)
        # self.hidden_layer = nn.Linear(n_feature, n_hidden)
        # self.output_layer = nn.Linear(n_hidden, n_output)

    def forward(self, feature):
        # outputs = self.hidden_layer(feature)
        # outputs = self.output_layer(outputs)
        outputs = self.output_layer(feature)
        outputs = abs(outputs)
        return outputs


net = Net(2, 10, 2)

optimizer = torch.optim.SGD(net.parameters(), lr=5e-2)
loss_func = nn.MSELoss()

for i in range(10000):
    output = net(x)
    y_output = F.softmax(output)[:, 1]
    # print(y_output)
    loss = loss_func(y_output, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i + 1) % 100 == 0:
        predict = torch.max(F.softmax(output), 1)[1]
        # print(predict)

        plt.cla()
        plt.scatter(x[:, 0], x[:, 1], c=predict)
        plt.pause(0.0001)
