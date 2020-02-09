import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self, n_inputs, n_hiddens, n_outputs):
        super(Net, self).__init__()

        self.hidden = nn.Linear(n_inputs, n_hiddens)
        self.output = nn.Linear(n_hiddens, n_outputs)

    def forward(self, inputs):
        hidden = self.hidden(inputs)
        output = self.output(hidden)
        return output


n_data = torch.ones(50, 2)
scale = 1
bias = 0.5

x0 = torch.normal(-scale * n_data, bias)
y0 = torch.zeros(n_data.size()[0])
# print(y0)

x1 = torch.normal(scale * n_data, bias)
y1 = torch.ones(n_data.size()[0])
# print(y1)

x = torch.cat((x0, x1), dim=0).type(torch.float32)
y = torch.cat((y0, y1), dim=0).type(torch.long)
print(x)
print(y)

# plt.scatter(x.data.numpy()[:50, 0], x.data.numpy()[:50, 1])
# plt.scatter(x.data.numpy()[50:, 0], x.data.numpy()[50:, 1])
# plt.pause(3)

net = Net(2, 10, 2)
print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=5e-5)
loss_func = nn.CrossEntropyLoss()

epochs = 15000
for i in range(epochs):
    output = net(x)
    loss = loss_func(output, y)
    # print(output)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i + 1) % 100 == 0:
        plt.cla()
        predict = torch.max(F.softmax(output), 1)[1]
        # print(torch.max(F.softmax(output), 1))
        # print(F.softmax(output))
        # print(torch.max(F.softmax(output), 1)[1])
        pred_y = predict.data.numpy().squeeze()
        target_y = y.data.numpy()

        plt.scatter(x.data.numpy()[:, 0],
                    x.data.numpy()[:, 1],
                    c=pred_y)  # c = pred_y, s = 100, lw = 0, cmap = 'RdYlGn')

        accuracy = sum(target_y == pred_y) / (n_data.size()[0] * 2)
        # plt.text(0, 0, 'Acc=%.4f' % accuracy)
        plt.title('Acc=%.4f' % accuracy)
        plt.pause(1e-4)
