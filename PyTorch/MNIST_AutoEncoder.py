import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import dataloader
from torchvision import datasets
import matplotlib.pyplot as plt

MNIST_set = datasets.mnist.MNIST(
    r'C:\Users\user\PycharmProjects\PyTorch\Datasets',
    train=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,))
    ])
)

batch_size = 64

image_gen = dataloader.DataLoader(
    MNIST_set,
    batch_size=batch_size,
    shuffle=True
)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            # nn.ReLU(),
            nn.Linear(64, 16),
            # nn.ReLU(),
            nn.Linear(16, 2),
            # nn.ReLU(),
        )

        # encode 28x28 image to 2 values (the MARROW)
        # use this 2 values to recover original 28x28 image

        self.decoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view((-1, 28 * 28))

        encoder = self.encoder(x)
        x = self.decoder(encoder)

        decoder = x.view((-1, 1, 28, 28))

        return encoder, decoder


auto_encoder = Autoencoder()
# print(G)
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(auto_encoder.parameters(), lr=5e-3)

for epoch in range(5):
    for step, (images, labels) in enumerate(image_gen):
        output = auto_encoder(images.view(-1, 1, 28, 28))
        loss = loss_func(output[1], images)

        # print(output[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step + 1) % 50 == 0:
            print("epoch:{}, step:{}, loss:{}".format(epoch, step, loss.data.numpy()))
            for i in range(2):
                # print(i)
                plt.subplot(i + 323)
                plt.imshow(output[1][i].view(28, 28).data.numpy(), cmap='binary')

            for i in range(2):
                # print(i)
                plt.subplot(i + 325)
                plt.imshow(images[i].view(28, 28).data.numpy(), cmap='binary')

            for _, (test_images, test_labels) in enumerate(image_gen):

                encoded_data, _ = auto_encoder(test_images)

                X = encoded_data.data[:, 0].numpy()
                Y = encoded_data.data[:, 1].numpy()
                plt.subplot(311)
                plt.scatter(X, Y, c=test_labels)
                # plt.text(X, Y, MNIST_set.train_labels[:200])
                # plt.show()
                break

            plt.pause(0.05)
            # break

plt.show()
