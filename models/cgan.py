import torch.nn as nn
import torch.nn.functional as F
import torch


def normal_init(m, mean, std):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.fc1_1 = nn.Linear(128, 256)
        self.fc1_1_bn = nn.BatchNorm1d(256)
        self.fc1_2 = nn.Linear(10, 256)
        self.fc1_2_bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(512, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc3_bn = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, 784)

    def weight_init(self, mean=0, std=0.02):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, inputs, label):
        x = F.relu(self.fc1_1_bn(self.fc1_1(inputs)))
        y = F.relu(self.fc1_2_bn(self.fc1_2(label)))
        x = torch.cat([x, y], 1)
        x2 = F.relu(self.fc2_bn(self.fc2(x)))
        x1 = F.relu(self.fc3_bn(self.fc3(x2)))
        x = torch.tanh(self.fc4(x1))

        return x, x1, x2


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1_1 = nn.Linear(784, 1024)
        self.fc1_2 = nn.Linear(10, 1024)
        self.fc2 = nn.Linear(2048, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.fc3_bn = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 1)

    def weight_init(self, mean=0, std=0.02):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, inputs, label):
        x = F.leaky_relu(self.fc1_1(inputs), 0.2)
        y = F.leaky_relu(self.fc1_2(label), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.fc2_bn(self.fc2(x)), 0.2)
        x = F.leaky_relu(self.fc3_bn(self.fc3(x)), 0.2)
        x = torch.sigmoid(self.fc4(x))

        return x
