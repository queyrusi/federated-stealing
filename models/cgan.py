import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils


class Generator(nn.Module):
    def __init__(
        self,
        nc,
        ngf,
        nz,
        num_classes: int = 10,
    ):
        super(Generator, self).__init__()
        self.nc = nc
        self.ngf = ngf
        self.nz = nz

        self.image = nn.Sequential(
            nn.ConvTranspose2d(self.nz, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),
        )
        self.label = nn.Sequential(
            nn.ConvTranspose2d(num_classes, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(inplace=True),
        )
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.ngf * 16, self.ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.ngf * 4, self.nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, image, label):
        image = self.image(image)
        label = self.label(label)
        incat = torch.cat((image, label), dim=1)
        return self.main(incat)


class GeneratorModerated(nn.Module):
    def __init__(self, nc, ngf, nz, num_classes: int = 10):
        super(GeneratorModerated, self).__init__()
        self.nc = nc
        self.ngf = ngf
        self.nz = nz

        self.image = nn.Sequential(
            nn.ConvTranspose2d(self.nz, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),
        )

        self.label = nn.Sequential(
            nn.ConvTranspose2d(num_classes, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(inplace=True),
        )

        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.ngf * 16, self.ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.ngf * 4, self.nc, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nc),  # Batch Normalization for the final layer
            nn.Tanh(),
        )

    def forward(self, image, label):
        image = self.image(image)
        label = self.label(label)
        incat = torch.cat((image, label), dim=1)
        return self.main(incat)


class Discriminator(nn.Module):
    def __init__(self, nc=1, nz=100, ndf=8, num_classes: int = 10):
        super(Discriminator, self).__init__()

        self.image = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
        )
        self.label = nn.Sequential(
            nn.Conv2d(num_classes, ndf, 4, 2, 1, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
        )
        self.main = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, image, label):
        image = self.image(image)
        label = self.label(label)
        incat = torch.cat((image, label), dim=1)
        return self.main(incat)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


# Conditional batch norm definition
class CategoricalConditionalBatchNorm(torch.nn.Module):
    # as in the chainer SN-GAN implementation, we keep per-cat weight and bias
    def __init__(
        self,
        num_features,
        num_cats,
        eps=2e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_cats = num_cats
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = torch.nn.Parameter(torch.Tensor(num_cats, num_features))
            self.bias = torch.nn.Parameter(torch.Tensor(num_cats, num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_var", None)
            self.register_parameter("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.fill_(1.0)
            self.bias.data.zero_()

    def forward(self, input, cats):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        out = torch.nn.functional.batch_norm(
            input,
            self.running_mean,
            self.running_var,
            None,
            None,
            self.training or not self.track_running_stats,
            exponential_average_factor,
            self.eps,
        )
        if self.affine:
            shape = [input.size(0), self.num_features] + (input.dim() - 2) * [1]
            weight = self.weight.index_select(0, cats).view(shape)
            bias = self.bias.index_select(0, cats).view(shape)
            out = out * weight + bias
        return out

    def extra_repr(self):
        return (
            "{num_features}, num_cats={num_cats}, eps={eps}, momentum={momentum}, affine={affine}, "
            "track_running_stats={track_running_stats}".format(**self.__dict__)
        )


# G(z)
class CelebAGenerator(nn.Module):
    # initializers
    def __init__(self, num_classes, d=128, n_labels=2):
        super(CelebAGenerator, self).__init__()
        self.deconv1_1 = nn.ConvTranspose2d(100, d * 4, 4, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(d * 4)

        self.deconv00_2 = nn.Conv2d(num_classes, int(d / 4), 1, 1, 0)
        self.deconv00_2_bn = nn.BatchNorm2d(int(d / 4))
        self.deconv0_2 = nn.ConvTranspose2d(int(d / 4), d, 4, 1, 0)
        self.deconv0_2_bn = nn.BatchNorm2d(d)
        self.deconv1_2 = nn.ConvTranspose2d(d, d * 4, 3, 1, 1)
        self.deconv1_2_bn = nn.BatchNorm2d(d * 4)

        self.deconv2 = nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1)
        self.deconv2_bn = CategoricalConditionalBatchNorm(d * 4, num_classes)
        self.deconv3 = nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1)
        self.deconv3_bn = CategoricalConditionalBatchNorm(d * 2, num_classes)
        self.deconv4 = nn.ConvTranspose2d(d * 2, d, 4, 2, 1)
        self.deconv4_bn = CategoricalConditionalBatchNorm(d, num_classes)
        self.deconv5 = nn.ConvTranspose2d(d, 3, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    # def forward(self, input):
    def forward(self, input, label, cat):
        x = F.leaky_relu(self.deconv1_1_bn(self.deconv1_1(input)), 0.2)

        y = F.leaky_relu(self.deconv00_2_bn(self.deconv00_2(label)), 0.2)
        y = F.leaky_relu(self.deconv0_2_bn(self.deconv0_2(y)), 0.2)
        y = F.leaky_relu(self.deconv1_2_bn(self.deconv1_2(y)), 0.2)

        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.deconv2_bn(self.deconv2(x), cat), 0.2)
        x = F.leaky_relu(self.deconv3_bn(self.deconv3(x), cat), 0.2)
        x = F.leaky_relu(self.deconv4_bn(self.deconv4(x), cat), 0.2)
        x = torch.tanh(self.deconv5(x))

        return x


class CelebADiscriminator(nn.Module):
    # initializers
    def __init__(self, d=128, n_labels=2):
        super(CelebADiscriminator, self).__init__()

        # original:
        self.conv1_1 = nn.Conv2d(3, int(d / 2), 4, 2, 1)
        self.conv0_2 = nn.Conv2d(2 * n_labels, int(d / 4), 1, 1, 0)
        self.conv1_2 = nn.Conv2d(int(d / 4), int(d / 2), 4, 2, 1)

        # after union
        self.conv2 = utils.spectral_norm(nn.Conv2d(d, d * 2, 4, 2, 1))
        self.conv3 = utils.spectral_norm(nn.Conv2d(d * 2, d * 4, 4, 2, 1))
        self.conv4 = utils.spectral_norm(nn.Conv2d(d * 4, d * 8, 4, 2, 1))
        self.conv5 = nn.Conv2d(d * 8, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    # def forward(self, input):
    def forward(self, input, label):
        # original:
        x = F.leaky_relu(self.conv1_1(input), 0.2)

        y = F.leaky_relu(self.conv0_2(label), 0.2)
        y = F.leaky_relu(self.conv1_2(y), 0.2)

        x = torch.cat([x, y], 1)  # Discriminator

        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = torch.sigmoid(self.conv5(x))

        return x
