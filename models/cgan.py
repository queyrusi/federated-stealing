# import torchvision module to handle image manipulation
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.optim as optim
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torch.autograd import Variable
from scipy import linalg
import utils


class Generator(nn.Module):
    def __init__(self,nc, ngf, nz, num_classes: int = 10, ):
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

            nn.ConvTranspose2d(num_classes, self.ngf * 8, 4, 1,0, bias=False),
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
           

            nn.ConvTranspose2d(self.ngf*4, self.nc, 4, 2, 1, bias=False),
            nn.Tanh()
           
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
            nn.Tanh()
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

            nn.Conv2d(nc, ndf, 4, 2, 1,1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3)

        )
        self.label = nn.Sequential(

            nn.Conv2d(num_classes, ndf, 4, 2, 1,1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3)

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
            nn.Sigmoid()
        )

    def forward(self, image, label):
        image = self.image(image)
        label = self.label(label)
        incat = torch.cat((image, label), dim=1)
        return self.main(incat)