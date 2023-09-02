from models.cgan import Generator, Discriminator
from gan_alg.show import show_cgan_result

from torchvision import datasets, transforms
import torch.optim as optim
import torch
import os
from torch.autograd import Variable

# M2 sillicon device
device = "mps" if torch.backends.mps.is_available() else "cpu"


class TrainConditionalGAN(object):

    def __init__(self, args):
        super(TrainConditionalGAN, self).__init__()
        self.G = Generator()
        self.D = Discriminator()

        self.args = args
        self.class_num = None
        self.train_loader = self.load_dataset()

        self.folder = 'res/train_res/FashionMNIST_CGAN_res'

        self.G.to(device)
        self.D.to(device)

    def load_dataset(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(root='data', train=True, download=True, transform=transform),
            batch_size=self.args.batch_size, shuffle=True)
        self.class_num = 10
        return train_loader

    def check_res_folders(self):
        if not os.path.isdir('res'):
            os.mkdir('res')
        if not os.path.isdir('res/train_res'):
            os.mkdir('res/train_res')
        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)

    def train_epoch(self, G_optimizer, D_optimizer, loss):
        D_loss_list = []
        G_loss_list = []

        for x, y in self.train_loader:

            print("x shape ", x.shape)
            print("y shape ", y.shape)
            # training D
            self.D.zero_grad()

            batch_size = x.size()[0]

            y_real = torch.ones(batch_size)
            y_fake = torch.zeros(batch_size)
            y_one_hot = torch.zeros(batch_size, self.class_num)
            y_one_hot.scatter_(1, y.view(batch_size, 1), 1)

            x = x.view(-1, 28 * 28)
            x, y_one_hot, y_real, y_fake = Variable(x.to(device)), Variable(y_one_hot.to(device)), Variable(
                y_real.to(device)), Variable(y_fake.to(device))
            D_output = self.D(x, y_one_hot).squeeze()
            D_real_loss = loss(D_output, y_real)

            z = torch.rand((batch_size, self.args.z_dim))
            y = (torch.rand(batch_size, 1) * self.class_num).type(torch.LongTensor)
            y_one_hot = torch.zeros(batch_size, self.class_num)
            y_one_hot.scatter_(1, y.view(batch_size, 1), 1)

            z, y_one_hot = Variable(z.to(device)), Variable(y_one_hot.to(device))

            G_output, G_mid2, Gmid3 = self.G(z, y_one_hot)

            D_output = self.D(G_output, y_one_hot).squeeze()
            D_fake_loss = loss(D_output, y_fake)

            D_train_loss = D_real_loss + D_fake_loss

            D_train_loss.backward()
            D_optimizer.step()

            # train G
            self.G.zero_grad()

            z = torch.rand((batch_size, self.args.z_dim)) # latent vector generation
            y = (torch.rand(batch_size, 1) * self.class_num).type(torch.LongTensor) # random class labels for conditional information
            y_one_hot = torch.zeros(batch_size, self.class_num)
            y_one_hot.scatter_(1, y.view(batch_size, 1), 1) # Convert random class labels to one-hot encoding

            z, y_one_hot = Variable(z.to(device)), Variable(y_one_hot.to(device))

            G_output, _mid2, _mid3 = self.G(z, y_one_hot)

            D_output = self.D(G_output, y_one_hot).squeeze()
            G_train_loss = loss(D_output, y_real)
            G_train_loss.backward()
            G_optimizer.step()

            G_loss_list.append(G_train_loss.cpu().detach().numpy())
            D_loss_list.append(D_train_loss.cpu().detach().numpy())

        return sum(D_loss_list) / len(D_loss_list), sum(G_loss_list) / len(G_loss_list)

    def train(self):
        self.check_res_folders()

        G_optimizer = optim.Adam(self.G.parameters(), lr=self.args.lr, betas=(0.5, 0.999))
        D_optimizer = optim.Adam(self.D.parameters(), lr=self.args.lr, betas=(0.5, 0.999))
        
        print("training start")

        # loss = torch.nn.BCELoss()
        loss = torch.nn.BCEWithLogitsLoss()

        for epoch in range(self.args.train_epoch):
            D_loss, G_loss = self.train_epoch(G_optimizer, D_optimizer, loss)

            fixed_p = self.folder+'/cGAN_' + str(epoch + 1) + '.png'
            show_cgan_result(self.G, (epoch + 1), self.args, save=True, path=fixed_p)

            print('[%d/%d] - loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), self.args.train_epoch, D_loss, G_loss))

        torch.save(self.G.state_dict(), self.folder + "/G.pth")
        torch.save(self.D.state_dict(), self.folder + "/D.pth")
