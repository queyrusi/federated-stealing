from models.cgan import Generator
from gan_alg.show import show_cgan_result

import torch.optim as optim
import torch
import os
from torch.autograd import Variable
import torch.nn.functional as F

device = "mps" if torch.backends.mps.is_available() else "cpu"


class StealConditionalGAN(object):

    def __init__(self, args):
        super(StealConditionalGAN, self).__init__()
        self.target = Generator()
        map_location = lambda storage, loc: storage.to(device)

        state_dict = torch.load('res/train_res/FashionMNIST_CGAN_res/G.pth', map_location=map_location)
        self.target.load_state_dict(state_dict)

        self.G = Generator()

        self.args = args
        self.class_num = 10

        self.folder = 'res/steal_res/FashionMNIST_mid%d_cgan_res' % (args.s_layers)

        self.target.to(device)
        self.G.to(device)

        self.z_batches = None
        self.y_batches = None

    def check_res_folders(self):
        if not os.path.isdir('res'):
            os.mkdir('res')
        if not os.path.isdir('res/steal_res'):
            os.mkdir('res/steal_res')
        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)

    def init_noise(self):
        noise_path = 'res/steal_res/noise'
        if not os.path.isdir(noise_path):
            print("save noise to files")

            os.mkdir(noise_path)

            n_batch = 1000
            z_batches = torch.rand((n_batch, self.args.batch_size, self.args.z_dim))
            y_batches = (torch.rand(n_batch, self.args.batch_size, 1) * self.class_num).type(torch.LongTensor)
            torch.save(z_batches, noise_path+'/z_batches')
            torch.save(y_batches, noise_path+'/y_batches')

        self.z_batches = torch.load(noise_path+'/z_batches')
        self.y_batches = torch.load(noise_path + '/y_batches')

    def get_batch(self, idx):
        return self.z_batches[idx], self.y_batches[idx]

    def steal(self):

        self.check_res_folders()

        self.init_noise()

        G_optimizer = optim.Adam(self.G.parameters(), lr=self.args.lr, betas=(0.5, 0.999))

        i = 0
        for epoch in range(self.args.n_epoch):
            for batch in range(self.args.n_batch):

                # train G
                self.G.zero_grad()

                batch_size = self.args.batch_size

                z, y = self.get_batch(i)

                i += 1

                y_one_hot = torch.zeros(batch_size, self.class_num)
                y_one_hot.scatter_(1, y.view(batch_size, 1), 1)

                z, y_one_hot = Variable(z.to(device)), Variable(y_one_hot.to(device))

                with torch.no_grad():
                    # API query
                    target_output, mid2_output, mid3_output = self.target(z, y_one_hot)

                G_output, G_mid2, G_mid3 = self.G(z, y_one_hot)

                G_train_loss = F.l1_loss(G_output, target_output)
                if self.args.s_layers >= 1:
                    G_train_loss += F.l1_loss(G_mid2, mid2_output)
                if self.args.s_layers == 2:
                    G_train_loss += F.l1_loss(G_mid3, mid3_output)

                G_train_loss.backward()
                G_optimizer.step()

            fixed_p = self.folder + '/cGAN_' + str(i) + '.png'
            show_cgan_result(self.G, i, self.args, save=True, path=fixed_p)

            print('stealing [%d/%d]' % (i, self.args.n_epoch*self.args.n_batch))
            torch.save(self.G.state_dict(), self.folder + ("/G_%d.pth" % (i)))

