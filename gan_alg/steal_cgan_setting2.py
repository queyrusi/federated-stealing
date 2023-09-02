from models.cgan import Generator
import gan_alg.utils as utils

import torch.optim as optim
import torch
import os
import numpy as np
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

        # self.G = Generator()

        # Declare thieves
        self.G = [Generator(), Generator()]

        # As a first approximation, all thieves have similar learning strategy
        self.args = args
        self.class_num = 10

        self.folder = 'res/steal_res/FashionMNIST_mid%d_cgan_res' % (args.s_layers)

        self.target.to(device)
        self.G[0].to(device)
        self.G[1].to(device)

        self.z_batches = []
        self.y_batches = []

    def check_res_folders(self):
        """Creates res/steal_res/FashionMNIST_mid2_cgan_res/[0-N] and /averaged_model
        """
        if not os.path.isdir('res'):
            os.mkdir('res')
        if not os.path.isdir('res/steal_res'):
            os.mkdir('res/steal_res')
        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        for n in range(len(self.G)): # create folder for each thief generator
            if not os.path.isdir(self.folder + f"/{n}"):
                os.mkdir(self.folder + f"/{n}")
        if not os.path.isdir(self.folder + "/averaged_model"):
            os.mkdir(self.folder + "/averaged_model")

    def init_noise(self): # TODO in the future this should take the T in argument 
        # TODO this should be called N time and not one time
        noise_path = 'res/steal_res/noise'
        noise_paths = [f"res/steal_res/noise/{n}" for n in range(len(self.G))]
        if not os.path.isdir(noise_path):
            print("save noise to files")
        if not os.path.isdir(noise_paths[0]):
            os.mkdir(noise_paths[0])
        if not os.path.isdir(noise_paths[1]):
            os.mkdir(noise_paths[1])

        for path in noise_paths:
            n_batch = int(self.args.n_epoch*self.args.n_batch) # TODO we assume same batch size for every thief
            z_batch = torch.rand((n_batch, self.args.batch_size, self.args.z_dim))
            y_batch = (torch.rand(n_batch, self.args.batch_size, 1) * self.class_num).type(torch.LongTensor)
            torch.save(z_batch, noise_path+'/z_batch')
            torch.save(y_batch, noise_path+'/y_batch')

            self.z_batches.append(torch.load(noise_path + '/z_batch'))
            self.y_batches.append(torch.load(noise_path + '/y_batch'))

    def get_batch(self, thief_num, idx):
        # print(self.z_batches[thief_num].shape)
        return self.z_batches[thief_num][idx], self.y_batches[thief_num][idx]

    def steal(self):

        self.check_res_folders()

        self.init_noise()

        G_optimizers = [optim.Adam(generator.parameters(), lr=self.args.lr, betas=(0.5, 0.999)) for generator in self.G]

        # iterate over all thieves
        for n, generator in enumerate(self.G):
            i = 0
            print(f"thief {n} starts training")
            for epoch in range(self.args.n_epoch): # TODO corect number of epoch?
                if i>=int(self.args.n_epoch*self.args.n_batch): # TODO we assume same batch size for every thief
                    break
                for batch in range(self.args.n_batch):
                    # train current thief
                    generator.zero_grad()

                    batch_size = self.args.batch_size
                    if i>=int(self.args.n_epoch*self.args.n_batch): # TODO we assume same batch size for every thief
                        break
                    z, y = self.get_batch(n, i)

                    i += 1

                    y_one_hot = torch.zeros(batch_size, self.class_num)
                    y_one_hot.scatter_(1, y.view(batch_size, 1), 1)

                    z, y_one_hot = Variable(z.to(device)), Variable(y_one_hot.to(device))

                    with torch.no_grad():
                        # API query
                        target_output, mid2_output, mid3_output = self.target(z, y_one_hot, i)

                    G_output, G_mid2, G_mid3 = generator(z, y_one_hot)

                    G_train_loss = F.l1_loss(G_output, target_output)
                    if self.args.s_layers >= 1:
                        G_train_loss += F.l1_loss(G_mid2, mid2_output)
                    if self.args.s_layers == 2:
                        G_train_loss += F.l1_loss(G_mid3, mid3_output)

                    G_train_loss.backward()
                    G_optimizers[n].step()

                fixed_p = self.folder + f"/{n}"+ '/cGAN_' + str(i) + '.png'
                # utils.show_cgan_result(generator, i, self.args, save=True, path=fixed_p)

                print('stealing [%d/%d]' % (i, self.args.n_epoch*self.args.n_batch))
                torch.save(generator.state_dict(), self.folder + f"/{n}" + ("/G_%d.pth" % (i)))
        import time
        for n, generator in enumerate(self.G):
            i = 0
            fixed_p = self.folder + f"/{n}"+ '/activation_compare_' + str(i) + '.png'
            utils.show_cgan_result(generator, 0, self.args, save=True, path=fixed_p)
            time.sleep(10)

    def model_avg(self):
        """Calculate the weighted average of models in the list self.G.

        This method takes a list of PyTorch models, self.G, assumes they have the same
        architecture, and calculates the weighted average of their weights. The averaged
        model is saved to a specified folder as 'averaged_model.pth'. 

        Returns:
            None

        Note:
            - Make sure all models in self.G have the same architecture.
        """
        # Assuming self.G is a list containing your models G1, G2, ...
        print("For G1 ", self.G[0].fc1_1.weight[:10])
        print("For G2 ", self.G[1].fc1_1.weight[:10])

        # Create a new model with the same architecture as G1 (assuming G1 and others have the same architecture)
        averaged_model = type(self.G[0])().to(device)  # Create an instance of the same class as the first model

        # Initialize a dictionary to keep track of the summed weights
        summed_weights = {}

        # Loop through the models and accumulate their weights
        num_models = len(self.G)
        for model in self.G:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if name not in summed_weights:
                        summed_weights[name] = param.data.clone()
                    else:
                        summed_weights[name] += param.data

        # Calculate the average of the accumulated weights
        for name, summed_weight in summed_weights.items():
            averaged_weight = summed_weight / num_models
            averaged_model.state_dict()[name].copy_(averaged_weight)

        # Save or use the averaged_model for further tasks
        AVG_G_DIR = self.folder + "/averaged_model" 
        AVG_G_PATH = AVG_G_DIR + "/averaged_model.pth"
        torch.save(averaged_model.state_dict(), AVG_G_PATH)

        print("model is averaged and stored at ", AVG_G_PATH)

        utils.show_cgan_result(averaged_model, "doesn't matter", self.args, save=True, path=AVG_G_DIR + '/cGAN_steal_script.png')
        # utils.calculate_fid_score(averaged_model, self.args)
        return averaged_model
