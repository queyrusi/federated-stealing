from models.cgan import Generator
import gan_alg.utils as utils

import copy
import torch.optim as optim
import torch
import os
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

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
        self.G = [Generator(), Generator(), Generator()]

        # As a first approximation, all thieves have similar learning strategy
        self.args = args
        self.class_num = 10

        self.folder = 'res/steal_res/FashionMNIST_mid%d_cgan_res' % (args.s_layers)

        self.target.to(device)
        for model in self.G:
            model.to(device)

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
        if not os.path.isdir(self.folder + "/federated_model"):
            os.mkdir(self.folder + "/federated_model")

    def init_noise(self): 
        noise_path = 'res/steal_res/noise'
        noise_paths = [f"res/steal_res/noise/{n}" for n in range(len(self.G))]
        if not os.path.isdir(noise_path):
            print("save noise to files")
        for noise_path in noise_paths:
            if not os.path.isdir(noise_path):
                os.mkdir(noise_path)

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


    def federated_steal(self):

        self.check_res_folders()

        self.init_noise()

        global_learning_rate = self.args.lr
        federated_G = Generator().to(device)  # Initialize the global model

        # Define an optimizer for the global model (federated_G)
        federated_G_optimizer = optim.Adam(federated_G.parameters(), lr=global_learning_rate, betas=(0.5, 0.999))

        i = 0  # Batch iterator
        for epoch in range(self.args.n_epoch):
            for batch in range(self.args.n_batch):

                # Initialize lists to store gradients from each device
                model_gradients = [torch.zeros_like(param) for param in self.G[0].parameters()]
                i += 1
                for model in self.G:
                    model.zero_grad()

                    batch_size = self.args.batch_size

                    z, y = self.get_batch(self.G.index(model), i)

                    y_one_hot = torch.zeros(batch_size, self.class_num)
                    y_one_hot.scatter_(1, y.view(batch_size, 1), 1)

                    z, y_one_hot = Variable(z.to(device)), Variable(y_one_hot.to(device))

                    with torch.no_grad():
                        # API query
                        target_output, mid2_output, mid3_output = self.target(z, y_one_hot)

                    G_output, G_mid2, G_mid3 = model(z, y_one_hot)

                    G_train_loss = F.l1_loss(G_output, target_output)
                    if self.args.s_layers >= 1:
                        G_train_loss += F.l1_loss(G_mid2, mid2_output)
                    if self.args.s_layers == 2:
                        G_train_loss += F.l1_loss(G_mid3, mid3_output)

                    # Calculate gradients
                    G_train_loss.backward()

                    # Accumulate gradients
                    for k, param in enumerate(model.parameters()):
                        model_gradients[k] += param.grad

                # Average gradients across all models
                for k in range(len(model_gradients)):
                    model_gradients[k] /= len(self.G)

                # Update the global model with the averaged gradients
                for global_param, averaged_grad in zip(federated_G.parameters(), model_gradients):
                    global_param.grad = averaged_grad

                # Update the global model's parameters
                federated_G_optimizer.step()
                # print("dir(federated_G) ", dir(federated_G))
                # print("federated_G.type() ", federated_G.type)
                # Replace models inside self.G with the updated global model
                self.G = [copy.deepcopy(federated_G) for _ in range(len(self.G))]


            fixed_p = self.folder + '/federated_model' + '/cGAN_' + str(i) + '.png'
            utils.show_cgan_result(federated_G, i, self.args, save=True, path=fixed_p)

            print('stealing [%d/%d]' % (i, self.args.n_epoch*self.args.n_batch))
            torch.save(federated_G.state_dict(), self.folder + '/federated_model' + ("/G_%d.pth" % (i)))

