import copy
import torch
import torch.nn.functional as F
import utils
from models.cgan import Generator
import time


if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


class ClueStealing():
    def __init__(self, args):
        self.queries_dl = None
        self.args = args
        self.LOG_DIR = None
        self.fixed_noise = None
        self.fixed_label = None
        self.fixed_label_natural = None
        self.remainder_noise_dl = None      
        self.remainder_queries_dl = None
        self.thief_gradients = None
        self.thieves = None
        self.noise_dl = None
        if self.args.defense == 'FGSM':
            self.fgsm_model = utils.load_lenet(trained_on=self.args.dataset).to(device)
        self.best_fid = 4242

    def init_fixed_noise(self):
        """Latent  vectors and labels used to assess progress of the generator"""

        self.fixed_noise = torch.randn(self.args.n_samples4fid, self.args.nz, 1, 1).to(device)
        self.fixed_label_natural = torch.randint(0, 10, (self.args.n_samples4fid,)).sort().values
        self.fixed_label_natural.to(device)
        self.fixed_label = utils.label_1hots()[self.fixed_label_natural]

    def init_dataloaders(self):
        """Get dataloaders for noise and victim replies. Create remainder dataloader when num_QperA 
        not a multiple of declared batch_size.

        Having batches of queries of all thieves lined up in one dataloader saves mem space and
        compute time.
        """
        Q, R = divmod(self.args.num_QperA, self.args.batch_size)
        if Q == 0:
            # override args batch size
            self.args.batch_size = R

        self.noise_dl = utils.get_noise_dataloader(self.args.n_thieves * Q, self.args.batch_size,
                                                   noise_file="noise_tensor.pth",
                                                   labels_file="labels_tensor.pth",
                                                   nz=self.args.nz)
        self.queries_dl = utils.get_queries_dataloader(self.noise_dl, dataset=self.args.dataset,
                                                       nz=self.args.nz)

        # Create remainder dataloader if num_QperA is not a multiple of batch_size
        if R:
            self.remainder_noise_dl = utils.get_noise_dataloader(self.args.n_thieves, R,
                                                                 nz=self.args.nz)
            self.remainder_queries_dl = utils.get_queries_dataloader(self.noise_dl,
                                                                     self.args.dataset,
                                                                     self.args.nz)
            print("self.remainder_noise_dl ", len(self.remainder_noise_dl))
            print("self.noise_dl ", len(self.noise_dl))

    def log_fid(self, model, epoch):
        """Get FID of model to real statistics"""

        model.eval()

        # Thief generates images
        thief_fixed = model(self.fixed_noise, self.fixed_label).cpu()

        # Save 2048 images for FID
        utils.save_batch_images(thief_fixed, "./dumpster/thief_batch") 

        with torch.no_grad():
            start_time = time.time()
            try:
                # Distance from thief to real

                real_stats_path = f"./dumpster/real_batch_stats_{self.args.dataset}.npz"
                frechet_dist = utils.subprocess_fid(real_stats_path,
                                                    "./dumpster/thief_batch")
                print(f"Frechet Distance thief<>real: {frechet_dist:.3f}")
                if frechet_dist < self.best_fid:
                    self.best_fid = frechet_dist

            except Exception as e:
                print(f"An exception occurred: {str(e)}")
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"(compute time {execution_time:.3f} seconds)")

    def accumulate(self, thief, thief_gradients, noise_batch, reply_batch):
        """Run thief on a reply (noise, label) batch and compare the output to reply batch.
         Add gradient to list"""
        thief.zero_grad()

        batch_noise, corresponding_label = noise_batch[0], utils.label_1hots()[noise_batch[1]]

        target_output, _ = reply_batch

        if self.args.defense == 'FGSM':
            target_output = utils.fgsm_attack(self.fgsm_model, target_output, corresponding_label,
                                              epsilon=0.01)
            if self.args.counter_measure == 'JPEG':
                target_output = utils.apply_jpeg_compression(target_output).to(device)

        thief_output = thief(batch_noise, corresponding_label)
        #                                 ^-- thief wants to generate this class

        # Compare what thief generated to what the target generated
        thief_train_loss = F.l1_loss(thief_output, target_output)

        thief_train_loss.backward()
        for k, param in enumerate(thief.parameters()):
            thief_gradients[k] += param.grad.clone()
        return thief_gradients

    def backlog(self, model, epoch, queries_batch=None):
        """Intermediate values/media logging policy"""

        if epoch % 40 == 0 and epoch != 0:
            self.log_fid(model, epoch)

    def accumulate_batch(self, batch_i, epoch, noise_batch, queries_batch):
        # Current thief's noise for his batch
        batch_noise, batch_label = noise_batch

        # Victim's reponse to a (batch_noise, batch_label) pair.
        # It was pre-computed in the init_dataloaders step)
        batch_reply, batch_label_ = queries_batch

        assert torch.equal(batch_label, batch_label_), "Reply is not the answer to query"

        if batch_i % 20 == 0 or batch_i + 1 == len(self.queries_dl):
            print(f'[{epoch}/{self.args.n_epoch}][{batch_i+1}/{len(self.queries_dl)}]')
        
        # Initialize lists to store gradients from each device
        if (batch_i == 0) or (batch_i % self.args.n_thieves == 0):
            self.thief_gradients = [torch.zeros_like(p) for p in self.thieves[0].parameters()]
            # ^-- len 13

        # To the existing gradients add new gradients from current thief
        self.thief_gradients = self.accumulate(self.thieves[batch_i % self.args.n_thieves],
                                               self.thief_gradients,  # list to accumulate
                                               (batch_noise, batch_label),  # query
                                               (batch_reply, batch_label))  # answer

    def steal(self):
        """
        For each thief,
        - Get the pre-generated (noise, label) and the pre-generated (reply, _) 
        - Make thief infer with input (noise, label) and compare its output to the (reply, _)
        - Resulting gradient is accumulated on a list. 
        When we're done with all thieves, divide list by N_thieves and make it the
        gradient of the global model. Update the global model and clone it N_thieves times.                                                                           
        """
        self.init_fixed_noise()

        # Getting dataset of pre-generated thief noises and already computed queries from victim
        self.init_dataloaders()

        # Initialize the global thief
        GT = Generator(nc=self.args.nc, ngf=self.args.ngf, nz=self.args.nz).to(device)

        # Declare thieves
        self.thieves = [copy.deepcopy(GT) for i in range(self.args.n_thieves)]

        # Define an optimizer for the global model
        GT_optimizer = torch.optim.Adam(GT.parameters(), lr=self.args.lr,
                                        betas=(self.args.b1, self.args.b2)) 

        print(f"[+] start stealing with N_a={self.args.n_thieves}, N_q/a={self.args.num_QperA}")

        # ClueS uses mini-batch update so epoch num is a convenience
        for epoch in range(self.args.n_epoch):
            noise_data_iter = iter(self.noise_dl)
            queries_data_iter = iter(self.queries_dl)

            for batch_i, (noise_batch, queries_batch) in enumerate(zip(noise_data_iter,
                                                                   queries_data_iter)):

                self.accumulate_batch(batch_i, epoch, noise_batch, queries_batch)

                # When all thieves have made their contribution, optimize:
                if (batch_i % self.args.n_thieves == 0):
                    # Zero out gradients for the global model (GT)
                    GT_optimizer.zero_grad()

                    # Average gradients across all thieves
                    for k in range(len(self.thief_gradients)):
                        self.thief_gradients[k] /= len(self.thieves)

                    # Update the global model with the averaged gradients
                    for global_param, averaged_grad in zip(GT.parameters(), self.thief_gradients):
                        global_param.grad = averaged_grad.clone()

                    # Update the global model's parameters
                    GT_optimizer.step()

                    # Replace models inside self.G with the updated global model
                    self.thieves = [copy.deepcopy(GT) for _ in range(len(self.thieves))]

                    self.backlog(GT, epoch, queries_batch)

            # If there are remainder queries (num_QperA not a multiple of batch_size), infer them
            # too
            if self.remainder_noise_dl:
                for batch_i, (noise_batch, queries_batch) in enumerate(zip(noise_data_iter,
                                                                       queries_data_iter)):
                    self.accumulate_batch(batch_i, epoch, noise_batch, queries_batch)

                    # Last optimization with remainder:

                    # Zero out gradients for the global model (GT)
                    GT_optimizer.zero_grad()

                    # Average gradients across all thieves
                    for k in range(len(self.thief_gradients)):
                        self.thief_gradients[k] /= len(self.thieves)

                    # Update the global model with the averaged gradients
                    for global_param, averaged_grad in zip(GT.parameters(), self.thief_gradients):
                        global_param.grad = averaged_grad.clone()

                    # Update the global model's parameters
                    GT_optimizer.step()

                    # Replace models inside self.G with the updated global model
                    self.thieves = [copy.deepcopy(GT) for _ in range(len(self.thieves))]

                    self.backlog(GT, epoch, queries_batch)
