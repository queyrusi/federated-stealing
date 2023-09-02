import torch
import os
import pickle
import numpy as np
import itertools
from torch import nn
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from scipy.linalg import sqrtm
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from torchvision.models import inception_v3
import matplotlib.pyplot as plt
from models.cgan import Generator
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

import torch
from torchvision import models, transforms
from PIL import Image
import urllib.request

# M2 sillicon device
device = "mps" if torch.backends.mps.is_available() else "cpu"

# temp_z_ = torch.rand(10, 100)
temp_z_ = torch.rand(10, 128)
fixed_z_ = temp_z_
fixed_y_ = torch.zeros(10, 1)
for i in range(9):
    fixed_z_ = torch.cat([fixed_z_, temp_z_], 0)
    temp = torch.ones(10, 1) + i
    fixed_y_ = torch.cat([fixed_y_, temp], 0)

fixed_z_ = Variable(fixed_z_.to(device))
fixed_y_label_ = torch.zeros(100, 10)
fixed_y_label_.scatter_(1, fixed_y_.type(torch.LongTensor), 1)
fixed_y_label_ = Variable(fixed_y_label_.to(device))


def show_cgan_result(G, num_epoch, args, show = False, save = False, path = 'result.png'):

    G.eval()

    test_images, mid2, mid3 = G(fixed_z_, fixed_y_label_, num_epoch)

    size_figure_grid = 10
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(10 * 10):
        i = k // 10
        j = k % 10
        ax[i, j].cla()
        ax[i, j].imshow(test_images[k].cpu().data.view(28, 28).numpy(), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def extract_and_save_features(dataloader, inception_model, device, features_folder):
    """
    Extracts features from a dataset using an Inception-v3 model and saves them to a file.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader for the FashionMNIST.
        inception_model (torch.nn.Module): Pretrained Inception-v3 model.
        device (torch.device): Device (CPU or GPU) to perform computations on.
        features_folder (str): Path to the folder where extracted features should be saved.

    Returns:
        torch.Tensor: Extracted features as a tensor.

    Raises:
        FileNotFoundError: If the specified features folder does not exist.

    Note:
        - Will create the 'data/FashionMNIST/fid_features' folder if not existing.
        - Takes about 7'.

    """
    inception_model.eval()
    # Define the fraction of the data you want to use (e.g., 1/100)
    fraction = 0.08  # Use 1/100 of the data

    # Compute the total number of batches to process
    total_batches = len(dataloader)
    num_batches_to_process = int(total_batches * fraction)
    real_features = []
    for i, batch in enumerate(tqdm(dataloader, desc="processing real batches")):
        if i >= num_batches_to_process:
            break  # Stop processing after the specified number of batches
        batch = batch[0].to(device)
        with torch.no_grad():
            features = inception_model(batch)
            features = features.view(features.size(0), -1)
        real_features.append(features)
    real_features = torch.cat(real_features, dim=0)
    
    # Check if the folder for features exists
    if not os.path.exists(features_folder):
        os.makedirs(features_folder)

    # Save the extracted features
    with open(os.path.join(features_folder, 'fid_features.pkl'), 'wb') as f:
        pickle.dump(real_features.cpu().numpy(), f)
    return real_features

def calculate_fid_score(G, args):
    folder = 'res/steal_res/FashionMNIST_mid%d_cgan_res' % (args.s_layers)
    show_cgan_result(G, "doesn't matter", save=True, path="dumpster/fided_model.png")

    FEATURES_DIR = 'data/FashionMNIST/fid_features'
    FEATURES_PATH = os.path.join(FEATURES_DIR, 'fid_features.pkl')

    # Transformation for Inception-v3
    inception_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert to RGB by duplicating the grayscale channel
        transforms.Resize((299, 299), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Initialize the Inception-v3 model. It will extract features on the real images + fake images to compare
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()

    # Check if features file exists
    if os.path.exists(FEATURES_PATH):
        # Load and use the precomputed features
        with open(FEATURES_PATH, 'rb') as f:
            real_features = torch.tensor(pickle.load(f))
    else:
        # Load the FashionMNIST dataset as a reference. Transform input so it matches inception model
        real_dataset_for_inception = FashionMNIST(root='data', train=True, download=True, transform=inception_transform)
        real_dataloader_for_inception = DataLoader(real_dataset_for_inception, batch_size=args.batch_size, shuffle=True)

        # Extract and save features
        real_features = extract_and_save_features(real_dataloader_for_inception, inception_model, device, FEATURES_DIR)


    mu_real = torch.mean(real_features, axis=0)
    print("mu_real.shape ", mu_real.shape)
    sigma_real = torch.cov(real_features.t()) # not too sure about the transpose
    print("sigma_real.shape ", sigma_real.shape)

    G_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    G.eval()

    # Prepare dataset for generator
    real_dataset = FashionMNIST(root='data', train=True, download=True, transform=G_transform) # Load the FashionMNIST dataset as a reference
    real_dataloader = DataLoader(real_dataset, batch_size=args.batch_size, shuffle=True) # Create a DataLoader for the real dataset

    # Transformation for Inception-v3
    preprocess = transforms.Compose([
         transforms.Resize((299, 299)),
         transforms.Grayscale(num_output_channels=3),  # Convert to RGB by duplicating grayscale channel
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
     ])

    # Generate fake samples using the GAN generator
    fake_features = []
    with torch.no_grad():
        num_samples = len(real_dataloader)
        for _ in tqdm(range(num_samples // args.batch_size), desc="processing fake batches"):
            z = torch.rand((args.batch_size, 128))  # TODO was self.args.z_dim
            y = (torch.rand(args.batch_size, 1) * 10).type(torch.LongTensor) # TODO was self.class_num
            y_one_hot = torch.zeros(args.batch_size, 10) # TODO was self.class_num
            y_one_hot.scatter_(1, y.view(args.batch_size, 1), 1)
            z, y_one_hot = Variable(z.to(device)), Variable(y_one_hot.to(device))
            fake_images, _, _ = G(z, y_one_hot) # Generate images that will be evaluated by FID
            # ^-- is of shape (64, 28 * 28)
            fake_images = fake_images.view(args.batch_size, 28, 28) 
            # ---------------------------------------------
            # Convert the tensor to a NumPy array
            tensor_as_numpy = fake_images[0].cpu().numpy()

            # Scale the values in the NumPy array to the range [0, 255]
            tensor_as_numpy = (tensor_as_numpy * 255).astype(np.uint8)

            # Create a PIL image from the NumPy array
            image = Image.fromarray(tensor_as_numpy)    

            # Save the image to the specified directory
            save_path = 'dumpster/random_image.jpg'
            image.save(save_path)
            # ---------------------------------------------
            input_batch = torch.empty(args.batch_size, 3, 299, 299).to(device) # Placeholder for the rescaled fake images
            # Convert and preprocess each image in the batch
            for i in range(args.batch_size):
                image_pil = transforms.ToPILImage()(fake_images[i])
                preprocessed_image = preprocess(image_pil)
                input_batch[i] = torch.tensor(preprocessed_image) # Fill placeholder with Inception-compatible pics
            with torch.no_grad():
                features = inception_model(input_batch)
                features = features.view(features.size(0), -1)
            fake_features.append(features)
    fake_features = torch.cat(fake_features, dim=0).cpu()

    mu_fake = torch.mean(fake_features, axis=0)
    sigma_fake = torch.cov(fake_features.t())

    # Calculate the FID score
    diff = mu_real - mu_fake
    covmean = sqrtm(torch.matmul(sigma_real,sigma_fake))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid_score = np.sum(diff.numpy()**2) + np.trace(sigma_real.numpy() + sigma_fake.numpy() - 2.0 * covmean)

    return fid_score
