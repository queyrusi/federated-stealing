import os
import numpy as np
import PIL.Image as Image

import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.utils import save_image
from models.cgan import CelebAGenerator as CelebaGenerator
import random
import subprocess
import re

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def fgsm_attack(model, images, labels, epsilon=0.03):
    # Clone the input images to ensure the original tensor is not modified
    images = images.clone().detach().requires_grad_(True)
    labels = labels.clone().detach().requires_grad_(True)

    # Forward pass:

    # We need the outputs to be the same dimension as the one hot labels...
    outputs = model(images)  # torch.Size([128, 10])
    # >>>outputs[0]
    # tensor([-10.1601,  -9.0096, -10.5265,  -9.6249, -11.2461,   2.4602,  -9.1701,
    #          10.9368,  -3.5861,  -1.1070], device='mps:0', rad_fn=<SelectBackward0>)

    # Squeeze the dimensions of the input one-hot labels
    squeezed_labels = labels.squeeze(dim=2).squeeze(dim=2)  # torch.Size([128, 10])

    # >>>squeezed_labels[0]
    # tensor([0., 0., 0., 1., 0., 0., 0., 0., 0., 0.], device='mps:0')

    loss = torch.nn.functional.mse_loss(outputs, squeezed_labels)

    model.zero_grad()
    loss.backward()

    # Collect the element-wise sign of the data gradient
    perturbation = epsilon * torch.sign(images.grad)

    # Create the adversarial example
    perturbed_images = images + perturbation

    # Clip the perturbed image to ensure it remains in the valid pixel range
    perturbed_images = torch.clamp(perturbed_images, 0, 1)

    return perturbed_images


def apply_jpeg_compression(image_tensor, quality=75):
    compressed_images = []
    for image in image_tensor:
        image_array = image.cpu().detach().numpy()

        # Convert to 8-bit unsigned integer
        image_array = (image_array * 255).astype(np.uint8)
        image_pil = Image.fromarray(image_array[0])

        # Save and reload the image with JPEG compression
        # image_pil.save("compressed_image.jpg", format="JPEG", quality=quality)
        compressed_image_pil = Image.open("compressed_image.jpg")

        # Convert the compressed image back to a tensor
        compressed_image_tensor = TF.to_tensor(compressed_image_pil)
        # ^-- [1, 32, 32]
        compressed_images.append(compressed_image_tensor.unsqueeze(1))
        #                                               ^-- add a dimension to get [1, 1, 32, 32]
    rearranged = torch.cat(compressed_images, dim=0)
    # ^-- [20, 1, 32, 32], just what the doctor ordered
    return rearranged


class QueriesDataset(Dataset):
    def __init__(self, data_file, labels_file, num_queries):
        self.data = torch.load(data_file, map_location=torch.device(device))[
            :num_queries
        ]
        self.labels = torch.load(labels_file, map_location=torch.device(device))[
            :num_queries
        ]

        print("[+] Creating a dataset of length ", num_queries)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def delete_helper_tensor_files(filenames):
    for filename in filenames:
        if os.path.exists(filename):
            os.remove(filename)
            print(f"[-] Deleted {filename}")
        else:
            print(f"{filename} does not exist")


def load_lenet(trained_on="FashionMNIST"):
    lenet = nn.Sequential()
    lenet.add_module(
        "conv1",
        nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),
    )
    lenet.add_module("relu1", nn.ReLU())
    lenet.add_module("avg_pool1", nn.AvgPool2d(kernel_size=2, stride=2))
    lenet.add_module(
        "conv2", nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
    )
    lenet.add_module("avg_pool2", nn.AvgPool2d(kernel_size=2, stride=2))
    lenet.add_module(
        "conv3", nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1)
    )
    lenet.add_module("relu2", nn.ReLU())
    lenet.add_module("flatten", nn.Flatten(start_dim=1))
    lenet.add_module("fc1", nn.Linear(in_features=480, out_features=84))
    lenet.add_module("relu3", nn.ReLU())
    lenet.add_module("fc2", nn.Linear(in_features=84, out_features=10))

    model_path = (
        f"./models/{device}/Lenet_{trained_on}{'_cuda' if device=='cuda' else ''}.pth"
    )

    lenet.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    return lenet


def save_batch_images(batch_data, folder_path):

    # Check if the folder exists, and if not, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Save the image from fake_fixed
    for i, image in enumerate(batch_data):
        image_path = os.path.join(folder_path, f"image_{i}.png")
        save_image(image, image_path)
    return


def subprocess_fid(path_to_dataset1, path_to_dataset2):

    # Run the command and capture the output
    command = f"python -m pytorch_fid {path_to_dataset1} {path_to_dataset2}\
     {'--device cuda:0' if device=='cuda' else ''}"
    output = subprocess.check_output(command, shell=True, text=True)

    # String containing FID value
    fid_string = output.strip()

    # re to extract FID value
    pattern = r"FID:\s+([\d.]+)"
    match = re.search(pattern, fid_string)

    if match:
        frechet_dist = float(match.group(1))
    else:
        print("FID value not found in the string.")

    return frechet_dist


def label_1hots():
    label_1hots = torch.zeros(10, 10)
    for i in range(10):
        label_1hots[i, i] = 1
    label_1hots = label_1hots.view(10, 10, 1, 1).to(device)
    return label_1hots


def label_fills():
    img_size = 32
    # Label one-hot for D
    label_fills = torch.zeros(10, 10, img_size, img_size)
    ones = torch.ones(img_size, img_size)
    for i in range(10):
        label_fills[i][i] = ones
    label_fills = label_fills.to(device)
    return label_fills


# Get the labels as a tensor for the generator from the random labels in a vector
def getGeneratorLabels(lbl, batch_size):
    if lbl.shape[1] == 1:

        a = torch.zeros([batch_size, 2, 1, 1])

        for i in range(batch_size):
            if lbl[i][0] == 1:
                a[i, 1, 0, 0] = 1
            else:
                a[i, 0, 0, 0] = 1

        return a

    elif lbl.shape[1] == 2:
        a = torch.zeros([batch_size, 4, 1, 1])

        for i in range(batch_size):
            if lbl[i][0] == 1:
                a[i, 1, 0, 0] = 1
            else:
                a[i, 0, 0, 0] = 1
            if lbl[i][1] == 1:
                a[i, 3, 0, 0] = 1
            else:
                a[i, 2, 0, 0] = 1

        return a
    else:
        a = torch.zeros([batch_size, 6, 1, 1])

        for i in range(batch_size):
            if lbl[i][0] == 1:
                a[i, 1, 0, 0] = 1
            else:
                a[i, 0, 0, 0] = 1
            if lbl[i][1] == 1:
                a[i, 3, 0, 0] = 1
            else:
                a[i, 2, 0, 0] = 1
            if lbl[i][2] == 1:
                a[i, 5, 0, 0] = 1
            else:
                a[i, 4, 0, 0] = 1

        return a


def getGeneratorCategories(lbl, batch_size):

    if lbl.shape[1] == 1:

        a = torch.zeros([batch_size])

        for i in range(batch_size):
            if lbl[i][0] == 1:
                a[i] = 1
            else:
                a[i] = 0

        b = a.long()
        return b

    elif lbl.shape[1] == 2:
        a = torch.zeros([batch_size])

        for i in range(batch_size):
            if lbl[i][0] == 0 and lbl[i][1] == 0:
                a[i] = 0
            elif lbl[i][0] == 0 and lbl[i][1] == 1:
                a[i] = 1
            elif lbl[i][0] == 0 and lbl[i][1] == 1:
                a[i] = 2
            else:
                a[i] = 3

        b = a.long()
        return b
    else:
        a = torch.zeros([batch_size])

        for i in range(batch_size):
            if lbl[i][0] == 0 and lbl[i][1] == 0 and lbl[i][2] == 0:
                a[i] = 0
            elif lbl[i][0] == 0 and lbl[i][1] == 0 and lbl[i][2] == 1:
                a[i] = 1
            elif lbl[i][0] == 0 and lbl[i][1] == 1 and lbl[i][2] == 0:
                a[i] = 2
            elif lbl[i][0] == 0 and lbl[i][1] == 1 and lbl[i][2] == 1:
                a[i] = 3
            elif lbl[i][0] == 1 and lbl[i][1] == 0 and lbl[i][2] == 0:
                a[i] = 4
            elif lbl[i][0] == 1 and lbl[i][1] == 0 and lbl[i][2] == 1:
                a[i] = 5
            elif lbl[i][0] == 1 and lbl[i][1] == 1 and lbl[i][2] == 0:
                a[i] = 6
            else:
                a[i] = 7

        b = a.long()
        return b


def get_real_images_dataloader_for_FID(batch_size):
    """Create a dataloader of real images (original dataset) to compute the FIDs of the victim
    (this FID shouldn't change since vicitm is not trained anymmore) and the target (FID lowers)
    """

    # create a set of transforms for the dataset
    dset_transforms = []
    dset_transforms.append(transforms.Resize(32))
    dset_transforms.append(transforms.ToTensor())
    dset_transforms.append(transforms.Normalize([0.5], [0.5]))
    dset_transforms = transforms.Compose(dset_transforms)

    # Use standard FashionMNIST dataset
    train_set = torchvision.datasets.FashionMNIST(
        root=".", train=True, download=False, transform=dset_transforms
    )
    dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    return dataloader


# Generate noise
def get_noise_dataloader(
    num_batches,
    batch_size,
    noise_file="noise_tensor.pth",
    labels_file="labels_tensor.pth",
    n_labels=10,
    nz=100,
    dataset="FashionMNIST",
    delete_after=True,
):
    """Dataloader of the noise/label sent to query the victim."""

    if not os.path.isfile(noise_file) or not os.path.isfile(labels_file):
        generate_and_save_noise(
            num_batches, batch_size, noise_file, labels_file, n_labels, nz, dataset
        )

    custom_dataset = QueriesDataset(
        noise_file, labels_file, num_queries=int(num_batches * batch_size)
    )
    dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False)
    if delete_after:
        delete_helper_tensor_files([noise_file])

    return dataloader


def generate_and_save_noise(
    num_batches,
    batch_size,
    noise_filename,
    labels_filename,
    n_labels=10,
    nz=100,
    dataset="FashionMNIST",
):
    """Generate noise images that will be 'queried' by the thieves to the victim"""

    print(f"[+] Generating {num_batches} batches of {batch_size} noises")
    for batch_idx in range(int(num_batches)):
        noise = torch.randn(batch_size, nz, 1, 1).to(device)

        if dataset == "CelebA":
            label_natural = np.random.randint(2, size=(batch_size, 2))

            # Convert the list to a PyTorch tensor
            random_labels = torch.tensor(label_natural)
        else:
            desired_length = batch_size
            shuffled_labels = list(range(n_labels))
            random.shuffle(shuffled_labels)
            label_natural = [
                shuffled_labels[i % n_labels] for i in range(desired_length)
            ]
            random_labels = torch.tensor(label_natural)
            # ^-- tensor([3, 5, 9, 8, 2, 4, ...

        if batch_idx == 0:
            # Save the first batch directly to the file
            torch.save(noise, noise_filename)
            torch.save(random_labels, labels_filename)
        else:
            # Append subsequent batches to the file
            existing_data = torch.load(noise_filename)
            existing_labels = torch.load(labels_filename)
            updated_noise = torch.cat((existing_data, noise), dim=0)
            updated_labels = torch.cat((existing_labels, random_labels), dim=0)
            torch.save(updated_noise, noise_filename)
            torch.save(updated_labels, labels_filename)


# Generate queries from the pre-generated noise
def get_queries_dataloader(noise_dataloader, dataset="FashionMNIST", nz=100):
    data_file = "data_tensor.pth"
    labels_file = "labels_tensor.pth"

    if not os.path.isfile(labels_file):
        raise ValueError(
            "Labels file is not present. Cannot generate the victim queries from\
                          thief noise"
        )
    if not os.path.isfile(data_file):
        generate_and_save_queries(noise_dataloader, data_file, dataset=dataset, nz=nz)

    num_queries = len(noise_dataloader) * noise_dataloader.batch_size
    custom_dataset = QueriesDataset(data_file, labels_file, num_queries=num_queries)
    dataloader = DataLoader(
        custom_dataset, batch_size=noise_dataloader.batch_size, shuffle=False
    )

    # This is why get_queries_dataloader should be launched after get_noise_dataloader:
    # they share the same labels and they are deleted in the next line
    delete_helper_tensor_files([data_file, labels_file])

    return dataloader


def generate_and_save_queries(
    noise_dataloader, data_filename, dataset="FashionMNIST", nz=100
):
    """Using pre-generated noise and labels, generate victim images that will be 'queried' by the
    thieves"""

    name_compl = "_cuda" if device == "cuda" else ""
    model_path = f"./models/{device}/Generator_model_{dataset}{name_compl}.pth"
    if dataset == "CelebA":
        n_labels = 2
        target = CelebaGenerator(n_labels * 2)
        target.load_state_dict(
            torch.load(
                f"./models/{device}/Generator_model_CelebA.pth",
                map_location=f"{device}",
            )
        )
        target.to(device)
    else:
        target = torch.load(model_path, map_location=f"{device}")
        target.to(device)

    print(
        f"[+] Creating {len(noise_dataloader)} batches of {noise_dataloader.batch_size} samples"
    )

    for batch_idx in range(len(noise_dataloader)):
        # Load noise and labels from the noise dataloader
        noise, random_labels = next(iter(noise_dataloader))

        if dataset == "CelebA":
            G_label = getGeneratorLabels(
                random_labels.cpu().numpy(), noise_dataloader.batch_size
            ).to(device)
            G_cat = getGeneratorCategories(
                random_labels, noise_dataloader.batch_size
            ).to(device)

            # Useless since we make queries on the go (we need the noise in steal_cgan.accumulate)
            target_output = target(noise, G_label, G_cat)

        else:
            G_label = label_1hots()[random_labels]

            target_output = target(noise, G_label)

        if batch_idx == 0:
            # Save the first batch directly to the file
            torch.save(target_output, data_filename)
        else:
            # Append subsequent batches to the file
            existing_data = torch.load(data_filename)
            updated_data = torch.cat((existing_data, target_output), dim=0)
            torch.save(updated_data, data_filename)
