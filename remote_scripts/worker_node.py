"""
Worker (agent) node for real implementation of collusive theft of a target cGAN.

SSH tunneling should be set between central server and workers, and between workers and victim.
This script is meant to be run before central server is launched as starting server will broadcast 
model to workers. By design (which can be changed easily), server will send the model to 5001, 
5002, ... , 500Na so each worker N ∈ {1 ... Na} should listen on 500N.

Example:

    $ # On victim's side
    $ python target_machine.py --port=5000

    $ # On worker 3's side
    $ python worker_node.py --port=5003 --victim_port=5000

    $ # On server's side
    $ python central_server.py --port=5004 --Na=3

"""

# Import necessary libraries
import torch
from torch.utils.data import DataLoader, TensorDataset
from flask import Flask, request, jsonify
import requests
from models.cgan import Generator
import utils
import numpy as np
import torch.nn.functional as F
import json
import argparse
import os
import time


# Initialize theft variables
if True:
    app = Flask(__name__)

    # Check if a GPU is available
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    parser = argparse.ArgumentParser(description='Run the Flask app with a specified port.')
    parser.add_argument('--port', type=int, default=5001, help='Port number for the Flask app.')
    parser.add_argument('--victim_port', type=int, default=5000, help='Port number of victim.')

    args = parser.parse_args()

    # Parameters of the theft
    num_requests, batch_size = 23, 128

    # Variable that stops querying
    out_of_queries = False

    # Placeholder
    received_model = Generator(1, 16, 100)


# Theft is triggered everytime central server sends updated model
@app.route('/receive_model', methods=['POST'])
def steal():
    
    global received_model, model_has_been_initialized, worker_id
    print(f"[W{worker_id}] Received model")

    json_data = request.data
    jsonLoads = json.loads(json_data)

    received_model = create_received_model(jsonLoads).to(device) 

    # If the worker has credits left, he sends a query.
    # Eveytime, he must load all data that he already got to compute gradients to send to
    # central server
    global out_of_queries, full_replies_dataloader

    if not out_of_queries:

        # Get victim samples
        try:
            noise_to_query, label_to_query = next(full_noise_dataloader_iter)
            print(f"[W{worker_id}] Querying")
            reply_images, corresponding_label = query(noise_to_query, label_to_query)
            save_reply_images(reply_images)

        # If worker falls out of queries, create a full_replies_dataloader containing all reply
        # images that will be used for endless training
        except StopIteration:
            print(f"[W{worker_id}] Reached the end of the noise DataLoader.")
            out_of_queries = True
            full_replies_dataloader, _ = get_train_dataloaders()

    # If there are no queries left, then images used for gradient are existing full dataloaders
    if out_of_queries:
        train_images_dl, train_noise_dl = full_replies_dataloader, full_noise_dataloader

    # If there are queries left, get dataloaders for the replies we have
    else:
        train_images_dl, train_noise_dl = get_train_dataloaders()

    #
    # --- Constitute gradient payload and respond to server
    #

    # Get gradients
    print(f"[W{worker_id}] Computing gradients from vicitm reply")
    thief_gradients = compute_gradients(received_model, train_images_dl, train_noise_dl)

    serialized_gradients = {}

    for idx, tensor in enumerate(thief_gradients):
        serialized_tensor = tensor.tolist()
        serialized_gradients[f'tensor{idx}'] = serialized_tensor

    # Create a JSON payload
    payload = {'thief_gradients': serialized_gradients,
               'worker_id': worker_id  # Worker identification needed so server knows it's a go
               }

    # Convert the dictionary to a JSON string
    json_payload = json.dumps(payload)

    print(f"[W{worker_id}] Uploading gradients")
    response = requests.post('http://localhost:5004/gradients', json=json_payload)
    return jsonify({"worker_id": 1})


# Get available (already queried) replies and corresponding noise
def get_train_dataloaders():

    # Load available reply images
    replies_data = torch.load(replies_filename)

    # Load full labels + clipping it to number of available replies (full labels is never updated)
    labels_data = torch.load(labels_filename)[:len(replies_data)]
    #                                          ^--- returned dataloader only has that many labels

    replay_dataset = TensorDataset(replies_data, labels_data)
    replies_dataloader = DataLoader(replay_dataset, batch_size=batch_size, shuffle=False)

    # Load full noise but clipping it to number of available replies (full noise is never updated)
    noise_data = torch.load(noise_filename)[:len(replies_data)]
    noise_dataset = TensorDataset(noise_data, labels_data)
    noise_dataloader = DataLoader(noise_dataset, batch_size=batch_size, shuffle=False)

    return replies_dataloader, noise_dataloader


# Query vicitm model
def query(noise, labels, target_url=f'http://localhost:{args.victim_port}/generate'):

    # Convert PyTorch tensors to NumPy arrays
    noise_array = noise.cpu().detach().numpy().tolist()
    labels_array = labels.cpu().detach().numpy().tolist()

    # Create a JSON payload
    payload = {
        'noise': noise_array,
        'labels': labels_array
    }

    # Make a POST request with the JSON payload
    response = requests.post(target_url, json=payload)

    generated_images = response.json()['generated_images']

    # Check the response
    print(f"[W{worker_id}] Target says:", response.status_code)

    # Extract noise and labels from the JSON payload
    generated_images_array = np.array(generated_images)

    # Convert NumPy arrays to PyTorch tensors
    generated_images_tensor = torch.tensor(generated_images_array).to(device, dtype=torch.float32)

    return generated_images_tensor, labels


def compute_gradients(thief_model, train_images_dl, train_noise_dl):
    # Placeholder for gradients
    thief_gradients = [torch.zeros_like(param) for param in thief_model.parameters()]

    thief_model.eval()

    for images_batch, noise_batch in zip(train_images_dl, train_noise_dl):
        # Assuming both dataloaders have the same labels for each batch
        reply_images, corresponding_label = images_batch
        noise, _ = noise_batch

        labels_1hot = utils.label_1hots()[corresponding_label]

        # Move tensors to the specified device
        reply_images = reply_images.to(device)
        noise = noise.to(device)
        labels_1hot = labels_1hot.to(device)

        # Pass
        thief_model.zero_grad()
        thief_output = thief_model(noise, labels_1hot)
        thief_train_loss = F.l1_loss(thief_output, reply_images)
        thief_train_loss.backward()

        # Accumulate gradients
        for k, param in enumerate(thief_model.parameters()):
            thief_gradients[k] += param.grad.clone()

    # Average the accumulated gradients over the entire dataset
    num_batches = len(train_images_dl)
    thief_gradients = [grad / num_batches for grad in thief_gradients]

    return thief_gradients


# Called every time victim answers query() within steal() to save queries for further training
def save_reply_images(reply_images):
    if not os.path.exists(replies_filename):
        # Save the first batch directly to the file
        torch.save(reply_images, replies_filename)
    else:
        # Append subsequent batches to the file
        existing_replies = torch.load(replies_filename)
        updated_replies = torch.cat((existing_replies, reply_images), dim=0)
        torch.save(updated_replies, replies_filename)


# Create+load received model from central server to avoid memory collisions
def create_received_model(jsonLoads):
    keys = [
        "image.0.weight", "image.1.weight", "image.1.bias", "image.1.running_mean",
        "image.1.running_var", "label.0.weight", "label.1.weight", "label.1.bias",
        "label.1.running_mean", "label.1.running_var", "main.0.weight", "main.1.weight",
        "main.1.bias", "main.1.running_mean", "main.1.running_var", "main.3.weight",
        "main.4.weight", "main.4.bias", "main.4.running_mean", "main.4.running_var",
        "main.6.weight"
    ]

    empy_model_state_dict = {key: '' for key in keys}
    # Serialize the state dictionary to a JSON-formatted string
    model_state_dict = jsonLoads

    # Convert NumPy arrays back to PyTorch tensors
    for key, value in model_state_dict.items():
        empy_model_state_dict[key] = torch.tensor(value, dtype=torch.float32)

    new_FT = Generator(1, 16, 100)
    new_FT.load_state_dict(empy_model_state_dict)
    return new_FT


if __name__ == '__main__':

    worker_id = int(str(args.port)[-1])

    # Specify the directory where the files are located
    directory_path = '.'

    # Specify the file patterns to be removed
    file_patterns = [f'noise_tensor_{worker_id}', f'replies_tensor_{worker_id}', 
                     f'labels_tensor_{worker_id}']

    # Iterate through the file patterns and remove matching files
    for pattern in file_patterns:
        files_to_remove = [f for f in os.listdir(directory_path) if f.startswith(pattern)]
        for file_name in files_to_remove:
            file_path = os.path.join(directory_path, file_name)
            os.remove(file_path)
            print(f"Removed: {file_path}")

    noise_filename = f"noise_tensor_{worker_id}.pth"
    labels_filename = f"labels_tensor_{worker_id}.pth"
    replies_filename = f"replies_tensor_{worker_id}.pth"

    full_replies_dataloader = None
    full_replay_iterator = None

    # This is all the noise that will be sent by this worker, i.e. num_requests batches of size 
    # batch_size
    full_noise_dataloader = utils.get_noise_dataloader(num_requests,
                                                       batch_size,
                                                       noise_file=noise_filename,
                                                       labels_file=labels_filename,
                                                       nz=100,
                                                       delete_after=False)
    full_noise_dataloader_iter = iter(full_noise_dataloader)

    app.run(host='0.0.0.0', port=args.port)
