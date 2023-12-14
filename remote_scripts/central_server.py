"""
Central server for real implementation of collusive theft of a target cGAN.

SSH tunneling should be set between central server and workers, and between workers and victim.
This script is meant to be run last (after workers and vicitm are online) as it will trigger the 
broadcast of the model to the workers. By default, server listens to 3 workers on 5004.
By design (which can be changed easily), workers listen on 500N, where N is worker id∈{1 ... Na}.

Example:

    $ # On victim's side
    $ python target_machine.py --port=5000

    $ # On worker 3's side
    $ python worker_node.py --port=5003 --victim_port=5000

    $ # On server's side
    $ python central_server.py --port=5004 --Na=3

"""

# Import necessary libraries
from flask import Flask, request, jsonify
import requests
import torch
import threading
from models.cgan import Generator
import io
import utils
import torchvision
from PIL import Image
import orjson as json
import argparse
import zipfile

# Initialize theft variables
if True:
    app = Flask(__name__)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    parser = argparse.ArgumentParser(description='Run the Flask app with a specified port.')
    parser.add_argument('--port', type=int, default=5004, help='Port number for the Flask app.')
    parser.add_argument('--Na', type=int, default=3, help='Number of workers in the theft')

    args = parser.parse_args()

    # Ack list to keep track of who sent their gradients. If all did, update is permitted
    has_worker_sent_gradients = dict()
    for worker_i in range(args.Na):
        has_worker_sent_gradients[worker_i + 1] = False

    # Parameters of the theft
    num_requests, batch_size = 50, 128

    GT = Generator(1, 16, 100).to(device)

    # Define an optimizer for the global model (Global Thief)
    GT_optimizer = torch.optim.Adam(GT.parameters(), lr=0.002, betas=(0.5, 0.5))   

    # List of gradient tensors
    gradient_aggregator = dict()

    round_has_ended = True

    epoch = 0


# Make fixed noise and labels to assess the learning
def make_fixed_noise():
    # Create batch of latent vectors and laebls that we will use to visualize the progression of 
    # the generator 
    fixed_noise = torch.randn(2048, 100, 1, 1).to(device)
    fixed_label_natural = torch.randint(0, 10, (2048,)).sort().values.to(device)
    fixed_label = utils.label_1hots()[fixed_label_natural]
    return fixed_noise.to(device), fixed_label.to(device)


# Initialize writers and noise
if True:
    # Create batch of latent vectors and laebls that we use to visualize progression of generator
    fixed_noise, fixed_label = make_fixed_noise()


# Generate picture with central version of the thief
def writer_backlog(model):
    model.eval()
    with torch.no_grad():

        fake_fixed = model(fixed_noise.to(device), fixed_label.to(device)).cpu()

        img_grid = torchvision.utils.make_grid(fake_fixed[:min(24, 128)], normalize=True)

        img_grid_normalized = (img_grid - img_grid.min()) / (img_grid.max() - img_grid.min())

        # Convert the tensor to a PIL Image
        img_array = (img_grid_normalized * 255).cpu().numpy()
        img_pil = Image.fromarray(img_array.astype('uint8').transpose((1, 2, 0)))

        # Save the image
        img_pil.save("./thief_generation_server_side.png")


# Aggregates received gradients then calls model_update()
def aggregate_gradients_then_update(thief_gradients, worker_id):

    def save_tensors_to_zip(tensor_dict, zip_file_path):
        with zipfile.ZipFile(zip_file_path, 'w') as archive:
            for key, value in tensor_dict.items():
                buffer = io.BytesIO()
                torch.save(value, buffer)
                buffer.seek(0)
                archive.writestr(key, buffer.read())

    def load_tensors_from_zip(zip_file_path):
        tensor_dict = {}
        with zipfile.ZipFile(zip_file_path, 'r') as archive:
            for entry in archive.infolist():
                key = entry.filename
                buffer = io.BytesIO(archive.read(entry))
                tensor_dict[key] = torch.load(buffer)
        return tensor_dict

    global gradient_aggregator, has_worker_sent_gradients

    zip_file_path = 'saved_tensors.zip'

    # Save the tensors to a zip file
    save_tensors_to_zip(thief_gradients, zip_file_path)

    # Load the tensors back from the zip file
    loaded_tensors = load_tensors_from_zip(zip_file_path)

    for key, value in loaded_tensors.items():

        # Check if the key exists in the dictionary
        if key in gradient_aggregator:
            # Key exists, add the tensor to the existing value
            gradient_aggregator[key] += torch.tensor(value).to(device)
        else:
            # Key doesn't exist, create the key in the dictionary
            gradient_aggregator[key] = torch.tensor(value).to(device)

    # If all workers have sent their gradients:
    for k in range(len(gradient_aggregator)):
        gradient_aggregator[f'tensor{k}'] /= len(has_worker_sent_gradients)

    print(f"[Server] Gradients from worker {worker_id} aggregated")
    has_worker_sent_gradients[worker_id] = True
    print(f"[Server] Worker ack list: {has_worker_sent_gradients}")

    if False not in has_worker_sent_gradients.values():
        model_update()


def model_update():
    print("[Server] Updating model")

    global gradient_aggregator, GT, GT_optimizer, round_has_ended, epoch
    # Update the global model with the averaged gradients
    for global_param, averaged_grad in zip(GT.parameters(), gradient_aggregator.values()):
        global_param.grad = averaged_grad.clone()

    # Manually update model parameters using optimizer
    GT_optimizer.step()

    # Clear gradients for the next iteration
    GT_optimizer.zero_grad()

    #
    # --- Reset aggreg
    #
    zeros_param = [torch.zeros_like(param) for param in GT.parameters()]

    # Check if the lengths match
    if len(gradient_aggregator) != len(zeros_param):
        raise ValueError("The lengths of the aggregator and the GT parameters list do not match.")

    # Update values in gradient_aggregator with corresponding elements from parameters_list
    for i, (key, value) in enumerate(gradient_aggregator.items()):
        zero_like = zeros_param[i]
        gradient_aggregator[key] = zero_like

    writer_backlog(GT)
    epoch += 1
    round_has_ended = True


# Theft routine
def steal():
    print("[Server] Started theft")
    global has_worker_sent_gradients, GT, round_has_ended

    while True and round_has_ended:
        round_has_ended = False

        # Reinitialize workers
        for worker_id in has_worker_sent_gradients:
            has_worker_sent_gradients[worker_id] = False

        # Cumbersome implementation but limits mem-crashes on cluster
        # Multi-threading easy to implement if needed.
        for worker_id in has_worker_sent_gradients:
            if has_worker_sent_gradients[worker_id] is False:

                GT_state_dict = GT.state_dict()

                # Convert tensors to NumPy arrays
                GT_state_dict_numpy = {
                    key: value.cpu().detach().numpy().tolist() if isinstance(value, torch.Tensor) 
                    else value for key, value in GT_state_dict.items()
                }

                # Serialize the state dictionary to a JSON-formatted string
                model_payload = json.dumps(GT_state_dict_numpy)

                # Send model to worker
                response = requests.post(f'http://localhost:500{worker_id}/receive_model',
                                         data=model_payload)

                # We keep 500N as a convention to prevent an additional handshake


# Once workers receive the model, they answer by sending new gradients on this endpoint
@app.route('/gradients', methods=['POST'])
def gradient_receiver():
    payload = json.loads(request.json)
    worker_id = payload['worker_id']
    gradient_payload = payload['thief_gradients']
    print(f"[Server] Got response from worker {worker_id}")

    aggregate_gradients_then_update(gradient_payload, worker_id)
    print("[Server] Done aggregating gradients and updated")
    has_worker_sent_gradients[worker_id] = True
    return jsonify({"message": "ack"})


# Start sending the model periodically in a separate thread
send_model_thread = threading.Thread(target=steal)
send_model_thread.daemon = True
send_model_thread.start()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=args.port)
