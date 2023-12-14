"""
Victim (target) model for real implementation of collusive theft of a target cGAN.

SSH tunneling should be set between central server and workers, and between workers and victim.
This script is meant to be run before central server is launched as starting server will broadcast 
model to workers that in turn immediately query the victim. By default, victim listens on 5000.

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
import numpy as np
from models.cgan import Generator
from flask import Flask, request, jsonify
import argparse


if True:
    app = Flask(__name__)

    # Check if a GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target = torch.load('models/Generator_model_FashionMNIST_cuda.pth').to(device)

    parser = argparse.ArgumentParser(description='Run the Flask app with a specified port.')
    parser.add_argument('--port', type=int, default=5000, help='Port number for the Flask app.')

    args = parser.parse_args()


@app.route('/generate', methods=['POST'])
def generate_images():
    data = request.get_json()

    # Extract noise and labels from the JSON payload
    noise_array = np.array(data.get('noise'))
    labels_array = np.array(data.get('labels'))

    # Convert NumPy arrays to PyTorch tensors
    noise_tensor = torch.tensor(noise_array).to(device, dtype=torch.float32)
    labels_tensor = torch.tensor(labels_array).to(device, dtype=torch.float32)

    # Generate images using the cGAN model
    generated_images = target(noise_tensor, labels_tensor)
    print("generated_images ", generated_images.shape)

    return jsonify({'generated_images': generated_images.tolist()})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=args.port)
