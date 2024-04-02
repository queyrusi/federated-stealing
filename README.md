# On Collusive Stealing of Conditional GANs

## Overview

This repository contains code related to the paper "On Collusive Stealing of Conditional GANs." The research focuses on collusive stealing attacks against Conditional Generative Adversarial Networks (cGANs).

## Requirements

Ensure you have the required dependencies installed. To install them, run:

```bash
pip install -r requirements.txt
```

## Usage
To run the experiments in simulation, run e.g.

```bash
python run_exp.py --num_QperA=200 --n_thieves=2 --defense='FGSM' --dataset='FashionMNIST'

```

## Concrete implementation
GeoD scripts are provided inside the `remote_scripts` folder: each script should be placed inside the machine with the dedicated role (several machines can be worker nodes) with a copy of the model architecture inside a `models/` folder and the utility file `utils.py` at the root:

```
machine_root/
│
├── central_server.py, worker_node.py or victim_machine.py
├── utils.py
└── models/
    └── cgan.py
```

Please ensure ssh connection is established between server and worker machines, and bewteen workers and victim machine as well prior to launch.
By design, workers and victim apps are to be started before the central server:

```bash
    # On victim's side
    python target_machine.py --port=5000
```
```bash
    # On worker 1's side
    python worker_node.py --port=5001 --victim_port=5000
```
```bash
    # On worker 2's side
    python worker_node.py --port=5002 --victim_port=5000
```
```bash
    # On server's side
    python central_server.py --port=5004 --Na=3
```

## Victim model $G^T$ training parameters

$G^T$ and $D^T$ were trained on public train splits of CelebA (condition on Male, Female, Black hair and non Black hair)

| Param                                   | Value                        |
|-----------------------------------------|------------------------------|
| Num. epochs                             | 20                           |
| Batch size                              | 128                          |
| Loss                                    | BCE                          |
| $n_z$ (latent space dim.)               | 100                          |
| Filters                                 | 16                           |
| Learning rate                           | 0.0002                       |
| Optimizer                               | Adam                         |
| $\beta_1, \beta_2$                      | 0.5,0.999                    |
| $p_z, p_y$                              | uniform                      |
