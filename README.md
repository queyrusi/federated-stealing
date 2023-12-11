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
python run_exp.py --num_QperA 200 --n_thieves 2 --defense 'FGSM' --dataset 'FashionMNIST'

```