# PINN Solver for Inverse Diffusion Problem

Welcome to the **PINN Solver for Inverse Diffusion Problem**! This repository contains a MATLAB implementation of a Physics-Informed Neural Network (PINN) designed to solve inverse diffusion problems using a 0-10 plaque domain with boundary conditions and a sinusoidal initial condition.

## Overview

The PINN Solver for Inverse Diffusion Problem is an advanced computational tool built to tackle inverse problems in diffusion dynamics using modern machine learning techniques. Leveraging Physics-Informed Neural Networks (PINNs), this solver integrates physical laws into the learning process, ensuring accurate and efficient solutions.

### Problem Formulation

- **Domain:** 0 ≤ x ≤ 10, 0 ≤ y ≤ 10
- **Boundary Conditions:** \( u(x, y, t) = 0 \) on the domain boundaries
- **Initial Condition:** Sinusoidal distribution 

This setup is ideal for testing and benchmarking PINN-based solutions for diffusion problems with known boundary and initial conditions.

## Features

- **MATLAB Implementation:** Ready-to-run code for MATLAB users.
- **PINN Integration:** Harnesses the power of PINNs to solve inverse diffusion problems efficiently.
- **Lightning Integration:** Utilizes PyTorch Lightning for streamlined training and evaluation of neural networks. (Note: This feature assumes PyTorch and PyTorch Lightning are installed and properly configured.)
- **Weights & Biases (wandb) Control:** Integrates with Weights & Biases for experiment tracking, hyperparameter tuning, and visualization. Ensure you have a wandb account and API key configured.

## Installation

To get started with the PINN Solver, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/LucasUnizar/PINN_pl_diff_2D.git

2. **Navigate to the Directory:**
   ```bash
   cd PINN_pl_diff_2D

3. **Add Dependencies:**
   ```bash
   pip install torch torchvision torchaudio pytorch-lightning wandb

## Script Usage
### Command-Line Arguments
--train (optional): If specified, the model will be trained. If not specified, the model will only be tested.
--input_dim (optional, default: 3): The dimension of the input features.
--hidden_dim (optional, default: [64, 64]): A list specifying the dimensions of hidden layers.
--output_dim (optional, default: 2): The dimension of the output.
--max_epochs (optional, default: 50): The maximum number of epochs for training.
--batch_size (optional, default: 128): The batch size used during training.
--lr (optional, default: 1e-4): The learning rate for training.
--wandb_project (optional, default: 'PINN-Project'): The name of the project in Weights & Biases.
--wandb_entity (optional): The entity name for Weights & Biases. This is required to log to a specific entity.

## Running the Script
**To Train the Model**
To train the model, run the script with the --train flag. You can customize other parameters as needed.
   ```bash
   python script.py --train --input_dim 4 --hidden_dim 128 64 --output_dim 3 --max_epochs 100 --batch_size 64 --lr 1e-5 --wandb_project MyPINNProject --wandb_entity my_wandb_entity

**To Test the Model**
To test the model, run the script without the --train flag. Ensure the model has been previously trained.
   ```bash
   python script.py --input_dim 4 --hidden_dim 128 64 --output_dim 3 --max_epochs 100 --batch_size 64 --lr 1e-5 --wandb_project MyPINNProject --wandb_entity my_wandb_entity




   
