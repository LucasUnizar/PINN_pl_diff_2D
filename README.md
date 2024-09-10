<div align="center"> 

# PINN Solver for Inverse Diffusion Problem

[![Project page](https://img.shields.io/badge/-Project%20page-green)](https://amb.unizar.es/people/lucas/tesan/)
[![Linkedln](https://img.shields.io/badge/-Linkdln%20page-blue)](https://www.linkedin.com/in/lucas-tesan-ingbiozar/)

</div>

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

|     Argument              |             Description                           | Options                                               |
|---------------------------| ------------------------------------------------- |------------------------------------------------------ |
| `--train`                 | Train mode                                        | `True`, `False`                                       |
| `--input_dim`             | Input dimension int                               | Default: `3`                                          |
| `--ioutput_dim`           | Output dimension int                              | Default: `2`                                          |
| `--dim_hidden`            | Dimension of hidden layers                        | Default: `70`                                         |
| `--lr`                    | Learning rate                                     | Default: `1e-4`                                       |
| `--batch_size`            | Training batch size                               | Default: `2`                                          |
| `--max_epoch`             | Maximum number of training epochs                 | Default: `1500`                                       |

## Running the Script
**To Train the Model**
To train the model, run the script with the --train flag. You can customize other parameters as needed.
   ```bash
   python script.py --train --input_dim 4 --hidden_dim 128 64 --output_dim 3 --max_epochs 100 --batch_size 64 --lr 1e-5 --wandb_project MyPINNProject --wandb_entity my_wandb_entity
   ```

**To Test the Model**
To test the model, run the script without the --train flag. Ensure the model has been previously trained.
   ```bash
   python script.py --input_dim 4 --hidden_dim 128 64 --output_dim 3 --max_epochs 100 --batch_size 64 --lr 1e-5 --wandb_project MyPINNProject --wandb_entity my_wandb_entity
   ```

## Plots and gifs
**Here is a visualization of the inference in the hole rollout for u(x,y,t):**

<div align="center">
<img src="/outputs/plots/pred_vs_gt_evolution_3D.gif" width="450">
</div>

**Here we can follow the evolution of the inference of the diffusion coefficent:**

<div align="center">
<img src="/outputs/plots/alpha_evolution_3D.gif" width="450">
</div>

**Finally, we can also plot the quadratic error for the simulation:**

<div align="center">
<img src="/outputs/plots/qerror_evolution_3D.gif" width="450">
</div>

## Neural Network Training Overview

This section provides an overview of the neural network training process, focusing on the decomposition of the total loss into its constituent components and the results of solving the inverse problem.
The total loss function is defined as the sum of several loss components, each measuring a different aspect of the model's performance:

**Data Loss:** Measures the error between the model predictions and observed data, guiding the network to fit the empirical data accurately.

**Physics Loss:** Ensures the model predictions adhere to physical laws and constraints, improving the consistency and plausibility of the results.

**BC and IC Loss:** Enforces the boundary and initial conditions, ensuring the model's predictions are accurate at critical points, enhancing overall reliability.

The total loss is defined as the sum of individual loss components:

\[
\text{Total Loss} = \sum_{i=1}^{n} \text{Loss}_i
\]

Where:
- \( \text{Loss}_i \) represents each individual loss component contributing to the total loss.

In addition, the training evolution of these metrics is shown as a line graph for 50 epoch:

<div align="center">
<img src="/graphic_material/lossplot.png" width="450">
</div>

On the other hand, the estimation of the diffusion coefficient is shown as part of the solution of the inverse problem:

<div align="center">
<img src="/graphic_material/Dplot.png" width="450">
</div>

With a final value of 9,54 and considering a ground thruth of 10 in de numerical solver.
