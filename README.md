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
- **Boundary Conditions:** Automatically enforces \( u(x, y, t) = 0 \) on the domain limits.
- **Initial Condition:** Utilizes a sinusoidal initial condition for realistic testing.
- **PINN Integration:** Harnesses the power of PINNs to solve inverse diffusion problems efficiently.

## Installation

To get started with the PINN Solver, follow these steps:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/LucasUnizar/PINN_pl_diff_2D.git
