#!/bin/bash

# Script for training the model
source /mnt/cephfs/home/amb/ltesan/miniconda3/bin/activate venv

# Execute the Python script with train argument and additional training parameters
python main.py --sweep --train

