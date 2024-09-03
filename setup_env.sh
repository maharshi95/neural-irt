#!/bin/bash

# Define the environment name and Python version
ENV_NAME="caimira"
PYTHON_VERSION="3.10"

# Check if conda is available
if ! command -v conda &>/dev/null; then
    echo "Conda could not be found. Please install it and try again."
    exit 1
fi

# Create the conda environment
echo "Creating a conda environment named $ENV_NAME with Python $PYTHON_VERSION"
conda create -y --name $ENV_NAME python=$PYTHON_VERSION

# Activate the environment
echo "Activating the conda environment: $ENV_NAME"
conda activate $ENV_NAME

# Install pip in the conda environment
echo "Installing pip in the conda environment"
conda install pip -y

pip install -U pip

