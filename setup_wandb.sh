#!/bin/bash

# Create wandb directories
mkdir -p ~/wandb/data ~/wandb/cache ~/wandb/config

# Set and export the environment variables
echo "Exporting wandb environment variables..."
export WANDB_DIR=~/wandb/data
export WANDB_CACHE_DIR=~/wandb/cache
export WANDB_CONFIG_DIR=~/wandb/config

# Ensure write permissions for the wandb directories
echo "Setting permissions for wandb directories..."
chmod -R 755 ~/wandb

# Add the exports to bashrc to persist the variables across sessions
echo "Persisting the environment variables to ~/.bashrc..."
echo 'export WANDB_DIR=~/wandb/data' >> ~/.bashrc
echo 'export WANDB_CACHE_DIR=~/wandb/cache' >> ~/.bashrc
echo 'export WANDB_CONFIG_DIR=~/wandb/config' >> ~/.bashrc

echo "Wandb setup completed!"

