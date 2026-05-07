#!/bin/bash

# Create conda environment and upgrade pip
env_name=chexagent-env
conda create --name ${env_name} python=3.11 -y # Modify environment name as desired
BASE_ENV_PATH=$(conda info | grep -i 'base environment' | awk '{print $4}')
source "${BASE_ENV_PATH}/etc/profile.d/conda.sh"
conda activate ${env_name}
conda install -y pip
pip install --upgrade pip

# Install CUDA toolkit
conda install -y nvidia::cuda-toolkit=12.8

# Install project
cd ../
pip install uv
uv pip install -e .[all]

# Install deepspeed
uv pip install --no-build-isolation deepspeed

# Install flash-attn (versions higher than 2.7.4.post1 require GLIBC >= 2.32)
uv pip install --no-build-isolation flash-attn==2.3.6

# Install MedVersa-specific dependencies
uv pip install transformers==4.40.0
uv pip install albumentations