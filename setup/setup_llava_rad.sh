#!/bin/bash

# Create conda environment and upgrade pip
env_name=llava-rad-env
conda create --name ${env_name} python=3.11 -y # Modify environment name as desired
BASE_ENV_PATH=$(conda info | grep -i 'base environment' | awk '{print $4}')
source "${BASE_ENV_PATH}/etc/profile.d/conda.sh"
conda activate ${env_name}
conda install -y pip
pip install --upgrade pip
pip install uv

# Install CUDA toolkit
#conda install -y nvidia::cuda-toolkit=12.8

# Install project
cd ../
uv pip install -e .[all]

# Install deepspeed
uv pip install --no-build-isolation deepspeed

# Install flash-attn (versions higher than 2.7.4.post1 require GLIBC >= 2.32)
uv pip install --no-build-isolation flash-attn==2.3.6

# Install LLaVA-Rad
uv pip install git+https://github.com/microsoft/llava-rad.git