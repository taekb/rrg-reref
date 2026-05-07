#!/bin/bash

# Create conda environment and upgrade pip
env_name=libra-env
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

# Install libra package
uv pip install "Libra[train,eval] @ git+https://github.com/X-iZhang/Libra.git"
