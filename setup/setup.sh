#!/bin/bash

# Create conda environment and upgrade pip
env_name=rrg-reref
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

# Install flash-attn from source (pre-built wheels may be compiled against a different
# torch/libstdc++; --no-binary forces a local source build against the installed torch)
#uv pip install --no-build-isolation --no-binary flash-attn flash-attn==2.7.4.post1
uv pip install --no-build-isolation flash-attn==2.7.4.post1

# Ensure the conda env's libstdc++ (which has CXXABI 1.3.15 from the GCC 14 toolchain)
# is used at runtime instead of the system's older copy. This is needed for flash-attn
# and other CUDA extensions compiled with the conda compiler.
ENV_PREFIX=$(conda info --envs | grep "^${env_name} " | awk '{print $NF}')
mkdir -p "${ENV_PREFIX}/etc/conda/activate.d"
mkdir -p "${ENV_PREFIX}/etc/conda/deactivate.d"

cat > "${ENV_PREFIX}/etc/conda/activate.d/env_vars.sh" << 'ACTIVATE'
#!/bin/bash
export _OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
ACTIVATE

cat > "${ENV_PREFIX}/etc/conda/deactivate.d/env_vars.sh" << 'DEACTIVATE'
#!/bin/bash
export LD_LIBRARY_PATH="${_OLD_LD_LIBRARY_PATH}"
unset _OLD_LD_LIBRARY_PATH
DEACTIVATE