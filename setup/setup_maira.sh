#!/bin/bash

# Create conda environment and upgrade pip
env_name=maira-env
conda create --name ${env_name} python=3.11 -y
BASE_ENV_PATH=$(conda info | grep -i 'base environment' | awk '{print $4}')
source "${BASE_ENV_PATH}/etc/profile.d/conda.sh"
conda activate ${env_name}
conda install -y pip
pip install --upgrade pip
pip install uv

# Install a newer libstdc++ via conda so that compiled C++ extensions
# (flash-attn, deepspeed) can find CXXABI_1.3.15+, which is missing from
# the host system's GCC 9 runtime on Ubuntu 20.04.
conda install -y -c conda-forge libstdcxx-ng>=12

# Set up conda activation hooks so LD_LIBRARY_PATH always includes the
# env's lib/ directory.  This ensures the conda-provided libstdc++ is
# found before the (older) system copy at /lib/x86_64-linux-gnu/.
mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d"
mkdir -p "${CONDA_PREFIX}/etc/conda/deactivate.d"

cat > "${CONDA_PREFIX}/etc/conda/activate.d/libstdcxx_ld_path.sh" <<'ACTIVATE_EOF'
#!/bin/sh
export _OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ACTIVATE_EOF

cat > "${CONDA_PREFIX}/etc/conda/deactivate.d/libstdcxx_ld_path.sh" <<'DEACTIVATE_EOF'
#!/bin/sh
export LD_LIBRARY_PATH="${_OLD_LD_LIBRARY_PATH:-}"
unset _OLD_LD_LIBRARY_PATH
DEACTIVATE_EOF

# Install project
cd ../
uv pip install -e .[all]

# Downgrade transformers version
uv pip install transformers==4.51.3

# Install deepspeed
uv pip install --no-build-isolation deepspeed

# Install flash-attn (versions higher than 2.7.4.post1 require GLIBC >= 2.32)
uv pip install --no-build-isolation flash-attn==2.7.4.post1