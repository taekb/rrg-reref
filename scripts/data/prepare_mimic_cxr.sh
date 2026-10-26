#!/bin/bash

# Load conda environment
BASE_ENV_PATH=$(conda info | grep -i 'base environment' | awk '{print $4}')
source "${BASE_ENV_PATH}/etc/profile.d/conda.sh"
conda activate rrg-reref

# Set up environment variables
SCRIPT_DIR=$(dirname $(realpath ${BASH_SOURCE[0]}))
PROJECT_ROOT=${SCRIPT_DIR}/../..
source ${PROJECT_ROOT}/configs/env_vars.sh

[ -z "${split}" ] && split=test
[ -z "${llm_name}" ] && llm_name=claude-sonnet-4-5
[ -z "${azure_openai}" ] && azure_openai=False

log_dir="${PROJECT_ROOT}/logs/data/mimic-cxr"
mkdir -p "${log_dir}"
log_path="${log_dir}/${split}_${llm_name}.log"

python3 ${PROJECT_ROOT}/rrg_reref/utils/mimic_cxr/prepare.py \
    --split ${split} \
    --llm_name ${llm_name} \
    --azure_openai ${azure_openai} \
    2>&1 | tee "${log_path}"
