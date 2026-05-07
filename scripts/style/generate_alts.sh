#!/bin/bash

# Load conda environment
BASE_ENV_PATH=$(conda info | grep -i 'base environment' | awk '{print $4}')
source "${BASE_ENV_PATH}/etc/profile.d/conda.sh"
conda activate rrg-reref

# Set up environment variables
SCRIPT_DIR=$(dirname $(realpath ${BASH_SOURCE[0]}))
PROJECT_ROOT=${SCRIPT_DIR}/../..
source ${PROJECT_ROOT}/configs/env_vars.sh

[ -z "${llm_generator_name}" ] && llm_generator_name=o3
[ -z "${llm_verifier_name}" ] && llm_verifier_name=o3
[ -z "${dataset_name}" ] && dataset_name=rexval
[ -z "${mimic_cxr_parser}" ] && mimic_cxr_parser=llm
[ -z "${controlled}" ] && controlled=True
[ -z "${n_alts}" ] && n_alts=1
[ -z "${temperature}" ] && temperature=0.7
[ -z "${top_p}" ] && top_p=None
[ -z "${max_tokens}" ] && max_tokens=16382
[ -z "${azure_openai}" ] && azure_openai=False
[ -z "${verify_alts}" ] && verify_alts=True
[ -z "${max_retries}" ] && max_retries=5

log_root="${PROJECT_ROOT}/logs/style"
if [[ "${controlled}" == "True" ]]; then
    log_dir="${log_root}/${dataset_name}/${llm_generator_name}/controlled"
else
    log_dir="${log_root}/${dataset_name}/${llm_generator_name}/random"
fi
if [[ "${dataset_name}" == "mimic-cxr" || "${dataset_name}" == "rexval" ]]; then
    log_dir="${log_dir}/${mimic_cxr_parser}"
fi
mkdir -p "${log_dir}"
log_path="${log_dir}/n_alts=${n_alts}_T=${temperature}_P=${top_p}_t=${max_tokens}.log"

python3 ${PROJECT_ROOT}/rrg_reref/style/generate_alts.py \
    --llm_generator_name ${llm_generator_name} \
    --llm_verifier_name ${llm_verifier_name} \
    --dataset_name ${dataset_name} \
    --mimic_cxr_parser ${mimic_cxr_parser} \
    --controlled ${controlled} \
    --n_alts ${n_alts} \
    --temperature ${temperature} \
    --top_p ${top_p} \
    --max_tokens ${max_tokens} \
    --azure_openai ${azure_openai} \
    --verify_alts ${verify_alts} \
    --max_retries ${max_retries} \
    2>&1 | tee "${log_path}"