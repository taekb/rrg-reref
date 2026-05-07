#!/bin/bash -l

#export CUDA_LAUNCH_BLOCKING=1
#export NCCL_ASYNC_ERROR_HANDLING=1
#export NCCL_BLOCKING_WAIT=1
#export NCCL_DEBUG=INFO
#export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True

# Load conda environment
BASE_ENV_PATH=$(conda info | grep -i 'base environment' | awk '{print $4}')
source "${BASE_ENV_PATH}/etc/profile.d/conda.sh"

# Set up environment variables
SCRIPT_DIR=$(dirname $(realpath ${BASH_SOURCE[0]}))
PROJECT_ROOT=${SCRIPT_DIR}/../..
source ${PROJECT_ROOT}/configs/env_vars.sh

[ -z "${model_name}" ] && model_name=google/medgemma-4b-it
[ -z "${dataset_name}" ] && dataset_name=rexval
[ -z "${output_dir}" ] && output_dir=${PROJECT_ROOT}/results
[ -z "${mimic_cxr_parser}" ] && mimic_cxr_parser=llm
[ -z "${section_to_generate}" ] && section_to_generate=findings
[ -z "${split}" ] && split=test
[ -z "${load_pretrained}" ] && load_pretrained=True
[ -z "${load_in_8bit}" ] && load_in_8bit=False
[ -z "${load_in_4bit}" ] && load_in_4bit=False
[ -z "${do_sample}" ] && do_sample=False
[ -z "${temperature}" ] && temperature=0
[ -z "${top_p}" ] && top_p=None
[ -z "${max_new_tokens}" ] && max_new_tokens=512
[ -z "${seed}" ] && seed=0
[ -z "${force_eval}" ] && force_eval=False

# Use different environments for different models
if [[ "${model_name}" =~ "maira-2" ]]; then
    conda activate maira-env
elif [[ "${model_name}" =~ "MedVersa" || "${model_name}" =~ "medversa" ]]; then
    conda activate medversa-env
elif [[ "${model_name}" =~ "CheXagent" || "${model_name}" =~ "chexagent" ]]; then
    conda activate chexagent-env
elif [[ "${model_name}" =~ "llava-rad" ]]; then
    conda activate llava-rad-env
elif [[ "${model_name}" =~ "libra" || "${model_name}" =~ "med-cxrgen" ]]; then
    conda activate libra-env
else
    conda activate rrg-reref
fi

[ -z "${gpu_num}" ] && gpu_num=$(python -c "import torch; print(torch.cuda.device_count())")

# FIXME: Figure out a better fix. There is an I/O issue that only occurs under DDP inference.
if [[ "${model_name}" =~ "llava-rad" || "${model_name}" =~ "libra" || "${model_name}" =~ "med-cxrgen" ]]; then
    gpu_num=1
fi

log_root="${PROJECT_ROOT}/logs/inference"
log_dir="${log_root}/${dataset_name}"
if [[ "${dataset_name}" == "mimic-cxr" || "${dataset_name}" == "rexval" ]]; then
    log_dir="${log_dir}/${mimic_cxr_parser}"
fi
mkdir -p "${log_dir}"

if [[ "${do_sample}" == "true" ]]; then
    log_path="${log_dir}/${model_name//\//--}_T=${temperature}_P=${top_p}_t=${max_new_tokens}.log"
else
    log_path="${log_dir}/${model_name//\//--}_t=${max_new_tokens}_greedy.log"
fi

# Auto-assign a free port so parallel runs don't collide on the default 29500.
main_process_port=$(python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')

accelerate_config_path=${PROJECT_ROOT}/configs/dist/inference.yaml
accelerate launch --num_processes=${gpu_num} --main_process_port ${main_process_port} --config_file=${accelerate_config_path} ${PROJECT_ROOT}/rrg_reref/eval/run_inference.py \
    --model_name ${model_name} \
    --dataset_name ${dataset_name} \
    --output_dir ${output_dir} \
    --mimic_cxr_parser ${mimic_cxr_parser} \
    --section_to_generate ${section_to_generate} \
    --split ${split} \
    --load_pretrained ${load_pretrained} \
    --load_in_8bit ${load_in_8bit} \
    --load_in_4bit ${load_in_4bit} \
    --do_sample ${do_sample} \
    --temperature ${temperature} \
    --top_p ${top_p} \
    --max_new_tokens ${max_new_tokens} \
    --seed ${seed} \
    --force_eval ${force_eval} \
    2>&1 | tee "${log_path}"