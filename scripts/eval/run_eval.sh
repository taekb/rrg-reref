#!/bin/bash -l

# Load conda environment
BASE_ENV_PATH=$(conda info | grep -i 'base environment' | awk '{print $4}')
source "${BASE_ENV_PATH}/etc/profile.d/conda.sh"
conda activate rrg-reref

# Set up environment variables
SCRIPT_DIR=$(dirname $(realpath ${BASH_SOURCE[0]}))
PROJECT_ROOT=${SCRIPT_DIR}/../..
source ${PROJECT_ROOT}/configs/env_vars.sh
export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

[ -z "${model_name}" ] && model_name=google/medgemma-4b-it
[ -z "${dataset_name}" ] && dataset_name=rexval
[ -z "${output_dir}" ] && output_dir=${PROJECT_ROOT}/results
[ -z "${mimic_cxr_parser}" ] && mimic_cxr_parser=llm
[ -z "${split}" ] && split=test
[ -z "${do_sample}" ] && do_sample=False
[ -z "${temperature}" ] && temperature=0
[ -z "${top_p}" ] && top_p=None
[ -z "${max_new_tokens}" ] && max_new_tokens=512
[ -z "${seed}" ] && seed=0
[ -z "${force_eval}" ] && force_eval=False
[ -z "${green_llm_judge_name}" ] && green_llm_judge_name=StanfordAIMI/GREEN-radllama2-7b
[ -z "${crimson_llm_judge_name}" ] && crimson_llm_judge_name=CRIMSONScore/medgemma-4b-it-crimson
[ -z "${use_vllm}" ] && use_vllm=True
[ -z "${include_alt_gt_reports}" ] && include_alt_gt_reports=True
[ -z "${alt_llm_name}" ] && alt_llm_name=o3
[ -z "${use_controlled_alts}" ] && use_controlled_alts=True
[ -z "${alt_sample_idx}" ] && alt_sample_idx=0

log_root="${PROJECT_ROOT}/logs/eval"
log_dir="${log_root}/${dataset_name}/"
if [[ "${dataset_name}" == "mimic-cxr" || "${dataset_name}" == "rexval" ]]; then
    log_dir="${log_dir}/${mimic_cxr_parser}"
fi
mkdir -p "${log_dir}"

if [[ "${do_sample}" == "true" ]]; then
    log_path="${log_dir}/${model_name//\//--}_T=${temperature}_P=${top_p}_t=${max_new_tokens}.log"
else
    log_path="${log_dir}/${model_name//\//--}_t=${max_new_tokens}_greedy.log"
fi

python3 ${PROJECT_ROOT}/rrg_reref/eval/run_eval.py \
    --model_name ${model_name} \
    --dataset_name ${dataset_name} \
    --output_dir ${output_dir} \
    --mimic_cxr_parser ${mimic_cxr_parser} \
    --split ${split} \
    --do_sample ${do_sample} \
    --temperature ${temperature} \
    --top_p ${top_p} \
    --max_new_tokens ${max_new_tokens} \
    --seed ${seed} \
    --force_eval ${force_eval} \
    --green_llm_judge_name ${green_llm_judge_name} \
    --crimson_llm_judge_name ${crimson_llm_judge_name} \
    --use_vllm ${use_vllm} \
    --include_alt_gt_reports ${include_alt_gt_reports} \
    --alt_llm_name ${alt_llm_name} \
    --use_controlled_alts ${use_controlled_alts} \
    --alt_sample_idx ${alt_sample_idx} \
    2>&1 | tee "${log_path}"