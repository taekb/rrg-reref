#!/bin/bash

# MODIFY ALL FIELDS AS NEEDED.
#
# This file centralizes paths and credentials used by the inference, evaluation,
# and perturbation pipelines. All shell scripts under ./scripts source this
# file, so set everything here once and the rest of the pipelines will pick it up.

# Project root directory
export PROJECT_ROOT="/path/to/rrg-reref"

# Azure OpenAI configs (only needed for LLM-based metrics / perturbations)
export AZURE_OPENAI_API_KEY=""
export AZURE_OPENAI_ENDPOINT=""
export AZURE_OPENAI_API_VERSION=""

# OpenAI configs
export OPENAI_API_KEY=""
export OPENAI_BASE_URL=""

# Anthropic configs
export ANTHROPIC_API_KEY=""

# HuggingFace configs
export HF_TOKEN=""
export HF_HUB_CACHE="/path/to/hf_models"
export HF_DATASETS_CACHE="/path/to/hf_datasets"

# Kaggle configs
export KAGGLE_API_TOKEN=""
export KAGGLEHUB_CACHE="/path/to/kagglehub"

# CheXbert model path (used by RadEval-based metrics)
export CHEXBERT_PATH="/path/to/chexbert/chexbert.pth"

# RadGraph model cache directory
export RADGRAPH_CACHE="/path/to/radgraph"

# MIMIC-CXR directory (PhysioNet release; used to read raw reports)
export MIMIC_CXR_DIR="/path/to/physionet.org/files/mimic-cxr/2.0.0"

# MIMIC-CXR-JPG directory (PhysioNet release; used for images)
export MIMIC_CXR_JPG_DIR="/path/to/physionet.org/files/mimic-cxr-jpg/2.0.0"

# IU X-ray directory (Kaggle release; downloaded via kagglehub)
export IU_XRAY_DIR="${KAGGLEHUB_CACHE}/datasets/raddar/chest-xrays-indiana-university/versions/2"
export IU_XRAY_IMAGE_DIR="${IU_XRAY_DIR}/images/images_normalized"
export IU_XRAY_R2GEN_DIR="/path/to/iu_xray"

# CheXpert Plus directory
export CHEXPERTPLUS_DIR="/path/to/chexpertplus"
export CHEXPERTPLUS_IMAGE_DIR="${CHEXPERTPLUS_DIR}/PNG"

# ReXGradient-160K directory (HuggingFace release)
export REXGRADIENT_DIR="${HF_DATASETS_CACHE}/datasets--rajpurkarlab--ReXGradient-160K/snapshots/<snapshot-hash>"
export REXGRADIENT_IMAGE_DIR="${REXGRADIENT_DIR}"
