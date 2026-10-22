# rrg-reref

This is official code accompanying the NeurIPS 2026 E&D Track paper *"Reporting Practice Matters: The Impact of Reference Choice on Chest X-ray Report Evaluation"* (Jeong et al., 2026).

This repository provides:
- The **stylistic perturbation pipeline** for generating LLM-rewritten alternative reference reports along controlled style axes (structure, scope, granularity, terminology, hedging, negation, etc.).
- The **inference pipeline** for running a suite of chest X-ray report-generation VLMs (MAIRA-2, MedGemma, MedVersa, CheXagent, LLaVA-Rad, Libra, Med-CXRGen, CheXOne, CheX-MIMIC).
- The **evaluation pipeline** with implementations of standard NLG and clinical-fidelity metrics (BLEU, ROUGE, BERTScore, CheXbert-F1, RadGraph-F1, RadGraph-XL-F1, RadCliQ-v1, RaTEScore, GREEN, CRIMSON), including evaluation against the LLM-rewritten alternative references.

## Repository layout
```
rrg-reref/
├── configs/
│   ├── env_vars.sh         # Paths and credentials (edit before running anything)
│   └── dist/inference.yaml # Accelerate config for multi-GPU inference
├── rrg_reref/
│   ├── eval/               # Inference and evaluation entry points
│   ├── style/              # Style-axis definitions and alternative-report generation
│   ├── models/             # Per-model inference wrappers
│   └── utils/              # Data loading, metric implementations (CRIMSON, GREEN, RadCliQ), MIMIC-CXR parsers
├── scripts/
│   ├── eval/               # Shell wrappers for run_inference.py / run_eval.py
│   └── style/              # Shell wrapper for generate_alts.py
└── setup/                  # Conda environment setup scripts (one per model family)
```

## Setup

### 1. Install the main environment

```bash
cd setup
./setup.sh
```

This creates a conda environment named `rrg-reref` (Python 3.11) and installs the project in editable mode along with all dependencies.

### 2. (Optional) Install per-model environments

Some models pin incompatible versions of `transformers` / `flash-attn` and need their own conda environment. Install only the ones you plan to run:

```bash
cd setup
./setup_maira.sh        # creates `maira-env` (MAIRA-2)
./setup_medversa.sh     # creates `medversa-env` (MedVersa)
./setup_chexagent.sh    # creates `chexagent-env` (CheXagent)
./setup_llava_rad.sh    # creates `llava-rad-env` (LLaVA-Rad)
./setup_libra.sh        # creates `libra-env` (Libra, Med-CXRGen)
```

`scripts/eval/run_inference.sh` automatically picks the right environment based on `model_name`. MedGemma, CheXOne, and CheX-MIMIC run in the base `rrg-reref` environment.

### 3. Configure paths and credentials

Edit `configs/env_vars.sh` and fill in:
- `PROJECT_ROOT` — absolute path to this repo.
- Dataset directories — `MIMIC_CXR_DIR`, `MIMIC_CXR_JPG_DIR`, `REXVAL_DIR`, `IU_XRAY_DIR`, `CHEXPERTPLUS_DIR`, `REXGRADIENT_DIR`. Datasets must be obtained from their original sources (PhysioNet, Kaggle, HuggingFace) and are not redistributed here.
- API keys — `OPENAI_API_KEY` / `AZURE_OPENAI_*` / `ANTHROPIC_API_KEY` (only needed for the LLM-driven perturbation step and for the GREEN/CRIMSON metrics when used with API-hosted judges).
- Model-cache directories — `HF_HUB_CACHE`, `HF_DATASETS_CACHE`, `KAGGLEHUB_CACHE`, `CHEXBERT_PATH`, `RADGRAPH_CACHE`.

All shell scripts under `scripts/` source this file automatically.

### 4. Prepare the dataset JSON files

The pipelines expect per-dataset JSON files at `data/<dataset>/<split>.json` (or `data/<dataset>/<split>_<parser>.json` for MIMIC-CXR / ReXVal). Each file maps `study_id → {report, findings, impression, image_paths, ...}`. Build these from the raw datasets before running anything; the `rrg_reref.utils.mimic_cxr.parse` module contains the LLM-based MIMIC-CXR section parser used to produce `*_llm.json`.

## Running the perturbation pipeline

The perturbation pipeline rewrites each ground-truth reference report into one or more alternative versions that vary along a specified style axis (e.g. structured vs. free-text, exhaustive vs. minimal findings scope). Variations are defined in `rrg_reref/style/axes.py` and the prompts in `rrg_reref/style/generation.py`.

```bash
cd scripts/style

# Defaults: dataset=rexval, generator=verifier=o3, controlled mode, 1 alt per study
bash generate_alts.sh

# Override any parameter via environment variables before invoking
dataset_name=mimic-cxr llm_generator_name=o3 n_alts=3 bash generate_alts.sh
```

Key parameters (set as env vars before calling the script):
- `dataset_name`: `rexval`, `mimic-cxr`, `chexpertplus`, `iu-xray`, `rexgradient`.
- `llm_generator_name` / `llm_verifier_name`: any model name supported by `rrg_reref.utils.llm_api.prompt_llm` (OpenAI, Azure OpenAI, or Anthropic).
- `controlled`: `True` → vary one style axis at a time; `False` → free-form rewrite.
- `n_alts`: number of alternative reports to sample per study.
- `verify_alts`: `True` → reject alternatives whose findings are not clinically equivalent to the original (rejection sampling, up to `max_retries`).
- `azure_openai`: route OpenAI calls through Azure.

The set of style axes to vary is configured inside `rrg_reref/style/generate_alts.py` (`style_axes_to_vary`). Outputs are written to `data/<dataset>/controlled-alts/<axis>/<variation>/<generator>/` (or `random-alts/<generator>/` when `controlled=False`).

You can also call the entry point directly:

```bash
python rrg_reref/style/generate_alts.py \
    --dataset_name rexval \
    --llm_generator_name o3 \
    --controlled True \
    --n_alts 1 \
    --verify_alts True
```

## Running the evaluation pipeline

The evaluation pipeline has two stages: (1) generating model predictions via `run_inference.py`, and (2) scoring predictions against the original (and optionally LLM-rewritten) references via `run_eval.py`.

### Stage 1 — model inference

```bash
cd scripts/eval

# Single model + dataset (defaults: medgemma-4b-it on rexval, findings section)
bash run_inference.sh

# Override defaults
model_name=chexagent dataset_name=mimic-cxr section_to_generate=findings bash run_inference.sh
```

Useful overrides:
- `model_name`: any alias from `rrg_reref/utils/constants.py::MODEL_ALIAS_TO_MODEL_NAME` (`maira-2`, `medgemma-4b-it`, `medversa`, `chexagent`, `llava-rad`, `libra`, `med-cxrgen`, `chexone`) or a full HuggingFace ID.
- `dataset_name`: `mimic-cxr`, `rexval`, `chexpertplus`, `iu-xray`, `rexgradient`.
- `section_to_generate`: `findings`, `impression`, or `findings-and-impression`.
- `mimic_cxr_parser`: `llm` (default) or `mimic_code` for MIMIC-CXR / ReXVal.
- `do_sample`, `temperature`, `top_p`, `max_new_tokens`, `seed`: generation hyperparameters.
- `force_eval=True`: overwrite existing predictions.

Predictions are written to `${output_dir}/<dataset>/[<parser>/]<model>/<split>_..._predicted-<section>.json`. The script picks the right conda environment for each model, auto-assigns a free `accelerate` port, and runs DDP across all visible GPUs.

### Stage 2 — metric evaluation

```bash
cd scripts/eval

# Defaults: medgemma-4b-it on rexval, includes alt-reference evaluation (controlled, axis sample 0, generator=o3)
bash run_eval.sh

# Override defaults
model_name=chexagent dataset_name=mimic-cxr include_alt_gt_reports=True bash run_eval.sh
```

Useful overrides:
- All of the inference-stage overrides above (the eval script reuses the same prediction-file naming convention).
- `include_alt_gt_reports`: `True` → also evaluate against LLM-rewritten alternative references generated by `generate_alts.py`.
- `use_controlled_alts`: `True` → use controlled (per-axis) alternatives; `False` → use random-style alternatives.
- `alt_llm_name`: which generator's alternatives to use (must match a directory under `data/<dataset>/controlled-alts/...`).
- `alt_sample_idx`: which sample index to evaluate against when multiple alternatives were generated.
- `green_llm_judge_name` / `crimson_llm_judge_name`: judge models for the GREEN and CRIMSON metrics. Set `use_vllm=True` to serve open-weights judges (e.g. `StanfordAIMI/GREEN-radllama2-7b`) locally via vLLM; set to `False` to call hosted APIs (`claude-sonnet-4-5`, `gpt-4`, etc.).

Metrics computed (configurable via `metrics_to_compute` in `rrg_reref/eval/run_eval.py`):
`BLEU-2`, `BLEU-4`, `ROUGE`, `BERTScore`, `CheXbert-F1`, `RadGraph-F1`, `RadGraph-XL-F1`, `RadCliQ-v1`, `RaTEScore`, `GREEN`, `CRIMSON`.

## Datasets

Datasets must be obtained from their original distributors and placed at the paths set in `configs/env_vars.sh`:

| Dataset | Source | Env var |
| --- | --- | --- |
| MIMIC-CXR (reports) | PhysioNet | `MIMIC_CXR_DIR` |
| MIMIC-CXR-JPG (images) | PhysioNet | `MIMIC_CXR_JPG_DIR` |
| ReXVal | PhysioNet | `REXVAL_DIR` |
| CheXpert Plus | Stanford AIMI | `CHEXPERTPLUS_DIR` |
| IU X-ray (Indiana University) | Kaggle (`raddar/chest-xrays-indiana-university`) | `IU_XRAY_DIR` |
| ReXGradient-160K | HuggingFace (`rajpurkarlab/ReXGradient-160K`) | `REXGRADIENT_DIR` |

This repository does not contain any patient data, model checkpoints, or generated reports.

## License

MIT.
