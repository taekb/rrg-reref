# Reporting Practice Matters: The Impact of Reference Choice on Chest X-ray Report Evaluation

This is the official repository for the NeurIPS 2026 E&D Track paper:
> Daniel P. Jeong, Charles Q. Li, Hossein Hosseiny, Nitya M. Bhalla, Fatma Uyar Morency, Pradeep Ravikumar, Zachary C. Lipton, Michael Oberst. [Reporting Practice Matters: The Impact of Reference Choice on Chest X-ray Report Evaluation](https://arxiv.org/abs/2609.19093). *Neural Information Processing Systems (NeurIPS) 2026 Evaluations and Datasets Track, 2026*

This repository includes:
- Our proposed **ReRef pipeline**, which rewrites reference radiology reports along the axes of our taxonomy of reporting practice variation (Structure, Section Assignment, Findings Scope, Anatomical Granularity, Quantitative Granularity, Terminology, Hedging, Negation) while preserving clinical interpretation. This is the pipeline used to construct our MIMIC-CXR-Ext-ReRef dataset.
- The **inference pipeline** for running a suite of chest X-ray report-generation VLMs (MAIRA-2, MedGemma, MedVersa, CheXagent, LLaVA-Rad, Libra, Med-CXRGen, CheXOne, CheX-MIMIC).
- The **evaluation pipeline** with implementations of standard NLG and clinical-fidelity metrics (BLEU, ROUGE, BERTScore, CheXbert-F1, RadGraph-F1, RadGraph-XL-F1, RadCliQ-v1, RaTEScore, GREEN, CRIMSON), including evaluation against the LLM-rewritten alternative references.

## Repository Layout
```
rrg-reref/
├── configs/
│   ├── env_vars.sh         # Paths and credentials (edit before running anything)
│   └── dist/inference.yaml # Accelerate config for multi-GPU inference
├── rrg_reref/
│   ├── eval/               # Inference and evaluation entry points
│   ├── reref/              # ReRef: taxonomy axes, Generator/Verifier prompts, alternative-reference generation
│   ├── models/             # Per-model inference wrappers
│   └── utils/              # Data loading, metric implementations (CRIMSON, GREEN, RadCliQ), MIMIC-CXR section extraction
├── scripts/
│   ├── data/               # Shell wrapper for MIMIC-CXR data preparation
│   ├── eval/               # Shell wrappers for run_inference.py / run_eval.py
│   └── reref/              # Shell wrapper for generate_alts.py
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
- Dataset directories — `MIMIC_CXR_DIR`, `MIMIC_CXR_JPG_DIR`, `IU_XRAY_DIR`, `CHEXPERTPLUS_DIR`, `REXGRADIENT_DIR`. Datasets must be obtained from their original sources (PhysioNet, Kaggle, HuggingFace) and are not redistributed here.
- API keys — `OPENAI_API_KEY` / `AZURE_OPENAI_*` / `ANTHROPIC_API_KEY` (only needed for the LLM-driven perturbation step and for the GREEN/CRIMSON metrics when used with API-hosted judges).
- Model-cache directories — `HF_HUB_CACHE`, `HF_DATASETS_CACHE`, `KAGGLEHUB_CACHE`, `CHEXBERT_PATH`, `RADGRAPH_CACHE`.

All shell scripts under `scripts/` source this file automatically.

### 4. Prepare the dataset JSON files

The pipelines expect per-dataset JSON files at `data/<dataset>/<split>.json`. Each file maps `study_id → {report, section_findings, section_impression, image_path, key_image_path, ...}`. Build these from the raw datasets before running anything.

For MIMIC-CXR, the Findings and Impression sections are not provided in parsed form, so we extract all sections from each full report with an LLM (Claude Sonnet 4.5), using the prompt in Appendix C of the paper (`rrg_reref/utils/mimic_cxr/parse.py`). To build `data/mimic-cxr/test.json` from the official test split:

```bash
cd scripts/data
bash prepare_mimic_cxr.sh   # requires MIMIC_CXR_DIR, MIMIC_CXR_JPG_DIR, and ANTHROPIC_API_KEY in configs/env_vars.sh
```

Studies with neither a Findings nor an Impression section are excluded. This makes one LLM call per study (3,269 studies in the test split).

## Running the Perturbation Pipeline (ReRef)

The perturbation pipeline (ReRef) rewrites each ground-truth reference report into one or more alternative versions that vary along a specified axis of reporting practice (e.g. structured vs. free-text, exhaustive vs. minimal findings scope). Axes are organized under four dimensions of our taxonomy (Organization, Completeness, Granularity, and Language), as described in Section 2.1 of the paper. Each alternative is sampled from a Generator LLM and checked by a Verifier LLM for clinical equivalence and validity of the perturbation (rejection sampling). Axes and their variations are defined in `rrg_reref/reref/axes.py` and the Generator/Verifier prompts in `rrg_reref/reref/generation.py`. All LLM-facing prompt text is kept verbatim as used to produce the results in the paper.

```bash
cd scripts/reref

# Defaults: dataset=mimic-cxr, generator=verifier=o3, 1 alt per study
bash generate_alts.sh

# Override any parameter via environment variables before invoking
dataset_name=mimic-cxr llm_generator_name=o3 n_alts=3 bash generate_alts.sh
```

Key parameters (set as env vars before calling the script):
- `dataset_name`: `mimic-cxr`, `chexpertplus`, `iu-xray`, `rexgradient`.
- `llm_generator_name` / `llm_verifier_name`: any model name supported by `rrg_reref.utils.llm_api.prompt_llm` (OpenAI, Azure OpenAI, or Anthropic).
- `n_alts`: number of alternative reports to sample per study.
- `verify_alts`: `True` → reject alternatives whose findings are not clinically equivalent to the original (rejection sampling, up to `max_retries`).
- `azure_openai`: route OpenAI calls through Azure.

Each run varies one axis at a time. The set of axes to vary is configured inside `rrg_reref/reref/generate_alts.py` (`axes_to_vary`). Outputs are written to `data/<dataset>/controlled-alts/<axis>/<variation>/<generator>/`.

You can also call the entry point directly:

```bash
python rrg_reref/reref/generate_alts.py \
    --dataset_name mimic-cxr \
    --llm_generator_name o3 \
    --n_alts 1 \
    --verify_alts True
```

## Running the evaluation pipeline

The evaluation pipeline has two stages: (1) generating model predictions via `run_inference.py`, and (2) scoring predictions against the original (and optionally LLM-rewritten) references via `run_eval.py`.

### Stage 1 — model inference

```bash
cd scripts/eval

# Single model + dataset (defaults: medgemma-4b-it on mimic-cxr, findings section)
bash run_inference.sh

# Override defaults
model_name=chexagent dataset_name=mimic-cxr section_to_generate=findings bash run_inference.sh
```

Useful overrides:
- `model_name`: any alias from `rrg_reref/utils/constants.py::MODEL_ALIAS_TO_MODEL_NAME` (`maira-2`, `medgemma-4b-it`, `medversa`, `chexagent`, `llava-rad`, `libra`, `med-cxrgen`, `chexone`) or a full HuggingFace ID.
- `dataset_name`: `mimic-cxr`, `chexpertplus`, `iu-xray`, `rexgradient`.
- `section_to_generate`: `findings`, `impression`, or `findings-and-impression`.
- `do_sample`, `temperature`, `top_p`, `max_new_tokens`, `seed`: generation hyperparameters.
- `force_eval=True`: overwrite existing predictions.

Predictions are written to `${output_dir}/<dataset>/<model>/<split>_..._predicted-<section>.json`. The script picks the right conda environment for each model, auto-assigns a free `accelerate` port, and runs DDP across all visible GPUs.

### Stage 2 — metric evaluation

```bash
cd scripts/eval

# Defaults: medgemma-4b-it on mimic-cxr, includes alt-reference evaluation (axis sample 0, generator=o3)
bash run_eval.sh

# Override defaults
model_name=chexagent dataset_name=mimic-cxr include_alt_gt_reports=True bash run_eval.sh
```

Useful overrides:
- All of the inference-stage overrides above (the eval script reuses the same prediction-file naming convention).
- `include_alt_gt_reports`: `True` → also evaluate against the alternative references generated by `generate_alts.py`, along each axis in `axis_names`.
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
| CheXpert Plus | Stanford AIMI | `CHEXPERTPLUS_DIR` |
| IU X-ray (Indiana University) | Kaggle (`raddar/chest-xrays-indiana-university`) | `IU_XRAY_DIR` |
| ReXGradient-160K | HuggingFace (`rajpurkarlab/ReXGradient-160K`) | `REXGRADIENT_DIR` |

This repository does not contain any patient data, model checkpoints, or generated reports.

## 🙂 Citing Our Work (BibTeX)

```bibtex
@article{jeong2026reporting,
  title={Reporting Practice Matters: The Impact of Reference Choice on Chest X-ray Report Evaluation},
  author={Jeong, Daniel P. and Li, Charles Q. and Hosseiny, Hossein and Bhalla, Nitya M. and Uyar Morency, Fatma and Ravikumar, Pradeep and Lipton, Zachary C. and Oberst, Michael},
  journal={arXiv preprint arXiv:2609.19093},
  year={2026}
}
```
