import os
import os.path as osp
from pathlib import Path
import fire
from easydict import EasyDict as edict
from tqdm import tqdm, trange
from typing import Optional, Union
from functools import partial
import numpy as np
import pandas as pd
import json
import re
from datetime import datetime

import torch
from accelerate import Accelerator
from accelerate.utils import gather_object

from rrg_reref.utils.constants import (
    INFERENCE_DEFAULT_SEED, 
    MODEL_ALIAS_TO_MODEL_NAME,
    MODEL_TO_SECTIONS,
    SUPPORTED_DATASETS
)
from rrg_reref.utils.models import load_model
from rrg_reref.utils.data import load_dataset
from rrg_reref.utils.logging import rank0_print

def get_image_dir(dataset_name: str) -> str:
    '''Returns the image directory for a given dataset.'''

    if dataset_name == 'mimic-cxr':
        return Path(os.getenv('MIMIC_CXR_JPG_DIR'))

    elif dataset_name == 'chexpertplus':
        return Path(os.getenv('CHEXPERTPLUS_IMAGE_DIR'))

    elif dataset_name == 'iu-xray':
        return Path(os.getenv('IU_XRAY_IMAGE_DIR'))

    elif dataset_name == 'rexgradient':
        return Path(os.getenv('REXGRADIENT_IMAGE_DIR'))

def infer(
    model_name: str,
    dataset_name: str,
    output_dir: Union[str,Path],
    section_to_generate: Optional[str] = None,
    split: Optional[str] = 'test',
    load_pretrained: Optional[bool] = True,
    load_in_8bit: Optional[bool] = False,
    load_in_4bit: Optional[bool] = False,
    double_quant: Optional[bool] = True,
    quant_type: Optional[str] = 'nf4',
    torch_dtype: Optional[torch.dtype] = torch.bfloat16,
    do_sample: Optional[bool] = False,
    temperature: Optional[float] = 0.7,
    top_p: Optional[float] = None,
    max_new_tokens: Optional[int] = 512,
    seed: Optional[int] = INFERENCE_DEFAULT_SEED,
    force_eval: Optional[bool] = False
):
    '''Main inference function.'''

    # Check if the dataset specified is supported
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f'Unknown dataset: {dataset_name}')

    # Check if the model specified is supported
    if model_name not in list(MODEL_TO_SECTIONS.keys()):
        # Check if the alias was used
        if model_name in list(MODEL_ALIAS_TO_MODEL_NAME.keys()):
            model_name = MODEL_ALIAS_TO_MODEL_NAME[model_name]
        else:
            raise ValueError(f'Unknown model: {model_name}')

    # Fetch image directory
    image_dir = get_image_dir(dataset_name)

    # Check if the specified model supports generation of the specified section
    supported_sections = MODEL_TO_SECTIONS[model_name]
    if len(supported_sections) == 1:
        section_to_generate = supported_sections[0]
        print(
            f'Model "{model_name}" only supports generation of the "{section_to_generate}" section.\n'
            f'The "{section_to_generate}" section will be generated regardless of the specified value.'
        )

    # Check if inference has already been done
    output_dir = Path(output_dir) / dataset_name / model_name.replace('/', '--')
    
    if do_sample:
        output_path = output_dir / f'{split}_T={temperature},p={top_p},t={max_new_tokens},seed={seed}_predicted-{section_to_generate}.json'
    else:
        output_path = output_dir / f'{split}_t={max_new_tokens}_predicted-{section_to_generate}_greedy.json'

    if output_path.exists() and not force_eval:
        print(f'Output file already exists: {output_path}. Use --force_eval to overwrite.')
        return

    # Otherwise, check if the model supports generating the specified section
    if section_to_generate not in supported_sections:
        # Stitch together the Findings and Impression sections if the model supports Findings and Impression generation separately
        if section_to_generate == 'findings-and-impression' and ('findings' in supported_sections and 'impression' in supported_sections):
            # Check if the Findings and Impression sections have already been generated
            findings_output_path = Path(str(output_path).replace(section_to_generate, 'findings'))
            impression_output_path = Path(str(output_path).replace(section_to_generate, 'impression'))

            # Check if the Findings and Impression sections have already been generated
            if findings_output_path.exists() and impression_output_path.exists():
                findings_df = pd.read_json(findings_output_path, convert_axes=False)
                findings_df = findings_df.T
                findings_df = findings_df.drop(columns=['study_id'], errors='ignore').reset_index()
                findings_df = findings_df.rename(columns={'index': 'study_id'})

                impression_df = pd.read_json(impression_output_path, convert_axes=False)
                impression_df = impression_df.T
                impression_df = impression_df.drop(columns=['study_id'], errors='ignore').reset_index()
                impression_df = impression_df.rename(columns={'index': 'study_id'})

                # Identify shared study IDs and align rows
                shared_study_ids = set(findings_df['study_id']) & set(impression_df['study_id'])
                findings_df = findings_df.query('study_id in @shared_study_ids').sort_values('study_id').reset_index(drop=True)
                impression_df = impression_df.query('study_id in @shared_study_ids').sort_values('study_id').reset_index(drop=True)
                joint_df = findings_df.copy()
                joint_df['model_prediction'] = (
                    'FINDINGS:\n\n' + findings_df['model_prediction'].values + '\n\n' + \
                    'IMPRESSION:\n\n' + impression_df['model_prediction'].values
                )
                joint_df_json = joint_df.set_index('study_id').T
                joint_df_json.to_json(output_path, indent=4)
                print(f'Saved JSON results to: {output_path}')

                # Also save separate CSV files for ground-truth and predicted reports
                gt_reports_df = joint_df[['study_id', 'section_findings', 'section_impression']].copy()
                gt_reports_df['report'] = (
                    'FINDINGS:\n\n' + gt_reports_df['section_findings'].fillna('') +
                    '\n\n' + 'IMPRESSION:\n\n' + gt_reports_df['section_impression'].fillna('')
                )
                gt_reports_df = gt_reports_df.drop(columns=['section_findings', 'section_impression'])
                gt_reports_df.to_csv(output_dir / f'{split}_ground-truth-{section_to_generate}.csv', index=False)
                print(f'Saved CSV results (ground-truth reports) to: {output_dir / f"{split}_ground-truth-{section_to_generate}.csv"}')

                pred_reports_df = joint_df[['study_id', 'model_prediction']]
                pred_reports_df = pred_reports_df.rename(columns={'model_prediction': 'report'})
                pred_reports_df.to_csv(output_path.with_suffix('.csv'), index=False)
                print(f'Saved CSV results (predicted reports) to: {output_path.with_suffix(".csv")}')

                return

            # Check if either of them exist but not the other
            # In this case, print a warning message and just treat the prediction for the other section as empty.
            # elif findings_output_path.exists() or impression_output_path.exists():
            #     if findings_output_path.exists():
            #         print('[WARNING] Only the Findings section predictions exist. Impression section predictions will be treated as empty.')
            #         findings_df = pd.read_json(findings_output_path)
            #         findings_df = findings_df.T.reset_index()
            #         findings_df = findings_df.rename(columns={'index': 'study_id'})
            #         joint_df = findings_df.copy()
            #         joint_df['model_prediction'] = (
            #             'FINDINGS:\n\n' + findings_df['model_prediction']
            #         )
                
            #     elif impression_output_path.exists():
            #         print('[WARNING] Only the Impression section predictions exist. Findings section predictions will be treated as empty.')
            #         impression_df = pd.read_json(impression_output_path)
            #         impression_df = impression_df.T.reset_index()
            #         impression_df = impression_df.rename(columns={'index': 'study_id'})
            #         joint_df = impression_df.copy()
            #         joint_df['model_prediction'] = (
            #             'IMPRESSION:\n\n' + impression_df['model_prediction']
            #         )

            #     joint_df_json = joint_df.set_index('study_id').T
            #     joint_df_json.to_json(output_path, indent=4)
            #     print(f'Saved JSON results to: {output_path}')

            #     # Also save separate CSV files for ground-truth and predicted reports
            #     if 'section_findings' not in joint_df.columns:
            #         joint_df['section_findings'] = ''
            #     if 'section_impression' not in joint_df.columns:
            #         joint_df['section_impression'] = ''
                
            #     gt_reports_df = joint_df[['study_id', 'section_findings', 'section_impression']].copy()
            #     gt_reports_df['report'] = (
            #         'FINDINGS:\n\n' + gt_reports_df['section_findings'].fillna('') +
            #         '\n\n' + 'IMPRESSION:\n\n' + gt_reports_df['section_impression'].fillna('')
            #     )
            #     gt_reports_df = gt_reports_df.drop(columns=['section_findings', 'section_impression'])
            #     gt_reports_df.to_csv(output_dir / f'{split}_ground-truth-{section_to_generate}.csv', index=False)
            #     print(f'Saved CSV results (ground-truth reports) to: {output_dir / f"{split}_ground-truth-{section_to_generate}.csv"}')

            #     pred_reports_df = joint_df[['study_id', 'model_prediction']]
            #     pred_reports_df = pred_reports_df.rename(columns={'model_prediction': 'report'})
            #     pred_reports_df.to_csv(output_path.with_suffix('.csv'), index=False)
            #     print(f'Saved CSV results (predicted reports) to: {output_path.with_suffix(".csv")}')

            #     return
            else:
                raise RuntimeError(f'Ensure that both the Findings and Impression sections have been generated first.')
        else:
            raise ValueError(f'Model "{model_name}" does not support generation of the "{section_to_generate}" section.')
    
    start_time = datetime.now()
    accelerator = Accelerator()
    device = accelerator.device
    rank0_print_ = partial(rank0_print, rank=accelerator.local_process_index)

    # Load model
    rank0_print_(f'Loading model: "{model_name}"...')
    if 'chex-mimic' in model_name.lower():
        model_load_name = f'chex-mimic-{section_to_generate}'
    elif 'cxrgen' in model_name.lower():
        model_load_name = f'X-iZhang/Med-CXRGen-{section_to_generate.upper()[0]}'
    else:
        model_load_name = model_name

    model, construct_prompt, generate_fn = load_model(
        model_name=model_load_name,
        load_pretrained=load_pretrained,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        double_quant=double_quant,
        quant_type=quant_type,
        torch_dtype=torch_dtype
    )
    model = model.to(device, dtype=torch_dtype)

    # Load dataset
    rank0_print_(f'Loading the JSON file for "{dataset_name}"...')
    dataset = load_dataset(dataset_name, split=split)

    # Iterate through examples and generate outputs
    study_ids = list(dataset.keys())
    generation_configs = edict(
        do_sample=do_sample,
        temperature=(temperature if do_sample else None),
        top_p=(top_p if do_sample else None),
        max_new_tokens=max_new_tokens,
        use_cache=True
    )
    accelerator.wait_for_everyone()
    with accelerator.split_between_processes(study_ids) as acc:
        dataset_with_outputs = {}
        pbar = tqdm(acc, disable=(not accelerator.is_main_process), total=len(acc))
        for study_id in pbar:
            # Load example: ['image_path', 'key_image_path', 'section_findings', 'section_impression', 'frontal_lateral', 'split', 'context', 'report', ...]
            example = dataset[study_id] # Contains all information for a given study
            inputs = construct_prompt(
                example=example, 
                image_dir=image_dir,
                section_to_generate=section_to_generate,
                dataset_name=dataset_name
            )
            try:
                inputs = inputs.to(device)
            except:
                inputs = {
                    k: v.to(device, dtype=torch_dtype) if k == 'images' else v.to(device) 
                    for k, v in inputs.items()
                }

            with torch.inference_mode():
                output_text = generate_fn(**inputs, **generation_configs)

            # Post-processing of the generated text
            # NOTE: Assuming that Impression always appears after the Findings section
            # TODO: Double-check that this works as intended for all models
            if section_to_generate in ['findings', 'impression']:
                # Remove section headers for Findings/Impression generation
                if output_text.lower().startswith(f'{section_to_generate}:'):
                    output_text = output_text[len(section_to_generate)+1:].strip()

                # Remove any irrelevant sections from the generated text
                if section_to_generate == 'findings':
                    match = re.search(r'(Impression|IMPRESSION):', output_text, re.IGNORECASE)
                    if match:
                        output_text = output_text[:match.start()].strip()
                
                elif section_to_generate == 'impression':
                    match = re.search(r'(Findings|FINDINGS):', output_text, re.IGNORECASE)
                    if match:
                        output_text = output_text[match.end():].strip()

            example_with_output = example.copy()
            example_with_output['model_prediction'] = output_text
            dataset_with_outputs[study_id] = example_with_output

    # Gather results - convert to list of (key, value) pairs for better serialization
    accelerator.wait_for_everyone()
    local_results = list(dataset_with_outputs.items())
    all_results = gather_object(local_results)

    if accelerator.is_main_process:
        # Merge all results into a single dictionary
        dataset_with_outputs = {}
        for study_id, example_with_output in all_results:
            dataset_with_outputs[study_id] = example_with_output

        # Write to JSON file
        os.makedirs(osp.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as fh:
            json.dump(dataset_with_outputs, fh, indent=4)

        rank0_print_(f'Saved JSON results to: {output_path}')

        # Also save separate CSV files for ground-truth and predicted reports
        gt_reports = []
        pred_reports = []
        for study_id, example_with_output in dataset_with_outputs.items():
            if section_to_generate == 'findings-and-impression':
                findings = example_with_output['section_findings']
                impression = example_with_output['section_impression']
                findings_text = '' if pd.isna(findings) else str(findings)
                impression_text = '' if pd.isna(impression) else str(impression)
                gt_report = (
                    'FINDINGS:\n\n' + findings_text + '\n\n' +
                    'IMPRESSION:\n\n' + impression_text
                )
            else:
                section_val = example_with_output[f'section_{section_to_generate}']
                gt_report = '' if pd.isna(section_val) else f'{section_to_generate.upper()}:\n\n{section_val}'
            
            pred_report = example_with_output['model_prediction']
            pred_report = '' if pd.isna(pred_report) else str(pred_report)

            gt_reports.append([study_id, gt_report])
            pred_reports.append([study_id, pred_report])

        gt_reports_df = pd.DataFrame(gt_reports, columns=['study_id', 'report'])
        pred_reports_df = pd.DataFrame(pred_reports, columns=['study_id', 'report'])

        gt_reports_path = output_dir / f'{split}_ground-truth-{section_to_generate}.csv'
        if do_sample:
            pred_reports_path = output_dir / f'{split}_T={temperature},p={top_p},t={max_new_tokens},seed={seed}_predicted-{section_to_generate}.csv'
        else:
            pred_reports_path = output_dir / f'{split}_t={max_new_tokens}_predicted-{section_to_generate}_greedy.csv'
        
        # Check for NaN values in 'report' column of ground-truth reports DataFrame
        if gt_reports_df['report'].isnull().values.any():
            print('WARNING: There are NaN values in "report" column of `gt_reports_df`.')

        # Check for NaN values in 'report' column of predicted reports DataFrame
        if pred_reports_df['report'].isnull().values.any():
            print('WARNING: There are NaN values in "report" column of `pred_reports_df`.')

        gt_reports_df.to_csv(gt_reports_path, index=False)
        rank0_print_(f'Saved CSV results (ground-truth reports) to: {gt_reports_path}')
        
        pred_reports_df.to_csv(pred_reports_path, index=False)
        rank0_print_(f'Saved CSV results (predicted reports) to: {pred_reports_path}')

        rank0_print_(f'Elapsed time: {str(datetime.now() - start_time)}')

if __name__ == '__main__':
    fire.Fire(infer)