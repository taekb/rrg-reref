import os
import os.path as osp
from pathlib import Path
import glob
import fire
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from typing import Optional, List
import json
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from datasets import Dataset
from openai import AzureOpenAI
from anthropic import Anthropic
from pydantic import BaseModel
import instructor

from rrg_reref.utils.data import load_dataset
from rrg_reref.reref.axes import get_axis_cls
from rrg_reref.reref.generation import (
    prompt_for_controlled_variation,
    prompt_for_verification
)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'

def generate_alts(
    llm_generator_name: Optional[str] = 'o3',
    llm_verifier_name: Optional[str] = 'o3',
    dataset_name: Optional[str] = 'mimic-cxr',
    axes_to_vary: Optional[list[str]] = [
        'structure',
        #'section_assignment',
        'findings_scope',
        #'anatomical_granularity',
        #'quantitative_granularity',
        #'terminology',
        #'hedging',
        #'negation'
    ],
    n_alts: Optional[int] = 1,
    temperature: Optional[float] = 0.7,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = 16382,
    azure_openai: Optional[bool] = True,
    verify_alts: Optional[bool] = True,
    max_retries: Optional[int] = 5,
):
    '''Generates alternative reports for a given dataset.'''

    start_time = datetime.now()

    # Load the dataset
    dataset = load_dataset(dataset_name, split='test')
    study_ids = list(dataset.keys())

    print(f'Generating {n_alts} alternative reports for "{dataset_name}"...')
    print(f'Generator LLM: {llm_generator_name}')
    print(f'Verifier LLM: {llm_verifier_name}')
    print(f'Temperature: {temperature}')
    print(f'Top P: {top_p}')
    print(f'Max Output Tokens: {max_tokens}')
    print(f'Rejection Sampling: {verify_alts}')
    print(f'Max Retries: {max_retries}\n')

    # Iterate through all axes to vary
    for axis_name in axes_to_vary:
        print(f'Generating alternative references along "{axis_name}" axis...')
        axis_obj = get_axis_cls(axis_name)()

        # Generate all versions if absolute axis
        if axis_obj.axis_type == 'absolute':
            variations_to_generate = axis_obj.variations
        elif axis_obj.axis_type == 'relative':
            variations_to_generate = ['alternative']

        # Generate samples for all possible variations
        for variation in variations_to_generate:
            print(f'Generating samples for variation: "{variation}"')
            outdir = DATA_DIR / dataset_name / 'controlled-alts' / axis_name / variation / llm_generator_name
            outdir.mkdir(parents=True, exist_ok=True)

            # Store the generated datasets for each study, along each variation of current axis
            new_datasets = [{} for _ in range(n_alts)]

            # Store the verification outputs for each alternative report
            verifications = [{} for _ in range(n_alts)]

            # Iterate through all examples in the dataset
            pbar = tqdm(study_ids, total=len(study_ids))
            for study_id in pbar:
                pbar.set_description(f'Study ID: "{study_id}"')
                example = dataset[study_id]
                report = example['report']

                pbar_ = trange(n_alts, desc='Sampling alternative reports', leave=False)
                for i in pbar_:
                    # Implement verification loop for rejection sampling
                    if verify_alts:
                        for _ in trange(max_retries, desc='Generating with validation', leave=False):
                            # Generate the alternative report
                            alt_sections = prompt_for_controlled_variation(
                                orig_report=report, 
                                axis=axis_obj,
                                variation_to_generate=variation,
                                model_name=llm_generator_name,
                                temperature=temperature,
                                top_p=top_p,
                                max_tokens=max_tokens,
                                azure_openai=azure_openai
                            )

                            # Verify the alternative report
                            verification = prompt_for_verification(
                                orig_report=report,
                                alt_sections=alt_sections, 
                                axis_name=axis_name, 
                                variation_generated=variation, 
                                model_name=llm_verifier_name,
                                temperature=0,
                                azure_openai=azure_openai
                            )

                            # Break loop if alternative report is verified
                            if verification.is_clinically_consistent and verification.is_valid_perturbation:
                                break
                    else:
                        alt_sections = prompt_for_controlled_variation(
                            orig_report=report,
                            axis=axis_obj,
                            variation_to_generate=variation,
                            model_name=llm_generator_name,
                            temperature=temperature,
                            top_p=top_p,
                            max_tokens=max_tokens,
                            azure_openai=azure_openai
                        )

                    # Save final sample
                    new_example = example.copy()
                    new_example['section_findings'] = alt_sections['findings']
                    new_example['section_impression'] = alt_sections['impression']
                    new_example['report'] = f'FINDINGS: {alt_sections["findings"]}\n\nIMPRESSION: {alt_sections["impression"]}'
                    new_datasets[i][study_id] = new_example

                    # Save the verification output
                    if verify_alts:
                        verifications[i][study_id] = verification.model_dump()

            for i in range(n_alts):
                # Save the generated datasets
                outpath = outdir / f'test_{i}.json'

                with open(outpath, 'w') as fh:
                    json.dump(new_datasets[i], fh, indent=4)

                print(f'[{i+1}/{n_alts}] Saved generated reports: {outpath}')

                # Save the verification output
                if verify_alts:
                    verification_outpath = outdir / f'test_verification_{i}.json'

                    with open(verification_outpath, 'w') as fh:
                        json.dump(verifications[i], fh, indent=4)

                    print(f'[{i+1}/{n_alts}] Saved verification output: {verification_outpath}')

    print(f'Elapsed: {str(datetime.now() - start_time)}')
    
if __name__ == '__main__':
    fire.Fire(generate_alts)