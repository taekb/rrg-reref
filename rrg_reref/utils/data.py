import sys
import os
import os.path as osp
from pathlib import Path
import glob
from collections import defaultdict
from typing import Union, Optional, Dict, List
from easydict import EasyDict as edict
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import Levenshtein
from redlines import Redlines
from IPython.display import display, Markdown

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'

'''
    Utilities for data loading & preprocessing and visualization.

'''

def load_dataset(
    dataset_name: str,
    split: Optional[str] = 'test',
    load_alts: Optional[bool] = False,
    axis_name: Optional[str] = None, # Name of reporting practice axis considered for controlled perturbation
    variation: Optional[str] = None, # Specific variation along reporting practice axis considered
    llm_name: Optional[str] = 'claude-sonnet-4-5',
    sample_idx: Optional[int] = 0, # Index of the LLM-generated sample
    load_verification: Optional[bool] = False
) -> Dict:
    '''Loads a given dataset.'''

    # Load alternative version of the dataset, with LLM-generated reports
    if load_alts:
        # Controlled perturbation along a single axis
        assert axis_name is not None and variation is not None, (
            'Both axis_name and variation must be specified when loading alternative references.'
        )
        json_dir = DATA_DIR / dataset_name / 'controlled-alts' / axis_name / variation / llm_name
        data_path = json_dir / f'{split}_{sample_idx}.json'
        verification_path = json_dir / f'{split}_verification_{sample_idx}.json'
    
    # Load the original ground-truth dataset
    else:
        data_path = DATA_DIR / dataset_name / f'{split}.json'
    
    with open(data_path) as fh:
        dataset = edict(json.load(fh))

    if load_alts and load_verification:
        with open(verification_path) as fh:
            verification = edict(json.load(fh))
        
        return dataset, verification
    else:
        return dataset

def exclude_empty_reports(
    reports_df: pd.DataFrame,
    verbose: Optional[bool] = False
) -> pd.DataFrame:
    '''Excludes reports that are empty.'''
    
    n_orig = len(reports_df)
    reports_df = reports_df[~reports_df['report'].isna()]
    reports_df = reports_df[reports_df['report'].str.strip() != '']
    reports_df = reports_df.reset_index(drop=True)
    n_filtered = n_orig - len(reports_df)
    if n_filtered > 0:
        if verbose:
            print(f'Excluded {n_filtered} rows with empty reports.')

    return reports_df

def gather_gt_sections(
    dataset: Dict, 
    section: str, 
    filter_missing_rows: bool = True, 
    verbose: Optional[bool] = False
) -> pd.DataFrame:
    '''Gathers the ground-truth sections for a given dataset.'''

    study_ids = list(dataset.keys())
    gt_reports_dict = defaultdict(list)
    for study_id in study_ids:
        if section == 'findings-and-impression':
            findings = dataset[study_id]['section_findings']
            impression = dataset[study_id]['section_impression']

            # Skip if both are empty
            if (pd.isna(findings) or findings.strip() == '') and (pd.isna(impression) or impression.strip() == ''):
                continue

            findings_text = '' if pd.isna(findings) else findings
            impression_text = '' if pd.isna(impression) else impression

            gt_reports_dict['study_id'].append(study_id)
            gt_reports_dict['report'].append(
                'FINDINGS:\n\n' + findings_text + '\n\n' + \
                'IMPRESSION:\n\n' + impression_text
            )
        else:
            gt_reports_dict['study_id'].append(study_id)
            gt_reports_dict['report'].append(dataset[study_id][f'section_{section}'])

    gt_reports_df = pd.DataFrame(gt_reports_dict)
    gt_reports_df = exclude_empty_reports(gt_reports_df, verbose=verbose)
    gt_reports_df = gt_reports_df.sort_values(by=['study_id']).reset_index(drop=True)

    return gt_reports_df

def visualize_diffs(
    orig_report: str,
    alt_report: str,
    render: Optional[bool] = True
):
    '''Annotates the differences between the original and alternative reports.'''

    diffs = Redlines(orig_report, alt_report).output_markdown

    if render:
        display(Markdown(diffs))
    else:
        return diffs

def compute_edit_distance(orig_report: str, alt_report: str) -> int:
    '''Computes the edit distance between the original and alternative reports.'''
    
    return Levenshtein.distance(orig_report, alt_report)

def retrieve_alts_for_study(
    dataset_name: str,
    study_id: int,
    split: Optional[str] = 'test',
    axis_names: Optional[List[str]] = [
        'structure',
        'section_assignment',
        'findings_scope',
        'anatomical_granularity',
        'quantitative_granularity',
        'terminology',
        'hedging',
        'negation'
    ],
    llm_name: Optional[str] = 'o3', 
    section_to_retrieve: Optional[str] = 'findings',
    sample_idx: Optional[Union[int, List[int]]] = 0,
    select_meaningful_alts: Optional[bool] = True,
    verbose: Optional[bool] = True
) -> Dict[str,Dict[str,Optional[str]]]:
    '''Retrieves all axis-wise alternative reports for a given study.'''

    if verbose:
        print(f'Retrieving alternative reports for study "{study_id}" in "{dataset_name}" dataset...')
    
    # Retrieve axis-wise variations
    try:
        from rrg_reref.reref.axes import get_axis_cls
    except:
        sys.path.append(
            str(Path(os.getenv('PROJECT_ROOT')) / 'rrg_reref' / 'reref')
        )
        from axes import get_axis_cls

    alt_sections_dict = defaultdict(dict)
    pbar = tqdm(axis_names, desc='Loading axis variations', disable=(not verbose))
    for axis_name in pbar:
        axis_obj = get_axis_cls(axis_name)()

        if axis_obj.axis_type == 'absolute':
            variations = axis_obj.variations
        else:
            variations = ['alternative']

        for variation in variations:
            alt_dataset, alt_verification = load_dataset(
                dataset_name=dataset_name,
                split=split,
                load_alts=True,
                axis_name=axis_name,
                variation=variation,
                llm_name=llm_name,
                sample_idx=sample_idx,
                load_verification=True
            )
            if section_to_retrieve == 'findings-and-impression':
                findings = alt_dataset[str(study_id)]['section_findings']
                impression = alt_dataset[str(study_id)]['section_impression']
                findings_text = '' if pd.isna(findings) else findings
                impression_text = '' if pd.isna(impression) else impression
                alt_section = (
                    'FINDINGS:\n\n' + findings_text + '\n\n' +
                    'IMPRESSION:\n\n' + impression_text
                )
            else:
                alt_section = alt_dataset[str(study_id)][f'section_{section_to_retrieve}']

            # Check verification output
            verification = alt_verification[str(study_id)]
            is_clinically_consistent = verification.is_clinically_consistent
            is_valid_perturbation = verification.is_valid_perturbation
            does_not_contain_required_info = verification.does_not_contain_required_info
            is_already_in_expected_form = verification.is_already_in_expected_form
            reasoning = verification.reasoning

            if is_clinically_consistent and is_valid_perturbation:
                # Also check if the alternative report is not a meaningful variation
                if select_meaningful_alts and (does_not_contain_required_info or is_already_in_expected_form):
                    alt_sections_dict[axis_name][variation] = None
                else:
                    alt_sections_dict[axis_name][variation] = alt_section
            else:
                print(f'No valid alternative found for "{axis_name}: {variation}".')
                alt_sections_dict[axis_name][variation] = None

    # Return the dictionary of alternative reports
    return dict(alt_sections_dict)
