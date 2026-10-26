import os
from pathlib import Path
import fire
import pandas as pd
from tqdm import tqdm
from typing import Optional
import json
from datetime import datetime

from rrg_reref.utils.mimic_cxr.parse import extract_sections_via_llm

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'

'''
    Builds the MIMIC-CXR JSON file expected by the pipelines (`data/mimic-cxr/<split>.json`),
    with sections extracted from each full report via `extract_sections_via_llm`.

'''

def prepare_mimic_cxr(
    split: Optional[str] = 'test',
    llm_name: Optional[str] = 'claude-sonnet-4-5',
    azure_openai: Optional[bool] = True
):
    '''Extracts the sections of all MIMIC-CXR reports in a given split and saves them as a JSON file.'''

    start_time = datetime.now()

    # MIMIC-CXR and MIMIC-CXR-JPG files
    mimic_cxr_dir = Path(os.getenv('MIMIC_CXR_DIR'))
    mimic_cxr_jpg_dir = Path(os.getenv('MIMIC_CXR_JPG_DIR'))
    mimic_cxr_study_list = pd.read_csv(mimic_cxr_dir / 'cxr-study-list.csv.gz') # Columns: [subject_id, study_id, path]
    mimic_cxr_metadata = pd.read_csv(mimic_cxr_jpg_dir / 'mimic-cxr-2.0.0-metadata.csv.gz')
    mimic_cxr_splits = pd.read_csv(mimic_cxr_jpg_dir / 'mimic-cxr-2.0.0-split.csv.gz')

    # Handle NaN values in ViewCodeSequence_CodeMeaning
    mimic_cxr_metadata['ViewCodeSequence_CodeMeaning'] = mimic_cxr_metadata['ViewCodeSequence_CodeMeaning'].apply(
        lambda x: 'unknown' if str(x) == 'nan' else x
    )

    # Fetch study IDs in the given split
    study_ids = mimic_cxr_splits.query('split == @split')['study_id'].unique().tolist()

    outdir = DATA_DIR / 'mimic-cxr'
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f'{split}.json'

    all_records = {}
    excluded_study_ids = []
    pbar = tqdm(study_ids, desc='Processing studies')
    for study_id in pbar:
        # Subject ID
        subject_id = str(mimic_cxr_study_list.query('`study_id` == @study_id')['subject_id'].values[0])

        # Fetch the full report for current study
        report_path = mimic_cxr_study_list.query('`study_id` == @study_id')['path'].values[0] # e.g., "files/p13/p13352405/s53780576.txt"
        report_path = mimic_cxr_dir / report_path
        with open(report_path, 'r') as fh:
            report = fh.read()

        # Extract the Indication, Comparison, Technique, Findings, and Impression sections
        sections = extract_sections_via_llm(report, llm_name=llm_name, azure_openai=azure_openai)

        # Exclude studies with entirely missing Findings and Impression sections
        if sections['findings'] is None and sections['impression'] is None:
            excluded_study_ids.append(study_id)
            continue

        # Image paths
        image_dir = Path(str(report_path).split('.txt')[0].split('files')[-1]) # e.g., "/p13/p13352405/s53780576"
        dicom_ids = mimic_cxr_metadata.query('`study_id` == @study_id')['dicom_id'].values
        image_paths = [str(image_dir / f'{dicom_id}.jpg') for dicom_id in dicom_ids]
        frontal_lateral = mimic_cxr_metadata.query('`study_id` == @study_id')['ViewCodeSequence_CodeMeaning'].values.tolist()
        try:
            key_image_dicom_id = mimic_cxr_metadata.query('`study_id` == @study_id and `ViewPosition` in ["AP", "PA"]')['dicom_id'].values[0]
        except IndexError:
            key_image_dicom_id = dicom_ids[0] # Fallback to first image if no AP/PA view found

        key_image_path = str(image_dir / f'{key_image_dicom_id}.jpg')

        # Combine everything into a record
        record = {
            'image_path': image_paths,
            'section_indication': sections['indication'],
            'section_comparison': sections['comparison'],
            'section_technique': sections['technique'],
            'section_findings': sections['findings'],
            'section_impression': sections['impression'],
            'frontal_lateral': frontal_lateral,
            'split': split,
            'subject_id': subject_id,
            'context': '',
            'key_image_path': key_image_path,
            'report': report
        }
        all_records[str(study_id)] = record

    print(f'Excluded {len(excluded_study_ids)} studies with missing Findings and Impression sections.')

    with open(outpath, 'w') as fh:
        json.dump(all_records, fh, indent=4)
        print(f'Saved {len(all_records)} records: {outpath}')

    print(f'Elapsed: {str(datetime.now() - start_time)}')

if __name__ == '__main__':
    fire.Fire(prepare_mimic_cxr)
