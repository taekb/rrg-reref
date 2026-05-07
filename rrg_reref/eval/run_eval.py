import os
import os.path as osp
from pathlib import Path
import fire
from tqdm import tqdm
from typing import Optional, Union, List
import pandas as pd
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from radgraph import F1RadGraph
from RadEval import RadEval

from rrg_reref.utils.constants import (
    EVAL_DEFAULT_SEED, 
    MODEL_ALIAS_TO_MODEL_NAME,
    MODEL_TO_SECTIONS, 
    SUPPORTED_DATASETS
)
from rrg_reref.utils.data import load_dataset, exclude_empty_reports, gather_gt_sections, retrieve_alts_for_study
from rrg_reref.utils.radcliq import CompositeMetric
from rrg_reref.utils.green import GREEN
from rrg_reref.utils.crimson import CRIMSONScore
from rrg_reref.style.axes import get_style_axis_cls

PROJECT_ROOT = Path(os.getenv('PROJECT_ROOT', Path(__file__).parent.parent.parent))

def summarize_metrics(
    df: pd.DataFrame, 
    metrics: List[str],
    summary_title: Optional[str] = 'Average Report-Level Metrics'
):
    '''Summarizes the metrics for the given DataFrame.'''

    start_str = f'\n=============== {summary_title} ===============\n'
    print(start_str)
    for metric in metrics:
        if df[metric].dtype in ['object', 'str']:
            continue
        
        print(f'- {metric}: {df[metric].mean():.4f}')

    end_str = f'\n==============={"="*(len(summary_title)+2)}===============\n'
    print(end_str)

# Adapted from CXR-Report-Metric/CXRMetric/run_eval.py
def calc_metric(
    gt: pd.DataFrame, 
    pred: pd.DataFrame, 
    model_output_dir: Union[str,Path],
    output_path: Union[str,Path], 
    metrics_to_compute: Optional[List[str]] = [
        'BLEU-2',
        'BLEU-4',
        'ROUGE',
        'BERTScore',
        'CheXbert-F1',
        'RadGraph-F1',
        'RadGraph-XL-F1',
        'RadCliQ-v1',
        'RaTEScore',
        'GREEN',
        'CRIMSON'
    ],
    green_llm_judge_name: Optional[str] = 'gpt-4o',
    crimson_llm_judge_name: Optional[str] = 'gpt-4o',
    use_vllm: Optional[bool] = True,
    azure_openai: Optional[bool] = True,
    summary_title: Optional[str] = 'Average Report-Level Metrics',
    **kwargs
):
    '''Calculates the specified reference-based metrics for each (candidate, reference) report pair.'''
    
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    print('Loading the ground-truth and generated reports...')

    # Keep intersection of study IDs
    gt_study_ids = set(gt['study_id'])
    pred_study_ids = set(pred['study_id'])
    shared_study_ids = gt_study_ids.intersection(pred_study_ids)
    print(f'Number of shared study IDs: {len(shared_study_ids)}')

    gt = gt.loc[gt['study_id'].isin(shared_study_ids)].reset_index(drop=True)
    pred = pred.loc[pred['study_id'].isin(shared_study_ids)].reset_index(drop=True)
    refs = gt['report'].tolist()
    hyps = pred['report'].tolist()
    assert len(gt) == len(pred)
    assert ('report' in gt.columns) and ('report' in pred.columns)
    assert (gt['study_id'].equals(pred['study_id']))

    crimson_sub_categories = []
    for metric in metrics_to_compute:
        # Compute BLEU-2 scores (corpus- and sample-level scores)
        if metric == 'BLEU-2':
            print('\nComputing the BLEU-2 scores...')
            evaluator = RadEval(do_bleu=True, do_details=True)
            scores = evaluator(refs=refs, hyps=hyps)
            corpus_bleu_score = scores['bleu']['bleu_2']['mean_score']
            sample_bleu_scores = scores['bleu']['bleu_2']['sample_scores']
            pred['BLEU-2'] = corpus_bleu_score # Same corpus-level score broadcasted to all rows
            pred['BLEU-2-Sample'] = sample_bleu_scores

        # Compute BLEU-4 scores (corpus- and sample-level scores)
        elif metric == 'BLEU-4':
            print('\nComputing the BLEU-4 scores...')
            evaluator = RadEval(do_bleu=True, do_details=True)
            scores = evaluator(refs=refs, hyps=hyps)
            corpus_bleu_score = scores['bleu']['bleu_4']['mean_score']
            sample_bleu_scores = scores['bleu']['bleu_4']['sample_scores']
            pred['BLEU-4'] = corpus_bleu_score # Same corpus-level score broadcasted to all rows
            pred['BLEU-4-Sample'] = sample_bleu_scores

        # Compute ROUGE scores (sample-level scores)
        elif metric == 'ROUGE':
            print('\nComputing the ROUGE scores...')
            evaluator = RadEval(do_rouge=True, do_details=True)
            scores = evaluator(refs=refs, hyps=hyps)
            rouge1_scores = scores['rouge']['rouge1']['sample_scores']
            rouge2_scores = scores['rouge']['rouge2']['sample_scores']
            rougeL_scores = scores['rouge']['rougeL']['sample_scores']
            pred['ROUGE-1'] = rouge1_scores
            pred['ROUGE-2'] = rouge2_scores
            pred['ROUGE-L'] = rougeL_scores

        # Compute BERTScore scores (sample-level scores)
        elif metric == 'BERTScore':
            print('\nComputing the BERTScore scores...')
            evaluator = RadEval(do_bertscore=True, do_details=True)
            scores = evaluator(refs=refs, hyps=hyps)
            bertscore_scores = scores['bertscore']['sample_scores']
            pred['BERTScore'] = bertscore_scores

        # Compute CheXbert-F1 score (corpus-level score)
        elif metric.startswith('CheXbert-F1'):
            print(f'\nComputing the {metric} scores...')
            level = metric.split('-')[-1]
            evaluator = RadEval(do_chexbert=True, do_details=True)
            scores = evaluator(refs=refs, hyps=hyps)

            # CheXbert-F1-5 scores
            micro_f1_score = scores['chexbert']['chexbert-5_micro avg_f1-score']
            macro_f1_score = scores['chexbert']['chexbert-5_macro avg_f1-score']
            pred['CheXbert-F1-5-Micro'] = micro_f1_score
            pred['CheXbert-F1-5-Macro'] = macro_f1_score

            # CheXbert-F1-All scores
            micro_f1_score = scores['chexbert']['chexbert-all_micro avg_f1-score']
            macro_f1_score = scores['chexbert']['chexbert-all_macro avg_f1-score']
            pred['CheXbert-F1-All-Micro'] = micro_f1_score
            pred['CheXbert-F1-All-Macro'] = macro_f1_score

        # Compute RadGraph-F1 scores (sample-level scores)
        elif metric == 'RadGraph-F1':
            print('\nComputing the RadGraph-F1 scores...')
            evaluator = RadEval(do_radgraph=True, do_details=True)
            evaluator.radgraph_scorer = F1RadGraph(reward_level='all', model_type='radgraph')
            scores = evaluator(refs=refs, hyps=hyps)
            f1_scores = scores['radgraph']['sample_scores'][-1] # NOTE: We use the "complete" F1 scores
            pred['RadGraph-F1'] = f1_scores

        # Compute RadGraph-XL-F1 scores (sample-level scores)
        elif metric == 'RadGraph-XL-F1':
            print('\nComputing the RadGraph-XL-F1 scores...')
            evaluator = RadEval(do_radgraph=True, do_details=True)
            scores = evaluator(refs=refs, hyps=hyps)
            f1_scores = scores['radgraph']['sample_scores'][-1] # NOTE: We use the "complete" F1 scores
            pred['RadGraph-XL-F1'] = f1_scores
        
        # Compute RadCliQ-v1 scores (sample-level scores)
        # NOTE: We use a slightly modified version of RadCliQ-v1 in RadEval to allow batched processing
        elif metric == 'RadCliQ-v1':
            print('\nComputing the RadCliQ-v1 scores...')
            radcliq_scorer = CompositeMetric()
            _, detail_scores = radcliq_scorer.predict(refs, hyps)
            radcliq_scores = detail_scores.tolist()
            # evaluator = RadEval(do_radcliq=True, do_details=True)
            # scores = evaluator(refs=refs, hyps=hyps)
            # radcliq_scores = scores['radcliq-v1']['sample_scores']
            pred['RadCliQ-v1'] = radcliq_scores

        # Compute RaTEScore scores (sample-level scores)
        # NOTE: RaTEScore silently drops ref/hyp pairs with text < 3 characters.
        # We fill in 0 for these values, since they will almost always correspond to empty candidates,
        # as we have already filtered out any empty references before running the evaluation.
        elif metric == 'RaTEScore':
            print('\nComputing the RaTEScore scores...')
            
            # Identify indices where ref/hyp is >= 3 characters
            valid_idx = [
                i for i in range(len(refs))
                if len(refs[i].strip()) >= 3
                and len(hyps[i].strip()) >= 3
            ]
            if len(valid_idx) < len(refs):
                print(f'[WARNING] {len(refs)-len(valid_idx)} entries have text < 3 characters and will be scored as 0.')
            
            valid_refs = [refs[i] for i in valid_idx]
            valid_hyps = [hyps[i] for i in valid_idx]

            evaluator = RadEval(do_ratescore=True, do_details=True)
            scores = evaluator(refs=valid_refs, hyps=valid_hyps)
            valid_scores = scores['ratescore']['sample_scores']
            ratescore_scores = [0.] * len(refs) # Placeholder for all scores, filled in later.
            for i, idx in enumerate(valid_idx):
                ratescore_scores[idx] = valid_scores[i]
            
            pred['RaTEScore'] = ratescore_scores

        # Compute GREEN scores
        elif metric.startswith('GREEN'):
            print(f'\nComputing GREEN scores using "{green_llm_judge_name}"...')
            green_scorer = GREEN(model_name=green_llm_judge_name, azure_openai=azure_openai, use_vllm=use_vllm)
            try:
                mean, std, green_score_list, summary, result_df = green_scorer(refs=refs, hyps=hyps)
                pred[f'GREEN_{green_llm_judge_name}'] = green_score_list
                pred[f'GREEN_{green_llm_judge_name}--Total'] = result_df['total_errors'].tolist()
                pred[f'GREEN_{green_llm_judge_name}--Text'] = result_df['green_analysis'].tolist()
                pred[f'GREEN_{green_llm_judge_name}--Matched'] = result_df['Matched Findings'].tolist()
                for i, sub_category in enumerate(green_scorer.sub_categories):
                    pred[f'GREEN_{green_llm_judge_name}--{i+1}'] = result_df[sub_category].tolist()
            finally:
                green_scorer.close()

        # Compute CRIMSON scores
        elif metric.startswith('CRIMSON'):
            if crimson_llm_judge_name == 'CRIMSONScore/medgemma-4b-it-crimson':
                api = 'vllm'
            else:
                api = 'openai'

            print(f'\nComputing CRIMSON scores using "{crimson_llm_judge_name}"...')
            crimson_scorer = CRIMSONScore(api=api)
            try:
                if api == 'openai':
                    results = []
                    pbar = tqdm(range(len(refs)), desc='Computing CRIMSON scores')
                    for i in pbar:
                        result = crimson_scorer.evaluate(reference_findings=refs[i], predicted_findings=hyps[i])
                        results.append(result)

                elif api == 'vllm':
                    results = crimson_scorer.evaluate_batch(reference_findings_list=refs, predicted_findings_list=hyps)
            finally:
                # Release vLLM GPU memory before any fallback / subsequent metric runs.
                crimson_scorer.close()

            # Check if there are any cases where the output was invalid
            # Fallback to using GPT-5.2 for such cases
            if api == 'vllm':
                invalid_idxs = [i for i, result in enumerate(results) if result is None]
                if len(invalid_idxs) > 0:
                    print(f'[WARNING] {len(invalid_idxs)} cases failed to parse/score. Falling back to GPT-5.2...')
                    fallback_crimson_scorer = CRIMSONScore(api='openai')
                    pbar = tqdm(invalid_idxs, desc='Computing CRIMSON scores (fallback)')
                    for i in pbar: # TODO: Consider adding a try-except statement here to prevent errors downstream
                        result = fallback_crimson_scorer.evaluate(reference_findings=refs[i], predicted_findings=hyps[i])
                        results[i] = result

            crimson_scores = [result['crimson_score'] for result in results]
            pred[f'CRIMSON_{crimson_llm_judge_name}'] = crimson_scores
            pred[f'CRIMSON_{crimson_llm_judge_name}--Total'] = [sum(result['error_counts'].values()) for result in results]
            pred[f'CRIMSON_{crimson_llm_judge_name}--Text'] = [json.dumps(result['raw_evaluation']) for result in results]
            pred[f'CRIMSON_{crimson_llm_judge_name}--False_Findings'] = [result['error_counts']['false_findings'] for result in results]
            pred[f'CRIMSON_{crimson_llm_judge_name}--Missing_Findings'] = [result['error_counts']['missing_findings'] for result in results]
            pred[f'CRIMSON_{crimson_llm_judge_name}--Attribute_Errors'] = [result['error_counts']['attribute_errors'] for result in results]
            pred[f'CRIMSON_{crimson_llm_judge_name}--Location_Errors'] = [result['error_counts']['location_errors'] for result in results]
            pred[f'CRIMSON_{crimson_llm_judge_name}--Severity_Errors'] = [result['error_counts']['severity_errors'] for result in results]
            pred[f'CRIMSON_{crimson_llm_judge_name}--Descriptor_Errors'] = [result['error_counts']['descriptor_errors'] for result in results]
            pred[f'CRIMSON_{crimson_llm_judge_name}--Measurement_Errors'] = [result['error_counts']['measurement_errors'] for result in results]
            pred[f'CRIMSON_{crimson_llm_judge_name}--Certainty_Errors'] = [result['error_counts']['certainty_errors'] for result in results]
            pred[f'CRIMSON_{crimson_llm_judge_name}--Unspecific_Errors'] = [result['error_counts']['unspecific_errors'] for result in results]
            pred[f'CRIMSON_{crimson_llm_judge_name}--Overinterpretation_Errors'] = [result['error_counts']['overinterpretation_errors'] for result in results]
            pred[f'CRIMSON_{crimson_llm_judge_name}--Temporal_Errors'] = [result['error_counts']['temporal_errors'] for result in results]

        # Compute CheXprompt scores
        elif metric == 'CheXprompt':
            import openai
            from chexprompt import ReportEvaluator
            openai.api_type = 'azure'
            openai.api_base = os.environ['AZURE_OPENAI_ENDPOINT']
            openai.api_version = os.environ['AZURE_OPENAI_API_VERSION']
            openai.api_key = os.environ['AZURE_OPENAI_API_KEY']

            # Fixed to OpenAI models (GPT-4o) for now
            print('\nComputing CheXprompt scores using "gpt-4o"...')
            evaluator = ReportEvaluator(engine='gpt-4o')
            refs = gt['report'].tolist()
            hyps = pred['report'].tolist()
            outputs = evaluator.evaluate(refs, hyps)
            sig_errors = [output['clinically_significant'] for output in outputs]
            sig_errors_df = pd.DataFrame.from_records(sig_errors)
            sig_errors_df['total'] = sig_errors_df.sum(axis=1)
            pred['CheXprompt_gpt-4o--Total'] = sig_errors_df['total'].tolist()
            pred['CheXprompt_gpt-4o--1'] = sig_errors_df['false_positive_finding'].tolist()
            pred['CheXprompt_gpt-4o--2'] = sig_errors_df['omission_finding'].tolist()
            pred['CheXprompt_gpt-4o--3'] = sig_errors_df['incorrect_location'].tolist()
            pred['CheXprompt_gpt-4o--4'] = sig_errors_df['incorrect_severity'].tolist()
            pred['CheXprompt_gpt-4o--5'] = sig_errors_df['false_positive_comparison'].tolist()
            pred['CheXprompt_gpt-4o--6'] = sig_errors_df['omission_comparison'].tolist()
        
        else:
            raise ValueError(f'Unknown metric: {metric}')

    # Reformat the DataFrame
    pred['Reference Report'] = gt['report']
    pred = pred.rename(columns={'study_id': 'Study ID', 'report': 'Candidate Report'})
    cols_order = ['Study ID', 'Candidate Report', 'Reference Report']

    if 'BLEU-2' in metrics_to_compute:
        cols_order += ['BLEU-2', 'BLEU-2-Sample']

    if 'BLEU-4' in metrics_to_compute:
        cols_order += ['BLEU-4', 'BLEU-4-Sample']

    if 'ROUGE' in metrics_to_compute:
        cols_order += ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']

    if 'BERTScore' in metrics_to_compute:
        cols_order += ['BERTScore']

    if 'CheXbert-F1' in metrics_to_compute:
        cols_order += [
            'CheXbert-F1-5-Micro', 
            'CheXbert-F1-5-Macro', 
            'CheXbert-F1-All-Micro', 
            'CheXbert-F1-All-Macro'
        ]

    if 'RadGraph-F1' in metrics_to_compute:
        cols_order += ['RadGraph-F1']
        
    if 'RadGraph-XL-F1' in metrics_to_compute:
        cols_order += ['RadGraph-XL-F1']

    if 'RadCliQ-v1' in metrics_to_compute:
        cols_order += ['RadCliQ-v1']

    if 'RaTEScore' in metrics_to_compute:
        cols_order += ['RaTEScore']

    if 'GREEN' in metrics_to_compute:
        cols_order += [
            f'GREEN_{green_llm_judge_name}',
            f'GREEN_{green_llm_judge_name}--Total', 
            f'GREEN_{green_llm_judge_name}--Text', 
            f'GREEN_{green_llm_judge_name}--Matched', 
            f'GREEN_{green_llm_judge_name}--1', 
            f'GREEN_{green_llm_judge_name}--2', 
            f'GREEN_{green_llm_judge_name}--3', 
            f'GREEN_{green_llm_judge_name}--4', 
            f'GREEN_{green_llm_judge_name}--5', 
            f'GREEN_{green_llm_judge_name}--6'
        ]
    
    if 'CRIMSON' in metrics_to_compute:
        cols_order += [
            f'CRIMSON_{crimson_llm_judge_name}',
            f'CRIMSON_{crimson_llm_judge_name}--Total',
            f'CRIMSON_{crimson_llm_judge_name}--Text',
            f'CRIMSON_{crimson_llm_judge_name}--False_Findings',
            f'CRIMSON_{crimson_llm_judge_name}--Missing_Findings',
            f'CRIMSON_{crimson_llm_judge_name}--Attribute_Errors',
            f'CRIMSON_{crimson_llm_judge_name}--Location_Errors',
            f'CRIMSON_{crimson_llm_judge_name}--Severity_Errors',
            f'CRIMSON_{crimson_llm_judge_name}--Descriptor_Errors',
            f'CRIMSON_{crimson_llm_judge_name}--Measurement_Errors',
            f'CRIMSON_{crimson_llm_judge_name}--Certainty_Errors',
            f'CRIMSON_{crimson_llm_judge_name}--Unspecific_Errors',
            f'CRIMSON_{crimson_llm_judge_name}--Overinterpretation_Errors',
            f'CRIMSON_{crimson_llm_judge_name}--Temporal_Errors',
        ]
    
    if 'CheXprompt' in metrics_to_compute:
        cols_order += [
            'CheXprompt_gpt-4o--Total', 
            'CheXprompt_gpt-4o--1', 
            'CheXprompt-gpt-4o--2', 
            'CheXprompt_gpt-4o--3', 
            'CheXprompt_gpt-4o--4', 
            'CheXprompt_gpt-4o--5', 
            'CheXprompt_gpt-4o--6'
        ]
    
    results_df = pred[cols_order] # Reorder columns
    metric_cols = [c for c in cols_order if c not in ['Study ID', 'Candidate Report', 'Reference Report']]

    # If output file exists, override and append the new results
    if output_path.exists():
        orig_results_df = pd.read_csv(output_path)
        for metric in metric_cols:
            orig_results_df[metric] = results_df[metric].tolist()
        
        results_df = orig_results_df

    # Summarize the results
    summarize_metrics(results_df, metrics=metric_cols, summary_title=summary_title)

    # Save the results
    results_df.to_csv(output_path, index=False)

def _evaluate(
    gt: pd.DataFrame,
    pred: pd.DataFrame,
    model_output_dir: Union[str,Path],
    output_path: Union[str,Path],
    force_eval: Optional[bool] = False,
    metrics_to_compute: Optional[List[str]] = [
        'BLEU-2',
        'BLEU-4',
        'ROUGE',
        'BERTScore',
        'CheXbert-F1',
        'RadGraph-F1',
        'RadGraph-XL-F1',
        'RadCliQ-v1',
        'RaTEScore',
        'GREEN',
        'CRIMSON'
    ],
    green_llm_judge_name: Optional[str] = 'gpt-4o',
    crimson_llm_judge_name: Optional[str] = 'gpt-4o',
    use_vllm: Optional[bool] = True,
    azure_openai: Optional[bool] = True,
    summary_title: Optional[str] = 'Average Report-Level Metrics',
    **kwargs
):
    '''Performs evaluation against the ground-truth report.'''

    # Output CSV file path
    if output_path.exists():
        results_df = pd.read_csv(output_path)
        metric_col_names = []
        for m in metrics_to_compute:
            if 'ROUGE' in m:
                metric_col_names.append(f'{m}-1')
                metric_col_names.append(f'{m}-2')
                metric_col_names.append(f'{m}-L')

            elif 'CheXbert-F1' in m:
                metric_col_names.append(f'{m}-5-Micro')
                metric_col_names.append(f'{m}-5-Macro')
                metric_col_names.append(f'{m}-All-Micro')
                metric_col_names.append(f'{m}-All-Macro')
            
            elif 'GREEN' in m:
                metric_col_names.append(f'{m}_{green_llm_judge_name}')
            
            elif 'CRIMSON' in m:
                metric_col_names.append(f'{m}_{crimson_llm_judge_name}')

            else:
                metric_col_names.append(m)
        
        all_metrics_exist = all([m in results_df.columns for m in metric_col_names])
        if all_metrics_exist and not force_eval:
            print('\nEvaluation results already exist. Skipping...')
            metric_cols = [c for c in results_df.columns if c not in ['Study ID', 'Candidate Report', 'Reference Report']]
            summarize_metrics(results_df, metrics=metric_cols, summary_title=summary_title)
            return
        
        # Unless forced to re-evaluate, only evaluate the missing metrics
        elif not force_eval:
            missing_metrics = set()
            for m in metric_col_names:
                if m not in results_df.columns:
                    if 'ROUGE' in m:
                        missing_metrics.add('ROUGE')
                    elif 'CheXbert-F1' in m:
                        missing_metrics.add('CheXbert-F1')
                    elif 'GREEN' in m:
                        missing_metrics.add('GREEN')
                    elif 'CRIMSON' in m:
                        missing_metrics.add('CRIMSON')
                    else:
                        missing_metrics.add(m)

            missing_metrics = list(missing_metrics)
            print(f'Missing {len(missing_metrics)} metrics: {missing_metrics}')
            metrics_to_compute = missing_metrics

    # Run evaluation
    calc_metric(
        gt,
        pred,
        model_output_dir,
        output_path,
        metrics_to_compute=metrics_to_compute,
        green_llm_judge_name=green_llm_judge_name,
        crimson_llm_judge_name=crimson_llm_judge_name,
        use_vllm=use_vllm,
        summary_title=summary_title,
        azure_openai=azure_openai,
        **kwargs
    )
    print(f'\nEvaluation results saved to: {str(output_path)}\n')

def evaluate(
    model_name: str,
    dataset_name: str,
    output_dir: Union[str,Path],
    mimic_cxr_parser: Optional[str] = 'llm',
    sections_to_eval: Optional[List[str]] = None,
    split: Optional[str] = 'test',
    do_sample: Optional[bool] = False,
    temperature: Optional[float] = 0.7,
    top_p: Optional[float] = None,
    max_new_tokens: Optional[int] = 512,
    seed: Optional[int] = EVAL_DEFAULT_SEED,
    force_eval: Optional[bool] = False,
    metrics_to_compute: Optional[Union[List[str],str]] = [
        'BLEU-2',
        'BLEU-4',
        'ROUGE',
        'BERTScore',
        'CheXbert-F1',
        'RadGraph-F1',
        'RadGraph-XL-F1',
        'RadCliQ-v1',
        'RaTEScore',
        'GREEN',
        'CRIMSON'
    ],
    green_llm_judge_name: Optional[str] = 'claude-sonnet-4-5',
    crimson_llm_judge_name: Optional[str] = 'claude-sonnet-4-5',
    use_vllm: Optional[bool] = True,
    azure_openai: Optional[bool] = True,
    include_alt_gt_reports: Optional[bool] = False,
    alt_llm_name: Optional[str] = 'o3',
    use_controlled_alts: Optional[bool] = True,
    alt_sample_idx: Optional[int] = 1,
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
    verbose: Optional[bool] = False,
    **kwargs
):
    '''Main evaluation function.'''

    start_time = datetime.now()

    # Check if specified dataset is supported
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f'Unknown dataset: {dataset_name}')

    # Check if specified model is supported
    if model_name not in list(MODEL_TO_SECTIONS.keys()):
        # Check if the alias was used
        if model_name in list(MODEL_ALIAS_TO_MODEL_NAME.keys()):
            model_name = MODEL_ALIAS_TO_MODEL_NAME[model_name]
        else:
            raise ValueError(f'Unknown model: {model_name}')

    # Parse metrics list
    if isinstance(metrics_to_compute, str):
        metrics_to_compute = metrics_to_compute.split(',')

    # Iterate over all possible sections supported by the model
    if sections_to_eval is None:
        sections_to_eval = list(MODEL_TO_SECTIONS[model_name])

    # Add Findings + Impression (if applicable)
    if 'findings-and-impression' not in sections_to_eval and ('findings' in sections_to_eval and 'impression' in sections_to_eval):
        sections_to_eval.append('findings-and-impression')

    # By default, we exclude evaluation on just the Impression section
    if 'impression' in sections_to_eval:
        sections_to_eval.remove('impression')

    for section in sections_to_eval:
        print(f'Evaluating the "{section.upper()}" sections generated by "{model_name}"...')

        # Check if model inference has already been run
        if 'oracle' in model_name:
            if dataset_name != 'rexval':
                raise ValueError('Metric-oracles are only available for the ReXVal dataset.')

            model_output_root = Path(output_dir) / dataset_name / model_name # No need to include mimic_cxr_parser for ReXVal metric-oracles
            pred_reports_path = model_output_root / f'{split}_candidate-{section}.csv'

            if not pred_reports_path.exists():
                print(f'Candidate {section} sections do not exist. Extract the ReXVal metric-oracles first.')
                continue
        else:
            if dataset_name in ['mimic-cxr', 'rexval']:
                model_output_root = Path(output_dir) / dataset_name / mimic_cxr_parser / model_name.replace('/', '--')
            else:
                model_output_root = Path(output_dir) / dataset_name / model_name.replace('/', '--')

            if do_sample:
                pred_reports_path = model_output_root / f'{split}_T={temperature},p={top_p},t={max_new_tokens},seed={seed}_predicted-{section}.csv'
            else:
                pred_reports_path = model_output_root / f'{split}_t={max_new_tokens}_predicted-{section}_greedy.csv'

            if not pred_reports_path.exists():
                section_not_supported = section not in MODEL_TO_SECTIONS[model_name]
                if section_not_supported:
                    print(f'Candidate {section} sections are not supported for "{model_name}". Skipping.')
                else:
                    print(f'Candidate {section} sections do not exist for "{model_name}". Please run inference first.')
                
                continue

        pred_reports_df = pd.read_csv(pred_reports_path)
        pred_reports_df['study_id'] = pred_reports_df['study_id'].astype(str) # NOTE: Should cast before sorting, as some IDs are integers
        pred_reports_df = pred_reports_df.sort_values(by=['study_id']).reset_index(drop=True)
        pred_reports_df['report'] = pred_reports_df['report'].fillna('') # Replace NaNs (if present) with empty strings

        # Fetch ground-truth reports to evaluate against
        # NOTE: `gather_gt_sections` automatically exludes examples where the ground-truth report is empty
        dataset = load_dataset(dataset_name, split=split, mimic_cxr_parser=mimic_cxr_parser)
        gt_reports_df = gather_gt_sections(dataset, section, verbose=verbose) # DataFrame with columns ['study_id', 'report'], sorted by study_id
        study_ids = gt_reports_df['study_id'].values.tolist() # NOTE: Should use these, as studies with missing sections are filtered out
        
        # Evaluate against ground-truth report
        print('\nEvaluating against the ground-truth reports...')
        gt_results_dir = Path(model_output_root) / 'ground-truth'
        gt_results_dir.mkdir(parents=True, exist_ok=True)
        gt_results_path = gt_results_dir / f'{split}_{section}_results.csv'
        _evaluate(
            gt_reports_df,
            pred_reports_df,
            gt_results_dir,
            gt_results_path,
            force_eval=force_eval,
            metrics_to_compute=metrics_to_compute,
            green_llm_judge_name=green_llm_judge_name,
            crimson_llm_judge_name=crimson_llm_judge_name,
            use_vllm=use_vllm,
            azure_openai=azure_openai,
            summary_title='Ground-Truth',
            **kwargs
        )
        
        # Evaluate against alternative ground-truth reports
        if include_alt_gt_reports:
            # Use controlled style variations
            if use_controlled_alts:
                # Evaluate along each style axis
                for axis_name in axis_names:
                    axis_obj = get_style_axis_cls(axis_name)()
                    if axis_obj.axis_type == 'absolute':
                        variations = axis_obj.variations
                    else:
                        variations = ['alternative']
                    
                    # Pre-load all (dataset, verification) pairs once per variation,
                    # instead of re-loading the same JSON files for every study_id.
                    variation_data = {}
                    for variation in variations:
                        alt_dataset, alt_verification = load_dataset(
                            dataset_name=dataset_name,
                            mimic_cxr_parser=mimic_cxr_parser,
                            split=split,
                            load_alts=True,
                            axis_name=axis_name,
                            variation=variation,
                            llm_name=alt_llm_name,
                            sample_idx=alt_sample_idx,
                            load_verification=True
                        )
                        variation_data[variation] = (alt_dataset, alt_verification)

                    # Gather alternative reports for each axis variation across all studies
                    variation_to_alt_reports = {
                        variation: dict(study_id=[], report=[]) for variation in variations
                    }
                    pbar = tqdm(study_ids, desc=f'Gathering alternative reports for "{axis_name}"')
                    for study_id in pbar:
                        sid = str(study_id)
                        for variation in variations:
                            alt_dataset, alt_verification = variation_data[variation]

                            if section == 'findings-and-impression':
                                findings = alt_dataset[sid]['section_findings']
                                impression = alt_dataset[sid]['section_impression']
                                findings_text = '' if pd.isna(findings) else findings
                                impression_text = '' if pd.isna(impression) else impression
                                alt_section = (
                                    'FINDINGS:\n\n' + findings_text + '\n\n' +
                                    'IMPRESSION:\n\n' + impression_text
                                )
                            else:
                                alt_section = alt_dataset[sid][f'section_{section}']

                            verification = alt_verification[sid]
                            if not (verification.is_clinically_consistent and verification.is_valid_perturbation):
                                if verbose:
                                    print(f'No valid alternative found for "{axis_name}: {variation}" (study {study_id}).')
                                continue
                            if verification.does_not_contain_required_info or verification.is_already_in_expected_form:
                                continue

                            variation_to_alt_reports[variation]['study_id'].append(study_id)
                            variation_to_alt_reports[variation]['report'].append(alt_section)

                    # Run evaluation on each axis variation
                    for variation in variations:
                        # Set up results directory
                        alt_results_dir = Path(model_output_root) / 'controlled-alts' / axis_name / variation / alt_llm_name
                        alt_results_dir.mkdir(parents=True, exist_ok=True)
                        alt_results_path = alt_results_dir / f'{split}_{section}_results_{alt_sample_idx}.csv'

                        # Prepare alternative reports in a DataFrame
                        alt_reports_df = pd.DataFrame(variation_to_alt_reports[variation])

                        if alt_reports_df.empty: # No alternative reports found for this variation
                            print(f'No alternative reference reports found for "{variation}". Skipping...')
                            with open(alt_results_dir / 'no_alts_found.txt', 'w') as fh:
                                fh.write(f'No alternative reference reports found for "{variation}".')
                            continue

                        alt_reports_df = exclude_empty_reports(alt_reports_df, verbose=verbose) # Filter down to studies with non-empty alternative reports
                        alt_reports_df = alt_reports_df.sort_values(by=['study_id']).reset_index(drop=True)
                    
                        # Run evaluation
                        summary_title = f'Alternative ({axis_name.replace("_", " ").title()} - {variation.replace("_", " ").title()})'
                        _evaluate(
                            alt_reports_df,
                            pred_reports_df,
                            alt_results_dir,
                            alt_results_path,
                            force_eval=force_eval,
                            metrics_to_compute=metrics_to_compute,
                            green_llm_judge_name=green_llm_judge_name,
                            crimson_llm_judge_name=crimson_llm_judge_name,
                            use_vllm=use_vllm,
                            azure_openai=azure_openai,
                            summary_title=summary_title,
                            **kwargs
                        )
            
            # Use random style variations
            else:
                alt_results_dir = Path(model_output_root) / 'random-alts'
                alt_results_dir.mkdir(parents=True, exist_ok=True)
                alt_results_path = alt_results_dir / f'{split}_{section}_results_{alt_sample_idx}.csv'

                # Gather alternative reports for each study
                # FIXME: alt_reports will be a list of lists. DataFrame construction won't work.
                # TODO: Should decide whether to update this for handling multiple samples per study.
                sample_indices = [alt_sample_idx] if isinstance(alt_sample_idx, int) else alt_sample_idx
                sample_data = {}
                for idx in sample_indices:
                    alt_dataset, alt_verification = load_dataset(
                        dataset_name=dataset_name,
                        mimic_cxr_parser=mimic_cxr_parser,
                        split=split,
                        load_alts=True,
                        llm_name=alt_llm_name,
                        sample_idx=idx,
                        load_verification=True
                    )
                    sample_data[idx] = (alt_dataset, alt_verification)

                alt_reports = []
                pbar = tqdm(study_ids, desc='Gathering randomly sampled alternative reports')
                for study_id in pbar:
                    sid = str(study_id)
                    study_alts = []
                    for idx in sample_indices:
                        alt_dataset, alt_verification = sample_data[idx]
                        alt_section = alt_dataset[sid][f'section_{section}']
                        verification = alt_verification[sid]
                        if verification.is_clinically_consistent:
                            study_alts.append(alt_section)
                        else:
                            if verbose:
                                print(f'No valid alternative found for sample index {idx} (study {study_id}).')
                            study_alts.append(None)
                    alt_reports.append(study_alts)
                
                alt_reports_df = pd.DataFrame(dict(study_id=study_ids, report=alt_reports))
                alt_reports_df = exclude_empty_reports(alt_reports_df, verbose=verbose)
                alt_reports_df = alt_reports_df.sort_values(by=['study_id']).reset_index(drop=True)

                # Run evaluation
                _evaluate(
                    alt_reports_df,
                    pred_reports_df,
                    alt_results_dir,
                    alt_results_path,
                    force_eval=force_eval,
                    metrics_to_compute=metrics_to_compute,
                    green_llm_judge_name=green_llm_judge_name,
                    crimson_llm_judge_name=crimson_llm_judge_name,
                    use_vllm=use_vllm,
                    azure_openai=azure_openai,
                    summary_title=f'Alternative (Random)',
                    **kwargs
                )
    
    print(f'Elapsed: {str(datetime.now() - start_time)}')

if __name__ == '__main__':
    fire.Fire(evaluate)