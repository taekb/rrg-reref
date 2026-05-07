import os
import os.path as osp
import re
import abc
import gc
import contextlib
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from datasets import Dataset
import warnings
warnings.filterwarnings('ignore')

from vllm import LLM, SamplingParams

import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer

from rrg_reref.utils.llm_api import prompt_llm

# From GREEN repository: https://github.com/Stanford-AIMI/GREEN/blob/main/green_score/utils.py
def strip_special_tokens(text):
    """Strips special tokens matching <|...|> and their associated bounding box content from report text."""
    text = re.sub(r'<\|box\|>.*?<\|/box\|>', '', text)
    text = re.sub(r'<\|[^|]*\|>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def make_prompt(text1, text2, max_len=300):
    """
    Creates a prompt for evaluating the accuracy of a candidate radiology report in comparison to a reference radiology report.

    Args:
        text1 (str): Reference radiology report.
        text2 (str): Candidate radiology report.

    Returns:
        str: Formatted prompt string.
    """
    
    text1 = strip_special_tokens(text1)
    text2 = strip_special_tokens(text2)
    text1 = " ".join(text1.split()[:max_len])
    text2 = " ".join(text2.split()[:max_len])
    prompt = f"Objective: Evaluate the accuracy of a candidate radiology report in comparison to a reference radiology report composed by expert radiologists.\n\n    Process Overview: You will be presented with:\n\n    1. The criteria for making a judgment.\n    2. The reference radiology report.\n    3. The candidate radiology report.\n    4. The desired format for your assessment.\n\n    1. Criteria for Judgment:\n\n    For each candidate report, determine:\n\n    The count of clinically significant errors.\n    The count of clinically insignificant errors.\n\n    Errors can fall into one of these categories:\n\n    a) False report of a finding in the candidate.\n    b) Missing a finding present in the reference.\n    c) Misidentification of a finding's anatomic location/position.\n    d) Misassessment of the severity of a finding.\n    e) Mentioning a comparison that isn't in the reference.\n    f) Omitting a comparison detailing a change from a prior study.\n    Note: Concentrate on the clinical findings rather than the report's writing style. Evaluate only the findings that appear in both reports.\n\n    2. Reference Report:\n    {text1}\n\n    3. Candidate Report:\n    {text2}\n\n    4. Reporting Your Assessment:\n\n    Follow this specific format for your output, even if no errors are found:\n    ```\n    [Explanation]:\n    <Explanation>\n\n    [Clinically Significant Errors]:\n    (a) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n    ....\n    (f) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n\n    [Clinically Insignificant Errors]:\n    (a) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n    ....\n    (f) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n\n    [Matched Findings]:\n    <The number of matched findings>. <Finding 1>; <Finding 2>; ...; <Finding n>\n    ```\n"
    
    return prompt

# From GREEN repository: https://github.com/Stanford-AIMI/GREEN/blob/main/green_score/utils.py
def clean_responses(response):
    '''Extracts the formatted output from the generated response.'''
    
    if "[Explanation]:" in response:
        if (
            "[Explanation]:\n    <Explanation>\n" or "[Explanation]:\n<Explanation>"
        ) in response:
            response = response.split("[Explanation]:")[1]
        else:
            response = response.split("[Explanation]:")[-1]
    
    return response

# Adapted from official GREEN implementation: https://github.com/Stanford-AIMI/GREEN/blob/main/green_score/green.py
# Works for OpenAI/Anthropic models and the fine-tuned model from GREEN.
class GREEN:
    def __init__(
        self,
        model_name: str,
        azure_openai: bool = True,
        use_vllm: bool = True,
        max_tokens: int = 4096,
        temperature: float = 0.,
        top_p: float = None,
        compute_summary_stats: bool = True
    ):
        super().__init__()
    
        self.model_name = model_name
        self.azure_openai = azure_openai
        self.use_vllm = use_vllm
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.compute_summary_stats = compute_summary_stats

        # Define categories and subcategories
        self.categories = [
            "Clinically Significant Errors",
            "Clinically Insignificant Errors",
            "Matched Findings",
        ]
        self.sub_categories = [
            "(a) False report of a finding in the candidate",
            "(b) Missing a finding present in the reference",
            "(c) Misidentification of a finding's anatomic location/position",
            "(d) Misassessment of the severity of a finding",
            "(e) Mentioning a comparison that isn't in the reference",
            "(f) Omitting a comparison detailing a change from a prior study",
        ]

        # Bookkeeping prompts, completions, GREEN scores & error counts
        self.prompts = None
        self.completions = None
        self.green_scores = None
        self.error_counts = None

        # Initialize the fine-tuned model with vLLM
        if self.model_name == 'StanfordAIMI/GREEN-radllama2-7b':
            if self.use_vllm:
                self.model = LLM(
                    self.model_name,
                    download_dir=os.getenv('HF_HUB_CACHE'),
                    tensor_parallel_size=1,
                    gpu_memory_utilization=0.5,
                    trust_remote_code=True
                )
                self.sampling_params = SamplingParams(
                    n=1,
                    temperature=(0 if self.temperature is None else self.temperature),
                    top_p=(1 if self.top_p is None else self.top_p),
                    max_tokens=self.max_tokens,
                    skip_special_tokens=True
                )

            else:
                self.accelerator = Accelerator()
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )
                self.model = self.model.to(self.accelerator.device)
                self.model = self.model.eval()
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    add_eos_token=True,
                    use_fast=True,
                    trust_remote_code=True,
                    padding_side='left'
                )
                chat_template = "{% for message in messages %}\n{% if message['from'] == 'human' %}\n{{ '<|user|>\n' + message['value'] + eos_token }}\n{% elif message['from'] == 'system' %}\n{{ '<|system|>\n' + message['value'] + eos_token }}\n{% elif message['from'] == 'gpt' %}\n{{ '<|assistant|>\n'  + message['value'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
                self.tokenizer.chat_template = chat_template
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.clean_up_tokenization_spaces = True
                assert self.tokenizer.padding_side == 'left'

            self.use_local_model = True
        else:
            self.use_local_model = False

    def close(self):
        '''Release the underlying vLLM engine (GPU memory, subprocess, NCCL).

        Safe to call multiple times; a no-op for API-backed or HF-backed models.
        '''
        if not getattr(self, 'use_local_model', False) or not getattr(self, 'use_vllm', False):
            return
        if getattr(self, 'model', None) is None:
            return

        llm = self.model
        self.model = None

        with contextlib.suppress(Exception):
            engine = getattr(llm, 'llm_engine', None)
            if engine is not None and hasattr(engine, 'shutdown'):
                engine.shutdown()

        del llm

        with contextlib.suppress(Exception):
            from vllm.distributed import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )
            destroy_model_parallel()
            destroy_distributed_environment()

        gc.collect()
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __call__(
        self,
        refs: List[str], # Reference reports
        hyps: List[str] # Candidate reports
    ):
        '''Performs GREEN evaluation on pairs of reference and candidate reports.'''
    
        start_time = datetime.now()

        # Load and preprocess dataset
        dataset = Dataset.from_dict({'reference': refs, 'prediction': hyps})
        self.dataset = self.process_data(dataset)

        # Run inference
        mean, std, green_scores, summary, results_df = self.infer()

        print(f'Elapsed: {str(datetime.now() - start_time)}')

        return mean, std, green_scores, summary, results_df

    def process_data(self, dataset: Dataset):
        '''Converts each (candidate, reference) pair into a prompt for the OpenAI model.'''

        def prompting(examples):
            return {
                'prompt': [
                    make_prompt(r, p)
                    for r, p in zip(examples['reference'], examples['prediction'])
                ]
            }
        
        print(f'Creating prompts...')
        dataset = dataset.map(prompting, batched=True)
        
        return dataset

    def infer(self):
        '''Queries the specified LLM for inference.'''

        print(f'Querying LLM: "{self.model_name}"\n')

        # vLLM-based local model inference
        if self.use_local_model:
            if self.use_vllm:
                prompts = self.dataset['prompt']
                outputs = self.model.generate(prompts, self.sampling_params)
                completions = [output.outputs[0].text for output in outputs]
            else:
                do_sample = (self.temperature is not None and self.temperature > 0)
                prompts = []
                completions = []
                pbar = tqdm(self.dataset.iter(batch_size=1), total=len(self.dataset), desc='Prompting')
                for batch in pbar:
                    prompts.extend(batch['prompt'])
                    input_ids = self.tokenizer(prompts[-1], return_tensors='pt').input_ids
                    input_ids = input_ids.to(self.accelerator.device)
                    input_len = input_ids.shape[-1]
                    print(prompts[-1])
                    print(input_ids)
                    print(torch.where(input_ids >= len(self.tokenizer)))
                    print(len(self.tokenizer) == self.model.get_input_embeddings().weight.shape[0])
                    with torch.inference_mode():
                        outputs = self.model.generate(
                            input_ids=input_ids,
                            do_sample=do_sample,
                            max_length=min(2048, self.max_tokens),
                            temperature=(self.temperature if do_sample else None),
                            top_p=(self.top_p if do_sample else None),
                            eos_token_id=self.tokenizer.eos_token_id,
                            use_cache=True
                        )
                        completions.append(
                            self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
                        )

        # API-based model inference
        else:
            prompts = []
            completions = []
            pbar = tqdm(self.dataset.iter(batch_size=1), total=len(self.dataset), desc='Prompting')
            for batch in pbar:
                prompts.extend(batch['prompt'])
                completion = prompt_llm(
                    model_name=self.model_name,
                    system_prompt='You are a helpful assistant.',
                    user_prompt=batch['prompt'][0],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    response_model=None,
                    azure_openai=self.azure_openai
                )
                completions.append(clean_responses(completion))

        self.prompts = prompts
        self.completions = completions

        print(f'Inference complete.')

        return self.process_results()

    def process_results(self):
        '''Extracts the error counts and computes the GREEN score.'''

        self.green_scores = [
            self.compute_green(response) for response in self.completions
        ]
        self.error_counts = pd.DataFrame(
            [self.compute_error_count(response) for response in self.completions],
            columns=self.sub_categories + ['Matched Findings']
        )

        results_df = pd.DataFrame(
            {
                'reference': self.dataset['reference'],
                'predictions': self.dataset['prediction'],
                'green_analysis': self.completions,
                'green_score': self.green_scores,
                **self.error_counts,
            }
        )
        results_df['total_errors'] = results_df[self.sub_categories].sum(axis=1)
        mean, std, summary = None, None, None

        if self.compute_summary_stats:
            mean, std, summary = self.compute_summary()

        return mean, std, self.green_scores, summary, results_df

    def compute_error_count(self, response):
        '''Counts the number of errors in each category.'''

        _, sig_errors = self.parse_error_counts(response, self.categories[0]) # "Clinically Significant"
        matched_findings, _ = self.parse_error_counts(response, self.categories[2]) # "Matched Findings"
        
        return sig_errors + [matched_findings] # Returns a list of error counts for each category

    def compute_green(self, response):
        '''Computes the GREEN score for a given response.'''

        sig_present, sig_errors = self.parse_error_counts(response, self.categories[0]) # "Clinically Significant"
        matched_findings, _ = self.parse_error_counts(response, self.categories[2]) # "Matched Findings"

        if matched_findings == 0:
            return 0

        if sig_present is None or matched_findings is None:
            return None

        return matched_findings / (matched_findings + sum(sig_errors))

    def parse_error_counts(self, text, category, for_reward=False):
        '''Extracts the error counts for each category.'''

        if category not in self.categories:
            raise ValueError(
                f'Category {category} is not a valid category. Please choose from {self.categories}.'
            )

        # Extract block of text corresponding to specified category
        # NOTE: Format solicited in prompt is replicated below.
        # [Clinically Significant Errors]:\n(a) <Error Type>: <The number of errors>. <Error 1>; ...; <Error n>
        # [Clinically Insignificant Errors]:\n(a) <Error Type>: <The number of errors>. <Error 1>; ...; <Error n>
        pattern = rf'\[{category}\]:\s*(.*?)(?:\n\s*\n|\Z)'
        category_text = re.search(pattern, text, re.DOTALL)

        sum_counts = 0
        sub_counts = [0 for i in range(len(self.sub_categories))]
        
        # Return 0 if the specified category is not present in the response
        if not category_text:
            if for_reward:
                return None, None
            return sum_counts, sub_counts
        
        # Return 0 if there are "No errors" found for the category
        if category_text.group(1).startswith('No'):
            return sum_counts, sub_counts

        # Check for matched findings
        # NOTE: Format solicited in prompt is replicated below.
        # [Matched Findings]:\n<The number of matched findings>. <Finding 1>; ...; <Finding n>
        if category == 'Matched Findings':
            counts = re.findall(r'^\b\d+\b(?=\.)', category_text.group(1))
            if len(counts) > 0:
                sum_counts = int(counts[0])
            
            return sum_counts, sub_counts
        
        # Check for errors in each subcategory, within the specified category
        else:
            # e.g., "(a) False report of a finding in the candidate" -> ["(a)", "False report of a finding in the candidate" -> "(a) "
            sub_categories = [s.split(" ", 1)[0] + " " for s in self.sub_categories]
            matches = sorted(re.findall(r"\([a-f]\) .*", category_text.group(1)))

            if len(matches) == 0:
                matches = sorted(re.findall(r"\([1-6]\) .*", category_text.group(1)))
                sub_categories = [
                    f"({i})" + " " for i in range(1, len(self.sub_categories) + 1)
                ]

            # Count the number of errors in each subcategory
            for position, sub_category in enumerate(sub_categories):
                for match in range(len(matches)): # Iterate through each matched subcategory
                    if matches[match].startswith(sub_category):
                        count = re.findall(r"(?<=: )\b\d+\b(?=\.)", matches[match]) # Extract the count
                        if len(count) > 0:
                            sub_counts[position] = int(count[0])
            
            return sum(sub_counts), sub_counts

    def compute_summary(self):
        '''
            Returns the proportion of (candidate, reference) pairs without clinically significant errors (per subcategory), 
            and the mean and standard deviation in GREEN scores across these pairs.
        
        '''
        
        print("Computing summary ...")
        #representative_sentences = self.get_representative_sentences(self.completions)
        accuracies = self.compute_accuracy(self.completions)
        mean = np.mean(self.green_scores)
        std = np.std(self.green_scores)

        summary = f'\n##### {self.model_name} #####\n'
        summary += f'GREEN Score: mean={mean:.4f}, std={std:.4f}\n'
        summary += f'Clinically Significant Errors:\n'
        for sub_category, accuracy in accuracies.items():
            summary += f'{sub_category}: {accuracy:.4f}\n'
        summary += '\n'

        return mean, std, summary

    def compute_accuracy(self, responses):
        '''
            For each subcategory, returns the proportion of (candidate, reference) pairs 
            where the number of significant errors is 0.
            
        '''
        
        counts = []
        for response in responses:
            _, sig_errors = self.parse_error_counts(response, self.categories[0]) # Clinically Significant
            counts.append(sig_errors)

        counts = np.array(counts)

        dict_acc = {}
        for i in range(len(self.sub_categories)):
            error_counts = counts[:,i] # Error counts for i-th subcategory
            accuracy = np.mean(error_counts == 0)
            dict_acc[self.sub_categories[i]] = accuracy

        return dict_acc