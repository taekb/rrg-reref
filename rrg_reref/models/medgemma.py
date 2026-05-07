from pathlib import Path
from typing import Optional, Union
from easydict import EasyDict as edict
from functools import partial
import re
from PIL import Image

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers import BitsAndBytesConfig

# TODO: Add separate prompt construction for the MedGemma base model.
# NOTE: For base model evaluation, it looks like they generated the "Findings" and then "Findings + Impression" altogether.
def construct_prompt( 
    example: Union[dict,edict],
    image_dir: Union[str,Path],
    processor: Optional[AutoProcessor] = None,
    use_key_image_only: Optional[bool] = True,
    dataset_name: Optional[str] = 'rexval',
    section_to_generate: Optional[str] = 'findings', 
    tokenize: Optional[bool] = True
):
    '''Constructs a prompt for MedGemma given an example.'''

    if processor is None:
        raise ValueError('Processor must be provided to construct the prompt.')
    
    # Unpack image input
    example, image_dir = edict(example), Path(image_dir)
    if use_key_image_only:
        key_image_path = str(example.key_image_path).lstrip('/')
        if dataset_name in ['rexval', 'mimic-cxr']:
            image_path = image_dir / 'files' / key_image_path
        elif dataset_name == 'chexpertplus':
            image_path = image_dir / key_image_path.replace('.jpg', '.png') # FIXME
        else:
            image_path = image_dir / key_image_path
        
        image = Image.open(image_path)
    else:
        raise ValueError('MedGemma was only evaluated on single-image inputs.')

    # Default messages
    # NOTE: The default question here is NOT what was considered in the original paper or ReXrank.
    system_prompt = 'You are an expert radiologist.'

    if section_to_generate in ['findings', 'impression']:
        user_prompt = (
            f'Generate the {section_to_generate.upper()} section of the radiology report for this CXR image.\n\n'
        )
    elif section_to_generate == 'findings-and-impression':
        user_prompt = (
            f'Generate the FINDINGS and IMPRESSION sections of the radiology report for this CXR image.\n\n'
        )
    
    # Check if context is available
    context_available = (
        example['section_indication'] is not None or
        example['section_technique'] is not None or
        example['section_comparison'] is not None
    )
    if context_available:
        user_prompt += 'Below are the prefilled sections of the report that you can consider:\n\n"""\n'

    # Add Technique, Comparison, and Indication, Technique, and Comparison if available
    if example['section_indication'] is not None:
        user_prompt += f'INDICATION: {example["section_indication"]}\n\n'
    
    if example['section_technique'] is not None:
        user_prompt += f'TECHNIQUE: {example["section_technique"]}\n\n'

    if example['section_comparison'] is not None:
        user_prompt += f'COMPARISON: {example["section_comparison"]}\n\n'

    if context_available:
        user_prompt = user_prompt[:-1] + '"""\n\n' # Exclude the last newline and close the context block
    
    if section_to_generate in ['findings', 'impression']:
        user_prompt += (
            f'DO NOT generate anything other than the {section_to_generate.upper()} section of the report, '
            'and DO NOT repeat the details of the prefilled sections in any way.\n\n'
        )
    elif section_to_generate == 'findings-and-impression': 
        user_prompt += (
            'DO NOT generate anything other than the FINDINGS and IMPRESSION sections of the report, '
            'and DO NOT repeat the details of the prefilled sections in any way.\n\n'
        )

    # Add generation prefix
    #user_prompt += f'{section_to_generate.upper()}: '

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image", "image": image}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        tokenize=tokenize,
        return_dict=True if tokenize else False, 
        return_tensors="pt" if tokenize else None
    )

    return inputs

def load_model(
    model_name: Optional[str] = 'google/medgemma-4b-it',
    load_pretrained: bool = True,
    quantization_config: Optional[BitsAndBytesConfig] = None,
    torch_dtype: Optional[torch.dtype] = None,
    use_key_image_only: Optional[bool] = True
):
    if load_pretrained:
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(
            model_name, 
            trust_remote_code=True,
            use_fast=True
        )
        model = model.eval()
    else:
        raise NotImplementedError('Loading non-pretrained MedGemma is not implemented.')

    construct_prompt_ = partial(
        construct_prompt, 
        processor=processor, 
        use_key_image_only=use_key_image_only
    )

    # Define generation function
    def generate_fn(**kwargs):
        input_len = kwargs['input_ids'].shape[-1]
        outputs = model.generate(
            pad_token_id=processor.tokenizer.eos_token_id,
            **kwargs
        )
        output_ids = outputs[0][input_len:]
        output_text = processor.decode(output_ids, skip_special_tokens=True)

        return output_text

    return model, construct_prompt_, generate_fn
