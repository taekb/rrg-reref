from pathlib import Path
from typing import Optional, Union
from easydict import EasyDict as edict
from functools import partial
from PIL import Image

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers import BitsAndBytesConfig

LATERAL_VIEWS = {'lateral', 'lat', 'll', 'rl', 'left lateral'}

def _resolve_image_path(
    rel_path: str,
    image_dir: Path,
    dataset_name: str
) -> Path:
    '''Resolve a relative image path to its full filesystem path.'''

    rel_path = str(rel_path).lstrip('/')
    if dataset_name == 'mimic-cxr':
        return image_dir / 'files' / rel_path
    elif dataset_name == 'chexpertplus':
        return image_dir / rel_path.replace('.jpg', '.png')
    else:
        return image_dir / rel_path

def construct_prompt(
    example: Union[dict,edict],
    image_dir: Union[str,Path],
    processor: Optional[AutoProcessor] = None,
    use_key_image_only: Optional[bool] = False,
    dataset_name: Optional[str] = 'mimic-cxr',
    section_to_generate: Optional[str] = 'findings'
):
    '''Constructs a prompt for MAIRA-2 given an example.'''

    if processor is None:
        raise ValueError('Processor must be provided to construct the prompt.')

    if section_to_generate != 'findings':
        raise ValueError(f'MAIRA-2 does not support generating the "{section_to_generate}" section.')

    # Unpack image input
    example, image_dir = edict(example), Path(image_dir)

    if use_key_image_only:
        image_path = _resolve_image_path(example.key_image_path, image_dir, dataset_name)
        frontal_image = Image.open(image_path)
        lateral_image = None
    else:
        views = example.get('frontal_lateral', [])
        paired = (
            list(zip(example.image_path, views))
            if views
            else [(p, '') for p in example.image_path]
        )

        # Separate frontal and lateral images
        frontals = [(p, v) for p, v in paired if v.lower() not in LATERAL_VIEWS]
        laterals = [(p, v) for p, v in paired if v.lower() in LATERAL_VIEWS]

        key_path = str(example.key_image_path).lstrip('/')

        # Select the main frontal and lateral images
        if frontals:
            # Multiple frontal images available
            if len(frontals) > 1:
                main_frontal = next(
                    (p for p, _ in frontals if str(p).lstrip('/') == key_path),
                    frontals[0][0],
                )
            # Single frontal image available
            else:
                main_frontal = frontals[0][0]
            
            main_lateral = laterals[0][0] if laterals else None
        
        # No frontal images available
        else:
            main_frontal = laterals[0][0]
            main_lateral = laterals[1][0] if len(laterals) > 1 else None

        frontal_image = Image.open(_resolve_image_path(main_frontal, image_dir, dataset_name))
        lateral_image = (
            Image.open(_resolve_image_path(main_lateral, image_dir, dataset_name))
            if main_lateral is not None
            else None
        )

    # Include Indication, Technique, and Comparison sections if available
    indication = example['section_indication'] if example['section_indication'] is not None else None
    technique = example['section_technique'] if example['section_technique'] is not None else None
    comparison = example['section_comparison'] if example['section_comparison'] is not None else None

    inputs = processor.format_and_preprocess_reporting_input(
        current_frontal=frontal_image,
        current_lateral=lateral_image,
        prior_frontal=None,
        indication=indication,
        technique=technique,
        comparison=comparison,
        prior_report=None,
        return_tensors='pt',
        get_grounding=False
    )

    return inputs

def load_model(
    model_name: Optional[str] = 'microsoft/maira-2',
    load_pretrained: Optional[bool] = True,
    quantization_config: Optional[BitsAndBytesConfig] = None,
    torch_dtype: Optional[torch.dtype] = None,
    use_key_image_only: Optional[bool] = False
):
    if load_pretrained:
        model = AutoModelForCausalLM.from_pretrained(
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
        raise NotImplementedError('Loading non-pretrained MAIRA-2 is not implemented.')
    
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
    