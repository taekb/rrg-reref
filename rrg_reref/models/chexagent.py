from pathlib import Path
from typing import Optional, Union
from easydict import EasyDict as edict
from functools import partial
from PIL import Image

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

def construct_prompt(
    example: Union[dict,edict],
    image_dir: Union[str,Path],
    tokenizer: Optional[AutoTokenizer] = None,
    use_key_image_only: Optional[bool] = False,
    dataset_name: Optional[str] = 'rexval',
    section_to_generate: Optional[str] = 'findings',
    tokenize: Optional[bool] = True
):
    '''Constructs a prompt for CheXagent given an example.'''

    if tokenizer is None:
        raise ValueError('Tokenizer must be provided to construct the prompt.')

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

        image_paths = [str(image_path)]
    else:
        # CheXagent-2-3b accepts multi-image inputs via from_list_format. CheXbench
        # caps at 2 images per study (evaluation_chexbench/.../run_findings_generation.py:35),
        # so match that. Prioritise the key frontal image, then other frontals, then laterals.
        LATERAL_VIEWS = {'lateral', 'lat', 'll', 'rl', 'left lateral'}
        views = example.get('frontal_lateral', [])
        paired = list(zip(example.image_path, views)) if views else [(p, '') for p in example.image_path]
        key_path = str(example.key_image_path).lstrip('/')
        paired.sort(key=lambda pv: (
            0 if str(pv[0]).lstrip('/') == key_path else 1,
            1 if pv[1].lower() in LATERAL_VIEWS else 0,
        ))

        image_paths = []
        for rel_path, _ in paired[:2]: # Max 2 images per study
            rel_path = str(rel_path).lstrip('/')
            if dataset_name in ['rexval', 'mimic-cxr']:
                resolved = image_dir / 'files' / rel_path
            elif dataset_name == 'chexpertplus':
                resolved = image_dir / rel_path.replace('.jpg', '.png')
            else:
                resolved = image_dir / rel_path
            
            image_paths.append(str(resolved))

    # Default messages
    system_prompt = 'You are a helpful assistant.'

    # Prompts follow the CheXinstruct training templates (data_chexinstruct/.../templates.py):
    # the "with Indication" form matches the authors' inference API (model_chexagent/chexagent.py)
    # and CheXbench evaluation; the no-indication form falls back to a training-valid template.
    indication = example.get('section_indication')
    has_indication = indication is not None and indication != ''
    if section_to_generate == 'findings':
        if has_indication:
            user_prompt = f'Given the indication: "{indication}", write a structured findings section for the CXR.'
        else:
            user_prompt = 'Write a structured findings section for the CXR.'
    
    elif section_to_generate == 'impression':
        if has_indication:
            user_prompt = f'Given the indication: "{indication}", write a structured impression section for the CXR.'
        else:
            user_prompt = 'Write a structured impression section for the CXR.'
    
    elif section_to_generate == 'findings-and-impression':
        raise NotImplementedError(
            'CheXagent-2-3b has no documented combined findings-and-impression prompt. '
            'The authors use a two-step pipeline (findings generation, then findings '
            'summarization). Run findings and impression separately instead.'
        )

    # Construct prompt
    query = tokenizer.from_list_format(
        [*[{'image': p} for p in image_paths], {'text': user_prompt}]
    )
    conv = [
        {'from': 'system', 'value': system_prompt},
        {'from': 'human', 'value': query}
    ]
    
    inputs = tokenizer.apply_chat_template(
        conv, 
        add_generation_prompt=True, 
        tokenize=tokenize,
        return_dict=True if tokenize else False,
        return_tensors='pt' if tokenize else None
    )

    return inputs

def load_model(
    model_name: Optional[str] = 'StanfordAIMI/CheXagent-2-3b',
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
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
    else:
        raise NotImplementedError('Loading non-pretrained CheXagent is not implemented.')

    construct_prompt_ = partial(
        construct_prompt,
        tokenizer=tokenizer,
        use_key_image_only=use_key_image_only
    )

    # Define generation function
    def generate_fn(**kwargs):
        input_len = kwargs['input_ids'].shape[-1]
        outputs = model.generate(
            #pad_token_id=tokenizer.eos_token_id,
            **kwargs
        )
        output_ids = outputs[0][input_len:]
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

        return output_text

    return model, construct_prompt_, generate_fn