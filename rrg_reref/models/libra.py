from pathlib import Path
from typing import Optional, Union
from easydict import EasyDict as edict
from functools import partial
from PIL import Image

import torch
from transformers import BitsAndBytesConfig

from libra.constants import (
    IMAGE_TOKEN_INDEX, 
    DEFAULT_IMAGE_TOKEN, 
    DEFAULT_IM_START_TOKEN, 
    DEFAULT_IM_END_TOKEN
)
from libra.conversation import conv_templates, SeparatorStyle
from libra.model.builder import load_pretrained_model
from libra.utils import disable_torch_init
from libra.mm_utils import (
    tokenizer_image_token, 
    process_images, 
    get_model_name_from_path, 
    KeywordsStoppingCriteria
)


SECTION_PROMPTS = {
    'findings': 'Provide a detailed description of the findings in the radiology image.',
    'impression': 'Provide a detailed description of the impression in the radiology image.',
}


def _concatenate_images(images):
    '''Horizontal stitching with a 10px black separator, per the Med-CXRGen (RRG-BioNLP-ACL2024) reference implementation.'''

    total_width = sum(img.width for img in images) + 10 * (len(images) - 1)
    height = max(img.height for img in images)
    new_img = Image.new('RGB', (total_width, height), (0, 0, 0))
    current_width = 0
    for img in images:
        new_img.paste(img, (current_width, 0))
        current_width += img.width + 10
    
    return new_img

def _get_conv_mode(model_name: str) -> str:
    '''Determine conversation mode from model name, matching Libra's auto-detection logic.'''
    
    name = model_name.lower()
    if 'libra-v1.0-3b' in name:
        return 'libra_llama_3'
    elif 'cxrgen' in name:
        return 'libra_v0'
    else:
        return 'libra_v1'


def construct_prompt(
    example: Union[dict, edict],
    image_dir: Union[str, Path],
    tokenizer=None,
    image_processor=None,
    model_config=None,
    conv_mode: str = 'libra_v1',
    use_key_image_only: bool = True,
    dataset_name: str = 'mimic-cxr',
    section_to_generate: str = 'findings',
):
    '''Constructs inputs for a Libra-based model given a dataset example.'''

    example, image_dir = edict(example), Path(image_dir)

    if use_key_image_only:
        key_image_path = str(example.key_image_path).lstrip('/')
        if dataset_name == 'mimic-cxr':
            image_path = image_dir / 'files' / key_image_path
        elif dataset_name == 'chexpertplus':
            image_path = image_dir / key_image_path.replace('.jpg', '.png')
        else:
            image_path = image_dir / key_image_path
        image = Image.open(str(image_path)).convert('RGB')
    
    # Multi-image inputs for Med-CXRGen models (see Section 4.1 of paper)
    elif conv_mode == 'libra_v0':
        LATERAL_VIEWS = {'lateral', 'lat', 'll', 'rl', 'left lateral'}
        views = example.get('frontal_lateral', [])
        paired = list(zip(example.image_path, views)) if views else [(p, '') for p in example.image_path]
        key_path = str(example.key_image_path).lstrip('/')
        paired.sort(key=lambda pv: (
            0 if str(pv[0]).lstrip('/') == key_path else 1, # Prioritize key image first
            1 if pv[1].lower() in LATERAL_VIEWS else 0, # Then prioritize frontals over laterals
        ))
        
        pil_images = []
        for rel_path, _ in paired[:4]: # Stitch images horizontally up to 4 images
            rel_path = str(rel_path).lstrip('/')
            if dataset_name == 'mimic-cxr':
                resolved = image_dir / 'files' / rel_path
            elif dataset_name == 'chexpertplus':
                resolved = image_dir / rel_path.replace('.jpg', '.png')
            else:
                resolved = image_dir / rel_path
            
            pil_images.append(Image.open(str(resolved)).convert('RGB'))
        
        image = _concatenate_images(pil_images) if len(pil_images) > 1 else pil_images[0]
    
    # Libra model only supports Findings generation
    else:
        raise ValueError('Libra was not trained or evaluated on multi-view input settings.')

    # Libra expects [2, num_images_per_timepoint, C, H, W] (current + prior).
    # With no prior available, duplicate the current image as a dummy.
    image_tensor = process_images([image], image_processor, model_config)[0]  # [C, H, W]
    image_tensor = torch.stack([
        image_tensor.unsqueeze(0),  # current  [1, C, H, W]
        image_tensor.unsqueeze(0),  # prior    [1, C, H, W]
    ])  # [2, 1, C, H, W]

    query = SECTION_PROMPTS[section_to_generate]

    # Append clinical context when available
    context_parts = []
    if example.get('section_indication') is not None:
        context_parts.append(f'INDICATION: {example["section_indication"]}')
    if example.get('section_technique') is not None:
        context_parts.append(f'TECHNIQUE: {example["section_technique"]}')
    if example.get('section_comparison') is not None:
        context_parts.append(f'COMPARISON: {example["section_comparison"]}')
    if context_parts:
        query += ' Following clinical context: ' + ' '.join(context_parts)

    # Prepend image token
    if getattr(model_config, 'mm_use_im_start_end', False):
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + query
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + query

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

    return dict(
        input_ids=input_ids,
        images=image_tensor,
        attention_mask=attention_mask,
    )


def load_model(
    model_name: str = 'X-iZhang/libra-v1.0-7b',
    load_pretrained: bool = True,
    quantization_config: Optional[BitsAndBytesConfig] = None,
    torch_dtype: Optional[torch.dtype] = None,
    use_key_image_only: bool = False,
):
    if not load_pretrained:
        raise NotImplementedError('Loading non-pretrained Libra models is not implemented.')

    # Enforce use_key_image_only=True for Libra
    if not use_key_image_only and 'libra-v1.0-7b' in model_name.lower():
        use_key_image_only = True

    disable_torch_init()
    model_name_short = get_model_name_from_path(model_name)

    # device_map=None avoids transformers 4.51+ auto-promoting device_map="auto" to
    # tp_plan="auto" under torchrun/accelerate (which sets WORLD_SIZE), which would
    # fail on torch < 2.5. The model loads on CPU; run_inference.py moves it to GPU.
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_name,
        model_base=None,
        model_name=model_name_short,
        device_map=None,
        device="cpu",
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = model.eval()

    conv_mode = _get_conv_mode(model_name_short)

    # Determine the stop string from the conversation template
    conv = conv_templates[conv_mode].copy()
    if conv.sep_style in {SeparatorStyle.TWO, SeparatorStyle.LLAMA_3, SeparatorStyle.MISTRAL}:
        stop_str = conv.sep2
    else:
        stop_str = conv.sep

    construct_prompt_ = partial(
        construct_prompt,
        tokenizer=tokenizer,
        image_processor=image_processor,
        model_config=model.config,
        conv_mode=conv_mode,
        use_key_image_only=use_key_image_only,
    )

    def generate_fn(**kwargs):
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        input_ids = kwargs.pop('input_ids')
        images = kwargs.pop('images')
        attention_mask = kwargs.pop('attention_mask', None)

        stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
        input_len = input_ids.shape[-1]

        output_ids = model.generate(
            input_ids=input_ids,
            images=images,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.pad_token_id,
            stopping_criteria=[stopping_criteria],
            **kwargs,
        )

        output_text = tokenizer.batch_decode(
            output_ids[:, input_len:], skip_special_tokens=True
        )[0].strip()
        if output_text.endswith(stop_str):
            output_text = output_text[:-len(stop_str)].strip()

        return output_text

    return model, construct_prompt_, generate_fn
