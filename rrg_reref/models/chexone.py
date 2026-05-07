import re
from pathlib import Path
from typing import Optional, Union
from easydict import EasyDict as edict
from functools import partial
from PIL import Image

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers import BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

def construct_prompt(
    example: Union[dict,edict],
    image_dir: Union[str,Path],
    processor: Optional[AutoProcessor] = None,
    use_key_image_only: Optional[bool] = False,
    dataset_name: Optional[str] = 'rexval',
    section_to_generate: Optional[str] = 'findings',
    reasoning_mode: Optional[bool] = True
):
    '''Constructs a prompt for CheXOne given an example.'''

    if processor is None:
        raise ValueError('Processor must be provided to construct the prompt.')

    example, image_dir = edict(example), Path(image_dir)
    if use_key_image_only:
        key_image_path = str(example.key_image_path).lstrip('/')
        if dataset_name in ['rexval', 'mimic-cxr']:
            image_path = image_dir / 'files' / key_image_path
        elif dataset_name == 'chexpertplus':
            image_path = image_dir / key_image_path.replace('.jpg', '.png') # FIXME
        else:
            image_path = image_dir / key_image_path

        images = [Image.open(image_path)]
    else:
        LATERAL_VIEWS = {'lateral', 'lat', 'll', 'rl', 'left lateral'}
        views = example.get('frontal_lateral', [])
        paired = list(zip(example.image_path, views)) if views else [(p, '') for p in example.image_path]
        key_path = str(example.key_image_path).lstrip('/')
        paired.sort(key=lambda pv: (
            0 if str(pv[0]).lstrip('/') == key_path else 1,
            1 if pv[1].lower() in LATERAL_VIEWS else 0,
        ))

        images = []
        for rel_path, _ in paired[:10]:
            rel_path = str(rel_path).lstrip('/')
            if dataset_name in ['rexval', 'mimic-cxr']:
                resolved = image_dir / 'files' / rel_path
            elif dataset_name == 'chexpertplus':
                resolved = image_dir / rel_path.replace('.jpg', '.png')
            else:
                resolved = image_dir / rel_path
            images.append(Image.open(resolved))

    # Check if the Indication section is available
    indication = example['section_indication'] if example['section_indication'] is not None else None

    # NOTE: Impression or Findings+Impression generation tasks are not explicitly discussed in the CheXOne paper.
    # As such, the prompts for either case are not provided by the authors.
    if section_to_generate == 'findings-and-impression':
        if indication is not None and indication != '':
            user_prompt = f'Given the indication: "{indication}", write example findings and impression sections for the CXR.'
        else:
            user_prompt = f'Write example findings and impression sections for the CXR.'
    else:
        if indication is not None and indication != '':
            user_prompt = f'Given the indication: "{indication}", write an example {section_to_generate} section for the CXR.'
        else:
            user_prompt = f'Write an example {section_to_generate} section for the CXR.'
    
    if reasoning_mode:
        user_prompt += f' Please reason step by step, and put your final answer within \\boxed{{}}.'

    content = [{'type': 'image', 'image': img} for img in images]
    content.append({'type': 'text', 'text': user_prompt})
    messages = [{'role': 'user', 'content': content}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors='pt',
    )

    return inputs

def load_model(
    model_name: Optional[str] = 'StanfordAIMI/CheXOne',
    load_pretrained: bool = True,
    quantization_config: Optional[BitsAndBytesConfig] = None,
    torch_dtype: Optional[torch.dtype] = None,
    use_key_image_only: Optional[bool] = False
):
    if load_pretrained:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            max_pixels=512 * 512,
        )
        model = model.eval()
    else:
        raise NotImplementedError('Loading non-pretrained CheXOne is not implemented.')

    construct_prompt_ = partial(
        construct_prompt,
        processor=processor,
        use_key_image_only=use_key_image_only
    )

    def generate_fn(**kwargs):
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        input_len = kwargs['input_ids'].shape[-1]
        outputs = model.generate(**kwargs)
        output_ids = outputs[0][input_len:]
        output_text = processor.batch_decode(
            [output_ids],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        boxed = re.search(r'\\boxed\{(.+)\}', output_text, re.DOTALL)
        if boxed:
            output_text = boxed.group(1).strip()

        return output_text

    return model, construct_prompt_, generate_fn
