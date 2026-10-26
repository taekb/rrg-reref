from pathlib import Path
from typing import Optional, Union
from easydict import EasyDict as edict
from functools import partial
from typing import Any
from PIL import Image

import torch
from transformers import BitsAndBytesConfig
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria

def construct_prompt(
    example: Union[dict,edict],
    image_dir: Union[str,Path],
    tokenizer: Optional[Any] = None,
    image_processor: Optional[Any] = None,
    use_key_image_only: Optional[bool] = True,
    dataset_name: Optional[str] = 'mimic-cxr',
    section_to_generate: Optional[str] = 'findings',
    tokenize: Optional[bool] = True
):
    '''Constructs a prompt for LLaVA-Rad given an example.'''

    if tokenizer is None:
        raise ValueError('Tokenizer must be provided to construct the prompt.')

    if section_to_generate != 'findings':
        raise ValueError(f'LLaVA-Rad does not support generating the "{section_to_generate}" section.')

    # Unpack image input
    example, image_dir = edict(example), Path(image_dir)
    if use_key_image_only:
        key_image_path = str(example.key_image_path).lstrip('/')
        if dataset_name == 'mimic-cxr':
            image_path = image_dir / 'files' / key_image_path
        elif dataset_name == 'chexpertplus':
            image_path = image_dir / key_image_path.replace('.jpg', '.png') # FIXME
        else:
            image_path = image_dir / key_image_path

        image_path = str(image_path)
    else:
        raise ValueError('LLaVA-Rad was only evaluated on single-image inputs.')

    # Construct prompt
    user_prompt = '<image>\nDescribe the findings of the chest x-ray.\n'
    conv = conv_templates['v1'].copy()
    conv.append_message(conv.roles[0], user_prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image = Image.open(image_path).convert('RGB')
    image_tensor = image_processor.preprocess(
        image, return_tensors='pt'
    )['pixel_values'][0].unsqueeze(0)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)

    inputs = dict(input_ids=input_ids, images=image_tensor)

    return inputs

def load_model(
    model_name: Optional[str] = 'microsoft/llava-rad',
    load_pretrained: Optional[bool] = True,
    quantization_config: Optional[BitsAndBytesConfig] = None, # FIXME: Not currently used
    torch_dtype: Optional[torch.dtype] = None,
    use_key_image_only: Optional[bool] = True
):
    if load_pretrained:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_name,
            'lmsys/vicuna-7b-v1.5',
            'llavarad'
        )
    else:
        raise NotImplementedError('Loading non-pretrained LLaVA-Rad is not implemented.')

    construct_prompt_ = partial(
        construct_prompt,
        tokenizer=tokenizer,
        image_processor=image_processor,
        use_key_image_only=use_key_image_only
    )

    # Define generation function
    def generate_fn(**kwargs):
        # Set up stopping criteria
        stopping_criteria = KeywordsStoppingCriteria(['</s>'], tokenizer, kwargs['input_ids'])
        
        input_len = kwargs['input_ids'].shape[-1]
        outputs = model.generate(
            stopping_criteria=[stopping_criteria],
            **kwargs
        )
        output_ids = outputs[0][input_len:]
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        return output_text

    return model, construct_prompt_, generate_fn