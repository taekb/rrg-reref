from pathlib import Path
from typing import Optional, Union
from easydict import EasyDict as edict
from functools import partial
from PIL import Image

import torch
from transformers import (
    BertTokenizer, 
    ViTImageProcessor,
    VisionEncoderDecoderModel,
    GenerationConfig,
    BitsAndBytesConfig
)

def construct_prompt(
    example: Union[dict,edict],
    image_dir: Union[str,Path],
    section_supported: Optional[str] = 'findings',
    tokenizer: Optional[BertTokenizer] = None,
    image_processor: Optional[ViTImageProcessor] = None,
    use_key_image_only: Optional[bool] = True,
    dataset_name: Optional[str] = 'mimic-cxr',
    section_to_generate: Optional[str] = 'findings',
    tokenize: Optional[bool] = True
):
    '''Constructs a prompt for CheXpertPlus-CheX-MIMIC given an example.'''

    if tokenizer is None:
        raise ValueError('Tokenizer must be provided to construct the prompt.')

    if section_supported != section_to_generate:
        raise ValueError(
            f'Mismatch in sections between loaded CheXpertPlus-CheX-MIMIC model and requested section: '
            f'Supported="{section_supported}" vs. Requested="{section_to_generate}".'
        )

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

        image = Image.open(image_path).convert('RGB')
    else:
        raise ValueError('CheXpertPlus-CheX-MIMIC was only evaluated on single-image inputs.')

    # Prepare image input
    pixel_values = image_processor(image, return_tensors='pt').pixel_values
    inputs = dict(images=pixel_values)

    return inputs

def load_model(
    model_name: Optional[str] = 'chex-mimic-findings',
    load_pretrained: Optional[bool] = True,
    quantization_config: Optional[BitsAndBytesConfig] = None,
    torch_dtype: Optional[torch.dtype] = None,
    use_key_image_only: Optional[bool] = True
):
    section_supported= model_name.split('-')[-1]
    repo_id = f'IAMJB/chexpert-mimic-cxr-{section_supported}-baseline'
    
    # Load model
    model = VisionEncoderDecoderModel.from_pretrained(
        repo_id,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    )
    tokenizer = BertTokenizer.from_pretrained(repo_id)
    image_processor = ViTImageProcessor.from_pretrained(repo_id)

    construct_prompt_ = partial(
        construct_prompt,
        section_supported=section_supported,
        tokenizer=tokenizer,
        image_processor=image_processor,
        use_key_image_only=use_key_image_only
    )

    # Define generation function
    def generate_fn(**kwargs):
        pixel_values = kwargs['images']
        generation_config = GenerationConfig(
            bos_token_id=model.config.bos_token_id,
            eos_token_id=model.config.eos_token_id,
            pad_token_id=model.config.pad_token_id,
            num_return_sequences=1,
            max_new_tokens=kwargs['max_new_tokens'],
            use_cache=True,
            beam_width=2,
            decoder_start_token_id=tokenizer.cls_token_id,
        )
        outputs = model.generate(
            pixel_values,
            generation_config=generation_config
        )
        output_ids = outputs[0]
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

        return output_text

    return model, construct_prompt_, generate_fn