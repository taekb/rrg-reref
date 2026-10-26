import sys
import os
from pathlib import Path
from typing import Optional, Union
from easydict import EasyDict as edict
from functools import partial
from PIL import Image

from huggingface_hub import snapshot_download

import torch
from transformers import BitsAndBytesConfig, StoppingCriteriaList, StoppingCriteria

HF_CACHE_DIR = os.getenv('HF_HUB_CACHE')
REPO_ID = 'hyzhou/MedVersa'
REVISION = '23c7cd57626d299e2639cb2ec1ebae07acf818e6'
SNAPSHOT_DIR = Path(HF_CACHE_DIR) / f'models--{REPO_ID.replace("/", "--")}'
sys.path.append(str(SNAPSHOT_DIR))

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[]):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False

def construct_prompt(
    example: Union[dict,edict],
    image_dir: Union[str,Path],
    model: Optional[torch.nn.Module] = None,
    use_key_image_only: Optional[bool] = False,
    dataset_name: Optional[str] = 'mimic-cxr',
    section_to_generate: Optional[str] = 'findings'
):
    '''Constructs a prompt for MedVersa given an example.'''

    # Raise error if model is not provided
    if model is None:
        raise ValueError('Model must be provided to construct the input embeddings for MedVersa.')

    # Load MedVersa utilities
    if str(SNAPSHOT_DIR) not in sys.path:
        sys.path.append(str(SNAPSHOT_DIR))

    from utils import read_image

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
        
        # Read and preprocess image
        n_images = 1
        image_tensors = [read_image(str(image_path), 'cxr', 'report')]
        image_tensor = torch.cat(image_tensors).to(model.llama_model.device, dtype=model.llama_model.dtype)
    else:
        MAX_IMAGES = 10
        LATERAL_VIEWS = {'lateral', 'lat', 'll', 'rl', 'left lateral'}
        views = example.get('frontal_lateral', [])
        paired = list(zip(example.image_path, views)) if views else [(p, '') for p in example.image_path]
        paired.sort(key=lambda pv: 0 if pv[1].lower() not in LATERAL_VIEWS else 1)
        paired = paired[:MAX_IMAGES]

        image_paths = []
        for image_path, _ in paired:
            image_path = str(image_path).lstrip('/')
            
            if dataset_name == 'mimic-cxr':
                image_path = image_dir / 'files' / image_path
            
            elif dataset_name == 'chexpertplus':
                image_path = image_dir / image_path.replace('.jpg', '.png')
            
            else:
                image_path = image_dir / image_path
            
            image_paths.append(image_path)
        
        image_tensors = [read_image(str(image_path), 'cxr', 'report') for image_path in image_paths]
        image_tensor = torch.cat(image_tensors).to(model.llama_model.device, dtype=model.llama_model.dtype)
        n_images = len(image_tensors)
    
    # Include Indication, Technique, and Comparison sections if available
    indication = example['section_indication'] if example['section_indication'] is not None else None
    technique = example['section_technique'] if example['section_technique'] is not None else None
    comparison = example['section_comparison'] if example['section_comparison'] is not None else None

    context = ''
    context += f'Indication: {indication}\n' if indication is not None else ''
    context += f'Technique: {technique}\n' if technique is not None else ''
    context += f'Comparison: {comparison}\n' if comparison is not None else ''
    if len(context) == 0:
        context = 'Comparison:None.'

    # Specify prompt
    # NOTE: Using the prompts from Table 5 of the MedVersa paper.
    image_tokens = ''.join(f'<img{i}>' for i in range(n_images))
    if section_to_generate == 'findings':
        prompt = f'How would you characterize the findings from {image_tokens}?'
    
    elif section_to_generate == 'impression':
        prompt = f'Can you please provide your overall impression of {image_tokens}?'

    elif section_to_generate == 'findings-and-impression':
        prompt = f'Can you provide a report of {image_tokens} with findings and impression?'

    # Construct prompt
    prompt = f'<context>{context}</context>{prompt}'
    placeholder = ['<ImageHere>'] * 9
    prefix = '###Human:' + ''.join([f'<img{i}>' + ''.join(placeholder) + f'</img{i}>' for i in range(n_images)])
    img_embeds, atts_img, img_embeds_list = model.encode_img(
        image_tensor.unsqueeze(0),
        modals=['cxr'],
        task_types=['report']
    )
    img_embeds, atts_img = model.prompt_wrap(img_embeds, atts_img, [prefix], [n_images])
    prompt += '###Assistant:'
    prompt_tokens = model.llama_tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(image_tensor.device)
    new_img_embeds, new_atts_img = model.prompt_concat(img_embeds, atts_img, prompt_tokens)

    return dict(inputs_embeds=new_img_embeds, attention_mask=new_atts_img)

def load_model(
    model_name: Optional[str] = 'hyzhou/MedVersa',
    load_pretrained: Optional[bool] = True,
    quantization_config: Optional[BitsAndBytesConfig] = None,
    torch_dtype: Optional[torch] = torch.bfloat16,
    use_key_image_only: Optional[bool] = False
):
    # Check if the model has already been downloaded
    if not SNAPSHOT_DIR.exists():
        snapshot_download(
            repo_id=REPO_ID,
            repo_type='model',
            revision=REVISION,
            local_dir=SNAPSHOT_DIR, 
            local_dir_use_symlinks=False,
            resume_download=True
        )
    
    # Set up path to model code and load the model
    sys.path.append(str(SNAPSHOT_DIR))
    from medomni.models.medomni import MedOmni
    from medomni.common.registry import Registry

    registry = Registry()
    model_cls = registry.get_model_class('medomni')
    model = model_cls.from_pretrained(SNAPSHOT_DIR)
    model = model.eval()

    construct_prompt_ = partial(
        construct_prompt,
        model=model,
        use_key_image_only=use_key_image_only
    )

    # Define generation function
    def generate_fn(**kwargs):
        device = kwargs['inputs_embeds'].device
        outputs = model.llama_model.generate(
            stopping_criteria=StoppingCriteriaList([
                StoppingCriteriaSub(
                    stops=[
                        torch.tensor([835]).type(torch.bfloat16).to(device),
                        torch.tensor([2277, 29937]).type(torch.bfloat16).to(device)
                    ]
                )
            ]),
            num_beams=1,
            min_length=1,
            repetition_penalty=1,
            length_penalty=1,
            **kwargs
        )[0]
        
        # Remove unknown token <unk> at the beginning
        if outputs[0] == 0:
            outputs = outputs[1:]

        # Remove start token <s> at the beginning
        if outputs[0] == 1:
            outputs = outputs[1:]

        output_text = model.llama_tokenizer.decode(outputs, skip_special_tokens=True)
        output_text = output_text.split('###')[0].split('Assistant:')[-1].strip()

        # Remove headers in Findings/Impression generation
        # if section_to_generate in ['findings', 'impression']:
        #     if output_text.startswith('Findings:'):
        #         output_text = output_text.split('Findings:')[-1].strip()
        #     elif output_text.startswith('Impression:'):
        #         output_text = output_text.split('Impression:')[-1].strip()

        # Adjust headers
        # elif section_to_generation == 'findings+impression':
        if 'Findings:' in output_text:
            output_text = output_text.replace('Findings:', 'FINDINGS: ')
        
        if 'Impression:' in output_text:
            output_text = output_text.replace('Impression:', '\n\nIMPRESSION: ')

        output_text = output_text.replace('  ', ' ')
        output_text = output_text.strip()

        return output_text

    return model, construct_prompt_, generate_fn