from typing import Any, Optional
from easydict import EasyDict as edict

import torch
from transformers import BitsAndBytesConfig

def load_model(
    model_name: str,
    load_pretrained: Optional[bool] = True,
    load_in_8bit: Optional[bool] = False,
    load_in_4bit: Optional[bool] = False,
    double_quant: Optional[bool] = True,
    quant_type: Optional[str] = 'nf4',
    torch_dtype: Optional[torch.dtype] = torch.bfloat16,
    **kwargs
):
    '''Loads the given model.'''

    # Prepare bitsandbytes config
    if load_in_4bit or load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=double_quant,
            bnb_4bit_quant_type=quant_type
        )
    else:
        quantization_config = None

    # Model load config
    load_config = edict(
        model_name=model_name,
        load_pretrained=load_pretrained,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype
    )

    # MAIRA-2
    if 'maira-2' in model_name.lower():
        from rrg_reref.models.maira2 import load_model
        return load_model(**load_config, **kwargs)
    
    # MedGemma
    elif 'medgemma' in model_name.lower():
        from rrg_reref.models.medgemma import load_model
        return load_model(**load_config, **kwargs)

    # MedVersa
    elif 'medversa' in model_name.lower():
        from rrg_reref.models.medversa import load_model
        return load_model(**load_config, **kwargs)

    # CheXagent
    elif 'chexagent' in model_name.lower():
        from rrg_reref.models.chexagent import load_model
        return load_model(**load_config, **kwargs)

    # LLaVA-Rad
    elif 'llava-rad' in model_name.lower():
        from rrg_reref.models.llava_rad import load_model
        return load_model(**load_config, **kwargs)

    # CheXpertPlus-CheX-MIMIC
    elif 'chex-mimic' in model_name.lower():
        from rrg_reref.models.chex_mimic import load_model
        return load_model(**load_config, **kwargs)

    # CheXOne
    elif 'chexone' in model_name.lower():
        from rrg_reref.models.chexone import load_model
        return load_model(**load_config, **kwargs)

    # Libra / Med-CXRGen (all use the Libra library)
    elif 'libra' in model_name.lower() or 'cxrgen' in model_name.lower():
        from rrg_reref.models.libra import load_model
        return load_model(**load_config, **kwargs)

    else:
        raise ValueError(f"Unknown model name: {model_name}")