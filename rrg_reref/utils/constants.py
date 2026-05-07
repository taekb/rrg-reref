EVAL_DEFAULT_SEED = 0

INFERENCE_DEFAULT_SEED = 0

MODEL_ALIAS_TO_MODEL_NAME = {
    'maira-2': 'microsoft/maira-2',
    'medgemma-4b-it': 'google/medgemma-4b-it',
    'medversa': 'hyzhou/MedVersa',
    'chexagent': 'StanfordAIMI/CheXagent-2-3b',
    'llava-rad': 'microsoft/llava-rad',
    'libra': 'X-iZhang/libra-v1.0-7b',
    'med-cxrgen-f': 'X-iZhang/Med-CXRGen-F',
    'med-cxrgen-i': 'X-iZhang/Med-CXRGen-I',
    'med-cxrgen': 'X-iZhang/Med-CXRGen',
    'chexone': 'StanfordAIMI/CheXOne',
}

MODEL_TO_SECTIONS = {
    'microsoft/maira-2': ['findings'],
    'google/medgemma-4b-it': ['findings', 'impression', 'findings-and-impression'],
    'hyzhou/MedVersa': ['findings', 'impression', 'findings-and-impression'],
    'StanfordAIMI/CheXagent-2-3b': ['findings', 'impression'],
    'microsoft/llava-rad': ['findings'],
    'chex-mimic-findings': ['findings'],
    'chex-mimic-impression': ['impression'],
    'chex-mimic': ['findings', 'impression'],
    'X-iZhang/libra-v1.0-7b': ['findings'],
    'X-iZhang/Med-CXRGen-F': ['findings'],
    'X-iZhang/Med-CXRGen-I': ['impression'],
    'X-iZhang/Med-CXRGen': ['findings', 'impression'],
    'StanfordAIMI/CheXOne': ['findings', 'impression', 'findings-and-impression'],

    # ReXVal metric-oracles
    'oracle--BERTScore': ['impression'],
    'oracle--BLEU': ['impression'],
    'oracle--CheXbert': ['impression'],
    'oracle--RadGraph-F1': ['impression']
}

SUPPORTED_DATASETS = [
    'rexval',
    'mimic-cxr',
    'chexpertplus',
    'iu-xray',
    'rexgradient'
]