import os
import re
from typing import Optional
from pydantic import BaseModel

from openai import OpenAI, AzureOpenAI
from anthropic import Anthropic
import instructor

def prompt_llm(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    temperature: Optional[float] = 0.7,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = 16382,
    response_model: BaseModel | None = None,
    azure_openai: Optional[bool] = True
) -> str:
    '''Prompts an LLM.'''

    # Fetch provider name
    if 'gpt' in model_name.lower() or re.match(r'^o\d', model_name.lower()):
        provider_name = 'azure_openai' if azure_openai else 'openai'
    elif 'claude' in model_name.lower():
        provider_name = 'anthropic'
    else:
        raise ValueError(f'Unsupported LLM: {model_name}')

    # Set up LLM endpoint
    client_kwargs = {}
    if 'openai' in provider_name and azure_openai:
        client_kwargs['api_version'] = os.getenv('AZURE_OPENAI_API_VERSION')
    
    client = instructor.from_provider(
        f'{provider_name}/{model_name}',
        **client_kwargs
    )

    # Compile generation config
    generation_kwargs = dict(response_model=response_model)

    # NOTE: o-series models: 
    # Use `max_completion_tokens` instead of `max_tokens`
    # Do not allow `temperature` or `top_p` to be specified
    if provider_name == 'openai':
        generation_kwargs['max_completion_tokens'] = max_tokens
    else:
        generation_kwargs['max_tokens'] = max_tokens
        if temperature is not None:
            generation_kwargs['temperature'] = temperature
        if top_p is not None:
            generation_kwargs['top_p'] = top_p
    
    # Structured output generation
    response = client.create(
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        **generation_kwargs
    )

    # Extract raw text if no response model is specified
    if response_model is None:
        if 'openai' in provider_name:
            response = response.choices[0].message.content
        
        elif 'anthropic' in provider_name:
            response = response.content[0].text

    return response