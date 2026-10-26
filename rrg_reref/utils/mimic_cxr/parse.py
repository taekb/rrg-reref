from typing import Optional, Dict, List
from pydantic import create_model

from rrg_reref.utils.llm_api import prompt_llm

'''
    LLM-based section parser for MIMIC-CXR reports, used to extract the Indication, Comparison,
    Technique, Findings, and Impression sections (Appendix C of the paper).

    The prompt is adapted from LLaVA-Rad (Zambrano Chaves et al., 2025; Supplementary Table 6),
    but instructs the LLM to extract each section verbatim.

'''

SYSTEM_PROMPT = (
    'You are an expert medical assistant AI capable of modifying clinical documents to user specifications. '
    'You make minimal changes to the original document to satisfy user requests. '
    'You never add information that is not already directly stated in the original document.'
)
SECTION_EXTRACTION_PROMPT_FORMAT = (
    "Extract the following sections from the input radiology report: [{section_names}]. "
    'Leave an extracted section as "N/A" if it does not exist in the original report. '
    'The output should be in JSON format. '
    'An Indication section can refer to the History, Indication or Reason for Study sections in the original report. '
    'An Impression section can include the Recommendations and Conclusion sections in the original report. '
    'The extracted sections should be verbatim from the original report and should not overlap with each other. ' 
    'Remove any linebreaks and whitespace that have been added only for better readability.\n\n'
    "Examples of inputs and expected outputs (when e.g., extracting the ['Examination', 'Indication', 'Findings', 'Impression'] sections):\n\n"
    'INPUT:\n'
    'EXAMINATION: XR CHEST AP PORTABLE\n'
    'INDICATION: Small right apical pneumothorax after lung biopsy.\n'
    'FINDINGS: Single portable view of the chest was obtained. Copared with 10:42 AM. '
    'The small right apical pneumothorax has decreased slightly in size, the improvement best appreciated laterally where '
    'it now measures 10 mm compared to 14 mm before. '
    'At the lung apex it now measures 1.6 and compared to 2.1 cm previously. '
    'A subtle right apical pulmonary contusion is grossly stable. '
    'Minor chest wall emphysema along the right exilla has not changed significant delay. '
    'There is no metastatic shift. '
    'No pleural effusion is evident.\n\n'
    'OUTPUT:\n'
    '{{\n'
    '\t"EXAMINATION": "XR CHEST AP PORTABLE.",\n'
    '\t"INDICATION": "Small right apical pneumothorax after lung biopsy.",\n'
    '\t"FINDINGS": "Single portable view of the chest was obtained. '
    'The small right apical pneumothorax measures 10mm. '
    'At the lung apex it measures 1.6cm. '
    'A subtle right apical pulmonary contusion is grossly stable. '
    'Minor chest wall emphysema is noted along the right exilla. '
    'There is no metastatic shift. '
    'No pleural effusion is evident.",\n'
    '\t"IMPRESSION": "N/A"\n'
    '}}\n\n'
    '**Here is the report that you should process**:\n\n' # Newly added
    '{report}\n'
)

def get_section_extraction_prompt(
    report: str,
    section_names: List[str]
) -> str:
    '''Constructs the prompt for LLM-based section extraction.'''

    prompt = SECTION_EXTRACTION_PROMPT_FORMAT.format(
        section_names=', '.join([f"'{section_name.capitalize()}'" for section_name in section_names]),
        report=report
    )

    return prompt

def extract_sections_via_llm(
    report: str,
    section_names: Optional[List[str]] = None,
    llm_name: Optional[str] = 'claude-sonnet-4-5',
    temperature: Optional[float] = 0.,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = 16384,
    azure_openai: Optional[bool] = True
) -> Dict[str,Optional[str]]:
    '''
        Extracts the sections from a full radiology report in MIMIC-CXR using an LLM.
        
        NOTE: Missing sections are returned as None.

    '''

    if section_names is None:
        section_names = ['indication', 'comparison', 'technique', 'findings', 'impression']

    # Define response schema dynamically based on the section names
    ReportSections = create_model(
        'ReportSections',
        **{section_name: (str, ...) for section_name in section_names}
    )

    # Compile generation config
    generation_kwargs = dict(
        max_tokens=max_tokens,
        response_model=ReportSections
    )
    if temperature is not None:
        generation_kwargs['temperature'] = temperature
    if top_p is not None:
        generation_kwargs['top_p'] = top_p

    # Extract sections via LLM
    prompt = get_section_extraction_prompt(
        report=report,
        section_names=section_names
    )
    sections = prompt_llm(
        model_name=llm_name,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=prompt,
        azure_openai=azure_openai,
        **generation_kwargs
    )
    new_sections = {}
    for section_name in section_names:
        new_sections[section_name] = getattr(sections, section_name)
        new_sections[section_name] = None if new_sections[section_name] == 'N/A' else new_sections[section_name]

    return new_sections
