import os
import re
from typing import Optional, Dict, List
from pydantic import BaseModel, create_model
import instructor

from rrg_reref.utils.llm_api import prompt_llm

'''
    MIMIC-CXR section parser (ours).

'''

NEXT_HEADER = r"""
(?=
    ^\s*(?:
        INDICATION|HISTORY|CLINICAL\s+HISTORY|CLINICAL\s+INFORMATION|REASON\s+FOR\s+EXAM|REASON\s+FOR\s+EXAMINATION|
        COMPARISON|COMPARISONS|
        TECHNIQUE|STUDY|TYPE\s+OF\s+EXAMINATION|EXAMINATION|EXAM|
        FINDINGS|
        IMPRESSION|
        FINAL\s+REPORT
    )\b
)
"""

def _compile_section(header_regex: str) -> re.Pattern:
    '''Compiles the regex for each section header.'''

    return re.compile(
        rf"""
        ^\s*(?:{header_regex})\s*:?\s*    # header line (optionally with colon)
        (.*?)                             # section body (non-greedy)
        (?:
            {NEXT_HEADER}                 # until next header
            |\Z                           # or end of string
        )
        """,
        re.IGNORECASE | re.MULTILINE | re.DOTALL | re.VERBOSE,
    )

SECTION_REGEX: Dict[str,re.Pattern] = {
    # Indication section headers
    'indication': _compile_section(
        r"INDICATION|HISTORY|CLINICAL\s+HISTORY|CLINICAL\s+INFORMATION|REASON\s+FOR\s+EXAM|REASON\s+FOR\s+EXAMINATION"
    ),

    # Comparison section headers
    'comparison': _compile_section(r"COMPARISON|COMPARISONS"),

    # Technique section headers
    'technique': _compile_section(
        r"TECHNIQUE|STUDY|TYPE\s+OF\s+EXAMINATION|EXAMINATION|EXAM"
    ),

    # Findings section headers
    'findings': _compile_section(r"FINDINGS"),

    # Impression section headers
    "impression": _compile_section(r"IMPRESSION")
}

def extract_sections_via_regex(
    report: str,
    section_names: Optional[List[str]] = None
) -> Dict[str,Optional[str]]:
    '''
        Extracts the sections from a full radiology report in MIMIC-CXR.

        NOTE: Missing sections are returned as None.

    '''

    if section_names is None:
        section_names = list(SECTION_REGEX.keys())
    
    sections = {}
    for section_name in section_names:
        regex = SECTION_REGEX[section_name]
        m = regex.search(report)
        sections[section_name] = m.group(1).strip() if m else None

    return sections

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
    # 'Remove any information not directly observable from the current imaging study. '
    # 'For instance, remove any patient demographic data, past medical history, or comparison to prior images or studies. '
    # "The generated 'Findings' and 'Impression' sections should not reference any changes based on prior images, studies, '
    # 'or external knowledge about the patient. '
    # 'Rewrite such comparisons as a status observation based only on the current image or study. '
    # 'Remember to remove any numbering or bullets.\n\n'
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
# SECTION_EXTRACTION_PROMPT_FORMAT = (
#     'Identify the text span corresponding to the "{section_name}" section the following radiology report:\n\n'
#     '```\n'
#     '{report}\n'
#     '```\n\n'
#     'It is possible that the section is NOT present in the report. '
#     'In this case, you should simply output "N/A" as the extracted section text.\n'
#     'MAKE SURE TO OUTPUT THE CORRECT SPAN FROM THE ORIGINAL REPORT *VERBATIM*.\n'
#     'DO NOT MODIFY, ADD, OR REMOVE ANY TEXT FROM THE ORIGINAL REPORT.'
# )

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

    # Fetch provider name
    if 'gpt' in llm_name.lower():
        provider_name = 'azure_openai' if azure_openai else 'openai'
    elif 'claude' in llm_name.lower():
        provider_name = 'anthropic'
    else:
        raise ValueError(f'Unsupported LLM: {llm_name}')

    # Set up LLM endpoint
    client_kwargs = {}
    if 'openai' in provider_name and azure_openai:
        client_kwargs['api_version'] = os.getenv('AZURE_OPENAI_API_VERSION')
    
    client = instructor.from_provider(
        f'{provider_name}/{llm_name}',
        **client_kwargs
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
        **generation_kwargs
    )
    new_sections = {}
    for section_name in section_names:
        new_sections[section_name] = getattr(sections, section_name)
        new_sections[section_name] = None if new_sections[section_name] == 'N/A' else new_sections[section_name]

    return new_sections

# def extract_sections_via_llm(
#     report: str,
#     section_names: Optional[List[str]] = None,
#     llm_name: Optional[str] = 'claude-sonnet-4-5',
#     temperature: Optional[float] = 0.,
#     top_p: Optional[float] = None,
#     max_tokens: Optional[int] = 16384,
#     azure_openai: Optional[bool] = True
# ) -> Dict[str,Optional[str]]:

#     '''
#         Extracts the sections from a full radiology report in MIMIC-CXR using an LLM.

#         NOTE: Missing sections are returned as None.

#     '''

#     # Fetch provider name
#     if 'gpt' in llm_name.lower():
#         provider_name = 'azure_openai' if azure_openai else 'openai'
#     elif 'claude' in llm_name.lower():
#         provider_name = 'anthropic'
#     else:
#         raise ValueError(f'Unsupported LLM: {llm_name}')

#     # Set up LLM endpoint
#     client_kwargs = {}
#     if 'openai' in provider_name and azure_openai:
#         client_kwargs['api_version'] = os.getenv('AZURE_OPENAI_API_VERSION')
    
#     client = instructor.from_provider(
#         f'{provider_name}/{llm_name}',
#         **client_kwargs
#     )

#     # Compile generation config
#     generation_kwargs = dict(
#         max_tokens=max_tokens,
#         response_model=ReportSection
#     )
#     if temperature is not None:
#         generation_kwargs['temperature'] = temperature
#     if top_p is not None:
#         generation_kwargs['top_p'] = top_p

#     # Iterate through specified sections
#     if section_names is None:
#         section_names = list(SECTION_REGEX.keys())

#     sections = {}
#     for section_name in section_names:
#         prompt = get_section_extraction_prompt(
#             report=report,
#             section_name=section_name
#         )
#         section = prompt_llm(
#             model_name=llm_name,
#             system_prompt='You are a medical expert in radiology.',
#             user_prompt=prompt,
#             **generation_kwargs
#         )
#         is_present = section.is_present
#         section_text = section.section_text
#         if not is_present:
#             assert (section_text is None) or (section_text == '')

#         sections[section_name] = section_text

#     return sections

# def extract_sections_via_regex_and_llm(
#     report: str,
#     section_names: Optional[List[str]] = None,
#     llm_name: Optional[str] = 'claude-sonnet-4-5',
#     temperature: Optional[float] = 0.,
#     top_p: Optional[float] = None,
#     max_tokens: Optional[int] = 16384,
#     azure_openai: Optional[bool] = True
# ) -> Dict[str,Optional[str]]:
#     '''
#         Extracts the specified sections from a full radiology report, 
#         first using a regex-based approach and then using an LLM to extract
#         section spans that may not conform to the structure assumed by the regexes.

#     '''

#     if section_names is None:
#         section_names = list(SECTION_REGEX.keys())

#     # Extract sections via regex
#     sections_regex = extract_sections_via_regex(
#         report=report,
#         section_names=section_names
#     )
    
#     # Check for missing sections
#     missing_section_names = [
#         section_name for section_name in section_names
#         if sections_regex[section_name] is None
#     ]
    
#     # Prompt an LLM to catch potentially missed sections
#     if len(missing_section_names) > 0:
#         sections_llm = extract_sections_via_llm(
#             report=report,
#             section_names=missing_section_names,
#             llm_name=llm_name,
#             temperature=temperature,
#             top_p=top_p,
#             max_tokens=max_tokens,
#             azure_openai=azure_openai
#         )
    
#     sections = {**sections_regex, **sections_llm}

#     return sections

'''
    MIMIC-CXR section parser based on mimic-code.

    Adapted from: https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv-cxr/txt/section_parser.py

'''

# Source: https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv-cxr/txt/section_parser.py
def section_text(text):
    """Splits text into sections.

    Assumes text is in a radiology report format, e.g.:

        COMPARISON:  Chest radiograph dated XYZ.

        IMPRESSION:  ABC...

    Given text like this, it will output text from each section, 
    where the section type is determined by the all caps header.

    Returns a three element tuple:
        sections - list containing the text of each section
        section_names - a normalized version of the section name
        section_idx - list of start indices of the text in the section
    """
    p_section = re.compile(
        r'\n ([A-Z ()/,-]+):\s', re.DOTALL)

    sections = list()
    section_names = list()
    section_idx = list()

    idx = 0
    s = p_section.search(text, idx)

    if s:
        sections.append(text[0:s.start(1)])
        section_names.append('preamble')
        section_idx.append(0)

        while s:
            current_section = s.group(1).lower()
            # get the start of the text for this section
            idx_start = s.end()
            # skip past the first newline to avoid some bad parses
            idx_skip = text[idx_start:].find('\n')
            if idx_skip == -1:
                idx_skip = 0

            s = p_section.search(text, idx_start + idx_skip)

            if s is None:
                idx_end = len(text)
            else:
                idx_end = s.start()

            sections.append(text[idx_start:idx_end])
            section_names.append(current_section)
            section_idx.append(idx_start)

    else:
        sections.append(text)
        section_names.append('full report')
        section_idx.append(0)

    section_names = normalize_section_names(section_names)

    # remove empty sections
    # this handles when the report starts with a finding-like statement
    #  .. but this statement is not a section, more like a report title
    #  e.g. p10/p10103318/s57408307
    #    CHEST, PA LATERAL:
    #
    #    INDICATION:   This is the actual section ....
    # it also helps when there are multiple findings sections
    # usually one is empty
    for i in reversed(range(len(section_names))):
        if section_names[i] in ('impression', 'findings'):
            if sections[i].strip() == '':
                sections.pop(i)
                section_names.pop(i)
                section_idx.pop(i)

    if ('impression' not in section_names) & ('findings' not in section_names):
        # create a new section for the final paragraph
        if '\n \n' in sections[-1]:
            sections.append('\n \n'.join(sections[-1].split('\n \n')[1:]))
            sections[-2] = sections[-2].split('\n \n')[0]
            section_names.append('last_paragraph')
            section_idx.append(section_idx[-1] + len(sections[-2]))

    return sections, section_names, section_idx


def normalize_section_names(section_names):
    # first, lower case all
    section_names = [s.lower().strip() for s in section_names]

    frequent_sections = {
        "preamble": "preamble",  # 227885
        "impression": "impression",  # 187759
        "comparison": "comparison",  # 154647
        "indication": "indication",  # 153730
        "findings": "findings",  # 149842
        "examination": "examination",  # 94094
        "technique": "technique",  # 81402
        "history": "history",  # 45624
        "comparisons": "comparison",  # 8686
        "clinical history": "history",  # 7121
        "reason for examination": "indication",  # 5845
        "notification": "notification",  # 5749
        "reason for exam": "indication",  # 4430
        "clinical information": "history",  # 4024
        "exam": "examination",  # 3907
        "clinical indication": "indication",  # 1945
        "conclusion": "impression",  # 1802
        "chest, two views": "findings",  # 1735
        "recommendation(s)": "recommendations",  # 1700
        "type of examination": "examination",  # 1678
        "reference exam": "comparison",  # 347
        "patient history": "history",  # 251
        "addendum": "addendum",  # 183
        "comparison exam": "comparison",  # 163
        "date": "date",  # 108
        "comment": "comment",  # 88
        "findings and impression": "impression",  # 87
        "wet read": "wet read",  # 83
        "comparison film": "comparison",  # 79
        "recommendations": "recommendations",  # 72
        "findings/impression": "impression",  # 47
        "pfi": "history",
        'recommendation': 'recommendations',
        'wetread': 'wet read',
        'ndication': 'impression',  # 1
        'impresson': 'impression',  # 2
        'imprression': 'impression',  # 1
        'imoression': 'impression',  # 1
        'impressoin': 'impression',  # 1
        'imprssion': 'impression',  # 1
        'impresion': 'impression',  # 1
        'imperssion': 'impression',  # 1
        'mpression': 'impression',  # 1
        'impession': 'impression',  # 3
        'findings/ impression': 'impression',  # ,1
        'finding': 'findings',  # ,8
        'findins': 'findings',
        'findindgs': 'findings',  # ,1
        'findgings': 'findings',  # ,1
        'findngs': 'findings',  # ,1
        'findnings': 'findings',  # ,1
        'finidngs': 'findings',  # ,2
        'idication': 'indication',  # ,1
        'reference findings': 'findings',  # ,1
        'comparision': 'comparison',  # ,2
        'comparsion': 'comparison',  # ,1
        'comparrison': 'comparison',  # ,1
        'comparisions': 'comparison',  # ,1

        # ADDED
        'study': 'examination', # Example: 50770541
        'report': 'findings', # Example: 50770541; TODO: Check if this applies to other studies as well
        #'preliminary report': 'history' # Example: 57798512
    }

    p_findings = [
        'chest',
        'portable',
        'pa and lateral',
        'lateral and pa',
        'ap and lateral',
        'lateral and ap',
        'frontal and',
        'two views',
        'frontal view',
        'pa view',
        'ap view',
        'one view',
        'lateral view',
        'bone window',
        'frontal upright',
        'frontal semi-upright',
        'ribs',
        'pa and lat'
    ]
    p_findings = re.compile('({})'.format('|'.join(p_findings)))

    main_sections = [
        'impression', 'findings', 'history', 'comparison',
        'addendum'
    ]
    for i, s in enumerate(section_names):
        if s in frequent_sections:
            section_names[i] = frequent_sections[s]
            continue

        main_flag = False
        for m in main_sections:
            if m in s:
                section_names[i] = m
                main_flag = True
                break
        if main_flag:
            continue

        m = p_findings.search(s)
        if m is not None:
            section_names[i] = 'findings'

        # if it looks like it is describing the entire study
        # it's equivalent to findings
        # group similar phrasings for impression

    return section_names


def custom_mimic_cxr_rules():
    custom_section_names = {
        's50913680': 'recommendations',  # files/p11/p11851243/s50913680.txt
        's59363654': 'examination',  # files/p12/p12128253/s59363654.txt
        's59279892': 'technique',  # files/p13/p13150370/s59279892.txt
        's59768032': 'recommendations',  # files/p13/p13249077/s59768032.txt
        's57936451': 'indication',  # files/p14/p14325424/s57936451.txt
        's50058765': 'indication',  # files/p14/p14731346/s50058765.txt
        's53356173': 'examination',  # files/p15/p15898350/s53356173.txt
        's53202765': 'technique',  # files/p16/p16076182/s53202765.txt
        's50808053': 'technique',  # files/p16/p16631485/s50808053.txt
        's51966317': 'indication',  # files/p10/p10817099/s51966317.txt
        's50743547': 'examination',  # files/p11/p11388341/s50743547.txt
        's56451190': 'note',  # files/p11/p11842879/s56451190.txt
        's59067458': 'recommendations',  # files/p11/p11984647/s59067458.txt
        's59215320': 'examination',  # files/p12/p12408912/s59215320.txt
        's55124749': 'indication',  # files/p12/p12428492/s55124749.txt
        's54365831': 'indication',  # files/p13/p13876470/s54365831.txt
        's59087630': 'recommendations',  # files/p14/p14267880/s59087630.txt
        's58157373': 'recommendations',  # files/p15/p15032392/s58157373.txt
        's56482935': 'recommendations',  # files/p15/p15388421/s56482935.txt
        's58375018': 'recommendations',  # files/p15/p15505556/s58375018.txt
        's54654948': 'indication',  # files/p17/p17090359/s54654948.txt
        's55157853': 'examination',  # files/p18/p18975498/s55157853.txt
        's51491012': 'history',  # files/p19/p19314266/s51491012.txt

    }

    custom_indices = {
        's50525523': [201, 349],  # files/p10/p10602608/s50525523.txt
        's57564132': [233, 554],  # files/p10/p10637168/s57564132.txt
        's59982525': [313, 717],  # files/p11/p11989982/s59982525.txt
        's53488209': [149, 475],  # files/p12/p12458657/s53488209.txt
        's54875119': [234, 988],  # files/p13/p13687044/s54875119.txt
        's50196495': [59, 399],  # files/p13/p13894879/s50196495.txt
        's56579911': [59, 218],  # files/p15/p15394326/s56579911.txt
        's52648681': [292, 631],  # files/p15/p15666238/s52648681.txt
        's59889364': [172, 453],  # files/p15/p15835529/s59889364.txt
        's53514462': [73, 377],  # files/p16/p16297706/s53514462.txt
        's59505494': [59, 450],  # files/p16/p16730991/s59505494.txt
        's53182247': [59, 412],  # files/p16/p16770442/s53182247.txt
        's51410602': [47, 320],  # files/p17/p17069955/s51410602.txt
        's56412866': [522, 822],  # files/p17/p17612000/s56412866.txt
        's54986978': [59, 306],  # files/p17/p17912487/s54986978.txt
        's59003148': [262, 505],  # files/p17/p17916384/s59003148.txt
        's57150433': [61, 394],  # files/p18/p18335791/s57150433.txt
        's56760320': [219, 457],  # files/p18/p18418794/s56760320.txt
        's59562049': [158, 348],  # files/p18/p18502016/s59562049.txt
        's52674888': [145, 296],  # files/p19/p19381919/s52674888.txt
        's55258338': [192, 568],  # files/p13/p13719117/s55258338.txt
        's59330497': [140, 655],  # files/p15/p15479218/s59330497.txt
        's52119491': [179, 454],  # files/p17/p17959278/s52119491.txt
        # below have no findings at all in the entire report
        's58235663': [0, 0],  # files/p11/p11573679/s58235663.txt
        's50798377': [0, 0],  # files/p12/p12632853/s50798377.txt
        's54168089': [0, 0],  # files/p14/p14463099/s54168089.txt
        's53071062': [0, 0],  # files/p15/p15774521/s53071062.txt
        's56724958': [0, 0],  # files/p16/p16175671/s56724958.txt
        's54231141': [0, 0],  # files/p16/p16312859/s54231141.txt
        's53607029': [0, 0],  # files/p17/p17603668/s53607029.txt
        's52035334': [0, 0],  # files/p19/p19349312/s52035334.txt
    }

    return custom_section_names, custom_indices

SECTION_NAME_TO_SUBSECTION_NAMES = {
    'indication': ['indication', 'history'],
    'comparison': ['comparison'],
    'technique': ['technique', 'examination'],
    'findings': ['findings'],
    'impression': ['impression', 'recommendations'],
    'other': [
        'preamble', 'notification', 'date', 'addendum', 
        'comment', 'wet read', 'last_paragraph', 'full report'
    ]
}
SUBSECTION_NAME_TO_SECTION_NAME = {
    'indication': 'indication',
    'history': 'indication',
    'comparison': 'comparison',
    'technique': 'technique',
    'examination': 'technique',
    'findings': 'findings',
    'impression': 'impression',
    'recommendations': 'impression',
    'preamble': 'other',
    'notification': 'other',
    'date': 'other',
    'addendum': 'other',
    'comment': 'other',
    'wet read': 'other',
    'last_paragraph': 'other',
    'full report': 'other'
}

def extract_sections_mimic_code(
    report: str,
    section_names: Optional[List[str]] = None
) -> Dict[str,Optional[str]]:
    '''
        Extracts the specified sections from a full radiology report.

        Groups fine-grained sections from mimic-code parser into the following sections:
        - Indication
        - Comparison
        - Technique
        - Findings
        - Impression

        By specifying `section_names`, you can extract a subset of the above sections.

    '''

    # Default section names to extract
    if section_names is None:
        section_names = ['indication', 'comparison', 'technique', 'findings', 'impression']

    # Extract all subsections
    subsection_texts, subsection_names, _ = section_text(report)
    
    # Tag the subsections
    tagged_section_names = []
    for subsection_name in subsection_names:
        if subsection_name in SUBSECTION_NAME_TO_SECTION_NAME:
            tagged_section_names.append(SUBSECTION_NAME_TO_SECTION_NAME[subsection_name])
        else:
            tagged_section_names.append('unknown')

    # Iterate through tagged section names and gather the subsection texts
    section_texts = {section_name: None for section_name in section_names}
    for i, tagged_section_name in enumerate(tagged_section_names):
        # Skip any irrelevant sections
        if tagged_section_name not in section_names:
            continue

        # Fetch the subsection text
        subsection_name = subsection_names[i]
        subsection_text = subsection_texts[i]
        subsection_text = re.sub(r'^\s+|\s+$', '', subsection_text) # Remove any whitespace before and after
        subsection_text = subsection_text.replace('\n', '').replace('  ', ' ') # Remove linebreaks and spaces added for readability

        # Add subsection headers for all sections other than Findings and Impression
        if tagged_section_name not in ['findings', 'impression']:
            subsection_text = f'{subsection_name.upper()}: {subsection_text}'

        if section_texts[tagged_section_name] is None:
            section_texts[tagged_section_name] = subsection_text
        else:
            section_texts[tagged_section_name] += f'\n\n{subsection_text}'
    
    return section_texts

'''
    MIMIC-CXR section parser based on LLaVA-Rad (Zambrano Chaves et al., 2025).

    Implemented is based on Supplementary Table 6 in the paper.

    NOTE: While the sections are meant to be extracted verbatim, any information that is not 
    fully grounded on the given imaging study is instructed to be removed (e.g., prior comparisons).

'''

SYSTEM_PROMPT_LLAVA_RAD = (
    'You are an expert medical assistant AI capable of modifying clinical documents to user specifications. '
    'You make minimal changes to the original document to satisfy user requests. '
    'You never add information that is not already directly stated in the original document.'
)
SECTION_EXTRACTION_PROMPT_FORMAT_LLAVA_RAD = (
    "Extract four sections from the input radiology report: 'Examination', 'Indication', 'Findings', and 'Impression'. "
    'Leave an extracted section as "N/A" if it does not exist in the original report. '
    'The output should be in JSON format. '
    'An Indication section can refer to the History, Indication or Reason for Study sections in the original report. '
    'Remove any information not directly observable from the current imaging study. '
    'For instance, remove any patient demographic data, past medical history, or comparison to prior images or studies. '
    "The generated 'Findings' and 'Impression' sections should not reference any changes based on prior images, studies, "
    'or external knowledge about the patient. '
    'Rewrite such comparisons as a status observation based only on the current image or study. '
    'Remember to remove any numbering or bullets.\n\n'
    'Examples of inputs and expected outputs:\n\n'
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
    '\t"IMPRESSION": N/A}}\n\n'
    '**Here is the report that you should process**:\n\n' # Newly added
    '{report}\n'
)

def get_section_extraction_prompt_llava_rad(report: str):
    '''Constructs the prompt for LLM-based section extraction used in LLaVA-Rad (Zambrano Chaves et al., 2025).'''

    prompt = SECTION_EXTRACTION_PROMPT_FORMAT_LLAVA_RAD.format(report=report)

    return prompt

def extract_sections_llava_rad(
    report: str,
    section_names: Optional[List[str]] = None, # NOTE: Not used at the moment
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

    class ReportSections(BaseModel):
        '''Response schema for the sections of a radiology report.'''

        examination: str
        indication: str
        findings: str
        impression: str

    # Fetch provider name
    if 'gpt' in llm_name.lower():
        provider_name = 'azure_openai' if azure_openai else 'openai'
    elif 'claude' in llm_name.lower():
        provider_name = 'anthropic'

    # Set up LLM endpoint
    client_kwargs = {}
    if 'openai' in provider_name and azure_openai:
        client_kwargs['api_version'] = os.getenv('AZURE_OPENAI_API_VERSION')
    
    client = instructor.from_provider(
        f'{provider_name}/{llm_name}',
        **client_kwargs
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

    # Extract sections
    prompt = get_section_extraction_prompt_llava_rad(report=report)
    sections = prompt_llm(
        model_name=llm_name,
        system_prompt=SYSTEM_PROMPT_LLAVA_RAD,
        user_prompt=prompt,
        **generation_kwargs
    )
    examination = sections.examination
    indication = sections.indication
    findings = sections.findings
    impression = sections.impression
    sections = {
        'examination': None if examination == 'N/A' else examination,
        'indication': None if indication == 'N/A' else indication,
        'findings': None if findings == 'N/A' else findings,
        'impression': None if impression == 'N/A' else impression
    }

    return sections