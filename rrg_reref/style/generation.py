from pydantic import BaseModel, Field
from typing import Optional
import json

from rrg_reref.style.axes import get_style_axis_cls, StyleAxis
from rrg_reref.utils.llm_api import prompt_llm

'''
    Utilities for simulating stylistic variations in radiology reports.

'''

# MAIN PROMPT FORMATS
CONTROLLED_STYLE_VARIATION_PROMPT_FORMAT = (
    'It is well-known that even for the same imaging study, radiologists can write the Findings and Impression sections in highly variable ways.\n'
    'Such variability can reflect differences in reporting style, structure, and interpretive emphasis, even when describing the same underlying findings.\n'
    'Given a ground-truth reference report for an imaging study, written by a radiologist, your task is to rewrite it '
    'along the specified stylistic axis, while preserving all of the clinical content and interpretations present in the original report.\n\n'
    '======================== ORIGINAL REPORT =========================\n'
    '{orig_report}\n\n'
    #'==================================================================\n\n'
    '======================= STYLE AXIS TO VARY =======================\n'
    '- NAME: {name}\n'
    '- DESCRIPTION: {description}\n'
    '- VARIATIONS:\n'
    '{variations}\n'
    '- EXAMPLES:\n'
    '{examples}\n\n'
    #'==================================================================\n\n'
    '===================== IMPORTANT CONSTRAINTS ======================\n'
    '{constraints}\n\n'
    #'==================================================================\n\n'
    '===================== VARIATION TO GENERATE ======================\n'
    '- {variation_to_generate}\n\n'
    #'==================================================================\n\n'
)

RANDOM_STYLE_VARIATION_PROMPT_FORMAT = (
    "It is well-known that even for the same imaging study, radiologists can write the Findings and Impression sections in highly variable ways.\n"
    "Such variability can reflect differences in reporting style, structure, and interpretive emphasis, even when describing the same underlying findings.\n"
    "\n"
    "In particular, radiologists may differ in the following ways:\n"
    "- **Report Structure**: Some use highly templated, sectioned formats (e.g., with headings for each anatomical region), while others use continuous free-text prose.\n"
    "- **Level of Detail**: Some radiologists exhaustively mention all normal structures (e.g., using words like \"normal\", \"unremarkable\"), while others selectively omit normal findings and focus on key abnormalities.\n"
    "- **Enumeration of Anatomical Regions**: Some radiologists enumerate each substructure (e.g., \"costophrenic sulci\", \"diaphragmatic contour\", \"trachea\", \"hila\"), while others cover only high-level structures (e.g., \"No pulmonary parenchymal opacity or consolidation.\").\n"
    "- **Precision of Measurements**: Some radiologists include exact numerical measurements (e.g., \"4 mm extra-axial collection\", \"13 mm extrusion\"), while others use qualitative descriptions only (e.g., \"mild\", \"severe\", \"numerous\", \"no depressed fracture\").\n"
    "- **Technical Quality and Artifacts**: Some radiologists include comments on imaging quality and sources of artifacts (e.g., \"streak and beam hardening artifact from aneurysm clip\"), while others may omit discussions of quality entirely or include only brief references.\n"
    "- **Lexical Style**: Some prefer standardized, concise terminology (e.g., \"No consolidation or effusion\"), while others use descriptive or narrative phrasing (e.g., \"The lungs are clear and well expanded\").\n"
    "- **Terminology Preferences**: Different radiologists may use different terms for describing the same findings (e.g., \"hila unremarkable\" vs. \"normal pulmonary vasculature\", \"The costophrenic sulci and diaphragmatic contour are sharp.\" vs. \"No pleural effusion or pneumothorax.\").\n"
    "- **Interpretive Tone**: Some include mild interpretation or qualifying phrases (e.g., \"likely degenerative\", \"nonspecific but probably chronic\"), while others remain strictly descriptive.\n"
    "- **Certainty and Confidence**: Some express diagostic uncertainty (e.g., \"possible\", \"suggested\"), while others describe findings definitively.\n"
    "- **Organization of Content**: The location of comparison statements or secondary findings may differ — for instance, some include comparison with prior exams in the Findings section, while others place it in the Impression.\n"
    "\n"
    "Below is a ground-truth reference report for an imaging study, written by a radiologist.\n"
    "Please generate an **alternative** version of the Findings and Impression sections that reflects a plausible yet distinct radiologist style, consistent with real-world reporting variability.\n"
    "\n"
    "IMPORTANT:\n"
    "- Do **not** add any new information not supported by the ground-truth reference report.\n"
    "- Do **not** simply paraphrase or reorder sentences — instead, vary structure, level of detail, tone, medical terminology, and phrasing as described above.\n"
    "- The alternative version(s) should remain clinically consistent with the ground-truth reference report.\n"
    "- You may choose to vary multiple stylistic aspects at once, as long as the meaning is preserved.\n"
    "- Make sure that the alternative versions are still diverse and stylistically distinct from each other.\n"
    "\n"
    "You may assume that all other (often pre-filled) sections (e.g., Indication, Technique) of the report remain fixed.\n"
    "You should only output the alternative Findings and Impression sections, in the format below:\n"
    "[\n"
    "    {{\n"
    "        \"findings\": \"alternative findings section\",\n"
    "        \"impression\": \"alternative impression section\"\n"
    "    }}\n"
    "]\n\n"
    "Do not output anything else. No explanations or commentary.\n"
    "\n"
    "Here is the full ground-truth reference report:\n"
    "```\n"
    "{orig_report}\n"
    "```\n"
)

VERIFICATION_PROMPT_FORMAT = (
    'You will be given a ground-truth reference report for an imaging study, written by a radiologist, '
    'and an alternative version of the report, generated by perturbing the ground-truth report along '
    'one or more stylistic axes (e.g., report structure, terminology preferences, level of detail).\n'
    #'For the ground-truth report, you will be given the full report, whereas for the alternative version, '
    #'you will only be given the Findings and Impression sections.\n'
    'Your task is to check if the alternative version is *clinically consistent* with the ground-truth report.\n\n'
    'IMPORTANT:\n'
    '- The alternative version should not add any new information that is not substantiated by the original ground-truth reference report.\n'
    '- Minor paraphrases and reordering (within and across sections) of clinical content are allowed, as long as they are clinically identical in meaning.\n\n'
    '======================== GROUND-TRUTH REFERENCE REPORT =========================\n'
    '{orig_report}\n\n'
    #'================================================================================\n\n'
    '====================== ALTERNATIVE FINDINGS + IMPRESSION =======================\n'
    'FINDINGS: {alt_findings}\n'
    'IMPRESSION: {alt_impression}\n\n'
    #'================================================================================\n\n'
)

# Also checks if the alternative version is a valid perturbation of the ground-truth report for the specified stylistic axis
VERIFICATION_WITH_PERTURBATION_CHECKING_PROMPT_FORMAT = (
    'You will be given a ground-truth reference report for an imaging study, written by a radiologist, '
    'and an alternative version of the report, generated by perturbing the ground-truth report along '
    'one or more stylistic axes (e.g., report structure, terminology preferences, level of detail).\n'
    'Your task is to check if the alternative version is *clinically consistent* with the ground-truth report.\n'
    'Additionally, you will be given the stylistic axis along which the alternative version was meant to be perturbed, '
    'and you should check if the alternative version is a valid perturbation of the ground-truth report for the specified stylistic axis.\n\n'
    'IMPORTANT:\n'
    '- The alternative version should NOT add any new information that is not substantiated by the original ground-truth reference report.\n'
    '- Minor paraphrases and reordering (within and across sections) of clinical content are allowed, as long as they are clinically identical in meaning.\n'
    '- There can be cases where the perturbation was not made, either because the ground-truth report '
    '(i) does not contain sufficient information to make the perturbation without adding new information, or '
    '(ii) contains sufficient information but is already in the form expected after the perturbation along the specified stylistic axis. '
    'In both cases, you should return "True" for the "is_valid_perturbation" field (otherwise, return "False"). '
    'For case (i), you should also return "True" for the "does_not_contain_required_info" field (otherwise, return "False"). '
    'For case (ii), you should also return "True" for the "is_already_in_expected_form" field (otherwise, return "False"). '
    'Meanwhile, provide a detailed explanation for all of the above fields in the "reasoning" field. '
    'Examples of case (i): '
    '1. The original report does not discuss technical quality, but the perturbation is to have it discussed; '
    '2. The original report does not contain any detailed quantitative measurements, but the perturbation is to include them. '
    'Examples of case (ii): '
    '1. The original report is already structured, but the perturbation is to make it structured; '
    '2. The original report already exhaustively lists all normal and abnormal findings, but the perturbation is to make the list of findings exhaustive.\n\n'
    '====================== GROUND-TRUTH REFERENCE REPORT =======================\n'
    '{orig_report}\n\n'
    #'============================================================================\n\n'
    '======================== STYLE AXIS CHOSEN TO VARY =========================\n'
    '- NAME: {axis_name}\n'
    '- DESCRIPTION: {axis_description}\n'
    '- VARIATIONS:\n'
    '{axis_variations}\n'
    '- EXAMPLES:\n'
    '{axis_examples}\n\n'
    #'============================================================================\n\n'
    '=========================== VARIATION GENERATED ============================\n'
    '- {variation_generated}\n\n'
    #'============================================================================\n\n'
    '===================== ALTERNATIVE FINDINGS + IMPRESSION ====================\n'
    'FINDINGS: {alt_findings}\n'
    'IMPRESSION: {alt_impression}\n\n'
    #'============================================================================\n\n'
)

# DEFAULT GENERATION CONSTRAINTS
DEFAULT_CONSTRAINTS = [
    'Assume that all other sections (e.g., Indication, Technique) of the alternative report '
    'are identical to those of the original ground-truth reference report.',
    'Do NOT add any new information that is not substantiated by the original ground-truth reference report. '
    'Notably, if the original report does not include details required for perturbation along the specified axis, '
    'you should output the original report unchanged.',
    'The alternative version should remain clinically consistent with the original version.',
    'Introduce minimal changes beyond the specified stylistic axis that the report is being perturbed along.',
    'You should only output the alternative Findings and Impression sections. '
    'Do NOT output anything else. No explanations or commentary.'
    # 'You should only output the alternative Findings and Impression sections, in the format below:',
    # '[\n',
    # '    {{\n',
    # "        \"findings\": \"alternative findings section\",\n",
    # '        \"impression\": \"alternative impression section\"\n',
    # '    }},\n',
    # ']\n\n',
    # 'Do NOT output anything else. No explanations or commentary.'
]

class AlternativeReport(BaseModel):
    '''Response schema for the generated alternative report.'''

    findings: str = Field(description='The Findings section of the alternative report.')
    impression: str = Field(description='The Impression section of the alternative report.')

class AlternativeReportList(BaseModel):
    '''Response schema for a list of generated alternative reports.'''

    reports: list[AlternativeReport] = Field(description='A list of alternative reports.')

class VerificationResult(BaseModel):
    '''Response schema for validating clinical consistency of an alternative report.'''
    
    is_clinically_consistent: bool = Field(
        description='Whether the alternative report is clinically consistent with the original report.'
    )
    reasoning: str = Field(
        description='The reasoning for why or why not the alternative report is clinically consistent with the original report.'
    )

class VerificationWithPerturbationCheckingResult(BaseModel):
    '''Response schema for validating clinical consistency and perturbation of an alternative report.'''
    
    is_clinically_consistent: bool = Field(description='Whether the alternative report is clinically consistent with the original report.')
    is_valid_perturbation: bool = Field(description='Whether the alternative report is a valid perturbation of the original report.')
    does_not_contain_required_info: bool = Field(
        description='Whether the alternative report does not contain the required information to make the perturbation along the specified stylistic axis.'
    )
    is_already_in_expected_form: bool = Field(
        description='Whether the original report is already in the expected form after the perturbation along the specified stylistic axis.'
    )
    reasoning: str = Field(
        description=(
            'The reasoning for why or why not the alternative report is clinically consistent with the original report, '
            'whether it is a valid perturbation of the original report, and '
            'whether the original report is already in the expected form for the specified stylistic axis.'
        )
    )

'''
    Controlled generation of alternative reports, along a single stylistic axis.

'''

def get_controlled_style_variation_prompt(
    orig_report: str,
    style_axis: str | StyleAxis,
    variation_to_generate: str
) -> str:
    '''Constructs the main prompt for generation of a specific stylistic variant of the original report.'''

    if isinstance(style_axis, str):
        style_axis = get_style_axis_cls(style_axis)()

    # Validate that the specified variation to generate is valid
    assert variation_to_generate in getattr(style_axis, 'variations', []), (
        f'Invalid variation to generate: {variation_to_generate}. Valid variations are: {style_axis.variations}'
    )

    # Compile the full list of constraints
    constraints = DEFAULT_CONSTRAINTS + style_axis.constraints
    constraints = '\n'.join([
        f'- {constraint}'
        for constraint in constraints
    ])

    # Construct variation descriptions string
    variations = '\n'.join([
        f'    - {variation}: {description}'
        for variation, description in style_axis.variation_descriptions.items()
    ])

    # Construct examples string (JSON format)
    examples = json.dumps(style_axis.examples, indent=4)
    examples = examples.replace('\n', '\n    ')
    examples = f'    {examples}'

    # Construct the main prompt
    prompt = CONTROLLED_STYLE_VARIATION_PROMPT_FORMAT.format(
        orig_report=orig_report,
        name=style_axis.name,
        description=style_axis.description,
        variations=variations,
        examples=examples,
        constraints=constraints,
        variation_to_generate=variation_to_generate
    )

    return prompt

def prompt_for_controlled_style_variation(
    orig_report: str,
    style_axis: str | StyleAxis,
    variation_to_generate: str,
    model_name: Optional[str] = 'claude-sonnet-4-5',
    temperature: Optional[float] = 0.7,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = 16382,
    azure_openai: Optional[bool] = True
) -> str:
    '''Constructs the main prompt for generating a stylistic variant of the original report.'''

    # Fetch the prompt
    prompt = get_controlled_style_variation_prompt(
        orig_report=orig_report,
        style_axis=style_axis,
        variation_to_generate=variation_to_generate
    )

    # Generate the alternative report
    alt_report = prompt_llm(
        model_name=model_name,
        system_prompt='You are a medical expert in radiology.',
        user_prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        response_model=AlternativeReport,
        azure_openai=azure_openai
    )
    alt_findings = alt_report.findings
    alt_impression = alt_report.impression
    alt_sections = dict(
        findings=alt_findings,
        impression=alt_impression
    )

    return alt_sections

'''
    Random generation of alternative reports, along multiple stylistic axes.

'''

def get_random_style_variation_prompt(orig_report: str) -> str:
    '''Constructs the main prompt for generating a random stylistic variant of the original report.'''

    prompt = RANDOM_STYLE_VARIATION_PROMPT_FORMAT.format(orig_report=orig_report)
    
    return prompt

def prompt_for_random_style_variation(
    orig_report: str,
    model_name: Optional[str] = 'claude-sonnet-4-5',
    temperature: Optional[float] = 0.7,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = 16382,
    azure_openai: Optional[bool] = True
) -> str:
    '''Constructs the main prompt for generating a random stylistic variant of the original report.'''

    # Fetch the prompt
    prompt = get_random_style_variation_prompt(orig_report=orig_report)
    
    # Generate the alternative report
    alt_report = prompt_llm(
        model_name=model_name,
        system_prompt='You are a medical expert in radiology.',
        user_prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        response_model=AlternativeReport,
        azure_openai=azure_openai
    )
    alt_findings = alt_report.findings
    alt_impression = alt_report.impression
    alt_sections = dict(
        findings=alt_findings,
        impression=alt_impression
    )
    
    return alt_sections

'''
    Verification of alternative reports with respect to the original report.

'''

def get_verification_prompt(
    orig_report: str,
    alt_sections: dict[str,str],
    check_perturbation: Optional[bool] = False,
    axis_name: Optional[str] = None,
    variation_generated: Optional[str] = None
) -> str:
    '''Constructs the main prompt for verifying the clinical consistency of the alternative report.'''

    # Construct the main prompt
    if check_perturbation:
        axis_obj = get_style_axis_cls(axis_name)()
        axis_description = axis_obj.description
        axis_variations = axis_obj.variations
        axis_examples = axis_obj.examples

        # Construct variation descriptions string
        axis_variations = '\n'.join([
            f'    - {variation}: {description}'
            for variation, description in axis_obj.variation_descriptions.items()
        ])

        # Construct examples string (JSON format)
        axis_examples = json.dumps(axis_obj.examples, indent=4)
        axis_examples = axis_examples.replace('\n', '\n    ')
        axis_examples = f'    {axis_examples}'

        prompt = VERIFICATION_WITH_PERTURBATION_CHECKING_PROMPT_FORMAT.format(
            orig_report=orig_report, 
            alt_findings=alt_sections['findings'], 
            alt_impression=alt_sections['impression'], 
            axis_name=axis_name, 
            axis_description=axis_description, 
            axis_variations=axis_variations, 
            axis_examples=axis_examples,
            variation_generated=variation_generated
        )
    else:
        prompt = VERIFICATION_PROMPT_FORMAT.format(orig_report=orig_report, alt_report=alt_report)

    return prompt

def prompt_for_verification(
    orig_report: str,
    alt_sections: dict[str,str],
    model_name: Optional[str] = 'o3',
    temperature: Optional[float] = 0,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = 16382,
    azure_openai: Optional[bool] = True,
    check_perturbation: Optional[bool] = False,
    axis_name: Optional[str] = None,
    variation_generated: Optional[str] = None
):
    '''Prompts an LLM to verify the clinical consistency (and perturbation validity) of an alternative report.'''

    # Fetch the prompt
    prompt = get_verification_prompt(
        orig_report=orig_report, 
        alt_sections=alt_sections,
        check_perturbation=check_perturbation,
        axis_name=axis_name,
        variation_generated=variation_generated
    )

    # Validate the clinical consistency of the alternative report
    verification = prompt_llm(
        model_name=model_name,
        system_prompt='You are a medical expert in radiology.',
        user_prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        response_model=VerificationWithPerturbationCheckingResult if check_perturbation else VerificationResult,
        azure_openai=azure_openai
    )
    
    return verification