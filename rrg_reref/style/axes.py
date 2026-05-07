from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class StyleAxis:
    '''Base class for all axes of stylistic variation in radiology reports.'''
    
    name: str = field(default='')
    axis_group: str = field(default='')
    axis_type: str = field(default='')
    description: str = field(default='')
    variations: List[str] = field(default_factory=lambda: [])
    variation_descriptions: Dict[str,str] = field(default_factory=lambda: {})
    examples: List[Dict[str,str]] = field(default_factory=lambda: [])
    constraints: List[str] = field(default_factory=lambda: [])

'''
    Higher level style axis groups.

'''

@dataclass
class Organization(StyleAxis):
    '''Axis controlling the organization of the report.'''
    
    axis_group: str = field(default='organization')

@dataclass
class Completeness(StyleAxis):
    '''Axis controlling the completeness of findings descriptions.'''
    
    axis_group: str = field(default='completeness')

@dataclass
class Granularity(StyleAxis):
    '''Axis controlling the anatomical, spatial, and quantitative granularity of findings descriptions.'''

    axis_group: str = field(default='granularity')

@dataclass
class Language(StyleAxis):
    '''Axis controlling the linguistic style of the report.'''

    axis_group: str = field(default='language')

'''
    Style axes corresponding to the "Organization" group.

'''

@dataclass
class Structure(Organization):
    name: str = field(default='structure')
    axis_type: str = field(default='absolute')
    description: str = field(default=(
        'Radiologists differ in their use of structured templates versus continuous free-text prose. '
        'Some use highly templated formats (e.g., explicit headings for each anatomical region, bullet points, numbered lists, subheadings for groups of findings), '
        'while others write in continuous free-text sentences and paragraphs.'
    ))
    variations: List[str] = field(default_factory=lambda: ['structured', 'free_text'])
    variation_descriptions: Dict[str,str] = field(default_factory=lambda: dict(
        structured=(
            'Written in highly structured templated format '
            '(e.g., explicit headings for each anatomical region, bullet points, numbered lists, subheadings for groups of findindgs).'
        ),
        free_text='Written in continuous free-text prose.'
    ))
    examples: List[Dict[str,str]] = field(default_factory=lambda: [
        dict(
            structured=(
                'CARDIOMEDIASTINAL: Normal silhouette. '
                'LUNGS: Well expanded and clear. '
                'PLEURA: Sharp costophrenic sulci and diaphragmatic contour. '
                'BONES: No acute abnormalities.'
            ),
            free_text=(
                'The cardiomediastinal silhouette is normal. '
                'The lungs are well expanded and clear. '
                'The costophrenic sulci and diaphragmatic contour are sharp. '
                'No acute osseous abnormalities.'
            )
        ),
        dict(
            structured=(
                '1. Single view of chest demonstrates no significant interval change in '
                'position of bilateral chest tubes. Median sternotomy wires are unchaged.\n'
                '2. Previously described left pneumothorax is not well seen on today\'s exam. '
                'No large pneumothorax is seen.\n'
                '3. Interval increase in left pleural effusion. Small right pleural effusion present.\n'
                'Summary: Possible significant abnormality and change. May need action.'
            ),
            free_text=(
                'Single view of chest demonstrates no significant interval change in '
                'position of bilateral chest tubes. Median sternotomy wires are unchaged. '
                'Previously described left pneumothorax is not well seen on today\'s exam. '
                'No large pneumothorax is seen. '
                'Interval increase in left pleural effusion. Small right pleural effusion present.\n'
                'Possible significant abnormality and change. May need action.'
            )
        ),
        dict(
            structured='1. Cardiomegaly. 2. Resolution of right basal infiltrate.',
            free_text='Cardiomegaly. Resolution of right basal infiltrate.'
        )
    ])
    constraints: List[str] = field(default_factory=lambda: [
        'The Findings and Impression sections need not be structured in the same way. '
        'For example, the Findings section can be more structured, while the Impression section '
        'is more free-text, and vice versa.'
    ])

# NOTE: Technically, there are also cases where the radiologist includes everything in 
# either the Findings or Impression section. However, simulating such cases results in 
# examples that do not have both the ground-truth and the alternative for the given section.
# As such, we do not simulate such cases in this study.
@dataclass
class SectionAssignment(Organization):
    name: str = field(default='section_assignment')
    axis_type: str = field(default='relative')
    description: str = field(default=(
        'Radiologists differ in where they place specific types of content within the report (Findings vs. Impression). '
        'For example, some discuss prior comparisons only in the Impression section to '
        'keep the Findings section fully grounded on the current study, '
        'while others discuss them in the Findings section to provide immediate clinical context. '
        'Meanwhile, there are also cases where comparison findings are discussed in both sections. '
        'As another example, some radiologists discuss the differentials in the Findings section (and in the Impression section), '
        'while others only discuss them in the Impression section.'
    ))
    variations: List[str] = field(default_factory=lambda: ['original', 'alternative'])
    variation_descriptions: Dict[str,str] = field(default_factory=lambda: dict(
        original='Retains original assignment of all contents to the Findings and Impression sections.',
        alternative='Reassigns contents to different sections, compared to the original version.'
    ))
    examples: List[Dict[str,str]] = field(default_factory=lambda: [
        dict(
            original=(
                'FINDINGS: Chronic left-sided rib fractures are again noted. '   
                'The cardiomediastinal and hilar contours are unchanged from ___. '
                'Pleural thickening and blunting at the right costophrenic angle is again demonstrated, '
                'and is stable from the prior exam in ___ and likely represents pleural scarring and a small pleural effusion. '
                'No focal consolidation or pneumothorax is identified.\n\n'
                'IMPRESSION: Multiple chronic appearing left-sided rib fractures. '
                'No pneumothorax. '
                'Blunting of the costophrenic angle on the right likely represents pleural scarring and a small effusion, '
                'not significantly changed from ___.'
            ),
            alternative=(
                'FINDINGS: Left-sided rib fractures are present. '
                'The cardiomediastinal and hilar contours are normal. '
                'Pleural thickening and blunting at the right costophrenic angle, '
                'likely representing pleural scarring and a small pleural effusion. '
                'No focal consolidation or pneumothorax is identified.\n\n'
                'IMPRESSION: Multiple chronic left-sided rib fractures, again noted and unchanged from ___. '
                'Cardiomediastinal and hilar contours remain stable. '
                'No pneumothorax. '
                'Blunting of the costophrenic angle on the right likely represents pleural scarring and a small effusion, '
                'again demonstrated and stable from prior exam in ___.'
            )
        ),
        dict(
            original=(
                'FINDINGS: A hazy opacity is present in the right lung, which may represent aspiration, pleural effusion or hemorrhage. '
                'Moderate cardiomegaly is stable. '
                'Slight prominence of the pulmonary vasculature with cephalization and enlarged pulmonary arteries are consistent with mild pulmonary edema. '
                'There are no displaced rib fractures.\n\n'
                'IMPRESSION: 1. Hazy opacity in the right lung, which may represent aspiration versus pleural effusion or hemorrhage. '
                '2. Mild pulmonary edema. '
                '3. No displaced rib fractures.'
            ),
            alternative=(
                'FINDINGS: A hazy opacity is present in the right lung. '
                'Moderate cardiomegaly is stable. '
                'Slight prominence of the pulmonary vasculature with cephalization and enlarged pulmonary arteries. '
                'There are no displaced rib fractures.\n\n'
                'IMPRESSION: 1. Hazy opacity in the right lung, which may represent aspiration versus pleural effusion or hemorrhage. '
                '2. Mild pulmonary edema. '
                '3. No displaced rib fractures.'
            )
        )
    ])
    constraints: List[str] = field(default_factory=lambda: [
        'Ensure that the union of clinical content in the Findings and Impression sections matches that of the original report. '
        'You should not (i) introduce any new details from other sections (e.g., Technique, Indication, Comparison), '
        'or (ii) omit any important clinical details from the Findings and Impression sections in the original report.'
    ])

'''
    Style axes corresponding to the "Completeness" group.

'''

@dataclass
class FindingsScope(Completeness):
    name: str = field(default='findings_scope')
    axis_type: str = field(default='absolute')
    description: str = field(default=(
        'Radiologists differ in how exhaustively they describe and normal and abnormal findings. '
        'Some exhaustively describe all normal and abnormal findings, while others selectively omit normal findings and '
        'focus on a minimal list of key abnormal findings. The minimal list of key abnormal findings typically includes '
        'abnormal findings that are most likely to be clinically relevant to the patient\'s condition and treatment.'
    ))
    variations: List[str] = field(default_factory=lambda: ['exhaustive', 'minimal'])
    variation_descriptions: Dict[str,str] = field(default_factory=lambda: dict(
        exhaustive='Exhaustively describes all normal and abnormal findings.',
        minimal='Selectively omits normal findings and focuses on a minimal list of key abnormal findings.'
    ))
    examples: List[Dict[str,str]] = field(default_factory=lambda: [
        dict(
            exhaustive=(
                'No significant technical limitations. '
                'Lung volumes are normal. '
                'No consolidative airspace disease. '
                'No pleural effusion or pneumothorax. '
                'No pulmonary nodule or mass noted. '
                'Pulmonary vasculature and the cardiomediastinal silhouette are within normal limits. '
                'Atherosclerosis in the thoracic aorta is noted.'
            ),
            minimal=(
                'No evidence of pneumothorax, pleural effusion, or pulmonary opacity. '
                'Cardiomediastinal silhouette is normal. '
                'Atherosclerosis in the thoracic aorta.'
            )
        ),
        dict(
            exhaustive=(
                'No exam quality limitations. '
                'Patient positioning is appropriate. '
                'Trachea is midline. '
                'Image of the abdomen is within normal limits. '
                'No radiopaque lines, tubes, or implants. '
                'Hila are normal. '
                'Lung volumes are normal. '
                'Heart size is normal. '
                'Mediastinal contours are unremarkable. '
                'Lungs are clear. '
                'No effusion or pneumothorax. '
                'Bones are intact.'
            ),
            minimal=(
                'Heart size is normal. '
                'No evidence of pneumothorax, pleural effusion, or pulmonary opacity. '
                'No acute abnormality.'
            )
        ),
        dict(
            exhaustive=(
                'QUALITY: Portable AP radiograph demonstrates low lung volumes with mild rotation. '
                'No motion artifact.\n'
                'The lungs are clear bilaterally. '
                'No pleural effusion or pneumothorax. '
                'The cardiomediastinal silhouette is within normal limits for this projection. '
                'Visualized osseous structures are unremarkable.'
            ),
            minimal=(
                'The lungs are clear bilaterally. '
                'No pleural effusion or pneumothorax. '
                'The cardiomediastinal silhouette is within normal limits. '
                'Visualized osseous structures are unremarkable.'
            )
        )
    ])
    constraints: List[str] = field(default_factory=lambda: [])

# FIXME: Obsolete; this has been merged with FindingsScope.
@dataclass
class TechnicalQuality(Completeness):
    name: str = field(default='technical_quality')
    axis_type: str = field(default='absolute')
    description: str = field(default=(
        'Some radiologists include comments on imaging quality, patient positioning, and/or technical issues or artifacts, '
        'while others omit such discussion entirely or include only brief references.'
    ))
    variations: List[str] = field(default_factory=lambda: ['discusses_technical_quality', 'does_not_discuss_technical_quality'])
    variation_descriptions: Dict[str,str] = field(default_factory=lambda: dict(
        discusses_technical_quality='Includes comments on imaging quality, patient positioning, technical issues or artifacts.',
        does_not_discuss_technical_quality='Does not include discussion of imaging quality, patient positioning, technical issues or artifacts.'
    ))
    examples: List[Dict[str,str]] = field(default_factory=lambda: [
        dict(
            discusses_technical_quality=(
                'QUALITY: Frontal and lateral views. The patient is slightly rotated to the right. '
                'Low lung volumes limit evaluation. Portable technique. No motion artifact.\n\n'
                'Small bilateral pleural effusions. Bibasilar opacities likely reflect atelectasis, '
                'though infection or aspiration cannot be excluded in the appropriate clinical setting. '
                'Mild pulmonary vascular congestion. Cardiomegaly. No pneumothorax. '
                'Atherosclerotic calcification of the thoracic aorta.'
            ),
            does_not_discuss_technical_quality=(
                'Small bilateral pleural effusions. Bibasilar opacities likely reflect atelectasis, '
                'though infection or aspiration cannot be excluded in the appropriate clinical setting. '
                'Mild pulmonary vascular congestion. Cardiomegaly. No pneumothorax. '
                'Atherosclerotic calcification of the thoracic aorta.'
            )
        ),
        dict(
            discusses_technical_quality=(
                'Portable AP radiograph demonstrates low lung volumes with mild rotation. '
                'The lungs are clear bilaterally. No pleural effusion or pneumothorax. '
                'The cardiomediastinal silhouette is within normal limits for this projection. '
                'Visualized osseous structures are unremarkable.'
            ),
            does_not_discuss_technical_quality=(
                'The lungs are clear bilaterally. No pleural effusion or pneumothorax. '
                'The cardiomediastinal silhouette is within normal limits. '
                'Visualized osseous structures are unremarkable.'
            )
        )
    ])
    constraints: List[str] = field(default_factory=lambda: [
        'Only perturb along this axis if the original report includes any discussion of technical quality.'
    ])

'''
    Style axes corresponding to the "Granularity" group.

'''

@dataclass
class AnatomicalGranularity(Granularity):
    name: str = field(default='anatomical_granularity')
    axis_type: str = field(default='absolute')
    description: str = field(default=(
        'Some radiologists exhaustively enumerate findings for each anatomical substructure separately, '
        'while others only enumerate high-level findings at the level of the organ or system.'
    ))
    variations: List[str] = field(default_factory=lambda: ['high', 'low'])
    variation_descriptions: Dict[str,str] = field(default_factory=lambda: dict(
        high='Describes findings for each anatomical substructure separately.',
        low='Describes findings at the level of the organ or system.'
    ))
    examples: List[Dict[str,str]] = field(default_factory=lambda: [
        dict(
            high=(
                'A pulmonary nodule is present in the right upper lobe anterior segment. '
                'A coronary artery stent is present in the left anterior descending artery. '
                'Small volume of fluid in the left posterior costophrenic sulcus. '
                'Right third through sixth healed rib fracture deformities. '
                'T9 anterior wedge compression fracture. '
                'Cholecystectomy clips are present.'
            ),
            low=(
                'Right lung pulmonary nodule. '
                'Prior coronary artery stenting. '
                'Small left pleural effusion. '
                'Healed rib fracture deformities. '
                'Thoracic compression fracture. '
                'Right upper quadrant surgical clips present.'
            )
        ),
        dict(
            high=(
                'The cardiac silhouette appears enlarged. '
                'Pulmonary vascular markings are prominent in the bilateral perihilar regions. '
                'There is mild interstitial prominence in the bilateral lower lungs. '
                'Trace fluid blunts the right posterior costophrenic sulcus. '
                'A left chest wall pacemaker is present with leads projecting over the cardiac silhouette. '
                'No focal airspace opacity. '
                'No pneumothorax.'
            ),
            low=(
                'Enlarged cardiac silhouette. '
                'Prominent pulmonary vasculature. '
                'Mild interstitial lung prominence. '
                'Trace right pleural effusion. '
                'Left-sided pacemaker present. '
                'No focal airspace opacity or pneumothorax.'
            )
        ),
        dict(
            high=(
                'The lungs are well expanded and clear without focal airspace opacity. '
                'The pulmonary vasculature is within normal limits. '
                'The costophrenic angles are sharp bilaterally. '
                'The diaphragms are smooth with normal contour. '
                'The cardiomediastinal silhouette is normal in size and shape. '
                'The trachea is midline. '
                'The hila are unremarkable. '
                'No pneumothorax is identified. '
                'The visualized osseous structures are intact.'
            ),
            low=(
                'Normal cardiomediastinal silhouette. '
                'Lungs are clear without consolidation, effusion, or pneumothorax. '
                'Pulmonary vasculature is within normal limits. '
                'No acute osseous abnormalities.'
            )
        )
    ])
    constraints: List[str] = field(default_factory=lambda: [])

# FIXME: Obsolete; this has been merged with AnatomicalGranularity.
@dataclass
class SpatialGranularity(Granularity):
    name: str = field(default='spatial_granularity')
    axis_type: str = field(default='absolute')
    description: str = field(default=(
        'Some radiologists describe locations with high precision, '
        'while others use broader regional descriptions that may not uniquely identify a single location.'
    ))
    variations: List[str] = field(default_factory=lambda: ['high', 'low'])
    variation_descriptions: Dict[str,str] = field(default_factory=lambda: dict(
        high='Describes locations with high precision.',
        low='Uses broader regional descriptions that may not uniquely identify a single location.'
    ))
    examples: List[Dict[str,str]] = field(default_factory=lambda: [
        dict(
            high='Opacity in the posterior basal segment of the left lower lobe.',
            low='Opacity in the left lung base.'
        ),
        dict(
            high='Pleural effusion layering along the right lateral costophrenic sulcus.',
            low='Right pleural effusion.'
        )
    ])
    constraints: List[str] = field(default_factory=lambda: [])

@dataclass
class QuantitativeGranularity(Granularity):
    name: str = field(default='quantitative_granularity')
    axis_type: str = field(default='absolute')
    description: str = field(default=(
        'Some radiologists include precise quantitative findings, while others include qualitative descriptions only.'
    ))
    variations: List[str] = field(default_factory=lambda: [
        'with_quantitative_findings', 'without_quantitative_findings'
    ])
    variation_descriptions: Dict[str,str] = field(default_factory=lambda: dict(
        with_quantitative_findings='Includes precise quantitative findings.',
        without_quantitative_findings='Includes qualitative descriptions only.'
    ))
    examples: List[Dict[str,str]] = field(default_factory=lambda: [
        dict(
            with_quantitative_findings='Left pleural effusion measuring 4 cm in height. Cardiothoracic ratio of 0.55.',
            without_quantitative_findings='Moderate left pleural effusion. Mild cardiomegaly.'
        ),
        dict(
            with_quantitative_findings='There is an endotracheal tube whose distal tip is seen 6.2 cm above the carina appropriately sited.',
            without_quantitative_findings='There is an endotracheal tube whose distal tip is appropriately sited above the carina.'
        ),
        dict(
            with_quantitative_findings=(
                'Moderate right apical pneumothorax has minimally decreased since yesterday. '
                'The maximum width at the apex measures 2.4 cm as compared to yesterday measuring 2.9 cm.'
            ),
            without_quantitative_findings=(
                'Moderate right apical pneumothorax has minimally decreased since yesterday.'
            )
        )
    ])
    constraints: List[str] = field(default_factory=lambda: [
        'Only modify along this axis if the original reference report includes precise quantitative findings, '
        'so as to avoid generating alternative reports with hallucinated quantitative findings.'
    ])

'''
    Style axes corresponding to the "Language" group.

'''

@dataclass
class Terminology(Language):
    name: str = field(default='terminology')
    axis_type: str = field(default='relative')
    description: str = field(default=(
        'Radiologists may use different terms for describing the same findings, '
        'depending on personal preference, training background, or subspecialty. '
        'These are near-synonymous phrasings that convey essentially the same clinical meaning.'
    ))
    variations: List[str] = field(default_factory=lambda: ['original', 'alternative'])
    variation_descriptions: Dict[str,str] = field(default_factory=lambda: dict(
        original='Retains original terminology.',
        alternative='Substitutes alternative terminology that conveys essentially the same clinical meaning.'
    ))
    examples: List[Dict[str,str]] = field(default_factory=lambda: [
        dict(
            original=(
                'No gross consolidation, atelectasis or infiltrate. '
                'No pleural fluid collection or pneumothorax. '
                'Cardiomediastinal silhouette is within normal limits.'
            ),
            alternative=(
                'The lungs are clear without evidence of focal airspace disease or consolidation. '
                'There is no evidence of pneumothorax or pleural effusion. '
                'The cardiac and mediastinal contours are within normal limits.'
            )
        ),
        dict(
            original='No pulmonary opacity.',
            alternative='Lungs are clear.'
        ),
        dict(
            original='Cardiomegaly.',
            alternative='Enlarged cardiac silhouette.'
        ),
        dict(
            original='Bones are intact.',
            alternative='Osseous structures are within normal limits.'
        ),
        dict(
            original='No evidence of intraabdominal free air.',
            alternative='Visualized upper abdomen is normal.'
        )
    ])
    constraints: List[str] = field(default_factory=lambda: [])

@dataclass
class Hedging(Language):
    name: str = field(default='hedging')
    axis_type: str = field(default='relative')
    description: str = field(default=(
        'Radiologists differ in how they express diagnostic uncertainty, even if their underlying interpretations are similar. '
        'Different phrases can convey similar levels of uncertainty but use different linguistic constructions.'
    ))
    variations: List[str] = field(default_factory=lambda: ['original', 'alternative'])
    variation_descriptions: Dict[str,str] = field(default_factory=lambda: dict(
        original='Retains original hedging language.',
        alternative='Substitutes alternative hedging language that conveys essentially the same level of uncertainty.'
    ))
    examples: List[Dict[str,str]] = field(default_factory=lambda: [
        dict(
            original='Peripheral wedge-shaped opacity is worrisome for pulmonary embolism.',
            alternative='Peripheral wedge-shaped opacity is concerning for pulmonary embolism.'
        ),
        dict(
            original='Opacity consistent with atelectasis.',
            alternative='Opacity likely represents atelectasis.'
        ),
        dict(
            original='Right lung opacity, query pneumonia.',
            alternative='Right lung opacity may represent pneumonia.'
        ),
        dict(
            original='Left lung opacity, query pneumonia.',
            alternative='Left lung opacity suggestive of pneumonia.'
        ),
        dict(
            original='Bilateral perihilar opacities could represent pulmonary edema.',
            alternative='Bilateral perihilar opacities suggest possible pulmonary edema.'
        )
    ])
    constraints: List[str] = field(default_factory=lambda: [
        'Ensure that the alternative version conveys the same level of uncertainty as the original version.'
    ])

@dataclass
class Negation(Language):
    name: str = field(default='negation')
    axis_type: str = field(default='absolute')
    description: str = field(default=(
        'Radiologists differ in how they express negation, using explicit statements or implicit language.'
    ))
    variations: List[str] = field(default_factory=lambda: ['explicit', 'implicit'])
    variation_descriptions: Dict[str,str] = field(default_factory=lambda: dict(
        explicit='Expresses negation explicitly.',
        implicit='Uses implicit language to express negation.'
    ))
    examples: List[Dict[str,str]] = field(default_factory=lambda: [
        dict(
            explicit='No pneumothorax. No pleural effusion.',
            implicit='Pleural spaces are clear.'
        ),
        dict(
            explicit='No focal consolidation. No pneumothorax. No pleural effusion.',
            implicit='The lungs are clear. Pleural spaces are normal.'
        ),
        dict(
            explicit='No mediastinal widening. No hilar adenopathy.',
            implicit='Mediastinal contours are within normal limits.'
        )
    ])
    constraints: List[str] = field(default_factory=lambda: [
        'Implicit versions must clearly convey the same clinical information as the explicit negations. '
        'Avoid ambiguous implicit statements that could be misinterpreted.'
    ])

def get_style_axis_cls(name: str) -> StyleAxis:
    '''Returns the corresponding style axis class for the given name.'''

    return dict(
        structure=Structure,
        section_assignment=SectionAssignment,
        findings_scope=FindingsScope,
        technical_quality=TechnicalQuality,
        anatomical_granularity=AnatomicalGranularity,
        spatial_granularity=SpatialGranularity,
        quantitative_granularity=QuantitativeGranularity,
        terminology=Terminology,
        hedging=Hedging,
        negation=Negation
    )[name]