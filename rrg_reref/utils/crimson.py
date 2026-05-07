import json
import csv
import os
import sys
import re
import gc
import contextlib
from typing import Optional

'''
    Adapted from the rajpurkarlab/CRIMSON repository.

    Version adapted from: https://github.com/rajpurkarlab/CRIMSON/blob/d3b9615

'''

# ---------------------------------------------------------------------------
# Regex patterns for cleaning radiology report text
# ---------------------------------------------------------------------------

# Collapse spaced-dot abbreviations produced by some models,
# e.g.  ". S. V. C." -> "SVC",  ". P. I. C. C" -> "PICC".
_SPACED_DOT_ABBREV_RE = re.compile(r'\. (?:[A-Z]\. ){1,}[A-Z]\.?')

# Hyphenated compounds split by dots, e.g. ". Port-. A-. Cath" -> "Port-A-Cath".
_HYPHENATED_DOT_RE = re.compile(r'\. (\w+(?:-\. \w+)+)')

# Stray/trailing dots: a lone '.' preceded by whitespace and followed by a
# non-whitespace character — these are tokenization artifacts, not real
# sentence-ending periods (which directly follow a word character).
_STRAY_DOT_RE = re.compile(r'(?<=\s)\. (?=\S)')

# Trailing stray dot(s) at the very end of a string, e.g. "leads. ." -> "leads."
_TRAILING_STRAY_DOT_RE = re.compile(r'\.\s*\.\s*$')

# ---------------------------------------------------------------------------
# Regex for fixing invalid JSON backslash escapes
# ---------------------------------------------------------------------------

# Fix invalid JSON backslash escapes (e.g. \_ or \L) that the model may
# produce when echoing back input text.  Valid JSON escapes
# (\" \\ \/ \b \f \n \r \t \uXXXX) are left untouched.
_INVALID_ESCAPE_RE = re.compile(r'\\(?!["\\\//bfnrtu])')

# ---------------------------------------------------------------------------
# Regex for removing orphan keys in JSON objects
# ---------------------------------------------------------------------------

# Matches a bare string value inside a JSON object — a quoted string preceded
# by a comma and followed by a comma or closing brace, but NOT followed by a
# colon.  Example:  ,"pred_old",  or  ,"pred_old"}
# The leading comma is consumed so removal doesn't leave double commas.
_ORPHAN_KEY_AFTER_COMMA_RE = re.compile(r',\s*"[^"]*"\s*(?=\s*[,}])(?!\s*:)')
# Orphan as the first item in an object:  {"orphan","key":"val"}
_ORPHAN_KEY_AT_START_RE = re.compile(r'(?<=\{)\s*"[^"]*"\s*,\s*(?=")')

# Malformed array-object hybrid:  ["":["R1"]  ->  ["R1"]
# The model starts an array, writes an empty-string "key" with a colon, then
# another array — as if confusing array and object syntax.
_MALFORMED_ARRAY_OBJ_RE = re.compile(r'\[""\s*:\s*\[')

# Missing opening quote on a JSON key:  ,attribute_errors":[  ->  ,"attribute_errors":[
# The model drops the opening " on a key name.  Only fires in structural
# positions: after a JSON value closer (] } " or digit) followed by a comma,
# or directly after an opening brace {.  This avoids false positives on
# commas inside string values like "persists, unchanged".
_MISSING_OPEN_QUOTE_RE = re.compile(
    r'([\]}"0-9]\s*,\s*|{\s*)([a-zA-Z_][a-zA-Z0-9_]*")'
)


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def _dedup_sentences(text):
    """Remove exact duplicate sentences from repetitive model predictions.

    Some models (CXRMate-RRG24) enter generation loops producing dozens of
    repeated sentences.  This preserves order of first occurrence and rejoins
    with '. '.
    """
    parts = text.split('. ')
    seen = set()
    unique = []
    for p in parts:
        stripped = p.strip()
        if stripped and stripped not in seen:
            seen.add(stripped)
            unique.append(p)
    return '. '.join(unique)


def clean_report_text(text):
    """Clean tokenization artifacts in radiology report text.

    Applied transformations (in order):
    1. Strip markdown bold markers (``**``)
    2. Collapse spaced-dot abbreviations ('. S. V. C.' -> 'SVC')
    3. Fix hyphenated compounds split by dots ('. Port-. A-. Cath' -> 'Port-A-Cath')
    4. Remove stray/trailing dots (tokenization artifacts where a lone '.'
       appears before a word, e.g. 'chest . X-ray' -> 'chest X-ray')
    5. Deduplicate exact-match repeated sentences
    """
    if not text:
        return text
    # 1. Markdown bold
    text = text.replace('**', '')
    # 2. Spaced-dot abbreviations
    text = _SPACED_DOT_ABBREV_RE.sub(
        lambda m: ''.join(re.findall(r'[A-Z]', m.group(0))), text
    )
    # 3. Hyphenated compounds
    text = _HYPHENATED_DOT_RE.sub(
        lambda m: m.group(0).replace('-. ', '-').lstrip('. '), text
    )
    # 4. Stray dots
    text = _STRAY_DOT_RE.sub('', text)
    text = _TRAILING_STRAY_DOT_RE.sub('.', text)
    # 5. Dedup
    text = _dedup_sentences(text)
    return text


# ---------------------------------------------------------------------------
# JSON response parsing
# ---------------------------------------------------------------------------

def _dedupe_keys(pairs):
    """JSON object_pairs_hook that keeps the first occurrence of duplicate keys."""
    result = {}
    for key, value in pairs:
        if key not in result:
            result[key] = value
    return result


def _loads(text):
    """json.loads with duplicate-key handling (keeps first occurrence)."""
    return json.loads(text, object_pairs_hook=_dedupe_keys)


def _is_structural_quote(text, i):
    """Check if the quote at position *i* is likely a JSON structural delimiter.

    Returns True for quotes that delimit JSON keys/values (after ``:``, before
    ``:`` etc.) and False for quotes that are part of prose inside a string
    value (e.g. the model quoting a phrase in an explanation).
    """
    before = text[i - 1] if i > 0 else ''
    after = text[i + 1] if i + 1 < len(text) else ''
    # Quotes right after : [ { are always structural (opening a value/key)
    if before in (':', '[', '{'):
        return True
    # Quotes right before : ] } are always structural (closing a key/value)
    if after in (':', ']', '}'):
        return True
    # For comma-adjacent quotes, check deeper context.
    # Structural ,"  is preceded by a JSON token closer: " ] } or digit
    # Text      ,"  is preceded by a word character (e.g. yesterday,")
    if before == ',':
        pre_comma = text[i - 2] if i >= 2 else ''
        return pre_comma in ('"', ']', '}') or pre_comma.isdigit()
    if after == ',':
        post_comma = text[i + 2] if i + 2 < len(text) else ''
        return post_comma in ('"', '[', '{') or post_comma.isdigit()
    return False


def _fix_unescaped_quotes(text, max_attempts=50):
    """Iteratively escape unescaped double-quotes that break JSON parsing.

    When the model uses literal " inside a JSON string value (e.g. to quote
    a phrase in an explanation), the JSON parser sees it as a string
    terminator.  This function locates each offending quote by parsing,
    catching the error position, searching backwards for a non-structural
    quote, escaping it, and retrying.
    """
    for _ in range(max_attempts):
        try:
            return _loads(text)
        except json.JSONDecodeError as e:
            pos = e.pos
            search_start = max(0, pos - 5)
            found = False
            for i in range(pos, search_start - 1, -1):
                if i < len(text) and text[i] == '"' and (i == 0 or text[i - 1] != '\\'):
                    if _is_structural_quote(text, i):
                        continue
                    text = text[:i] + '\\"' + text[i + 1:]
                    found = True
                    break
            if not found:
                raise
    raise json.JSONDecodeError("Max quote-fix attempts reached", text, 0)


def _fix_orphan_keys(text):
    """Remove bare string values inside JSON objects (orphan keys with no value).

    The model sometimes hallucinates partial key fragments (e.g. ``"pred_old"``)
    that appear as bare values in objects, producing invalid JSON like::

        {"ref_id":"R3","pred_old","pred_id":"P2"}

    This function strips them so the JSON becomes parsable.
    """
    text = _ORPHAN_KEY_AFTER_COMMA_RE.sub('', text)
    text = _ORPHAN_KEY_AT_START_RE.sub('', text)
    return text


def parse_json_response(response, batch_idx=None):
    """Parse model response as JSON, applying progressive fixes.

    Fix pipeline (each step only attempted if prior steps fail):
    1. Raw parse
    2. Remove orphan keys (bare strings in object context)
    3. Fix invalid backslash escapes (``\\L``, ``\\_``, etc.)
    4. Fix unescaped double-quotes inside string values

    Args:
        response: Raw model output string.
        batch_idx: Optional index for error context in batch evaluation.

    Returns:
        Parsed JSON object (dict).

    Raises:
        ValueError: If the response cannot be parsed as JSON.
    """
    # 0. Pre-parse text fixes
    # Escape curly/smart quotes — they always appear inside JSON string values
    response = response.replace('\u201c', '\\"').replace('\u201d', '\\"')
    response = response.replace('\u2018', "\\'").replace('\u2019', "\\'")
    # Fix malformed array-object hybrids:  ["":["R1"] -> ["R1"]
    response = _MALFORMED_ARRAY_OBJ_RE.sub('[', response)
    # Fix missing opening quote on keys:  ,attribute_errors"  ->  ,"attribute_errors"
    response = _MISSING_OPEN_QUOTE_RE.sub(r'\1"\2', response)
    # 1. Raw
    try:
        return _loads(response)
    except json.JSONDecodeError:
        pass
    # 2. Orphan keys (bare strings in object context, e.g. "pred_old")
    deorphaned = _fix_orphan_keys(response)
    try:
        return _loads(deorphaned)
    except json.JSONDecodeError:
        pass
    # 3. Invalid backslash escapes
    escaped = _INVALID_ESCAPE_RE.sub(r'\\\\', response)
    try:
        return _loads(escaped)
    except json.JSONDecodeError:
        pass
    # 4. Unescaped quotes (try on both raw and escape-fixed variants)
    for text in (response, escaped):
        try:
            return _fix_unescaped_quotes(text)
        except (json.JSONDecodeError, ValueError):
            pass
    ctx = f" for batch item {batch_idx}" if batch_idx is not None else ""
    raise ValueError(
        f"Failed to parse model response as JSON{ctx}\n"
        f"Response ({len(response)} chars):\n{response}"
    )

# ---------------------------------------------------------------------------
# vLLM model resolution
# ---------------------------------------------------------------------------

def resolve_model_for_vllm(model_name: str) -> str:
    """Resolve a HF model name to a local path suitable for vLLM.

    Works around a known issue where a stale
    ``model.safetensors.index.json`` on HuggingFace Hub references
    sharded weight files that don't exist alongside a consolidated
    ``model.safetensors``.  vLLM's weight loader trusts the index and
    ends up finding zero matching files.

    If the model is already a local path, it is returned unchanged.
    """
    if os.path.isdir(model_name):
        return model_name

    from huggingface_hub import snapshot_download
    # MODIFIED HERE
    try:
        snapshot_dir = snapshot_download(
            model_name,
            cache_dir=os.getenv('HF_HUB_CACHE'),
            local_files_only=True
        )
    except Exception:
        snapshot_dir = snapshot_download(
            model_name,
            cache_dir=os.getenv('HF_HUB_CACHE')
        )

    index_path = os.path.join(snapshot_dir, "model.safetensors.index.json")
    consolidated_path = os.path.join(snapshot_dir, "model.safetensors")
    if os.path.exists(index_path) and os.path.exists(consolidated_path):
        # Check if the index references files that don't actually exist
        with open(index_path) as f:
            index = json.load(f)
        referenced = set(index.get("weight_map", {}).values())
        missing = [
            fn for fn in referenced
            if not os.path.exists(os.path.join(snapshot_dir, fn))
        ]
        if missing:
            # Build a clean local copy without the stale index
            import tempfile
            clean_dir = os.path.join(
                tempfile.gettempdir(),
                f"vllm-{model_name.replace('/', '--')}",
            )
            os.makedirs(clean_dir, exist_ok=True)
            for entry in os.listdir(snapshot_dir):
                if entry == "model.safetensors.index.json":
                    continue
                dst = os.path.join(clean_dir, entry)
                if not os.path.exists(dst):
                    src = os.path.realpath(os.path.join(snapshot_dir, entry))
                    os.symlink(src, dst)
            return clean_dir

    return snapshot_dir

"""
Composable prompt parts for CRIMSON evaluation.

"""

OBJECTIVE = """\
Objective:

Evaluate the accuracy of predicted chest X-ray (CXR) findings compared to reference (ground truth) findings.
Only evaluate positive findings, not normal findings. Focus on clinical accuracy."""

TASK_STEPS = """\
Task:
Perform this evaluation in TWO STEPS:

STEP 1: Extract All Positive Findings
For BOTH reference and predicted reports, identify each ABNORMAL/POSITIVE finding exactly as written, including all descriptors, locations, severity terms, and measurements. List each finding as it appears in the original text.
If a single sentence contains multiple distinct findings, split them into separate entries (e.g., "bilateral pleural effusions and bibasilar atelectasis" should be two findings: "bilateral pleural effusions" and "bibasilar atelectasis").
When splitting, carry over shared qualifiers (certainty terms like "suggestive of", "consistent with", "suspicious for", as well as shared descriptors like location or severity) to EACH resulting finding so no context is lost.
For each finding (both reference and predicted), assign a clinical_significance level.
Assign each reference finding a sequential ID starting with "R" (R1, R2, R3, ...) and each predicted finding a sequential ID starting with "P" (P1, P2, P3, ...).

IMPORTANT: Include all abnormal findings in the reference_findings and predicted_findings lists (not just clinically significant ones). Do not include normal findings.

STEP 2: Compare and Classify Errors
Using the assigned IDs, match findings between reference and predicted, then identify errors in these 3 categories."""

CLINICAL_SIGNIFICANCE_LEVELS = """\
Clinical significance levels (used for findings):
    - "urgent": Requires immediate action / life-threatening
    - "actionable_not_urgent": Would change treatment plan but not immediately critical
    - "not_actionable_not_urgent": Minor discrepancy with minimal clinical impact (but still worth noting)
    - "benign_expected": Benign/expected finding with no clinical relevance (age-related changes)"""

SIGNIFICANCE_APPLICATION = """\
Apply these significance levels as follows:
- reference_findings and predicted_findings: classify each FINDING's clinical significance
- missing_findings (b) and false_findings (a): Do NOT include clinical_significance.
  Their weights are derived automatically from reference_findings and predicted_findings respectively."""

ATTRIBUTE_SEVERITY_LEVELS = """\
Attribute error severity (used ONLY for attribute_errors):
    - "significant": The attribute difference affects treatment or could lead to mismanagement
    - "negligible": The attribute difference is clinically insignificant and would not affect treatment"""

ATTRIBUTE_ERROR_INSTRUCTIONS = """\
For attribute_errors, also specify which error types apply from: ["location", "severity", "descriptor", "measurement", "certainty", "unspecific", "overinterpretation", "temporal"]
For each attribute_error, provide the ref_id and pred_id of the matched pair, and a brief "explanation" describing what the discrepancy is.

IMPORTANT: A matched finding pair can appear in attribute_errors MULTIPLE TIMES if it has multiple distinct issues.
Each attribute error should be a separate entry with its own severity classification and explanation.
For example, if a finding has both a location error and a severity error, create two separate entries with the same ref_id and pred_id."""

ERROR_CATEGORIES = """\
Error categories:
    a) False findings: Positive findings in predicted that are NOT present in reference
    b) Missing findings: Positive findings in reference but MISSING from predicted
    c) Attribute errors: Matched finding has incorrect or incomplete attributes including:
        - Location/laterality: Errors in anatomical location or side specification (left vs right lung, different lobes). 
        - Severity/extent: Errors in the degree or size characterization of the finding (small vs large, mild vs severe, minimal vs moderate)
        - Morphological descriptors: Errors in the shape, appearance, or characteristics of the finding (well-defined vs irregular margins, solid vs ground-glass density)
        - Measurements: Errors in quantitative size specifications (3 cm vs 1.5 cm for lesions, nodule diameter discrepancies)
        - Certainty: Errors in the confidence level of the finding (definite finding vs "possible" or "suspicious for")
        - Unspecific: Predicted finding is too vague or non-specific compared to the reference (reference says "pneumothorax" but predicted says "lucency")
        - Overinterpretation: Predicted finding makes an unsupported diagnostic leap beyond the imaging finding (reference says "opacity" but predicted says "mass")
        - Temporal/comparison: Errors in temporal descriptors or comparison with prior studies (missing "new", "worsening", "resolved", "stable", "unchanged")"""

MATCHING_CRITERIA = """\
Matching criteria:
- A finding matches if the core finding is present in both (e.g., "pleural effusion", "pneumothorax")
- A finding still matches even if location differs (location mismatch is an attribute error, not a match failure)
- One reference finding CAN match multiple predicted findings (one-to-many). For example, if R1 is "bilateral pleural effusions" and the predicted report splits it into P1 "left pleural effusion" and P2 "right pleural effusion", then both {{R1, P1}} and {{R1, P2}} are valid matches.
- For matched findings, check all attributes (location, severity, descriptors, measurements, certainty, unspecific, overinterpretation, temporal)
- If any attributes differ, add separate entries to attribute_errors for each distinct issue, referencing both the reference ID (ref_id) and predicted ID (pred_id)
- Any reference finding whose ID does not appear in matched_findings MUST be in missing_findings
- Any predicted finding whose ID does not appear in matched_findings MUST be in false_findings"""

OUTPUT_FORMAT = """\
Output Format:
Return ONLY valid JSON (no markdown, no explanation) in this exact format:
{{
    "reference_findings": [
        {{"id": "R1", "finding": "<finding text>", "clinical_significance": "urgent|actionable_not_urgent|not_actionable_not_urgent|benign_expected"}},
        ...
    ],
    "predicted_findings": [
        {{"id": "P1", "finding": "<finding text>", "clinical_significance": "urgent|actionable_not_urgent|not_actionable_not_urgent|benign_expected"}},
        ...
    ],
    "matched_findings": [
        {{"ref_id": "R1", "pred_id": "P1"}},
        {{"ref_id": "R1", "pred_id": "P2"}},
        ...
    ],
    "errors": {{
        "false_findings": [
            "P3",
            ...
        ],
        "missing_findings": [
            "R2",
            ...
        ],
        "attribute_errors": [
            {{"ref_id": "R1", "pred_id": "P1", "severity": "significant|negligible", "error_types": ["location|severity|descriptor|measurement|certainty|unspecific|overinterpretation|temporal"], "explanation": "<brief description of the discrepancy>"}},
            ...
        ]
    }}
}}"""

CONTEXT_GUIDELINES = """\
Context guidelines:

Age-appropriate findings:
- Elderly patients (≥65): Expected degenerative changes such as aortic calcification, vascular tortuosity, degenerative spine changes, unfolding of the aorta, and sternal wire/median sternotomy findings (if chronic) should be classified under benign_expected — UNLESS the finding is directly related to the clinical indication.

Indication-finding concordance:
- A finding's clinical significance depends on whether it is EXPECTED given the indication and age, or UNEXPECTED and therefore more clinically important.
- The SAME finding can be urgent, actionable, or incidental depending on the clinical context:
    - Pleural effusion in a known heart failure patient with dyspnea: actionable_not_urgent (expected clinical course)
    - Pleural effusion in a young patient with palpitations: actionable_not_urgent or urgent (unexpected, needs workup)
    - Rib fractures in a young child with chest pain: urgent (concern for non-accidental injury)
    - Rib fractures in an adult post-RTA: actionable_not_urgent (trauma workup, possible hemothorax)
    - Rib fractures in an elderly patient with known old trauma: not_actionable_not_urgent or benign_expected (chronic/expected)
- When the indication directly explains a finding (e.g., "known heart failure" + cardiomegaly), the finding may be less actionable than the same finding without a known cause.
- When the indication does NOT explain a finding, it is generally more clinically significant and warrants higher classification.

Post-procedural, post-surgical context, and expectation of complication:
- If the indication references a recent procedure (line placement, intubation, chest tube insertion, post-CABG, post-thoracotomy), expected post-procedural findings should generally be classified as not_actionable_not_urgent:
- When a complication is clinically anticipated or known to be a high-probability outcome, its presence carries lower urgency than the same complication appearing unexpectedly:
    - Post-intubation aspiration (known risk): actionable_not_urgent even if new
    - Pneumothorax after central line placement (known procedural risk): actionable_not_urgent if small and anticipated
    - Mediastinal free air (Pneumomediastinum) after recent sternotomy or esophageal procedure: actionable_not_urgent if small and stable (expected post-surgical finding, but requires follow-up imaging)
    - New pleural effusion after cardiac surgery (expected post-op reaction): not_actionable_not_urgent if small and stable
- However, post-procedural COMPLICATIONS and UNEXPECTED findings remain at their usual or higher severity:
    - Effusion in a young, otherwise healthy patient: actionable_not_urgent (unexpected, needs workup)
    - Mediastinal widening without known prior dissection or trauma: urgent

Pediatric considerations:
- In young children, thymic shadow is normal and should not be classified as cardiomegaly or mediastinal widening.
- Unexpected findings in children (fractures, effusions) without a clear traumatic mechanism should be weighted higher due to concern for non-accidental injury.
- Lobar atelectasis in a young child may represent foreign body aspiration and should be classified as urgent, especially if clinical context suggests acute onset or no other explanation.
- Pediatric cardiac and mediastinal silhouette norms differ from adults — use age-appropriate thresholds.

Trauma context:
- When the indication mentions trauma, fall, MVC/RTA, or assault, fractures and pneumothorax should be classified at least actionable_not_urgent.
- Associated findings such as pleural effusion in trauma should raise concern for hemothorax (actionable_not_urgent or urgent depending on size).
- In elderly patients with a history of old/remote trauma, stable chronic findings (healed fractures, old deformities) are not_actionable_not_urgent or benign_expected.
"""

IMPORTANT_NOTES = """\
IMPORTANT NOTES:
- If a category has no items, use an empty array [].
- A single reference finding ID (R1, R2, ...) may appear in multiple matched_findings pairs (one-to-many matching). Any reference ID that does not appear in ANY matched pair MUST be in missing_findings.
- Every predicted finding ID (P1, P2, ...) must appear in EXACTLY ONE place: either in matched_findings or in false_findings.
- A missing finding CANNOT also appear in attribute_errors. If a reference finding is completely absent from the predicted findings, its ID goes ONLY in missing_findings.
- If a finding is classified as unspecific (in attribute_errors with "unspecific" error type), it is NOT a missing finding. Unspecific findings are PRESENT in the predicted report but too vague - they should appear in predicted_findings list and NOT in missing_findings."""

SIGNIFICANCE_EXAMPLES = """\
Examples of typical classifications (use guidelines and clinical judgment for patient-specific context):

Urgent findings (require immediate action):
    - Pneumothorax (especially tension pneumothorax)
    - Mediastinal widening 
    - Pneumomediastinum
    - Pneumoperitoneum
    - Mediastinal shift
    - Foreign Bodies 
    - Large pericardial effusion / Cardiomegaly
    - Malpositioned devices (e.g., endotracheal tube in right mainstem bronchus, misplaced central venous catheter, malpositioned chest tube)

Actionable non-urgent findings (change treatment but not emergent):
    - Moderate pleural effusion
    - Pulmonary nodules requiring follow-up
    - Pneumonia / Consolidation / Airspace disease or opacity / Infiltrates
    - Pulmonary edema (prominent perihilar / bronchial / vascular markings)
    - Mild to moderate cardiomegaly
    - Small non-tension pneumothorax
    - Masses
    - Fractures
    - Adenopathy

Not actionable, not urgent findings (minor clinical discrepancy, but still documented):
    - Stable chronic findings (old granulomas, chronic scarring)
    - Atelectasis
    - Trace pleural thickening / effusion 
    - Osteopenia
    - Appropriately positioned support devices and lines (NG tube, ETT, central lines, chest tubes in correct position) 

Benign/expected findings (no clinical relevance, age-related changes):
    - Age-appropriate atherosclerotic calcifications (e.g., aortic calcification in elderly)
    - Age-appropriate vascular tortuosity
    - Age-appropriate degenerative spine changes (in elderly patients)
    - Incidental stable benign findings (healed rib fracture, granuloma mentioned as "unchanged")
    - Expected physiologic findings (gastric air, bowel gas pattern)"""

ATTRIBUTE_SEVERITY_GUIDELINES = """\
Specific guidelines for attribute error severity:

Location/laterality errors:
    - NOT an error: If the reference finding does not specify a location, the predicted report adding location detail is acceptable and should NOT be flagged as a location error.
    - Significant: Wrong lung (left vs right), non-adjacent lobes (e.g., right upper lobe vs right lower lobe), non-adjacent zones (e.g., upper zone vs lower zone)
    - Negligible: Adjacent/touching lobes (e.g., right upper vs right middle, right middle vs right lower, left upper vs left lower), adjacent zones (e.g., upper vs mid zone, mid vs lower zone), minor positional differences within the same lobe (e.g., "apical" vs "lateral")

Severity/extent errors:
    - Significant: Changes clinical urgency or management (e.g., "small" vs "large", "mild" vs "severe", "minimal" vs "moderate", "simple" vs "tension")
    - Negligible: Stylistic differences that do not affect management (e.g., "small" vs "tiny")

Morphological descriptor errors:
    - Significant: Changes diagnostic considerations or management (e.g., "well-defined" vs "irregular margins", "solid" vs "ground-glass density", "simple" vs "complex")
    - Negligible: Stylistic differences that do not affect interpretation (e.g., "opacity" vs "opacification", "density" vs "opacification")

Measurement errors:
    - For nodules (< 3 cm): Significant if difference exceeds margin of error (for nodules < 6mm use 2mm margin; for nodules ≥ 6mm use 4mm margin). Negligible if within margin (e.g., "4mm" vs "5mm", "8mm" vs "11mm").
    - For masses (≥ 3 cm): Significant if difference exceeds 20% of the reference size. Negligible if within 20%. However, if the measurement error reclassifies a mass as a nodule (crossing below 3 cm), it is always significant regardless of percentage.

Certainty errors:
    - Significant: Adding or removing hedging language that changes management decisions (e.g., definite "pneumothorax" vs "possible pneumothorax" when clinical urgency would differ)
    - Negligible: Minor differences in hedging that do not affect management (e.g., "likely" vs "probable", "suggestive of" vs "compatible with")

Unspecific and Overinterpretation errors:
    Unspecific means the candidate is vaguer than the reference, overinterpretation means the candidate is more specific/diagnostic than the reference.
    - Significant: The specificity gap changes clinical management — either the vague term would not prompt the same workup, or the more specific term makes an unsupported diagnostic leap
        - "pneumothorax" and "lucency"
        - "pneumonia" and "opacity"
        - "mass" or "tumor" or "malignancy" and "opacity"
        - "pleural effusion" and "costophrenic angle blunting"
        - "consolidation" and "pulmonary edema" or "hemorrhage"
        - "lucency" and "pneumothorax" or "emphysema" or "cavitation"
        - "ground-glass opacity" and "pulmonary hemorrhage" or "early ARDS"
        - "nodule" and "granuloma" or "carcinoma"
        - "mediastinal widening" and "aortic dissection" or "lymphoma" or "mediastinal mass"
        - "pleural thickening" and "mesothelioma" or "empyema"
    - Negligible: The terms are commonly accepted as interchangeable, describe the same pathophysiological process at a slightly different point on its spectrum, and management would not meaningfully change
        - "consolidation" and "pneumonia"
        - "infiltrate" and "pneumonia"
        - "airspace opacity" and "airspace disease"
    - Note: If the overinterpreted or unspecific finding is in the same location as the reference finding, treat it as a MATCH with an attribute error, NOT as a false finding

Temporal/comparison errors:
    - Significant: Missing or incorrect "new", "worsening", "increasing", "resolved", or falsely adding these when not indicated (changes clinical urgency)
    - Negligible: Missing "stable", "unchanged", "chronic", "longstanding", or minor phrasing differences
    - Note: Do not penalize absence of temporal descriptors if no prior study is referenced in either report"""

def build_prompt(
    reference_findings,
    predicted_findings,
    patient_context=None,
    include_significance_examples=True,
    include_attribute_guidelines=True,
    include_context_guidelines=True,
):
    """Build the full evaluation prompt from composable parts.

    Args:
        reference_findings: Ground truth findings (string or list of strings)
        predicted_findings: Predicted findings (string or list of strings)
        patient_context: Optional dict with patient context (age, indication, etc.)
        include_significance_examples: Include examples of typical classifications
        include_attribute_guidelines: Include specific guidelines for attribute severity
        include_context_guidelines: Include context guidelines (only used when patient_context is provided)

    Returns:
        Formatted prompt string
    """
    # Convert lists to strings if needed
    if isinstance(reference_findings, list):
        reference_findings = " ".join(reference_findings)
    if isinstance(predicted_findings, list):
        predicted_findings = " ".join(predicted_findings)

    # Patient context block
    context_str = ""
    if patient_context:
        context_str = "Patient Context:\n"
        for key, value in patient_context.items():
            context_str += f"    {key}: {value}\n"

    # Assemble prompt sections
    sections = [
        OBJECTIVE,
        context_str,
        f"Reference Findings (Ground Truth):\n{reference_findings}",
        f"Predicted Findings (Candidate):\n{predicted_findings}",
        TASK_STEPS,
        CLINICAL_SIGNIFICANCE_LEVELS,
    ]

    if include_significance_examples:
        sections.append(SIGNIFICANCE_EXAMPLES)

    sections.append(SIGNIFICANCE_APPLICATION)
    sections.append(ATTRIBUTE_SEVERITY_LEVELS)

    sections.append(ERROR_CATEGORIES)
    sections.append(MATCHING_CRITERIA)
    sections.append(ATTRIBUTE_ERROR_INSTRUCTIONS)

    if include_attribute_guidelines:
        sections.append(ATTRIBUTE_SEVERITY_GUIDELINES)

    if include_context_guidelines and patient_context:
        sections.append(CONTEXT_GUIDELINES)

    sections.append(OUTPUT_FORMAT)
    sections.append(IMPORTANT_NOTES)

    return "\n\n".join(s for s in sections if s)

def _fix_gemma3_rope_config(config):
    """Workaround for a vLLM (<0.15) + transformers v4 bug where top-level
    rope_theta is injected into nested rope_parameters, breaking the
    is_rope_parameters_nested check for Gemma 3 hybrid attention models.

    The nested sub-dicts already carry their own rope_theta values, so the
    top-level attribute is redundant and safe to clear.
    """
    rp = getattr(config, "rope_parameters", None)
    if isinstance(rp, dict) and any(isinstance(v, dict) for v in rp.values()):
        for attr in ("rope_theta", "partial_rotary_factor",
                     "original_max_position_embeddings"):
            if getattr(config, attr, None) is not None:
                setattr(config, attr, None)
    return config

# Main CRIMSONScore class
class CRIMSONScore:
    """
    Calculate CRIMSON scores using various models (GPT, etc.) to evaluate
    radiology report generation quality.
    """
    
    DEFAULT_HF_MODEL = "CRIMSONScore/medgemma-4b-it-crimson"
    DEFAULT_MAX_NEW_TOKENS = 8192

    def __init__(
        self,
        api="hf",
        model_name=None,
        device=None,
    ):
        """
        Initialize CRIMSON scorer.

        Args:
            api: Model type to use ("hf", "huggingface", "openai").
                 Defaults to "hf".
            model_name: Specific model name / deployment name.
                       Defaults to "CRIMSONScore/medgemma-4b-it-crimson" for HF.
                       For OpenAI, defaults to "gpt-5.2".
            device: Device to use for HuggingFace models. Defaults to "cuda".
        """
        self.api = api

        if api == "openai":
            from openai import OpenAI
            self.model_name = model_name or "gpt-5.2"
            self.client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
            )
        elif api in ["huggingface", "hf"]:
            import torch
            from transformers import pipeline
            self.model_name = model_name or self.DEFAULT_HF_MODEL
            self.device = device or "cuda"
            self.torch_dtype = torch.bfloat16

            print(f"Loading HuggingFace model: {self.model_name}")
            self.pipe = pipeline(
                "text-generation",
                model=self.model_name,
                dtype=self.torch_dtype,
                device_map="auto",
            )
            # Left-pad for efficient batch generation with decoder-only models
            if self.pipe.tokenizer.padding_side != "left":
                self.pipe.tokenizer.padding_side = "left"
            # If the model has a generation_config.json, use it instead of
            # injecting our own generation kwargs at inference time.
            self._has_generation_config = not self.pipe.model.generation_config._from_model_config
            print(f"Model loaded.")
        elif api == "vllm":
            from vllm import LLM, SamplingParams
            self.model_name = model_name or self.DEFAULT_HF_MODEL
            model_path = resolve_model_for_vllm(self.model_name)
            print(f"Loading vLLM model: {self.model_name}")
            self.llm = LLM(
                model=model_path,
                dtype="bfloat16",
                max_model_len=8192,
                tensor_parallel_size=1, # MODIFIED HERE
                gpu_memory_utilization=0.5, # MODIFIED HERE
                trust_remote_code=True, # MODIFIED HERE
                hf_overrides=_fix_gemma3_rope_config, # MODIFIED HERE
            )
            self.sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=self.DEFAULT_MAX_NEW_TOKENS,
            )
            print(f"vLLM model loaded.")
        else:
            raise ValueError(f"Unsupported model: {api}. Use 'openai', 'huggingface', 'hf', or 'vllm'.")

    def close(self):
        """Release the underlying vLLM engine (GPU memory, subprocess, NCCL).

        Safe to call multiple times; a no-op for non-vLLM APIs.
        """
        if getattr(self, 'api', None) != 'vllm':
            return
        if getattr(self, 'llm', None) is None:
            return

        llm = self.llm
        self.llm = None

        with contextlib.suppress(Exception):
            engine = getattr(llm, 'llm_engine', None)
            if engine is not None and hasattr(engine, 'shutdown'):
                engine.shutdown()

        del llm

        with contextlib.suppress(Exception):
            from vllm.distributed import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )
            destroy_model_parallel()
            destroy_distributed_environment()

        gc.collect()
        with contextlib.suppress(Exception):
            import torch
            torch.cuda.empty_cache()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def _chat_completion(self, prompt):
        """
        Make a chat completion request (similar to process_reports.py pattern).
        
        Args:
            prompt: The prompt to send to the model
            
        Returns:
            The model's response text
        """
        if self.api == "openai":
            messages = [
                {"role": "system", "content": "You are an expert radiology evaluator that assesses the accuracy of radiology reports."},
                {"role": "user", "content": prompt}
            ]
            
            kwargs = {
                "model": self.model_name,
                "messages": messages,
                "seed": 42,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            }
            
            completion = self.client.chat.completions.create(**kwargs)
            return completion.choices[0].message.content
        
        elif self.api in ["huggingface", "hf"]:
            messages = [
                {"role": "system", "content": "You are an expert radiology evaluator that assesses the accuracy of radiology reports."},
                {"role": "user", "content": prompt},
            ]

            if self._has_generation_config:
                outputs = self.pipe(messages, generation_config=self.pipe.model.generation_config)
            else:
                outputs = self.pipe(
                    messages,
                    max_new_tokens=self.DEFAULT_MAX_NEW_TOKENS,
                    max_length=None,
                    do_sample=False,
                )
            return outputs[0]['generated_text'][-1]['content']

        elif self.api == "vllm":
            messages = [
                {"role": "system", "content": "You are an expert radiology evaluator that assesses the accuracy of radiology reports."},
                {"role": "user", "content": prompt},
            ]
            outputs = self.llm.chat([messages], sampling_params=self.sampling_params, use_tqdm=False)
            return outputs[0].outputs[0].text

    def _chat_completion_batch(self, prompts, batch_size=1):
        """
        Make completion requests for multiple prompts.

        For HuggingFace models, prompts are processed in local GPU batches of
        batch_size. OpenAI is not supported.

        Args:
            prompts: List of prompt strings
            batch_size: Number of prompts to process per forward pass. Defaults to 1.

        Returns:
            List of response strings in the same order as prompts
        """
        if not prompts:
            return []

        if self.api == "vllm":
            messages_batch = [
                [
                    {"role": "system", "content": "You are an expert radiology evaluator that assesses the accuracy of radiology reports."},
                    {"role": "user", "content": prompt},
                ]
                for prompt in prompts
            ]
            # vLLM handles batching internally with continuous batching
            outputs = self.llm.chat(messages_batch, sampling_params=self.sampling_params, use_tqdm=True)
            return [out.outputs[0].text for out in outputs]

        if self.api not in ["huggingface", "hf"]:
            raise ValueError(f"Batch not supported yet for API: {self.api}")

        responses = []
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start:start + batch_size]
            messages_batch = [
                [
                    {"role": "system", "content": "You are an expert radiology evaluator that assesses the accuracy of radiology reports."},
                    {"role": "user", "content": prompt},
                ]
                for prompt in chunk
            ]
            if self._has_generation_config:
                outputs = self.pipe(messages_batch, generation_config=self.pipe.model.generation_config, batch_size=batch_size)
            else:
                outputs = self.pipe(
                    messages_batch,
                    max_new_tokens=self.DEFAULT_MAX_NEW_TOKENS,
                    max_length=None,
                    do_sample=False,
                    batch_size=batch_size,
                )

            # Handle list of dicts or list of lists of dicts
            if isinstance(outputs, list) and len(outputs) > 0 and isinstance(outputs[0], list):
                batch_responses = [out[0]['generated_text'][-1]['content'] for out in outputs]
            else:
                if batch_size == 1:
                    batch_responses = [outputs[0]['generated_text'][-1]['content']]
                else:
                    batch_responses = [output['generated_text'][-1]['content'] for output in outputs]
            responses.extend(batch_responses)

        return responses
    
    def _build_evaluation_prompt(
        self, 
        reference_findings, 
        predicted_findings,
        patient_context=None,
        include_guidelines=True,
    ):
        """
        Build the evaluation prompt for CRIMSON scoring.
        Delegates to prompt_parts.build_prompt for composable prompt assembly.
        
        Args:
            reference_findings: Ground truth findings (string or list of strings)
            predicted_findings: Predicted findings (string or list of strings)
            patient_context: Optional patient context (age, indication, etc.)
            include_guidelines: If False, disables significance examples,
                attribute severity guidelines, and context guidelines.
            
        Returns:
            Formatted prompt string
        """

        # For the MedGemmaCRIMSON, exclude guidelines. the model was trained without them.
        if self.api in ["huggingface", "hf", "vllm"] and self.model_name == self.DEFAULT_HF_MODEL:
            include_guidelines = False
        
        return build_prompt( # MODIFIED HERE
            reference_findings,
            predicted_findings,
            patient_context=patient_context,
            include_significance_examples=include_guidelines,
            include_attribute_guidelines=include_guidelines,
            include_context_guidelines=include_guidelines,
        )

    @staticmethod
    def _parse_json_response(response, batch_idx=None):
        """Parse model response as JSON, fixing invalid backslash escapes."""
        return parse_json_response(response, batch_idx=batch_idx)

    def evaluate(
        self,
        reference_findings,
        predicted_findings,
        patient_context=None,
        include_guidelines=True,
    ):
        """
        Evaluate findings and calculate CRIMSON score.
        
        Args:
            reference_findings: Ground truth findings
            predicted_findings: Model-generated findings  
            patient_context: Optional patient context
            include_guidelines: If False, disables guidelines in the prompt
            
        Returns:
            Dictionary containing:
                - raw_evaluation: Raw model output with errors by category
                - error_counts: Counts of each error type
                - metrics: Individual metric components
                - crimson_score: Final CRIMSON score
        """
        # Get evaluation from model
        prompt = self._build_evaluation_prompt(
            reference_findings, 
            predicted_findings, 
            patient_context,
            include_guidelines=include_guidelines,
        )
        
        response = self._chat_completion(prompt)
        evaluation = self._parse_json_response(response)
        
        # Calculate CRIMSON score
        crimson_result = self._calculate_crimson(evaluation)
        
        return crimson_result

    def evaluate_batch(
        self,
        reference_findings_list,
        predicted_findings_list,
        patient_contexts: Optional[list] = None,
        include_guidelines=True,
        batch_size=1,
    ):
        """
        Evaluate multiple report pairs in one call.

        HuggingFace path uses local batched inference with the given batch_size.
        OpenAI is not supported for batch evaluation.

        Args:
            reference_findings_list: List of ground truth findings
            predicted_findings_list: List of model-generated findings
            patient_contexts: Optional list of patient contexts, aligned by index
            include_guidelines: If False, disables guidelines in the prompt
            batch_size: Number of prompts per forward pass (HF only). Defaults to 1.

        Returns:
            List of CRIMSON result dictionaries (same format as evaluate)
        """
        if len(reference_findings_list) != len(predicted_findings_list):
            raise ValueError("reference_findings_list and predicted_findings_list must have the same length")

        if patient_contexts is None:
            patient_contexts = [None] * len(reference_findings_list)
        elif len(patient_contexts) != len(reference_findings_list):
            raise ValueError("patient_contexts must be None or have the same length as input lists")

        prompts = [
            self._build_evaluation_prompt(
                reference_findings,
                predicted_findings,
                patient_context,
                include_guidelines=include_guidelines,
            )
            for reference_findings, predicted_findings, patient_context in zip(
                reference_findings_list,
                predicted_findings_list,
                patient_contexts,
            )
        ]

        responses = self._chat_completion_batch(prompts, batch_size=batch_size)

        results = []
        for idx, response in enumerate(responses):
            try:
                evaluation = self._parse_json_response(response, batch_idx=idx)
                results.append(self._calculate_crimson(evaluation))
            except Exception as e:
                # MODIFIED HERE
                #print(f"  [warn] sample {idx} failed: {e}")
                print(f"  [warn] sample {idx} failed")
                results.append(None)

        n_fail = sum(1 for r in results if r is None)
        if n_fail:
            print(f"  [warn] {n_fail}/{len(results)} samples failed to parse/score")

        return results
    
    def _calculate_crimson(
        self,
        evaluation
    ):
        """
        Calculate CRIMSON score from evaluation results.
        
        Args:
            evaluation: Parsed evaluation with matched findings and errors
            
        Returns:
            Dictionary with scores and metrics
        """
        errors = evaluation.get("errors", {})
        matched = evaluation.get("matched_findings", [])
        reference_findings_list = evaluation.get("reference_findings", [])
        predicted_findings_list = evaluation.get("predicted_findings", [])
        
        # Weight mapping for clinical significance
        significance_weights = {
            "urgent": 1.0,
            "actionable_not_urgent": 0.5,
            "not_actionable_not_urgent": 0.25,
            "benign_expected": 0.0,
        }
        
        # Weight mapping for attribute error severity
        attribute_severity_weights = {
            "significant": 0.5,
            "negligible": 0.0,
        }
        
        def calculate_weighted_count(error_list, weights=significance_weights, key="clinical_significance"):
            """Calculate weighted count based on significance/severity."""
            return sum(weights.get(error.get(key, ""), 0.25) for error in error_list)
        
        ref_weight_by_id = {
            ref["id"]: significance_weights.get(ref.get("clinical_significance", ""), 0.25)
            for ref in reference_findings_list
        }
        pred_weight_by_id = {
            pred["id"]: significance_weights.get(pred.get("clinical_significance", ""), 0.25)
            for pred in predicted_findings_list
        }
        
        E_false = sum(pred_weight_by_id.get(f_id, 0.0) for f_id in errors.get("false_findings", []))
        
        E_miss = sum(ref_weight_by_id.get(m_id, 0.0) for m_id in errors.get("missing_findings", []))
        
        attr_errors = errors.get("attribute_errors", [])
        n_location = sum(1 for e in attr_errors if "location" in e.get("error_types", []))
        n_severity = sum(1 for e in attr_errors if "severity" in e.get("error_types", []))
        n_descriptor = sum(1 for e in attr_errors if "descriptor" in e.get("error_types", []))
        n_measurement = sum(1 for e in attr_errors if "measurement" in e.get("error_types", []))
        n_certainty = sum(1 for e in attr_errors if "certainty" in e.get("error_types", []))
        n_unspecific = sum(1 for e in attr_errors if "unspecific" in e.get("error_types", []))
        n_overinterpretation = sum(1 for e in attr_errors if "overinterpretation" in e.get("error_types", []))
        n_temporal = sum(1 for e in attr_errors if "temporal" in e.get("error_types", []))
        
        attr_errors_by_ref_id = {}
        for err in attr_errors:
            ref_id = err["ref_id"]
            if ref_id not in attr_errors_by_ref_id:
                attr_errors_by_ref_id[ref_id] = []
            attr_errors_by_ref_id[ref_id].append(err)
        
        N_G = calculate_weighted_count(reference_findings_list)
        if N_G == 0 and not reference_findings_list:
            N_G = len(matched) + E_miss
        
        E_penalty = E_false
        
        # Weighted sum of matched findings with partial credit for attribute errors
        matched_ref_ids = set()
        correct = 0.0
        for m in matched:
            ref_id = m["ref_id"]
            if ref_id in matched_ref_ids:
                continue  # Already counted this reference finding
            matched_ref_ids.add(ref_id)
            base_weight = ref_weight_by_id.get(ref_id, 0.0)
            
            finding_attr_errors = attr_errors_by_ref_id.get(ref_id, [])
            
            if not finding_attr_errors:
                correct += base_weight
            else:
                sum_error_weights = sum(
                    attribute_severity_weights.get(err.get("severity", ""), 0.25) for err in finding_attr_errors
                )
                credit_factor = base_weight / (base_weight + sum_error_weights) if (base_weight + sum_error_weights) > 0 else 0.0
                correct += base_weight * credit_factor
        
        errors_more_than_correct = E_penalty - correct
        
        if N_G == 0:
            S = 1.0 if E_penalty == 0 and E_miss == 0 else -(E_penalty + E_miss + 1)
        else:
            S = (correct - E_penalty) / N_G
        
        if S >= 0:
            crimson = S
        else:
            if errors_more_than_correct > 0:
                crimson = -1 * errors_more_than_correct / (1 + errors_more_than_correct)
            else:
                crimson = 0
        
        return {
            "raw_evaluation": evaluation,
            "error_counts": {
                "false_findings": len(errors.get("false_findings", [])),
                "missing_findings": len(errors.get("missing_findings", [])),
                "attribute_errors": len(attr_errors),
                "location_errors": n_location,
                "severity_errors": n_severity,
                "descriptor_errors": n_descriptor,
                "measurement_errors": n_measurement,
                "certainty_errors": n_certainty,
                "unspecific_errors": n_unspecific,
                "overinterpretation_errors": n_overinterpretation,
                "temporal_errors": n_temporal
            },
            "weighted_error_counts": {
                "false_findings": E_false,
                "missing_findings": E_miss,
                "attribute_errors": calculate_weighted_count(attr_errors, attribute_severity_weights, "severity")
            },
            "metrics": {
                "N_G": N_G,  # Number of ground truth findings
                "E_penalty": E_penalty,
                "correct": correct,
                "errors_more_than_correct": errors_more_than_correct,
                "S": S
            },
            "crimson_score": round(crimson, 4)
        }