"""
Stage 2: Rule-Based NLP Baseline with NegEx Negation Detection
Implements CVD ontology covering 8 condition categories.
Applies regex pattern matching + NegEx negation suppression.
Also checks Other_Diagnosis / Other_Diagnosis_2 structured fields.
"""

import re
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import os

IN_PATH = "data/neiss2023_clean.parquet"
OUT_PATH = "data/neiss2023_rule_based.parquet"

# ---------------------------------------------------------------------------
# CVD Ontology: 8 condition categories with synonyms/abbreviations
# ---------------------------------------------------------------------------

CVD_ONTOLOGY: Dict[str, List[str]] = {
    "coronary_artery_disease": [
        r"\bcad\b", r"\bcoronary artery disease\b", r"\bcoronary disease\b",
        r"\bangina\b", r"\bstemi\b", r"\bnstemi\b", r"\bami\b",
        r"\bmyocardial infarction\b", r"\bheart attack\b",
        r"\bcoronary artery bypass\b", r"\bcabg\b",
        r"\bpercutaneous coronary\b", r"\bpci\b", r"\bstent\b",
        r"\bischemic heart disease\b",
    ],
    "heart_failure": [
        r"\bchf\b", r"\bcongestive heart failure\b", r"\bheart failure\b",
        r"\bcardiac failure\b", r"\bleft ventricular failure\b",
        r"\blvf\b", r"\bejection fraction\b",
        r"\bhfref\b", r"\bhfpef\b",
        r"\bdecompensated heart failure\b", r"\bdecompensated hf\b",
        r"\bpulmonary edema\b",                  # Added per analysis plan §3.1
    ],
    "atrial_fibrillation": [
        r"\bafib\b", r"\ba\.?fib\b", r"\batrial fibrillation\b",
        r"\batrial flutter\b", r"\barrhythmia\b",
        r"\bparoxysmal atrial\b",
        r"\bpacemaker\b",                         # Added per analysis plan §3.1
        r"\bimplantable cardioverter\b", r"\bicd\b",  # Added
        r"\bsvt\b", r"\bsupraventricular tachycardia\b",  # Added
        r"\bventricular tachycardia\b", r"\bvt\b(?= |$)",  # Added
    ],
    "hypertension": [
        r"\bhtn\b", r"\bhypertension\b", r"\bhigh blood pressure\b",
        r"\bhbp\b", r"\bhypertensive\b", r"\belevated blood pressure\b",
    ],
    "stroke_tia": [
        r"\bstroke\b", r"\bcva\b", r"\btia\b",
        r"\btransient ischemic attack\b", r"\bcerebrovascular accident\b",
        r"\bischemic stroke\b", r"\bhemorrhagic stroke\b",
        r"\bcerebrovascular\b",
        r"\bcarotid stenosis\b",                  # Added per analysis plan §3.1
    ],
    "peripheral_vascular_disease": [
        r"\bpad\b", r"\bperipheral artery disease\b",
        r"\bperipheral vascular disease\b", r"\bpvd\b",
        r"\bclaudication\b", r"\bperipheral arterial\b",
        r"\barterial insufficiency\b",            # Added per analysis plan §3.1
    ],
    "cardiomyopathy": [
        r"\bcardiomyopathy\b", r"\bdilated cardiomyopathy\b",
        r"\bhypertrophic cardiomyopathy\b", r"\bhocm\b",
        r"\bischemic cardiomyopathy\b",
        r"\brestrictive cardiomyopathy\b",        # Added per analysis plan §3.1
    ],
    "valvular_disease": [
        r"\bvalvular\b", r"\baortic stenosis\b", r"\baortic regurgitation\b",
        r"\bmitral stenosis\b", r"\bmitral regurgitation\b",
        r"\bmitral valve\b", r"\baortic valve\b",
        r"\bvalve replacement\b", r"\bvalvuloplasty\b",
        r"\bmitral valve prolapse\b", r"\bmvp\b",  # Added per analysis plan §3.1
        r"\bvalve repair\b",                       # Added
    ],
}

# Compile patterns
COMPILED_PATTERNS: Dict[str, List[re.Pattern]] = {
    category: [re.compile(p, re.IGNORECASE) for p in patterns]
    for category, patterns in CVD_ONTOLOGY.items()
}

# ---------------------------------------------------------------------------
# NegEx Implementation
# ---------------------------------------------------------------------------

NEGATION_PRE = [
    r"\bno\b", r"\bnot\b", r"\bdenies\b", r"\bdenied\b",
    r"\bno history of\b", r"\bwithout\b", r"\bnegative for\b",
    r"\bno known\b", r"\bnone\b", r"\bno h/o\b", r"\bno hx\b",
]

NEGATION_POST = [
    r"\bfree\b", r"\bnot present\b", r"\bnot detected\b",
]

NEGATION_PRE_COMPILED = [re.compile(p, re.IGNORECASE) for p in NEGATION_PRE]
NEGATION_POST_COMPILED = [re.compile(p, re.IGNORECASE) for p in NEGATION_POST]

CONTEXT_WINDOW = 5  # tokens


def negex_check(text: str, match_start: int, match_end: int) -> bool:
    """
    Return True if the match is negated.
    Checks a 5-token window before and after the matched span.
    """
    tokens = text.split()
    # find approximate token positions
    pre_text = text[:match_start]
    post_text = text[match_end:]

    pre_tokens = pre_text.split()[-CONTEXT_WINDOW:]
    post_tokens = post_text.split()[:CONTEXT_WINDOW]

    pre_window = " ".join(pre_tokens)
    post_window = " ".join(post_tokens)

    for pattern in NEGATION_PRE_COMPILED:
        if pattern.search(pre_window):
            return True
    for pattern in NEGATION_POST_COMPILED:
        if pattern.search(post_window):
            return True
    return False


# ---------------------------------------------------------------------------
# NEISS structured field CVD codes (Other_Diagnosis ICD proxy codes)
# ---------------------------------------------------------------------------

CVD_DIAG_CODES = {
    "53",  # Heart disease (NEISS diagnosis code)
}

# Not all NEISS releases include ICD codes; we do keyword match on the text
CVD_OTHER_DIAG_KEYWORDS = re.compile(
    r"\bchf\b|\bhtn\b|\bafib\b|\bcad\b|\bheart failure\b|\bhypertension\b",
    re.IGNORECASE
)


def extract_cvd_from_narrative(narrative: str) -> Tuple[bool, Dict[str, bool]]:
    """
    Apply rule-based extraction with NegEx to a single narrative string.
    Returns (cvd_positive, per_category_labels).
    """
    condition_labels: Dict[str, bool] = {}
    narrative_lower = narrative.lower()

    for category, patterns in COMPILED_PATTERNS.items():
        found = False
        for pattern in patterns:
            for m in pattern.finditer(narrative_lower):
                if not negex_check(narrative_lower, m.start(), m.end()):
                    found = True
                    break
            if found:
                break
        condition_labels[category] = found

    cvd_positive = any(condition_labels.values())
    return cvd_positive, condition_labels


def extract_cvd_from_structured_fields(row: pd.Series) -> bool:
    """
    Check Other_Diagnosis and Other_Diagnosis_2 text fields.
    """
    for field in ["Other_Diagnosis", "Other_Diagnosis_2"]:
        val = str(row.get(field, "")).lower()
        if CVD_OTHER_DIAG_KEYWORDS.search(val):
            return True
    return False


def apply_rule_based_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full rule-based NLP + structured field detection."""
    print("  Extracting CVD from narratives (rule-based + NegEx)...")

    results = df["Narrative_1"].apply(extract_cvd_from_narrative)
    df["CVD_Rule_Narrative"] = [r[0] for r in results]

    # Per-category labels
    category_dfs = pd.DataFrame(
        [r[1] for r in results],
        index=df.index
    ).add_prefix("CVD_")
    df = pd.concat([df, category_dfs], axis=1)

    print("  Checking structured diagnosis fields...")
    df["CVD_Rule_Structured"] = df.apply(extract_cvd_from_structured_fields, axis=1)

    # Final label: union of narrative and structured field detection
    df["CVD_Rule_Label"] = (df["CVD_Rule_Narrative"] | df["CVD_Rule_Structured"]).astype(int)

    n_pos = df["CVD_Rule_Label"].sum()
    print(f"  Rule-based CVD positive: {n_pos:,} ({n_pos/len(df)*100:.2f}%)")

    return df


def main():
    print("=" * 60)
    print("Stage 1: Rule-Based NLP with NegEx")
    print("=" * 60)

    df = pd.read_parquet(IN_PATH)
    print(f"Loaded {len(df):,} records")

    df = apply_rule_based_pipeline(df)

    os.makedirs("data", exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")

    # Summary
    print("\nCondition-level breakdown (rule-based positives):")
    cvd_cols = [c for c in df.columns if c.startswith("CVD_") and c not in
                ["CVD_Rule_Narrative", "CVD_Rule_Structured", "CVD_Rule_Label"]]
    for col in cvd_cols:
        n = df[col].sum()
        print(f"  {col}: {n:,} ({n/len(df)*100:.3f}%)")


if __name__ == "__main__":
    main()
