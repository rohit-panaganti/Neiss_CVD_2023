"""
Stage 4: LLM-Based Structured Extraction (Anthropic Claude API)
Applied to all rule-based CVD positive narratives (n ≈ 2,260).
Extracts: cvd_present, cvd_conditions, confidence, evidence_span,
          negated, causal_flag.
"""

import anthropic
import pandas as pd
import numpy as np
import json
import os
import time
from tqdm import tqdm

IN_PATH = "data/neiss2023_rule_based.parquet"
OUT_PATH = "data/llm_extracted.parquet"
CAUSAL_OUT = "outputs/llm_causal_flags.csv"

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 512
RATE_LIMIT_SLEEP = 0.5  # seconds between API calls

SYSTEM_PROMPT = """You are a clinical NLP system trained to identify cardiovascular disease (CVD)
comorbidities in emergency department injury narratives. You will be given a short clinical narrative
from the National Electronic Injury Surveillance System (NEISS).

Your task is to extract structured CVD comorbidity information and return ONLY a valid JSON object
with NO additional text, markdown, or explanation.

CVD conditions to detect:
- Coronary artery disease (CAD, STEMI, NSTEMI, angina, heart attack, CABG, stent, PCI)
- Heart failure (CHF, congestive heart failure, EF, HFrEF, HFpEF)
- Atrial fibrillation/arrhythmia (AFib, A-fib, atrial flutter, arrhythmia)
- Hypertension (HTN, high blood pressure, HBP)
- Stroke/TIA (CVA, TIA, transient ischemic attack, cerebrovascular accident)
- Peripheral vascular disease (PAD, PVD, claudication)
- Cardiomyopathy (dilated, hypertrophic, HOCM)
- Valvular disease (aortic stenosis, mitral regurgitation, valve replacement)

JSON schema:
{
  "cvd_present": true/false,
  "cvd_conditions": ["list of detected conditions"],
  "confidence": 0.0-1.0,
  "evidence_span": "exact quoted text supporting detection (empty string if none)",
  "negated": true/false,
  "causal_flag": true/false  // true if CVD plausibly precipitated the injury event
}
"""

USER_TEMPLATE = """Narrative: {narrative}

Return ONLY the JSON object."""


def call_claude_api(client: anthropic.Anthropic, narrative: str) -> dict:
    """Call Claude API for a single narrative. Returns parsed JSON."""
    message = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": USER_TEMPLATE.format(narrative=narrative)}
        ]
    )

    raw = message.content[0].text.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    return json.loads(raw)


def default_extraction() -> dict:
    """Return a safe default if API call fails."""
    return {
        "cvd_present": False,
        "cvd_conditions": [],
        "confidence": 0.0,
        "evidence_span": "",
        "negated": False,
        "causal_flag": False,
    }


def run_llm_extraction(df_positives: pd.DataFrame,
                       client: anthropic.Anthropic) -> pd.DataFrame:
    """
    Run LLM extraction on all CVD-positive records.
    """
    results = []
    errors = 0

    for _, row in tqdm(df_positives.iterrows(), total=len(df_positives),
                       desc="LLM extraction"):
        narrative = str(row.get("Narrative_1", ""))

        try:
            extracted = call_claude_api(client, narrative)
        except json.JSONDecodeError:
            extracted = default_extraction()
            errors += 1
        except Exception as e:
            extracted = default_extraction()
            errors += 1

        extracted["CPSC_Case_Number"] = row.get("CPSC_Case_Number")
        results.append(extracted)
        time.sleep(RATE_LIMIT_SLEEP)

    print(f"\n  Total processed: {len(results):,} | Errors: {errors:,}")
    return pd.DataFrame(results)


def compute_llm_probability(llm_df: pd.DataFrame) -> np.ndarray:
    """
    Convert LLM structured output to a numeric probability for ensemble.
    P = confidence if cvd_present=True and negated=False, else 1-confidence.
    """
    proba = np.where(
        (llm_df["cvd_present"].astype(bool)) & (~llm_df["negated"].astype(bool)),
        llm_df["confidence"].clip(0, 1),
        1 - llm_df["confidence"].clip(0, 1)
    )
    return proba


def main():
    print("=" * 60)
    print("Stage 4: LLM-Based Structured Extraction")
    print("=" * 60)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable not set.\n"
            "Export it: export ANTHROPIC_API_KEY=sk-ant-..."
        )

    client = anthropic.Anthropic(api_key=api_key)

    df = pd.read_parquet(IN_PATH)
    df_positives = df[df["CVD_Rule_Label"] == 1].copy()
    print(f"CVD-positive records to process: {len(df_positives):,}")

    print("\nRunning LLM extraction...")
    llm_df = run_llm_extraction(df_positives, client)

    # Add probability column
    llm_df["proba_llm"] = compute_llm_probability(llm_df)

    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    llm_df.to_parquet(OUT_PATH, index=False)
    print(f"LLM extraction saved: {OUT_PATH}")

    # Causal flag summary
    n_causal = llm_df["causal_flag"].sum()
    n_total = len(llm_df)
    pct_causal = n_causal / n_total * 100 if n_total > 0 else 0
    print(f"\nCausal flag (CVD triggered injury): {n_causal:,} / {n_total:,} ({pct_causal:.1f}%)")

    causal_df = llm_df[llm_df["causal_flag"].astype(bool)][
        ["CPSC_Case_Number", "cvd_conditions", "evidence_span", "confidence"]
    ]
    causal_df.to_csv(CAUSAL_OUT, index=False)
    print(f"Causal cases saved: {CAUSAL_OUT}")

    # Summary statistics
    print("\nLLM Extraction Summary:")
    print(f"  CVD present (LLM): {llm_df['cvd_present'].sum():,}")
    print(f"  Negated: {llm_df['negated'].sum():,}")
    print(f"  Mean confidence: {llm_df['confidence'].mean():.3f}")
    print(f"  Causal flag: {n_causal:,} ({pct_causal:.1f}%)")

    top_conditions = (
        llm_df["cvd_conditions"].explode().value_counts().head(10)
    )
    print("\nTop 10 extracted conditions:")
    print(top_conditions.to_string())


if __name__ == "__main__":
    main()
