"""
Stage 1: Data Preprocessing
Loads NEISS 2023 Public Use File, applies CPSC infant age encoding correction,
and saves a clean analytic dataset.
"""

import re
import pandas as pd
import numpy as np
import os

RAW_PATH = "data/neiss2023.tsv"
OUT_PATH = "data/neiss2023_clean.parquet"

# ---------------------------------------------------------------------------
# NEISS-specific abbreviation expansion table (Analysis Plan §3.1 deliverable)
# Expands clinical shorthand before NLP stages
# ---------------------------------------------------------------------------
ABBREVIATION_MAP = {
    r"\bhtn\b": "hypertension",
    r"\bafib\b": "atrial fibrillation",
    r"\ba\.?fib\b": "atrial fibrillation",
    r"\bchf\b": "congestive heart failure",
    r"\bcad\b": "coronary artery disease",
    r"\bcva\b": "cerebrovascular accident",
    r"\btia\b": "transient ischemic attack",
    r"\bpad\b": "peripheral artery disease",
    r"\bpvd\b": "peripheral vascular disease",
    r"\bami\b": "acute myocardial infarction",
    r"\bstemi\b": "ST-elevation myocardial infarction",
    r"\bnstemi\b": "non-ST-elevation myocardial infarction",
    r"\bhf\b": "heart failure",
    r"\bhocm\b": "hypertrophic obstructive cardiomyopathy",
    r"\bhbp\b": "high blood pressure",
    r"\bcabg\b": "coronary artery bypass graft",
    r"\bpci\b": "percutaneous coronary intervention",
    r"\bicd\b": "implantable cardioverter defibrillator",
    r"\bsvt\b": "supraventricular tachycardia",
    r"\bvt\b": "ventricular tachycardia",
    r"\bmvp\b": "mitral valve prolapse",
    r"\bef\b": "ejection fraction",
    r"\bas\b(?= \d|\d)": "aortic stenosis",   # context-specific
}

# NEISS 2023 column names (adjust if CPSC releases updated header)
NEISS_COLS = [
    "CPSC_Case_Number", "Treatment_Date", "Age", "Sex", "Race",
    "Other_Race", "Hispanic", "Body_Part", "Diagnosis", "Other_Diagnosis",
    "Other_Diagnosis_2", "Body_Part_2", "Diagnosis_2", "Stratum", "PSU",
    "Weight", "Narrative_1", "Narrative_2", "Product_1", "Product_2",
    "Product_3", "Disposition", "Location", "Fire_Involvement"
]


def load_neiss(path: str) -> pd.DataFrame:
    """Load NEISS TSV with flexible column handling."""
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        encoding="latin-1",
        low_memory=False
    )
    df.columns = [c.strip() for c in df.columns]
    return df


def recode_infant_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    CPSC encodes infants 0-23 months as Age 200-223.
    Recode to fractional years: Age_Years = (Age - 200) / 12
    Affects ~19,321 records (5.7% of sample).
    """
    df = df.copy()
    df["Age_Numeric"] = pd.to_numeric(df["Age"], errors="coerce")

    infant_mask = (df["Age_Numeric"] >= 200) & (df["Age_Numeric"] <= 223)
    df.loc[infant_mask, "Age_Years"] = (df.loc[infant_mask, "Age_Numeric"] - 200) / 12
    df.loc[~infant_mask, "Age_Years"] = df.loc[~infant_mask, "Age_Numeric"]

    n_infant = infant_mask.sum()
    print(f"  Infant age recoding applied to {n_infant:,} records ({n_infant/len(df)*100:.1f}%)")
    return df


def clean_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize key variables."""
    df = df.copy()

    # Weight (survey weight)
    df["Weight_Numeric"] = pd.to_numeric(df.get("Weight", df.get("weight")), errors="coerce")

    # Sex: 1=Male, 2=Female
    df["Sex_Numeric"] = pd.to_numeric(df.get("Sex"), errors="coerce")
    df["Female"] = (df["Sex_Numeric"] == 2).astype(int)

    # Disposition: 4 = admitted/hospitalized
    df["Disposition_Numeric"] = pd.to_numeric(df.get("Disposition"), errors="coerce")
    df["Hospitalized"] = (df["Disposition_Numeric"] == 4).astype(int)
    df["ED_Death"] = (df["Disposition_Numeric"] == 8).astype(int)
    df["Transferred"] = (df["Disposition_Numeric"].isin([5, 6])).astype(int)

    # Race (NEISS codes: 1=White, 2=Black, 3=Other, 4=Asian/PI, 5=Am Indian, 6=Hispanic)
    df["Race_Numeric"] = pd.to_numeric(df.get("Race"), errors="coerce")
    df["Race_Missing"] = df["Race_Numeric"].isna().astype(int)

    # Narrative: fill NaN
    for col in ["Narrative_1", "Narrative_2"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip().str.lower()

    return df


def expand_abbreviations(text: str) -> str:
    """Expand NEISS clinical abbreviations (Analysis Plan §3.2)."""
    for pattern, expansion in ABBREVIATION_MAP.items():
        text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
    return text


def detect_narrative_sections(text: str) -> dict:
    """
    Attempt to separate injury description from background medical history.
    NEISS narratives often follow pattern: [injury event] [PMH context].
    Returns dict with 'injury_description' and 'medical_background'.
    (Analysis Plan §3.2: Section detection)
    """
    # Heuristic split: background history usually follows keywords like
    # "history of", "h/o", "pmh", "past medical", "with known"
    background_markers = re.compile(
        r"\b(h/o|hx of|history of|past medical|pmh|with known|has a history)\b",
        re.IGNORECASE
    )
    match = background_markers.search(text)
    if match:
        return {
            "injury_description": text[:match.start()].strip(),
            "medical_background": text[match.start():].strip(),
        }
    return {"injury_description": text, "medical_background": ""}


def flag_short_narratives(df: pd.DataFrame, min_tokens: int = 10) -> pd.DataFrame:
    """
    Flag narratives with fewer than 10 tokens as potentially uninformative.
    (Analysis Plan §3.2: Length filtering)
    """
    df = df.copy()
    df["Narrative_Token_Count"] = df["Narrative_1"].str.split().str.len()
    df["Narrative_Short_Flag"] = (df["Narrative_Token_Count"] < min_tokens).astype(int)
    n_short = df["Narrative_Short_Flag"].sum()
    print(f"  Short narratives (<{min_tokens} tokens): {n_short:,} "
          f"({n_short/len(df)*100:.1f}%)")
    return df


def preprocess_narratives(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full narrative preprocessing pipeline (Analysis Plan §3.2):
    1. Lowercasing + whitespace normalization
    2. NEISS abbreviation expansion
    3. Section detection
    4. Length flagging
    """
    df = df.copy()

    # Step 1: lowercase + normalize
    df["Narrative_1"] = (df["Narrative_1"]
                         .str.lower()
                         .str.replace(r"\s+", " ", regex=True)
                         .str.strip())

    # Step 2: abbreviation expansion (store expanded version for NLP)
    print("  Expanding clinical abbreviations...")
    df["Narrative_Expanded"] = df["Narrative_1"].apply(expand_abbreviations)

    # Step 3: section detection
    sections = df["Narrative_Expanded"].apply(detect_narrative_sections)
    df["Narrative_Injury"] = [s["injury_description"] for s in sections]
    df["Narrative_Background"] = [s["medical_background"] for s in sections]

    # Step 4: length flag
    df = flag_short_narratives(df, min_tokens=10)

    return df


def create_mechanism_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Map NEISS fields to broad injury mechanism groups."""
    df = df.copy()
    narr = df["Narrative_Expanded"].str.lower()  # use expanded narratives

    df["Mech_Fall"] = narr.str.contains(
        r"\bfall\b|\bfell\b|\btripped\b|\bslipped\b|\bfalls\b", regex=True
    ).astype(int)
    df["Mech_MVC"] = narr.str.contains(
        r"\bmvc\b|\bmotor vehicle\b|\bcar accident\b|\bcollision\b|\bcrash\b", regex=True
    ).astype(int)
    df["Mech_Assault"] = narr.str.contains(
        r"\bassault\b|\bstruck\b|\bfight\b|\battack\b|\bbattery\b", regex=True
    ).astype(int)
    df["Mech_Sports"] = narr.str.contains(
        r"\bsport\b|\bbasketball\b|\bfootball\b|\bsoccer\b|\btennis\b|\bcycl", regex=True
    ).astype(int)

    # Alcohol involvement (Analysis Plan §2.3 covariate)
    df["Alcohol_Involved"] = narr.str.contains(
        r"\balcohol\b|\bdrunk\b|\bintoxicat\b|\bethanol\b|\bbeer\b|\bwine\b|\bliquor\b",
        regex=True
    ).astype(int)

    # Season from Treatment_Date (Analysis Plan §2.3)
    if "Treatment_Date" in df.columns:
        try:
            df["Treatment_Date_Parsed"] = pd.to_datetime(
                df["Treatment_Date"], errors="coerce", infer_datetime_format=True
            )
            df["Month"] = df["Treatment_Date_Parsed"].dt.month
            df["Quarter"] = df["Treatment_Date_Parsed"].dt.quarter
            df["Season"] = df["Month"].map({
                12: "Winter", 1: "Winter", 2: "Winter",
                3: "Spring", 4: "Spring", 5: "Spring",
                6: "Summer", 7: "Summer", 8: "Summer",
                9: "Fall_Season", 10: "Fall_Season", 11: "Fall_Season",
            })
            df["Season_Winter"] = (df["Season"] == "Winter").astype(int)
            df["Season_Spring"] = (df["Season"] == "Spring").astype(int)
            df["Season_Summer"] = (df["Season"] == "Summer").astype(int)
        except Exception:
            pass

    return df


def main():
    print("=" * 60)
    print("NEISS 2023 Data Preprocessing")
    print("=" * 60)

    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(
            f"{RAW_PATH} not found.\n"
            "Download NEISS 2023 Public Use File from "
            "https://www.cpsc.gov/Research--Statistics/NEISS-Injury-Data"
        )

    print(f"\nLoading: {RAW_PATH}")
    df = load_neiss(RAW_PATH)
    print(f"  Raw records: {len(df):,}")

    print("\nRecoding infant ages...")
    df = recode_infant_age(df)

    print("\nCleaning variables...")
    df = clean_variables(df)

    print("\nPreprocessing narratives (abbreviation expansion, section detection, length filter)...")
    df = preprocess_narratives(df)

    print("\nCreating mechanism groups, alcohol flag, and season variables...")
    df = create_mechanism_groups(df)

    print(f"\nFinal dataset: {len(df):,} records, {df.shape[1]} columns")
    print(f"  Hospitalized: {df['Hospitalized'].sum():,} ({df['Hospitalized'].mean()*100:.1f}%)")
    print(f"  Missing race: {df['Race_Missing'].sum():,} ({df['Race_Missing'].mean()*100:.1f}%)")

    os.makedirs("data", exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
