# NEISS 2023 CVD Comorbidity Extraction

**Cardiovascular Disease Comorbidity Among Emergency Department Injury Patients in the United States: A Nationally Representative Cross-Sectional Study with Multistage NLP, Transformer, and LLM Comorbidity Extraction from NEISS 2023**

Rohit Panaganti, MS  
Graduate Program in Computer and Information Science, University of Alabama at Birmingham

---

## Overview

This repository contains all analytic code for a multistage NLP pipeline that extracts CVD comorbidity from free-text NEISS 2023 narratives, benchmarks five model architectures, and conducts nationally representative epidemiological analysis with IPTW and BERTopic phenotyping.

### Pipeline Stages

| Stage | Description |
|-------|-------------|
| 1 | Rule-based regex + NegEx negation detection |
| 2 | Gold-standard annotation (Label Studio, n=3,000) |
| 3a | TF-IDF Logistic Regression |
| 3b | TF-IDF Gradient Boosting |
| 3c | Fine-tuned BioClinicalBERT |
| 4 | LLM-based structured extraction (Claude API) |
| 5 | Weighted hybrid ensemble |

### Key Results

- Hybrid ensemble: **AUROC 0.9250**, **AUPRC 0.9034**, F1 0.900  
- CVD prevalence: **0.81%** (weighted N ≈ 103,541)  
- CVD associated hospitalization: **aOR 8.84** (95% CI 7.02–11.13)  
- IPTW-adjusted aOR: **7.91** (95% CI 6.18–10.12)  
- BERTopic: **6 clinically coherent phenotypes**  
- LLM causal flag: CVD-triggered injury in **24.2%** of CVD-positive patients  

---

## Repository Structure

```
neiss_cvd/
├── src/
│   ├── 01_data_preprocessing.py       # NEISS data loading & age recoding
│   ├── 02_stage1_rule_based_nlp.py    # Rule-based NLP + NegEx
│   ├── 03_stage2_annotation_prep.py   # Annotation sample construction
│   ├── 04_stage3_tfidf_classifiers.py # TF-IDF LR & Gradient Boosting
│   ├── 05_stage3_bioclinicalbert.py   # BioClinicalBERT fine-tuning
│   ├── 06_stage4_llm_extraction.py    # Claude API structured extraction
│   ├── 07_ensemble.py                 # Weighted hybrid ensemble
│   ├── 08_model_evaluation.py         # AUROC/AUPRC/SHAP/calibration
│   ├── 09_bertopic_clustering.py      # BERTopic + UMAP + HDBSCAN
│   └── 10_statistical_analysis.py    # Survey-weighted regression + IPTW
├── data/                              # Place NEISS 2023 public use file here
├── outputs/                           # Model outputs and figures
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/rpanagan/neiss_2023_CVD_analysis.git
cd neiss_2023_CVD_analysis
pip install -r requirements.txt
```

### Data

Download the NEISS 2023 Public Use File from [CPSC](https://www.cpsc.gov/Research--Statistics/NEISS-Injury-Data) and place it in `data/neiss2023.tsv`.

---

## Usage

Run stages in order:

```bash
python src/01_data_preprocessing.py
python src/02_stage1_rule_based_nlp.py
python src/03_stage2_annotation_prep.py   # Export to Label Studio for annotation
# ... annotate in Label Studio, export as gold_standard.csv ...
python src/04_stage3_tfidf_classifiers.py
python src/05_stage3_bioclinicalbert.py
python src/06_stage4_llm_extraction.py    # Requires ANTHROPIC_API_KEY env var
python src/07_ensemble.py
python src/08_model_evaluation.py
python src/09_bertopic_clustering.py
python src/10_statistical_analysis.py
```

---

## Citation

Panaganti R. Cardiovascular Disease Comorbidity Among Emergency Department Injury Patients in the United States: A Nationally Representative Cross-Sectional Study with Multistage NLP, Transformer, and LLM Comorbidity Extraction from NEISS 2023. *University of Alabama at Birmingham*, 2024.

---

## License

MIT License
