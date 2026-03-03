"""
Stage 8: Survey-Weighted Epidemiological Analysis
- Survey-weighted logistic regression (hospitalization as outcome)
- Interaction terms: CVD × Age, CVD × Fall mechanism
- Sandwich standard errors
- Survey-weighted multinomial logistic regression (4-category disposition)
- Mechanism-stratified analyses
- IPTW propensity score sensitivity analysis (stabilized weights, SMD check)
- E-value calculation for unmeasured confounding
- Multiple imputation for missing race (5 datasets, Rubin's rules)
- HTN-exclusion and age ≥40 sensitivity analyses
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import DescrStatsW
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
import os

warnings.filterwarnings("ignore")

ENSEMBLE_LABELS = "data/neiss2023_ensemble_labels.parquet"
OUT_DIR = "outputs/statistics"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_analytic_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} records")

    # Standardize age (with quadratic term)
    df["Age_Z"] = (df["Age_Years"] - df["Age_Years"].mean()) / df["Age_Years"].std()
    df["Age_Z2"] = df["Age_Z"] ** 2

    # Reference: treat & release (Disposition=1/2)
    df["Disposition_4Cat"] = np.where(
        df["Disposition_Numeric"] == 4, "Admitted",
        np.where(df["Disposition_Numeric"].isin([5, 6]), "Transferred",
                 np.where(df["Disposition_Numeric"] == 8, "ED_Death", "Treat_Release"))
    )

    # CVD binary
    df["CVD"] = df["CVD_Ensemble_Label"].astype(int)

    # Impute missing race with mode for primary analysis
    mode_race = df["Race_Numeric"].mode()[0]
    df["Race_Imputed"] = df["Race_Numeric"].fillna(mode_race)

    # Dummy race
    df["Race_White"] = (df["Race_Imputed"] == 1).astype(int)
    df["Race_Black"] = (df["Race_Imputed"] == 2).astype(int)
    df["Race_Other"] = df["Race_Imputed"].isin([3, 4, 5, 6]).astype(int)

    return df


def weighted_prevalence(df: pd.DataFrame):
    """Survey-weighted CVD prevalence."""
    total_w = df["Weight_Numeric"].sum()
    cvd_w = df.loc[df["CVD"] == 1, "Weight_Numeric"].sum()
    pct = cvd_w / total_w * 100

    n_cvd = df["CVD"].sum()
    print(f"\nCVD Prevalence:")
    print(f"  Unweighted: {n_cvd:,} / {len(df):,} ({n_cvd/len(df)*100:.2f}%)")
    print(f"  Weighted N: {cvd_w:,.0f} / {total_w:,.0f} ({pct:.2f}%)")

    # Hospitalization by CVD status
    for cvd_val, label in [(0, "No CVD"), (1, "CVD")]:
        mask = df["CVD"] == cvd_val
        sub = df[mask]
        hosp_w = sub.loc[sub["Hospitalized"] == 1, "Weight_Numeric"].sum()
        total_sub_w = sub["Weight_Numeric"].sum()
        print(f"  Hospitalization rate ({label}): {hosp_w/total_sub_w*100:.1f}%")


def primary_logistic_regression(df: pd.DataFrame) -> pd.DataFrame:
    """
    Survey-weighted logistic regression: hospitalization ~ CVD + covariates + interactions.
    With sandwich standard errors.
    """
    formula = (
        "Hospitalized ~ CVD + Age_Z + Age_Z2 + Female + "
        "Race_Black + Race_Other + "
        "Mech_Fall + Mech_MVC + Mech_Assault + "
        "CVD:Age_Z + CVD:Mech_Fall"
    )

    # Drop rows with missing weight
    analytic = df.dropna(subset=["Weight_Numeric", "Hospitalized"])

    model = smf.glm(
        formula=formula,
        data=analytic,
        family=sm.families.Binomial(),
        freq_weights=analytic["Weight_Numeric"],
    )

    result = model.fit(cov_type="HC1")  # Sandwich SEs

    print("\nPrimary Logistic Regression (Hospitalization):")
    print(result.summary2().tables[1].round(4))

    # Odds ratios + 95% CI
    or_df = pd.DataFrame({
        "OR": np.exp(result.params),
        "CI_Lower": np.exp(result.conf_int()[0]),
        "CI_Upper": np.exp(result.conf_int()[1]),
        "P_Value": result.pvalues,
    })

    print("\nOdds Ratios:")
    print(or_df.round(3).to_string())

    return or_df


def mechanism_stratified_analysis(df: pd.DataFrame):
    """
    Mechanism-stratified logistic regressions.
    Strata: Falls, MVC, Assault/Struck, Sports.
    """
    strata = {
        "Falls": df["Mech_Fall"] == 1,
        "MVC": df["Mech_MVC"] == 1,
        "Assault_Struck": df["Mech_Assault"] == 1,
        "Sports_Recreation": df["Mech_Sports"] == 1,
    }

    formula = "Hospitalized ~ CVD + Age_Z + Female + Race_Black + Race_Other"

    results = []
    print("\nMechanism-Stratified Analysis:")
    for stratum, mask in strata.items():
        sub = df[mask].dropna(subset=["Weight_Numeric", "Hospitalized"])
        n_cvd = sub["CVD"].sum()

        if n_cvd < 10:
            print(f"  {stratum}: Insufficient CVD cases (n={n_cvd}) — skipped")
            continue

        try:
            model = smf.glm(
                formula=formula,
                data=sub,
                family=sm.families.Binomial(),
                freq_weights=sub["Weight_Numeric"],
            ).fit(cov_type="HC1")

            cvd_or = np.exp(model.params["CVD"])
            cvd_ci = np.exp(model.conf_int().loc["CVD"])
            cvd_p = model.pvalues["CVD"]

            print(f"  {stratum}: n={len(sub):,}, CVD+=  {n_cvd}, "
                  f"aOR={cvd_or:.2f} (95% CI {cvd_ci[0]:.2f}–{cvd_ci[1]:.2f}), p={cvd_p:.4f}")

            results.append({
                "stratum": stratum, "n": len(sub), "n_cvd": n_cvd,
                "aOR": cvd_or, "CI_Lower": cvd_ci[0], "CI_Upper": cvd_ci[1], "p": cvd_p
            })
        except Exception as e:
            print(f"  {stratum}: Model failed — {e}")

    return pd.DataFrame(results)


def compute_propensity_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit propensity score model for CVD ~ age, sex, race, mechanism, body part, season.
    Compute stabilized IPTW weights.
    """
    print("\nIPTW Propensity Score Analysis...")

    ps_features = ["Age_Z", "Female", "Race_Black", "Race_Other",
                   "Mech_Fall", "Mech_MVC", "Mech_Assault", "Mech_Sports"]

    analytic = df.dropna(subset=ps_features + ["CVD"]).copy()

    X = analytic[ps_features].values
    y = analytic["CVD"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ps_model = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    ps_model.fit(X_scaled, y)
    ps = ps_model.predict_proba(X_scaled)[:, 1]
    analytic["ps"] = ps

    # Stabilized IPTW
    p_treated = y.mean()
    analytic["iptw"] = np.where(
        y == 1,
        p_treated / ps,
        (1 - p_treated) / (1 - ps)
    )

    # Trim extreme weights at 99th percentile
    p99 = np.percentile(analytic["iptw"], 99)
    analytic["iptw_trim"] = analytic["iptw"].clip(upper=p99)

    # SMD check
    print("\n  Standardized Mean Differences (pre/post IPTW):")
    print(f"  {'Feature':30} {'Pre-IPTW SMD':>14} {'Post-IPTW SMD':>14}")
    for feat in ps_features:
        treated = analytic.loc[analytic["CVD"] == 1, feat]
        control = analytic.loc[analytic["CVD"] == 0, feat]
        pooled_sd = np.sqrt((treated.var() + control.var()) / 2 + 1e-9)
        smd_pre = abs(treated.mean() - control.mean()) / pooled_sd

        # Post-IPTW
        w_treated = analytic.loc[analytic["CVD"] == 1, "iptw_trim"]
        w_control = analytic.loc[analytic["CVD"] == 0, "iptw_trim"]
        mean_t = np.average(treated, weights=w_treated)
        mean_c = np.average(control, weights=w_control)
        smd_post = abs(mean_t - mean_c) / pooled_sd

        flag = "⚠" if smd_post > 0.10 else "✓"
        print(f"  {feat:30} {smd_pre:14.3f} {smd_post:14.3f} {flag}")

    return analytic


def iptw_outcome_model(analytic: pd.DataFrame) -> pd.DataFrame:
    """
    IPTW-adjusted logistic regression.
    """
    formula = "Hospitalized ~ CVD + Age_Z + Female + Race_Black + Race_Other + Mech_Fall"

    model = smf.glm(
        formula=formula,
        data=analytic.dropna(subset=["Hospitalized"]),
        family=sm.families.Binomial(),
        freq_weights=analytic.dropna(subset=["Hospitalized"])["iptw_trim"]
        * analytic.dropna(subset=["Hospitalized"])["Weight_Numeric"],
    ).fit(cov_type="HC1")

    cvd_or = np.exp(model.params["CVD"])
    cvd_ci = np.exp(model.conf_int().loc["CVD"])
    cvd_p = model.pvalues["CVD"]

    print(f"\nIPTW-Adjusted CVD aOR: {cvd_or:.2f} "
          f"(95% CI {cvd_ci[0]:.2f}–{cvd_ci[1]:.2f}), p={cvd_p:.4f}")

    return pd.DataFrame([{
        "model": "IPTW-Adjusted", "aOR": cvd_or,
        "CI_Lower": cvd_ci[0], "CI_Upper": cvd_ci[1], "p": cvd_p
    }])


def compute_e_value(aor: float, se: float = None, ci_lower: float = None) -> float:
    """
    E-value for unmeasured confounding (VanderWeele & Ding, 2017).
    E = aOR + sqrt(aOR * (aOR - 1))
    For CI lower bound: same formula applied to CI_lower.
    """
    def evalue(effect):
        if effect < 1:
            effect = 1 / effect
        return effect + np.sqrt(effect * (effect - 1))

    e_point = evalue(aor)
    print(f"\nE-Value Analysis:")
    print(f"  aOR (point estimate): {aor:.2f}")
    print(f"  E-value (point):      {e_point:.2f}")

    if ci_lower is not None:
        e_ci = evalue(ci_lower)
        print(f"  E-value (CI lower):   {e_ci:.2f}")

    return e_point


def sensitivity_htn_exclusion(df: pd.DataFrame):
    """Sensitivity: exclude pure HTN-only CVD cases."""
    htn_only = df["CVD_hypertension"].astype(bool) & ~(
        df["CVD_coronary_artery_disease"].astype(bool) |
        df["CVD_heart_failure"].astype(bool) |
        df["CVD_atrial_fibrillation"].astype(bool) |
        df["CVD_stroke_tia"].astype(bool) |
        df["CVD_peripheral_vascular_disease"].astype(bool)
    )

    df_excl = df[~(df["CVD"] == 1) | ~htn_only].copy()
    n_excl = htn_only.sum()
    print(f"\nHTN-Exclusion Sensitivity: Excluded {n_excl:,} HTN-only CVD cases")
    print(f"  Remaining CVD+: {df_excl['CVD'].sum():,}")
    return df_excl


def sensitivity_age_restriction(df: pd.DataFrame, min_age: float = 40.0):
    """Sensitivity: restrict to age >= 40."""
    df_aged = df[df["Age_Years"] >= min_age].copy()
    print(f"\nAge ≥{min_age:.0f} Sensitivity: {len(df_aged):,} records "
          f"({len(df_aged)/len(df)*100:.1f}% of sample)")
    return df_aged


def multiple_imputation_race(df: pd.DataFrame, n_imputations: int = 5) -> pd.DataFrame:
    """
    Multiple imputation for missing race (predictive mean matching proxy).
    Returns dataset with imputed race, pooled estimates via Rubin's rules.
    """
    print(f"\nMultiple Imputation for Race (n_imputations={n_imputations})...")
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer

    impute_features = ["Race_Numeric", "Age_Z", "Female", "CVD", "Hospitalized"]
    imputer = IterativeImputer(max_iter=10, random_state=42, n_nearest_features=5)

    imputed_datasets = []
    for i in range(n_imputations):
        df_imp = df.copy()
        imputer.random_state = i * 100
        df_imp[impute_features] = imputer.fit_transform(df[impute_features])
        df_imp["Race_Numeric"] = df_imp["Race_Numeric"].round().astype(int).clip(1, 6)
        df_imp["Race_Black"] = (df_imp["Race_Numeric"] == 2).astype(int)
        df_imp["Race_Other"] = df_imp["Race_Numeric"].isin([3, 4, 5, 6]).astype(int)
        imputed_datasets.append(df_imp)
        print(f"  Imputation {i+1}/{n_imputations} complete")

    return imputed_datasets


def rubin_pooling(models: list) -> dict:
    """
    Rubin's rules for pooling MI estimates.
    models: list of (estimate, variance) tuples.
    """
    m = len(models)
    Q_bar = np.mean([q for q, _ in models])
    W = np.mean([v for _, v in models])   # within-imputation variance
    B = np.var([q for q, _ in models], ddof=1)  # between-imputation variance
    T = W + (1 + 1/m) * B  # total variance

    return {"estimate": Q_bar, "se": np.sqrt(T),
            "ci_lower": Q_bar - 1.96 * np.sqrt(T),
            "ci_upper": Q_bar + 1.96 * np.sqrt(T)}


def main():
    print("=" * 60)
    print("Stage 8: Survey-Weighted Statistical Analysis")
    print("=" * 60)

    if not os.path.exists(ENSEMBLE_LABELS):
        raise FileNotFoundError(f"{ENSEMBLE_LABELS} not found. Run ensemble stage first.")

    df = load_analytic_data(ENSEMBLE_LABELS)
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Prevalence ---
    weighted_prevalence(df)

    # --- Primary Regression ---
    or_df = primary_logistic_regression(df)
    or_df.to_csv(f"{OUT_DIR}/primary_odds_ratios.csv")

    # Extract CVD aOR for E-value
    if "CVD" in or_df.index:
        cvd_aor = or_df.loc["CVD", "OR"]
        cvd_ci_lower = or_df.loc["CVD", "CI_Lower"]
        compute_e_value(cvd_aor, ci_lower=cvd_ci_lower)

    # --- Mechanism-Stratified ---
    strat_df = mechanism_stratified_analysis(df)
    strat_df.to_csv(f"{OUT_DIR}/stratified_analysis.csv", index=False)

    # --- IPTW ---
    analytic_iptw = compute_propensity_scores(df)
    iptw_df = iptw_outcome_model(analytic_iptw)
    iptw_df.to_csv(f"{OUT_DIR}/iptw_results.csv", index=False)

    # --- Sensitivity: HTN Exclusion ---
    df_htn_excl = sensitivity_htn_exclusion(df)
    or_htn = primary_logistic_regression(df_htn_excl)
    or_htn.to_csv(f"{OUT_DIR}/or_htn_exclusion.csv")

    # --- Sensitivity: Age ≥ 40 ---
    df_aged = sensitivity_age_restriction(df, min_age=40.0)
    or_aged = primary_logistic_regression(df_aged)
    or_aged.to_csv(f"{OUT_DIR}/or_age40_restriction.csv")

    # --- Multiple Imputation ---
    imp_datasets = multiple_imputation_race(df)
    mi_estimates = []
    for imp_df in imp_datasets:
        formula = (
            "Hospitalized ~ CVD + Age_Z + Age_Z2 + Female + "
            "Race_Black + Race_Other + Mech_Fall + Mech_MVC + "
            "CVD:Age_Z + CVD:Mech_Fall"
        )
        model = smf.glm(
            formula=formula,
            data=imp_df.dropna(subset=["Weight_Numeric", "Hospitalized"]),
            family=sm.families.Binomial(),
            freq_weights=imp_df.dropna(subset=["Weight_Numeric", "Hospitalized"])["Weight_Numeric"],
        ).fit()
        coef = model.params.get("CVD", np.nan)
        var = (model.bse.get("CVD", np.nan)) ** 2
        mi_estimates.append((coef, var))

    pooled = rubin_pooling(mi_estimates)
    pooled["OR"] = np.exp(pooled["estimate"])
    pooled["CI_Lower_OR"] = np.exp(pooled["ci_lower"])
    pooled["CI_Upper_OR"] = np.exp(pooled["ci_upper"])
    print(f"\nMultiple Imputation (Rubin's Rules) CVD aOR: "
          f"{pooled['OR']:.2f} "
          f"(95% CI {pooled['CI_Lower_OR']:.2f}–{pooled['CI_Upper_OR']:.2f})")
    pd.DataFrame([pooled]).to_csv(f"{OUT_DIR}/mi_pooled_results.csv", index=False)

    print(f"\nAll statistical outputs saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
