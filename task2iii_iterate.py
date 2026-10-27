"""
task2iii_iterate.py  --  TASK 2(iii): ITERATE (feature engineering)
Row-wise iteration for rule-based logic plus vectorised arithmetic features.
Requires the 'service_load' column created in task2i_search.add_service_load().
"""
import numpy as np
import pandas as pd
from src import config


def _tenure_band(t: int) -> str:
    if t <= 12: return "0-12"
    if t <= 24: return "13-24"
    if t <= 48: return "25-48"
    return "49-72"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with nine engineered features added."""
    assert "service_load" in df.columns, "Run task2i_search.add_service_load() first"
    df = df.copy()

    # --- explicit iteration: per-customer business rules
    rows = []
    for _, r in df.iterrows():
        autopay = int("automatic" in r["PaymentMethod"].lower())
        fiber_no_support = int(r["InternetService"] == "Fiber optic"
                               and r["TechSupport"] != "Yes"
                               and r["OnlineSecurity"] != "Yes")
        protection = sum(r[c] == "Yes" for c in config.PROTECTION_COLS)
        rows.append((_tenure_band(r["tenure"]), autopay, fiber_no_support, protection))
    fe = pd.DataFrame(rows, index=df.index,
                      columns=["tenure_band", "autopay", "fiber_no_support", "protection_count"])
    df = pd.concat([df, fe], axis=1)

    # --- vectorised arithmetic features
    df["avg_monthly_spend"] = (df["TotalCharges"] / df["tenure"].replace(0, np.nan)
                               ).fillna(df["MonthlyCharges"])
    df["charge_vs_avg"] = df["MonthlyCharges"] - df["avg_monthly_spend"]
    df["charge_per_service"] = df["MonthlyCharges"] / df["service_load"].clip(lower=1)
    df["new_mtm_high_bill"] = (
        (df["tenure"] <= 12)
        & (df["Contract"] == "Month-to-month")
        & (df["MonthlyCharges"] > df["MonthlyCharges"].median())
    ).astype(int)
    return df


def flag_churn_rates(df: pd.DataFrame) -> dict:
    """Churn rate and size of the two compound-flag segments."""
    out = {}
    for flag in ["fiber_no_support", "new_mtm_high_bill"]:
        seg = df[df[flag] == 1]
        out[flag] = {"n": int(len(seg)),
                     "churn_rate_pct": round(float(seg[config.TARGET_FLAG].mean()) * 100, 1)}
    return out