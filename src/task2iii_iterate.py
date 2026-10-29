"""
task2iii_iterate.py  --  TASK 2(iii): ITERATE (feature engineering)
Row-wise iteration for rule-based logic plus vectorised ratio features.
"""
import numpy as np
import pandas as pd
from src import config


def _tenure_band(t: int) -> str:
    if t <= 24: return "0-24"
    if t <= 32: return "25-32"
    if t <= 38: return "33-38"
    return "39-47"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with nine engineered features added."""
    df = df.copy()
    lo = df[config.CALL_VOLUME_COL].quantile(0.25)
    hi = df[config.CALL_VOLUME_COL].quantile(config.HIGH_VOLUME_QUANTILE)

    # --- explicit iteration: per-customer business rules
    rows = []
    for _, r in df.iterrows():
        calls, secs = r[config.CALL_VOLUME_COL], r["Seconds_of_Use"]
        if calls == 0 or secs == 0:
            usage = "dormant"
        elif calls < lo:
            usage = "light"
        elif calls >= hi:
            usage = "heavy"
        else:
            usage = "moderate"
        red_flags = (int(r["Complains"] == 1) + int(r["Status"] == 2)
                     + int(r["Charge_Amount"] == 0) + int(usage in ("dormant", "light")))
        complained_inactive = int(r["Complains"] == 1 and r["Status"] == 2)
        rows.append((_tenure_band(r["Subscription_Length"]), usage, red_flags, complained_inactive))
    fe = pd.DataFrame(rows, index=df.index,
                      columns=["tenure_band", "usage_segment", "red_flags", "complained_inactive"])
    df = pd.concat([df, fe], axis=1)

    # --- vectorised ratio features
    calls = df[config.CALL_VOLUME_COL]
    df["calls_per_month"] = calls / df["Subscription_Length"]
    df["avg_call_seconds"] = (df["Seconds_of_Use"] / calls.replace(0, np.nan)).fillna(0)
    df["sms_per_call"] = df["Frequency_of_SMS"] / (calls + 1)
    df["failure_rate"] = df["Call_Failure"] / (calls + 1)
    df["value_per_month"] = df["Customer_Value"] / df["Subscription_Length"]
    return df


def flag_churn_rates(df: pd.DataFrame) -> dict:
    """Churn rate and size of the engineered-flag segments."""
    segs = {"complained_inactive": df[df["complained_inactive"] == 1],
            "red_flags_3_or_more": df[df["red_flags"] >= 3],
            "dormant_usage": df[df["usage_segment"] == "dormant"]}
    return {k: {"n": int(len(v)),
                "churn_rate_pct": round(float(v[config.TARGET_FLAG].mean()) * 100, 1)}
            for k, v in segs.items()}