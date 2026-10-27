"""
task2i_search.py  --  TASK 2(i): SEARCH
Identify customers with high service volume.
The dataset has no call-log column, so 'service_load' (number of subscribed
services) is used as a documented proxy for support demand.
"""
import pandas as pd
from src import config


def add_service_load(df: pd.DataFrame) -> pd.DataFrame:
    """Add service_load = count of subscribed services (0-9)."""
    df = df.copy()
    df["service_load"] = (
        df[config.SERVICE_COLS].eq("Yes").sum(axis=1)
        + (df["InternetService"] != "No").astype(int)
    )
    return df


def high_volume_threshold(df: pd.DataFrame) -> float:
    return float(df["service_load"].quantile(config.HIGH_VOLUME_QUANTILE))


def high_volume_customers(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean-mask search: customers at or above the 75th percentile."""
    return df[df["service_load"] >= high_volume_threshold(df)]


def support_gap_segment(df: pd.DataFrame) -> pd.DataFrame:
    """Compound search with DataFrame.query():
    high load AND month-to-month AND no tech support."""
    q = high_volume_threshold(df)
    return df.query("service_load >= @q and Contract == 'Month-to-month' and TechSupport != 'Yes'")


def segment_table(df: pd.DataFrame) -> pd.DataFrame:
    """Churn rate and size for each searched segment (Table 2 of the paper)."""
    hv = high_volume_customers(df)
    lv = df.drop(hv.index)
    segs = {
        "All customers": df,
        "High service load": hv,
        "Lower service load": lv,
        "Support gap (high load, month-to-month, no tech support)": support_gap_segment(df),
    }
    return pd.DataFrame(
        [(k, len(v), round(v[config.TARGET_FLAG].mean() * 100, 1)) for k, v in segs.items()],
        columns=["segment", "n", "churn_rate_pct"],
    )