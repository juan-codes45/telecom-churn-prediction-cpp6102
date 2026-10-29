"""
task2i_search.py  --  TASK 2(i): SEARCH
Identify customers with high service call volume (Frequency_of_use, the number
of calls) and related risk segments, using boolean masks and DataFrame.query().
"""
import pandas as pd
from src import config


def high_volume_threshold(df: pd.DataFrame) -> float:
    return float(df[config.CALL_VOLUME_COL].quantile(config.HIGH_VOLUME_QUANTILE))


def high_volume_customers(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean-mask search: customers at or above the 75th percentile of calls."""
    return df[df[config.CALL_VOLUME_COL] >= high_volume_threshold(df)]


def high_failure_customers(df: pd.DataFrame) -> pd.DataFrame:
    """Customers in the top quartile of failed calls."""
    q = float(df["Call_Failure"].quantile(config.HIGH_VOLUME_QUANTILE))
    return df[df["Call_Failure"] >= q]


def complainers(df: pd.DataFrame) -> pd.DataFrame:
    return df.query("Complains == 1")


def support_gap_segment(df: pd.DataFrame) -> pd.DataFrame:
    """Compound search with DataFrame.query():
    complained AND high call failures AND the account is inactive (Status == 2)."""
    q = float(df["Call_Failure"].quantile(config.HIGH_VOLUME_QUANTILE))
    return df.query("Complains == 1 and Call_Failure >= @q and Status == 2")


def segment_table(df: pd.DataFrame) -> pd.DataFrame:
    """Churn rate and size for each searched segment (Table 2 of the paper)."""
    hv = high_volume_customers(df)
    segs = {
        "All customers": df,
        "High call volume (top quartile)": hv,
        "Lower call volume": df.drop(hv.index),
        "High call failures (top quartile)": high_failure_customers(df),
        "Filed a complaint": complainers(df),
        "Support gap (complaint + high failures + inactive)": support_gap_segment(df),
    }
    return pd.DataFrame(
        [(k, len(v), round(float(v[config.TARGET_FLAG].mean()) * 100, 1)) for k, v in segs.items()],
        columns=["segment", "n", "churn_rate_pct"],
    )


def high_volume_preview(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """The n customers with the most calls: the actual people the search found."""
    cols = [config.ID_COL, config.CALL_VOLUME_COL, "Seconds_of_Use", "Call_Failure",
            "Complains", config.TARGET_FLAG]
    return high_volume_customers(df).sort_values(config.CALL_VOLUME_COL, ascending=False).head(n)[cols]