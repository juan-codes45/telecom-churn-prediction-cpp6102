"""
task2ii_sort.py  --  TASK 2(ii): SORT
Order customers by tenure (Subscription_Length) and charges, and build a
retention watch-list of high-value customers who are still active.
"""
import pandas as pd
from src import config


def sort_by_tenure_and_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Newest customers first; within the same tenure, highest charge band first."""
    return df.sort_values(["Subscription_Length", "Charge_Amount"], ascending=[True, False])


def retention_watchlist(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Active customers (not churned) who have complained, ranked by customer
    value: the accounts most worth saving."""
    active = df.query("Churn == 0 and Complains == 1")
    cols = [config.ID_COL, "Subscription_Length", "Charge_Amount", "Customer_Value",
            "Call_Failure", "Status"]
    return (active.sort_values(["Customer_Value", "Subscription_Length"], ascending=[False, True])
            .head(n)[cols])


def validation_sorts(df: pd.DataFrame, n: int = 100) -> dict:
    """Churn rate among the n longest-tenured and n highest-value customers."""
    longest = df.sort_values("Subscription_Length", ascending=False).head(n)
    valuable = df.sort_values("Customer_Value", ascending=False).head(n)
    return {
        f"top{n}_tenure_churn_pct": round(float(longest[config.TARGET_FLAG].mean()) * 100, 1),
        f"top{n}_value_churn_pct": round(float(valuable[config.TARGET_FLAG].mean()) * 100, 1),
    }


def sorted_preview(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """First n rows of the tenure-then-charges sort."""
    cols = [config.ID_COL, "Subscription_Length", "Charge_Amount", "Customer_Value", config.TARGET_FLAG]
    return sort_by_tenure_and_charges(df).head(n)[cols]