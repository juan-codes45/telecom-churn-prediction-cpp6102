"""
task2ii_sort.py  --  TASK 2(ii): SORT
Order customers by tenure and charges, and build a retention watch-list.
"""
import pandas as pd
from src import config


def sort_by_tenure_and_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Newest customers first; within the same tenure, highest bill first."""
    return df.sort_values(["tenure", "MonthlyCharges"], ascending=[True, False])


def retention_watchlist(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Retained month-to-month customers who are newest and highest-billed:
    the accounts that could still be lost."""
    active = df.query("Churn == 'No' and Contract == 'Month-to-month'")
    cols = [config.ID_COL, "tenure", "MonthlyCharges", "service_load", "Contract"]
    return sort_by_tenure_and_charges(active).head(n)[cols]


def validation_sorts(df: pd.DataFrame, n: int = 100) -> dict:
    """Churn rate among the n longest-tenured and n highest-billed customers."""
    longest = df.sort_values("tenure", ascending=False).head(n)
    biggest = df.sort_values("MonthlyCharges", ascending=False).head(n)
    return {
        f"top{n}_tenure_churn_pct": round(float(longest[config.TARGET_FLAG].mean()) * 100, 1),
        f"top{n}_charges_churn_pct": round(float(biggest[config.TARGET_FLAG].mean()) * 100, 1),
    }