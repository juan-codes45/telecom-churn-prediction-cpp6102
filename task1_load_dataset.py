"""
task1_load_dataset.py  --  TASK 1: Load the Dataset
Reads the CSV, fixes the TotalCharges data-type problem and creates the
numeric target flag. Returns a clean DataFrame and a small profile dictionary.
"""
import pandas as pd
from src import config


def load_data(path=config.DATA_PATH) -> pd.DataFrame:
    """Read the CSV and apply minimal cleaning."""
    df = pd.read_csv(path)
    # TotalCharges is text because 11 rows are blank (customers with tenure 0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df[config.TARGET_FLAG] = (df[config.TARGET] == "Yes").astype(int)
    return df


def profile(df: pd.DataFrame) -> dict:
    """Summary numbers reported in Table 1 of the paper."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]) - 1,  # exclude the ChurnFlag we added
        "missing_total_charges": int(df["TotalCharges"].isna().sum()),
        "duplicates": int(df.duplicated().sum()),
        "churned": int(df[config.TARGET_FLAG].sum()),
        "retained": int((df[config.TARGET_FLAG] == 0).sum()),
        "churn_rate_pct": round(float(df[config.TARGET_FLAG].mean()) * 100, 2),
    }