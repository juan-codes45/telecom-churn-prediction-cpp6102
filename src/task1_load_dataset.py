"""
task1_load_dataset.py  --  TASK 1: Load the Dataset
Reads Customer_Churn.csv, tidies the column names, creates a row_id (the file
has no customer ID) and removes exact duplicate rows.
"""
import pandas as pd
from src import config


def load_data(path=config.DATA_PATH) -> pd.DataFrame:
    """Read the CSV and apply minimal cleaning."""
    df = pd.read_csv(path)
    # 'Call  Failure' has a double space -> 'Call_Failure'
    cleaned = df.columns.str.strip().str.replace(r"\s+", "_", regex=True)
    df.attrs["renamed_columns"] = int((cleaned != df.columns).sum())
    df.columns = cleaned
    df.insert(0, config.ID_COL, range(1, len(df) + 1))  # original CSV row number
    raw_rows = len(df)
    if config.DROP_DUPLICATES:
        data_cols = [c for c in df.columns if c != config.ID_COL]
        df = df.drop_duplicates(subset=data_cols).reset_index(drop=True)
    df.attrs["raw_rows"] = raw_rows
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """Show what was loaded: shape, first rows, data types, missing values, summary."""
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))
    print("\nData types:")
    print(df.dtypes.to_string())
    print("\nMissing values per column:", int(df.isna().sum().sum()), "in total")
    print("\nSummary statistics:")
    print(df.drop(columns=config.ID_COL).describe().T[["mean", "min", "50%", "max"]].round(2).to_string())


def profile(df: pd.DataFrame) -> dict:
    """Summary numbers reported in Table 1 of the paper."""
    raw = df.attrs.get("raw_rows", len(df))
    return {
        "raw_rows": int(raw),
        "renamed_columns": int(df.attrs.get("renamed_columns", 0)),
        "duplicates_removed": int(raw - len(df)),
        "rows": int(len(df)),
        "columns": int(df.shape[1]) - 1,  # exclude the row_id we added
        "missing_values": int(df.isna().sum().sum()),
        "churned": int(df[config.TARGET_FLAG].sum()),
        "retained": int((df[config.TARGET_FLAG] == 0).sum()),
        "churn_rate_pct": round(float(df[config.TARGET_FLAG].mean()) * 100, 2),
    }