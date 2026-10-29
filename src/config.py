"""
config.py
Central settings shared by every module: paths, random seed and column groups.
Change a value here and the whole project follows.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "Customer_Churn.csv"
OUTPUT_DIR = ROOT / "outputs"

SEED = 42            # reproducibility
TEST_SIZE = 0.20     # 80/20 stratified split
CV_FOLDS = 5

TARGET = "Churn"            # already coded 1 = churned, 0 = retained
TARGET_FLAG = "Churn"       # same column; kept as a separate name for readability
ID_COL = "row_id"           # the CSV has no customer ID, so task1 creates one

DROP_DUPLICATES = True      # 300 exact duplicate rows would leak between train and test

# Task 2(i): "service call volume" = number of calls made (Frequency_of_use)
CALL_VOLUME_COL = "Frequency_of_use"
HIGH_VOLUME_QUANTILE = 0.75  # top quartile = "high volume"

# Numeric codes that are really categories (one-hot encoded in the pipeline)
CODED_CATEGORICALS = ["Age_Group", "Tariff_Plan", "Status"]

# Task 3: put the key message in each chart title (for managers)
FIGURE_TITLES = True

# Task 2(iv) sensitivity check: columns removed to test reliance on Complains/Status
SENSITIVITY_DROP_COLS = ["Complains", "Status", "complained_inactive", "red_flags"]