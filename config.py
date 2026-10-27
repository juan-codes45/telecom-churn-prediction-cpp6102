"""
config.py
Central settings shared by every module: paths, random seed and column groups.
Change a value here and the whole project follows.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "telco.csv"
OUTPUT_DIR = ROOT / "outputs"

SEED = 42            # reproducibility
TEST_SIZE = 0.20     # 80/20 stratified split
CV_FOLDS = 5

TARGET = "Churn"            # original Yes/No column
TARGET_FLAG = "ChurnFlag"   # 1 = churned, 0 = retained (created in task1_load_dataset)
ID_COL = "customerID"

# Services counted when building the 'service_load' proxy for call volume
SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]
PROTECTION_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]

HIGH_VOLUME_QUANTILE = 0.75  # top quartile of service_load = "high volume"