from pathlib import Path

# Base directory of the repository
BASE_DIR = Path(__file__).resolve().parent

# Data directories
DATA_DIR = BASE_DIR / "Data"
RAW_DIR = DATA_DIR / "Raw"
PROCESSED_DIR = DATA_DIR / "Processed"

# Outputs directories
OUTPUTS_DIR = BASE_DIR / "Outputs"
MODEL_QUALITY_DIR = BASE_DIR / "Model_Quality"
TRADING_DIR = OUTPUTS_DIR / "Trading"

# Default file paths
DB_PATH = RAW_DIR / "bazaar.db"
PROCESSED_CSV = PROCESSED_DIR / "improved_normalized_labeled.csv"
PREDICTIONS_CSV = OUTPUTS_DIR / "xgb_predictions.csv"
TRADE_LOG_CSV = TRADING_DIR / "trade_log.csv"
CONFUSION_MATRIX_PLOT = MODEL_QUALITY_DIR / "confusion_matrix_xgbcv.png"

# File used for RL training (processed features, no labels)
RL_DATASET_CSV = PROCESSED_DIR / "rl_dataset.csv"

