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
TCN_TRADING_DIR = TRADING_DIR/ "TCN_Trading"
XGB_TRADING_DIR = TRADING_DIR/ "XGB_Trading"

# Default file paths
DB_PATH = RAW_DIR / "bazaar.db"
PROCESSED_CSV = PROCESSED_DIR / "improved_normalized_labeled.csv"
PREDICTIONS_CSV = OUTPUTS_DIR / "xgb_predictions.csv"
XGB_TRADE_LOG_CSV = XGB_TRADING_DIR / "XGB_trade_log.csv"
XGB_CONFUSION_MATRIX_PLOT = MODEL_QUALITY_DIR / "xgb_confusion_matrix_xgbcv.png"
TCN_CONFUSION_MATRIX_PLOT = MODEL_QUALITY_DIR / "tcn_confusion_matrix_xgbcv.png"
TCN_TRADE_LOG_CSV = TCN_TRADING_DIR / "TCN_trade_log.csv"



# File used for RL training (processed features, no labels)
RL_DATASET_CSV = PROCESSED_DIR / "rl_dataset.csv"
# Trade log generated from RL agent evaluation
RL_TRADE_LOG_CSV = TRADING_DIR / "rl_trade_log.csv"

# Trade log generated from RL agent evaluation
RL_TRADE_LOG_CSV = TRADING_DIR / "rl_trade_log.csv"
# Location to save the trained RL model
RL_MODEL_PATH = OUTPUTS_DIR / "rl_model.zip"

ENABLE_OPTUNA = True  # Enable Optuna hyperparameter tuning for XGBoost