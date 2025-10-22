from pathlib import Path


# Base directory of the repository
BASE_DIR = Path(__file__).resolve().parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Outputs directories
OUTPUTS_DIR = BASE_DIR / "outputs"
MODEL_QUALITY_DIR = BASE_DIR / "model_quality"
TRADING_DIR = OUTPUTS_DIR / "trading"
TCN_TRADING_DIR = TRADING_DIR / "TCN_trading"
XGB_TRADING_DIR = TRADING_DIR / "XGB_trading"

# Default file paths
DB_PATH = RAW_DIR / "bazaar.db"
PROCESSED_CSV = PROCESSED_DIR / "improved_normalized_labeled.csv"
PREDICTIONS_CSV = OUTPUTS_DIR / "xgb_predictions.csv"
XGB_TRADE_LOG_CSV = XGB_TRADING_DIR / "XGB_trade_log.csv"
XGB_CONFUSION_MATRIX_PLOT = MODEL_QUALITY_DIR / "xgb_confusion_matrix_xgbcv.png"
TCN_CONFUSION_MATRIX_PLOT = MODEL_QUALITY_DIR / "tcn_confusion_matrix_xgbcv.png"
TCN_TRADE_LOG_CSV = TCN_TRADING_DIR / "TCN_trade_log.csv"


# RL artifacts
RL_DATASET_CSV = PROCESSED_DIR / "rl_dataset.csv"  # processed features for RL
RL_TRADE_LOG_CSV = TRADING_DIR / "rl_trade_log.csv"
RL_MODEL_PATH = OUTPUTS_DIR / "rl_model.zip"

# Feature flags
ENABLE_OPTUNA = True  # Enable Optuna hyperparameter tuning for XGBoost


def ensure_directories() -> None:
    """Create expected directories if they do not exist.

    Safe to call multiple times. Keeps repo structure consistent before first run.
    """
    dirs = [
        DATA_DIR,
        RAW_DIR,
        PROCESSED_DIR,
        OUTPUTS_DIR,
        MODEL_QUALITY_DIR,
        TRADING_DIR,
        TCN_TRADING_DIR,
        XGB_TRADING_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def project_path(*parts: str) -> Path:
    """Convenience for building paths under the repo root.

    Example: project_path("Outputs", "new.csv")
    """
    return BASE_DIR.joinpath(*parts)

