# Bazaar Laplace's Demon (BLD)

This repository analyzes Hypixel Bazaar data in order to research trading strategies driven by machine learning.

## Project Goals

*(Add high level goals here)*

## Repository Layout

- `Data/` – raw and processed CSV data
- `Scripts/` – data collection, preparation, modelling and backtesting scripts
- `Model_Quality/` – evaluation plots and diagnostics
- `Outputs/` – model predictions and trade logs

A small sample dataset is provided in `Data/Processed/Smaller processed data.csv` so the
scripts can be executed without fetching new data.

## Getting Started

1. Install Python requirements.
2. Review `config.py` and adjust any paths if you wish to store data elsewhere.
3. Collect data with `Scripts/Snapshot.py` (requires your Hypixel API key).
4. Prepare features with `Scripts/Data Preparer.py`.
5. Train the XGBoost model using `Scripts/Model.py` which saves predictions to
   `Data/xgb_predictions.csv`.
6. Backtest these predictions via `python Scripts/run_simulation.py`.

## Configuration

All file locations are defined in `config.py`. By default paths are relative to the
repository root:

```python
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
RAW_DIR = DATA_DIR / "Raw"
PROCESSED_DIR = DATA_DIR / "Processed"
```

Adjust these variables if you want to place the database or outputs in different
directories.

## Next Steps

- Populate the SQLite database with a larger history of snapshots.
- Tweak feature engineering and model parameters to improve profitability.
- Consider more advanced strategies such as reinforcement learning.

