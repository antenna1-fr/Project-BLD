# Bazaar Laplace's Demon (BLD)

This repository analyzes Hypixel Bazaar data in order to research trading strategies driven by machine learning.

## Project Goals
Use machine learning to generate a trading strategy that is profitable over the medium term while being hands off
Similarly generate short term aritrage strategy for active management

## Repository Layout

- `Data/` – raw and processed CSV data
- `Scripts/` – data collection, preparation, modelling and backtesting scripts
- `Model_Quality/` – evaluation plots and diagnostics
- `Outputs/` – model predictions and trade logs

A small sample dataset is provided in `Data/Processed/improved_normalized_labeled.csv` so the
scripts can be executed without fetching new data.

## Getting Started

1. Install Python requirements by running `bash setup.sh`.
2. Review `config.py` and adjust any paths if you wish to store data elsewhere.
3. Collect data with `Scripts/Snapshot.py` (requires your Hypixel API key).
4. Prepare features with `Scripts/Data Preparer.py`.
5. Train the XGBoost model using `Scripts/Model.py` which saves predictions to
   `Data/xgb_predictions.csv`.
6. Backtest these predictions via `python Scripts/run_simulation.py`.
7. Generate an RL-ready dataset using `python Scripts/prepare_rl_dataset.py`.
8. Train the reinforcement learning agent with
   `python Scripts/rl_train_template.py --timesteps 5000`.
9. Evaluate the trained agent via
   `python Scripts/evaluate_rl_model.py`.

9. Evaluate the trained agent via
   `python Scripts/evaluate_rl_model.py`.

 Reinforcement-Learning
main
8. Train **and** evaluate the reinforcement learning agent using
   `python Scripts/train_eval_rl.py --timesteps 5000`.

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

## Reinforcement Learning

Reinforcement learning uses a Gymnasium environment defined in
`Scripts/train_eval_rl.py`. First build the dataset with
`prepare_rl_dataset.py` then run the training script. The environment now
manages a portfolio of many bazaar items simultaneously and will automatically
close or adjust positions based on cross‑item trends. The script trains a PPO
agent and evaluates it on a held‑out portion of the data, reporting profit
statistics and feature usefulness. Results, including the trained model and
trade log, are saved under `Outputs/`.

