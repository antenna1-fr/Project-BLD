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

A small sample dataset is provided in `Data/Processed/Smaller processed data.csv` so the
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
<<<<<< codex/set-up-architecture-for-rl-model
9. Evaluate the trained agent via
   `python Scripts/evaluate_rl_model.py`.

 xfh0n2-codex/set-up-architecture-for-rl-model
9. Evaluate the trained agent via
   `python Scripts/evaluate_rl_model.py`.

 Reinforcement-Learning
main

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

The project includes a Gymnasium environment (`Scripts/rl_env.py`) enabling
training RL agents on historical data. Build the dataset with
`prepare_rl_dataset.py` then train an agent with
`rl_train_template.py`.  The trainer uses PPO and periodically evaluates the
agent, saving the best model under `Outputs/rl_model.zip`.  The environment is
kept simple so the repository can run quickly but it provides a solid base for
<<<<< codex/set-up-architecture-for-rl-model
=======
set-up-architecture-for-rl-model
main
experimentation.  After training, run `evaluate_rl_model.py` which leverages
`run_simulation.py` to backtest the agent using the same trading logic as the
baseline.  The `setup.sh` script installs all necessary packages, including
`gymnasium` and Stable-Baselines3.
codex/set-up-architecture-for-rl-model
experimentation.  The `setup.sh` script installs all necessary packages,
including `gymnasium` and Stable-Baselines3.
Reinforcement-Learning
main

