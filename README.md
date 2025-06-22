# Bazaar Laplace's Demon (BLD)

Bazaar Laplace's Demon (BLD) analyzes Hypixel Bazaar market data and applies machine learning to develop profitable trading strategies.

## Project Goals
- Develop a medium‑term strategy that trades automatically for profit.
- Explore short‑term arbitrage strategies for active players.

## Repository Layout
- `Data/` – raw and processed CSV data. A small sample RL dataset is included at `Data/Processed/rl_dataset.csv`.
- `Scripts/` – data collection, feature preparation, modelling and backtesting scripts.
- `Model_Quality/` – evaluation plots and diagnostics.
- `Outputs/` – saved model predictions and trade logs.

## Getting Started
1. Install dependencies:
   ```bash
   bash setup.sh
   ```
2. Review `config.py` to adjust any file paths.
3. (Optional) Collect raw snapshots via `python Scripts/Snapshot.py` (requires a Hypixel API key).
4. Generate the training dataset using `python "Scripts/Data Preparer.py"`.
5. Train the XGBoost model with `python Scripts/Model.py`. Predictions are saved to `Outputs/xgb_predictions.csv`.
6. Backtest predictions using `python Scripts/run_simulation.py`.
7. Prepare the RL dataset with `python Scripts/prepare_rl_dataset.py` (a sample is already provided).
8. Train and evaluate the RL agent:
   ```bash
   python Scripts/train_eval_rl.py --timesteps 5000
   ```

## Configuration
Paths for data and outputs are defined in `config.py`:
```python
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
RAW_DIR = DATA_DIR / "Raw"
PROCESSED_DIR = DATA_DIR / "Processed"
```
Adjust these variables if you wish to store data elsewhere.

## Reinforcement Learning
`Scripts/train_eval_rl.py` provides a Gymnasium environment that manages multiple bazaar items. After generating the dataset, run the script to train a PPO agent and evaluate it. Results, including the trained model and trade log, are written to `Outputs/`.
