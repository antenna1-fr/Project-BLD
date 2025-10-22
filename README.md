# Bazaar Laplace's Demon (BLD)

Quantitative trading on the Hypixel Bazaar using classical ML and RL.

## Structure
- `Data/`: Input data
  - `Processed/`: Processed datasets (sample `rl_dataset.csv` included)
- `Scripts/`: Data prep, modeling, simulation, RL utilities
- `Model_Quality/`: Evaluation plots and diagnostics
- `Outputs/`: Model predictions and trade logs
- `bld/`: Python package (data, features, models, RL)
- `notebooks/`: Prototyping notebooks

## Quickstart
1) Create env and install deps
- Windows (PowerShell): `./setup.ps1`
- macOS/Linux: `bash setup.sh`

2) Configure paths and API key
- Review `config.py` and adjust directories as needed
- Set Hypixel API key for data collection: `HYPX_API_KEY=<your_key>`

3) Typical workflow
- Collect snapshots (optional): `python Scripts/Snapshot.py`
- Prepare features: `python "Scripts/Data Preparer.py"`
- Train XGBoost: `python Scripts/Model.py` → writes `Outputs/xgb_predictions.csv`
- Backtest: `python Scripts/run_simulation.py`
- RL dataset: `python Scripts/prepare_rl_dataset.py`
- Train RL agent: `python Scripts/train_eval_rl.py --timesteps 500000`

## Configuration
Centralized paths live in `config.py` (data, outputs, plots, logs). Use the provided helpers to create folders before first run. Example:
```python
from config import ensure_directories
ensure_directories()
```

Environment variables
- `HYPX_API_KEY`: Hypixel API key for `Snapshot.py` or `bld.data.snapshot`

## Notes on dependencies
- Torch and GPU: If you need CUDA, install the matching `torch` build from pytorch.org; XGBoost GPU requires a CUDA-enabled build as well.
- Parquet: `pyarrow` enables fast parquet IO if you use it in datasets.

## Outputs
- Predictions: `Outputs/xgb_predictions.csv`, RL artifacts under `Outputs/`
- Quality: confusion matrices and diagnostics in `Model_Quality/`

## Troubleshooting
- Paths: If files are not found, verify `config.py` points to your data directories and call `ensure_directories()`.
- Packages: If import errors occur, re-run setup and check CUDA-specific installs.

