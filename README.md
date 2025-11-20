Project-BLD
============

Purpose
-------
Project-BLD is a modular toolkit for developing, training, and evaluating bazaar trading models with careful, leak-proof dataset construction and a canonical T+1 backtest engine. It collects utilities and production-ready plumbing from prototype notebooks, including:

- Leak-proof tabular dataset construction and directional label creation.
- Sequence windowing and a PyTorch-friendly `SequenceDataset` for temporal models (TCN, transformers).
- A contrastive "Bazaar" encoder and pretraining pipeline (InfoNCE + masked reconstruction).
- XGBoost edge model wrapper and TCN sequence model wrapper with consistent `BazaarModel` interface.
- Canonical T+1 backtest engine implementing tradability gates, impact caps, exposure limits, and sensible position sizing.
- Model registry, storage abstractions, and trainer scripts for common flows.

This repository is intended as a starting point for research & development on market edge models and synthetic backtests.

Requirements
------------
The project uses Python 3.8+ and the key dependencies are listed in `requirements.txt`. Major libraries include:

- pandas, numpy
- scikit-learn, xgboost
- torch 
- shap, optuna
- gymnasium, stable-baselines3 (if RL components are used)
- tensorboard, tqdm, rich

Install (Windows cmd example)
----------------------------
Open a cmd.exe shell in the repository root and run:

```bat
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Project layout (key files & directories)
---------------------------------------
- config.py
  - Central path constants and convenience helpers (ensure_directories).

- data/
  - `src/data/storage.py` — thin DataStorage wrapper (loads processed CSV by default).
  - `src/data/tabular_dataset.py` — leak-proof dataset construction and label creation.
  - `src/data/sequence_dataset.py` — sliding-window logic and `SequenceDataset` for PyTorch.
  - `src/data/ssl_window_dataset.py` and `src/data/make_features.py` — feature engineering and SSL window dataset.

- models/
  - `src/models/base.py` — `BazaarModel` abstract interface (fit/predict/save/load).
  - `src/models/registry.py` — model registry and loader helpers.
  - `src/models/bazaar_encoder.py` — transformer-style encoder used with contrastive pretraining.
  - `src/models/tabular/xgb_model.py` — XGBoost wrapper.
  - `src/models/seq/tcn_model.py` — TCN wrapper for sequence modeling.

- backtest/
  - `src/backtest/engine.py` — canonical backtest engine (T+1 semantics, tradability, impact caps).
  - `src/backtest/splits.py` — time-based splitting helpers.
  - `src/backtest/contracts.py` — `BacktestConfig` and `BacktestResult` dataclasses.

- trainers/
  - `src/trainers/train_xgb.py` — data load, XGB training, predict & backtest wrapper.
  - `src/trainers/train_tcn.py` — TCN training orchestration and backtest.
  - `src/trainers/pretrain_bazaar_encoder.py` — encoder pretraining loop.

- ensemble/
  - `src/ensemble/conductor.py` — meta-ensemble conductor (weights blending experts).

- scripts/
  - Notebook-derived scripts and utilities (data_preparer, callbacks, dump_embeddings, linear_xgb_probe, ...).

Quickstart / Examples
---------------------
1) Sanity import tests

```bat
python test_imports.py
python test_refactoring.py
```

These scripts perform lightweight import checks and simple functional smoke tests.

2) Build / prepare the canonical processed dataset

- The canonical processed dataset now lives at `data/processed/improved_normalized_labeled.parquet` (DuckDB-compatible Parquet). A legacy CSV at `data/processed/improved_normalized_labeled.csv` is kept for backward compatibility and will be rebuilt automatically.
- To run the two-pass feature pipeline (the repository exposes a wrapper):

```py
from src.features.data_preparer import build_full_processed_dataset
build_full_processed_dataset(raw_db='data/raw/bazaar.db')
```

Note: the wrapper expects the legacy `scripts/data_preparer.py` script to be importable. If you prefer, run the script directly or adapt the wrapper to your environment.

### Processed dataset storage & IO upgrades

- DuckDB handles all processed dataset IO (storage abstraction, trainers, comparison scripts). Consumers transparently read from Parquet via DuckDB, falling back to CSV only if needed.
- The CSV → Parquet migration shrinks disk usage from 34.73 GB to 10.16 GB (≈71% reduction) thanks to columnar encoding + ZSTD compression.
- DuckDB read benchmarks on the first 1,000,000 rows show ~3.4× faster ingestion (CSV 24.29 s → Parquet 7.22 s) with identical downstream DataFrames and labels.

3) Train XGBoost and run backtest (example)

```py
from src.trainers.train_xgb import train_and_backtest_xgb
train_and_backtest_xgb(n_estimators=100, max_depth=6)
```

Output artifacts are written under `outputs/xgb/` (model, predictions, summary JSON).

4) Train TCN and run backtest (example)

```py
from src.trainers.train_tcn import train_and_backtest
train_and_backtest(epochs=3, window=64)
```

Output artifacts are written under `outputs/tcn/`.

5) Pretrain encoder & dump embeddings

- Pretrain the contrastive encoder (writes checkpoint according to encoder Paths config):

```py
from src.trainers.pretrain_bazaar_encoder import pretrain
pretrain()
```

- Dump embeddings for downstream probes:

```py
from src.scripts.dump_embeddings import dump
dump()
```

API notes (useful entry points)
------------------------------
- Dataset & features
  - `src.data.storage.DataStorage` — .load_full_table(), .load_features_and_labels(), iter_item_windows()
  - `src.data.tabular_dataset.load_leak_proof_dataset()` — returns (df, y_dir, y_enc, feature_cols)
  - `src.data.sequence_dataset.SequenceDataset` — PyTorch Dataset for sequence models

- Models & training
  - All models implement `BazaarModel` interface (fit, predict, save, load)
  - `src.models.registry.load_default_edge_model()` — load preset default model artifact

- Backtest
  - `src.backtest.engine.run_backtest(df, ...)` — canonical backtest; `df` must include timestamp, item, mid_price, pred_label, pred_proba_buy

Structure & design decisions
---------------------------
- Strict T+1 execution semantics and tradability checks in the backtest to avoid lookahead and to mimic realistic exchange microstructure constraints.
- Leak-proof feature selection: passthrough columns and target_* columns are excluded from model features to prevent label leakage.
- Trainer scripts mirror notebook flows to make reproducing experiments straightforward.
- Minimal registry for model artifacts (add entries and paths in `src/models/registry.py`).

Testing & verification
----------------------
- Quick checks: `test_imports.py`, `test_refactoring.py`, and `src/verify/refactor_checks.py` provide smoke tests.
- Recommended: add pytest tests for dataset label creation, sequence windowing, and backtest invariants (T+1, no target leakage).


Next steps / TODOs
------------------
- Add a CI pipeline with unit tests and lint checks.
- Expand the model registry to inclue more models and pretrained artifacts.
- Implement `Conductor.update()` online learning (placeholder in `src/ensemble/conductor.py`).
- Provide a small sample dataset or scripted data generator for end-to-end unit tests.

Contact / Notes
---------------
- This repository is derived from prototype notebooks and intended as a research platform. If you need help reproducing experiments, start with the trainers and backtest on small subsets of your processed CSV.



