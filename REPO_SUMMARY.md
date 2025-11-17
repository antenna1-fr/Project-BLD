# Project-BLD — Repository Summary

Generated: 2025-11-16

This document summarizes the repository's purpose, layout, and the responsibilities of the most important files and notebooks. It was created by scanning the repository and reading key files (README, config, core scripts, training & data-prep code, and the prototype notebooks).

---

## High-level purpose

Bazaar Laplace's Demon (BLD) is a quantitative-trading codebase that builds engineered features from Hypixel Bazaar snapshots, trains ML models (XGBoost, TCN; also encoder-based SSL pretraining), and evaluates strategies with a discrete, causal backtester and RL dataset preparation. The repo includes data-collection tooling (Hypixel API snapshots), feature engineering, classical ML training/HPO, time-series deep models (TCN), a patch/transformer-like Bazaar encoder (SSL pretraining), and backtesting/trading wrappers.


## Top-level inventory (concise)

- `README.md` — project description & quickstart.
- `AGENTS.md` — guidance for AI agents and project conventions.
- `config.py` — central path, artifact, and folder helpers plus `ensure_directories()`.
- `requirements.txt` — Python requirements for running the project.
- `treefinder.py` — small utility to print project tree (excludes venv, cache).
- `data/`
  - `raw/bazaar.db` — SQLite raw snapshots (if present).
  - `processed/` — prepared artifacts: `improved_normalized_labeled.csv`, `rl_dataset.csv`, `feature_scaler.pkl`, `preparer_meta.json`, `categorical_encoders.json`, `allowed_items.txt`, `tmp/`.
- `scripts/`
  - `TCN_prototype.ipynb`, `XGB_Prototype.ipynb` — two prototyping notebooks (training, diagnostics, backtester integration).
  - `data preparer.py` — heavy-duty feature engineering and scaler fitting pipeline (writes `improved_normalized_labeled.csv`, scaler, meta).
  - `snapshot.py` — Hypixel API collector that writes `bazaar_raw` and `orderbook_raw` tables into an SQLite DB.
  - `callbacks.py` — stable-baselines3 RL callback(s), CSV + TB logging for interval rewards.
  - `snapshot.py` and others write/read to/from `config.py` paths.
- `src/` — core package-like code
  - `configs/encoder_config.py` — typed configs for encoder/training I/O (DataCfg, FeatCfg, ModelCfg, Paths, etc.).
  - `data/` — `make_features.py` (alternate/smaller features builder), `ssl_window_dataset.py` (Dataset for SSL windows), `time_features.py`.
  - `models/` — `bazaar_encoder.py` (patchify + transformer encoder to produce latent embeddings and reconstructions).
  - `scripts/` — `dump_embeddings.py`, `linear_xgb_probe.py` (analyses that use embeddings + XGB/logistic probes).
  - `trainers/` — `pretrain_bazaar_encoder.py` (SSL pretraining loop for BazaarEncoder).
  - `utils/` — `losses.py`, `masking.py`, `schedule.py` (supporting training utilities).
- `outputs/` — predictions, model artifacts and subfolders
  - `outputs/xgb/xgb_best_params.json`, `xgb_model.ubj`, `xgb_predictions.csv`, `xgb_feature_importance_top25.png`
  - `outputs/tcn/` — several TCN model artifacts: `tcn_model.pt`, `tcn_model_best.pt`, `tcn_scripted.pt`, `tb/` (TB logs), and `tcn_predictions.csv`.
- `model_quality/` — stored diagnostic images: confusion matrices.
- `venv311/` — local virtual env (not part of distribution but present in workspace).


## Important files — what they do (reading summary)

Below are concise function-level summaries from the key files that were read.

### `README.md`
- Project description, quickstart, workflow, and notes on CUDA/XGBoost.
- Mentions central `config.py` and typical script sequence (snapshots → data prep → model → backtest → RL dataset).

### `AGENTS.md`
- Guidance for AI agents and conventions for automated code generation and PRs.
- Not critical to runtime; documentation for contributors/agents.

### `config.py`
- Declares central Paths (BASE_DIR, DATA_DIR, RAW_DIR, PROCESSED_DIR, OUTPUTS_DIR, MODEL_QUALITY_DIR) and default file names (DB, processed CSV, prediction CSV, trade-log paths).
- Exposes `ensure_directories()` to create the expected folder tree and `project_path()` to build paths relative to repository root.
- Also defines RL artifact paths and feature flags (e.g., ENABLE_OPTUNA).

### `treefinder.py`
- Small script to print the repo tree excluding common folders like `venv311`, `__pycache__`, `tmp`, `.git`, `db`, `tb`.


### `scripts/data preparer.py`
- Large, production-focused feature engineering pipeline.
- Connects to `data/raw/bazaar.db`, selects a set of liquid items (`top_items()`), fetches per-item bars + orderbook (`fetch_item()`), computes orderbook aggregates (`compute_orderbook_features()`), core market features (`compute_core_features()`), and a suite of forward-looking targets (`make_targets()`).
- Implements a 2-pass flow:
  - PASS-1: per-item feature building + parquet shards using ProcessPoolExecutor.
  - Fit a scaler using reservoir sampling across per-item shards.
  - PASS-2: scale per-item shards and write per-item CSV shards, then concatenate into `PROCESSED_CSV`.
- Writes artifacts to `data/processed`: scaler (`feature_scaler.pkl`), `preparer_meta.json`, and `allowed_items.txt`.
- Implements many numeric, robust, and leak-safe helpers (strict future-only windows, sliding-window quantile utilities, winsorization, per-row tradable gating).

Why it's important: this file produces the canonical feature CSV used by notebooks and model training (XGB/TCN).

### `scripts/snapshot.py`
- Hypixel API snapshot collector that queries `https://api.hypixel.net/skyblock/bazaar` with a (user-supplied) API key and writes raw rows into SQLite tables:
  - `bazaar_raw` — per-timestamp item-level summary (mid prices, volumes, moving-week stats).
  - `orderbook_raw` — flattened top-30 buy/sell levels per timestamp.
  - `election_snapshot` — wide table for Skyblock election data (mayor/minister/candidates).
- Runs as a loop with retries and WAL pragmas for reader performance.
- Intended to be run continuously for data collection.

### `scripts/callbacks.py`
- RL training callback `IntervalRewardLogger` (stable-baselines3 BaseCallback) that:
  - Logs interval reward(s) to TensorBoard scalars (rollout/*)
  - Appends rows to a CSV `interval_reward_log.csv` with wall_time, env_idx, key, value.

### `src/data/make_features.py`
- A smaller/alternate feature builder that uses `src.configs.encoder_config` typed config objects.
- Produces returns at multiple horizons, proxies for order imbalance, realized vol, and writes feature parquet via configured paths (DataCfg().features_parquet).
- Calls `add_time_columns` to inject time-of-day/dow trigonometric features.

### `src/data/ssl_window_dataset.py`
- `SSLWindowDataset` (Torch Dataset) that builds fixed-length windows per item for SSL pretraining.
- Stores/creates item2id mapping and writes `Paths().id_map_json`.
- Each __getitem__ returns (x[T,F], item_id, time_feature_embedding surrogate).

### `src/models/bazaar_encoder.py`
- `BazaarEncoder`: patchify-style encoder that
  - Splits windows into patches, projects patches, appends item embedding,
  - Uses transformer encoder to produce CLS global embedding (latent) and per-patch reconstructions.
  - Exposes outputs: z_global (normalized), recon (for masked reconstruction), and optional delta forecasting head.
- Designed for contrastive/SSL + masked reconstruction pretext tasks.

### `src/trainers/pretrain_bazaar_encoder.py`
- SSL pretraining loop:
  - Loads `SSLWindowDataset`, builds model, performs augmentations (gaussian noise + contiguous dropout), span masking, InfoNCE contrastive loss on CLS embeddings, masked reconstruction loss on per-patch reconstructions, optional forecasting regularizer.
  - Uses gradient accumulation, AMP, summary writer, checkpointing to `Paths().ckpt_out`.

### `src/scripts/dump_embeddings.py`
- Loads a trained `BazaarEncoder` checkpoint, runs inference over `SSLWindowDataset` to produce embeddings for each endpoint, merges with metadata (`item`, `timestamp`), and writes embeddings to `Paths().embeddings_out` parquet.

### `src/scripts/linear_xgb_probe.py`
- Example script that merges precomputed embeddings, hand-crafted features, and labels, then runs:
  - LogisticRegression linear probe on embeddings
  - XGBoost on manual features and hybrid features
- Prints classification reports for comparison.

### `outputs/xgb/xgb_best_params.json`
- Example hyperparameter JSON (Optuna best params) used for XGBoost training in the notebook.
- Shows GPU usage (`device: cuda`) and tuned tree params.

### `data/processed/preparer_meta.json` and `categorical_encoders.json`
- `preparer_meta.json` mirrors knobs used by `data preparer.py` (window lists, EMAs, OB buckets, scaler kind).
- `categorical_encoders.json` stores small categorical mapping tables used for election snapshot encoding.


## Notebooks

- `scripts/XGB_Prototype.ipynb` — Full XGBoost training & HPO notebook with leak-safe feature selection, asymmetric label construction, Optuna HPO, training, calibration, prediction assembly, backtester integration, diagnostics (opportunity coverage helper), plotting, and saving predictions for backtesting.
  - Contains a production-grade backtester (T+1, tradability-aware), calibration, and joblib+Optuna orchestration for backtest knob tuning.
  - Produces `outputs/xgb/...` artifacts, confusion matrices, and prediction CSVs used by the backtester.

- `scripts/TCN_prototype.ipynb` — TCN sequence model prototype for direct sequence-to-label modeling.
  - Includes data ingestion, label construction (same as XGB notebook), long-window dataset creation, model definitions (causal TCN with TemporalBlocks), training loop with checkpointing, temperature calibration, inference helpers that do per-item streaming inference to avoid PCIe traffic, and a backtester variant tuned for TCN output.
  - Produces `outputs/tcn/...` artifacts and diagnostic plots.

Why these matter: the notebooks are the authoritative, reproducible experiments for XGB and TCN pipelines and contain the production backtester & plotting logic.


## Notable repo issues & observations

- `scripts/data preparer.py` is named `data preparer.py` (contains a space). While functional, this can cause awkward imports/argument quoting; renaming to `data_preparer.py` would be more conventional.
- There was an earlier `list_dir` error when trying to list a `backtests/` path — some directory references exist in notebooks (`backtests/db`) but the top-level `backtests/` directory listing call failed in the environment the scanner used. If `backtests/` exists, I can list it again; otherwise notebooks create `backtests/db` at runtime.
- `venv311/` exists in the workspace. It should be removed or added to .gitignore if the repo is published.


---

End of summary.

