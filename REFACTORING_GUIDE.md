# Symphony Architecture - Refactoring Guide

## Overview

This document describes the modular refactoring of Project-BLD into the "Symphony" architecture. The refactoring transforms the notebook-based workflow into a clean, importable package structure that supports:

- Multiple expert models (XGB, TCN, transformers, GNNs)
- Meta-ensemble orchestration (Conductor)
- Portfolio allocation strategies
- Transaction cost modeling
- Unified backtesting framework
- Future RL integration

## New Directory Structure

```
src/
├── __init__.py
├── data/
│   ├── __init__.py
│   ├── storage.py              # Data access abstraction
│   ├── make_features.py        # (existing)
│   ├── ssl_window_dataset.py   # (existing)
│   └── time_features.py        # (existing)
├── features/
│   ├── __init__.py
│   └── data_preparer.py        # Clean API wrapper for feature pipeline
├── models/
│   ├── __init__.py
│   ├── base.py                 # BazaarModel abstract base class
│   ├── bazaar_encoder.py       # (existing, now properly integrated)
│   ├── tabular/
│   │   ├── __init__.py
│   │   └── xgb_model.py        # XGBoost wrapper (W1)
│   └── seq/
│       ├── __init__.py
│       └── tcn_model.py        # TCN wrapper (S1/S2 proto)
├── ensemble/
│   ├── __init__.py
│   └── conductor.py            # Meta-ensemble coordinator
├── portfolio/
│   ├── __init__.py
│   └── allocator.py            # Position sizing strategies
├── execution/
│   ├── __init__.py
│   └── slippage_model.py       # Cost and slippage models
├── backtest/
│   ├── __init__.py
│   ├── contracts.py            # Backtest config/result dataclasses
│   └── engine.py               # Unified backtest engine
├── pipeline/
│   ├── __init__.py
│   └── live_loop.py            # End-to-end orchestration
├── trainers/
│   ├── __init__.py
│   └── pretrain_bazaar_encoder.py  # (existing)
├── utils/
│   ├── __init__.py
│   ├── losses.py               # (existing)
│   ├── masking.py              # (existing)
│   └── schedule.py             # (existing)
├── scripts/
│   ├── __init__.py
│   └── dump_embeddings.py      # (existing)
└── rl/
    └── envs/                   # (existing, for future use)
```

## Core Components

### 1. Data Layer (`src/data/`, `src/features/`)

**DataStorage** (`src/data/storage.py`)
- Abstraction over processed data access
- Currently wraps CSV, easily switchable to Parquet/DuckDB
- Provides `load_full_table()`, `load_features_and_labels()`, `iter_item_windows()`

**Data Preparer** (`src/features/data_preparer.py`)
- Clean API wrapper around `scripts/data_preparer.py`
- Main function: `build_full_processed_dataset(raw_db, output_csv)`
- All downstream code uses this instead of ad-hoc CSV loading

### 2. Model Layer (`src/models/`)

**BazaarModel** (`src/models/base.py`)
- Abstract base class defining the model interface
- All models implement: `fit()`, `predict()`, `save()`, `load()`
- Ensures consistent interaction patterns across model types

**XGBEdgeModel** (`src/models/tabular/xgb_model.py`)
- Wraps XGBoost in the BazaarModel interface
- This is the **W1** slot in Symphony (tabular window model)
- Includes feature importance extraction

**TCNSequenceModel** (`src/models/seq/tcn_model.py`)
- Skeleton for TCN integration
- Will be **S1/S2** prototype (sequence models)
- TODO: Move TCN architecture from notebook

**BazaarEncoder** (`src/models/bazaar_encoder.py`)
- Already exists, now properly integrated
- Future **S1** with forecasting head

### 3. Ensemble Layer (`src/ensemble/`)

**Conductor** (`src/ensemble/conductor.py`)
- Meta-ensemble coordinator
- Blends predictions from multiple experts (S1, S2, W1, W2, etc.)
- Current: Simple weighted average
- Future: OnlineRidge + IsotonicRegression for adaptive weighting

### 4. Portfolio Layer (`src/portfolio/`)

**Allocators** (`src/portfolio/allocator.py`)
- `alloc_greedy()`: Risk-adjusted edge ranking
- `alloc_risk_parity()`: Equal risk contribution
- `alloc_mean_variance()`: TODO - Markowitz optimization

### 5. Execution Layer (`src/execution/`)

**SimpleCostModel** (`src/execution/slippage_model.py`)
- Linear cost model: `cost = fee + α·spread + β·size + γ·queue`
- Can be calibrated from historical execution data
- **AdaptiveCostModel**: TODO - Online learning version

### 6. Backtest Layer (`src/backtest/`)

**BacktestConfig** (`src/backtest/contracts.py`)
- Configuration dataclass for backtest parameters
- Defines fee_bp, slippage_bp, position limits, etc.

**BacktestResult** (`src/backtest/contracts.py`)
- Result dataclass with trades, equity curve, summary metrics

**run_backtest()** (`src/backtest/engine.py`)
- Unified T+1 backtest engine
- TODO: Integrate actual logic from XGB/TCN notebooks
- All models use this for consistent evaluation

### 7. Pipeline Layer (`src/pipeline/`)

**run_offline_cycle()** (`src/pipeline/live_loop.py`)
- End-to-end pipeline demonstration
- Flow: data → features → model → conductor → allocator
- This is the integration point for all components

## Usage Examples

### Example 1: Load Data

```python
from src.data.storage import DataStorage

storage = DataStorage()
df = storage.load_full_table()

# Or iterate over windows for sequence models
for item_id, window in storage.iter_item_windows(window_size=60):
    print(f"Processing {item_id}: {len(window)} rows")
```

### Example 2: Train XGB Model

```python
from src.models.tabular.xgb_model import XGBEdgeModel
from src.data.storage import DataStorage

storage = DataStorage()
df = storage.load_full_table()

# Define features and label
feature_cols = [c for c in df.columns if c.startswith('feat_')]
model = XGBEdgeModel(
    feature_cols=feature_cols,
    label_col='label_up',
    n_estimators=100,
    max_depth=6
)

# Train
model.fit(df)

# Predict
predictions = model.predict(df)

# Save
model.save(Path('outputs/xgb/my_model.pkl'))
```

### Example 3: Run Backtest

```python
from src.backtest.engine import run_backtest
from src.backtest.contracts import BacktestConfig

# Add predictions to DataFrame
df['xgb_pred_prob'] = model.predict(df)

# Configure backtest
cfg = BacktestConfig(
    label_col='label_up',
    prediction_col='xgb_pred_prob',
    timestamp_col='timestamp',
    item_id_col='item',
    fee_bp=10.0,
    min_edge_bp=1.0
)

# Run
result = run_backtest(df, cfg)
result.print_summary()
```

### Example 4: Full Pipeline

```python
from src.pipeline.live_loop import run_offline_cycle

# Run the complete pipeline
positions = run_offline_cycle(
    max_gross_exposure=1000.0,
    per_item_cap=50.0
)

print(f"Allocated {len(positions)} positions")
```

## Migration Path

### Phase 1: Infrastructure (DONE ✓)
- [x] Create package structure with `__init__.py` files
- [x] Rename `data preparer.py` → `data_preparer.py`
- [x] Create `src/data/storage.py`
- [x] Create `src/features/data_preparer.py` wrapper
- [x] Create model base class and XGB wrapper
- [x] Create backtest contracts and engine
- [x] Create ensemble/portfolio/execution modules
- [x] Create pipeline orchestration

### Phase 2: Notebook Migration (TODO)
1. **XGB Notebook** (`scripts/XGB_Prototype.ipynb`)
   - Replace data loading with `DataStorage`
   - Replace backtest code with `run_backtest()`
   - Use `XGBEdgeModel` wrapper

2. **TCN Notebook** (`scripts/TCN_prototype.ipynb`)
   - Move TCN architecture to `src/models/seq/tcn_model.py`
   - Implement `fit()` and `predict()` methods
   - Use shared backtest engine

3. **Encoder Training**
   - Already in `src/trainers/pretrain_bazaar_encoder.py`
   - Ensure it uses `DataStorage` for data access

### Phase 3: Symphony Integration (TODO)
1. Add multiple experts to Conductor
2. Implement OnlineRidge + IsotonicRegression
3. Add RL policy integration
4. Implement live trading loop

## Key Benefits

### 1. Modularity
- Each component has a single responsibility
- Easy to swap implementations (e.g., XGB → LightGBM)
- Clean separation between data, models, ensemble, execution

### 2. Testability
- Each module can be tested independently
- Consistent interfaces make mocking easy
- Backtest engine ensures reproducible evaluation

### 3. Extensibility
- Adding new models: implement `BazaarModel` interface
- Adding new allocators: add function to `allocator.py`
- Adding new cost models: extend `SimpleCostModel`

### 4. Maintainability
- No more notebook copy-paste
- Shared backtest logic means bugs fixed once
- Clear import paths: `from src.models.tabular import XGBEdgeModel`

### 5. Production Ready
- Everything is importable and callable
- No hidden state in notebooks
- Easy to deploy as a service or scheduled job

## Next Steps

### Immediate
1. Test the basic imports:
   ```bash
   python -c "from src.data.storage import DataStorage; print('OK')"
   python -c "from src.models.tabular.xgb_model import XGBEdgeModel; print('OK')"
   ```

2. Update one notebook to use the new structure
   - Start with XGB notebook as reference implementation

3. Move actual backtest logic from notebook to `src/backtest/engine.py`

### Short Term
1. Complete TCN model wrapper
2. Integrate encoder training with new data layer
3. Add unit tests for core modules
4. Add example scripts demonstrating each component

### Long Term
1. Implement full Conductor with online learning
2. Add RL policy integration
3. Add more model types (PatchTST, TFT, GNN)
4. Implement live trading infrastructure

## File Locations Reference

| Component | Location | Status |
|-----------|----------|--------|
| Data Access | `src/data/storage.py` | ✓ Created |
| Feature Pipeline | `src/features/data_preparer.py` | ✓ Created |
| Model Base | `src/models/base.py` | ✓ Created |
| XGB Model | `src/models/tabular/xgb_model.py` | ✓ Created |
| TCN Model | `src/models/seq/tcn_model.py` | ✓ Skeleton |
| Conductor | `src/ensemble/conductor.py` | ✓ Stub |
| Allocator | `src/portfolio/allocator.py` | ✓ Created |
| Cost Model | `src/execution/slippage_model.py` | ✓ Created |
| Backtest Engine | `src/backtest/engine.py` | ✓ Placeholder |
| Pipeline | `src/pipeline/live_loop.py` | ✓ Created |

## Configuration

The refactored code uses the existing `config.py` at the project root. New alias added:
- `RAW_DB_PATH = DB_PATH` for consistency

All modules automatically add the project root to `sys.path` for clean imports.

---

**Last Updated**: November 16, 2025
**Version**: 1.0 (Initial Refactoring Complete)

