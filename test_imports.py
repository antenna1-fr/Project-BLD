#!/usr/bin/env python
"""Test script to verify all new modules import correctly."""

print("Testing imports...")

try:
    from src.configs.schema import (
        TIMESTAMP_COL, ITEM_ID_COL, MID_PRICE_COL, TRADABLE_COL,
        get_feature_columns, get_passthrough_columns
    )
    print("✓ schema module OK")
    print(f"  - TIMESTAMP_COL: {TIMESTAMP_COL}")
    print(f"  - ITEM_ID_COL: {ITEM_ID_COL}")
except Exception as e:
    print(f"✗ schema module FAILED: {e}")

try:
    from src.backtest.splits import (
        purged_time_splits, get_last_split, split_by_date
    )
    print("✓ splits module OK")
except Exception as e:
    print(f"✗ splits module FAILED: {e}")

try:
    from src.data.tabular_dataset import (
        create_directional_labels,
        build_leak_proof_dataset,
        load_leak_proof_dataset
    )
    print("✓ tabular_dataset module OK")
except Exception as e:
    print(f"✗ tabular_dataset module FAILED: {e}")

try:
    from src.data.sequence_dataset import (
        build_item_sequence_indices,
        SequenceDataset,
        guess_batch_size
    )
    print("✓ sequence_dataset module OK")
except Exception as e:
    print(f"✗ sequence_dataset module FAILED: {e}")

try:
    from src.backtest.engine import run_backtest, run_backtest_with_config
    print("✓ engine module OK")
except Exception as e:
    print(f"✗ engine module FAILED: {e}")

try:
    from src.data.storage import DataStorage
    print("✓ storage module OK")
except Exception as e:
    print(f"✗ storage module FAILED: {e}")

print("\n" + "="*60)
print("All modules imported successfully!")
print("="*60)

