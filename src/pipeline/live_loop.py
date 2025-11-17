# src/pipeline/live_loop.py
"""
Minimal offline 'live loop' using historical data.

This module orchestrates the full Symphony pipeline:
  data → features → model(s) → ensemble → allocation → execution / backtest

This is the dry run that will later evolve into:
  - real-time ingestion
  - multiple experts
  - RL policy
"""

from typing import Dict, Optional
from pathlib import Path
import sys

import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.storage import DataStorage
from src.models.tabular.xgb_model import XGBEdgeModel
from src.ensemble.conductor import Conductor
from src.portfolio.allocator import alloc_greedy
from src.execution.slippage_model import SimpleCostModel


def run_offline_cycle(
    data_path: Optional[Path] = None,
    max_gross_exposure: float = 1000.0,
    per_item_cap: float = 50.0
) -> Dict[str, float]:
    """
    Minimal offline 'live loop' using historical data.

    This demonstrates the full pipeline integration:
    1. Load data via DataStorage
    2. Train/load a baseline model (XGB)
    3. Generate predictions
    4. Apply cost model
    5. Use Conductor to blend (single expert for now)
    6. Allocate positions via greedy allocator

    Args:
        data_path: Optional path to data file (uses config default if None)
        max_gross_exposure: Maximum total position size
        per_item_cap: Maximum position size per item

    Returns:
        Dictionary mapping item IDs to position sizes
    """
    # 1) Load data
    print("[pipeline] Loading data...")
    storage = DataStorage(data_path)
    df = storage.load_full_table()

    if df.empty:
        print("[pipeline] No data loaded!")
        return {}

    print(f"[pipeline] Loaded {len(df)} rows, {len(df.columns)} columns")

    # 2) Identify feature columns (adapt to your actual schema)
    # This is a heuristic - in production, you'd have explicit feature lists
    feature_cols = [c for c in df.columns if c not in {
        'item', 'timestamp', 'mid_price', 'tradable',
        'target_min_abs', 'target_max_abs', 'target_min_rel', 'target_max_rel',
        'target_q_up_abs', 'target_q_dn_abs', 'target_q_up_rel', 'target_q_dn_rel'
    }]

    # Filter to numeric features only
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]

    if not feature_cols:
        print("[pipeline] No feature columns found!")
        return {}

    print(f"[pipeline] Using {len(feature_cols)} features")

    # Create a simple binary label if needed (example: up if target_max_rel > threshold)
    if 'label_up' not in df.columns and 'target_max_rel' in df.columns:
        df['label_up'] = (df['target_max_rel'] > 0.01).astype(int)

    label_col = 'label_up' if 'label_up' in df.columns else 'target_max_rel'

    # Drop rows with NaN in features or labels
    valid_rows = df[feature_cols + [label_col]].notna().all(axis=1)
    df = df[valid_rows].copy()
    print(f"[pipeline] After dropping NaN: {len(df)} rows")

    if len(df) < 100:
        print("[pipeline] Insufficient data for modeling")
        return {}

    # 3) Fit or load XGB baseline (W1)
    print("[pipeline] Training XGB model...")
    model = XGBEdgeModel(
        feature_cols=feature_cols,
        label_col=label_col,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )

    # Use first 80% for training (simple split)
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    model.fit(train_df)
    print("[pipeline] XGB training complete")

    # 4) Generate predictions on test set
    print("[pipeline] Generating predictions...")
    prob_up = model.predict(test_df, return_proba=True)
    test_df = test_df.copy()
    test_df['prob_up'] = prob_up

    # Convert probability to edge estimate (basis points)
    # Simple mapping: edge_bp = (prob - 0.5) * scaling_factor
    test_df['edge_bp'] = (test_df['prob_up'] - 0.5) * 100  # Max ±50bp edge

    # 5) Apply cost model
    print("[pipeline] Applying cost model...")
    cost_model = SimpleCostModel(fee_bp=10.0)

    # Get spread if available, else use a default
    spread_bp = test_df.get('rel_qspread', 0.01) * 10000  # Convert to bp
    if isinstance(spread_bp, pd.Series):
        spread_bp = spread_bp.fillna(10.0)
    else:
        spread_bp = 10.0

    test_df['cost_bp'] = cost_model.predict_bp(
        spread_bp=spread_bp,
        rel_size=0.5,  # TODO: Make dynamic based on volume
        queue_util=0.5  # TODO: Use actual queue data if available
    )

    # 6) Conductor: blend experts (single expert for now)
    print("[pipeline] Running conductor...")
    conductor = Conductor(n_experts=1, edge_buffer_bp=1.0)

    # For single expert, just pass through with buffer and costs subtracted
    test_df['exp_net_edge_bp'] = test_df['edge_bp'] - test_df['cost_bp'] - 1.0

    # 7) Allocator: greedy allocation on most recent snapshot
    print("[pipeline] Allocating positions...")

    # Take the most recent data (last 100 items for example)
    snapshot = test_df.tail(100).copy()

    # Filter to items with positive expected edge
    snapshot = snapshot[snapshot['exp_net_edge_bp'] > 0]

    if len(snapshot) == 0:
        print("[pipeline] No positive edge opportunities found")
        return {}

    items = snapshot['item'].values if 'item' in snapshot.columns else np.arange(len(snapshot))
    exp_edge = snapshot['exp_net_edge_bp'].values

    # Crude risk proxy: proportional to absolute edge
    risk_bp = 10.0 * np.abs(exp_edge)

    positions = alloc_greedy(
        items=items,
        exp_edge_bp=exp_edge,
        risk_bp=risk_bp,
        max_gross_exposure=max_gross_exposure,
        per_item_cap=per_item_cap,
    )

    # Filter out zero positions
    positions = {k: v for k, v in positions.items() if v > 0}

    print(f"[pipeline] Allocated {len(positions)} positions")
    print(f"[pipeline] Total exposure: {sum(positions.values()):.2f}")

    return positions


def main():
    """Run the offline pipeline as a standalone script."""
    print("="*60)
    print("Symphony Pipeline - Offline Cycle")
    print("="*60)

    positions = run_offline_cycle()

    print("\n" + "="*60)
    print("FINAL POSITIONS")
    print("="*60)

    if positions:
        for item, size in sorted(positions.items(), key=lambda x: -x[1])[:10]:
            print(f"{item:30s}: {size:8.2f}")
        if len(positions) > 10:
            print(f"... and {len(positions) - 10} more positions")
    else:
        print("No positions allocated")

    print("="*60)


if __name__ == "__main__":
    main()


__all__ = ['run_offline_cycle']

