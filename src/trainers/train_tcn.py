# src/trainers/train_tcn.py
"""Trainer script for TCN: builds dataset, trains model, runs backtest, and saves artifacts.

Usage (as module):
    from src.trainers.train_tcn import train_and_backtest
    train_and_backtest()

This script is intentionally minimal and mirrors the notebook flow.
"""
from pathlib import Path
import logging
import json

import numpy as np
import pandas as pd

from src.data.tabular_dataset import load_leak_proof_dataset
from src.data.sequence_dataset import build_item_sequence_indices
from src.models.seq.tcn_model import TCNSequenceModel
from src.backtest.engine import run_backtest

LOG = logging.getLogger(__name__)


def train_and_backtest(
    storage=None,
    model_out: str = "outputs/tcn/tcn_model.pt",
    epochs: int = 3,
    window: int = 64,
    stride: int = 4,
    val_frac: float = 0.05,
    verbose: bool = True,
):
    # 1) Load canonical dataset
    df, y_dir, y_enc, feature_cols = load_leak_proof_dataset(storage=storage, verbose=verbose)

    # 2) Simple val mask: hold out the last val_frac of each item
    n = len(df)
    rng = np.random.default_rng(0)
    # For deterministic small val set, choose randomly per-row
    val_mask = rng.choice([False, True], size=n, p=[1.0 - val_frac, val_frac])

    # 3) Build model config
    config = {
        "nfeat": len(feature_cols),
        "feature_cols": feature_cols,
        "window": window,
        "channels": [64, 64, 64],
        "kernel_size": 3,
        "dropout": 0.1,
        "num_classes": 3,
        "device": None,
    }

    model = TCNSequenceModel(config)

    # 4) Train
    model.fit(df=df, y_enc=y_enc, feature_cols=feature_cols, window=window, stride=stride,
              val_mask=val_mask, epochs=epochs, batch_size=None, lr=1e-3, weight_decay=1e-6, verbose=verbose)

    # 5) Save model
    out_path = Path(model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)
    LOG.info(f"Saved model to {out_path}")

    # 6) Predict across dataset
    preds_df = model.predict(df=df, feature_cols=feature_cols, window=window, stride=stride)
    merged = pd.concat([df.reset_index(drop=True), preds_df.reset_index(drop=True)], axis=1)

    # 7) Run canonical backtest
    trades, summary, equity = run_backtest(merged, pred_label_col="pred_label", pred_proba_buy_col="pred_proba_buy", verbose=verbose)

    # 8) Save predictions and summary
    preds_out = Path("outputs/tcn/tcn_predictions.csv")
    preds_out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(preds_out, index=False)
    with open(Path("outputs/tcn/tcn_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    LOG.info(f"Backtest summary: {summary}")
    return model, trades, summary, equity


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    train_and_backtest()

