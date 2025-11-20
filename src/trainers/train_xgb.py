# src/trainers/train_xgb.py
"""Trainer script for XGB: builds dataset, trains XGBEdgeModel, runs backtest, and saves artifacts.

Usage:
    from src.trainers.train_xgb import train_and_backtest_xgb
    train_and_backtest_xgb()
"""
from pathlib import Path
import json
import logging

import numpy as np
import pandas as pd

from src.data.tabular_dataset import load_leak_proof_dataset
from src.models.tabular.xgb_model import XGBEdgeModel
from src.backtest.engine import run_backtest

LOG = logging.getLogger(__name__)


def train_and_backtest_xgb(
    storage=None,
    model_out: str = "outputs/xgb/xgb_model.joblib",
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    verbose: bool = True,
):
    # 1) Load canonical dataset
    df, y_dir, y_enc, feature_cols = load_leak_proof_dataset(storage=storage, verbose=verbose)

    # For XGB we'll train a simple classifier to predict upward move (y_dir==1)
    df = df.copy()
    df['label_up'] = (y_dir == 1).astype(int)

    label_col = 'label_up'

    # Drop rows with NaN in features or label
    valid = df[feature_cols + [label_col]].notna().all(axis=1)
    df = df.loc[valid].reset_index(drop=True)

    if len(df) < 50:
        raise RuntimeError("Insufficient data for XGB training")

    # simple split
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    model = XGBEdgeModel(feature_cols=feature_cols, label_col=label_col,
                         n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                         use_label_encoder=False, eval_metric='logloss')

    model.fit(train_df)

    out_path = Path(model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)
    LOG.info(f"Saved XGB model to {out_path}")

    # Predict
    prob_up = model.predict(test_df, return_proba=True)
    test_df = test_df.copy()
    test_df['pred_proba_buy'] = prob_up
    test_df['pred_label'] = (prob_up > 0.5).astype(int) * 2  # align encoding: 2 for buy, keep simple mapping

    trades, summary, equity = run_backtest(test_df, pred_label_col='pred_label', pred_proba_buy_col='pred_proba_buy', verbose=verbose)

    preds_out = Path('outputs/xgb/xgb_predictions.csv')
    preds_out.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(preds_out, index=False)
    with open(Path('outputs/xgb/xgb_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    LOG.info(f"XGB backtest summary: {summary}")
    return model, trades, summary, equity


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    train_and_backtest_xgb()

