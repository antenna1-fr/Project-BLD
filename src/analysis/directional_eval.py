# src/analysis/regression_outputs.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


def softmax_three_way_from_regression(
    y_pred: np.ndarray,
    k_softmax: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Map a continuous regression prediction y_pred into 3-way pseudo-probabilities:

        class -1: "down"
        class  0: "flat"
        class +1: "up"

    via a softmax over scores [-k*y, 0, +k*y].

    Returns
    -------
    Pm1, P0, Pp1 : np.ndarray
        Probabilities for (-1, 0, +1), all float32.
    """
    y_pred = np.asarray(y_pred, dtype="float32")
    k = float(k_softmax)

    score_up = k * y_pred
    score_dn = -k * y_pred
    score_neu = np.zeros_like(y_pred, dtype="float32")

    scores = np.stack([score_dn, score_neu, score_up], axis=1)  # [-1, 0, +1]
    scores -= scores.max(axis=1, keepdims=True)  # numerical stability
    exp_scores = np.exp(scores)
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    Pm1 = probs[:, 0].astype("float32")
    P0  = probs[:, 1].astype("float32")
    Pp1 = probs[:, 2].astype("float32")
    return Pm1, P0, Pp1


def build_preds_test_regression(
    df: pd.DataFrame,
    val_idx: np.ndarray,
    y_val_pred: np.ndarray,
    *,
    vol_est_col: str = "vol_est",
    k_softmax: float = 20.0,
) -> pd.DataFrame:
    """
    Build preds_test DataFrame for regression models:

      - includes meta: timestamp, item, mid_price, vol_est
      - adds pseudo 3-class probabilities: Pm1, P0, Pp1
      - adds y_pred (raw regression output)
      - adds score_buy, score_sell, score

    This is the regression analogue of the old classification preds_test.
    """
    META_COLS = ["timestamp", "item", "mid_price"]

    # Validation slice indices and meta
    idx = df.index[val_idx]
    test_meta = df.loc[idx, META_COLS].reset_index(drop=True)

    if vol_est_col in df.columns:
        test_meta["vol_est"] = df.loc[idx, vol_est_col].to_numpy()
    else:
        test_meta["vol_est"] = 1.0

    y_val_pred = np.asarray(y_val_pred, dtype="float32")
    Pm1, P0, Pp1 = softmax_three_way_from_regression(
        y_val_pred, k_softmax=k_softmax
    )

    proba_df = pd.DataFrame(
        {
            "Pm1": Pm1,
            "P0": P0,
            "Pp1": Pp1,
            "y_pred": y_val_pred,
        },
        index=test_meta.index,
    )

    preds_test = pd.concat([test_meta, proba_df], axis=1)

    # Trading scores
    p_pos = preds_test["Pp1"].astype(float).to_numpy()
    p_neg = preds_test["Pm1"].astype(float).to_numpy()

    preds_test["score_buy"] = p_pos - p_neg
    preds_test["score_sell"] = p_neg - p_pos

    if "score" not in preds_test.columns:
        preds_test["score"] = preds_test["score_buy"]

    return preds_test


def compute_regression_metrics(
    y_val: np.ndarray,
    y_val_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute basic regression metrics: RMSE and R^2.
    """
    y_val = np.asarray(y_val, dtype="float32")
    y_val_pred = np.asarray(y_val_pred, dtype="float32")

    mse = mean_squared_error(y_val, y_val_pred)
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_val, y_val_pred))

    return {"rmse": rmse, "r2": r2}


def build_backtest_pred_df_regression(
    df: pd.DataFrame,
    val_idx: np.ndarray,
    y_val: np.ndarray,
    y_val_pred: np.ndarray,
    primary_target: str,
    *,
    up_tau_reg: float = 0.02,
    dn_tau_reg: float = 0.02,
) -> pd.DataFrame:
    """
    Build a regression-based prediction DataFrame for backtesting:

      - maps continuous y_true / y_pred into discrete directions:
          -1 (short), 0 (flat), +1 (long)
      - uses thresholds on best-case return (UP_TAU_REG / DN_TAU_REG)
      - adds a smooth pseudo P(buy) from regression output
      - preserves PRIMARY_TARGET and y_pred for analysis
      - carries 'tradable' flag if present on df
    """
    y_true_cont = np.asarray(y_val, dtype="float32")
    y_pred_cont = np.asarray(y_val_pred, dtype="float32")

    up_tau = float(up_tau_reg)
    dn_tau = float(dn_tau_reg)

    true_dir = np.zeros_like(y_true_cont, dtype="int8")
    true_dir[y_true_cont >= up_tau] = 1
    true_dir[y_true_cont <= -dn_tau] = -1

    pred_dir = np.zeros_like(y_pred_cont, dtype="int8")
    pred_dir[y_pred_cont >= up_tau] = 1
    pred_dir[y_pred_cont <= -dn_tau] = -1

    # Smooth pseudo P(buy) from regression
    k = 4
    pred_proba_buy = 1.0 / (1.0 + np.exp(-k * y_pred_cont))
    pred_proba_buy = pred_proba_buy.astype("float32")

    idx = df.index[val_idx]
    ts = df.loc[idx, "timestamp"].to_numpy()
    items = df.loc[idx, "item"].to_numpy() if "item" in df.columns else ""
    mids = df.loc[idx, "mid_price"].to_numpy() if "mid_price" in df.columns else np.nan

    pred_df = pd.DataFrame(
        {
            "timestamp": ts,
            "item": items,
            "mid_price": mids,
            "true_dir": true_dir,
            "pred_dir": pred_dir,
            "pred_label": pred_dir,
            "pred_proba_buy": pred_proba_buy,
            primary_target: y_true_cont,
            "y_pred": y_pred_cont,
        }
    )

    if "tradable" in df.columns:
        pred_df["tradable"] = (
            df.loc[idx, "tradable"].astype("int8").to_numpy()
        )
    else:
        pred_df["tradable"] = 1

    return pred_df


def run_regression_outputs(
    *,
    df: pd.DataFrame,
    val_idx: np.ndarray,
    reg,
    X_val,
    y_val: np.ndarray,
    primary_target: str,
    output_predictions: Path | str,
    vol_est_col: str = "vol_est",
    k_softmax: float = 20.0,
    up_tau_reg: float = 0.02,
    dn_tau_reg: float = 0.02,
) -> Dict[str, object]:
    """
    High-level helper that:

      1) predicts y_val_pred from reg on X_val
      2) builds preds_test with pseudo 3-class probs and scores
      3) computes regression metrics
      4) builds backtest pred_df and writes it to CSV

    Returns
    -------
    dict with keys:
        'y_val_pred', 'preds_test', 'metrics', 'pred_df'
    """
    # 1) Predict
    Xv = np.asarray(X_val, dtype="float32")
    y_val_pred = reg.predict(Xv).astype("float32")

    # 2) preds_test (pseudo-probs + scores)
    preds_test = build_preds_test_regression(
        df=df,
        val_idx=val_idx,
        y_val_pred=y_val_pred,
        vol_est_col=vol_est_col,
        k_softmax=k_softmax,
    )

    # 3) metrics
    metrics = compute_regression_metrics(y_val=y_val, y_val_pred=y_val_pred)

    # 4) backtest pred_df
    pred_df = build_backtest_pred_df_regression(
        df=df,
        val_idx=val_idx,
        y_val=y_val,
        y_val_pred=y_val_pred,
        primary_target=primary_target,
        up_tau_reg=up_tau_reg,
        dn_tau_reg=dn_tau_reg,
    )

    out_path = Path(output_predictions)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_path, index=False)

    return {
        "y_val_pred": y_val_pred,
        "preds_test": preds_test,
        "metrics": metrics,
        "pred_df": pred_df,
        "output_path": out_path,
    }
