# src/analysis/directional_eval.py

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from src.analysis.opportunity_coverage import opportunity_coverage  # your existing function


# ---------------------------
# 1) Build preds_test from proba
# ---------------------------
def build_validation_pred_frame_from_proba(
    df: pd.DataFrame,
    val_idx: np.ndarray,
    proba_val: np.ndarray,
    classes_dec: np.ndarray,
    *,
    meta_cols: Sequence[str] = ("timestamp", "item", "mid_price"),
    vol_col: str = "vol_est",
) -> pd.DataFrame:
    """
    Build preds_test for validation data given probabilities and decoded classes.

    - df: full feature/label DataFrame
    - val_idx: indices used for validation
    - proba_val: shape [N_val, num_classes], in encoded class order
    - classes_dec: decoded labels in original space (e.g. [-1, 0, 1])
    """
    idx = df.index[val_idx]
    test_meta = df.loc[idx, list(meta_cols)].reset_index(drop=True).copy()
    if vol_col in df.columns:
        test_meta["vol_est"] = df.loc[idx, vol_col].values
    else:
        test_meta["vol_est"] = 1.0

    # Columns like P-1, P0, P1 based on classes_dec
    proba_df = pd.DataFrame(proba_val, columns=[f"P{c}" for c in classes_dec])

    # Normalize names P-1 -> Pm1, P0 -> P0, P1 -> Pp1
    rename = {}
    for c in classes_dec:
        if c == -1:
            rename[f"P{c}"] = "Pm1"
        elif c == 0:
            rename[f"P{c}"] = "P0"
        elif c == 1:
            rename[f"P{c}"] = "Pp1"
    proba_df = proba_df.rename(columns=rename)

    for col in ["Pm1", "P0", "Pp1"]:
        if col not in proba_df.columns:
            proba_df[col] = 0.0

    preds_test = pd.concat([test_meta, proba_df], axis=1).reset_index(drop=True)
    assert {"timestamp", "item", "mid_price", "Pm1", "P0", "Pp1"}.issubset(preds_test.columns)

    return preds_test


# ---------------------------
# 2) Add score columns (model-agnostic)
# ---------------------------
def add_scores(preds_test: pd.DataFrame) -> pd.DataFrame:
    """
    Add score_buy, score_sell, score columns based on available
    P(+1), P(-1), P(0) variants. Works regardless of model.
    """
    def _get_first(df, names, default=np.nan):
        for n in names:
            if n in df.columns:
                return df[n].astype(float)
        return np.full(len(df), default, dtype=float)

    p_pos = _get_first(preds_test, ["Pp1", "P1", "P(1)", "prob_1", "proba_1"], default=np.nan)
    p_neg = _get_first(preds_test, ["Pm1", "P-1", "P(-1)", "prob_-1", "proba_-1"], default=np.nan)
    p_neu = _get_first(preds_test, ["P0", "P(0)", "prob_0", "proba_0"], default=np.nan)

    has_real_sell = np.isfinite(p_neg).any() and (np.nanmax(p_neg) > 0)

    if has_real_sell:
        preds_test["score_buy"] = p_pos - p_neg
        preds_test["score_sell"] = p_neg - p_pos
    else:
        preds_test["score_buy"] = p_pos - (p_neu if np.isfinite(p_neu).any() else 0.0)
        preds_test["score_sell"] = np.nan

    if "score" not in preds_test.columns:
        preds_test["score"] = preds_test["score_buy"]

    return preds_test


# ---------------------------
# 3) Opportunity coverage wrapper
# ---------------------------
def compute_opportunity_coverage(
    df: pd.DataFrame,
    val_idx: np.ndarray,
    preds_test: pd.DataFrame,
    *,
    horizon: int = 60,
    up_tau: float = 0.03,
    dn_tau: float = 0.03,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Validation-slice wrapper around opportunity_coverage.
    """
    prices_test = df.loc[val_idx, ["timestamp", "item", "mid_price"]].copy()
    opp_metrics, opp_joined = opportunity_coverage(
        prices_test,
        preds_test,
        horizon=horizon,
        up_tau=up_tau,
        dn_tau=dn_tau,
    )
    return opp_metrics, opp_joined


# ---------------------------
# 4) Classification metrics + CM plot (no model object required)
# ---------------------------
def compute_validation_metrics_and_plot_cm(
    y_true_enc: np.ndarray,
    y_pred_enc: np.ndarray,
    proba_val: Optional[np.ndarray],
    *,
    output_plot_path: Path | str,
    label_names: Sequence[str] = ("-1", "0", "1"),
) -> Dict[str, Any]:
    """
    Compute F1 metrics, classification report, and confusion matrix plot.

    y_true_enc, y_pred_enc are encoded labels (0,1,2).
    """
    output_plot_path = Path(output_plot_path)

    macro_f1 = f1_score(y_true_enc, y_pred_enc, average="macro")
    weighted_f1 = f1_score(y_true_enc, y_pred_enc, average="weighted")
    report = classification_report(
        y_true_enc,
        y_pred_enc,
        target_names=list(label_names),
        digits=4,
    )

    cm = confusion_matrix(y_true_enc, y_pred_enc, labels=[0, 1, 2])

    fig = plt.figure(figsize=(4, 4))
    ax = plt.gca()
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("3-class Direction (val) — Confusion Matrix")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(label_names)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(label_names)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    fig.savefig(output_plot_path, dpi=160)
    plt.close(fig)

    metrics = {
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "report": report,
    }
    return metrics


# ---------------------------
# 5) Build backtest predictions from arrays
# ---------------------------
def build_backtest_predictions_from_arrays(
    df: pd.DataFrame,
    val_idx: np.ndarray,
    y_true_enc: np.ndarray,
    y_pred_enc: np.ndarray,
    proba_val: Optional[np.ndarray],
    classes_dec: np.ndarray,
    inv_labels: np.ndarray,
    *,
    output_predictions_path: Path | str,
) -> pd.DataFrame:
    """
    Build backtest-ready prediction frame and save to CSV.

    Model-agnostic: you just give it encoded labels and proba.
    """
    output_predictions_path = Path(output_predictions_path)

    true_dir = inv_labels[y_true_enc]
    pred_dir = inv_labels[y_pred_enc]

    pred_df = pd.DataFrame(
        {
            "timestamp": df.loc[val_idx, "timestamp"].values,
            "item": df.loc[val_idx, "item"].values if "item" in df.columns else "",
            "mid_price": df.loc[val_idx, "mid_price"].values
            if "mid_price" in df.columns
            else np.nan,
            "true_dir": true_dir,
            "pred_dir": pred_dir,
        }
    )

    pred_df["tradable"] = (
        df.loc[val_idx, "tradable"].values.astype("int8")
        if "tradable" in df.columns
        else 1
    )

    if proba_val is not None:
        # proba_val is [N, num_classes] in encoded order; classes_dec is decoded [-1,0,1]
        for k, lab in enumerate(classes_dec):
            pred_df[f"proba_{lab}"] = proba_val[:, k].astype("float32")

    pred_df["pred_label"] = pred_dir.astype("int8")

    if proba_val is not None:
        # find index for decoded label +1 (BUY)
        if 1 in classes_dec:
            buy_idx = int(np.where(classes_dec == 1)[0][0])
            pred_df["pred_proba_buy"] = proba_val[:, buy_idx].astype("float32")
        else:
            pred_df["pred_proba_buy"] = (pred_df["pred_dir"] == 1).astype("float32")
    else:
        pred_df["pred_proba_buy"] = (pred_df["pred_dir"] == 1).astype("float32")

    output_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_predictions_path, index=False)
    return pred_df
