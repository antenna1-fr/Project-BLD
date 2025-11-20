# src/analysis/discrete_diagnostics.py

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc

import config  # assumes your global config module


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_fig(fig, plots_dir: Path, name: str, dpi: int = 160) -> Path:
    _ensure_dir(plots_dir)
    path = plots_dir / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def run_discrete_backtest_diagnostics(
    *,
    trades: Optional[pd.DataFrame],
    equity: Optional[pd.DataFrame],
    clf: Any = None,
    X_val: Optional[Any] = None,
    y_val: Optional[np.ndarray] = None,
    proba_val: Optional[np.ndarray] = None,
    classes_dec: Optional[np.ndarray] = None,
    opp_joined: Optional[pd.DataFrame] = None,
    preds_test: Optional[pd.DataFrame] = None,
    plots_dir: Optional[Path | str] = None,
) -> Dict[str, Path]:
    """
    Run diagnostics & sanity graphs for the discrete backtester / XGB model.

    Parameters
    ----------
    trades : DataFrame or None
        Trades from run_backtest (or None to skip trade plots).
    equity : DataFrame or None
        Equity curve from run_backtest (or None to skip equity/exposure plots).
    clf : fitted classifier (XGBClassifier) or None
        Used for confusion matrix, calibration, PR curve, and feature importance.
    X_val : array-like or None
        Validation features (for clf.predict / predict_proba if needed).
    y_val : array-like or None
        Validation labels (for confusion matrix and PR / calibration).
    proba_val : array-like or None
        Precomputed validation probabilities; if None and clf/X_val provided,
        will compute via clf.predict_proba(X_val).
    classes_dec : np.ndarray or None
        Decoded classes in {-1,0,1}; if None, will infer from clf.classes_.
    opp_joined : DataFrame or None
        Joined opportunities DataFrame (for score vs forward opportunity plot).
    preds_test : DataFrame or None
        Predictions DataFrame for class-prevalence-over-time plot.
    plots_dir : Path or str or None
        Directory to save plots. If None, uses config.XGB_TRADING_DIR.

    Returns
    -------
    dict
        Mapping from logical plot name to saved Path.
    """
    if plots_dir is None:
        plots_dir = Path(str(config.XGB_TRADING_DIR))
    else:
        plots_dir = Path(plots_dir)
    _ensure_dir(plots_dir)

    print(f"[diag] Saving plots to: {plots_dir}")

    saved: Dict[str, Path] = {}

    # ---------- 1) Equity & exposure ----------
    if isinstance(equity, pd.DataFrame) and len(equity):
        # Equity curve
        fig = plt.figure(figsize=(9, 3))
        ax = plt.gca()
        ax.plot(equity["timestamp"].values, equity["equity"].values)
        ax.set_title("Equity Over Time (Discrete Backtester)")
        ax.set_xlabel("timestamp")
        ax.set_ylabel("equity")
        saved["equity_curve"] = _save_fig(fig, plots_dir, "equity_curve_discrete.png")

        # Invested % curve
        if "invested_pct" in equity.columns:
            fig = plt.figure(figsize=(9, 3))
            ax = plt.gca()
            ax.plot(
                equity["timestamp"].values,
                100 * equity["invested_pct"].values,
            )
            ax.set_title("Invested % of Equity Over Time (Discrete)")
            ax.set_xlabel("timestamp")
            ax.set_ylabel("invested %")
            saved["invested_curve"] = _save_fig(
                fig, plots_dir, "invested_curve_discrete.png"
            )

        # Drawdown curve
        e = equity["equity"].astype(float).to_numpy()
        peaks = np.maximum.accumulate(e)
        dd = (e - peaks) / np.maximum(peaks, 1e-12)
        fig = plt.figure(figsize=(9, 2.5))
        ax = plt.gca()
        ax.plot(equity["timestamp"].values, 100 * dd)
        ax.set_title(f"Drawdown (%) — Min: {100 * dd.min():.2f}%")
        ax.set_xlabel("timestamp")
        ax.set_ylabel("drawdown %")
        saved["drawdown_curve"] = _save_fig(
            fig, plots_dir, "drawdown_curve_discrete.png"
        )
    else:
        print("[diag] Skipping equity/exposure plots — 'equity' missing or empty.")

    # ---------- 2) Trade diagnostics ----------
    if isinstance(trades, pd.DataFrame) and len(trades):
        # PnL histogram
        if "pnl" in trades.columns:
            fig = plt.figure(figsize=(7, 3))
            ax = plt.gca()
            ax.hist(trades["pnl"].astype(float).values, bins=80)
            ax.set_title("Trade PnL Histogram")
            ax.set_xlabel("PnL")
            ax.set_ylabel("count")
            saved["trades_pnl_hist"] = _save_fig(
                fig, plots_dir, "trades_pnl_hist.png"
            )

        # Duration vs return scatter
        if {"duration_min", "return"}.issubset(trades.columns):
            fig = plt.figure(figsize=(6, 4))
            ax = plt.gca()
            x = trades["duration_min"].astype(float).values
            y = trades["return"].astype(float).values
            ax.scatter(x, y, s=8, alpha=0.5)
            ax.set_title("Trade Return vs Duration")
            ax.set_xlabel("duration (min)")
            ax.set_ylabel("return")
            saved["trades_return_vs_duration"] = _save_fig(
                fig, plots_dir, "trades_return_vs_duration.png"
            )

        # Top/Bottom items by total PnL
        if "item" in trades.columns and "pnl" in trades.columns:
            agg = (
                trades.groupby("item", as_index=False)["pnl"]
                .sum()
                .sort_values("pnl", ascending=False)
            )
            topN = 20

            head = agg.head(topN)
            fig = plt.figure(figsize=(10, 4))
            ax = plt.gca()
            ax.bar(range(len(head)), head["pnl"].values)
            ax.set_xticks(range(len(head)))
            ax.set_xticklabels(head["item"].values, rotation=90)
            ax.set_title(f"Top {topN} Items by Total PnL")
            ax.set_xlabel("item")
            ax.set_ylabel("PnL")
            saved["top_items_pnl_discrete"] = _save_fig(
                fig, plots_dir, "top_items_pnl_discrete.png"
            )

            tail = agg.tail(topN)
            fig = plt.figure(figsize=(10, 4))
            ax = plt.gca()
            ax.bar(range(len(tail)), tail["pnl"].values)
            ax.set_xticks(range(len(tail)))
            ax.set_xticklabels(tail["item"].values, rotation=90)
            ax.set_title(f"Worst {topN} Items by Total PnL")
            ax.set_xlabel("item")
            ax.set_ylabel("PnL")
            saved["worst_items_pnl_discrete"] = _save_fig(
                fig, plots_dir, "worst_items_pnl_discrete.png"
            )
    else:
        print("[diag] Skipping trade diagnostics — 'trades' missing or empty.")

    # ---------- 3) Confusion matrix (normalized) ----------
    try:
        if clf is not None and y_val is not None:
            if X_val is not None:
                y_val_hat = clf.predict(X_val)
            else:
                raise ValueError("X_val is required for confusion matrix.")

            cm = confusion_matrix(y_val, y_val_hat, labels=[0, 1, 2])
            cm_norm = cm.astype(float) / np.maximum(
                cm.sum(axis=1, keepdims=True), 1.0
            )

            fig = plt.figure(figsize=(4.5, 4.5))
            ax = plt.gca()
            im = ax.imshow(cm_norm, interpolation="nearest")
            ax.set_title("Confusion Matrix (Normalized)")
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(["-1", "0", "1"])
            ax.set_yticks([0, 1, 2])
            ax.set_yticklabels(["-1", "0", "1"])
            for (i, j), v in np.ndenumerate(cm_norm):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center")
            fig.colorbar(im, fraction=0.046, pad=0.04)
            saved["confusion_matrix_normalized"] = _save_fig(
                fig, plots_dir, "confusion_matrix_normalized.png"
            )
    except Exception as e:
        print(f"[diag] Confusion matrix skipped: {e}")

    # ---------- 4) Calibration & PR curve for the BUY class (+1) ----------
    try:
        if clf is not None and y_val is not None:
            if proba_val is None:
                if X_val is None:
                    raise ValueError("X_val is required to compute proba_val.")
                proba_val = clf.predict_proba(X_val)

            if classes_dec is None:
                classes_enc = np.asarray(
                    getattr(clf, "classes_", np.array([0, 1, 2]))
                )
                _inv = np.array([-1, 0, 1])
                classes_dec = _inv[classes_enc]

            buy_idx = int(np.where(classes_dec == 1)[0][0])
            y_true_buy = (y_val == buy_idx).astype(int)
            p_buy = proba_val[:, buy_idx].astype(float)

            # Reliability diagram
            bins = np.linspace(0, 1, 21)
            inds = np.digitize(p_buy, bins) - 1
            cal_x, cal_y = [], []
            for b in range(len(bins) - 1):
                m = inds == b
                if not m.any():
                    continue
                cal_x.append(np.mean(p_buy[m]))
                cal_y.append(np.mean(y_true_buy[m]))

            fig = plt.figure(figsize=(5, 5))
            ax = plt.gca()
            ax.plot([0, 1], [0, 1], linestyle="--")
            ax.plot(cal_x, cal_y, marker="o")
            ax.set_title("Calibration — BUY class")
            ax.set_xlabel("predicted P(+1)")
            ax.set_ylabel("empirical freq (+1)")
            saved["calibration_buy"] = _save_fig(
                fig, plots_dir, "calibration_buy.png"
            )

            # Precision–Recall
            prec, rec, _ = precision_recall_curve(y_true_buy, p_buy)
            pr_auc = auc(rec, prec)
            fig = plt.figure(figsize=(5, 4))
            ax = plt.gca()
            ax.plot(rec, prec)
            ax.set_title(f"Precision–Recall — BUY (AUC={pr_auc:.3f})")
            ax.set_xlabel("recall")
            ax.set_ylabel("precision")
            saved["pr_curve_buy"] = _save_fig(
                fig, plots_dir, "pr_curve_buy.png"
            )

            # Probability histogram
            fig = plt.figure(figsize=(6, 3))
            ax = plt.gca()
            ax.hist(p_buy, bins=50)
            ax.set_title("Histogram of predicted P(+1)")
            ax.set_xlabel("P(+1)")
            ax.set_ylabel("count")
            saved["proba_hist_buy"] = _save_fig(
                fig, plots_dir, "proba_hist_buy.png"
            )
    except Exception as e:
        print(f"[diag] Calibration/PR skipped: {e}")

    # ---------- 5) Score vs Forward Opportunity (from opp_joined) ----------
    try:
        if isinstance(opp_joined, pd.DataFrame) and len(opp_joined):
            dj = opp_joined.dropna(subset=["score", "fwd_up_ret"]).copy()
            if len(dj) > 0:
                dj["dec"] = pd.qcut(dj["score"], 10, duplicates="drop")
                g = dj.groupby("dec")["fwd_up_ret"].mean()
                fig = plt.figure(figsize=(8, 3))
                ax = plt.gca()
                ax.plot(range(len(g)), g.values, marker="o")
                ax.set_xticks(range(len(g)))
                ax.set_xticklabels([f"{i+1}" for i in range(len(g))])
                ax.set_title("Mean forward up-return by score decile")
                ax.set_xlabel("score decile (low→high)")
                ax.set_ylabel("mean fwd_up_ret")
                saved["score_vs_fwd_up_ret_deciles"] = _save_fig(
                    fig, plots_dir, "score_vs_fwd_up_ret_deciles.png"
                )
        else:
            print(
                "[diag] Skipping score→opportunity plot — 'opp_joined' missing or empty."
            )
    except Exception as e:
        print(f"[diag] Score→opportunity plot skipped: {e}")

    # ---------- 6) Class prevalence over time (predicted) ----------
    try:
        if isinstance(preds_test, pd.DataFrame) and len(preds_test):
            pt = preds_test.copy()
            if "timestamp" in pt.columns:
                pt["minute"] = (pt["timestamp"] // 60).astype(int)
                has_cols = {"Pm1", "P0", "Pp1"}.issubset(pt.columns)
                if has_cols:
                    pt["pred_idx"] = np.argmax(
                        pt[["Pm1", "P0", "Pp1"]].to_numpy(), axis=1
                    )
                    pt["pred_label"] = pt["pred_idx"].map(
                        {0: -1, 1: 0, 2: 1}
                    ).astype(int)
                    m = pt.groupby("minute")["pred_label"].apply(
                        lambda s: pd.Series(
                            {
                                "share_buy": (s == 1).mean(),
                                "share_sell": (s == -1).mean(),
                                "share_flat": (s == 0).mean(),
                            }
                        )
                    ).unstack()
                    fig = plt.figure(figsize=(9, 3))
                    ax = plt.gca()
                    ax.plot(
                        m.index.values,
                        100 * m["share_buy"].values,
                        label="buy",
                    )
                    ax.plot(
                        m.index.values,
                        100 * m["share_sell"].values,
                        label="sell",
                    )
                    ax.plot(
                        m.index.values,
                        100 * m["share_flat"].values,
                        label="flat",
                    )
                    ax.legend()
                    ax.set_title("Predicted class share over time")
                    ax.set_xlabel("minute")
                    ax.set_ylabel("share (%)")
                    saved["predicted_class_share_over_time"] = _save_fig(
                        fig, plots_dir, "predicted_class_share_over_time.png"
                    )
    except Exception as e:
        print(f"[diag] Class prevalence over time skipped: {e}")

    # ---------- 7) Simple feature importance (tree-based gain, if available) ----------
    try:
        if clf is not None:
            booster = getattr(clf, "get_booster", lambda: None)()
            if booster is not None:
                try:
                    score = booster.get_score(importance_type="gain")
                except Exception:
                    score = booster.get_score(importance_type="weight")
                if score:
                    imp = (
                        pd.DataFrame(
                            {
                                "feature": list(score.keys()),
                                "importance": list(score.values()),
                            }
                        )
                        .sort_values("importance", ascending=False)
                        .head(25)
                    )
                    fig = plt.figure(figsize=(8, 5))
                    ax = plt.gca()
                    ax.barh(
                        imp["feature"].values[::-1],
                        imp["importance"].values[::-1],
                    )
                    ax.set_title("Top 25 Features (XGBoost importance)")
                    ax.set_xlabel("importance")
                    ax.set_ylabel("feature")
                    saved["xgb_feature_importance_top25"] = _save_fig(
                        fig, plots_dir, "xgb_feature_importance_top25.png"
                    )
                else:
                    print(
                        "[diag] Booster returned empty importance; skipping feature plot."
                    )
            else:
                print(
                    "[diag] No XGBoost booster available; skipping feature plot."
                )
    except Exception as e:
        print(f"[diag] Feature importance plot skipped: {e}")

    print("[diag] Done.")
    return saved
