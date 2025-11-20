# src/metrics/opportunity.py

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def _forward_extrema(df_item: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Compute forward max/min returns for a single item's time series.

    Parameters
    ----------
    df_item : DataFrame
        Must have columns: ['timestamp', 'item', 'mid_price'].
    horizon : int
        Forward window length (number of bars).

    Returns
    -------
    DataFrame
        Original columns + fwd_up_ret, fwd_dn_ret.
    """
    x = df_item["mid_price"].to_numpy()
    if len(x) < horizon:
        out = df_item[["timestamp", "item", "mid_price"]].copy()
        out["fwd_up_ret"] = np.nan
        out["fwd_dn_ret"] = np.nan
        return out

    win = sliding_window_view(x, horizon)
    fwd_max = np.concatenate([win.max(1), np.full(horizon - 1, np.nan)])
    fwd_min = np.concatenate([win.min(1), np.full(horizon - 1, np.nan)])

    out = df_item[["timestamp", "item", "mid_price"]].copy()
    out["fwd_up_ret"] = (fwd_max - x) / x
    out["fwd_dn_ret"] = (x - fwd_min) / x
    return out


def opportunity_coverage(
    prices: pd.DataFrame,
    preds: pd.DataFrame,
    *,
    horizon: int = 60,
    up_tau: float = 0.03,
    dn_tau: float = 0.03,
    score_col: str = "score",
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Compute opportunity coverage metrics.

    Measures how well the model captures good trading opportunities
    within the specified horizon.

    Parameters
    ----------
    prices : DataFrame
        Must have ['timestamp', 'item', 'mid_price'].
    preds : DataFrame
        Must have ['timestamp', 'item'] and either `score_col` or
        columns ['Pp1','Pm1'] to synthesize score.
    horizon : int
        Forward window length (bars).
    up_tau : float
        Threshold for "good up" opportunity.
    dn_tau : float
        Threshold for "good down" opportunity.
    score_col : str
        Column name in preds to use as score. If missing, will fall back
        to Pp1 - Pm1 if available.

    Returns
    -------
    metrics : Series
        Contains recall@good_up, precision@buy, recall@good_dn,
        precision@sell, coverage%.
    joined : DataFrame
        preds joined with forward extrema and flags.
    """
    preds = preds.copy()

    # If score not present, try to synthesize from Pp1 - Pm1
    if score_col not in preds.columns:
        if {"Pp1", "Pm1"}.issubset(preds.columns):
            preds["score"] = preds["Pp1"] - preds["Pm1"]
        else:
            raise ValueError(
                f"score_col='{score_col}' not found and cannot synthesize from Pp1/Pm1."
            )

    prices = prices.sort_values(["item", "timestamp"]).copy()

    # Compute forward up/down returns per item
    opp = (
        prices.groupby("item", group_keys=False)
        .apply(lambda g: _forward_extrema(g, horizon))
    )

    # Join predictions with forward opportunity data
    joined = preds.merge(
        opp, on=["item", "timestamp"], how="inner", suffixes=("", "_price")
    )

    joined["good_up"] = joined["fwd_up_ret"] >= up_tau
    joined["good_dn"] = joined["fwd_dn_ret"] >= dn_tau

    # Default score: positive -> buy, negative -> sell
    joined["flag_buy"] = joined[score_col] > 0
    joined["flag_sell"] = joined[score_col] < 0

    metrics = {
        "recall@good_up": (
            (joined["flag_buy"] & joined["good_up"]).sum()
            / max(1, joined["good_up"].sum())
        ),
        "precision@buy": (
            (joined["flag_buy"] & (joined["fwd_up_ret"] > 0)).sum()
            / max(1, joined["flag_buy"].sum())
        ),
        "recall@good_dn": (
            (joined["flag_sell"] & joined["good_dn"]).sum()
            / max(1, joined["good_dn"].sum())
        ),
        "precision@sell": (
            (joined["flag_sell"] & (joined["fwd_dn_ret"] > 0)).sum()
            / max(1, joined["flag_sell"].sum())
        ),
        "coverage%": 100
        * joined.groupby("timestamp")["flag_buy"].max().mean(),
    }

    return pd.Series(metrics), joined
