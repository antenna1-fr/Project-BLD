# src/backtest/engine.py
"""
Canonical T+1, tradability-aware backtest engine + Optuna-based tuning.

Implements:
- T+1 execution semantics (decisions at t execute at t+1)
- Tradability gates (no entry when tradable=0, defer exits)
- Microstructure guards (impact caps, spreads, median filtering)
- Hard exposure caps per item
- Minimum holding periods
- Exec-first ordering for causal impact caps

Also provides:
- Parallel Optuna tuning of backtest hyperparameters via joblib (loky)
  with sqlite-backed study storage.

This file is designed to be a drop-in replacement for the original
backtest engine, while also bundling the functionality of the
"parallel backtest tuning via joblib (loky)" script.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import os
import math
import inspect
import traceback
import warnings

import numpy as np
import pandas as pd
import optuna
from joblib import Parallel, delayed

from .contracts import BacktestConfig, BacktestResult
from src.configs.schema import TIMESTAMP_COL, ITEM_ID_COL, MID_PRICE_COL, TRADABLE_COL


# ======================================================================
# Core backtest (unchanged API)
# ======================================================================

def run_backtest(
    df_sim: pd.DataFrame,
    *,
    initial_capital: float = 1_000_000_000.0,
    min_trade_amount: float = 10_000_000.0,
    max_trades_per_minute: int = 2,
    min_confidence: float = 0.90,
    exit_profit_threshold: float = 0.20,
    stop_loss_threshold: float = 0.20,
    fee_bps: float = 100,
    slippage_bps: float = 0.0,
    spread_bps: float = 10.0,
    persist_bars: int = 20,
    min_confidence_streak: float = 0.60,
    median_window: int = 120,
    impact_cap_bps: float = 200.0,
    max_positions_per_item: int = 1,
    cooldown_minutes: int = 0,
    bar_seconds: int = 60,
    alpha: float = 7.0,
    per_item_exposure_cap_pct: float = 0.20,
    min_hold_minutes: int = 2,
    pred_label_col: str = "pred_label",
    pred_proba_buy_col: str = "pred_proba_buy",
    timestamp_col: str = TIMESTAMP_COL,
    item_col: str = ITEM_ID_COL,
    mid_price_col: str = MID_PRICE_COL,
    tradable_col: str = TRADABLE_COL,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Causal backtest with microstructure guards + tradability gate.

    Args:
        df_sim: DataFrame with columns: timestamp, item, mid_price, pred_label,
                pred_proba_buy, (optional) tradable
        initial_capital: Starting capital
        min_trade_amount: Minimum trade size
        max_trades_per_minute: Maximum entries per minute
        min_confidence: Minimum confidence to enter
        exit_profit_threshold: Take-profit threshold (fraction)
        stop_loss_threshold: Stop-loss threshold (fraction)
        fee_bps: Round-trip fee in basis points (split per side)
        slippage_bps: Extra slippage per side in basis points
        spread_bps: Synthetic spread in basis points
        persist_bars: Consecutive signal bars required before action
        min_confidence_streak: Lenient threshold to build streaks
        median_window: Rolling median window for price stability
        impact_cap_bps: Max effective price move per bar for fills
        max_positions_per_item: Max concurrent positions per item
        cooldown_minutes: Minutes to wait after exit before re-entry
        bar_seconds: Seconds per bar (for T+1 calculation)
        alpha: Position sizing exponent
        per_item_exposure_cap_pct: Max % equity per item at entry (mark-to-bid)
        min_hold_minutes: Minimum holding period in minutes
        pred_label_col: Name of prediction label column
        pred_proba_buy_col: Name of buy probability column
        timestamp_col: Name of timestamp column
        item_col: Name of item ID column
        mid_price_col: Name of mid price column
        tradable_col: Name of tradability flag column
        verbose: Print progress/diagnostics

    Returns:
        Tuple of (trades_df, summary_dict, equity_df)
    """
    # Validate required columns
    required = {timestamp_col, item_col, mid_price_col, pred_label_col, pred_proba_buy_col}
    missing = required - set(df_sim.columns)
    if missing:
        raise ValueError(f"Backtest missing columns: {sorted(missing)}")

    has_tradable = tradable_col in df_sim.columns

    # 0) Dedup & strict ordering
    df = (
        df_sim.sort_values([timestamp_col, item_col])
        .drop_duplicates([item_col, timestamp_col], keep="last")
        .reset_index(drop=True)
    )

    # 1) Exec time = next bar (T+1)
    df["exec_time"] = (df[timestamp_col] + bar_seconds).astype("float64")
    df["exec_minute"] = (df["exec_time"] // bar_seconds).astype("int64")

    # 2) Rolling median mid per item for stability
    df = df.sort_values([item_col, timestamp_col], kind="mergesort")
    df["mid_med"] = (
        df.groupby(item_col, sort=False)[mid_price_col]
        .rolling(median_window, min_periods=1)
        .median()
        .reset_index(level=0, drop=True)
    )

    # 3) State initialization
    capital = float(initial_capital)
    inventory = []  # List of dicts: item, entry_time, entry_price, size, confidence, entry_minute
    trade_log = []

    half_spread = (spread_bps / 20000.0)
    fee_buy_mult = 1.0 + (fee_bps / 20000.0) + (slippage_bps / 10000.0)
    fee_sell_mult = 1.0 - (fee_bps / 20000.0) - (slippage_bps / 10000.0)

    equity_rows = []
    current_minute = None
    trades_this_minute = 0

    def minute_of(ts: float) -> int:
        return int(ts // bar_seconds)

    # --- FAST grouping via indices ---
    raw_idx = df.groupby(timestamp_col, sort=False).indices
    exec_idx = df.groupby("exec_time", sort=False).indices

    # Pre-extract columns to NumPy once
    ts_arr = df[timestamp_col].to_numpy(dtype=np.float64)
    exec_arr = df["exec_time"].to_numpy(dtype=np.float64)
    _item_cat = df[item_col].astype("category")
    item_codes = _item_cat.cat.codes.to_numpy(dtype=np.int32)
    item_categories = _item_cat.cat.categories
    item_to_id = {str(cat): int(i) for i, cat in enumerate(item_categories)}
    mid_med_arr = df["mid_med"].to_numpy(dtype=np.float64)
    pred_arr = df[pred_label_col].to_numpy(dtype=np.int8)
    pbuy_arr = df[pred_proba_buy_col].to_numpy(dtype=np.float32)
    tradable_arr = df[tradable_col].to_numpy(dtype=np.int8) if has_tradable else None

    n_items = int(item_codes.max()) + 1

    # Per-item array state
    last_seen_mid = np.full(n_items, np.nan, dtype=np.float64)
    last_seen_ts = np.zeros(n_items, dtype=np.float64)
    streak_up_arr = np.zeros(n_items, dtype=np.int32)
    streak_dn_arr = np.zeros(n_items, dtype=np.int32)
    last_exit_minute_arr = np.full(n_items, -10_000_000, dtype=np.int32)

    # Exposure tracking (mark-to-bid)
    per_item_exposure_bid = np.zeros(n_items, dtype=np.float64)
    invested_mark = 0.0

    # Build global timeline
    timeline = np.unique(np.concatenate([ts_arr, exec_arr]))
    timeline.sort()

    if verbose:
        print(f"[bt] steps={len(timeline):,} | rows={len(df):,} | items={n_items:,}")

    # Tail hygiene (no new entries near the end)
    TAIL_BARS = 5
    end_raw_ts = float(np.nanmax(ts_arr)) if len(ts_arr) else 0.0
    end_exec_ts = float(np.nanmax(exec_arr)) if len(exec_arr) else end_raw_ts

    def snapshot(minute: int, ts: float):
        """Record equity snapshot at this minute."""
        if minute is None:
            return
        invested = 0.0
        for pos in inventory:
            iid = item_to_id.get(pos["item"], None)
            if iid is None:
                mm = pos["entry_price"]
            else:
                mm = last_seen_mid[iid]
                if not np.isfinite(mm) or mm <= 0:
                    mm = pos["entry_price"]
            bid = mm * (1.0 - half_spread)
            invested += float(pos["size"]) * float(bid)
        equity = capital + invested
        equity_rows.append(
            {
                "minute": minute,
                "timestamp": ts,
                "capital": capital,
                "invested": invested,
                "equity": equity,
                "invested_pct": invested / equity if equity > 0 else 0.0,
                "num_positions": len(inventory),
            }
        )

    # =========================
    # MAIN LOOP (EXEC-FIRST, then RAW update)
    # =========================
    for ts in timeline:
        minute = minute_of(ts)

        # Minute snapshot on boundary
        if current_minute is None:
            current_minute = minute
            trades_this_minute = 0
        elif minute != current_minute:
            snapshot(current_minute, ts)
            current_minute = minute
            trades_this_minute = 0

        # ---- (A) EXEC actions at this exec_time (exits then entries) ----
        eidx = exec_idx.get(ts)
        if eidx is not None:
            # Slice arrays once for this timestamp
            ii = item_codes[eidx]
            preds = pred_arr[eidx]
            pbuy = pbuy_arr[eidx]
            midm = mid_med_arr[eidx]
            trad = (tradable_arr[eidx] if tradable_arr is not None else None)
            minute = int(ts // bar_seconds)

            # --- Exits first ---
            for j in range(len(eidx)):
                item_id = int(ii[j])
                if trad is not None and int(trad[j]) == 0:
                    continue  # Defer exit if not tradable

                # Signal & streaks
                p_buy = float(pbuy[j])
                p_neg = max(0.0, 1.0 - p_buy)
                if p_buy >= min_confidence_streak and p_buy >= p_neg:
                    sig = 1
                elif p_neg >= min_confidence_streak and p_neg > p_buy:
                    sig = -1
                else:
                    sig = 0

                if sig == 1:
                    streak_up_arr[item_id] += 1
                    streak_dn_arr[item_id] = 0
                elif sig == -1:
                    streak_up_arr[item_id] = 0
                    streak_dn_arr[item_id] += 1
                else:
                    streak_up_arr[item_id] = 0
                    streak_dn_arr[item_id] = 0

                # Impact-capped filtered bid from mid_med
                prev_mid = last_seen_mid[item_id]
                m = float(midm[j])
                if np.isfinite(prev_mid):
                    max_move = (impact_cap_bps / 10000.0)
                    delta = np.clip(m / max(prev_mid, 1e-12), 1.0 - max_move, 1.0 + max_move)
                    m = prev_mid * delta
                bid = m * (1.0 - half_spread)

                item = str(item_categories[item_id])

                updated = []
                for pos in inventory:
                    if pos["item"] != item:
                        updated.append(pos)
                        continue

                    # Enforce min-hold
                    if minute < (pos["entry_minute"] + int(min_hold_minutes)):
                        updated.append(pos)
                        continue

                    entry_price = pos["entry_price"]
                    mkt_exit_price = bid * fee_sell_mult
                    held_ret = (mkt_exit_price - entry_price) / entry_price

                    tp_fill_price = entry_price * (1.0 + float(exit_profit_threshold))
                    sl_fill_price = entry_price * (1.0 - float(stop_loss_threshold))

                    flip_ready = (streak_dn_arr[item_id] >= persist_bars)
                    tp_hit = held_ret >= float(exit_profit_threshold)
                    sl_hit = held_ret <= -float(stop_loss_threshold)

                    if tp_hit:
                        exit_price = min(mkt_exit_price, tp_fill_price)
                        reason = "tp"
                    elif sl_hit:
                        exit_price = max(mkt_exit_price, sl_fill_price)
                        reason = "sl"
                    else:
                        exit_price = mkt_exit_price
                        reason = "flip_persist" if flip_ready else None

                    should_exit = tp_hit or sl_hit or flip_ready
                    if should_exit:
                        # Exposure decrement (mark-to-bid)
                        ex_bid_val = float(bid) * float(pos["size"])
                        per_item_exposure_bid[item_id] = max(
                            0.0, per_item_exposure_bid[item_id] - ex_bid_val
                        )
                        invested_mark = max(0.0, invested_mark - ex_bid_val)

                        pnl = (exit_price - entry_price) * pos["size"]
                        capital += pos["size"] * exit_price
                        dur_sec = float(ts - pos["entry_time"])
                        trade_log.append(
                            {
                                "item": item,
                                "entry_time": pos["entry_time"],
                                "exit_time": ts,
                                "duration_sec": dur_sec,
                                "duration_min": dur_sec / 60.0,
                                "entry_price": entry_price,
                                "exit_price": exit_price,
                                "size": pos["size"],
                                "pnl": pnl,
                                "return": (exit_price - entry_price) / entry_price,
                                "confidence": pos["confidence"],
                                "reason": reason,
                            }
                        )
                        last_exit_minute_arr[item_id] = minute
                    else:
                        updated.append(pos)
                inventory = updated

            # --- Entries after exits ---
            if not (TAIL_BARS > 0 and ts >= end_exec_ts - TAIL_BARS * bar_seconds):
                for j in range(len(eidx)):
                    item_id = int(ii[j])
                    if trad is not None and int(trad[j]) == 0:
                        continue  # No entry if not tradable

                    # Score buy: P(+1) minus P(-1)
                    p_buy = float(pbuy[j])
                    p_neg = max(0.0, 1.0 - p_buy)
                    score_buy = p_buy - p_neg
                    if np.isnan(score_buy) or score_buy < 0.0:
                        continue

                    conf = p_buy
                    if conf < min_confidence or trades_this_minute >= max_trades_per_minute:
                        continue
                    if cooldown_minutes > 0 and (minute - last_exit_minute_arr[item_id]) < cooldown_minutes:
                        continue
                    if streak_up_arr[item_id] < persist_bars:
                        continue

                    # Impact-capped filtered ask from mid_med
                    prev_mid = last_seen_mid[item_id]
                    m = float(midm[j])
                    if np.isfinite(prev_mid):
                        max_move = (impact_cap_bps / 10000.0)
                        delta = np.clip(m / max(prev_mid, 1e-12), 1.0 - max_move, 1.0 + max_move)
                        m = prev_mid * delta
                    ask = m * (1.0 + half_spread)
                    entry_price = ask * fee_buy_mult
                    if entry_price <= 0:
                        continue

                    # Disallow entries that cannot meet min hold before data end
                    if (ts + min_hold_minutes * bar_seconds) > end_exec_ts:
                        continue

                    # Position cap: count current opens
                    item = str(item_categories[item_id])
                    n_open = sum(1 for p in inventory if p["item"] == item)
                    if n_open >= max_positions_per_item:
                        continue

                    # --- per-item exposure cap (mark-to-bid) ---
                    current_equity = capital + invested_mark
                    if current_equity <= 0.0:
                        continue
                    cap_value = float(per_item_exposure_cap_pct) * current_equity
                    bid_now = m * (1.0 - half_spread)
                    remaining_value = cap_value - per_item_exposure_bid[item_id]
                    if remaining_value <= 0.0:
                        continue  # Already at/over cap

                    # Budget sizing from confidence
                    scaled = ((conf - min_confidence) / max(1e-6, 1.0 - min_confidence)) ** alpha
                    budget = float(scaled * capital)
                    if budget < min_trade_amount:
                        continue
                    size = int(budget / entry_price)
                    if size < 1:
                        continue

                    # Clamp by remaining per-item headroom
                    max_size_by_cap = int(remaining_value // max(bid_now, 1e-12))
                    if max_size_by_cap <= 0:
                        continue
                    if size > max_size_by_cap:
                        size = max_size_by_cap
                    if size < 1:
                        continue

                    # Open position
                    capital -= size * entry_price
                    inventory.append(
                        {
                            "item": item,
                            "entry_time": ts,
                            "entry_minute": minute,
                            "entry_price": entry_price,
                            "size": size,
                            "confidence": conf,
                        }
                    )

                    # Exposure increment (mark-to-bid)
                    trade_bid_val = bid_now * size
                    per_item_exposure_bid[item_id] += trade_bid_val
                    invested_mark += trade_bid_val

                    # Reset streaks for this item after entry
                    streak_dn_arr[item_id] = 0
                    streak_up_arr[item_id] = 0

                    last_seen_mid[item_id] = m
                    last_seen_ts[item_id] = ts
                    trades_this_minute += 1

        # ---- (B) RAW UPDATE after exec: update last_seen from raw rows at this timestamp ----
        ridx = raw_idx.get(ts)
        if ridx is not None:
            ii = item_codes[ridx]
            mm = mid_med_arr[ridx]
            last_seen_mid[ii] = mm
            last_seen_ts[ii] = ts

    # Final snapshot and forced exits at bid
    if current_minute is not None and len(df):
        snapshot(current_minute, float(df[timestamp_col].iloc[-1]))

    for pos in inventory:
        iid = item_to_id.get(pos["item"], None)
        if iid is None:
            mm = pos["entry_price"]
            last_time = pos["entry_time"]
        else:
            mm = last_seen_mid[iid]
            if not np.isfinite(mm) or mm <= 0:
                mm = pos["entry_price"]
            last_time = last_seen_ts[iid]
            if not np.isfinite(last_time) or last_time <= 0:
                last_time = pos["entry_time"]

        # Ensure exit strictly after entry (T+1)
        exit_time = max(float(last_time), float(pos["entry_time"]) + float(bar_seconds))
        bid = float(mm) * (1.0 - half_spread)
        exit_price = bid * fee_sell_mult
        pnl = (exit_price - pos["entry_price"]) * pos["size"]
        capital += pos["size"] * exit_price
        dur_sec = float(exit_time - pos["entry_time"])
        trade_log.append(
            {
                "item": pos["item"],
                "entry_time": pos["entry_time"],
                "exit_time": exit_time,
                "duration_sec": dur_sec,
                "duration_min": dur_sec / 60.0,
                "entry_price": pos["entry_price"],
                "exit_price": exit_price,
                "size": pos["size"],
                "pnl": pnl,
                "return": (exit_price - pos["entry_price"]) / pos["entry_price"],
                "confidence": pos["confidence"],
                "reason": "final_close",
            }
        )

    # Build DataFrames safely even if there are zero trades / snapshots
    if len(trade_log) > 0:
        trades = (
            pd.DataFrame(trade_log)
            .sort_values("pnl", ascending=False)
            .reset_index(drop=True)
        )
    else:
        trades = pd.DataFrame(
            columns=[
                "item",
                "entry_time",
                "exit_time",
                "duration_sec",
                "duration_min",
                "entry_price",
                "exit_price",
                "size",
                "pnl",
                "return",
                "confidence",
                "reason",
            ]
        )

    if len(equity_rows) > 0:
        eq = (
            pd.DataFrame(equity_rows)
            .sort_values("minute")
            .reset_index(drop=True)
        )
    else:
        eq = pd.DataFrame(
            columns=[
                "minute",
                "timestamp",
                "capital",
                "invested",
                "equity",
                "invested_pct",
                "num_positions",
            ]
        )

    # Invariants
    if len(trades):
        assert (
            trades["exit_time"] > trades["entry_time"]
        ).all(), "Exit must be strictly after entry (T+1)."
        if (trades["return"] > 1.0).any():
            print(
                "[WARN] Trades with >100% return detected — "
                "check timestamp units and price scaling."
            )

    # Summary
    gross_profit = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    gross_loss = trades.loc[trades["pnl"] < 0, "pnl"].sum()
    avg_win = (
        trades.loc[trades["pnl"] > 0, "return"].mean()
        if (trades["pnl"] > 0).any()
        else 0.0
    )
    avg_loss = (
        trades.loc[trades["pnl"] < 0, "return"].mean()
        if (trades["pnl"] < 0).any()
        else 0.0
    )
    avg_duration = trades["duration_min"].mean() if len(trades) else 0.0

    summary = {
        "Final Capital": f"{capital:,.0f}",
        "Total Profit": f"{capital - initial_capital:,.0f}",
        "Num Trades": len(trades),
        "Win Rate": f"{(trades['pnl'] > 0).mean():.2%}"
        if len(trades)
        else "n/a",
        "Average Return / Trade": f"{trades['return'].mean():.2%}"
        if len(trades)
        else "n/a",
        "Average Win": f"{avg_win:.2%}",
        "Average Loss": f"{avg_loss:.2%}",
        "Gross Profit": f"{gross_profit:,.0f}",
        "Gross Loss": f"{gross_loss:,.0f}",
        "Profit Factor": f"{abs(gross_profit / gross_loss):.2f}"
        if gross_loss < 0
        else "∞",
        "Average Duration (min)": f"{avg_duration:.2f}",
    }

    if verbose:
        print("\n" + "=" * 60)
        print("BACKTEST SUMMARY")
        print("=" * 60)
        for key, val in summary.items():
            print(f"{key:30s}: {val}")
        print("=" * 60 + "\n")

    return trades, summary, eq


def run_backtest_with_config(df: pd.DataFrame, cfg: BacktestConfig) -> BacktestResult:
    """
    Run backtest using BacktestConfig (for backwards compatibility).

    Maps BacktestConfig to the detailed run_backtest parameters.
    """
    trades, summary, equity = run_backtest(
        df,
        initial_capital=cfg.initial_capital,
        fee_bps=cfg.fee_bp,
        slippage_bps=cfg.slippage_bp,
        pred_label_col=cfg.prediction_col,
        timestamp_col=cfg.timestamp_col,
        item_col=cfg.item_id_col,
        mid_price_col="mid_price",  # Assume default
        verbose=False,
        **cfg.extra_params,  # Allow extra params through config
    )

    return BacktestResult(
        trades=trades,
        equity_curve=equity,
        summary=summary,
        config=cfg,
    )


# ======================================================================
# Optuna-based tuning (functional copy of your joblib/Optuna script)
# ======================================================================

def _load_preds_from_csv(
    path: Path,
    timestamp_col: str = "timestamp",
    item_col: str = "item",
    pred_label_col: str = "pred_label",
    pred_dir_col: str = "pred_dir",
    proba_col: str = "proba_1",
    pred_proba_buy_col: str = "pred_proba_buy",
    tradable_col: str = "tradable",
) -> pd.DataFrame:
    """
    Load and normalize prediction CSV to the format expected by run_backtest.
    Mirrors the behavior of _load_preds() in your tuning script.
    """
    dfp = pd.read_csv(path)
    dfp = (
        dfp.sort_values([timestamp_col, item_col])
        .drop_duplicates([item_col, timestamp_col], keep="last")
    )

    if pred_label_col not in dfp.columns and pred_dir_col in dfp.columns:
        dfp[pred_label_col] = dfp[pred_dir_col].astype("int8")

    if pred_proba_buy_col not in dfp.columns:
        if proba_col in dfp.columns:
            dfp[pred_proba_buy_col] = dfp[proba_col].astype("float32")
        else:
            dfp[pred_proba_buy_col] = (dfp[pred_label_col] == 1).astype("float32")

    if tradable_col not in dfp.columns:
        dfp[tradable_col] = 1

    return dfp

def load_backtest_preds(path: Path) -> pd.DataFrame:
    """
    Public helper to load predictions for backtesting/tuning.

    Uses the same normalization logic as `_load_preds_from_csv`.
    """
    return _load_preds_from_csv(Path(path))

def _days_in_preds(preds_df: pd.DataFrame, timestamp_col: str = "timestamp") -> float:
    if len(preds_df) == 0:
        return 1.0
    t0 = float(preds_df[timestamp_col].min())
    t1 = float(preds_df[timestamp_col].max())
    days = max((t1 - t0) / 86400.0, 1e-6)
    return days


def tune_backtest_knobs(
    *,
    preds: Optional[pd.DataFrame] = None,
    pred_path: Optional[Path] = None,
    study_name: str = "bt_knobs_v4",
    storage_dir: Path | str = "backtests/db",
    total_trials: int = 200,
    n_workers: Optional[int] = None,
    initial_capital: float = 1_000_000_000.0,
    max_trades_per_minute: int = 2,
    spread_bps: float = 10.0,
    fee_bps: float = 100.0,
    slippage_bps: float = 0.0,
    median_window: int = 30,
    impact_cap_bps: float = 200.0,
    max_positions_per_item: int = 1,
    cooldown_minutes: int = 0,
    bar_seconds: int = 60,
    trade_log_path: Optional[Path] = None,
    timestamp_col: str = "timestamp",
    item_col: str = "item",
) -> Tuple[optuna.Study, Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """
    Parallel Optuna tuning of backtest parameters, functionally equivalent
    to the "Parallel backtest tuning via joblib (loky)" script.

    Either `preds` OR `pred_path` must be provided.

    Returns:
        (study, best_summary, best_trades_df, best_equity_df)
    """
    if preds is None and pred_path is None:
        raise ValueError("Must provide either `preds` or `pred_path`.")

    # Load predictions if a path is given
    if preds is None:
        preds = _load_preds_from_csv(Path(pred_path))

    storage_dir = Path(storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)

    storage_url = f"sqlite:///{str(storage_dir / f'{study_name}.db')}"

    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, 32)

    # Avoid BLAS oversubscription inside each worker
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    # Create or load study (idempotent)
    _ = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    def _bt_objective(trial, preds_local: pd.DataFrame):
        # ---- sample knobs (ranges mirror your script) ----
        sampled = {
            "min_trade_amount": trial.suggest_float("min_trade_amount", 5e6, 5e7, log=True),
            "min_confidence": trial.suggest_float("min_confidence", 0.55, 0.95),
            "exit_profit_threshold": trial.suggest_float("exit_profit_threshold", 0.02, 0.40),
            "stop_loss_threshold": trial.suggest_float("stop_loss_threshold", 0.02, 0.40),
            "persist_bars": trial.suggest_int("persist_bars", 1, 30),
            "alpha": trial.suggest_float("alpha", 1.0, 12.0),
            "min_confidence_streak": trial.suggest_float("min_confidence_streak", 0.50, 0.80),
        }

        # Only pass args that run_backtest accepts
        try:
            allowed = set(inspect.signature(run_backtest).parameters.keys())
            bt_kwargs = {k: v for k, v in sampled.items() if k in allowed}
        except Exception:
            bt_kwargs = sampled

        # ---- compute target trade count from data span ----
        days = _days_in_preds(preds_local, timestamp_col=timestamp_col)
        target_trades = max(1.0, 25.0 * days)  # 25 round-trips/day (tunable)
        sigma = max(1.0, 0.40 * target_trades)  # ~40% width

        try:
            trades, summary, eq = run_backtest(
                preds_local,
                initial_capital=initial_capital,
                max_trades_per_minute=max_trades_per_minute,
                spread_bps=spread_bps,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                median_window=median_window,
                impact_cap_bps=impact_cap_bps,
                max_positions_per_item=max_positions_per_item,
                cooldown_minutes=cooldown_minutes,
                bar_seconds=bar_seconds,
                verbose=False,
                **bt_kwargs,
            )

            ntr = len(trades)
            if ntr == 0:
                return -1e9  # avoid best being a zero-trade config

            # Extract PF and profit
            pf_txt = summary.get("Profit Factor", "0")
            pf = float("inf") if pf_txt == "∞" else float(pf_txt)
            if not np.isfinite(pf) or pf <= 0:
                try:
                    tot_profit = float(
                        summary.get("Total Profit", "0").replace(",", "")
                    )
                except Exception:
                    tot_profit = 0.0
                pf = max(tot_profit / initial_capital, 0.0)

            # Trade-count shaping (Gaussian centered at target_trades)
            trade_weight = math.exp(-0.5 * ((ntr - target_trades) / sigma) ** 2)

            # Profit nudge
            try:
                tot_profit = float(summary.get("Total Profit", "0").replace(",", ""))
            except Exception:
                tot_profit = 0.0
            profit_nudge = 1.0 + 0.15 * max(tot_profit / initial_capital, 0.0)

            score = float(pf * trade_weight * profit_nudge)

            # Penalty for extremely churny configs
            if ntr > 4.0 * target_trades:
                score *= 0.90

            return score

        except Exception:
            traceback.print_exc()
            return -1e12

    def _worker_chunk(n_trials: int):
        preds_local = preds.copy()  # local copy per worker
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        study.optimize(
            lambda tr: _bt_objective(tr, preds_local),
            n_trials=n_trials,
            n_jobs=1,
            gc_after_trial=True,
            catch=(Exception,),
        )

    # Split trials across workers like your script
    per = total_trials // n_workers
    rem = total_trials % n_workers
    trial_counts = [per + (1 if i < rem else 0) for i in range(n_workers)]
    trial_counts = [c for c in trial_counts if c > 0]

    if len(trial_counts) == 0:
        warnings.warn("No trials scheduled; check total_trials and n_workers.")
    else:
        Parallel(n_jobs=n_workers, backend="loky")(
            delayed(_worker_chunk)(c) for c in trial_counts
        )

    # Read best result
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    completed = [
        t for t in study.get_trials(deepcopy=False)
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    print(f"[optuna] completed trials: {len(completed)}")
    print("Best params:", study.best_params)
    print("Best value:", study.best_value)

    # Final backtest with best params
    best_params = {
        k: v
        for k, v in study.best_params.items()
        if k in run_backtest.__code__.co_varnames
    }
    best_trades, best_summary, best_eq = run_backtest(
        preds,
        initial_capital=initial_capital,
        max_trades_per_minute=max_trades_per_minute,
        spread_bps=spread_bps,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        median_window=median_window,
        impact_cap_bps=impact_cap_bps,
        max_positions_per_item=max_positions_per_item,
        cooldown_minutes=cooldown_minutes,
        bar_seconds=bar_seconds,
        verbose=False,
        **best_params,
    )

    print("Summary:", best_summary)

    if trade_log_path is not None:
        best_trades.to_csv(trade_log_path, index=False)

    return study, best_summary, best_trades, best_eq


__all__ = [
    "run_backtest",
    "run_backtest_with_config",
    "tune_backtest_knobs",
    "load_backtest_preds",
]
