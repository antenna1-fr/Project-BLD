# Scripts/Backtest.py
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# project config
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as config

# === Default Paths ===
LOAD_DEFAULT = str(config.PREDICTIONS_CSV)
SAVE_DEFAULT = str(config.TRADE_LOG_CSV)
print(f"[backtest] defaults: load={LOAD_DEFAULT}  save={SAVE_DEFAULT}")

def run_backtest(
    df_sim: pd.DataFrame,
    *,
    initial_capital: float = 1_000_000_000.0,
    min_trade_amount: float = 75_000.0,
    max_trades_per_minute: int = 2,
    min_confidence: float = 0.60,
    exit_profit_threshold: float = 0.50,
    fee_bps: float = 100.0,         # trading fee (buy+sell) in basis points
    slippage_bps: float = 1.0,    # per side slippage in basis points
    max_positions_per_item: int = 1,
    cooldown_minutes: int = 0,    # optional cooldown after exit before re-entering same item
) -> tuple[pd.DataFrame, dict]:
    """
    Simple long-only minute-paced backtest on predictions CSV.
    Expects columns: timestamp, item, mid_price, pred_label {-1,0,1}, pred_proba_buy [0..1]
    """

    required = {"timestamp", "item", "mid_price", "pred_label", "pred_proba_buy"}
    missing = required - set(df_sim.columns)
    if missing:
        raise ValueError(f"Backtest missing columns: {sorted(missing)}")

    # sort by time; enforce integer minute groups
    df = df_sim.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["time_group"] = (df["timestamp"] // 60).astype("int64")

    capital = float(initial_capital)
    inventory: list[dict] = []   # [{item, entry_time, entry_price, size, confidence}]
    trade_log = []
    last_seen = {}               # item -> (price, ts)
    last_exit_time = {}          # item -> last exit minute (cooldown)

    current_group = None
    trades_this_period = 0

    # precompute fee/slippage multipliers
    fee_mult_buy  = 1.0 + (fee_bps / 10000.0) + (slippage_bps / 10000.0)
    fee_mult_sell = 1.0 - (fee_bps / 10000.0) - (slippage_bps / 10000.0)

    for _, row in df.iterrows():
        ts = float(row["timestamp"])
        minute = int(row["time_group"])
        if current_group != minute:
            current_group = minute
            trades_this_period = 0

        item = str(row["item"])
        price = float(row["mid_price"])
        pred = int(row["pred_label"])
        conf = float(row["pred_proba_buy"])
        last_seen[item] = (price, ts)

        # === EXIT logic for this item's open positions ===
        updated_inventory = []
        for pos in inventory:
            if pos["item"] == item:
                # mark-to-market return with sell fees/slippage
                exit_price = price * fee_mult_sell
                held_ret = (exit_price - pos["entry_price"]) / pos["entry_price"]
                should_exit = (held_ret >= exit_profit_threshold) or (pred == -1)
                if should_exit:
                    pnl = (exit_price - pos["entry_price"]) * pos["size"]
                    capital += pos["size"] * exit_price
                    trade_log.append({
                        "item": item,
                        "entry_time": pos["entry_time"],
                        "exit_time": ts,
                        "entry_price": pos["entry_price"],
                        "exit_price": exit_price,
                        "size": pos["size"],
                        "pnl": pnl,
                        "return": held_ret,
                        "confidence": pos["confidence"],
                        "reason": ("tp" if held_ret >= exit_profit_threshold else "flip_down"),
                    })
                    last_exit_time[item] = minute
                else:
                    updated_inventory.append(pos)
            else:
                updated_inventory.append(pos)
        inventory = updated_inventory

        # === ENTRY logic (long-only) ===
        if pred == 1 and trades_this_period < max_trades_per_minute and conf >= min_confidence:
            # optional cooldown
            if cooldown_minutes > 0 and item in last_exit_time:
                if minute - last_exit_time[item] < cooldown_minutes:
                    continue

            # enforce max concurrent positions per item
            n_open_item = sum(1 for p in inventory if p["item"] == item)
            if n_open_item >= max_positions_per_item:
                continue

            # confidence-scaled sizing (smooth; 0 at min_confidence → 1 at conf=1)
            # alpha controls aggressiveness
            alpha = 4.0
            scaled = ((conf - min_confidence) / max(1e-6, (1.0 - min_confidence))) ** alpha
            trade_budget = float(scaled * capital)

            if trade_budget >= min_trade_amount and price > 0:
                # buy with fees/slippage applied to entry price
                entry_price = price * fee_mult_buy
                num_items = int(trade_budget / entry_price)
                if num_items >= 1:
                    cost = num_items * entry_price
                    capital -= cost
                    inventory.append({
                        "item": item,
                        "entry_time": ts,
                        "entry_price": entry_price,
                        "size": num_items,
                        "confidence": conf,
                    })
                    trades_this_period += 1

    # === Close remaining positions at last seen prices ===
    for pos in inventory:
        item = pos["item"]
        last_price, last_time = last_seen.get(item, (pos["entry_price"], pos["entry_time"]))
        exit_price = last_price * fee_mult_sell
        pnl = (exit_price - pos["entry_price"]) * pos["size"]
        capital += pos["size"] * exit_price
        trade_log.append({
            "item": item,
            "entry_time": pos["entry_time"],
            "exit_time": last_time,
            "entry_price": pos["entry_price"],
            "exit_price": exit_price,
            "size": pos["size"],
            "pnl": pnl,
            "return": (exit_price - pos["entry_price"]) / pos["entry_price"],
            "confidence": pos["confidence"],
            "reason": "final_close",
        })

    trades = pd.DataFrame(trade_log).sort_values(by="pnl", ascending=False).reset_index(drop=True)
    summary = {
        "final_capital": float(capital),
        "total_profit": float(capital - initial_capital),
        "num_trades": int(len(trades)),
        "average_return": float(trades["return"].mean()) if len(trades) else 0.0,
        "win_rate": float((trades["pnl"] > 0).mean()) if len(trades) else 0.0,
    }
    return trades, summary


def main():
    ap = argparse.ArgumentParser(description="Run backtest on model predictions.")
    ap.add_argument("--input", default=None, help="Path to predictions CSV")
    ap.add_argument("--output", default=None, help="Path to save trade log CSV")
    ap.add_argument("--fee_bps", type=float, default=0.0)
    ap.add_argument("--slippage_bps", type=float, default=0.0)
    ap.add_argument("--cooldown", type=int, default=0)
    args = ap.parse_args()

    load_path = args.input or LOAD_DEFAULT
    save_path = args.output or SAVE_DEFAULT
    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)

    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Could not find input file at: {load_path}")

    print(f"[backtest] loading: {load_path}")
    df_sim = pd.read_csv(load_path)

    trades, summary = run_backtest(
        df_sim,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        cooldown_minutes=args.cooldown,
    )

    print("\n[summary]")
    for k, v in summary.items():
        print(f"{k}: {v}")

    trades.to_csv(save_path, index=False)
    print(f"[backtest] trade log → {save_path}")


if __name__ == "__main__":
    main()
