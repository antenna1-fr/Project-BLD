
import pandas as pd
import numpy as np
import argparse
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config

# === Default Paths ===
load_data_path_default = str(config.PREDICTIONS_CSV)
save_data_path_default = str(config.TRADE_LOG_CSV)
print(f"Default paths: {load_data_path_default}, {save_data_path_default}")

def run_backtest(df_sim,
                 initial_capital=25_000_000,
                 min_trade_amount=75_000,
                 min_trade_pct=0.03,
                 exit_profit_threshold=0.005):
    """Backtest trades in time order respecting item identity."""

    capital = initial_capital
    inventory = []  # open positions [{'item', 'entry_time', 'entry_price', 'size', 'confidence'}]
    trade_log = []
    last_seen = {}

    df_sim = df_sim.sort_values('timestamp').reset_index(drop=True)
    df_sim['time_group'] = (df_sim['timestamp'] // 60).astype(int)

    current_group = None
    trades_this_period = 0

    for _, row in df_sim.iterrows():
        ts_group = row['time_group']
        if current_group != ts_group:
            current_group = ts_group
            trades_this_period = 0

        item = row['item']
        price = row['mid_price']
        pred = row['pred_label']
        ts = row['timestamp']
        last_seen[item] = (price, ts)

        # === Exit Logic only for positions of this item ===
        updated_inventory = []
        for pos in inventory:
            if pos['item'] == item:
                held_return = (price - pos['entry_price']) / pos['entry_price']
                should_exit = (held_return >= exit_profit_threshold or pred == -1)
                if should_exit:
                    pnl = (price - pos['entry_price']) * pos['size']
                    capital += pos['size'] * price
                    trade_log.append({
                        'item': item,
                        'entry_time': pos['entry_time'],
                        'exit_time': ts,
                        'entry_price': pos['entry_price'],
                        'exit_price': price,
                        'size': pos['size'],
                        'pnl': pnl,
                        'return': held_return,
                        'confidence': pos['confidence']
                    })
                else:
                    updated_inventory.append(pos)
            else:
                updated_inventory.append(pos)
        inventory = updated_inventory

        # === Entry Logic (BUY only) ===
        if pred == 1 and trades_this_period < 2:
            confidence = row['pred_proba_buy']
            cash_available = capital
            max_trade = min(min_trade_amount, cash_available * min_trade_pct)
            if max_trade >= min_trade_amount:
                num_items = int(max_trade / price)
                if num_items >= 1:
                    cost = num_items * price
                    capital -= cost
                    inventory.append({
                        'item': item,
                        'entry_time': ts,
                        'entry_price': price,
                        'size': num_items,
                        'confidence': confidence
                    })
                    trades_this_period += 1

    # === Final Exit for Remaining Positions ===
    for pos in inventory:
        item = pos['item']
        last_price, last_time = last_seen.get(item, (pos['entry_price'], pos['entry_time']))
        pnl = (last_price - pos['entry_price']) * pos['size']
        capital += pos['size'] * last_price
        trade_log.append({
            'item': item,
            'entry_time': pos['entry_time'],
            'exit_time': last_time,
            'entry_price': pos['entry_price'],
            'exit_price': last_price,
            'size': pos['size'],
            'pnl': pnl,
            'return': (last_price - pos['entry_price']) / pos['entry_price'],
            'confidence': pos['confidence']
        })

    trade_df = pd.DataFrame(trade_log).sort_values(by='pnl', ascending=False).reset_index(drop=True)
    summary = {
        "final_capital": capital,
        "total_profit": capital - initial_capital,
        "num_trades": len(trade_df),
        "average_return": trade_df['return'].mean() if len(trade_df) > 0 else 0,
        "win_rate": (trade_df['pnl'] > 0).mean() if len(trade_df) > 0 else 0
    }

    return trade_df, summary

def main():
    parser = argparse.ArgumentParser(description="Run backtest on model predictions.")
    parser.add_argument('--input', default=None, help='Path to input df_sim CSV')
    parser.add_argument('--output', default=None, help='Path to save trade log CSV')
    args = parser.parse_args()

    # Use top-level default paths unless overridden
    load_path = args.input or load_data_path_default
    save_path = args.output or save_data_path_default

    if not os.path.exists(load_path):
        raise FileNotFoundError(f" Could not find input file at: {load_path}")

    print(f" Loading predictions from: {load_path}")
    df_sim = pd.read_csv(load_path)

    trade_df, summary = run_backtest(df_sim)

    print("\n Trading Summary:")
    for k, v in summary.items():
        print(f"{k}: {v}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    trade_df.to_csv(save_path, index=False)
    print(f"\n Trade log saved to: {save_path}")

if __name__ == "__main__":
    main()
