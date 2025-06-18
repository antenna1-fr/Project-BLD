
import pandas as pd
import numpy as np
import argparse
import os

# === Default Paths ===
base_dir = os.path.dirname(os.path.dirname(__file__))  # go up from Scripts/
load_data_path_default = os.path.join(base_dir, "Data", "xgb_predictions.csv")
save_data_path_default = os.path.join(base_dir, "Outputs", "trade_log.csv")
print(f"Default paths: {load_data_path_default}, {save_data_path_default}")

def run_backtest(df_sim,
                 initial_capital=25_000_000,
                 min_trade_amount=75_000,
                 min_trade_pct=0.02,
                 exit_profit_threshold=0.03):
    capital = initial_capital
    inventory = []
    trade_log = []

    for i in range(len(df_sim)):
        row = df_sim.iloc[i]
        price = row['mid_price']
        pred = row['pred_label']
        ts = row['timestamp']

        # === Exit Logic ===
        still_open = []
        for pos in inventory:
            held_return = (price - pos['entry_price']) / pos['entry_price']

            # Exit only if profit target hit or a SELL signal occurs
            should_exit = (
                held_return >= exit_profit_threshold or
                pred == -1
            )

            if should_exit:
                pnl = (price - pos['entry_price']) * pos['size']
                capital += pos['size'] * price
                trade_log.append({
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
                still_open.append(pos)
        inventory = still_open

        # === Entry Logic (BUY only) ===
        if pred == 1:
            confidence = row['pred_proba_buy']
            cash_available = capital
            max_trade = min(min_trade_amount, cash_available * min_trade_pct)
            if max_trade >= min_trade_amount:
                item_price = price
                num_items = int(max_trade / item_price)
                if num_items >= 1:
                    cost = num_items * item_price
                    capital -= cost
                    inventory.append({
                        'entry_time': ts,
                        'entry_price': item_price,
                        'size': num_items,
                        'confidence': confidence
                    })

    # === Final Exit for Remaining Positions ===
    last_price = df_sim['mid_price'].iloc[-1]
    last_time = df_sim['timestamp'].iloc[-1]
    for pos in inventory:
        pnl = (last_price - pos['entry_price']) * pos['size']
        capital += pos['size'] * last_price
        trade_log.append({
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
        "average_return": trade_df['return'].mean(),
        "win_rate": (trade_df['pnl'] > 0).mean()
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
    df_sim = pd.read_csv(load_path, parse_dates=['timestamp'])

    trade_df, summary = run_backtest(df_sim)

    print("\n Trading Summary:")
    for k, v in summary.items():
        print(f"{k}: {v}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    trade_df.to_csv(save_path, index=False)
    print(f"\n Trade log saved to: {save_path}")

if __name__ == "__main__":
    main()
