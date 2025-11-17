# src/backtest/engine.py
"""Canonical T+1, tradability-aware backtest engine."""
from typing import Tuple
import numpy as np
import pandas as pd
from .contracts import BacktestConfig, BacktestResult

def run_backtest(df: pd.DataFrame, cfg: BacktestConfig) -> BacktestResult:
    """Canonical T+1 backtest. TODO: Replace with actual logic from notebooks."""
    required_cols = [cfg.timestamp_col, cfg.item_id_col, cfg.prediction_col, cfg.label_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    df = df.sort_values(cfg.timestamp_col).reset_index(drop=True)
    
    # Placeholder implementation
    threshold = 0.5
    df['signal'] = 0
    df.loc[df[cfg.prediction_col] > threshold + (cfg.min_edge_bp / 10000), 'signal'] = 1
    df.loc[df[cfg.prediction_col] < threshold - (cfg.min_edge_bp / 10000), 'signal'] = -1
    
    trades = []
    positions = {}
    cash = cfg.initial_capital
    equity = [cash]
    timestamps = [df[cfg.timestamp_col].iloc[0]]
    
    for idx in range(len(df) - 1):
        row = df.iloc[idx]
        next_row = df.iloc[idx + 1]
        item_id = row[cfg.item_id_col]
        signal = row['signal']
        
        if signal != 0 and item_id not in positions:
            position_size = min(cfg.max_position_size, cash * 0.1)
            if position_size > 0:
                entry_price = next_row.get('mid_price', 100.0)
                total_cost_bp = cfg.fee_bp + cfg.slippage_bp
                effective_price = entry_price * (1 + signal * total_cost_bp / 10000)
                positions[item_id] = {'size': signal * position_size, 'entry_price': effective_price, 'entry_time': next_row[cfg.timestamp_col]}
                cash -= abs(position_size)
        
        total_pv = 0
        for pos_item, pos in positions.items():
            current_price = row.get('mid_price', pos['entry_price'])
            pnl = pos['size'] * (current_price - pos['entry_price']) / pos['entry_price']
            total_pv += abs(pos['size']) + pnl
        
        equity.append(cash + total_pv)
        timestamps.append(row[cfg.timestamp_col])
    
    for item_id, pos in positions.items():
        exit_price = df.iloc[-1].get('mid_price', pos['entry_price'])
        pnl = pos['size'] * (exit_price - pos['entry_price']) / pos['entry_price']
        trades.append({'item_id': item_id, 'entry_time': pos['entry_time'], 'exit_time': df.iloc[-1][cfg.timestamp_col], 'entry_price': pos['entry_price'], 'exit_price': exit_price, 'size': pos['size'], 'pnl': pnl})
        cash += abs(pos['size']) + pnl
    
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    equity_curve = pd.DataFrame({cfg.timestamp_col: timestamps, 'equity': equity})
    
    total_pnl = equity[-1] - cfg.initial_capital
    returns = np.diff(equity) / np.array(equity[:-1])
    returns = returns[np.isfinite(returns)]
    sharpe = 0.0
    if len(returns) > 0 and np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
    
    summary = {'total_pnl': total_pnl, 'return_pct': (total_pnl / cfg.initial_capital) * 100, 'num_trades': len(trades), 'sharpe_ratio': sharpe, 'final_equity': equity[-1], 'max_equity': max(equity), 'min_equity': min(equity)}
    if len(trades) > 0:
        summary['avg_pnl_per_trade'] = trades_df['pnl'].mean()
        summary['win_rate_pct'] = (trades_df['pnl'] > 0).mean() * 100
    
    return BacktestResult(trades=trades_df, equity_curve=equity_curve, summary=summary, config=cfg)

__all__ = ['run_backtest']
