# src/backtest/contracts.py
"""Backtest configuration and result data contracts."""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import pandas as pd

@dataclass
class BacktestConfig:
    """Configuration for backtesting a trading strategy."""
    label_col: str
    prediction_col: str
    timestamp_col: str
    item_id_col: str
    side_col: Optional[str] = None
    fee_bp: float = 10.0
    min_edge_bp: float = 1.0
    slippage_bp: float = 5.0
    max_position_size: float = 100.0
    initial_capital: float = 10000.0
    extra_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BacktestResult:
    """Results from a backtest run."""
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    summary: Dict[str, float]
    config: Optional[BacktestConfig] = None
    
    def print_summary(self) -> None:
        print("\n" + "="*60)
        print("BACKTEST SUMMARY")
        print("="*60)
        for key, value in self.summary.items():
            if isinstance(value, float):
                print(f"{key:30s}: {value:12.4f}")
            else:
                print(f"{key:30s}: {value}")
        print("="*60 + "\n")

__all__ = ['BacktestConfig', 'BacktestResult']
