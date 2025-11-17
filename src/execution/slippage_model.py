# src/execution/slippage_model.py
"""Transaction cost and slippage modeling."""
from typing import Optional
import numpy as np

class SimpleCostModel:
    """Simple linear cost model for execution estimation."""
    def __init__(self, fee_bp: float = 10.0):
        self.fee_bp = fee_bp
        self.coeffs = np.array([0.6, 0.25, 0.15])
    
    def predict_bp(self, spread_bp: float | np.ndarray, rel_size: float | np.ndarray, queue_util: float | np.ndarray) -> float | np.ndarray:
        x = np.array([spread_bp, rel_size, queue_util])
        slip = float(np.dot(self.coeffs, x)) if x.ndim == 1 else np.dot(x.T, self.coeffs)
        return self.fee_bp + slip
    
    def calibrate(self, spread_bp: np.ndarray, rel_size: np.ndarray, queue_util: np.ndarray, realized_cost_bp: np.ndarray) -> None:
        y = realized_cost_bp - self.fee_bp
        X = np.column_stack([spread_bp, rel_size, queue_util])
        self.coeffs = np.linalg.lstsq(X, y, rcond=None)[0]

class AdaptiveCostModel:
    """Adaptive cost model (TODO: implement online learning)."""
    def __init__(self, base_fee_bp: float = 10.0, decay: float = 0.99):
        self.base_fee_bp = base_fee_bp
        self.decay = decay
        self.coeffs = np.array([0.6, 0.25, 0.15])
    
    def predict_bp(self, spread_bp: float, rel_size: float, queue_util: float) -> float:
        x = np.array([spread_bp, rel_size, queue_util])
        slip = float(np.dot(self.coeffs, x))
        return self.base_fee_bp + slip
    
    def update(self, spread_bp: float, rel_size: float, queue_util: float, realized_cost_bp: float) -> None:
        pass  # TODO: implement online update

__all__ = ['SimpleCostModel', 'AdaptiveCostModel']
