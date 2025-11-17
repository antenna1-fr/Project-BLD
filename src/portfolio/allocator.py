# src/portfolio/allocator.py
"""Portfolio allocation strategies."""
from typing import Dict, List
import numpy as np

def alloc_greedy(items: List[str] | np.ndarray, exp_edge_bp: np.ndarray, risk_bp: np.ndarray, max_gross_exposure: float, per_item_cap: float) -> Dict[str, float]:
    """Greedy allocator by edge per unit risk."""
    edge_per_risk = exp_edge_bp / np.maximum(risk_bp, 1e-6)
    idx = np.argsort(-edge_per_risk)
    budget = max_gross_exposure
    positions = {str(item): 0.0 for item in items}
    
    for j in idx:
        i = str(items[j])
        if exp_edge_bp[j] <= 0:
            continue
        size = min(per_item_cap, budget)
        if size <= 0:
            break
        positions[i] = size
        budget -= size
        if budget <= 0:
            break
    
    return positions

def alloc_mean_variance(items: List[str] | np.ndarray, exp_returns: np.ndarray, cov_matrix: np.ndarray, max_gross_exposure: float, risk_aversion: float = 1.0) -> Dict[str, float]:
    """Mean-variance optimal allocation (TODO)."""
    raise NotImplementedError("Mean-variance allocation coming soon")

def alloc_risk_parity(items: List[str] | np.ndarray, risk_bp: np.ndarray, max_gross_exposure: float) -> Dict[str, float]:
    """Risk parity allocation."""
    inv_risk = 1.0 / np.maximum(risk_bp, 1e-6)
    weights = inv_risk / inv_risk.sum()
    return {str(items[i]): weights[i] * max_gross_exposure for i in range(len(items))}

__all__ = ['alloc_greedy', 'alloc_mean_variance', 'alloc_risk_parity']
