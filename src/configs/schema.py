"""
Canonical schema definitions for the dataset.

This module defines the single source of truth for:
- Column names used throughout the pipeline
- Feature selection rules
- Columns that must never be used as features (to prevent leakage)

Source: Extracted from TCN notebook's most up-to-date assumptions.
"""

from typing import Set, List

# ============================================================================
# Core Column Names
# ============================================================================

TIMESTAMP_COL = "timestamp"
"""Main timestamp column for temporal ordering."""

ITEM_ID_COL = "item"
"""Item/product identifier column."""

MID_PRICE_COL = "mid_price"
"""Mid price column."""

TRADABLE_COL = "tradable"
"""Tradability flag column (1 = tradable, 0 = not tradable)."""

# ============================================================================
# Label Columns
# ============================================================================

# Future return targets (used to construct labels, never features)
TARGET_MIN_ABS = "target_min_abs"
TARGET_MAX_ABS = "target_max_abs"
TARGET_MIN_REL = "target_min_rel"
TARGET_MAX_REL = "target_max_rel"
TARGET_Q_UP_REL = "target_q_up_rel"
TARGET_Q_DN_REL = "target_q_dn_rel"

# Execution-aware regression labels (from execution-aware data preparer v2)
Y_LONG_BEST      = "y_long_best"
Y_LONG_DRAWDOWN  = "y_long_drawdown"
Y_SHORT_BEST     = "y_short_best"
Y_SHORT_DRAWUP   = "y_short_drawup"

# Derived label (if present)
LABEL_COL = "label"  # Encoded label: -1 -> 0, 0 -> 1, 1 -> 2

# ============================================================================
# Passthrough Columns (metadata, never features)
# ============================================================================

PASSTHROUGH_BASE: Set[str] = {
    ITEM_ID_COL,
    TIMESTAMP_COL,
    MID_PRICE_COL,
    TRADABLE_COL,
    TARGET_MIN_ABS,
    TARGET_MAX_ABS,
    TARGET_MIN_REL,
    TARGET_MAX_REL,
    # Execution-aware labels must never be used as features
    Y_LONG_BEST,
    Y_LONG_DRAWDOWN,
    Y_SHORT_BEST,
    Y_SHORT_DRAWUP,
}
"""Base set of columns that are metadata and should never be features."""

# ============================================================================
# Leak Detection
# ============================================================================

LEAK_PREFIXES: tuple[str, ...] = ("target_",)
"""Column name prefixes that indicate potential data leakage."""

def get_leak_columns(df_columns: List[str]) -> Set[str]:
    """
    Identify all columns with leak-prone prefixes.

    Args:
        df_columns: List of column names from a DataFrame

    Returns:
        Set of column names that start with leak prefixes
    """
    return {c for c in df_columns if any(c.startswith(p) for p in LEAK_PREFIXES)}

def get_passthrough_columns(df_columns: List[str]) -> Set[str]:
    """
    Get all passthrough columns (metadata + leak columns).

    Args:
        df_columns: List of column names from a DataFrame

    Returns:
        Set of all columns that should not be used as features
    """
    leak_cols = get_leak_columns(df_columns)
    return PASSTHROUGH_BASE | leak_cols

def get_feature_columns(df_columns: List[str]) -> List[str]:
    """
    Extract feature columns from a DataFrame's columns.

    Features are all numeric columns that are NOT:
    - In the passthrough set
    - Starting with leak prefixes

    Args:
        df_columns: List of column names from a DataFrame

    Returns:
        List of valid feature column names
    """
    passthrough = get_passthrough_columns(df_columns)
    return [c for c in df_columns if c not in passthrough]

# ============================================================================
# Label Construction Parameters (from TCN notebook)
# ============================================================================

# Default thresholds for asymmetric directional labels
UP_TAU_DEFAULT = 0.2
DN_TAU_BASE_DEFAULT = 0.05
DN_TAU_MIN_DEFAULT = 0.015
DN_VOL_MULT_DEFAULT = 0.75
VOL_COL_DEFAULT = "vol_60"

# Label encoding: direction (-1, 0, 1) -> encoded (0, 1, 2)
LABEL_MAP = {-1: 0, 0: 1, 1: 2}
LABEL_INV = [-1, 0, 1]  # inverse: encoded -> direction

__all__ = [
    'TIMESTAMP_COL',
    'ITEM_ID_COL',
    'MID_PRICE_COL',
    'TRADABLE_COL',
    'TARGET_MIN_ABS',
    'TARGET_MAX_ABS',
    'TARGET_MIN_REL',
    'TARGET_MAX_REL',
    'TARGET_Q_UP_REL',
    'TARGET_Q_DN_REL',
    'Y_LONG_BEST',
    'Y_LONG_DRAWDOWN',
    'Y_SHORT_BEST',
    'Y_SHORT_DRAWUP',
    'LABEL_COL',
    'PASSTHROUGH_BASE',
    'LEAK_PREFIXES',
    'get_leak_columns',
    'get_passthrough_columns',
    'get_feature_columns',
    'UP_TAU_DEFAULT',
    'DN_TAU_BASE_DEFAULT',
    'DN_TAU_MIN_DEFAULT',
    'DN_VOL_MULT_DEFAULT',
    'VOL_COL_DEFAULT',
    'LABEL_MAP',
    'LABEL_INV',
]
