# src/backtest/splits.py
"""
Walk-forward time-based splitting utilities for backtesting.

Provides shared date-based splitting logic (train/val/test windows) used by:
- TCN trainer
- XGB trainer
- Backtester

Source: Extracted from TCN notebook's approach to train/validation/test splitting.
"""

from typing import Iterator, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.configs.schema import TIMESTAMP_COL


def purged_time_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    embargo: int = 0,
    timestamp_col: str = TIMESTAMP_COL
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate purged time-series splits with optional embargo.

    Creates rolling time-based train/validation splits with an embargo period
    between train and validation to prevent leakage from autocorrelation.

    Args:
        df: DataFrame to split (must be sorted by timestamp)
        n_splits: Number of splits to generate
        embargo: Number of rows to exclude at the end of train set (purging period)
        timestamp_col: Name of timestamp column (for verification)

    Yields:
        Tuple of (train_indices, validation_indices) for each split

    Example:
        >>> splits = list(purged_time_splits(df, n_splits=5, embargo=10))
        >>> train_idx, val_idx = splits[-1]  # Get last split
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"DataFrame missing timestamp column: {timestamp_col}")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    n_rows = len(df)
    idx = np.arange(n_rows)

    for train, val in tscv.split(idx):
        if embargo > 0:
            # Remove the last 'embargo' rows from training set
            val_start = val.min()
            train = train[train < max(0, val_start - embargo)]
        yield train, val


def get_last_split(
    df: pd.DataFrame,
    n_splits: int = 5,
    embargo: int = 0,
    timestamp_col: str = TIMESTAMP_COL
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the final train/validation split (most common use case).

    Args:
        df: DataFrame to split
        n_splits: Number of splits
        embargo: Embargo period in rows
        timestamp_col: Name of timestamp column

    Returns:
        Tuple of (train_indices, validation_indices)
    """
    splits = list(purged_time_splits(df, n_splits, embargo, timestamp_col))
    if not splits:
        raise ValueError("No splits generated")
    return splits[-1]


def get_date_range_mask(
    df: pd.DataFrame,
    start_timestamp: float = None,
    end_timestamp: float = None,
    timestamp_col: str = TIMESTAMP_COL
) -> np.ndarray:
    """
    Create a boolean mask for a specific date range.

    Args:
        df: DataFrame with timestamp column
        start_timestamp: Start timestamp (inclusive), None means no lower bound
        end_timestamp: End timestamp (exclusive), None means no upper bound
        timestamp_col: Name of timestamp column

    Returns:
        Boolean array mask
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"DataFrame missing timestamp column: {timestamp_col}")

    ts = df[timestamp_col].values
    mask = np.ones(len(df), dtype=bool)

    if start_timestamp is not None:
        mask &= (ts >= start_timestamp)
    if end_timestamp is not None:
        mask &= (ts < end_timestamp)

    return mask


def split_by_date(
    df: pd.DataFrame,
    train_days: int,
    val_days: int,
    test_days: int = None,
    timestamp_col: str = TIMESTAMP_COL,
    return_masks: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data by rolling date windows.

    Constructs train/val/test splits based on day counts. The test set is optional
    and represents the most recent data.

    Args:
        df: DataFrame to split (must be sorted by timestamp)
        train_days: Number of days for training
        val_days: Number of days for validation
        test_days: Number of days for testing (None = use remaining data)
        timestamp_col: Name of timestamp column
        return_masks: If True, return boolean masks instead of DataFrames

    Returns:
        If return_masks=False: (train_df, val_df, test_df)
        If return_masks=True: (train_mask, val_mask, test_mask)
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"DataFrame missing timestamp column: {timestamp_col}")

    ts = df[timestamp_col].values
    min_ts = ts.min()
    max_ts = ts.max()

    # Convert days to seconds (assuming timestamp is in seconds)
    train_seconds = train_days * 86400
    val_seconds = val_days * 86400
    test_seconds = test_days * 86400 if test_days is not None else None

    # Define boundaries
    train_end = min_ts + train_seconds
    val_end = train_end + val_seconds

    # Create masks
    train_mask = (ts >= min_ts) & (ts < train_end)
    val_mask = (ts >= train_end) & (ts < val_end)

    if test_seconds is not None:
        test_end = val_end + test_seconds
        test_mask = (ts >= val_end) & (ts < test_end)
    else:
        test_mask = (ts >= val_end) & (ts <= max_ts)

    if return_masks:
        return train_mask, val_mask, test_mask
    else:
        return (
            df[train_mask].copy(),
            df[val_mask].copy(),
            df[test_mask].copy()
        )


__all__ = [
    'purged_time_splits',
    'get_last_split',
    'get_date_range_mask',
    'split_by_date',
]

