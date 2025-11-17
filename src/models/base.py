# src/models/base.py
"""
Base interface that all models should implement.

This provides a consistent API for training, inference, and serialization
across different model types (tabular, sequence, transformer-based, etc.).
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional
import pandas as pd


class BazaarModel(ABC):
    """
    Minimal interface that all models (XGB, TCN, Encoder+Probe, future PatchTST/TFT)
    should satisfy for training and inference.

    This abstract base class ensures consistent interaction patterns across
    different model implementations, making it easy to swap models in the
    Symphony ensemble.
    """

    @abstractmethod
    def fit(self, df: pd.DataFrame, **kwargs) -> None:
        """
        Train the model on the provided data.

        Args:
            df: Training data DataFrame
            **kwargs: Additional training parameters
        """
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame, **kwargs) -> pd.Series | pd.DataFrame:
        """
        Generate predictions for the provided data.

        Args:
            df: Input data DataFrame
            **kwargs: Additional prediction parameters

        Returns:
            Predictions as a Series or DataFrame
        """
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """
        Save the model to disk.

        Args:
            path: Path where the model should be saved
        """
        pass

    @abstractmethod
    def load(self, path: Path) -> "BazaarModel":
        """
        Load the model from disk.

        Args:
            path: Path to the saved model

        Returns:
            The loaded model instance (self)
        """
        pass

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance if the model supports it.

        Returns:
            DataFrame with feature names and importance scores, or None
        """
        return None


__all__ = ['BazaarModel']

