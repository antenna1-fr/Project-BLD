# src/models/tabular/xgb_model.py
"""
XGBoost-based edge prediction model (W1 in Symphony architecture).
"""
from pathlib import Path
from typing import Iterable, Optional, Dict, Any
import sys
import joblib
import pandas as pd
import numpy as np

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.base import BazaarModel

class XGBEdgeModel(BazaarModel):
    def __init__(self, feature_cols: Iterable[str], label_col: str, **xgb_kwargs):
        if XGBClassifier is None:
            raise ImportError("xgboost is required")
        self.feature_cols = list(feature_cols)
        self.label_col = label_col
        self.model = XGBClassifier(**xgb_kwargs)
    
    def fit(self, df: pd.DataFrame, **kwargs) -> None:
        X = df[self.feature_cols]
        y = df[self.label_col]
        self.model.fit(X, y, **kwargs)
    
    def predict(self, df: pd.DataFrame, return_proba: bool = True, **kwargs) -> pd.Series:
        X = df[self.feature_cols]
        if return_proba:
            proba = self.model.predict_proba(X, **kwargs)[:, 1]
            return pd.Series(proba, index=df.index, name="prob_up")
        else:
            pred = self.model.predict(X, **kwargs)
            return pd.Series(pred, index=df.index, name="prediction")
    
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"feature_cols": self.feature_cols, "label_col": self.label_col, "model": self.model}, path)
    
    def load(self, path: Path) -> "XGBEdgeModel":
        payload = joblib.load(path)
        self.feature_cols = payload["feature_cols"]
        self.label_col = payload["label_col"]
        self.model = payload["model"]
        return self
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        if not hasattr(self.model, 'feature_importances_'):
            return None
        return pd.DataFrame({'feature': self.feature_cols, 'importance': self.model.feature_importances_}).sort_values('importance', ascending=False)
    
    def get_params(self) -> Dict[str, Any]:
        return self.model.get_params()

__all__ = ['XGBEdgeModel']
