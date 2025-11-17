# src/models/seq/tcn_model.py
"""TCN model wrapper (S1/S2 proto in Symphony)."""
from pathlib import Path
from typing import Dict, Any
import sys
import pandas as pd

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.base import BazaarModel

class TCNSequenceModel(BazaarModel):
    def __init__(self, config: Dict[str, Any]):
        if torch is None:
            raise ImportError("torch is required")
        self.config = config
        self.model = self._build_model(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.model is not None:
            self.model.to(self.device)
    
    def _build_model(self, config):
        # TODO: Move TCN architecture from notebook
        return None
    
    def fit(self, df: pd.DataFrame, **kwargs) -> None:
        raise NotImplementedError("TCN training to be implemented")
    
    def predict(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        raise NotImplementedError("TCN inference to be implemented")
    
    def save(self, path: Path) -> None:
        if self.model is None:
            raise ValueError("No model to save")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.model.state_dict(), "config": self.config}, path)
    
    def load(self, path: Path) -> "TCNSequenceModel":
        ckpt = torch.load(path, map_location=self.device)
        self.config = ckpt["config"]
        self.model = self._build_model(self.config)
        if self.model is not None:
            self.model.load_state_dict(ckpt["state_dict"])
            self.model.to(self.device)
        return self

__all__ = ['TCNSequenceModel']
