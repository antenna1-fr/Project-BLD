# src/models/registry.py
"""Simple model registry to map friendly names to loader functions and default model keys.

This registry supports lightweight loading of saved model artifacts and returning
an instantiated model ready for inference.
"""
from pathlib import Path
from typing import Dict, Any

# Lazy imports to avoid heavy deps at import time


def _load_xgb(path: Path, **kwargs):
    from src.models.tabular.xgb_model import XGBEdgeModel
    # Create placeholder then load if artifact exists
    m = XGBEdgeModel(feature_cols=kwargs.get('feature_cols', []), label_col=kwargs.get('label_col', 'label_up'))
    if path.exists():
        m.load(path)
    return m


def _load_tcn(path: Path, config: Dict[str, Any] = None):
    from src.models.seq.tcn_model import TCNSequenceModel
    cfg = config or {}
    m = TCNSequenceModel(cfg)
    if path.exists():
        m.load(path)
    return m


# Registry entries map to loader type + path + optional config
_REGISTRY: Dict[str, Dict[str, Any]] = {
    "edge_xgb_v1": {
        "type": "xgb",
        "path": Path("outputs/xgb/xgb_model.joblib"),
    },
    "edge_tcn_v1": {
        "type": "tcn",
        "path": Path("outputs/tcn/tcn_model.pt"),
        "config": {},
    }
}

_DEFAULTS: Dict[str, str] = {
    "edge_model": "edge_xgb_v1"
}


def list_models():
    return list(_REGISTRY.keys())


def register(name: str, entry: Dict[str, Any]):
    _REGISTRY[name] = entry


def load_named_model(name: str):
    entry = _REGISTRY.get(name)
    if entry is None:
        raise KeyError(f"Unknown model name: {name}")
    mtype = entry.get("type")
    path = Path(entry.get("path"))
    if mtype == "xgb":
        return _load_xgb(path, **entry.get('meta', {}))
    if mtype == "tcn":
        return _load_tcn(path, entry.get('config'))
    raise NotImplementedError(f"Unknown model type: {mtype}")


def load_default_edge_model():
    name = _DEFAULTS.get("edge_model")
    if name is None:
        raise KeyError("No default edge model configured")
    return load_named_model(name)


__all__ = ["list_models", "register", "load_named_model", "load_default_edge_model", "_REGISTRY", "_DEFAULTS"]

