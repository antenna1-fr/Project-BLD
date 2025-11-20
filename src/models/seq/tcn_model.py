# src/models/seq/tcn_model.py
"""TCN model wrapper (S1/S2 proto in Symphony).

This file implements a compact, notebook-derived TCN suitable for
sequence-to-label classification used by the TCN prototype.

Features:
- TemporalBlock (dilated causal conv + residuals + dropout)
- Stacked TCN (configurable channels & levels)
- fit / predict / save / load to satisfy BazaarModel interface

Notes:
- fit() expects the leak-proof dataframe + y_enc array produced by
  src.data.tabular_dataset.build_leak_proof_dataset / load_leak_proof_dataset
- predict() returns a pandas Series aligned to the input df index where
  only rows that are valid sequence end indices receive predictions.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Sequence
import sys
import math
import logging

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    import torch.nn.functional as F
except Exception:
    torch = None
    nn = None

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.base import BazaarModel
from src.data.sequence_dataset import SequenceDataset, build_item_sequence_indices, guess_batch_size
from src.backtest.engine import run_backtest

LOG = logging.getLogger(__name__)


class TemporalBlock(nn.Module):
    """A single Temporal Block (dilated causal conv -> ReLU -> Dropout -> Residual)."""

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout):
        super().__init__()
        # Store padding for later cropping to maintain causal receptive field
        self._padding = padding
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        if self.conv1.bias is not None:
            nn.init.constant_(self.conv1.bias, 0)
        if self.downsample is not None:
            nn.init.constant_(self.downsample.weight, 0.0)

    def forward(self, x):
        out = self.net(x)
        # conv was applied with padding for causal conv; crop the extra right-side padding
        if self._padding and self._padding > 0:
            out = out[:, :, : -self._padding]
        res = x if self.downsample is None else self.downsample(x)
        return F.relu(out + res)


class TCN(nn.Module):
    """Stacked TCN composed of TemporalBlocks.

    Input shape: (batch, channels, seq_len)
    Output: logits over classes
    """

    def __init__(self, n_inputs: int, num_classes: int, channels: Sequence[int],
                 kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        num_levels = len(channels)
        for i in range(num_levels):
            in_ch = n_inputs if i == 0 else channels[i - 1]
            out_ch = channels[i]
            dilation_size = 2 ** i
            padding = (kernel_size - 1) * dilation_size
            layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size, stride=1,
                              dilation=dilation_size, padding=padding, dropout=dropout)
            )
        self.network = nn.Sequential(*layers)
        self.final = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        # x: [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)
        y = self.network(x)
        # take last timestep features
        out = y[:, :, -1]
        logits = self.final(out)
        return logits


class TCNSequenceModel(BazaarModel):
    """High-level model wrapper implementing fit / predict / save / load.

    config keys (defaults provided in trainer):
        nfeat: int (number of input features)
        window: int (sequence window length)
        channels: list of ints (TCN channels per level)
        kernel_size: int
        dropout: float
        num_classes: int (usually 3)
        epochs: int
        batch_size: int | None
        lr: float
        weight_decay: float
        device: str or torch.device
    """

    def __init__(self, config: Dict[str, Any]):
        if torch is None:
            raise ImportError("torch is required for TCNSequenceModel")
        self.config = dict(config)
        self.device = torch.device(self.config.get("device") if self.config.get("device") is not None
                                   else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: Optional[nn.Module] = None
        # placeholder for scaler/feature_cols if needed later
        self.feature_cols = self.config.get("feature_cols")
        # build model lazily when nfeat is known
        if "nfeat" in self.config and self.config.get("nfeat") is not None:
            self.model = self._build_model(self.config)
            self.model.to(self.device)

    def _build_model(self, config: Dict[str, Any]):
        nfeat = int(config["nfeat"])
        num_classes = int(config.get("num_classes", 3))
        channels = config.get("channels", [64, 64, 64])
        kernel_size = int(config.get("kernel_size", 3))
        dropout = float(config.get("dropout", 0.1))
        return TCN(n_inputs=nfeat, num_classes=num_classes, channels=channels,
                   kernel_size=kernel_size, dropout=dropout)

    def fit(self, df: pd.DataFrame, y_enc: Optional[np.ndarray] = None, feature_cols: Optional[Sequence[str]] = None,
            window: int = 64, stride: int = 4, val_mask: Optional[np.ndarray] = None, epochs: int = 3,
            batch_size: Optional[int] = None, lr: float = 1e-3, weight_decay: float = 0.0,
            verbose: bool = True, **kwargs) -> None:
        """Train the TCN on sequences built from df + y_enc.

        Args:
            df: leak-proof DataFrame (sorted by timestamp) produced by load_leak_proof_dataset
            y_enc: encoded labels (np.ndarray, same length as df)
            feature_cols: list of feature columns to use (required)
            window: sequence length
            stride: stride between windows
            val_mask: boolean mask for validation rows (same length as df)
            epochs: number of epochs
            batch_size: batch size (guessed if None)
            lr: learning rate
        """
        if y_enc is None:
            raise ValueError("y_enc (encoded labels) must be provided to fit()")
        if feature_cols is None:
            if self.feature_cols is None:
                raise ValueError("feature_cols must be provided (either in config or as argument)")
            feature_cols = self.feature_cols

        X_df = df[feature_cols].astype("float32")
        df_meta = df[["timestamp", "item", "mid_price"]].copy() if set(["timestamp", "item"]).issubset(df.columns) else df

        # compute end indices per item (train/val split)
        end_idx_all, end_idx_tr, end_idx_val, _ = build_item_sequence_indices(
            df=df,
            y_enc=y_enc,
            window=window,
            stride=stride,
            val_mask=val_mask,
            verbose=verbose,
        )

        # If no val indices found, keep a small fraction as val
        if len(end_idx_val) == 0 and len(end_idx_tr) > 100:
            # simple holdout: take 5% for val
            nval = max(1, int(0.05 * len(end_idx_tr)))
            end_idx_val = end_idx_tr[-nval:]
            end_idx_tr = end_idx_tr[:-nval]

        # Build datasets
        tr_ds = SequenceDataset(X_df, y_enc, df_meta, end_idx_tr, window)
        val_ds = SequenceDataset(X_df, y_enc, df_meta, end_idx_val, window) if len(end_idx_val) else None

        nfeat = X_df.shape[1]
        self.config.update({"nfeat": nfeat, "window": window, "feature_cols": list(feature_cols)})
        if self.model is None:
            self.model = self._build_model(self.config)
            self.model.to(self.device)

        # batch size guess
        if batch_size is None:
            batch_size = guess_batch_size(nfeat=nfeat, window=window)
        batch_size = int(min(batch_size, len(tr_ds)))

        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0) if val_ds is not None else None

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

        best_val = math.inf
        best_state = None

        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0
            n_seen = 0
            for batch in tr_loader:
                x, y, *_ = batch  # x: [B,T,C]
                x = x.to(self.device, dtype=torch.float32)
                y = y.to(self.device, dtype=torch.long)
                optimizer.zero_grad()
                logits = self.model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item()) * x.size(0)
                n_seen += x.size(0)
            train_loss = running_loss / max(1, n_seen)

            # validation
            val_loss = None
            if val_loader is not None:
                self.model.eval()
                running_v = 0.0
                n_v = 0
                with torch.no_grad():
                    for batch in val_loader:
                        x, y, *_ = batch
                        x = x.to(self.device, dtype=torch.float32)
                        y = y.to(self.device, dtype=torch.long)
                        logits = self.model(x)
                        loss = criterion(logits, y)
                        running_v += float(loss.item()) * x.size(0)
                        n_v += x.size(0)
                val_loss = running_v / max(1, n_v)
                scheduler.step(val_loss)

            if verbose:
                LOG.info(f"[tcn] epoch={epoch}/{epochs} train_loss={train_loss:.6f} val_loss={val_loss if val_loss is not None else 'n/a'}")

            if val_loss is not None and val_loss < best_val:
                best_val = val_loss
                best_state = {"state_dict": {k: v.cpu() for k, v in self.model.state_dict().items()},
                              "config": dict(self.config)}

        if best_state is not None:
            # load best
            self.model.load_state_dict(best_state["state_dict"])

    def predict(self, df: pd.DataFrame, feature_cols: Optional[Sequence[str]] = None,
                window: Optional[int] = None, stride: int = 4, batch_size: Optional[int] = None,
                return_proba: bool = True) -> pd.Series:
        """Apply the model to sequences constructed from `df` and return aligned predictions.

        Returns a pandas Series of predicted encoded labels (0/1/2) indexed like `df`.
        Rows that are not valid sequence ends are NaN.
        If return_proba is True, also returns a DataFrame with probability for buy class (encoded 2).
        """
        if self.model is None:
            raise ValueError("Model has not been built/trained")
        feature_cols = feature_cols or self.config.get("feature_cols")
        if feature_cols is None:
            raise ValueError("feature_cols must be provided to predict()")
        if window is None:
            window = int(self.config.get("window", 64))

        X_df = df[list(feature_cols)].astype("float32")
        df_meta = df[["timestamp", "item", "mid_price"]].copy() if set(["timestamp", "item"]).issubset(df.columns) else df

        # Build all valid end indices
        end_idx_all, _, _, _ = build_item_sequence_indices(df=df, y_enc=np.zeros(len(df), dtype=np.int8), window=window, stride=stride, val_mask=None, verbose=False)

        ds = SequenceDataset(X_df, np.zeros(len(df), dtype=np.int8), df_meta, end_idx_all, window)
        if batch_size is None:
            batch_size = guess_batch_size(nfeat=X_df.shape[1], window=window)
        loader = DataLoader(ds, batch_size=int(min(batch_size, len(ds))), shuffle=False, num_workers=0)

        # Prepare output containers
        preds = np.full(len(df), fill_value=np.nan)
        proba_buy = np.full(len(df), fill_value=np.nan)

        self.model.eval()
        with torch.no_grad():
            idx_ptr = 0
            for batch in loader:
                x, y, ts, it, mid = batch
                bsize = x.shape[0]
                x = x.to(self.device, dtype=torch.float32)
                logits = self.model(x)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                lab = probs.argmax(axis=1)
                # fill into preds according to end_idx_all ordering
                for i in range(bsize):
                    pos = int(end_idx_all[idx_ptr])
                    preds[pos] = int(lab[i])
                    # buy class is encoded as 2 (LABEL_MAP maps 1 -> 2 in tabular_dataset)
                    proba_buy[pos] = float(probs[i, 2]) if probs.shape[1] > 2 else float(probs[i, -1])
                    idx_ptr += 1

        # Return a DataFrame with two columns for convenience
        out = pd.DataFrame({"pred_label": pd.Series(preds, index=df.index),
                            "pred_proba_buy": pd.Series(proba_buy, index=df.index)})
        return out

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
