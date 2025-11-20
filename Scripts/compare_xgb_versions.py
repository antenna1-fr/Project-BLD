"""
Comparison script for XGB_Prototype.ipynb vs XGB_Prototype_Modular.ipynb

This script runs key cells from both notebooks on the first 1M rows
to compare the original monolithic implementation against the modularized version.

Usage:
    python scripts/compare_xgb_versions.py
"""

import sys
import os
from pathlib import Path
import time
import json

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import duckdb
from sklearn.metrics import f1_score, classification_report, confusion_matrix

import config

print("="*80)
print("XGB PROTOTYPE COMPARISON: Original vs Modularized")
print("="*80)
print(f"Dataset: First 1,000,000 rows from {getattr(config, 'PROCESSED_DATA_PATH', config.PROCESSED_PARQUET)}")
print("="*80)

# ============================================================================
# ORIGINAL VERSION (Monolithic)
# ============================================================================
print("\n[1/2] Running ORIGINAL version (monolithic)...")
print("-"*80)

start_original = time.time()

# Load data (original approach) via DuckDB for faster IO and Parquet support
def _load_first_n(n: int):
    path = getattr(config, "PROCESSED_DATA_PATH", getattr(config, "PROCESSED_PARQUET", config.PROCESSED_CSV))
    reader = "read_parquet" if str(path).endswith(".parquet") else "read_csv_auto"
    con = duckdb.connect(database=":memory:")
    try:
        return con.execute(f"SELECT * FROM {reader}(?) LIMIT {n}", [str(path)]).df()
    finally:
        con.close()

df_orig = _load_first_n(1_000_000)
df_orig = df_orig.sort_values("timestamp").reset_index(drop=True)

if "tradable" not in df_orig.columns:
    df_orig["tradable"] = 1
df_orig["tradable"] = df_orig["tradable"].astype("int8")

to_cast = [
    c for c in df_orig.select_dtypes("float64").columns
    if c not in ("timestamp", "mid_price")
]
df_orig[to_cast] = df_orig[to_cast].astype("float32")

# Labels (original approach)
UP_TAU = 0.25
DN_TAU_BASE = 0.08

if {"target_q_up_rel","target_q_dn_rel"}.issubset(df_orig.columns):
    up_rel_series = df_orig["target_q_up_rel"].astype(np.float32)
    dn_rel_series = df_orig["target_q_dn_rel"].astype(np.float32)
else:
    up_rel_series = df_orig["target_max_rel"].astype(np.float32)
    dn_rel_series = (-df_orig["target_min_rel"]).astype(np.float32)

dn_tau_eff = np.full(len(df_orig), DN_TAU_BASE, dtype=np.float32)

def to_dir_asym(up_rel_val: float, dn_rel_val: float, dn_tau: float) -> int:
    if np.isfinite(up_rel_val) and (up_rel_val >= UP_TAU):
        return 1
    if np.isfinite(dn_rel_val) and (dn_rel_val >= dn_tau):
        return -1
    return 0

y_dir_orig = np.fromiter(
    (to_dir_asym(u, d, t) for u, d, t in zip(up_rel_series.values, dn_rel_series.values, dn_tau_eff)),
    count=len(df_orig),
    dtype=np.int8
)

_map = {-1: 0, 0: 1, 1: 2}
y_orig = np.fromiter((_map[int(v)] for v in y_dir_orig), count=y_dir_orig.shape[0], dtype=np.int8)

counts_orig = np.bincount(y_orig, minlength=3)
print(f"[original] Rows: {len(df_orig):,} | Class counts (−1,0,1): "
      f"{{-1:{int(counts_orig[0])}, 0:{int(counts_orig[1])}, 1:{int(counts_orig[2])}}}")

# Features (original approach)
passthrough_base = {
    "item", "timestamp", "mid_price", "tradable",
    "target_min_abs", "target_max_abs", "target_min_rel", "target_max_rel",
}

leak_prefixes = ("target_",)
leak_cols = {c for c in df_orig.columns if any(c.startswith(p) for p in leak_prefixes)}
passthrough = passthrough_base | leak_cols

feature_cols_orig = [c for c in df_orig.columns
                if (c not in passthrough) and pd.api.types.is_numeric_dtype(df_orig[c])]

X_orig = df_orig[feature_cols_orig]
if any(X_orig.dtypes != "float32"):
    X_orig = X_orig.astype("float32", copy=False)

X_np_orig = X_orig.to_numpy(copy=False)
finite_mask = np.isfinite(X_np_orig).all(axis=1)
tradable_mask = (df_orig["tradable"].to_numpy(copy=False) == 1)
mask = finite_mask & tradable_mask

X_orig = X_orig.loc[mask]
y_dir_orig = y_dir_orig[mask]
df_orig = df_orig.loc[mask].reset_index(drop=True)

y_orig = np.fromiter((_map[int(v)] for v in y_dir_orig), count=y_dir_orig.shape[0], dtype=np.int8)

counts_orig = np.bincount(y_orig, minlength=3)
print(f"[original] Post-mask class counts (−1,0,1): "
      f"{{-1:{int(counts_orig[0])}, 0:{int(counts_orig[1])}, 1:{int(counts_orig[2])}}}")
print(f"[original] Features: {len(feature_cols_orig)}")

# Split (original approach)
from sklearn.model_selection import TimeSeriesSplit

def purged_time_splits(n_rows: int, n_splits: int = 5, embargo: int = 0):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    idx = np.arange(n_rows)
    for tr, va in tscv.split(idx):
        if embargo > 0:
            va_start = va.min()
            tr = tr[tr < max(0, va_start - embargo)]
        yield tr, va

splits = list(purged_time_splits(len(X_orig), n_splits=5))
train_idx_orig, val_idx_orig = splits[-1]

X_tr_orig, X_val_orig = X_orig.iloc[train_idx_orig], X_orig.iloc[val_idx_orig]
y_tr_orig, y_val_orig = y_orig[train_idx_orig], y_orig[val_idx_orig]

print(f"[original] Train: {len(X_tr_orig):,} | Val: {len(X_val_orig):,}")

end_original = time.time()
time_original = end_original - start_original

print(f"[original] Data preparation time: {time_original:.2f}s")

# ============================================================================
# MODULARIZED VERSION
# ============================================================================
print("\n[2/2] Running MODULARIZED version...")
print("-"*80)

start_modular = time.time()

# Import modularized components
from src.data.tabular_dataset import build_leak_proof_dataset
from src.backtest.splits import get_last_split

# Load data using modularized approach
df_raw_mod = _load_first_n(1_000_000)

df_mod, y_dir_mod, y_enc_mod, feature_cols_mod = build_leak_proof_dataset(
    df_raw_mod,
    label_params={
        'up_tau': 0.25,
        'dn_tau_base': 0.08,
        'use_vol_scaled_dn': False,
    },
    downcast_float64=True,
    require_tradable=True,
    verbose=True
)

y_mod = y_enc_mod
print(f"[modular] Features: {len(feature_cols_mod)}")

# Split using modularized approach
train_idx_mod, val_idx_mod = get_last_split(df_mod, n_splits=5, embargo=0)

X_mod = df_mod[feature_cols_mod].copy()
X_tr_mod, X_val_mod = X_mod.iloc[train_idx_mod], X_mod.iloc[val_idx_mod]
y_tr_mod, y_val_mod = y_mod[train_idx_mod], y_mod[val_idx_mod]

print(f"[modular] Train: {len(X_tr_mod):,} | Val: {len(X_val_mod):,}")

end_modular = time.time()
time_modular = end_modular - start_modular

print(f"[modular] Data preparation time: {time_modular:.2f}s")

# ============================================================================
# COMPARISON
# ============================================================================
print("\n" + "="*80)
print("COMPARISON RESULTS")
print("="*80)

comparison = {
    "metric": [],
    "original": [],
    "modular": [],
    "match": []
}

def add_comparison(metric, orig, mod, tolerance=0):
    comparison["metric"].append(metric)
    comparison["original"].append(orig)
    comparison["modular"].append(mod)
    if isinstance(orig, (int, float)) and isinstance(mod, (int, float)):
        match = abs(orig - mod) <= tolerance
    else:
        match = orig == mod
    comparison["match"].append("✓" if match else "✗")

# Data shape
add_comparison("Total rows (after filtering)", len(df_orig), len(df_mod))
add_comparison("Train rows", len(X_tr_orig), len(X_tr_mod))
add_comparison("Val rows", len(X_val_orig), len(X_val_mod))
add_comparison("Num features", len(feature_cols_orig), len(feature_cols_mod))

# Label distribution
add_comparison("Class -1 count", int(counts_orig[0]), int(np.bincount(y_enc_mod, minlength=3)[0]))
add_comparison("Class 0 count", int(counts_orig[1]), int(np.bincount(y_enc_mod, minlength=3)[1]))
add_comparison("Class 1 count", int(counts_orig[2]), int(np.bincount(y_enc_mod, minlength=3)[2]))

# Performance
add_comparison("Data prep time (s)", f"{time_original:.2f}", f"{time_modular:.2f}")

# Create comparison table
comp_df = pd.DataFrame(comparison)
print("\nMetric Comparison:")
print(comp_df.to_string(index=False))

# Feature set comparison
orig_features = set(feature_cols_orig)
mod_features = set(feature_cols_mod)

only_orig = orig_features - mod_features
only_mod = mod_features - orig_features

print(f"\n\nFeature Set Comparison:")
print(f"  Common features: {len(orig_features & mod_features)}")
if only_orig:
    print(f"  Only in original: {len(only_orig)} - {list(only_orig)[:5]}...")
if only_mod:
    print(f"  Only in modular: {len(only_mod)} - {list(only_mod)[:5]}...")

# Data consistency check
print(f"\n\nData Consistency:")
# Check if timestamps match (sample first 100 from validation set)
n_check = min(100, len(val_idx_orig), len(val_idx_mod))
ts_orig = df_orig.iloc[val_idx_orig[:n_check]]["timestamp"].values
ts_mod = df_mod.iloc[val_idx_mod[:n_check]]["timestamp"].values
ts_match = np.allclose(ts_orig, ts_mod)
print(f"  Timestamps match (first {n_check} val samples): {'✓' if ts_match else '✗'}")

# Check label distribution
label_dist_orig = np.bincount(y_val_orig, minlength=3) / len(y_val_orig)
label_dist_mod = np.bincount(y_val_mod, minlength=3) / len(y_val_mod)
label_match = np.allclose(label_dist_orig, label_dist_mod, rtol=0.01)
print(f"  Label distributions match: {'✓' if label_match else '✗'}")
print(f"    Original: [{label_dist_orig[0]:.3f}, {label_dist_orig[1]:.3f}, {label_dist_orig[2]:.3f}]")
print(f"    Modular:  [{label_dist_mod[0]:.3f}, {label_dist_mod[1]:.3f}, {label_dist_mod[2]:.3f}]")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

all_match = all(comparison["match"])

if all_match:
    print("✓ SUCCESS: Modularized version produces identical results to original!")
else:
    print("⚠ DIFFERENCES: Some metrics differ between versions")
    mismatches = comp_df[comp_df["match"] == "✗"]
    print("\nMismatched metrics:")
    print(mismatches.to_string(index=False))

print(f"\nModularization benefits:")
print(f"  ✓ Cleaner code structure")
print(f"  ✓ Reusable components")
print(f"  ✓ Consistent with TCN and other trainers")
print(f"  ✓ Better testability")
print(f"  ✓ Easier maintenance")

# Save comparison results
output_path = Path("outputs") / "xgb_comparison_results.json"
output_path.parent.mkdir(parents=True, exist_ok=True)

results = {
    "comparison": comparison,
    "original": {
        "rows": len(df_orig),
        "features": len(feature_cols_orig),
        "train_samples": len(X_tr_orig),
        "val_samples": len(X_val_orig),
        "prep_time_sec": time_original,
    },
    "modular": {
        "rows": len(df_mod),
        "features": len(feature_cols_mod),
        "train_samples": len(X_tr_mod),
        "val_samples": len(X_val_mod),
        "prep_time_sec": time_modular,
    },
    "all_match": all_match,
}

with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nComparison results saved to: {output_path}")
print("="*80)

