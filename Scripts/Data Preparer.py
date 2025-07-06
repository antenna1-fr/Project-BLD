import os
import sqlite3
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as config

# === CONFIG ===
DATA_DIR         = str(config.PROCESSED_DIR)
DB_PATH          = str(config.DB_PATH)
OUTPUT_CSV       = str(config.PROCESSED_CSV)

ITEM_TO_PLOT     = 'GRIFFIN_FEATHER'  
# long windows in minutes (hourly+ trends)
LONG_WINDOWS     = [60, 120, 240, 720, 1440, 2880, 5760]    # 1h, 2h, 4h, 12h, 1d, 2d, 4d
EMA_SPANS        = [60, 240, 720, 2880]              # smoothing spans
LOOKAHEAD_UP     = 1440                    # label lookahead (in minutes)
LOOKAHEAD_DOWN   = 1440                    # label lookahead (in minutes)
PROFIT_UP        = 1.08                   # +8% profit threshold
PROFIT_DOWN      = 0.96                   # -4% loss threshold
VOLUME_THRESHOLD = 300                    # minimum weekly volume to consider item
MID_PRICE_THRESHOLD = 10_000              # minimum mid-price to consider item
TOP_LIQUIDITY_ITEMS = 120  # keep top N items by liquidity score
# ============================================================================
# 1) LOAD historical feature table from SQLite
# ============================================================================
con = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("SELECT * FROM model_data", con)
con.close()

# convert UNIX timestamp → datetime, sort
df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
df = df.sort_values(['item', 'datetime']).reset_index(drop=True)

# ============================================================================
# 2) COMPUTE hourly+ moving averages, std, and EMAs
# ============================================================================
# mid_price if not already present
if 'mid_price' not in df.columns:
    df['mid_price'] = (df['buy_price'] + df['sell_price']) / 2

df = df[df["sell_moving_week"] >=  VOLUME_THRESHOLD]  # filter by weekly volume
df = df[df["mid_price"]    >= MID_PRICE_THRESHOLD]  # filter by mid-price

# Rank & keep top amount by liquidity_score
df["liquidity_score"] = df["sell_moving_week"] * df["mid_price"]
topamount = (
    df.groupby("item")["liquidity_score"]
      .mean()
      .sort_values(ascending=False)
      .head(TOP_LIQUIDITY_ITEMS)
      .index
)
df = df[df["item"].isin(topamount)].copy()
df.drop(columns="liquidity_score", inplace=True)


# group-transform for each window
for w in LONG_WINDOWS:
    df[f'mid_ma{w}']  = df.groupby('item')['mid_price']\
                           .transform(lambda x: x.rolling(window=w, min_periods=1).mean())
    df[f'mid_std{w}'] = df.groupby('item')['mid_price']\
                           .transform(lambda x: x.rolling(window=w, min_periods=1).std())

# EMAs for smoothing
for span in EMA_SPANS:
    df[f'mid_ema{span}'] = df.groupby('item')['mid_price']\
                              .transform(lambda x: x.ewm(span=span, adjust=False).mean())

price_min = float(df["mid_price"].min())
price_max = float(df["mid_price"].max())

# ============================================================================
# 3) LABEL based on future profit over LOOKAHEAD window
# ============================================================================
def label_group(g):
    prices = g['mid_price'].values
    n = len(prices)
    labels = np.zeros(n, dtype=int)

    # use the larger window to make sure we don't run past the array
    max_window = max(LOOKAHEAD_UP, LOOKAHEAD_DOWN)

    for i in range(n - max_window):
        # look ahead with different horizons
        future_up   = prices[i+1 : i+1+LOOKAHEAD_UP]
        future_down = prices[i+1 : i+1+LOOKAHEAD_DOWN]

        # label 1 if an “up” threshold is hit within LOOKAHEAD_UP
        if future_up.max() > prices[i] * PROFIT_UP:
            labels[i] = 1
        # label -1 if a “down” threshold is hit within LOOKAHEAD_DOWN
        elif future_down.min() < prices[i] * PROFIT_DOWN:
            labels[i] = -1
        # otherwise 0 (hold)
        else:
            labels[i] = 0

    # the last max_window points remain at the default 0
    g['label'] = labels
    return g
df = df.groupby('item', group_keys=False).apply(label_group)
# ============================================================================
# 4) NORMALIZE all numeric feature columns (excluding identifiers & label)
# ============================================================================
# 4) SCALE numeric feature columns to the range [0, 1]
exclude  = {"item", "timestamp", "datetime", "label"}
num_cols = [c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0.00000001, 1.0))
df[num_cols] = scaler.fit_transform(df[num_cols])

 
joblib.dump(scaler, os.path.join(DATA_DIR, "feature_scaler.pkl"))

import json, pathlib
scale_meta = {"mid_price": {"min": price_min, "max": price_max}}
meta_path  = pathlib.Path("Data/Processed/rl_scaling.json")
meta_path.write_text(json.dumps(scale_meta, indent=2))
print("Wrote scaling meta →", meta_path)

# ============================================================================
# 5) SAVE to CSV
# ============================================================================
os.makedirs(DATA_DIR, exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved normalized & labeled data to {OUTPUT_CSV}")

# ============================================================================
# 6) PLOT features for a single item
# ============================================================================
item_df = df[df['item'] == ITEM_TO_PLOT]
if item_df.empty:
    raise ValueError(f"No data found for item '{ITEM_TO_PLOT}'")

plot_dir = os.path.join(DATA_DIR, 'Plots', ITEM_TO_PLOT)
os.makedirs(plot_dir, exist_ok=True)

# plot each numeric column
for col in num_cols + ['label']:
    plt.figure(figsize=(10, 3))
    plt.plot(item_df['datetime'], item_df[col], linewidth=1)
    plt.title(f"{ITEM_TO_PLOT} – {col}")
    plt.xlabel("Time")
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{col}.png"))
    plt.close()

print(f"Plots saved under {plot_dir}")
# ============================================================================  
# 7) PRINT LABEL DISTRIBUTION PERCENTAGES
# ============================================================================
label_counts = df['label'].value_counts(normalize=True) * 100
print("Label distribution percentages:")
for label, pct in label_counts.sort_index().items():
    print(f"Label {label}: {pct:.2f}%")
