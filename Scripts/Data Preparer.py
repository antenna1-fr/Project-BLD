import os
import sqlite3
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# === CONFIG ===
DATA_DIR         = r'C:/Users/reyno/Desktop/Quant Finance Skyblock/Data/Processed'
DB_PATH          = os.path.join(DATA_DIR, 'bazaar.db')
OUTPUT_CSV       = os.path.join(DATA_DIR, 'improved_normalized_labeled.csv')
ITEM_TO_PLOT     = 'GRIFFIN_FEATHER'  
# long windows in minutes (hourly+ trends)
LONG_WINDOWS     = [60, 120, 240, 480]    # 1h, 2h, 4h, 8h
EMA_SPANS        = [60, 240]              # smoothing spans
LOOKAHEAD_MIN    = 40                    # label lookahead (in minutes)
PROFIT_UP        = 1.002                   # +.2% profit threshold
PROFIT_DOWN      = 0.998                   # -.2% loss threshold

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

# ============================================================================
# 3) LABEL based on future profit over LOOKAHEAD window
# ============================================================================
def label_group(g):
    prices = g['mid_price'].values
    labels = np.zeros(len(g), dtype=int)
    window = LOOKAHEAD_MIN
    for i in range(len(g) - window):
        future = prices[i+1:i+1+window]
        if future.max() > prices[i] * PROFIT_UP:
            labels[i] = 1
        elif future.min() < prices[i] * PROFIT_DOWN:
            labels[i] = -1
        else:
            labels[i] = 0
    g['label'] = labels
    return g

df = df.groupby('item', group_keys=False).apply(label_group)
# ============================================================================
# 4) NORMALIZE all numeric feature columns (excluding identifiers & label)
# ============================================================================
exclude = {'item','timestamp','datetime','label'}
num_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
scaler   = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])



# ============================================================================
# 5) SAVE to CSV
# ============================================================================
os.makedirs(DATA_DIR, exist_ok=True)
df_trimmed = df.iloc[:-300]  # remove last 300 rows to avoid future data leakage
df_trimmed.to_csv(OUTPUT_CSV, index=False)
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
label_counts = df_trimmed['label'].value_counts(normalize=True) * 100
print("Label distribution percentages:")
for label, pct in label_counts.sort_index().items():
    print(f"Label {label}: {pct:.2f}%")
