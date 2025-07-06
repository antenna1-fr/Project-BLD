import os
import sqlite3
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as config

# === CONFIG ===
DATA_DIR    = str(config.DATA_DIR)
DB_PATH     = str(config.DB_PATH)
OLD_CSV     = os.path.join(DATA_DIR, 'old_history.csv')  # your existing long CSV

# Fields must exactly match your old CSV’s column names
RAW_FIELDS = [
    'timestamp','item',
    'buy_price','sell_price',
    'buy_volume','sell_volume',
    'sell_orders','buy_orders',
    'sell_moving_week','buy_moving_week'
]
FEATURE_FIELDS = [
    'mid_price_lag1','mid_price_lag5','mid_price_lag10',
    'mid_return_lag1','mid_return_lag5','mid_return_lag10',
    'mid_ma3','mid_std3','mid_mom3',
    'mid_ma5','mid_std5','mid_mom5',
    'mid_ma10','mid_std10','mid_mom10',
    'rsi14',
    'bollinger_upper','bollinger_lower','bollinger_width',
    'spread_price','order_imbalance',
    'buy_vol_change','sell_vol_change'
]

def ensure_db_schema(con):
    """Re-create tables/indexes if missing (same as your collector)."""
    cur = con.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS bazaar_raw (
          timestamp        REAL,
          item             TEXT,
          buy_price        REAL,
          sell_price       REAL,
          buy_volume       REAL,
          sell_volume      REAL,
          sell_orders      INTEGER,
          buy_orders       INTEGER,
          sell_moving_week REAL,
          buy_moving_week  REAL
        );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_raw_item_time ON bazaar_raw(item, timestamp);")

    all_cols = RAW_FIELDS + FEATURE_FIELDS
    cols_def = ",\n  ".join(
        f"{c} REAL" if c not in ('item',) else f"{c} TEXT"
        for c in all_cols
    )
    cur.execute(f"""
      CREATE TABLE IF NOT EXISTS model_data (
        {cols_def}
      );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_md_item_time ON model_data(item, timestamp);")
    con.commit()

def main():
    # 1) load old CSV
    df = pd.read_csv(OLD_CSV)

    # 2) open DB & ensure schema
    os.makedirs(DATA_DIR, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=30)
    ensure_db_schema(con)
    cur = con.cursor()

    # 3) prepare INSERT statements
    raw_cols        = ",".join(RAW_FIELDS)
    raw_ph         = ",".join("?" for _ in RAW_FIELDS)
    raw_sql         = f"INSERT INTO bazaar_raw ({raw_cols}) VALUES ({raw_ph})"

    all_cols        = RAW_FIELDS + FEATURE_FIELDS
    md_cols         = ",".join(all_cols)
    md_ph           = ",".join("?" for _ in all_cols)
    md_sql          = f"INSERT INTO model_data ({md_cols}) VALUES ({md_ph})"

    # 4) iterate & bulk-insert (commit every 1000 rows)
    for i, row in df.iterrows():
        raw_vals  = tuple(row[field] for field in RAW_FIELDS)
        feat_vals = tuple(row.get(field, 0.0) for field in FEATURE_FIELDS)
        cur.execute(raw_sql, raw_vals)
        cur.execute(md_sql, raw_vals + feat_vals)

        if i % 1000 == 0:
            con.commit()

    con.commit()
    con.close()
    print(f"Imported {len(df)} rows from {OLD_CSV} into {DB_PATH}.")

if __name__ == "__main__":
    main()
