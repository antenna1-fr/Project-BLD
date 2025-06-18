import os
import time
import sqlite3
import requests
import signal
import sys

# === CONFIG ===
API_KEY           = ''  # ← your Hypixel API key
DATA_DIR          = r'C:/Users/reyno/Desktop/Quant Finance Skyblock/Data/Raw'
DB_PATH           = os.path.join(DATA_DIR, 'bazaar.db')
INTERVAL_SECONDS  = 60    # fetch every 60 seconds
MA_WINDOW_SECONDS = 60 * 60 * 24 * 7   # 1 week
VOL_WINDOW        = 5                    # last 5 samples
RSI_PERIOD        = 14
BB_PERIOD         = 20

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

STOP = False
def _signal_handler(signum, frame):
    global STOP
    STOP = True

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

def ensure_db():
    os.makedirs(DATA_DIR, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=30)
    cur = con.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    # raw table
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
    # model_data table
    all_cols = RAW_FIELDS + FEATURE_FIELDS
    cols_def = ",\n  ".join(
        f"{col} REAL" if col not in ('item',) else f"{col} TEXT"
        for col in all_cols
    )
    cur.execute(f"""
      CREATE TABLE IF NOT EXISTS model_data (
        {cols_def}
      );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_md_item_time ON model_data(item, timestamp);")
    con.commit()
    return con

def fetch_all_bazaar_data():
    url = f"https://api.hypixel.net/skyblock/bazaar?key={API_KEY}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()['products']

def compute_features(history):
    feat = {f: 0.0 for f in FEATURE_FIELDS}
    n = len(history)
    if n < 2:
        return feat

    mids = [(h['buy_price'] + h['sell_price'])/2 for h in history]
    # 1) lags & returns
    for lag in (1,5,10):
        if n > lag:
            prev, curr = mids[-1-lag], mids[-1]
            feat[f'mid_price_lag{lag}']  = prev
            feat[f'mid_return_lag{lag}'] = ((curr - prev)/prev*100) if prev else 0
    # 2) rolling stats & momentum
    for w in (3,5,10):
        if n >= w:
            window = mids[-w:]
            avg = sum(window)/w
            var = sum((x-avg)**2 for x in window)/(w-1) if w>1 else 0
            feat[f'mid_ma{w}']  = avg
            feat[f'mid_std{w}'] = var**0.5
            feat[f'mid_mom{w}'] = mids[-1] - window[0]
    # 3) RSI
    deltas = [mids[i+1]-mids[i] for i in range(n-1)]
    gains = [d for d in deltas if d>0]
    losses = [-d for d in deltas if d<0]
    avg_g = sum(gains[-RSI_PERIOD:])/(RSI_PERIOD) if len(gains)>=RSI_PERIOD else (sum(gains)/len(gains) if gains else 0)
    avg_l = sum(losses[-RSI_PERIOD:])/(RSI_PERIOD) if len(losses)>=RSI_PERIOD else (sum(losses)/len(losses) if losses else 0)
    feat['rsi14'] = 100.0 if avg_l==0 else 100.0 - (100.0/(1+avg_g/avg_l))
    # 4) Bollinger
    if n >= BB_PERIOD:
        window = mids[-BB_PERIOD:]
        ma = sum(window)/BB_PERIOD
        sd = (sum((x-ma)**2 for x in window)/(BB_PERIOD-1))**0.5
        feat['bollinger_upper'] = ma + 2*sd
        feat['bollinger_lower'] = ma - 2*sd
        feat['bollinger_width'] = ((feat['bollinger_upper'] - feat['bollinger_lower'])/ma) if ma else 0
    # 5) spread & imbalance
    curr = history[-1]
    feat['spread_price'] = curr['buy_price'] - curr['sell_price']
    tot = curr['buy_orders']+curr['sell_orders']
    feat['order_imbalance'] = ((curr['buy_orders']-curr['sell_orders'])/tot) if tot else 0
    # 6) volume change
    prev = history[-2]
    feat['buy_vol_change']  = ((curr['buy_volume']-prev['buy_volume'])/prev['buy_volume']) if prev['buy_volume'] else 0
    feat['sell_vol_change'] = ((curr['sell_volume']-prev['sell_volume'])/prev['sell_volume']) if prev['sell_volume'] else 0
    return feat

def run_loop():
    con = ensure_db()
    cur = con.cursor()

    while not STOP:
        ts = time.time()
        products = fetch_all_bazaar_data()

        for item_id, prod in products.items():
            snap = {
                'timestamp':       ts,
                'item':            item_id,
                'buy_price':       prod['quick_status']['buyPrice'],
                'sell_price':      prod['quick_status']['sellPrice'],
                'buy_volume':      prod['quick_status']['buyVolume'],
                'sell_volume':     prod['quick_status']['sellVolume'],
                'sell_orders':     prod['quick_status']['sellOrders'],
                'buy_orders':      prod['quick_status']['buyOrders'],
                'sell_moving_week':prod['quick_status']['sellMovingWeek'],
                'buy_moving_week': prod['quick_status']['buyMovingWeek']
            }

            # insert raw
            placeholders = ",".join("?" for _ in RAW_FIELDS)
            cols = ",".join(RAW_FIELDS)
            vals = tuple(snap[f] for f in RAW_FIELDS)
            cur.execute(f"INSERT INTO bazaar_raw ({cols}) VALUES ({placeholders})", vals)

            # fetch recent history for features
            cur.execute(
                "SELECT " + ",".join(RAW_FIELDS) +
                " FROM bazaar_raw WHERE item=? ORDER BY timestamp DESC LIMIT ?",
                (item_id, BB_PERIOD)
            )
            rows = cur.fetchall()
            history = [dict(zip(RAW_FIELDS, row)) for row in reversed(rows)]

            feats = compute_features(history)

            # insert model_data
            all_cols = RAW_FIELDS + FEATURE_FIELDS
            placeholders = ",".join("?" for _ in all_cols)
            cols = ",".join(all_cols)
            vals = tuple(snap[f] for f in RAW_FIELDS) + tuple(feats[f] for f in FEATURE_FIELDS)
            cur.execute(f"INSERT INTO model_data ({cols}) VALUES ({placeholders})", vals)

        con.commit()
        # sleep until next interval
        for _ in range(INTERVAL_SECONDS):
            if STOP:
                break
            time.sleep(1)

    con.close()
    print("Shutting down gracefully.")

if __name__ == "__main__":
    run_loop()
