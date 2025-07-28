import os
import time
import sqlite3
import requests
import signal
import sys
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as config

# === CONFIG ===
API_KEY           = ''  # ← your Hypixel API key
DATA_DIR          = str(config.RAW_DIR)
DB_PATH           = str(config.DB_PATH)
INTERVAL_SECONDS  = 60  # fetch every 60 seconds

RAW_FIELDS = [
    'timestamp', 'item',
    'buy_price', 'sell_price',
    'buy_volume', 'sell_volume',
    'sell_orders', 'buy_orders',
    'sell_moving_week', 'buy_moving_week'
]

ORDERBOOK_FIELDS = [
    'timestamp', 'item'
] + [
    f'{side}_price_{i}' for side in ('buy', 'sell') for i in range(30)
] + [
    f'{side}_amount_{i}' for side in ('buy', 'sell') for i in range(30)
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

    orderbook_cols = ",\n  ".join(
        f"{col} REAL" if col != 'item' else f"{col} TEXT"
        for col in ORDERBOOK_FIELDS
    )
    cur.execute(f"""
      CREATE TABLE IF NOT EXISTS orderbook_raw (
        {orderbook_cols}
      );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_orderbook_item_time ON orderbook_raw(item, timestamp);")

    con.commit()
    return con

def fetch_all_bazaar_data():
    url = f"https://api.hypixel.net/skyblock/bazaar?key={API_KEY}"
    for attempt in range(30):
        try:
            resp = session.get(url, timeout=(5, 30))
            resp.raise_for_status()
            print("Snapshot Taken")
            return resp.json()['products']
        except (requests.exceptions.RequestException, ValueError) as e:
            wait = 1.35 ** attempt
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Fetch error (attempt {attempt+1}/30): {e!r}; retrying in {wait:.2f}s...")
            time.sleep(wait)
    raise RuntimeError("Failed to fetch bazaar data after 30 attempts")

def run_loop():
    con = ensure_db()
    cur = con.cursor()

    while not STOP:
        ts = time.time()
        products = fetch_all_bazaar_data()

        for item_id, prod in products.items():
            quick = prod['quick_status']
            snap = {
                'timestamp':        ts,
                'item':             item_id,
                'buy_price':        quick['buyPrice'],
                'sell_price':       quick['sellPrice'],
                'buy_volume':       quick['buyVolume'],
                'sell_volume':      quick['sellVolume'],
                'sell_orders':      quick['sellOrders'],
                'buy_orders':       quick['buyOrders'],
                'sell_moving_week': quick['sellMovingWeek'],
                'buy_moving_week':  quick['buyMovingWeek']
            }
            placeholders = ",".join("?" for _ in RAW_FIELDS)
            cols = ",".join(RAW_FIELDS)
            vals = tuple(snap[f] for f in RAW_FIELDS)
            cur.execute(f"INSERT INTO bazaar_raw ({cols}) VALUES ({placeholders})", vals)

            buy = prod.get("buy_summary", [])[:30]
            sell = prod.get("sell_summary", [])[:30]
            while len(buy) < 30:
                buy.append({'pricePerUnit': 0.0, 'amount': 0.0})
            while len(sell) < 30:
                sell.append({'pricePerUnit': 0.0, 'amount': 0.0})

            ob_row = {
                'timestamp': ts,
                'item': item_id
            }
            for i in range(30):
                ob_row[f'buy_price_{i}'] = buy[i]['pricePerUnit']
                ob_row[f'buy_amount_{i}'] = buy[i]['amount']
                ob_row[f'sell_price_{i}'] = sell[i]['pricePerUnit']
                ob_row[f'sell_amount_{i}'] = sell[i]['amount']

            ob_fields = list(ob_row.keys())
            placeholders = ",".join("?" for _ in ob_fields)
            cols = ",".join(ob_fields)
            vals = tuple(ob_row[f] for f in ob_fields)
            cur.execute(f"INSERT INTO orderbook_raw ({cols}) VALUES ({placeholders})", vals)
            if item_id == "ENCHANTED_DIAMOND":
                print("=== ENCHANTED_DIAMOND Snapshot ===")
                print("Quick Status:")
                print(prod['quick_status'])

                print("\nTop 30 Buy Summary:")
                for entry in prod.get("buy_summary", [])[:30]:
                    print(entry)

                print("\nTop 30 Sell Summary:")
                for entry in prod.get("sell_summary", [])[:30]:
                    print(entry)
                print("===================================")
                


        con.commit()

        for _ in range(INTERVAL_SECONDS):
            if STOP:
                break
            time.sleep(1)

    con.close()
    print("Shutting down gracefully.")

if __name__ == "__main__":
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD", "OPTIONS"]
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    run_loop()
