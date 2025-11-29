# Data Preparer v2 — execution-aware, volume-aware, streaming, anti-phantom-liquidity

import os, gc, json, math, sqlite3, sys
from pathlib import Path
from typing import List, Tuple
import duckdb
import joblib, numpy as np, pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

# --- project config import (unchanged)
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as config

# =================== CONFIG ===================
DATA_DIR        = Path(str(config.PROCESSED_DIR))
DB_PATH         = Path(str(config.DB_PATH))
OUTPUT_DATASET  = Path(str(config.PROCESSED_DATA_PATH))
OUTPUT_PARQUET  = Path(str(config.PROCESSED_PARQUET))

VOLUME_THRESHOLD      = 500
MID_PRICE_THRESHOLD   = 5_000
TOP_LIQUIDITY_ITEMS   = 800   # keep as you logged

RET_WINDOWS     = [60, 240, 720, 1440, 2880, 5760, 14400]
EMA_SPANS       = [5, 15, 60, 240, 720, 1440, 2880, 5760, 14400]
VOL_WINDOWS     = [15, 60, 240, 720, 1440, 2880, 5760, 14400]

# --- Parallelism knobs ---
MAX_WORKERS = max(1, (os.cpu_count() or 2) - 1)

OB_LEVELS       = 10
OB_BUCKETS      = (1, 5, 10)

# ==== EXECUTION-AWARE LABEL CONFIG (NEW) ====
# Long horizon = how far forward we look for "best long exit"
# Short horizon = how far forward we look for "best short exit"
LOOKAHEAD_UP    = 10000  # minutes forward for long-side PnL (e.g. 7 days)
LOOKAHEAD_DN    = 1440  # minutes forward for short-side PnL (e.g. 1 day)

# Keep legacy names for meta/backward-compat; now defined off the above
LOOKAHEAD_MIN   = LOOKAHEAD_DN
LOOKAHEAD_MAX   = LOOKAHEAD_UP

# Label trade size config (per item, then per-row clamped by depth)
LABEL_Q_DEPTH_PERCENTILE = 25.0   # percentile of min(depth_buy_10, depth_sell_10)
LABEL_Q_FRACTION_DEPTH   = 0.25   # fraction of that depth used as base label qty
LABEL_Q_MIN_ITEMS        = 1.0    # min items for a label
LABEL_Q_MAX_ITEMS        = 50_000.0  # absolute cap for very liquid items
LABEL_Q_MAX_DEPTH_FRAC   = 0.9    # per-row: can't exceed 90% of available depth

# Jump regime detection
JUMP_RET_K               = 4.0    # |ret_1| > K * vol_60 => jump
JUMP_REL_SPREAD_THRESH   = 0.10   # or rel_qspread > 10%
# Robust forward-window label config (legacy mid-based, still kept)
ROBUST_FWD_WINDOW = 120        # minutes forward for robust stat
ROBUST_Q_UP = 0.50
ROBUST_Q_DN = 0.50

# per-direction robust lookahead/window (legacy mid-based)
LOOKAHEAD_MAX_UP   = LOOKAHEAD_UP
LOOKAHEAD_MAX_DN   = LOOKAHEAD_DN
ROBUST_FWD_WINDOW_UP = ROBUST_FWD_WINDOW
ROBUST_FWD_WINDOW_DN = ROBUST_FWD_WINDOW
ROBUST_Q_UP_DIR    = ROBUST_Q_UP
ROBUST_Q_DN_DIR    = ROBUST_Q_DN

# Trim last N minutes per item (avoid unlabeled rows)
TRIM_MINUTES    = 0  # 0 to disable

SCALER_KIND             = "robust"   # "robust" or "standard"
ROBUST_QUANTILE_RANGE   = (10, 90)
MAX_SCALER_SAMPLES      = 2_000_000
RANDOM_SEED             = 42

TMP_DIR = DATA_DIR / "tmp"
ARTIFACTS = {
    "scaler": DATA_DIR / "feature_scaler.pkl",
    "meta":   DATA_DIR / "preparer_meta.json",
    "items":  DATA_DIR / "allowed_items.txt",
    "items_final": DATA_DIR / "allowed_items_final.txt",
}

# Row-gating safety defaults
if 'DROP_NONTRADABLE_ROWS' not in globals(): DROP_NONTRADABLE_ROWS = False
if 'POSTHOC_TRADABLE_MIN'  not in globals(): POSTHOC_TRADABLE_MIN  = 0.60
if 'POSTHOC_MIN_ITEM_ROWS' not in globals(): POSTHOC_MIN_ITEM_ROWS = 5000

# =================== UTILS ===================

def _escape_identifier(name: str) -> str:
    """
    Minimal DuckDB-safe identifier escaper.

    Wraps column/table names in double quotes and doubles any internal quotes.
    """
    s = str(name)
    return '"' + s.replace('"', '""') + '"'


def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)


def new_sql_conn() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, timeout=60, check_same_thread=False)
    try:
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        con.execute("PRAGMA temp_store=MEMORY;")
        con.execute("PRAGMA mmap_size=134217728;")
    except Exception:
        pass
    return con

def sql(conn: sqlite3.Connection, q: str, params=()) -> pd.DataFrame:
    return pd.read_sql_query(q, conn, params=params)

def top_items(conn: sqlite3.Connection) -> List[str]:
    """
    Pick items that are reliably liquid: price, volume, and spread sane for most bars.
    """
    MAX_REL_SPREAD = 0.20
    MIN_TRADABLE_RATIO = 0.90
    MIN_MID_OK_RATIO = 0.90
    MIN_VOL_OK_RATIO = 0.80

    q = f"""
    WITH base AS (
      SELECT
        item,
        (buy_price + sell_price)/2.0 AS mid_price,
        sell_moving_week,
        buy_orders, sell_orders,
        CASE
          WHEN buy_orders > 0 AND sell_orders > 0
               AND ( (sell_price - buy_price) /
                     NULLIF((buy_price + sell_price)/2.0,0) ) <= {MAX_REL_SPREAD}
          THEN 1 ELSE 0
        END AS tradable_flag,
        CASE WHEN (buy_price + sell_price)/2.0 >= ? THEN 1 ELSE 0 END AS mid_ok,
        CASE WHEN sell_moving_week >= ?            THEN 1 ELSE 0 END AS vol_ok
      FROM bazaar_raw
    ),
    agg AS (
      SELECT
        item,
        AVG(CAST(tradable_flag AS FLOAT)) AS tradable_ratio,
        AVG(CAST(mid_ok AS FLOAT))        AS mid_ok_ratio,
        AVG(CAST(vol_ok AS FLOAT))        AS vol_ok_ratio,
        AVG(sell_moving_week * mid_price) AS liq_score
      FROM base
      GROUP BY item
    )
    SELECT item
    FROM agg
    WHERE tradable_ratio >= {MIN_TRADABLE_RATIO}
      AND mid_ok_ratio   >= {MIN_MID_OK_RATIO}
      AND vol_ok_ratio   >= {MIN_VOL_OK_RATIO}
    ORDER BY liq_score DESC
    LIMIT ?
    """
    df = sql(conn, q, (MID_PRICE_THRESHOLD, VOLUME_THRESHOLD, TOP_LIQUIDITY_ITEMS))
    return df["item"].tolist()

# =================== FETCH ===================
def fetch_item(conn: sqlite3.Connection, item: str) -> pd.DataFrame:
    ob_cols = []
    for side in ("buy", "sell"):
        for k in range(OB_LEVELS):
            ob_cols += [f"{side}_price_{k}", f"{side}_amount_{k}"]
    ob_select = ", ".join([f"o.{c}" for c in ob_cols])

    q = f"""
    SELECT
      b.timestamp, b.item,
      b.buy_price, b.sell_price,
      b.buy_volume, b.sell_volume,
      b.sell_orders, b.buy_orders,
      b.sell_moving_week, b.buy_moving_week,
      {ob_select}
    FROM bazaar_raw b
    LEFT JOIN orderbook_raw o
      ON b.item = o.item AND b.timestamp = o.timestamp
    WHERE b.item = ?
    ORDER BY b.timestamp ASC
    """
    df = sql(conn, q, (item,))
    float_cols = [c for c in df.columns if c != "item"]
    df[float_cols] = df[float_cols].astype(np.float32)
    return df

# =================== ORDERBOOK FEATURES ===================
def compute_orderbook_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orderbook features:
      - L0 best prices/amounts & microprice
      - Bucket aggregates for k in OB_BUCKETS (depth, VWAPs, imbalance, slopes, depth-weighted spread)
      - 'tradable' flag
      - Keep bucket aggregates, prune most raw L2 columns
    """
    # L0 best prices/amounts
    bb = df.get("buy_price_0").astype(np.float32)
    ba = df.get("sell_price_0").astype(np.float32)
    bv0 = df.get("buy_amount_0").astype(np.float32)
    sv0 = df.get("sell_amount_0").astype(np.float32)

    mid0 = ((bb + ba) * 0.5).astype(np.float32)

    base = {
        "best_bid": bb,
        "best_ask": ba,
        "ba_spread": (ba - bb).astype(np.float32),
        "rel_spread": np.where((bb + ba) > 0,
                               (ba - bb) / ((bb + ba) * 0.5),
                               0.0).astype(np.float32),
        "microprice_l1": ((ba * bv0 + bb * sv0) /
                          (bv0 + sv0 + 1e-8)).astype(np.float32),
    }
    base_df = pd.DataFrame(base, index=df.index)
    base_df["microprice_l1"] = base_df["microprice_l1"].where(
        np.isfinite(base_df["microprice_l1"]), mid0
    )
    df = pd.concat([df, base_df], axis=1, copy=False)

    # Row-level tradability
    df["tradable"] = (
        (df["best_bid"] > 0) & (df["best_ask"] > 0) &
        (df.get("buy_amount_0", 0) > 0) & (df.get("sell_amount_0", 0) > 0) &
        (df["rel_spread"] <= 0.05)
    ).astype(np.int8)

    # Bucket aggregates
    for k in OB_BUCKETS:
        buy_amt = np.zeros(len(df), dtype=np.float32)
        sell_amt = np.zeros(len(df), dtype=np.float32)
        vwap_bid_num = np.zeros(len(df), dtype=np.float32)
        vwap_ask_num = np.zeros(len(df), dtype=np.float32)

        for i in range(k):
            pb = df.get(f"buy_price_{i}", 0.0).astype(np.float32)
            pa = df.get(f"sell_price_{i}", 0.0).astype(np.float32)
            ab = df.get(f"buy_amount_{i}", 0.0).astype(np.float32)
            aa = df.get(f"sell_amount_{i}", 0.0).astype(np.float32)
            buy_amt += ab
            sell_amt += aa
            vwap_bid_num += (pb * ab)
            vwap_ask_num += (pa * aa)

        denom_b = np.where(buy_amt > 0, buy_amt, np.nan)
        denom_s = np.where(sell_amt > 0, sell_amt, np.nan)
        vwap_bid = (vwap_bid_num / denom_b)
        vwap_ask = (vwap_ask_num / denom_s)
        vwap_mid = (vwap_bid + vwap_ask) * 0.5

        dws = (vwap_ask - vwap_bid) / np.where(mid0 > 0, mid0, np.nan)

        new_cols = {
            f"depth_buy_{k}": buy_amt,
            f"depth_sell_{k}": sell_amt,
            f"vwap_bid_{k}": vwap_bid.astype(np.float32),
            f"vwap_ask_{k}": vwap_ask.astype(np.float32),
            f"vwap_mid_{k}": vwap_mid.astype(np.float32),
            f"ob_imb_{k}": np.where(
                (buy_amt + sell_amt) > 0,
                (buy_amt - sell_amt) / (buy_amt + sell_amt),
                0.0
            ).astype(np.float32),
            f"bid_slope_{k}": (bb - df.get(f"buy_price_{k-1}", bb).astype(np.float32)) / max(k - 1, 1),
            f"ask_slope_{k}": (df.get(f"sell_price_{k-1}", ba).astype(np.float32) - ba) / max(k - 1, 1),
            f"dws_{k}": dws.astype(np.float32),
        }
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1, copy=False)

    # PRUNE low-signal raw OB columns after aggregates:
    drop_cols = []
    for i in range(1, OB_LEVELS):
        drop_cols += [f"buy_price_{i}", f"sell_price_{i}"]
    for i in range(2, OB_LEVELS):
        drop_cols += [f"buy_amount_{i}", f"sell_amount_{i}"]
    drop_cols += ["bid_slope_1", "ask_slope_1"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors="ignore")

    return df

# =================== CORE FEATURES ===================
def compute_core_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Core market features:
      - mid, spreads, imbalance
      - log returns & multi-scale returns
      - realized vol
      - vol regime ratios
      - EMAs
      - jump flag (NEW)
    """
    # Mid & spreads
    df["mid_price"] = ((df["buy_price"] + df["sell_price"]) * 0.5).astype(np.float32)
    df["quoted_spread"] = (df["sell_price"] - df["buy_price"]).astype(np.float32)
    df["rel_qspread"]   = np.where(df["mid_price"] > 0,
                                   df["quoted_spread"] / df["mid_price"],
                                   0.0).astype(np.float32)

    # Order/flow imbalance and normalized flow ratios
    df["order_imbalance"] = np.where(
        (df["buy_orders"] + df["sell_orders"]) > 0,
        (df["buy_orders"] - df["sell_orders"]) / (df["buy_orders"] + df["sell_orders"]),
        0.0
    ).astype(np.float32)
    df["flow_vol_ratio"] = (df["buy_volume"] / (df["buy_volume"] + df["sell_volume"] + 1e-8)).astype(np.float32)
    df["flow_ord_ratio"] = (df["buy_orders"] / (df["buy_orders"] + df["sell_orders"] + 1e-8)).astype(np.float32)
    df["flow_week_ratio"] = (df["buy_moving_week"] / (df["buy_moving_week"] + df["sell_moving_week"] + 1e-8)).astype(np.float32)

    # Log returns
    eps = 1e-8
    df["log_mid"] = np.log(df["mid_price"] + eps).astype(np.float32)
    df["ret_1"] = df["log_mid"].diff(1).astype(np.float32)

    # Longer-horizon returns
    for w in RET_WINDOWS:
        if w == 1:
            continue
        df[f"ret_{w}"] = (df["log_mid"].diff(w)).astype(np.float32)

    # Realized vol (ret_1)
    for w in VOL_WINDOWS:
        df[f"vol_{w}"] = df["ret_1"].rolling(
            w, min_periods=max(2, int(w/3))
        ).std().astype(np.float32)

    # Volatility regime ratios
    if "vol_15" in df and "vol_1440" in df:
        df["vol_ratio_15_1440"] = (df["vol_15"] / (df["vol_1440"] + 1e-8)).astype(np.float32)
    if "vol_60" in df and "vol_1440" in df:
        df["vol_ratio_60_1440"] = (df["vol_60"] / (df["vol_1440"] + 1e-8)).astype(np.float32)
    if "vol_240" in df and "vol_1440" in df:
        df["vol_ratio_240_1440"] = (df["vol_240"] / (df["vol_1440"] + 1e-8)).astype(np.float32)

    # Ensure dws_k stays float32
    for k in OB_BUCKETS:
        vk = f"dws_{k}"
        if vk in df.columns:
            df[vk] = df[vk].astype(np.float32)

    # EMAs
    ema_cols = {}
    for span in EMA_SPANS:
        ema_cols[f"mid_ema_{span}"]  = df["mid_price"].ewm(span=span, adjust=False).mean().astype(np.float32)
        ema_cols[f"imb_ema_{span}"]  = df["order_imbalance"].ewm(span=span, adjust=False).mean().astype(np.float32)
        ema_cols[f"qsp_ema_{span}"]  = df["rel_qspread"].ewm(span=span, adjust=False).mean().astype(np.float32)
        ema_cols[f"ret1_ema_{span}"] = df["ret_1"].ewm(span=span, adjust=False).mean().astype(np.float32)
    df = pd.concat([df, pd.DataFrame(ema_cols, index=df.index)], axis=1, copy=False)

    # Jump regime flag (NEW)
    if "vol_60" in df.columns:
        vol60 = df["vol_60"].fillna(0.0).to_numpy()
        ret_abs = df["ret_1"].abs().fillna(0.0).to_numpy()
        spread = df["rel_qspread"].fillna(0.0).to_numpy()
        jump = ((ret_abs > (JUMP_RET_K * (vol60 + 1e-8))) |
                (spread > JUMP_REL_SPREAD_THRESH))
        df["jump_flag"] = jump.astype(np.int8)
    else:
        df["jump_flag"] = 0

    return df

# =================== TARGET HELPERS ===================
from collections import deque
from numpy.lib.stride_tricks import sliding_window_view

def future_window_extrema(series: pd.Series, horizon: int) -> Tuple[pd.Series, pd.Series]:
    """
    Forward-only min/max over next `horizon` steps: [t+1 .. t+horizon].
    """
    x = series.values.astype(np.float32)
    n = len(x)
    if n == 0 or horizon <= 0:
        z = np.full(n, np.nan, dtype=np.float32)
        return pd.Series(z, index=series.index), pd.Series(z, index=series.index)

    min_deq, max_deq = deque(), deque()
    fmin = np.full(n, np.nan, dtype=np.float32)
    fmax = np.full(n, np.nan, dtype=np.float32)

    j_start, j_end = 1, min(horizon, n - 1)
    for j in range(j_start, j_end + 1):
        while min_deq and x[min_deq[-1]] >= x[j]:
            min_deq.pop()
        min_deq.append(j)
        while max_deq and x[max_deq[-1]] <= x[j]:
            max_deq.pop()
        max_deq.append(j)

    for i in range(n):
        fmin[i] = x[min_deq[0]] if min_deq else np.nan
        fmax[i] = x[max_deq[0]] if max_deq else np.nan
        leave_idx = i + 1
        enter_idx = i + 1 + horizon
        if min_deq and min_deq[0] == leave_idx:
            min_deq.popleft()
        if max_deq and max_deq[0] == leave_idx:
            max_deq.popleft()
        if enter_idx < n:
            while min_deq and x[min_deq[-1]] >= x[enter_idx]:
                min_deq.pop()
            min_deq.append(enter_idx)
            while max_deq and x[max_deq[-1]] <= x[enter_idx]:
                max_deq.pop()
            max_deq.append(enter_idx)
    return pd.Series(fmin, index=series.index), pd.Series(fmax, index=series.index)

def future_best_quantile_over_windows(series: pd.Series,
                                      lookahead: int,
                                      win: int,
                                      q: float,
                                      pick: str) -> pd.Series:
    """
    SAME as your v1: mid-based robust lookahead. Kept for diagnostics.
    """
    x = series.to_numpy(dtype=np.float32, copy=False)
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float32)
    if n == 0 or lookahead <= 0 or win <= 0 or q < 0 or q > 1:
        return pd.Series(out, index=series.index)

    if n < win:
        return pd.Series(out, index=series.index)
    windows = sliding_window_view(x, win)

    per_win_q = np.quantile(windows, q, axis=1).astype(np.float32)
    m = len(per_win_q)

    L = lookahead - win + 1
    if L <= 0 or m < L:
        return pd.Series(out, index=series.index)

    # moving extreme over per_win_q with window L
    dq = deque()
    best = np.empty(m - L + 1, dtype=np.float32)
    is_max = (pick == "max")

    def better(a, b):
        return a >= b if is_max else a <= b

    for i, val in enumerate(per_win_q):
        while dq and better(val, per_win_q[dq[-1]]):
            dq.pop()
        dq.append(i)
        if dq[0] <= i - L:
            dq.popleft()
        if i >= L - 1:
            best[i - L + 1] = per_win_q[dq[0]]

    max_t = min(len(best) - 2, n - lookahead - 1)
    if max_t >= 0:
        out[0:max_t+1] = best[1:max_t+2]
    return pd.Series(out, index=series.index)

# =================== EXECUTION-AWARE HELPERS ===================

def compute_label_base_qty_for_item(df: pd.DataFrame) -> float:
    """
    Compute an item-level base label size from orderbook depth.
    Use min(depth_buy_10, depth_sell_10) at a given percentile, then fraction of that.
    """
    depth_buy = df.get("depth_buy_10", pd.Series(0.0, index=df.index)).to_numpy()
    depth_sell = df.get("depth_sell_10", pd.Series(0.0, index=df.index)).to_numpy()
    depth_min = np.minimum(depth_buy, depth_sell)
    depth_pos = depth_min[depth_min > 0]

    if depth_pos.size == 0:
        return 0.0

    depth_ref = float(np.nanpercentile(depth_pos, LABEL_Q_DEPTH_PERCENTILE))
    base_q = depth_ref * LABEL_Q_FRACTION_DEPTH
    base_q = max(LABEL_Q_MIN_ITEMS, min(base_q, LABEL_Q_MAX_ITEMS))
    return float(base_q)

def compute_exec_vwap_for_qty(df: pd.DataFrame,
                              side: str,
                              qty: np.ndarray) -> pd.Series:
    """
    Compute per-row VWAP execution price for a given quantity (per-row),
    using bucket aggregates: top-1, 1-5, 6-10 levels.
      side: "bid" (instasell) or "ask" (instabuy)
      qty: np.ndarray of shape (n,) with requested size in items (may vary per row)
    """
    if side not in ("bid", "ask"):
        raise ValueError("side must be 'bid' or 'ask'")

    n = len(df)
    q = qty.astype(np.float32)
    out = np.full(n, np.nan, dtype=np.float32)

    if n == 0:
        return pd.Series(out, index=df.index)

    if side == "bid":
        p0 = df["best_bid"].to_numpy(dtype=np.float32)
        depth1 = df.get("depth_buy_1", pd.Series(0.0, index=df.index)).to_numpy(dtype=np.float32)
        depth5 = df.get("depth_buy_5", pd.Series(0.0, index=df.index)).to_numpy(dtype=np.float32)
        depth10 = df.get("depth_buy_10", pd.Series(0.0, index=df.index)).to_numpy(dtype=np.float32)
        vwap5 = df.get("vwap_bid_5", pd.Series(np.nan, index=df.index)).to_numpy(dtype=np.float32)
        vwap10 = df.get("vwap_bid_10", pd.Series(np.nan, index=df.index)).to_numpy(dtype=np.float32)
    else:
        p0 = df["best_ask"].to_numpy(dtype=np.float32)
        depth1 = df.get("depth_sell_1", pd.Series(0.0, index=df.index)).to_numpy(dtype=np.float32)
        depth5 = df.get("depth_sell_5", pd.Series(0.0, index=df.index)).to_numpy(dtype=np.float32)
        depth10 = df.get("depth_sell_10", pd.Series(0.0, index=df.index)).to_numpy(dtype=np.float32)
        vwap5 = df.get("vwap_ask_5", pd.Series(np.nan, index=df.index)).to_numpy(dtype=np.float32)
        vwap10 = df.get("vwap_ask_10", pd.Series(np.nan, index=df.index)).to_numpy(dtype=np.float32)

    depth1 = np.maximum(depth1, 0.0)
    depth5 = np.maximum(depth5, depth1)
    depth10 = np.maximum(depth10, depth5)

    # Clamp qty to available depth (to avoid NaNs from insufficient depth)
    depth_total = depth10
    q_clamped = np.minimum(q, depth_total)
    q_clamped = np.where(depth_total > 0, q_clamped, 0.0)

    # Costs for full buckets
    cost1 = depth1 * p0
    cost5 = depth5 * vwap5
    cost10 = depth10 * vwap10

    # Segments: [0..1], (1..5], (5..10]
    seg_vol_2_5 = np.maximum(depth5 - depth1, 1e-6)
    seg_cost_2_5 = np.maximum(cost5 - cost1, 0.0)
    seg_price_2_5 = seg_cost_2_5 / seg_vol_2_5

    seg_vol_6_10 = np.maximum(depth10 - depth5, 1e-6)
    seg_cost_6_10 = np.maximum(cost10 - cost5, 0.0)
    seg_price_6_10 = seg_cost_6_10 / seg_vol_6_10

    valid = (q_clamped > 0) & (depth_total > 0)

    # Case 0: q <= depth1 -> all at best price
    mask0 = valid & (q_clamped <= depth1)
    out[mask0] = p0[mask0]

    # Case 1: depth1 < q <= depth5
    mask1 = valid & (q_clamped > depth1) & (q_clamped <= depth5)
    if np.any(mask1):
        q1 = q_clamped[mask1]
        d1 = depth1[mask1]
        p_best = p0[mask1]
        p_2_5 = seg_price_2_5[mask1]
        cost = d1 * p_best + (q1 - d1) * p_2_5
        out[mask1] = cost / q1

    # Case 2: depth5 < q <= depth10
    mask2 = valid & (q_clamped > depth5)
    if np.any(mask2):
        q2 = q_clamped[mask2]
        d1 = depth1[mask2]
        d5 = depth5[mask2]
        p_best = p0[mask2]
        p_2_5 = seg_price_2_5[mask2]
        p_6_10 = seg_price_6_10[mask2]
        cost = (
            d1 * p_best
            + (d5 - d1) * p_2_5
            + (q2 - d5) * p_6_10
        )
        out[mask2] = cost / q2

    return pd.Series(out, index=df.index)

# =================== TARGETS ===================
def make_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    v2 targets:
      - Execution-aware regression labels:
          y_long_best, y_long_drawdown, y_short_best, y_short_drawup
        based on instabuy/instasell VWAP at a volume-aware label size.
      - Legacy mid-based targets kept for diagnostics: target_* and target_q_*.
    """

    # ---- 0) Label quantity per row (volume-aware) ----
    base_q = compute_label_base_qty_for_item(df)
    if base_q <= 0.0:
        # No usable depth for this item; leave legacy targets only.
        df["label_qty"] = np.nan
    else:
        depth_buy10 = df.get("depth_buy_10", pd.Series(0.0, index=df.index)).to_numpy(dtype=np.float32)
        depth_sell10 = df.get("depth_sell_10", pd.Series(0.0, index=df.index)).to_numpy(dtype=np.float32)
        depth_min = np.minimum(depth_buy10, depth_sell10)
        # Per-row label qty: base_q, but cannot exceed LABEL_Q_MAX_DEPTH_FRAC * depth_min
        max_per_row = depth_min * LABEL_Q_MAX_DEPTH_FRAC
        q_row = np.minimum(base_q, max_per_row)
        q_row = np.where(max_per_row > LABEL_Q_MIN_ITEMS, q_row, 0.0)
        df["label_qty"] = q_row.astype(np.float32)

    # ---- 1) Execution prices at label_qty (instabuy/instasell) ----
    q_vec = df["label_qty"].fillna(0.0).to_numpy(dtype=np.float32)
    px_insta_long = compute_exec_vwap_for_qty(df, side="ask", qty=q_vec)   # pay asks -> long entry
    px_insta_short = compute_exec_vwap_for_qty(df, side="bid", qty=q_vec)  # hit bids -> short entry

    df["px_entry_long"] = px_insta_long.astype(np.float32)
    df["px_entry_short"] = px_insta_short.astype(np.float32)

    # Execution spread at label size
    df["exec_spread_rel_labelQ"] = np.where(
        df["mid_price"] > 0,
        (df["px_entry_long"] - df["px_entry_short"]) / df["mid_price"],
        np.nan
    ).astype(np.float32)

    # Liquidity pressure at 10-level depth
    for k in (5, 10):
        db = df.get(f"depth_buy_{k}", pd.Series(0.0, index=df.index)).to_numpy(dtype=np.float32)
        ds = df.get(f"depth_sell_{k}", pd.Series(0.0, index=df.index)).to_numpy(dtype=np.float32)
        q = df["label_qty"].fillna(0.0).to_numpy(dtype=np.float32)
        df[f"liq_pressure_buy_{k}"] = np.where(db > 0, q / (db + 1e-8), 0.0).astype(np.float32)
        df[f"liq_pressure_sell_{k}"] = np.where(ds > 0, q / (ds + 1e-8), 0.0).astype(np.float32)

    # ---- 2) Execution-aware forward PnL paths (long/short) ----
    # Long uses future instasell (bid side) as exit
    H_up = int(globals().get("LOOKAHEAD_MAX_UP", LOOKAHEAD_UP))
    H_dn = int(globals().get("LOOKAHEAD_MAX_DN", LOOKAHEAD_DN))

    fmin_sell_up, fmax_sell_up = future_window_extrema(df["px_entry_short"], H_up)
    fmin_buy_dn,  fmax_buy_dn  = future_window_extrema(df["px_entry_long"], H_dn)

    entry_long = df["px_entry_long"].to_numpy(dtype=np.float32)
    entry_short = df["px_entry_short"].to_numpy(dtype=np.float32)

    fmin_sell_up = fmin_sell_up.to_numpy(dtype=np.float32)
    fmax_sell_up = fmax_sell_up.to_numpy(dtype=np.float32)
    fmin_buy_dn = fmin_buy_dn.to_numpy(dtype=np.float32)
    fmax_buy_dn = fmax_buy_dn.to_numpy(dtype=np.float32)

    # Long best: max future instasell vs entry_long
    # Long drawdown: min future instasell vs entry_long
    y_long_best = np.full(len(df), np.nan, dtype=np.float32)
    y_long_ddown = np.full(len(df), np.nan, dtype=np.float32)

    valid_long = (entry_long > 0) & (fmax_sell_up > 0)
    y_long_best[valid_long] = (fmax_sell_up[valid_long] / entry_long[valid_long]) - 1.0

    valid_long_dd = (entry_long > 0) & (fmin_sell_up > 0)
    y_long_ddown[valid_long_dd] = (fmin_sell_up[valid_long_dd] / entry_long[valid_long_dd]) - 1.0

    # Short best: sell now, buy later at best future instabuy (min)
    # Short drawup: worst adverse move (max future instabuy)
    y_short_best = np.full(len(df), np.nan, dtype=np.float32)
    y_short_drawup = np.full(len(df), np.nan, dtype=np.float32)

    valid_short = (entry_short > 0) & (fmin_buy_dn > 0)
    y_short_best[valid_short] = (entry_short[valid_short] / fmin_buy_dn[valid_short]) - 1.0

    valid_short_du = (entry_short > 0) & (fmax_buy_dn > 0)
    y_short_drawup[valid_short_du] = (entry_short[valid_short_du] / fmax_buy_dn[valid_short_du]) - 1.0

    df["y_long_best"] = y_long_best
    df["y_long_drawdown"] = y_long_ddown
    df["y_short_best"] = y_short_best
    df["y_short_drawup"] = y_short_drawup

    # ==== 3) Legacy mid-based targets (kept for diagnostics / backwards-compat) ====
    # Extremes over mid_price
    fmin_min, _ = future_window_extrema(df["mid_price"], LOOKAHEAD_MIN)
    _, fmax_max = future_window_extrema(df["mid_price"], LOOKAHEAD_MAX)

    cur = df["mid_price"].astype(np.float32)
    df["target_min_abs"] = (fmin_min - cur).astype(np.float32)
    df["target_max_abs"] = (fmax_max - cur).astype(np.float32)
    df["target_min_rel"] = np.where(cur > 0, (fmin_min - cur) / cur, np.nan).astype(np.float32)
    df["target_max_rel"] = np.where(cur > 0, (fmax_max - cur) / cur, np.nan).astype(np.float32)

    # Robust "best-of" window quantiles on mid (old v1 logic)
    H_up_mid  = int(globals().get("LOOKAHEAD_MAX_UP", LOOKAHEAD_MAX))
    H_dn_mid  = int(globals().get("LOOKAHEAD_MAX_DN", LOOKAHEAD_MIN))
    win_up = int(globals().get("ROBUST_FWD_WINDOW_UP", ROBUST_FWD_WINDOW))
    win_dn = int(globals().get("ROBUST_FWD_WINDOW_DN", ROBUST_FWD_WINDOW))
    q_up   = float(globals().get("ROBUST_Q_UP_DIR", ROBUST_Q_UP))
    q_dn   = float(globals().get("ROBUST_Q_DN_DIR", ROBUST_Q_DN))

    best_up_mid = future_best_quantile_over_windows(
        df["mid_price"], lookahead=H_up_mid, win=win_up, q=q_up, pick="max"
    )
    best_dn_mid = future_best_quantile_over_windows(
        df["mid_price"], lookahead=H_dn_mid, win=win_dn, q=q_dn, pick="min"
    )

    df["target_q_up_abs"] = (best_up_mid - cur).astype(np.float32)
    df["target_q_dn_abs"] = (cur - best_dn_mid).astype(np.float32)
    df["target_q_up_rel"] = np.where(cur > 0, (best_up_mid - cur) / cur, np.nan).astype(np.float32)
    df["target_q_dn_rel"] = np.where(cur > 0, (cur - best_dn_mid) / cur, np.nan).astype(np.float32)

    return df

# =================== SCALER / COLUMNS ===================
def select_feature_columns(df: pd.DataFrame):
    """
    Passthrough: identifiers + mid + tradable + all target/label columns.
    Everything else numeric gets scaled.
    """
    passthrough = [
        "item", "timestamp",
        "mid_price",
        "tradable",
        # execution-aware labels & primitives
        "label_qty",
        "px_entry_long", "px_entry_short",
        "y_long_best", "y_long_drawdown",
        "y_short_best", "y_short_drawup",
        "exec_spread_rel_labelQ",
        # legacy mid-based targets
        "target_min_abs", "target_max_abs",
        "target_min_rel", "target_max_rel",
        "target_q_up_abs", "target_q_dn_abs",
        "target_q_up_rel", "target_q_dn_rel",
    ]
    numerics = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    to_scale = [c for c in numerics if c not in set(passthrough)]
    return to_scale, passthrough

def reservoir_add(buf: pd.DataFrame | None,
                  block: pd.DataFrame,
                  max_rows: int,
                  rng: np.random.Generator) -> pd.DataFrame:
    if buf is None or buf.empty:
        take = min(len(block), max_rows)
        return block.sample(n=take, random_state=int(rng.integers(0, 2**31-1)))
    total = len(buf) + len(block)
    if total <= max_rows:
        return pd.concat([buf, block], axis=0, ignore_index=True)
    keep_new = max(1, min(len(block), int(max_rows * (len(block)/total))))
    new_take = block.sample(n=keep_new, random_state=int(rng.integers(0, 2**31-1)))
    replace_idx = rng.choice(len(buf), size=len(new_take), replace=False)
    out = buf.copy()
    out.iloc[replace_idx] = new_take.values
    return out

def fit_scaler(sample_block: pd.DataFrame, kind: str):
    if kind == "standard":
        scaler = StandardScaler(with_mean=True, with_std=True)
    else:
        scaler = RobustScaler(with_centering=True, with_scaling=True,
                              quantile_range=ROBUST_QUANTILE_RANGE)
    scaler.fit(sample_block.values)
    return scaler

# =================== PASS-1 / PASS-2 WORKERS ===================
from concurrent.futures import ProcessPoolExecutor, as_completed

def _pass1_build_item(item: str,
                      feature_schema_cols: List[str],
                      trim_minutes: int,
                      drop_nontradable_rows: bool) -> tuple[str, int]:
    """
    Worker: fetch -> features -> targets -> row-gate -> trim -> winsorize -> write parquet.
    """
    con = new_sql_conn()
    try:
        df = fetch_item(con, item)
        if df.empty:
            return (item, 0)

        df = compute_orderbook_features(df)
        df = compute_core_features(df)
        df = make_targets(df)

        # Optional row-level liquidity gate (if defined elsewhere)
        if 'apply_row_liquidity_gate' in globals():
            df, trad_ratio = apply_row_liquidity_gate(df)
            if ('POSTHOC_TRADABLE_MIN' in globals() and trad_ratio < POSTHOC_TRADABLE_MIN) or \
               ('POSTHOC_MIN_ITEM_ROWS' in globals() and len(df) < POSTHOC_MIN_ITEM_ROWS):
                return (item, 0)

        # Trim tail (avoid unlabeled horizon)
        if trim_minutes and trim_minutes > 0 and len(df):
            df = df.sort_values(['item', 'timestamp']).reset_index(drop=True)
            grp_size = df.groupby('item')['timestamp'].transform('size')
            rank_in_item = df.groupby('item').cumcount()
            df = df[rank_in_item < (grp_size - trim_minutes)].copy()

        # Winsorize numeric (keep mid_price untouched)
        if len(df):
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'mid_price' in numeric_cols:
                numeric_cols.remove('mid_price')
            if numeric_cols:
                q_lo = df[numeric_cols].quantile(0.0005, numeric_only=True)
                q_hi = df[numeric_cols].quantile(0.9995, numeric_only=True)
                df[numeric_cols] = df[numeric_cols].clip(lower=q_lo, upper=q_hi, axis=1)

        # Ensure schema alignment
        if feature_schema_cols:
            for c in feature_schema_cols:
                if c not in df.columns:
                    # default numeric 0.0, others NaN
                    df[c] = 0.0 if pd.api.types.is_numeric_dtype(float) else np.nan
            df = df[feature_schema_cols]

        out_path = TMP_DIR / f"{item}.parquet"
        if len(df):
            df.to_parquet(out_path, index=False)
        return (item, int(len(df)))
    finally:
        con.close()

def _pass2_scale_item(args) -> tuple[str, int]:
    """
    Worker: read per-item parquet -> scale features -> write scaled shard.
    """
    item, scaler_path, feature_schema_cols, to_scale, passthrough = args
    tmp_path = TMP_DIR / f"{item}.parquet"
    if not tmp_path.exists():
        return (item, 0)

    df = pd.read_parquet(tmp_path)
    df = df[feature_schema_cols]
    if df.empty:
        return (item, 0)

    scaler = joblib.load(str(scaler_path))
    X = df[to_scale].astype(np.float32).values
    X_scaled = scaler.transform(X)
    df_scaled = pd.DataFrame(X_scaled, columns=to_scale, index=df.index)

    out = pd.concat([df[passthrough].reset_index(drop=True),
                     df_scaled.reset_index(drop=True)], axis=1)

    shard_dir = TMP_DIR / "s2_parquet"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"{item}.parquet"
    out.to_parquet(shard_path, index=False, compression="zstd")
    return (item, int(len(out)))

# =================== MAIN ORCHESTRATION ===================
def run_full_pipeline(raw_db: Path | str = None,
                      output_path: Path | str = None) -> None:
    """
    Orchestrates PASS-1, scaler fit, PASS-2, and final concatenation.
    """
    global DB_PATH, OUTPUT_DATASET, OUTPUT_PARQUET
    if raw_db is not None:
        DB_PATH = Path(raw_db)
    if output_path is not None:
        OUTPUT_DATASET = Path(output_path)
        OUTPUT_PARQUET = Path(output_path)

def main():
    np.random.seed(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)
    ensure_dirs()

    # ----- Items -----
    con0 = new_sql_conn()
    items = top_items(con0)
    con0.close()
    print(f"[prep] Candidate items: {len(items)}")

    try:
        ARTIFACTS["items"].write_text("\n".join(items))
        print(f"[prep] Wrote allowed items → {ARTIFACTS['items']}")
    except Exception as e:
        print(f"[warn] Could not write items list: {e}")

    if not items:
        raise RuntimeError("No items to process.")

    # ===== PASS-1 pilot to establish schema =====
    pilot_item = items[0]
    con1 = new_sql_conn()
    df0 = fetch_item(con1, pilot_item)
    con1.close()
    if df0.empty:
        raise RuntimeError(f"Pilot item {pilot_item} has no data.")

    df0 = compute_orderbook_features(df0)
    df0 = compute_core_features(df0)
    df0 = make_targets(df0)

    if 'apply_row_liquidity_gate' in globals():
        df0, trad_ratio0 = apply_row_liquidity_gate(df0)
        if ('POSTHOC_TRADABLE_MIN' in globals() and trad_ratio0 < POSTHOC_TRADABLE_MIN) or \
           ('POSTHOC_MIN_ITEM_ROWS' in globals() and len(df0) < POSTHOC_MIN_ITEM_ROWS):
            raise RuntimeError("Pilot item failed tradability—adjust gates or pick a different pilot.")

    if TRIM_MINUTES and TRIM_MINUTES > 0:
        df0 = df0.sort_values(['item', 'timestamp']).reset_index(drop=True)
        grp_size = df0.groupby('item')['timestamp'].transform('size')
        rank_in_item = df0.groupby('item').cumcount()
        df0 = df0[rank_in_item < (grp_size - TRIM_MINUTES)].copy()

    # Meta & schema
    to_scale, passthrough = select_feature_columns(df0)
    feature_schema_cols = passthrough + to_scale
    meta = {
        "RET_WINDOWS": RET_WINDOWS,
        "EMA_SPANS": EMA_SPANS,
        "VOL_WINDOWS": VOL_WINDOWS,
        "LOOKAHEAD_MIN": LOOKAHEAD_MIN,
        "LOOKAHEAD_MAX": LOOKAHEAD_MAX,
        "LOOKAHEAD_UP": LOOKAHEAD_UP,
        "LOOKAHEAD_DN": LOOKAHEAD_DN,
        "OB_LEVELS": OB_LEVELS,
        "OB_BUCKETS": OB_BUCKETS,
        "SCALER_KIND": SCALER_KIND,
        "ROBUST_QUANTILE_RANGE": ROBUST_QUANTILE_RANGE,
        "VOLUME_THRESHOLD": VOLUME_THRESHOLD,
        "MID_PRICE_THRESHOLD": MID_PRICE_THRESHOLD,
        "TOP_LIQUIDITY_ITEMS": TOP_LIQUIDITY_ITEMS,
        "TRIM_MINUTES": TRIM_MINUTES,
        "ROBUST_FWD_WINDOW": ROBUST_FWD_WINDOW,
        "ROBUST_Q_UP": ROBUST_Q_UP,
        "ROBUST_Q_DN": ROBUST_Q_DN,
        "LABEL_Q_DEPTH_PERCENTILE": LABEL_Q_DEPTH_PERCENTILE,
        "LABEL_Q_FRACTION_DEPTH": LABEL_Q_FRACTION_DEPTH,
        "LABEL_Q_MIN_ITEMS": LABEL_Q_MIN_ITEMS,
        "LABEL_Q_MAX_ITEMS": LABEL_Q_MAX_ITEMS,
        "LABEL_Q_MAX_DEPTH_FRAC": LABEL_Q_MAX_DEPTH_FRAC,
        "JUMP_RET_K": JUMP_RET_K,
        "JUMP_REL_SPREAD_THRESH": JUMP_REL_SPREAD_THRESH,
    }
    ARTIFACTS["meta"].write_text(json.dumps(meta, indent=2))

    # Write pilot parquet
    (TMP_DIR / f"{pilot_item}.parquet").write_bytes(b"")
    df0[feature_schema_cols].to_parquet(TMP_DIR / f"{pilot_item}.parquet", index=False)

    accepted_items = [pilot_item]

    # ===== PASS-1 parallel for remaining items =====
    rest = items[1:]
    wrote = 1
    if rest:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
            futs = [
                exe.submit(_pass1_build_item, it, feature_schema_cols, TRIM_MINUTES, DROP_NONTRADABLE_ROWS)
                for it in rest
            ]
            for f in as_completed(futs):
                it, nrows = f.result()
                if nrows > 0:
                    accepted_items.append(it)
                wrote += 1
                if wrote % 10 == 0 or wrote == len(items):
                    print(f"[pass1] {wrote}/{len(items)} items (accepted so far: {len(accepted_items)})")

    try:
        ARTIFACTS["items_final"].write_text("\n".join(accepted_items))
        print(f"[prep] Wrote final accepted items → {ARTIFACTS['items_final']} ({len(accepted_items)})")
    except Exception as e:
        print(f"[warn] Could not write final accepted items: {e}")

    if not accepted_items:
        raise RuntimeError("No items survived PASS-1.")

    # ===== Scaler fit =====
    sample_buf = None
    for idx, it in enumerate(accepted_items, 1):
        p = TMP_DIR / f"{it}.parquet"
        if not p.exists():
            continue
        try:
            d = pd.read_parquet(p, columns=to_scale)
            if d.empty:
                continue
            take = min(len(d), max(500, MAX_SCALER_SAMPLES // max(1, len(accepted_items)//2)))
            block = d.sample(n=take, random_state=int(rng.integers(0, 2**31-1)))
            sample_buf = reservoir_add(sample_buf, block, MAX_SCALER_SAMPLES, rng)
        except Exception as e:
            print(f"[warn] sampling {it}: {e}")
        if idx % 25 == 0 or idx == len(accepted_items):
            print(f"[scaler] sampling {idx}/{len(accepted_items)} shards; buf={0 if sample_buf is None else len(sample_buf)}")

    if sample_buf is None or sample_buf.empty:
        raise RuntimeError("No data accumulated to fit scaler; check PASS-1 outputs.")
    scaler = fit_scaler(sample_buf, SCALER_KIND)
    joblib.dump(scaler, str(ARTIFACTS["scaler"]))
    del sample_buf; gc.collect()
    print(f"[scaler] Fitted {SCALER_KIND} scaler.")

    # ===== PASS-2 scaling =====
    args_list = [(it, ARTIFACTS["scaler"], feature_schema_cols, to_scale, passthrough)
                 for it in accepted_items]
    scaled_count = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futs = [exe.submit(_pass2_scale_item, args) for args in args_list]
        for f in as_completed(futs):
            it, n = f.result()
            scaled_count += 1
            if scaled_count % 20 == 0 or scaled_count == len(args_list):
                print(f"[pass2] {scaled_count}/{len(args_list)} scaled")

    # ===== Final concat (DuckDB) =====
    shard_dir = TMP_DIR / "s2_parquet"
    shards = sorted(shard_dir.glob("*.parquet"))
    if not shards:
        raise RuntimeError("No PASS-2 shards to write.")

    output_path = OUTPUT_DATASET if OUTPUT_DATASET.suffix == ".parquet" else OUTPUT_DATASET.with_suffix(".parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    cols_sql = ", ".join(_escape_identifier(c) for c in feature_schema_cols) if feature_schema_cols else "*"
    shard_list_sql = ", ".join(
        "'" + p.as_posix().replace("'", "''") + "'" for p in shards
    )

    con_final = duckdb.connect(database=":memory:")
    try:
        con_final.execute(
            f"""
            COPY (
              SELECT {cols_sql}
              FROM read_parquet([{shard_list_sql}])
            ) TO '{output_path.as_posix()}'
            (FORMAT 'parquet', COMPRESSION 'zstd');
            """
        )
    finally:
        con_final.close()

    print(f"[finalize] Wrote {len(shards)} shards -> {output_path.name}")
    print("[done] Parquet + scaler + meta ready.")

if __name__ == "__main__":
    main()
