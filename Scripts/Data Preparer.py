# Data Preparer.py — election-free, streaming, optimized, anti-phantom-liquidity
import os, gc, json, math, sqlite3, sys
from pathlib import Path
from typing import List, Tuple
import joblib, numpy as np, pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

# --- project config import (unchanged)
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as config

# =================== CONFIG ===================
DATA_DIR        = Path(str(config.PROCESSED_DIR))
DB_PATH         = Path(str(config.DB_PATH))
OUTPUT_CSV      = Path(str(config.PROCESSED_CSV))

VOLUME_THRESHOLD      = 500
MID_PRICE_THRESHOLD   = 5_000
TOP_LIQUIDITY_ITEMS   = 800   # keep as you logged

RET_WINDOWS     = [60, 240, 720, 1440, 2880]
EMA_SPANS       = [5, 15, 60, 240, 720, 1440, 2880, 5760]
VOL_WINDOWS     = [15, 60, 240, 720, 1440, 2880, 5760]

# --- Parallelism knobs (NEW) ---
MAX_WORKERS = max(1, (os.cpu_count() or 2) - 1)   # tune if disk gets saturated


OB_LEVELS       = 10
OB_BUCKETS      = (1, 5, 10)

LOOKAHEAD_MIN   = 2880
LOOKAHEAD_MAX   = 1440

# Robust forward-window label config
ROBUST_FWD_WINDOW = 120        # minutes forward for robust stat (e.g., median over 30)
ROBUST_Q_UP = 0.50            # 0.50 = median for UP side (e.g., try 0.80 for quantile)
ROBUST_Q_DN = 0.50            # 0.50 = median for DOWN side (e.g., try 0.20 for quantile)

# --- per-direction robust lookahead/window (new) ---
LOOKAHEAD_MAX_UP   = 2880   # 1 day
LOOKAHEAD_MAX_DN   = 720    # 12 hours
ROBUST_FWD_WINDOW_UP = ROBUST_FWD_WINDOW     # reuse 30 by default
ROBUST_FWD_WINDOW_DN = ROBUST_FWD_WINDOW     # reuse 30 by default
ROBUST_Q_UP_DIR    = ROBUST_Q_UP             # reuse 0.50 by default
ROBUST_Q_DN_DIR    = ROBUST_Q_DN             # reuse 0.50 by default



# Trim last N minutes per item (avoid unlabeled rows)
TRIM_MINUTES    = 0  # set 0 to disable

SCALER_KIND             = "robust"   # "robust" or "standard"
ROBUST_QUANTILE_RANGE   = (10, 90)
MAX_SCALER_SAMPLES      = 2_000_000
RANDOM_SEED             = 42

TMP_DIR = DATA_DIR / "tmp"
ARTIFACTS = {
    "scaler": DATA_DIR / "feature_scaler.pkl",
    "meta":   DATA_DIR / "preparer_meta.json",
    "items":  DATA_DIR / "allowed_items.txt",     # NEW: write the admitted items
}

# =================== UTILS ===================
def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

def new_sql_conn() -> sqlite3.Connection:
    """
    Create a reader-friendly SQLite connection for use in workers.
    Each process must have its own connection.
    """
    con = sqlite3.connect(DB_PATH, timeout=60, check_same_thread=False)
    # Reader-optimized pragmas (WAL allows multiple readers; we don't write)
    try:
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        con.execute("PRAGMA temp_store=MEMORY;")
        con.execute("PRAGMA mmap_size=134217728;")  # 128MB if available
    except Exception:
        pass
    return con


def sql(conn: sqlite3.Connection, q: str, params=()) -> pd.DataFrame:
    return pd.read_sql_query(q, conn, params=params)

def top_items(conn: sqlite3.Connection) -> List[str]:
    """
    Pick items that are reliably liquid by requiring a share of bars to pass:
      - mid_price >= MID_PRICE_THRESHOLD
      - sell_moving_week >= VOLUME_THRESHOLD
      - both sides present and sane spread
    Items are then ranked by average (sell_moving_week * mid_price).
    """
    MAX_REL_SPREAD = 0.20          # per-bar spread cap (20%)
    MIN_TRADABLE_RATIO = 0.90      # >=60% bars have quotes+size+spread ok
    MIN_MID_OK_RATIO = 0.90        # >=60% bars have mid >= MID_PRICE_THRESHOLD
    MIN_VOL_OK_RATIO = 0.80        # >=60% bars have weekly sell volume >= VOLUME_THRESHOLD

    q = f"""
    WITH base AS (
      SELECT
        item,
        (buy_price + sell_price)/2.0 AS mid_price,
        sell_moving_week,
        buy_orders, sell_orders,
        CASE
          WHEN buy_orders > 0 AND sell_orders > 0
               AND ( (sell_price - buy_price) / NULLIF((buy_price + sell_price)/2.0,0) ) <= {MAX_REL_SPREAD}
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

def future_window_quantile(series: pd.Series, horizon: int, q: float) -> pd.Series:
    """
    For each index t, compute the quantile(q) of series[t+1 : t+1+horizon].
    Returns a Series aligned to 'series' with NaN for the last 'horizon' rows.
    """
    x = series.to_numpy(dtype=np.float32, copy=False)
    n = len(x)
    if n == 0 or horizon <= 0:
        return pd.Series(np.full(n, np.nan, dtype=np.float32), index=series.index)
    m = n - horizon   # number of valid windows
    if m <= 0:
        return pd.Series(np.full(n, np.nan, dtype=np.float32), index=series.index)

    from numpy.lib.stride_tricks import sliding_window_view
    # windows over x[1:] to ensure strict "future-only"
    win = sliding_window_view(x[1:], window_shape=horizon)  # shape: (n-1 - (h-1)) = (n-h)
    qvals = np.nanquantile(win, q, axis=1).astype(np.float32)  # length m = n - horizon
    out = np.concatenate([qvals, np.full(horizon, np.nan, dtype=np.float32)])
    return pd.Series(out, index=series.index)



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

# =================== FEATURES ===================
def compute_orderbook_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orderbook features:
      - Keep only L0 (best) raw levels; build bucket aggregates for k in OB_BUCKETS
      - Derive depth, VWAP bid/ask/mid, OB imbalance, and slopes at larger buckets
      - Add depth-weighted spread per bucket
      - PRUNE low-signal raw L2..L9 prices/amounts and k=1 slopes
      - NEW: add per-row 'tradable' flag (no phantom liquidity)
    """
    # L0 best prices/amounts
    bb = df.get("buy_price_0").astype(np.float32)
    ba = df.get("sell_price_0").astype(np.float32)
    bv0 = df.get("buy_amount_0").astype(np.float32)
    sv0 = df.get("sell_amount_0").astype(np.float32)

    # Mid estimate available here (exact mid added in core later)
    mid0 = ((bb + ba) * 0.5).astype(np.float32)

    # Base L1 microprice using L0 amount weights
    base = {
        "best_bid": bb,
        "best_ask": ba,
        "ba_spread": (ba - bb).astype(np.float32),
        "rel_spread": np.where((bb + ba) > 0, (ba - bb) / ((bb + ba) * 0.5), 0.0).astype(np.float32),
        "microprice_l1": ((ba * bv0 + bb * sv0) / (bv0 + sv0)).astype(np.float32),
    }
    base_df = pd.DataFrame(base, index=df.index)
    base_df["microprice_l1"] = base_df["microprice_l1"].where(np.isfinite(base_df["microprice_l1"]), mid0)
    df = pd.concat([df, base_df], axis=1, copy=False)

    # NEW: Per-row tradability (both quotes & some size; spread sane)
    df["tradable"] = (
        (df["best_bid"] > 0) & (df["best_ask"] > 0) &
        (df.get("buy_amount_0", 0) > 0) & (df.get("sell_amount_0", 0) > 0) &
        (df["rel_spread"] <= 0.05)
    ).astype(np.int8)

    # Bucket aggregates (k in OB_BUCKETS)
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
            buy_amt += ab; sell_amt += aa
            vwap_bid_num += (pb * ab); vwap_ask_num += (pa * aa)

        denom_b = np.where(buy_amt > 0, buy_amt, np.nan)
        denom_s = np.where(sell_amt > 0, sell_amt, np.nan)
        vwap_bid = (vwap_bid_num / denom_b)
        vwap_ask = (vwap_ask_num / denom_s)
        vwap_mid = (vwap_bid.fillna(bb) + vwap_ask.fillna(ba)) * 0.5

        # depth-weighted spread normalized by mid
        dws = (vwap_ask.fillna(ba) - vwap_bid.fillna(bb)) / np.where(mid0 > 0, mid0, np.nan)

        new_cols = {
            f"depth_buy_{k}": buy_amt,
            f"depth_sell_{k}": sell_amt,
            f"vwap_bid_{k}": vwap_bid.astype(np.float32),
            f"vwap_ask_{k}": vwap_ask.astype(np.float32),
            f"vwap_mid_{k}": vwap_mid.astype(np.float32),
            f"ob_imb_{k}": np.where((buy_amt + sell_amt) > 0, (buy_amt - sell_amt) / (buy_amt + sell_amt), 0.0).astype(np.float32),
            f"bid_slope_{k}": (bb - df.get(f"buy_price_{k-1}", bb).astype(np.float32)) / max(k - 1, 1),
            f"ask_slope_{k}": (df.get(f"sell_price_{k-1}", ba).astype(np.float32) - ba) / max(k - 1, 1),
            f"dws_{k}": dws.astype(np.float32),  # depth-weighted spread / mid
        }
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1, copy=False)

    # PRUNE low-signal raw OB columns after aggregates:
    drop_cols = []
    for i in range(1, OB_LEVELS):  # prices: drop 1..9
        drop_cols += [f"buy_price_{i}", f"sell_price_{i}"]
    for i in range(2, OB_LEVELS):  # amounts: drop 2..9
        drop_cols += [f"buy_amount_{i}", f"sell_amount_{i}"]
    drop_cols += ["bid_slope_1", "ask_slope_1"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors="ignore")

    return df


def compute_core_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Core market features:
      - mid, spreads, normalized order/flow ratios
      - log returns: always compute ret_1; add longer-horizon returns per RET_WINDOWS
      - realized volatility over VOL_WINDOWS (rolling std of ret_1)
      - volatility regime ratios (short/long)
      - batched EMAs for mid, imbalance, rel spread, ret_1 (keep long spans that mattered)
    """
    # Mid & spreads
    df["mid_price"] = ((df["buy_price"] + df["sell_price"]) * 0.5).astype(np.float32)
    df["quoted_spread"] = (df["sell_price"] - df["buy_price"]).astype(np.float32)
    df["rel_qspread"]   = np.where(df["mid_price"] > 0, df["quoted_spread"] / df["mid_price"], 0.0).astype(np.float32)

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
    df["ret_1"] = df["log_mid"].diff(1).astype(np.float32)  # ALWAYS compute, used for vol

    # Longer-horizon returns
    for w in RET_WINDOWS:
        if w == 1:
            continue
        df[f"ret_{w}"] = (df["log_mid"].diff(w)).astype(np.float32)

    # Realized volatility over ret_1
    for w in VOL_WINDOWS:
        df[f"vol_{w}"] = df["ret_1"].rolling(w, min_periods=max(2, int(w/3))).std().astype(np.float32)

    # Volatility regime ratios (short vs long)
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

    # Batched EMAs
    ema_cols = {}
    for span in EMA_SPANS:
        ema_cols[f"mid_ema_{span}"]  = df["mid_price"].ewm(span=span, adjust=False).mean().astype(np.float32)
        ema_cols[f"imb_ema_{span}"]  = df["order_imbalance"].ewm(span=span, adjust=False).mean().astype(np.float32)
        ema_cols[f"qsp_ema_{span}"]  = df["rel_qspread"].ewm(span=span, adjust=False).mean().astype(np.float32)
        ema_cols[f"ret1_ema_{span}"] = df["ret_1"].ewm(span=span, adjust=False).mean().astype(np.float32)
    df = pd.concat([df, pd.DataFrame(ema_cols, index=df.index)], axis=1, copy=False)

    return df


# =================== TARGETS ===================
def future_window_extrema(series: pd.Series, horizon: int) -> Tuple[pd.Series, pd.Series]:
    x = series.values.astype(np.float32)
    n = len(x)
    if n == 0 or horizon <= 0:
        z = np.full(n, np.nan, dtype=np.float32)
        return pd.Series(z), pd.Series(z)
    from collections import deque
    min_deq, max_deq = deque(), deque()
    fmin = np.full(n, np.nan, dtype=np.float32)
    fmax = np.full(n, np.nan, dtype=np.float32)

    j_start, j_end = 1, min(horizon, n - 1)
    for j in range(j_start, j_end + 1):
        while min_deq and x[min_deq[-1]] >= x[j]: min_deq.pop()
        min_deq.append(j)
        while max_deq and x[max_deq[-1]] <= x[j]: max_deq.pop()
        max_deq.append(j)

    for i in range(n):
        fmin[i] = x[min_deq[0]] if min_deq else np.nan
        fmax[i] = x[max_deq[0]] if max_deq else np.nan
        leave_idx = i + 1
        enter_idx = i + 1 + horizon
        if min_deq and min_deq[0] == leave_idx: min_deq.popleft()
        if max_deq and max_deq[0] == leave_idx: max_deq.popleft()
        if enter_idx < n:
            while min_deq and x[min_deq[-1]] >= x[enter_idx]: min_deq.pop()
            min_deq.append(enter_idx)
            while max_deq and x[max_deq[-1]] <= x[enter_idx]: max_deq.pop()
            max_deq.append(enter_idx)
    return pd.Series(fmin, index=series.index), pd.Series(fmax, index=series.index)

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from collections import deque

def moving_extreme(arr: np.ndarray, window: int, kind: str) -> np.ndarray:
    """
    O(n) deque moving max/min over 1D array (no NaNs assumed).
    Returns array of length len(arr)-window+1.
    """
    if window <= 0 or window > len(arr):
        return np.full(0, np.nan, dtype=np.float32)
    dq = deque()
    out = np.empty(len(arr) - window + 1, dtype=np.float32)
    comp = (lambda a, b: a >= b) if kind == "max" else (lambda a, b: a <= b)
    for i, x in enumerate(arr):
        while dq and comp(x, arr[dq[-1]]):
            dq.pop()
        dq.append(i)
        if dq[0] <= i - window:
            dq.popleft()
        if i >= window - 1:
            out[i - window + 1] = arr[dq[0]]
    return out

def future_best_quantile_over_windows(series: pd.Series,
                                      lookahead: int,
                                      win: int,
                                      q: float,
                                      pick: str) -> pd.Series:
    """
    For each t, consider windows of length `win` starting at t+1 .. t+lookahead-win+1.
    Compute the per-window quantile (q in [0,1]) of price inside each window.
    Return BEST over those window-quantiles:
      pick='max' -> highest q-quantile; pick='min' -> lowest q-quantile.
    Output aligned to `series` with NaN for last `lookahead` rows.
    Leak-safe (strictly 'future-only').
    """
    x = series.to_numpy(dtype=np.float32, copy=False)
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float32)
    if n == 0 or lookahead <= 0 or win <= 0 or q < 0 or q > 1:
        return pd.Series(out, index=series.index)

    # All rolling windows of length 'win' across the full series
    # windows[i] == x[i : i+win]
    if n < win:
        return pd.Series(out, index=series.index)
    windows = sliding_window_view(x, win)  # shape: (n - win + 1, win)

    # Quantile within each length-'win' window
    # Note: nan-robust; if you expect NaNs, use np.nanquantile
    per_win_q = np.quantile(windows, q, axis=1).astype(np.float32)  # length m = n - win + 1
    m = len(per_win_q)

    # We need best over ranges of length L = lookahead - win + 1, starting at i0 = t+1
    L = lookahead - win + 1
    if L <= 0:
        # No room to place a 'win'-sized window inside lookahead
        return pd.Series(out, index=series.index)
    if m < L:
        return pd.Series(out, index=series.index)

    # Moving extreme over per_win_q with window length L
    kind = "max" if pick == "max" else "min"
    if kind == "max":
        best = moving_extreme(per_win_q, L, kind="max")  # length m - L + 1
    else:
        # For min, invert and use max for numerical stability if desired; direct min works fine here
        best = moving_extreme(per_win_q, L, kind="min")

    # Align: for each t, the range starts at i0 = t+1 -> index into 'best' is r = t+1
    # Valid t satisfy (t+1) <= len(best)-1  ->  t <= len(best)-2
    # But also require t <= n - lookahead - 1 to stay in-bounds
    max_t = min(len(best) - 2, n - lookahead - 1)
    if max_t >= 0:
        out[0:max_t+1] = best[1:max_t+2]  # best[r] with r = t+1

    # The last 'lookahead' bars remain NaN (no full lookahead available)
    return pd.Series(out, index=series.index)


def make_targets(df: pd.DataFrame) -> pd.DataFrame:
    # --- legacy extrema (kept for backward compatibility) ---
    fmin_min, _ = future_window_extrema(df["mid_price"], LOOKAHEAD_MIN)
    _, fmax_max = future_window_extrema(df["mid_price"], LOOKAHEAD_MAX)

    cur = df["mid_price"].astype(np.float32)
    df["target_min_abs"] = (fmin_min - cur).astype(np.float32)
    df["target_max_abs"] = (fmax_max - cur).astype(np.float32)
    df["target_min_rel"] = np.where(cur > 0, (fmin_min - cur) / cur, np.nan).astype(np.float32)
    df["target_max_rel"] = np.where(cur > 0, (fmax_max - cur) / cur, np.nan).astype(np.float32)

    # --- robust "best-of" window quantiles, per-direction ---
    cur = df["mid_price"].astype(np.float32)

    H_up  = int(globals().get("LOOKAHEAD_MAX_UP", LOOKAHEAD_MAX))
    H_dn  = int(globals().get("LOOKAHEAD_MAX_DN", LOOKAHEAD_MIN))
    win_up = int(globals().get("ROBUST_FWD_WINDOW_UP", ROBUST_FWD_WINDOW))
    win_dn = int(globals().get("ROBUST_FWD_WINDOW_DN", ROBUST_FWD_WINDOW))
    q_up   = float(globals().get("ROBUST_Q_UP_DIR", ROBUST_Q_UP))
    q_dn   = float(globals().get("ROBUST_Q_DN_DIR", ROBUST_Q_DN))

    # Highest q_up window-quantile in next H_up minutes (e.g., highest 30-min median inside 2 days)
    best_up = future_best_quantile_over_windows(
        df["mid_price"], lookahead=H_up, win=win_up, q=q_up, pick="max"
    )

    # Lowest  q_dn window-quantile in next H_dn minutes (e.g., lowest 30-min median inside 12 hours)
    best_dn = future_best_quantile_over_windows(
        df["mid_price"], lookahead=H_dn, win=win_dn, q=q_dn, pick="min"
    )

    # Absolute & relative deltas
    df["target_q_up_abs"] = (best_up - cur).astype(np.float32)
    df["target_q_dn_abs"] = (cur - best_dn).astype(np.float32)
    df["target_q_up_rel"] = np.where(cur > 0, (best_up - cur) / cur, np.nan).astype(np.float32)
    df["target_q_dn_rel"] = np.where(cur > 0, (cur - best_dn) / cur, np.nan).astype(np.float32)


    return df


# =================== SCALER / COLUMNS ===================
def select_feature_columns(df: pd.DataFrame):
    # Make 'tradable' passthrough (do NOT scale the mask)
    passthrough = [
        "item","timestamp","mid_price","tradable",
        # legacy extrema targets
        "target_min_abs","target_max_abs","target_min_rel","target_max_rel",
        # robust window targets (NEW)
        "target_q_up_abs","target_q_dn_abs","target_q_up_rel","target_q_dn_rel",
    ]
    numerics = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    to_scale = [c for c in numerics if c not in set(passthrough)]
    return to_scale, passthrough


def reservoir_add(buf: pd.DataFrame | None, block: pd.DataFrame, max_rows: int, rng: np.random.Generator) -> pd.DataFrame:
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
        scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=ROBUST_QUANTILE_RANGE)
    scaler.fit(sample_block.values)
    return scaler

from concurrent.futures import ProcessPoolExecutor, as_completed

def _pass1_build_item(item: str,
                      feature_schema_cols: List[str],
                      trim_minutes: int,
                      drop_nontradable_rows: bool) -> tuple[str, int]:
    """
    Worker: fetch -> features -> targets -> row-gate -> trim -> winsorize -> write parquet.
    Returns (item, n_rows_written). Uses its own SQLite connection.
    """
    con = new_sql_conn()
    try:
        df = fetch_item(con, item)
        if df.empty:
            return (item, 0)

        df = compute_orderbook_features(df)
        df = compute_core_features(df)
        df = make_targets(df)

        # Row liquidity gate (if you added apply_row_liquidity_gate earlier)
        if 'apply_row_liquidity_gate' in globals():
            df, trad_ratio = apply_row_liquidity_gate(df)
            # Respect your posthoc knobs if defined
            if ('POSTHOC_TRADABLE_MIN' in globals() and trad_ratio < POSTHOC_TRADABLE_MIN) or \
               ('POSTHOC_MIN_ITEM_ROWS' in globals() and len(df) < POSTHOC_MIN_ITEM_ROWS):
                return (item, 0)

        # Trim to avoid unlabeled tail
        if trim_minutes and trim_minutes > 0 and len(df):
            df = df.sort_values(['item','timestamp']).reset_index(drop=True)
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

        # Ensure schema alignment (add any missing cols as 0/NaN)
        if feature_schema_cols:
            for c in feature_schema_cols:
                if c not in df.columns:
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
    Worker: read per-item parquet -> scale -> write per-item CSV shard.
    Returns (item, n_rows).
    """
    item, scaler_path, feature_schema_cols, to_scale, passthrough = args
    tmp_path = TMP_DIR / f"{item}.parquet"
    if not tmp_path.exists():
        return (item, 0)

    df = pd.read_parquet(tmp_path)
    df = df[feature_schema_cols]  # enforce order
    if df.empty:
        return (item, 0)

    scaler = joblib.load(str(scaler_path))
    X = df[to_scale].astype(np.float32).values
    X_scaled = scaler.transform(X)
    df_scaled = pd.DataFrame(X_scaled, columns=to_scale, index=df.index)

    out = pd.concat([df[passthrough].reset_index(drop=True),
                     df_scaled.reset_index(drop=True)], axis=1)

    shard_dir = TMP_DIR / "s2_csv"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"{item}.csv"
    out.to_csv(shard_path, index=False)
    return (item, int(len(out)))

# ---- Safety defaults for row-gating knobs (prevents NameError) ----
if 'DROP_NONTRADABLE_ROWS' not in globals(): DROP_NONTRADABLE_ROWS = False
if 'POSTHOC_TRADABLE_MIN'  not in globals(): POSTHOC_TRADABLE_MIN  = 0.60
if 'POSTHOC_MIN_ITEM_ROWS' not in globals(): POSTHOC_MIN_ITEM_ROWS = 5000

# =================== MAIN ===================
def main():
    np.random.seed(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)
    ensure_dirs()

    # ----- Items (still via SQL picker) -----
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

    # ===== PASS-1 (pilot sequential) =====
    # Build 1 item to establish schema + meta
    pilot_item = items[0]
    con1 = new_sql_conn()
    df0 = fetch_item(con1, pilot_item)
    con1.close()
    if df0.empty:
        raise RuntimeError(f"Pilot item {pilot_item} has no data.")

    df0 = compute_orderbook_features(df0)
    df0 = compute_core_features(df0)
    df0 = make_targets(df0)

    # Row-level gate (if present)
    if 'apply_row_liquidity_gate' in globals():
        df0, trad_ratio0 = apply_row_liquidity_gate(df0)
        if ('POSTHOC_TRADABLE_MIN' in globals() and trad_ratio0 < POSTHOC_TRADABLE_MIN) or \
           ('POSTHOC_MIN_ITEM_ROWS' in globals() and len(df0) < POSTHOC_MIN_ITEM_ROWS):
            raise RuntimeError("Pilot item failed tradability—pick a different pilot or loosen gates.")

    # Trim pilot tail
    if TRIM_MINUTES and TRIM_MINUTES > 0:
        df0 = df0.sort_values(['item','timestamp']).reset_index(drop=True)
        grp_size = df0.groupby('item')['timestamp'].transform('size')
        rank_in_item = df0.groupby('item').cumcount()
        df0 = df0[rank_in_item < (grp_size - TRIM_MINUTES)].copy()

    # Meta & schema
    to_scale, passthrough = select_feature_columns(df0)
    feature_schema_cols = passthrough + to_scale
    meta = {
        "RET_WINDOWS": RET_WINDOWS, "EMA_SPANS": EMA_SPANS, "VOL_WINDOWS": VOL_WINDOWS,
        "LOOKAHEAD_MIN": LOOKAHEAD_MIN, "LOOKAHEAD_MAX": LOOKAHEAD_MAX,
        "OB_LEVELS": OB_LEVELS, "OB_BUCKETS": OB_BUCKETS,
        "SCALER_KIND": SCALER_KIND, "ROBUST_QUANTILE_RANGE": ROBUST_QUANTILE_RANGE,
        "VOLUME_THRESHOLD": VOLUME_THRESHOLD, "MID_PRICE_THRESHOLD": MID_PRICE_THRESHOLD,
        "TOP_LIQUIDITY_ITEMS": TOP_LIQUIDITY_ITEMS, "TRIM_MINUTES": TRIM_MINUTES,
        "ROBUST_FWD_WINDOW": ROBUST_FWD_WINDOW,"ROBUST_Q_UP": ROBUST_Q_UP,"ROBUST_Q_DN": ROBUST_Q_DN,

    }
    ARTIFACTS["meta"].write_text(json.dumps(meta, indent=2))

    # Write pilot parquet
    (TMP_DIR / f"{pilot_item}.parquet").write_bytes(b"")  # ensure path exists on errors
    df0[feature_schema_cols].to_parquet(TMP_DIR / f"{pilot_item}.parquet", index=False)

    accepted_items = [pilot_item]

    # ===== PASS-1 (parallel for the rest) =====
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

    # Record final accepted list
    try:
        ARTIFACTS["items_final"].write_text("\n".join(accepted_items))
        print(f"[prep] Wrote final accepted items → {ARTIFACTS['items_final']} ({len(accepted_items)})")
    except Exception as e:
        print(f"[warn] Could not write final accepted items: {e}")

    if not accepted_items:
        raise RuntimeError("No items survived PASS-1 gates.")

    # ===== Scaler fit (single-threaded, but fast I/O) =====
    # Reservoir-sample across per-item parquets to fit scaler
    sample_buf = None
    for idx, it in enumerate(accepted_items, 1):
        p = TMP_DIR / f"{it}.parquet"
        if not p.exists():
            continue
        try:
            d = pd.read_parquet(p, columns=to_scale)
            if d.empty:
                continue
            # small sample from each shard (adaptive reservoir)
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

    # ===== PASS-2 (parallel scaling to per-item shards) =====
    args_list = [(it, ARTIFACTS["scaler"], feature_schema_cols, to_scale, passthrough) for it in accepted_items]
    scaled_count = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futs = [exe.submit(_pass2_scale_item, args) for args in args_list]
        for f in as_completed(futs):
            it, n = f.result()
            scaled_count += 1
            if scaled_count % 20 == 0 or scaled_count == len(args_list):
                print(f"[pass2] {scaled_count}/{len(args_list)} scaled")

    # ===== Final concat (single writer) =====
    shard_dir = TMP_DIR / "s2_csv"
    shards = sorted(shard_dir.glob("*.csv"))
    if not shards:
        raise RuntimeError("No PASS-2 shards to write.")
    header_written = False
    with open(OUTPUT_CSV, "w", newline="") as fout:
        for i, sp in enumerate(shards, 1):
            with open(sp, "r", newline="") as fin:
                if header_written:
                    next(fin)  # skip header
                for line in fin:
                    fout.write(line)
            if not header_written:
                header_written = True
            if i % 50 == 0 or i == len(shards):
                print(f"[finalize] {i}/{len(shards)} shards → {OUTPUT_CSV.name}")

    print("[done] CSV + scaler + meta ready.")


if __name__ == "__main__":
    main()
