
"""
usage:
```bash
python bazaar_rl_optimized.py --timesteps 1_200_000 \
       --data "optional/override/path.csv"
```
If you omit `--data`, the script uses `config.RL_DATASET_CSV`.
"""

from __future__ import annotations

# ───────────────────────── Imports ──────────────────────────
import argparse
import math
import os
import random
import sys
import json
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Any, Dict, List, Tuple


import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import datetime
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from imitation.algorithms.bc import BC as BehavioralCloning
from imitation.data.types import Transitions
from callbacks import IntervalRewardLogger

# project‑level config (paths only)
sys.path.append(str(Path(__file__).resolve().parent.parent)) # repo root\n
import config as config  # must live in PYTHONPATH or same dir

# silence TF “oneDNN” spam
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# global RNG for reproducibility
GLOBAL_RNG = np.random.default_rng(42)

# Scaling meta for raw values
SCALE_META = json.loads(Path(config.PROCESSED_DIR / "rl_scaling.json").read_text())
P_MIN = SCALE_META["mid_price"]["min"]
P_MAX = SCALE_META["mid_price"]["max"]

# ───────────────────────── Hyper‑Config ─────────────────────
class C:
    """Centralised hyper‑parameters (non‑path)."""

    # dataset & outputs (paths from external config)
    DATA_PATH: Path = config.RL_DATASET_CSV
    OUTPUT_DIR: Path = config.OUTPUTS_DIR / "rl_opt"
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    TRADING_DIR: Path = config.OUTPUTS_DIR / "trading"

    # time base
    TICKS_PER_MIN: int = 120          # 0.5 s snapshots
    MIN_EPISODE_MIN: int = 4320        # 3 d
    MAX_EPISODE_MIN: int = 5760      # 4 d

    # position constraints
    MIN_HOLD_TICKS: int = 60_000       # 9 h
    MAX_HOLD_TICKS: int = 240_000     # 36 h
    MAX_PROFIT_PCT: float = 0.5
    MAX_LOSS_PCT: float = 0.25

    # trading frictions
    FEE_BPS: float = 100            # 1 % side
    LATENCY_MEAN_S: float = 4.0
    SLIPPAGE_PCT_PER_STACK: float = 0.01

    # reward weights
    W_REALISED: float = 5
    W_UNREALISED: float = 0.5
    W_DRAWDOWN: float = 0.3
    W_TRADE_COUNT: float = 0.3
    MIN_PROFIT_BP: float = 3       # trades <2 bp treated as spam
    MAX_BP_CLIP: float = 1000         # safety cap per interval
    HOLD_BONUS_BP: float = 0   # +0.1 bp per tick per open position

    # BC warm‑start 
    N_BC_EPOCHS: int = 1

    # PPO
    REWARD_INTERVAL: int = 7.5  # minutes
    PPO_N_STEPS: int = 900
    PPO_BATCH_SIZE: int = 900
    PPO_EPOCHS: int = 4
    PPO_GAMMA: float = 0.9999995
    PPO_LAMBDA: float = 0.98
    PPO_CLIP: float = 0.075
    PPO_LR_MAX: float = 5e-4
    PPO_LR_MIN: float = 1e-4
    PPO_ENT_START: float = 0.5
    PPO_ENT_END: float = 1e-4
    TOTAL_TIMESTEPS: int = 3_000_000
    GRAD_NORM_CLIP: float = 1.0
    MAX_ENVS: int = 16


# ───────────────────── Data Utilities ───────────────────────

def _read_one(fp: Path) -> pd.DataFrame:
    if fp.suffix == ".parquet":
        return pd.read_parquet(fp)
    if fp.suffix in {".csv", ".gz", ".bz2"}:
        return pd.read_csv(fp)
    raise ValueError(f"Unsupported file type: {fp.suffix}")


def load_dataset(path_like: Path | str) -> pd.DataFrame:
    """Load either a single file or concatenation of a glob pattern."""
    path = Path(path_like)
    if path.is_file():
        dfs = [_read_one(path)]
    else:
        files = list(Path().glob(str(path)))
        if not files:
            raise FileNotFoundError(f"No files match {path}")
        dfs = [_read_one(fp) for fp in files]
    df = pd.concat(dfs, ignore_index=True)
    df["ret_60s"] = df["mid_price"].pct_change(120).fillna(0)
    float_cols = df.select_dtypes("float64").columns
    df[float_cols] = df[float_cols].astype(np.float32)
    df.sort_values("timestamp", inplace=True, ignore_index=True)
    return df


def split_dataset(df: pd.DataFrame, test_ratio: float = 0.25):
    k = int(len(df) * (1 - test_ratio))
    return df.iloc[:k].reset_index(drop=True), df.iloc[k:].reset_index(drop=True)

def softclip(x, limit=1_000):
    """
    Smoothly compress |x| beyond ±limit bp but keep a non-zero slope.
    """
    return limit * np.tanh(x / limit)

class EWMA:
    """Cheap exponential running stats (decay in ticks)."""
    def __init__(self, decay=0.99999):
        self.decay, self.mean, self.var = decay, 0.0, 1e-6
    def update(self, x):
        d = 1.0 - self.decay
        self.mean = self.decay * self.mean + d * x
        self.var  = self.decay * self.var  + d * (x - self.mean) ** 2


# ────────────────── Environment Structures ──────────────────
@dataclass
class Order:
    item: str
    qty: int
    direction: str  # "buy" or "sell"
    submit_ts: float
    exec_ts: float
    price: float  # net price per item (incl. slip)


@dataclass
class Position:
    item: str
    entry_ts: float
    qty: int
    total_cost: float     # cumulative cost basis (including fees/slippage)
    tp_pct: float
    sl_pct: float
    hold_ticks: int
    age : int = 0

    @property
    def avg_price(self) -> float:
        """Weighted average entry price per unit."""
        return self.total_cost / self.qty

    @property
    def cost(self) -> float:
        """Alias for total_cost to match your existing code."""
        return self.total_cost
    
    
    @property
    def entry_cash(self) -> float:
        """Alias so _close_pos can do proceeds – entry_cash."""
        return self.total_cost

  
class BazaarTradingEnv(gym.Env):
    """Offline RL env with latency, slippage & advanced reward."""

    metadata = {"render.modes": []}

    # action indices
    DIR_I, FRAC_I, TP_I, SL_I, HOLD_I, IDLE_I = range(6)

    def __init__(self, data: pd.DataFrame, *, initial_capital: float = 1_000_000_000, trade_fee: float = 1e-2, explore: bool = True):
        df = data
        
        if "item_code" not in df.columns:
            df = df.copy()              # only copy if we need to mutate
            df["item_code"] = (
                df["item"].astype("category").cat.codes.astype(np.float32)
            )
            df["item_code"] = 2 * (df["item_code"] - df["item_code"].min()) / (
                df["item_code"].max() - df["item_code"].min() + 1
            ) - 1
        self.df = df.reset_index(drop=True)
        self.initial_capital = float(initial_capital)
        self.explore = explore

        # feature extraction (+ item code)
        base_feats = [c for c in self.df.columns if c not in {"timestamp", "item", "datetime"}]
        self.features = base_feats
        self.df["item_code"] = (
            self.df["item"].astype("category").cat.codes.astype(np.float32)
        )
        self.df["item_code"] = 2 * (self.df["item_code"] - self.df["item_code"].min()) / (
            self.df["item_code"].max() - self.df["item_code"].min() + 1e-6
        ) - 1  # scale to −1…1

        self.X_feats = self.df[self.features].to_numpy(np.float32)
        self.X_item_raw = self.df["item"].astype("category").cat.codes.to_numpy(np.int16)
        self.items = list(self.df["item"].astype("category").cat.categories)
        self.X_price = self.df["mid_price"].to_numpy(np.float32)
        self.X_time = self.df["timestamp"].to_numpy(np.float64)

        scale = (P_MAX - P_MIN)
        self.X_price_raw = self.X_price * scale + P_MIN

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.features) + 6,), dtype=np.float32
        )

        self.epsilon, self.eps_decay, self.eps_min = (
            (0.30, 0.9999, 0.0) if explore else (0.0, 1.0, 0.0)
        )
        self.r_stats = {
            "real":  EWMA(),
            "unreal":EWMA(),
            "dd":    EWMA(),
            "trades":EWMA(),
            "comp":  EWMA(),
        }
        self.rng = GLOBAL_RNG
        self.orders: list[Order] = []
        self.trade_log: list[dict] = []  
        self.reset()

    # ───────────────────── Helpers ──────────────────────────
    def _rand_ep_len(self):
        mins = self.rng.integers(C.MIN_EPISODE_MIN, C.MAX_EPISODE_MIN + 1)
        return mins * C.TICKS_PER_MIN

    def _sample_start(self):
        # Clamp episode length to dataset size – 2 rows (start + 1 valid step)
        max_startable = max(1, len(self.df) - self.ep_len - 1)
        if max_startable <= 1:
            # dataset shorter than episode → shorten the episode on-the-fly
            self.ep_len = len(self.df) - 2
            max_startable 
        return int(self.rng.integers(0, max_startable))

    def _get_obs(self):
        feats = self.X_feats[self.idx]
        pv = self.capital
        unreal, ages = [], []
        for itm, pos in self.positions.items():
            last = self.last_price.get(itm, pos.avg_price)
            pv += pos.qty * last * (1 - self.fee)
            unreal.append((last - pos.avg_price) / pos.avg_price)
            ages.append(pos.age)
        stats = np.array([
            self.capital / self.initial_capital,
            pv / self.initial_capital,
            len(self.positions),
            float(np.mean(unreal)) if unreal else 0.0,
            float(np.mean(ages)) if ages else 0.0,
            self.drawdown / self.initial_capital,
        ], np.float32)
        return np.concatenate([feats, stats])

    # ───────────────────── Gym API ───────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.fee = C.FEE_BPS / 10_000.0
        self.capital = self.initial_capital
        self.positions: Dict[str, Position] = {}
        self.last_price: Dict[str, float] = {}
        self.hwm = self.initial_capital
        self.drawdown = 0.0


        # interval stats
        self.eval_trades = []
        self.realised_bp = 0.0
        self.unrealised_bp = 0.0
        self.trade_cnt = 0
        self.dd_interval = 0.0
        self.interval_start_val = self.initial_capital
        self.agg_logret = 0.0

        # --- state for v4 reward engine ---
        self.prev_dd            = 0.0
        self.prev_pv            = self.initial_capital     # last PV seen
        self.downside_variance  = 0.0
        self.realised_pnl       = 0.0                      # coins, not bp
        self.unrealised_pnl     = 0.0

        self.ep_len = self._rand_ep_len()
        self.idx = self._sample_start()
        self.end_idx = min(len(self.df) - 1, self.idx + self.ep_len)
        self.ticks = 0
        self.orders.clear()
        self.trade_log.clear()
        return self._get_obs(), {}

    # ---------------------------------------------------------
    def step(self, action):
        # ε‑greedy
        if self.explore and self.rng.random() < self.epsilon:
            action = self.rng.uniform(-1, 1, 6).astype(np.float32)
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

        dir_raw, frac_raw, tp_raw, sl_raw, hold_raw, idle_raw = action
        idle = idle_raw > 0.2
        direction = "buy" if dir_raw > 0.33 else "sell" if dir_raw < -0.33 else "hold"
        frac = np.clip((frac_raw + 1) / 2, 0.1, 1)
        tp_pct = ((tp_raw + 1) / 2) * C.MAX_PROFIT_PCT
        sl_pct = ((sl_raw + 1) / 2) * C.MAX_LOSS_PCT
        hold_ticks = int(
            np.interp((hold_raw + 1) / 2, [0, 1], [C.MIN_HOLD_TICKS, C.MAX_HOLD_TICKS])
        )

        item_idx = self.X_item_raw[self.idx]
        item = self.items[item_idx]
        mid = float(self.X_price_raw[self.idx])
        ts = float(self.X_time[self.idx])
        self.last_price[item] = mid

        # fill matured orders
        self._fill_orders(ts)

        # ─── PRE-MIN-HOLD gating: only if pos.age < MIN_HOLD_TICKS do we block sells ───
        MIN_TIME = C.MIN_HOLD_TICKS
        if direction == "sell":
            pos = self.positions.get(item)
            if pos is not None and pos.age < MIN_TIME:
                # has TP or SL actually been hit?
                last   = self.last_price.get(item, pos.avg_price)
                up_bp  = (last - pos.avg_price) / pos.avg_price * 1e4
                hit_tp = up_bp >= pos.tp_pct * 1e4
                hit_sl = up_bp <= -pos.sl_pct * 1e4
                if not (hit_tp or hit_sl):
                    # still too early & no TP/SL → force hold
                    idle      = True
                    direction = "hold"

        if not idle:
            self._do_action(item, ts, mid, direction, frac, tp_pct, sl_pct, hold_ticks)

        # mark‑to‑market
        pv = self._mark_to_market()
        self.hwm = max(self.hwm, pv)
        self.drawdown = max(self.drawdown, self.hwm - pv)
        self.dd_interval = max(self.dd_interval, self.drawdown)

        ratio = (pv + 1e-8) / (self.interval_start_val + 1e-8)
        self.agg_logret += math.log(max(ratio, 1e-8))   # guarded

        # advance
        self.idx += 1
        self.ticks += 1
        done = self.idx >= self.end_idx

        if done:
            for itm, pos in list(self.positions.items()):
                # --- robust last-seen mid price ---
                #  • some items occasionally record a 0.0 or NaN mid_price
                #  • guard it so all downstream math (ratios, log, bp) is safe
                last_raw = self.last_price.get(itm, pos.avg_price)
                last     = max(float(last_raw) if np.isfinite(last_raw) else pos.avg_price, 1e-8)
                self._close_pos(
                    Order(itm, pos.qty, "sell", ts, ts, max(last, 1e-8))
                )
            pv = self.capital                     # all positions are closed

        reward = 0.0
        if self.ticks % (C.REWARD_INTERVAL * C.TICKS_PER_MIN) == 0 or done:
            reward = self._interval_reward(pv, done)
            self.interval_start_val = pv        # reset anchor for next window
        
        info_out = {"pv": pv}

        # ★ first tick *after* the reward was computed
        if "interval" in locals():            # reward was just computed this step
            info_out["interval_reward"] = self._last_interval_reward

        if done:                                            # terminal snapshot
            info_out["eval_trades"] = list(self.eval_trades)
        # ─── minimal holding bonus ───
        # give +HOLD_BONUS_BP basis-points for each open position each tick
        n_pos = len(self.positions)
        if n_pos:
            # convert bp to your reward units: reward is already in bp‐like scale
            reward += C.HOLD_BONUS_BP * n_pos

        return self._get_obs(), reward, done, False, info_out

    # ---------------------------------------------------------
    def _do_action(self, item, ts, mid, direction, frac, tp, sl, hold):
        if direction in ("buy", "sell"):
            budget = self.capital * frac
            qty    = max(1, int(budget / mid))
        if direction == "buy":
            budget = self.capital * frac
            qty = int(budget / mid)
            eff_price = self._slip(mid, qty, "buy")
            cost = qty * eff_price * (1 + self.fee)
            if cost > self.capital:
                qty = int(self.capital / (eff_price * (1 + self.fee)))
                cost = qty * eff_price * (1 + self.fee)
            if qty <= 0:
                return
            if self.capital < cost:
                print(f"Insufficient capital for {item} at {mid:.2f} (cost: {cost:.2f})")
                return
            self.capital -= cost
            exec_ts = ts + self.rng.exponential(C.LATENCY_MEAN_S)
            self.orders.append(Order(item, qty, "buy", ts, exec_ts, eff_price))
        elif direction == "sell" and item not in self.positions:
            self.trade_cnt += 1         
            return
        elif direction == "sell" and item in self.positions:
            pos = self.positions[item]
            qty = pos.qty
            eff_price = self._slip(mid, qty, "sell")
            exec_ts = ts + self.rng.exponential(C.LATENCY_MEAN_S)
            self.orders.append(Order(item, qty, "sell", ts, exec_ts, eff_price))
        # update thresholds
        if item in self.positions:
            p = self.positions[item]
            p.tp_pct, p.sl_pct, p.hold_ticks = tp, sl, hold

    def _fill_orders(self, current_ts: float):
        """
        Pull any pending Orders whose exec_ts ≤ current_ts
        and turn them into opens or closes.
        """
        for od in list(self.orders):
            if current_ts >= od.exec_ts:
                if od.direction == "buy":
                    self._open_pos(od)
                else:
                    self._close_pos(od)
                self.orders.remove(od)

    # ---------------------------------------------------------
    def _open_pos(self, od: Order):
        """
        When a buy order executes, either start a new Position
        or add to an existing one—so we never lose cost basis.
        """
        # compute the cash you actually spent on this fill
        cost = od.qty * od.price * (1 + self.fee)

        if od.item in self.positions:
            # existing position → top it up
            pos = self.positions[od.item]
            # update qty and total cost
            pos.qty        += od.qty
            pos.total_cost += cost
            # you might want to reset age on top-ups, or leave existing age
            # pos.age = 0  
        else:
            # brand-new position
            self.positions[od.item] = Position(
                item       = od.item,
                entry_ts   = od.submit_ts,
                qty        = od.qty,
                total_cost = cost,
                tp_pct     = C.MAX_PROFIT_PCT,
                sl_pct     = C.MAX_LOSS_PCT,
                hold_ticks = C.MAX_HOLD_TICKS,
            )

        # log the trade as before
        self.trade_log.append({
            "item":  od.item,
            "side":  "BUY",
            "qty":   od.qty,
            "price": od.price,
            "ts":    od.exec_ts,
        })

    def _close_pos(self, od: Order):
        pos = self.positions.get(od.item)
        if not pos:
            return
        proceeds = od.qty * od.price * (1 - self.fee)
        pnl = proceeds - pos.entry_cash

        self.trade_log.append(
            {
                "item":  od.item,
                "side":  "SELL",
                "qty":   od.qty,
                "price": od.price,
                "ts":    od.exec_ts,
                "pnl":   pnl,                          # realised PnL in coins
            }
        )
        # update stats
        pnl_bp = pnl / self.initial_capital * 1e4
        if pnl_bp < C.MIN_PROFIT_BP:
            # only losing or tiny-win trades get penalised
            pnl_bp -= C.MIN_PROFIT_BP
            self.trade_cnt += 1
        # winners ≥ MIN_PROFIT_BP don’t increment trade_cnt
        self.realised_bp += pnl_bp
        self.capital    += proceeds


        # ─── log the closed trade ───
        self.eval_trades.append(
            dict(
                item=od.item,
                qty=od.qty,
                entry_time=pos.entry_ts,
                exit_time=od.exec_ts,
                entry_price=pos.avg_price,
                exit_price=od.price,
                pnl=pnl,
                pnl_bp=pnl_bp,
            )
        )
        held_ticks   = pos.age
        held_minutes = held_ticks / C.TICKS_PER_MIN  # since C.TICKS_PER_MIN ticks = 1 min
        print(
            f"[CLOSE t={self.ticks:6d}] SELL item={od.item:27s} "
            f"pnl_bp={pnl_bp:7.2f}   held={held_ticks:5d} ticks "
            f"({held_minutes:5.1f} min)"  
            f"pnl={pnl:13.1f}"  
            f"Return={((od.price/pos.avg_price)-1):5.3f}"
         )

        del self.positions[od.item]

    # --------------------------- Market State Functions ------------------------------
    def _slip(self, mid_raw, qty, dir_):
        depth = qty / 71_040
        slip = depth * C.SLIPPAGE_PCT_PER_STACK
        return mid_raw * (1 + slip if dir_ == "buy" else 1 - slip)

    def _interval_reward(self, pv, done):
        """
        Self-balancing 10-min reward, returned in raw bp then soft-clipped.
        """
        # ---------- portfolio bp helpers ----------
        scale = 1e4 / self.initial_capital
        realised_bp   = self.realised_pnl      * scale          # coins → bp
        unreal_bp     = self.unrealised_pnl    * scale
        new_dd_bp     = max(0, self.dd_interval - self.prev_dd) * scale
        trade_pen_bp  = max(0, self.trade_cnt - 5) *  2.0       # 2 bp per extra order
        compound_bp   = self.agg_logret * 1e4                   # already log-ret

        # ---------- EWMA adaptive weights ----------
        # update running RMS (√var) BEFORE using them
        for k, x in zip(("real","unreal","dd","trades","comp"),
                        (realised_bp, unreal_bp, new_dd_bp, trade_pen_bp, compound_bp)):
            self.r_stats[k].update(x)

        rms = {k: max(1.0, np.sqrt(s.var)) for k, s in self.r_stats.items()}  # avoid /0

        # inverse-variance weights so each term contributes ~equally
        w = {k: 1.0 / rms[k] for k in rms}
        # keep a minimum influence for realised PnL
        w["real"] = max(w["real"], 1.0 * max(w.values()))

        # ---------- weighted sum ----------
        raw_total = (
            w["real"]   * realised_bp
        + w["unreal"] * unreal_bp
        - w["dd"]     * new_dd_bp
        - w["trades"] * trade_pen_bp
        + w["comp"]   * compound_bp
        )

        # early clip to ±1k bp before VecNormalize
        reward = softclip(raw_total, limit=C.MAX_BP_CLIP)
        self.prev_dd = self.dd_interval

        # ------ terminal bonus: Sortino ratio ------
        if done:
            port_ret   = np.log(max(pv / self.initial_capital, 1e-8))
            downside_sd= np.sqrt(max(self.downside_variance, 1e-12))
            sortino    = port_ret / downside_sd if downside_sd > 0 else 0.0
            reward    += sortino * 100.0                        # bring to bp scale

        # ---------- reset interval stats ----------
        self.realised_pnl = self.unrealised_pnl = 0.0
        self.trade_cnt = 0
        self.prev_pv   = pv
        self.agg_logret = 0.0
        self.dd_interval = 0.0

        # ---------- introspection dict ----------
        self._last_interval_reward = dict(
            r_real=realised_bp, r_unreal=unreal_bp, r_dd=-new_dd_bp,
            r_trades=-trade_pen_bp, r_comp=compound_bp,
            w_real=w["real"], w_unreal=w["unreal"], w_dd=-w["dd"],
            w_trades=-w["trades"], w_comp=w["comp"],
            total=reward,
        )
        return reward
    

    def _mark_to_market(self):
        """
        Re-value the portfolio, queue auto-close orders, and return current PV.

        Side-effects
        ------------
        • Updates:
            - self.unrealised_pnl      (coins)
            - self.dd_interval         (coins)
            - self.downside_variance   (bp², for Sortino)
            - self.prev_pv             (coins)
        • Increments   pos.age
        • Enqueues      sell Orders   when TP / SL / hold_age hit
        """
        scale = 1e4 / self.initial_capital          # coins → bp
        pv: float = self.capital                    # start with idle cash
        self.unrealised_pnl = 0.0                   # reset each tick

        for itm, pos in list(self.positions.items()):
            # -------- latest mid price for this item --------
            last_raw = self.last_price.get(itm, pos.avg_price)
            last     = max(float(last_raw), 1e-8)    # guard against 0 / NaN

            # -------- update portfolio value & unrealised PnL --------
            pos_value = pos.qty * last * (1 - self.fee)
            pv += pos_value
            self.unrealised_pnl += (last - pos.avg_price) * pos.qty

            # -------- position ageing & exit checks --------
            pos.age += 1
            pos_up_bp = (last - pos.avg_price) / pos.avg_price * 1e4

            hit_tp  = pos_up_bp >=  pos.tp_pct * 1e4
            hit_sl  = pos_up_bp <= -pos.sl_pct * 1e4
            hit_age = pos.age   >=  pos.hold_ticks

            if hit_tp or hit_sl or hit_age:
                self.orders.append(
                    Order(
                        item       = itm,
                        qty        = pos.qty,
                        direction  = "sell",
                        submit_ts  = float(self.X_time[self.idx]),
                        exec_ts    = float(self.X_time[self.idx])
                                    + self.rng.exponential(C.LATENCY_MEAN_S),
                        price      = last,
                    )
                )

        # -------- draw-down & downside variance (Sortino) --------
        self.hwm = max(self.hwm, pv)
        dd_now   = self.hwm - pv
        self.dd_interval = max(self.dd_interval, dd_now)

        delta_bp = (pv - self.prev_pv) * scale      # bp change this tick
        if delta_bp < 0:                            # downside only
            self.downside_variance += delta_bp ** 2

        self.prev_pv = pv
        return pv

# ───────────── Teacher Dataset (BC warm‑start) ─────────────

def make_teacher_dataset(df: pd.DataFrame):
        # ---------- feature block ----------
    base_feats = [c for c in df.columns if c not in {"timestamp", "item", "datetime"}]

    # add a scaled item_code feature
    df["item_code"] = df["item"].astype("category").cat.codes.astype(np.float32)
    df["item_code"] = 2 * (df["item_code"] - df["item_code"].min()) / (
        df["item_code"].max() - df["item_code"].min() + 1
    ) - 1

    feats = df[base_feats + ["item_code"]].to_numpy(np.float32)

    # ---------- pad obs to env length ----------
    env_probe = BazaarTradingEnv(df, explore=False)
    obs_dim   = env_probe.observation_space.shape[0]
    if feats.shape[1] < obs_dim:
        pad = np.zeros((len(df), obs_dim - feats.shape[1]), dtype=np.float32)
        obs = np.hstack([feats, pad])
    elif feats.shape[1] > obs_dim:
        raise ValueError(
            f"Raw features ({feats.shape[1]}) exceed env obs dim ({obs_dim})"
        )
    else:
        obs = feats

    # ---------- build action vector ----------
    act_dim = env_probe.action_space.shape[0]        # ← 6 in your current code

    # 1. pull raw predictions
    labels = df["pred_label"].astype(int)                       # ∈ {−1,0,1}
    conf   = df["pred_class_confidence"].to_numpy(np.float32)   # ∈ [0,1]

    # 2. direction head: only buy if label==1 AND conf≥0.6; sell if label==-1; else hold
    dir_raw = np.zeros_like(conf)
    buy_mask  = (labels == 1) & (conf >= 0.6)
    sell_mask = (labels == -1)
    dir_raw[ buy_mask] = +1.0
    dir_raw[sell_mask] = -1.0
    # leave dir_raw==0 for “hold” otherwise

    # 3. fraction head: only meaningful for buys; zero elsewhere
    frac_raw = np.zeros_like(conf)
    frac_raw[buy_mask] = 2 * (conf[buy_mask] - 0.6) / 0.4 - 1  # map [0.6→1.0] → [−1→+1]

    # 4. idle head: skip action whenever not buy _or_ sell
    idle_raw = np.ones_like(conf) * +1.0   # default: idle
    idle_raw[buy_mask | sell_mask] = -1.0 # take action

    tp_raw    = GLOBAL_RNG.uniform(0.33, 0.35, len(df)).astype(np.float32)
    sl_raw    = GLOBAL_RNG.uniform(-.3, -.2, len(df)).astype(np.float32)
    hold_raw  = GLOBAL_RNG.uniform(.75, 1, len(df)).astype(np.float32)
    direction = dir_raw.astype(np.float32)

    acts = np.column_stack([
        dir_raw.astype(np.float32),
        frac_raw.astype(np.float32),
        tp_raw,        
        sl_raw,
        hold_raw,
        idle_raw.astype(np.float32),
    ])            # −1,0,+1


    # trim or pad to match env action size
    if act_dim < acts.shape[1]:
        acts = acts[:, :act_dim]
    elif act_dim > acts.shape[1]:
        pad_a = np.zeros((len(df), act_dim - acts.shape[1]), dtype=np.float32)
        acts  = np.hstack([acts, pad_a])

    return obs, acts

# ───────────────────────── Evaluation Helpers ─────────────────────────

def simulate_policy(env_df: pd.DataFrame, policy, *, csv_path: Path | None = None):
    """Run `policy` on a fresh env built from `env_df` and return a trade-log df."""
    env = BazaarTradingEnv(env_df, explore=False)
    obs, _ = env.reset()
    done   = False
    while not done:
        act, _ = policy.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(act)

    trade_df = pd.DataFrame(env.eval_trades)   # ← already complete & tidy
    
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        trade_df.to_csv(csv_path, index=False)
        print(f"✓ Teacher trade-log saved → {csv_path}")

    # ⇓⇓⇓  **DON’T forget to return it**  ⇓⇓⇓
    return trade_df            # ← add / keep this line

def evaluate(model: PPO, env: BazaarTradingEnv, limit_steps: int | None = None):
    obs, _ = env.reset()
    done, steps = False, 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        steps += 1
        if limit_steps and steps >= limit_steps:
            break
    return info["pv"]

def evaluate_bc(bc_policy: ActorCriticPolicy,
                env_df: pd.DataFrame,
                n_steps: int = 200_000):
    """Run the behavioural-cloning policy in the env and report PV."""
    env = BazaarTradingEnv(env_df, explore=False)
    obs, _ = env.reset()
    done, steps = False, 0
    while not done and steps < n_steps:
        # policy acts deterministically: mean of Gaussian
        with torch.no_grad():
            action, _ = bc_policy.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        steps += 1
    return info["pv"]

# ─────────────────────────── Training Schedules ───────────────────────

def lr_schedule(progress: float):
    # progress ∈ [0, 1]
    return C.PPO_LR_MIN + (C.PPO_LR_MAX - C.PPO_LR_MIN) * (1 - progress)


def ent_schedule(progress: float):
    k = math.log(C.PPO_ENT_START / C.PPO_ENT_END)
    adj = (progress - 0.5) / 0.5
    return max(C.PPO_ENT_END, C.PPO_ENT_START * math.exp(-k * adj))


class AutoScheduleCB(EvalCallback):
    """Keeps LR & entropy in sync with overall progress."""

    def __init__(self, *a, total_steps: int, **kw):
        super().__init__(*a, **kw)
        self.total_steps = total_steps

    def _on_step(self) -> bool:
        prog = self.model.num_timesteps / self.total_steps
        self.model.lr_schedule = lambda _: lr_schedule(prog)
        self.model.ent_coef = ent_schedule(prog)
        # —— freeze a minimum log-std to keep exploration alive 
        with torch.no_grad():
            self.model.policy.log_std.clamp_(min=-1.5)
        # —— reset optimizer state every 1 000 000 steps
        if self.model.num_timesteps % 1_000_000 == 0:
            self.model.policy.optimizer.state = {}
        return super()._on_step()


# ─────────────────────────────── Main  ────────────────────────────────

def main(total_timesteps: int = C.TOTAL_TIMESTEPS):
    # ---------- data ----------
    raw = load_dataset(C.DATA_PATH)
    train_df, test_df = split_dataset(raw)

    # ---------- BC warm-start ----------
    obs_t, act_t = make_teacher_dataset(train_df)
    demos = Transitions(
        obs=obs_t,
        acts=act_t,
        next_obs=obs_t.copy(),
        dones=np.zeros(len(obs_t), dtype=bool),
        infos=np.empty(len(obs_t), dtype=object),
    )

    env_tmp = BazaarTradingEnv(train_df, explore=False)
    policy_kwargs = dict(net_arch=[1024, 512, 256, 128, 128, 128], log_std_init=-2.0)

    bc_policy = ActorCriticPolicy(
        observation_space=env_tmp.observation_space,
        action_space=env_tmp.action_space,
        lr_schedule=lambda _: 1e-3,
        **policy_kwargs,
    )

    bc = BehavioralCloning(
        observation_space=env_tmp.observation_space,   # ← required
        action_space=env_tmp.action_space,             # ← required
        demonstrations=demos,
        policy=bc_policy,
        batch_size=2048,
        optimizer_kwargs=dict(lr=5e-4),
        rng=GLOBAL_RNG,
        device="cuda",          # change to "cuda" if you have a GPU
    )
    bc.train(n_epochs=C.N_BC_EPOCHS)
    teacher_log = simulate_policy(
        test_df,
        bc_policy,                                   # the policy you just trained
        csv_path=C.OUTPUT_DIR / "teacher_trades.csv"
    )
    if not teacher_log.empty:
        print("Teacher summary:")
        print(teacher_log["pnl"].describe())
    else:
        print("Teacher produced no completed trades.")
    pv_bc = evaluate_bc(bc_policy, test_df)
    print(f"BC-only final PV on hold-out: {pv_bc:,.0f} coins "
      f"(net {pv_bc-1_000_000_000:+,.0f})")
    pretrained_state = bc.policy.state_dict()
    del bc, env_tmp

    # ---------- vectorised envs ----------
    def make_env(df_part, explore=True):
        return lambda: Monitor(BazaarTradingEnv(df_part, explore=explore))

    n_envs = max(1, min(C.MAX_ENVS, os.cpu_count() // 2))
    train_env = SubprocVecEnv([make_env(train_df) for _ in range(n_envs)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    test_env = DummyVecEnv([make_env(test_df, explore=True)])
    test_env = VecNormalize(test_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    test_env.obs_rms, test_env.ret_rms = train_env.obs_rms, train_env.ret_rms
    test_env.training = False

    # ---------- PPO ----------
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=50_000,
        policy_kwargs=policy_kwargs,
        n_steps=C.PPO_N_STEPS,
        batch_size=C.PPO_BATCH_SIZE,
        n_epochs=C.PPO_EPOCHS,
        gamma=C.PPO_GAMMA,
        gae_lambda=C.PPO_LAMBDA,
        clip_range=C.PPO_CLIP,
        ent_coef=C.PPO_ENT_START,
        learning_rate=lr_schedule,
        max_grad_norm=C.GRAD_NORM_CLIP,
        tensorboard_log=str(C.OUTPUT_DIR / "tb"),
        device="cpu",
    )
    # model.policy.load_state_dict(pretrained_state, strict=False)


    eval_cb = AutoScheduleCB(
        test_env,
        best_model_save_path=str(C.OUTPUT_DIR),
        log_path=str(C.OUTPUT_DIR),
        eval_freq=C.PPO_N_STEPS*n_envs*4,
        deterministic=False,
        total_steps=total_timesteps,
    )
    
    interval_cb = IntervalRewardLogger(
        csv_path=C.OUTPUT_DIR / "interval_reward_log.csv"
    )

    # pass BOTH callbacks – SB3 lets you compose via list
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_cb, interval_cb],
    )


    # ---------- final evaluation (normalized) ----------
    print("Model action space:", model.action_space)
    print("Environment action space:", test_env.action_space)
    def evaluate_and_log(model, eval_env, n_episodes: int, log_path: Path):
        def unwrap_vec_env(vec_env, idx: int = 0):
            # same as before: peel off VecNormalize / VecEnv / Monitor → BazaarTradingEnv
            env = vec_env
            while hasattr(env, "venv"):
                env = env.venv
            if hasattr(env, "envs"):
                env = env.envs[idx]
            while hasattr(env, "env"):
                env = env.env
            return env

        all_trades = []
        for ep in range(n_episodes):
            # ——— reset & unwrap (unchanged) ———
            obs = eval_env.reset()[0]
            done, state = False, None
            base_env = unwrap_vec_env(eval_env)

            # ——— rollout loop (unchanged except for grabbing trades via info) ———
            while not done:
                action, state = model.predict(obs, state, deterministic=True)

                # force the right shape for VecEnv
                if isinstance(action, np.ndarray) and action.ndim == 1:
                    action = action.reshape(1, -1)

                # step through the wrapped env
                obs, reward, done, infos = eval_env.step(action)
                info = infos[0]

                # ——— as soon as done, snap out the trades ———
                if done:
                    trades = info.get("eval_trades", [])
                    print(f"Episode {ep+1}/{n_episodes} — closed trades: {len(trades)}")
                    all_trades.extend(trades)

            # (Optional) print a summary per episode if you like
            print(f"Episode {ep+1}/{n_episodes} done")

        # ---- sanitise & save ----
        def safe(tr):
            return {k:
                (float(v) if isinstance(v, (np.floating, np.integer))
                else str(v) if isinstance(v, (np.datetime64, datetime.datetime, pd.Timestamp))
                else v)
                for k, v in tr.items()}
        pd.DataFrame([safe(t) for t in all_trades]).to_csv(log_path, index=False)
        print(f"✓ Evaluation trade log saved → {log_path}  ({len(all_trades)} rows)")

    evaluate_and_log(
        model, test_env, n_episodes=5, log_path=C.TRADING_DIR / "rl_trade_log.csv"
    )



# ───────────────────────── CLI entry-point ────────────────────────────
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(
        description="Train & evaluate the Bazaar PPO agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=C.TOTAL_TIMESTEPS,
        help="Total environment steps for PPO training",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(C.DATA_PATH),
        help="Path *or glob pattern* to the RL dataset (CSV/Parquet). "
             "Overrides config.RL_DATASET_CSV at run-time.",
    )
    args = parser.parse_args()

    # honour any dataset override provided via CLI
    C.DATA_PATH = Path(args.data)

    # make sure the output directory exists (helpful when OUTPUT_DIR is on another drive)
    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # kick off the full training / evaluation pipeline
    main(total_timesteps=args.timesteps)
