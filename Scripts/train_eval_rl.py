# ─────────────────────────────── Imports ────────────────────────────────
from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Dict, List

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv  # swap to DummyVecEnv if pickling errors occur

# Append project root so `import config` works when running as script
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


# ──────────────────────────── Data Structures ────────────────────────────

@dataclass
class Position:
    qty: float
    entry_cash: float  # cash laid out INCLUDING entry fee
    entry_time: float
    avg_entry_price: float  # entry_cash / qty (fee‑adjusted basis)
    age: int
    threshold: float  # fractional profit target (0.01 … 0.2)
    hold_period: int  # steps to force‑close (30 … 1000)

    @property
    def cash_cost(self) -> float:  # entry_cash already includes fee
        return self.entry_cash


# ───────────────────────────── Environment ───────────────────────────────

class BazaarTradingEnv(gym.Env):
    """Custom Gymnasium environment for multi‑item Bazaar trading.

    Action  (continuous) = [direction, fraction, profit_threshold, hold_period]
        direction ∈ [‑1, 1]  : <‑0.33 → **sell**, ‑0.33…0.33 → **hold**, >0.33 → **buy**
        fraction  ∈ [0, 1]   : share of available capital to deploy when buying
        profit_threshold ∈ [0, 0.2]
        hold_period      ∈ [0, 10] : mapped to 30…1000 env steps

    Observation = engineered price features  + 5 account stats
                 [capital_norm, portfolio_value_norm, n_positions, avg_profit, avg_age]
    """

    metadata: Dict[str, list] = {"render.modes": []}  # no visualisation yet

    # ───────────────────────────── Init ──────────────────────────────────

    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 25_000_000,
        trade_fee: float = 0.001,
    ) -> None:
        super().__init__()

        # Static config
        self.initial_capital: float = float(initial_capital)
        self.trade_fee: float = float(trade_fee)
        self.df: pd.DataFrame = data.reset_index(drop=True)

        # Features exclude raw identifiers / timestamps
        self.features: List[str] = [
            c for c in self.df.columns if c not in {"timestamp", "item", "datetime"}
        ]

        # Action space (continuous)
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 0.2, 10.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation space
        obs_len = len(self.features) + 5
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float32
        )

        # ε‑greedy exploration hyper‑parameters
        self.epsilon: float = 0.30  # start ε
        self.epsilon_decay: float = 0.99995
        self.epsilon_min: float = 0.01

        self.reset()

    # ─────────────────────────── Helpers ────────────────────────────────

        # ────────────── BazaarTradingEnv._get_obs ─────────────
    def _get_obs(self) -> np.ndarray:
        row = self.df.iloc[self.step_idx]
        feats = row[self.features].values.astype(np.float32)

        portfolio_value = self.capital
        profit_pcts, ages = [], []

        # NOTE: iterate over both key and value
        for itm, pos in self.positions.items():
            last_price = self.last_prices.get(itm, pos.avg_entry_price)
            portfolio_value += pos.qty * last_price
            profit_pcts.append((last_price - pos.avg_entry_price) / pos.avg_entry_price)
            ages.append(pos.age)

        avg_profit = float(np.mean(profit_pcts)) if profit_pcts else 0.0
        avg_age    = float(np.mean(ages))        if ages        else 0.0

        obs = np.concatenate(
            [
                feats,
                [
                    self.capital / self.initial_capital,
                    portfolio_value / self.initial_capital,
                    len(self.positions),
                    avg_profit,
                    avg_age,
                ],
            ]
        )
        return obs.astype(np.float32)

# ──────────────────────── Core Gym API ──────────────────────────────

    # ───────────────────────── Episode control ───────────────────────────
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.capital      = float(self.initial_capital)
        self.positions    = {}      # type: dict[str, Position]
        self.last_prices  = {}      # type: dict[str, float]
        self.prev_value   = self.initial_capital
        self.step_idx     = 0
        self.trade_log    = []
        # ← return *both* obs and info
        return self._get_obs(), {}

    def step(self, action):
        """Run one step of the environment."""
        # Exploration – draw random action with prob ε
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if np.random.rand() < self.epsilon:
            action = np.random.uniform(self.action_space.low, self.action_space.high).astype(
                np.float32
            )

        direction, fraction, profit_th_raw, hold_period_raw = action.astype(float)

        # Current market row
        row = self.df.iloc[self.step_idx]
        item: str = row["item"]
        price: float = float(row["mid_price"])
        timestamp: float = float(row["timestamp"])
        self.last_prices[item] = price

        # Decode intent
        intent = "hold"
        if direction > 0.33:
            intent = "buy"
        elif direction < -0.33:
            intent = "sell"

        opened_this_step: set[str] = set()

        # --------------------------- SELL ---------------------------
        if intent == "sell" and item in self.positions:
            self._close_position(item, price, timestamp)

        # --------------------------- BUY ---------------------------
        if intent == "buy":
            fraction = float(np.clip(fraction, 0.002, 1.0))
            budget_with_fee = self.capital * fraction  # cash OUT including entry fee
            if budget_with_fee > 0:
                cost_before_fee = budget_with_fee / (1 + self.trade_fee)
                qty = cost_before_fee / price

                # Update capital immediately (reservation of cash)
                self.capital -= budget_with_fee

                threshold = float(np.clip(profit_th_raw, 0.01, 0.2))
                hold_period = int(np.interp(hold_period_raw, [0.0, 10.0], [30, 1000]))

                if item in self.positions:
                    pos = self.positions[item]
                    new_qty = pos.qty + qty
                    new_entry_cash = pos.entry_cash + budget_with_fee
                    avg_entry_price = new_entry_cash / new_qty / (1 - self.trade_fee)

                    # Merge position atomically
                    self.positions[item] = Position(
                        qty=new_qty,
                        entry_cash=new_entry_cash,
                        entry_time=pos.entry_time,  # keep oldest entry time
                        avg_entry_price=avg_entry_price,
                        age=0,
                        threshold=threshold,  # overwrite TP rules with latest
                        hold_period=hold_period,
                    )
                else:
                    self.positions[item] = Position(
                        qty=qty,
                        entry_cash=budget_with_fee,
                        entry_time=timestamp,
                        avg_entry_price=cost_before_fee / qty,
                        age=0,
                        threshold=threshold,
                        hold_period=hold_period,
                    )
                opened_this_step.add(item)

        # --------------------- Update existing positions ---------------------
        for itm, pos in list(self.positions.items()):
            if itm in opened_this_step:
                continue  # cannot close immediately after opening

            last = price if itm == item else self.last_prices.get(itm, pos.avg_entry_price)
            current_val = pos.qty * last * (1 - self.trade_fee)
            profit = (current_val - pos.cash_cost) / pos.cash_cost

            # auto‑close rules
            if pos.age >= 2 and (profit >= pos.threshold or pos.age >= pos.hold_period):
                self._close_position(itm, last, timestamp)
                continue

            # otherwise just age the position
            pos.age += 1

        # -------------------- Advance time & compute reward -------------------
        self.step_idx += 1
        terminated = self.step_idx >= len(self.df) - 1
        truncated = False

        portfolio_value = self.capital + sum(
            p.qty * self.last_prices.get(itm, p.avg_entry_price) * (1 - self.trade_fee)
            for itm, p in self.positions.items()
        )

        raw_r = np.log((portfolio_value + 1e-8) / (self.prev_value + 1e-8))
        reward = float(np.clip(raw_r, -0.05, 0.05))
        self.prev_value = portfolio_value

        return self._get_obs(), reward, terminated, truncated, {"account_value": portfolio_value}

    # ──────────────────────────── Internals ─────────────────────────────

    def _close_position(self, item: str, price: float, timestamp: float) -> None:
        """Close *item* at *price* and realise PnL."""
        pos = self.positions[item]
        exit_cash = pos.qty * price * (1 - self.trade_fee)
        pnl = exit_cash - pos.entry_cash

        self.capital += exit_cash

        self.trade_log.append(
            {
                "item": item,
                "entry_time": pos.entry_time,
                "exit_time": timestamp,
                "qty": pos.qty,
                "entry_price": pos.avg_entry_price,
                "exit_price": price,
                "pnl": pnl,
                "return": pnl / pos.entry_cash,
            }
        )

        del self.positions[item]


# ──────────────────────────── Utilities ────────────────────────────────

def split_dataset(df: pd.DataFrame, test_size: float = 0.2):
    split = int(len(df) * (1 - test_size))
    return df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(drop=True)


def evaluate(model: PPO, env: BazaarTradingEnv, limit_steps: int | None = None):
    obs, _ = env.reset()
    terminated, truncated, steps = False, False, 0
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        steps += 1
        if limit_steps is not None and steps >= limit_steps:
            break

    trades = pd.DataFrame(env.trade_log).sort_values("pnl", ascending=False)
    portfolio_value = env.capital + sum(
        p.qty * env.last_prices.get(itm, p.avg_entry_price) * (1 - env.trade_fee)
        for itm, p in env.positions.items()
    )

    summary = {
        "final_capital": portfolio_value,
        "total_profit": portfolio_value - env.initial_capital,
        "num_trades": len(trades),
        "average_return": trades["return"].mean() if not trades.empty else 0.0,
        "win_rate": (trades["pnl"] > 0).mean() if not trades.empty else 0.0,
    }
    return trades, summary


def permutation_importance(model: PPO, base_env: BazaarTradingEnv, n_steps: int = 200):
    _, base_summary = evaluate(model, base_env, limit_steps=n_steps)
    base_profit = base_summary["total_profit"]

    importances = []
    for feat in base_env.features:
        df_perm = base_env.df.copy()
        df_perm[feat] = np.random.permutation(df_perm[feat].values)
        env = BazaarTradingEnv(df_perm, base_env.initial_capital, base_env.trade_fee)
        _, summary = evaluate(model, env, limit_steps=n_steps)
        importances.append({"feature": feat, "profit_drop": base_profit - summary["total_profit"]})

    return pd.DataFrame(importances).sort_values("profit_drop", ascending=False)


# ────────────────────────────── Main ────────────────────────────────────

def main(timesteps: int = 40_000):
    df = pd.read_csv(config.RL_DATASET_CSV)

    # down‑cast floats for memory savings
    float_cols = df.select_dtypes(include="float64").columns
    df[float_cols] = df[float_cols].astype(np.float32)

    train_df, test_df = split_dataset(df, test_size=0.25)

    # Env validation catches shape/dtype mismatches early
    check_env(BazaarTradingEnv(train_df), skip_render_check=True)

    train_env = SubprocVecEnv([lambda: Monitor(BazaarTradingEnv(train_df)) for _ in range(4)])
    test_env = SubprocVecEnv([lambda: Monitor(BazaarTradingEnv(test_df)) for _ in range(4)])

    eval_cb = EvalCallback(
        test_env,
        best_model_save_path=str(config.OUTPUTS_DIR),
        log_path=str(config.OUTPUTS_DIR),
        eval_freq=1_000,
        deterministic=True,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        policy_kwargs={"net_arch": [512, 256, 128]},
        n_steps=1_024,
        batch_size=512,
        tensorboard_log=str(config.OUTPUTS_DIR / "tensorboard_logs"),
        device="cpu",
    )

    model.learn(total_timesteps=timesteps, callback=eval_cb)
    model.save(config.RL_MODEL_PATH)

    # -------------------------- Evaluation --------------------------
    eval_env = BazaarTradingEnv(test_df)
    trades, summary = evaluate(model, eval_env)

    print("\nEvaluation summary")
    for k, v in summary.items():
        print(f"  {k:15}: {v}")

    important = permutation_importance(model, eval_env)
    print("\nFeature importances (profit drop ↑ worse):\n", important.to_string(index=False))

    print("\nTop 5 trades:\n", trades.head(5).to_string(index=False))

    Path(config.TRADING_DIR).mkdir(parents=True, exist_ok=True)
    trades.to_csv(config.RL_TRADE_LOG_CSV, index=False)
    print(f"\nTrade log saved → {config.RL_TRADE_LOG_CSV}")


# ─────────────────────────── CLI entry‑point ───────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate Bazaar PPO agent")
    parser.add_argument("--timesteps", type=int, default=40_000, help="Total training timesteps")
    parsed = parser.parse_args()
    main(parsed.timesteps)
