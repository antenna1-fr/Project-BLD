"""Gymnasium environment for trading simulation.

This environment iterates through historical data produced by
``prepare_rl_dataset.py`` and allows an RL agent to take ``buy``/``sell``/``hold``
actions. The reward at each step is the incremental change in portfolio
value so an RL algorithm can directly optimise for profit.  The
environment is intentionally lightweight so it can run on the small
dataset included in the repository but the design mirrors a full trading
simulator.
"""

import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import Env, spaces
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config


class BazaarEnv(Env):
    """Simple trading simulator compatible with Gymnasium."""

    def __init__(self, csv_path=None, initial_capital=25_000_000, trade_fee=0.001):
        super().__init__()
        self.csv_path = csv_path or config.RL_DATASET_CSV
        self.initial_capital = float(initial_capital)
        self.trade_fee = float(trade_fee)

        self.df = pd.read_csv(self.csv_path)
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)

        exclude = {'timestamp', 'item', 'datetime'}
        self.feature_cols = [c for c in self.df.columns if c not in exclude]

        # action: 0 hold, 1 buy, 2 sell
        self.action_space = spaces.Discrete(3)
        # observation: features + [cash, position]
        obs_len = len(self.feature_cols) + 2
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_len,), dtype=np.float32)

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.capital = float(self.initial_capital)
        self.position = 0.0  # number of items held
        self.entry_price = 0.0
        self.prev_account_value = self.initial_capital
        self.step_idx = 0
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.df.iloc[self.step_idx]
        features = row[self.feature_cols].values.astype(np.float32)
        position_val = self.position * row['mid_price']
        obs = np.concatenate([features, [self.capital, position_val]])
        return obs

    def step(self, action):
        row = self.df.iloc[self.step_idx]
        price = row['mid_price']

        # --- execute action -------------------------------------------------
        if action == 1 and self.position == 0:  # buy
            # purchase as many items as possible minus trading fee
            qty = (self.capital * (1 - self.trade_fee)) / price
            self.position = qty
            self.capital = 0.0
        elif action == 2 and self.position > 0:  # sell
            self.capital = self.position * price * (1 - self.trade_fee)
            self.position = 0.0

        # -------------------------------------------------------------------

        self.step_idx += 1
        done = self.step_idx >= len(self.df) - 1

        account_value = self.capital + self.position * price
        reward = account_value - self.prev_account_value
        self.prev_account_value = account_value

        obs = self._get_obs()
        info = {"account_value": account_value}
        terminated = done
        truncated = False
        return obs, reward, terminated, truncated, info


def sample_run(steps=5):
    env = BazaarEnv()
    obs, _ = env.reset()
    for i in range(steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        print(f"step {i}: action={action}, reward={reward}")
        if terminated or truncated:
            break


if __name__ == "__main__":
    sample_run()
