import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config


class BazaarTradingEnv(gym.Env):
    """Trading environment supporting a portfolio of many items."""

    def __init__(self, data: pd.DataFrame, initial_capital: float = 25_000_000, trade_fee: float = 0.001):
        super().__init__()
        self.df = data.reset_index(drop=True)
        exclude = {"timestamp", "item", "datetime"}
        self.features = [c for c in self.df.columns if c not in exclude]
        self.initial_capital = float(initial_capital)
        self.trade_fee = float(trade_fee)

        low = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 0.2, 10.0], dtype=np.float32)
        self.action_space = spaces.Box(low, high, dtype=np.float32)

        # observation includes portfolio statistics
        obs_len = len(self.features) + 5
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_len,), dtype=np.float32)
        self.reset()

    def reset(self, *, seed: int | None = None, options=None):  # noqa: D401
        super().reset(seed=seed)
        self.capital = float(self.initial_capital)
        self.positions: dict[str, dict] = {}
        self.last_prices: dict[str, float] = {}
        self.prev_value = self.initial_capital
        self.step_idx = 0
        self.trade_log: list[dict] = []
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        row = self.df.iloc[self.step_idx]
        feats = row[self.features].values.astype(np.float32)
        portfolio_value = self.capital
        profit_pcts = []
        ages = []
        for item, pos in self.positions.items():
            last = self.last_prices.get(item, pos["entry_price"])
            portfolio_value += pos["qty"] * last
            profit_pcts.append((last - pos["entry_price"]) / pos["entry_price"])
            ages.append(pos["age"])
        avg_profit = float(np.mean(profit_pcts)) if profit_pcts else 0.0
        avg_age = float(np.mean(ages)) if ages else 0.0
        obs = np.concatenate(
            [feats, [self.capital, portfolio_value, len(self.positions), avg_profit, avg_age]]
        )
        return obs

    def _close_position(self, item: str, price: float, timestamp: float):
        pos = self.positions[item]
        pnl = (price - pos["entry_price"]) * pos["qty"]
        self.capital += pos["qty"] * price * (1 - self.trade_fee)
        self.trade_log.append(
            {
                "item": item,
                "entry_time": pos["entry_time"],
                "exit_time": timestamp,
                "entry_price": pos["entry_price"],
                "exit_price": price,
                "size": pos["qty"],
                "pnl": pnl,
                "return": (price - pos["entry_price"]) / pos["entry_price"],
            }
        )
        del self.positions[item]

    def step(self, action):
        row = self.df.iloc[self.step_idx]
        price = row["mid_price"]
        item = row["item"]
        self.last_prices[item] = price
        direction, fraction, profit_th, hold_period = action

        # interpret discrete action from direction value
        act = 0
        if direction > 0.33:
            act = 1  # buy
        elif direction < -0.33:
            act = 2  # sell

        if act == 2 and item in self.positions:
            self._close_position(item, price, row["timestamp"])

        if act == 1:
            budget = self.capital * float(np.clip(fraction, 0.0, 1.0))
            qty = (budget * (1 - self.trade_fee)) / price
            if qty > 0:
                self.capital -= budget
                self.positions[item] = {
                    "qty": self.positions.get(item, {"qty": 0}).get("qty", 0) + qty,
                    "entry_price": price,
                    "entry_time": row["timestamp"],
                    "age": 0,
                    "threshold": float(np.clip(profit_th, 0.0, 0.2)),
                    "hold_period": int(np.clip(hold_period, 0.0, 10.0)),
                }

        # age all positions and automatically close if thresholds are hit
        for itm in list(self.positions.keys()):
            pos = self.positions[itm]
            pos["age"] += 1
            last = self.last_prices.get(itm, pos["entry_price"])
            if itm == item:
                last = price
            profit = (last - pos["entry_price"]) / pos["entry_price"]
            if profit >= pos["threshold"] or pos["age"] >= pos["hold_period"]:
                self._close_position(itm, last, row["timestamp"])

        self.step_idx += 1
        done = self.step_idx >= len(self.df) - 1
        portfolio_value = self.capital
        for itm, pos in self.positions.items():
            portfolio_value += pos["qty"] * self.last_prices.get(itm, pos["entry_price"])
        reward = portfolio_value - self.prev_value
        self.prev_value = portfolio_value
        obs = self._get_obs()
        info = {"account_value": portfolio_value}
        terminated = done
        truncated = False
        return obs, reward, terminated, truncated, info


def split_dataset(df: pd.DataFrame, test_size: float = 0.2):
    split = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split].reset_index(drop=True)
    test_df = df.iloc[split:].reset_index(drop=True)
    return train_df, test_df


def evaluate(model: PPO, env: BazaarTradingEnv, limit_steps: int | None = None) -> tuple[pd.DataFrame, dict]:
    obs, _ = env.reset()
    done = False
    steps = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        steps += 1
        if limit_steps is not None and steps >= limit_steps:
            break
    trades = pd.DataFrame(env.trade_log)
    if not trades.empty:
        trades = trades.sort_values("pnl", ascending=False)
    portfolio_value = env.capital
    for itm, pos in env.positions.items():
        last = env.last_prices.get(itm, pos["entry_price"])
        portfolio_value += pos["qty"] * last
    summary = {
        "final_capital": portfolio_value,
        "total_profit": portfolio_value - env.initial_capital,
        "num_trades": len(trades),
        "average_return": trades["return"].mean() if len(trades) > 0 else 0.0,
        "win_rate": (trades["pnl"] > 0).mean() if len(trades) > 0 else 0.0,
    }
    return trades, summary


def permutation_importance(model: PPO, base_env: BazaarTradingEnv, n_steps: int = 200) -> pd.DataFrame:
    _, base_summary = evaluate(model, base_env, limit_steps=n_steps)
    base_profit = base_summary["total_profit"]
    importances = []
    for feat in base_env.features:
        df_perm = base_env.df.copy()
        df_perm[feat] = np.random.permutation(df_perm[feat].values)
        env = BazaarTradingEnv(df_perm, base_env.initial_capital, base_env.trade_fee)
        _, summary = evaluate(model, env, limit_steps=n_steps)
        profit_drop = base_profit - summary["total_profit"]
        importances.append({"feature": feat, "profit_drop": profit_drop})
    return pd.DataFrame(importances).sort_values("profit_drop", ascending=False)


def main(timesteps: int = 5000):
    df = pd.read_csv(config.RL_DATASET_CSV)
    train_df, test_df = split_dataset(df)

    train_env = DummyVecEnv([lambda: Monitor(BazaarTradingEnv(train_df))])
    test_env = DummyVecEnv([lambda: Monitor(BazaarTradingEnv(test_df))])
    eval_callback = EvalCallback(test_env, best_model_save_path=str(config.OUTPUTS_DIR), log_path=str(config.OUTPUTS_DIR), eval_freq=1000, deterministic=True)

    model = PPO("MlpPolicy", train_env, verbose=1, policy_kwargs={"net_arch": [128, 128]}, n_steps=16, batch_size=16)
    model.learn(total_timesteps=timesteps, callback=eval_callback)

    model.save(config.RL_MODEL_PATH)
    print(f"Model saved to {config.RL_MODEL_PATH}")

    # evaluation
    eval_env = BazaarTradingEnv(test_df)
    trades, summary = evaluate(model, eval_env)
    print("\nEvaluation Summary:")
    for k, v in summary.items():
        print(f"{k}: {v}")

    important = permutation_importance(model, eval_env)
    print("\nFeature Importance (profit drop when permuted):")
    print(important.to_string(index=False))

    top_trades = trades.head(5)
    print("\nTop Trades:")
    print(top_trades.to_string(index=False))

    Path(config.TRADING_DIR).mkdir(parents=True, exist_ok=True)
    trades.to_csv(config.RL_TRADE_LOG_CSV, index=False)
    print(f"\nTrade log saved to {config.RL_TRADE_LOG_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate RL agent")
    parser.add_argument("--timesteps", type=int, default=5000, help="Total training timesteps")
    args = parser.parse_args()
    main(args.timesteps)
