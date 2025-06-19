"""Evaluate a trained RL model using the existing backtest logic."""

import argparse
import os
from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO

from rl_env import BazaarEnv
import run_simulation as sim
import config


def generate_predictions(model: PPO, env: BazaarEnv) -> pd.DataFrame:
    """Run the model in the environment and log predicted actions."""
    obs, _ = env.reset()
    records = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        row = env.df.iloc[env.step_idx]
        pred_label = 1 if action == 1 else -1 if action == 2 else 0
        # Use a fixed high confidence for buys to maximise trade size
        prob_buy = 0.9 if action == 1 else 0.1
        records.append(
            {
                "item": row["item"],
                "timestamp": row["timestamp"],
                "mid_price": row["mid_price"],
                "pred_label": pred_label,
                "pred_proba_buy": prob_buy,
            }
        )
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return pd.DataFrame(records)


def main(model_path: str, output_path: str) -> None:
    """Generate trades from an RL model and evaluate profit."""
    model = PPO.load(model_path)
    env = BazaarEnv()

    pred_df = generate_predictions(model, env)
    try:
        trade_df, summary = sim.run_backtest(pred_df)
    except KeyError:
        # run_backtest fails if no trades are taken
        trade_df = pd.DataFrame()
        summary = {
            "final_capital": env.initial_capital,
            "total_profit": 0.0,
            "num_trades": 0,
            "average_return": 0.0,
            "win_rate": 0.0,
        }

    print("\nRL Trading Summary:")
    for k, v in summary.items():
        print(f"{k}: {v}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    trade_df.to_csv(output_path, index=False)
    print(f"\nTrade log saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained RL model")
    parser.add_argument(
        "--model",
        default=str(Path("Outputs/rl_model.zip")),
        help="Path to the trained RL model",
    )
    parser.add_argument(
        "--output",
        default=str(config.TRADING_DIR / "rl_trade_log.csv"),
        help="Where to save the trade log CSV",
    )
    args = parser.parse_args()
    main(args.model, args.output)
