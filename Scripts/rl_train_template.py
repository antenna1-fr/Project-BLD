"""Train a reinforcement learning agent on the :class:`BazaarEnv`.

The script uses Stable-Baselines3's PPO implementation with a modest network
architecture so it can train quickly on the included sample data.  The purpose
is to provide a ready-to-run starting point for more sophisticated experiments.
Run ``prepare_rl_dataset.py`` beforehand to build the dataset used by
``BazaarEnv``.
"""

import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from rl_env import BazaarEnv


def main(total_timesteps: int = 5000):
    """Train PPO on the trading environment.

    Parameters
    ----------
    total_timesteps : int
        Number of environment steps to train for.
    """

    env = DummyVecEnv([lambda: Monitor(BazaarEnv())])
    eval_env = DummyVecEnv([lambda: Monitor(BazaarEnv())])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="Outputs",
        log_path="Outputs",
        eval_freq=1000,
        deterministic=True,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        policy_kwargs={"net_arch": [128, 128]},
        n_steps=16,
        batch_size=16,
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    model.save("Outputs/rl_model")
    print("Model saved to Outputs/rl_model")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train RL agent for Bazaar")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=5000,
        help="Number of environment steps to train for",
    )
    args = parser.parse_args()
    main(args.timesteps)
