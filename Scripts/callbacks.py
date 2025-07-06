# callbacks.py
from stable_baselines3.common.callbacks import BaseCallback
import pathlib, time, csv

class IntervalRewardLogger(BaseCallback):
    """
    Collects info["interval_reward"] from every sub-env, writes it to:
      • TensorBoard scalars   rollout/interval_reward  (+ sub-fields if dict)
      • CSV                   interval_reward_log.csv
    """
    def __init__(self, csv_path: str | pathlib.Path, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = pathlib.Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        # header
        with self.csv_path.open("w", newline="") as f:
            csv.writer(f).writerow(
                ["wall_time", "env_idx", "key", "value"]
            )

    # ---------------------------------------------------
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for env_idx, info in enumerate(infos):
            if "interval_reward" not in info:
                continue

            reward_blob = info["interval_reward"]

            # ---- 1) TensorBoard ----
            if isinstance(reward_blob, dict):
                # log each component
                for k, v in reward_blob.items():
                    self.logger.record_mean(f"rollout/{k}", v)
            else:   # scalar
                self.logger.record_mean("rollout/interval_reward", reward_blob)

            # ---- 2) CSV ----
            ts = time.time()
            if isinstance(reward_blob, dict):
                rows = [(ts, env_idx, k, v) for k, v in reward_blob.items()]
            else:
                rows = [(ts, env_idx, "total", reward_blob)]

            with self.csv_path.open("a", newline="") as f:
                csv.writer(f).writerows(rows)

        return True
