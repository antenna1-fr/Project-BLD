"""Utility to generate a dataset suitable for RL training.

It simply strips the classification label column and sorts the data
so an RL environment can iterate over it chronologically.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config

INPUT_CSV = config.PROCESSED_CSV
OUTPUT_CSV = config.RL_DATASET_CSV


def main():
    df = pd.read_csv(INPUT_CSV)

    # Drop label column if present
    df = df.drop(columns=[c for c in ['label'] if c in df.columns])

    df = df.sort_values(['item', 'timestamp']).reset_index(drop=True)

    # Replace missing/inf values that might destabilise RL training
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"RL dataset saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
