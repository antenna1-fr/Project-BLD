"""Utility to generate a dataset suitable for RL training.

It simply strips the classification label column and sorts the data
so an RL environment can iterate over it chronologically.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as config

INPUT_CSV = config.PROCESSED_CSV
OUTPUT_CSV = config.RL_DATASET_CSV


def main():
    df = pd.read_csv(INPUT_CSV)
    df = df.drop(columns=[c for c in ['label'] if c in df.columns])
    xgb_output_df = pd.read_csv(config.PREDICTIONS_CSV)[['pred_label','pred_class_confidence']]

    n_rows = df.shape[0]
    insert_at = int(n_rows * 0.8)

    df['pred_label']            = 0
    df['pred_class_confidence'] = 0

    # Add prediction info to RL dataframe

    print(len(df), len(xgb_output_df))
    print(insert_at, len(df.loc[insert_at:]), len(xgb_output_df.loc[insert_at:]))

    df.loc[insert_at:, 'pred_label']            = xgb_output_df.loc[insert_at:, 'pred_label'].values
    df.loc[insert_at:, 'pred_class_confidence'] = xgb_output_df.loc[insert_at:, 'pred_class_confidence'].values


    df = df.sort_values(['timestamp', 'item']).reset_index(drop=True)

    # Replace missing/inf values that might destabilise RL training
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)  

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"RL dataset saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
