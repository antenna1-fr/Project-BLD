import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as config

INPUT_CSV        = config.PROCESSED_CSV
OUTPUT_CSV       = config.RL_DATASET_CSV
PREDICTIONS_CSV  = config.PREDICTIONS_CSV

def main():
    # 1) Load & clean the base RL DataFrame
    print(f"→ Reading {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    df.drop(columns=['label'], errors='ignore', inplace=True)
    df.sort_values(['timestamp','item'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 2) Load & check the predictions
    print(f"→ Reading {PREDICTIONS_CSV}")
    preds = pd.read_csv(PREDICTIONS_CSV)
    required = {'timestamp','item','pred_label','pred_class_confidence'}
    if not required.issubset(preds.columns):
        raise ValueError(f"PREDICTIONS_CSV must contain {required}")
    preds.sort_values(['timestamp','item'], inplace=True)
    preds = preds[list(required)]

    # 3) Merge & fill
    print("→ Merging predictions")
    df = df.merge(preds, on=['timestamp','item'], how='left')
    df['pred_label'] = df['pred_label'].fillna(0).astype(int)
    df['pred_class_confidence'] = df['pred_class_confidence'].fillna(0.0)

    # 4) Cleanup & save
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ RL dataset saved to {OUTPUT_CSV} "
          f"({df['pred_label'].astype(bool).sum()} non-zero predictions)")

if __name__ == "__main__":
    main()
