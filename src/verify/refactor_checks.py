# src/verify/refactor_checks.py
"""Quick verification checks for refactor phases 4-7.

Runs lightweight checks that don't require the full processed CSV dataset.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.data.sequence_dataset import build_item_sequence_indices, SequenceDataset
from src.models.seq.tcn_model import TCNSequenceModel
from src.backtest.engine import run_backtest
from src.models.registry import list_models, load_named_model


def seq_dataset_check():
    print('== SequenceDataset check ==')
    rows = []
    for item in ['A','B']:
        for t in range(12):
            rows.append({'item': item, 'timestamp': float(t), 'mid_price': 100.0 + t, 'f0': float(t%5), 'f1': float((t*2)%7), 'tradable':1})
    df = pd.DataFrame(rows)
    y_dir = np.array([1 if int(r['timestamp'])%7==6 else 0 for r in rows], dtype=np.int8)
    LABEL_MAP = {-1:0, 0:1, 1:2}
    y_enc = np.array([LABEL_MAP[int(v)] for v in y_dir], dtype=np.int8)
    end_all, end_tr, end_val, item_to_pos = build_item_sequence_indices(df, y_enc, window=5, stride=2, val_mask=None, verbose=True)
    assert len(end_all) > 0
    X_df = df[['f0','f1']].astype('float32')
    meta = df[['timestamp','item','mid_price']]
    ds = SequenceDataset(X_df, y_enc, meta, end_tr, window=5)
    x,y,ts,it,mid = ds[0]
    assert x.shape == (5,2)
    print('SequenceDataset check passed')


def tcn_smoke_check():
    print('\n== TCN smoke check ==')
    rows = []
    for t in range(40):
        rows.append({'item': 'X', 'timestamp': float(t), 'mid_price': 100.0 + t, 'f0': float(t%5), 'f1': float((t*2)%7), 'tradable':1})
    df = pd.DataFrame(rows)
    y_dir = np.array([1 if int(r['timestamp'])%7==6 else 0 for r in rows], dtype=np.int8)
    LABEL_MAP = {-1:0, 0:1, 1:2}
    y_enc = np.array([LABEL_MAP[int(v)] for v in y_dir], dtype=np.int8)
    config = {'nfeat':2, 'feature_cols':['f0','f1'], 'window':8, 'channels':[16,16], 'kernel_size':3, 'dropout':0.1, 'num_classes':3}
    model = TCNSequenceModel(config)
    model.fit(df=df, y_enc=y_enc, feature_cols=['f0','f1'], window=8, stride=2, epochs=1, batch_size=8, lr=1e-3, verbose=False)
    out = model.predict(df=df, feature_cols=['f0','f1'], window=8, stride=2)
    assert 'pred_label' in out.columns and 'pred_proba_buy' in out.columns
    print('TCN smoke check passed (predictions produced)')


def backtest_simple_check():
    print('\n== Backtest simple check ==')
    # Build a tiny timeline where price rises; place a buy signal at t=0 so it executes at t=1
    ts = np.array([0.0, 60.0, 120.0, 180.0])
    rows = []
    for i,t in enumerate(ts):
        rows.append({'timestamp': float(t), 'item': 'P', 'mid_price': 100.0 + i*1.0, 'tradable':1, 'pred_label': np.nan, 'pred_proba_buy': np.nan})
    df = pd.DataFrame(rows)
    # Put a buy decision at t=0 (encoded 2), it will execute at t=60
    df.loc[0, 'pred_label'] = 2
    df.loc[0, 'pred_proba_buy'] = 0.95
    trades, summary, equity = run_backtest(df, verbose=False)
    print('Backtest summary:', json.dumps(summary, indent=2))
    print('Backtest simple check completed')


def registry_check():
    print('\n== Registry check ==')
    print('Registered models:', list_models())
    try:
        m = load_named_model(list_models()[0])
        print('Loaded model type:', type(m).__name__)
    except Exception as e:
        print('Registry load (expected if artifacts missing):', e)
    print('Registry check done')


if __name__ == '__main__':
    seq_dataset_check()
    tcn_smoke_check()
    backtest_simple_check()
    registry_check()
    print('\nAll refactor checks completed')

