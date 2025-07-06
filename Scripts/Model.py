"""
xgb_train_top100_gpu_es.py ― XGBoost pipeline with
 • volume/price filtering ➜ top-100 items  
 • class-weighted cost-sensitive learning  
 • early-stopping on a rolling validation window  
 • GPU acceleration
 • SHAP model-explainability report

─────────────────────────────────────────────────────────────────────────────
Prereqs
-------
pip install -U xgboost shap scikit-learn matplotlib pandas numpy

-------------------
Columns:  
  • item (str)                         – Bazaar item name  
  • timestamp (int/float)              – epoch seconds  
  • mid_price (float)                  – mid-point price [coins]  
  • moving_weekly (float) or volume    – 7-day moving trade volume  
  • label   (int)  {-1:sell, 0:hold, 1:buy}

"""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path
import xgboost as xgb
import matplotlib.pyplot as plt
sys.path.append(str(Path(__file__).resolve().parent.parent)) # repo root\n
import config as config  # must live in PYTHONPATH or same dir
import numpy as np
import pandas as pd
import shap
from packaging import version
import optuna
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping,EvaluationMonitor

# ─────────────────────────── project paths ──────────────────────────────
XGB_VER = version.parse(xgb.__version__)
print(f"Using XGBoost version:", XGB_VER)
CSV_PATH = str(config.PROCESSED_CSV)
OUTPUT_PLOT = str(config.CONFUSION_MATRIX_PLOT)
OUTPUT_PREDICTIONS = str(config.PREDICTIONS_CSV)
OUTPUT_SHAP = str(config.OUTPUTS_DIR / "shap_feature_importance.csv")

TEST_SIZE = 0.20
RANDOM_STATE = 42
TOP_N = 100

# ────────────────── 1) LOAD & DROP RECENT ITEMS ───────────────────
df = pd.read_csv(CSV_PATH)

# Sort chronologically; drop final 400 snapshots per item to avoid leakage
df = df.sort_values(["item", "timestamp"]).reset_index(drop=True)
df = (
    df.groupby("item")
    .apply(lambda g: g.iloc[:-1440] if len(g) > 1440 else g.iloc[0:0])
    .reset_index(drop=True)
)
df = df.sort_values("timestamp").reset_index(drop=True)

# Down-cast to float32 for memory efficiency
float_cols = df.select_dtypes("float64").columns
df[float_cols] = df[float_cols].astype("float32")

item_series = df["item"].copy()  # keep original for sim output
df = pd.get_dummies(df, columns=["item"], prefix="item")

# ────────────────── 2) FEATURE MATRIX / LABEL SPLIT ─────────────────────
drop_cols = {"timestamp", "datetime", "label"}
feature_cols = [c for c in df.columns if c not in drop_cols]

X_all = df[feature_cols]
y_all = df["label"]

split_idx = int(len(df) * (1 - TEST_SIZE))
X_train_full, X_test = X_all.iloc[:split_idx], X_all.iloc[split_idx:]
y_train_full, y_test = y_all.iloc[:split_idx], y_all.iloc[split_idx:]

# Reserve the last 10 % of the training chunk for early-stopping validation
val_split = int(len(X_train_full) * 0.9)
X_train, X_val = X_train_full.iloc[:val_split], X_train_full.iloc[val_split:]
y_train, y_val = y_train_full.iloc[:val_split], y_train_full.iloc[val_split:]

# ────────────────── 3) OVERSAMPLE + COST-SENSITIVE WEIGHTS ──────────────
train_df = pd.concat([X_train, y_train.rename("label")], axis=1)
df_hold = train_df[train_df["label"] == 0]
df_buy = train_df[train_df["label"] == 1]
df_sell = train_df[train_df["label"] == -1]

train_bal = (
    pd.concat([df_hold, df_buy, df_sell]).sample(frac=1, random_state=RANDOM_STATE)
)

X_tr_bal = train_bal[feature_cols].fillna(0).astype("float32")
y_tr_bal = train_bal["label"]

# Cost-sensitive sample weights: inverse frequency + manual penalty boost
class_counts = y_tr_bal.value_counts().to_dict()
base_weights = {0: 1, -1: 1, 1: 1}
# Extra 2× penalty to SELL mis-classifications (business-driven)
base_weights[-1] *= 2.0
sample_weights = y_tr_bal.map(base_weights).values

# ────────────────── 4) LABEL ENCODING {-1,0,1} → {0,1,2} ───────────────
le = LabelEncoder()
y_tr_enc = le.fit_transform(y_tr_bal)
y_val_enc = le.transform(y_val)

LABEL2IDX = {lbl: idx for idx, lbl in enumerate(le.classes_)}
IDX2LABEL = {idx: lbl for lbl, idx in LABEL2IDX.items()}

# ────────────────── 5) XGB CLASSIFIER (GPU if available) ────────────────

# A) Decide tree_method safely
def choose_tree_method() -> tuple[str, str]:
    try:
        # Works on xgboost 2.x with CUDA build
        if xgb.cuda.is_cuda_array_available():
            return "gpu_hist", "gpu_predictor"
    except Exception:  # pragma: no cover
        pass
    return "hist", "auto"

tree_method, predictor = choose_tree_method()
print(f"🚀  Training with tree_method={tree_method}")

# B) Objective Definition

def objective(trial):
    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "tree_method": tree_method,
        "predictor": predictor,
        "learning_rate": trial.suggest_float("learning_rate", 0.08, 0.13),
        "max_depth": trial.suggest_int("max_depth", 6, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 3, 7),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", .5, 2),
        "alpha": trial.suggest_float("alpha", 0.0, 1),
        "eval_metric": "mlogloss",
        "n_estimators": 750,
        "seed": RANDOM_STATE,
    }

    model = XGBClassifier(**params)
    model.fit(
        X_tr_bal,
        y_tr_enc,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val_enc)],
        verbose=False
    )

    preds = model.predict(X_val)
    f1 = f1_score(y_val_enc, preds, average="macro")
    return f1
if __name__ == "__main__" and config.ENABLE_OPTUNA:
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15) 

    print("✅ Best params:", study.best_params)
    print("🎯 Best macro-F1:", study.best_value)
    best_params = study.best_params 
else:
    best_params = {'learning_rate': 0.11592016476593922, 'max_depth': 7, 'min_child_weight': 5, 'subsample': 0.9439252674820234, 'colsample_bytree': 0.7658074839270397, 'gamma': 1.045760656283378, 'alpha': 0.028667196702399807}

# C) Build model


best_params.update({
    "objective": "multi:softprob",
    "num_class": 3,
    "tree_method": tree_method,
    "predictor": predictor,
    "eval_metric": "mlogloss",
    "n_estimators": 500,
    "early_stopping_rounds": 150,
    "seed": RANDOM_STATE,
})

xgb_model = XGBClassifier(**best_params)
import inspect

print("XGBoost version:", xgb.__version__)
print("Model type:", type(xgb_model))
print("Model class location:", inspect.getfile(type(xgb_model)))
print("Has 'callbacks' in fit signature?:", 'callbacks' in inspect.signature(xgb_model.fit).parameters)

xgb_model.fit(
    X_tr_bal,
    y_tr_enc,
    sample_weight=sample_weights,
    eval_set=[(X_val, y_val_enc)],
    verbose=200
)
print("✅  Best boosting round:", xgb_model.best_iteration)


# ────────────────── 6) PREDICTION WITH CONF. THRESHOLD ──────────────────
X_test_num   = X_test.fillna(0).astype("float32")
y_pred_prob  = xgb_model.predict_proba(X_test_num)

# plain arg-max in probability space
raw_pred_enc = np.argmax(y_pred_prob, axis=1)
y_pred       = le.inverse_transform(raw_pred_enc)

# ----------------------------------------------------------------------
# OPTIONAL: add a confidence filter *after* arg-max
# (leave threshold = None to disable)
threshold = None            # e.g. 0.40 to force min confidence
if threshold is not None:
    forced = []
    for i, p in enumerate(y_pred_prob):
        if p[raw_pred_enc[i]] < threshold:
            forced.append(LABEL2IDX[0])        # fallback to HOLD
        else:
            forced.append(raw_pred_enc[i])
    y_pred = le.inverse_transform(forced)

print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=4))

# ────────────────── 7) CONFUSION MATRIX PLOT ────────────────────────────
labels = [-1, 0, 1]
cm = confusion_matrix(y_test, y_pred, labels=labels)
fig, ax = plt.subplots(figsize=(5, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set(
    xticks=range(len(labels)),
    yticks=range(len(labels)),
    xticklabels=labels,
    yticklabels=labels,
    xlabel="Predicted",
    ylabel="Actual",
    title="XGBoost Confusion Matrix",
)
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            color="white" if cm[i, j] > cm.max() / 2 else "black",
        )
plt.colorbar(im)
plt.tight_layout()
os.makedirs(os.path.dirname(OUTPUT_PLOT), exist_ok=True)
plt.savefig(OUTPUT_PLOT)
plt.show()
print(f"📈  Confusion matrix saved → {OUTPUT_PLOT}")

# ────────────────── 9) SIMULATION CSV OUTPUT ────────────────────────────
df_sim = pd.DataFrame(
    {
        "timestamp": df.iloc[y_test.index]["timestamp"].values,
        "item": item_series.iloc[y_test.index].values,
        "mid_price": df.iloc[y_test.index]["mid_price"].values,
        "true_label": y_test.values,
        "pred_label": y_pred,
        "pred_proba_buy": y_pred_prob[:, LABEL2IDX[1]],
        "pred_proba_sell": y_pred_prob[:, LABEL2IDX[-1]],
        "pred_proba_hold": y_pred_prob[:, LABEL2IDX[0]],
        "pred_class_confidence": y_pred_prob.max(axis=1),
    }
)
os.makedirs(os.path.dirname(OUTPUT_PREDICTIONS), exist_ok=True)
df_sim.to_csv(OUTPUT_PREDICTIONS, index=False)
print(f"💾  Prediction CSV saved → {OUTPUT_PREDICTIONS}")

# ────────────────── 10) CLEAN-UP  ───────────────────────────────────────
del X_all, X_train_full, X_train, X_val, X_test
gc.collect()
