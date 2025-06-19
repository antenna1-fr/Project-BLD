import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import gc
import sys
from pathlib import Path

# project configuration
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config

from sklearn.model_selection import TimeSeriesSplit,GridSearchCV,train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, f1_score

# === CONFIG ===
CSV_PATH       = str(config.PROCESSED_CSV)
TEST_SIZE      = 0.20
RANDOM_STATE   = 42
OUTPUT_PLOT    = str(config.CONFUSION_MATRIX_PLOT)
OUTPUT_PREDICTIONS = str(config.PREDICTIONS_CSV)

# -----------------------------------------------------------------------------
# 1) LOAD & PREPARE DATA
# -----------------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
df = df.sort_values(['item', 'timestamp']).reset_index(drop=True)

# Remove last 40 snapshots per item
df = df.groupby('item').apply(lambda g: g.iloc[:-40] if len(g) > 40 else g.iloc[0:0]).reset_index(drop=True)
df = df.sort_values('timestamp').reset_index(drop=True)

# keep item labels for later use in the simulation
item_series = df['item']

# one-hot encode items for the model
df = pd.get_dummies(df, columns=['item'], prefix='item')

# define features & label
drop_cols   = {'timestamp', 'datetime', 'label'}
feature_cols = [
    c for c in df.columns
    if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])
]
X = df[feature_cols]
y = df['label']


# time-based 80/20 split
split_idx      = int(len(df) * (1 - TEST_SIZE))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]


print("TRAIN shape:", X_train.shape, y_train.shape)
print(" TEST shape:", X_test.shape, y_test.shape)

print("TRAIN dist:\n", y_train.value_counts(normalize=True))
print(" TEST dist:\n", y_test.value_counts(normalize=True))

# -----------------------------------------------------------------------------
# 2) OVERSAMPLE TRAINING MINORITIES
# -----------------------------------------------------------------------------
train_df = pd.concat([X_train, y_train.rename('label')], axis=1)
df_hold  = train_df[train_df['label'] == 0]    # HOLD is label == 0
df_buy   = train_df[train_df['label'] == 1]    # BUY   is label == 1
df_sell  = train_df[train_df['label'] == -1]   # SELL  is label == -1

n_hold = len(df_hold)
target = int(n_hold * 0.25)   # e.g. make buy & sell each 0.25x the size of hold
df_buy_os  = resample(df_buy,  replace=True, n_samples=target, random_state=RANDOM_STATE)
df_sell_os = resample(df_sell, replace=True, n_samples=target, random_state=RANDOM_STATE)
train_bal  = pd.concat([df_hold, df_buy_os, df_sell_os]) \
                 .sample(frac=1, random_state=RANDOM_STATE)

X_tr_bal = train_bal[feature_cols]
y_tr_bal = train_bal['label']

print("\nBALANCED TRAIN dist:\n", y_tr_bal.value_counts(normalize=True))

# -----------------------------------------------------------------------------
# 3) LABEL-ENCODE for XGBoost (must be 0,1,2)
# -----------------------------------------------------------------------------
le      = LabelEncoder()
y_tr_enc = le.fit_transform(y_tr_bal)    # maps {-1,0,1} → {0,1,2}

# ----------------------------------------------------------------------
# 4) TimeSeriesSplit + Hyperparameter Tuning with GridSearchCV
# ----------------------------------------------------------------------

# Use balanced training data
X_bal_num = X_tr_bal.fillna(0).astype('float32')
y_bal_enc = le.fit_transform(y_tr_bal)  # Label encoding if not done yet

# Define time series split (e.g. 5 folds)
tscv = TimeSeriesSplit(n_splits=5)

# XGBClassifier wrapper (scikit-learn compatible)
xgb_clf = XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    tree_method='hist',
    seed=RANDOM_STATE,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

# Define param grid
param_grid = {
    'learning_rate': [.12, .13],
    'max_depth': [10,11],
    'subsample': [.75],
    'colsample_bytree': [0.8],
    'n_estimators': [250]  # keep high for early stopping
}

# GridSearchCV with macro F1 as scoring
grid_search = GridSearchCV(
    estimator=xgb_clf,
    param_grid=param_grid,
    cv=tscv,
    scoring=make_scorer(f1_score, average='macro'),
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_bal_num, y_bal_enc)
best_model = grid_search.best_estimator_
print(f"\n✅ Best params: {grid_search.best_params_}")

# ----------------------------------------------------------------------
# 5) Predict with Thresholding
# ----------------------------------------------------------------------

X_test_num = X_test.fillna(0).astype('float32')
y_pred_prob = best_model.predict_proba(X_test_num)

# Get encoded index for labels
idx_buy  = np.where(le.classes_ == 1)[0][0]
idx_sell = np.where(le.classes_ == -1)[0][0]
idx_hold = np.where(le.classes_ == 0)[0][0]

# Apply confidence threshold
threshold = 0.6
y_pred_enc = []
for prob in y_pred_prob:
    if prob[idx_buy] > threshold:
        y_pred_enc.append(idx_buy)
    elif prob[idx_sell] > threshold:
        y_pred_enc.append(idx_sell)
    else:
        y_pred_enc.append(idx_hold)

# Decode back to {-1, 0, 1}
y_pred = le.inverse_transform(y_pred_enc)


print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, digits=4))

# -----------------------------------------------------------------------------
# 6) CONFUSION MATRIX PLOT
# -----------------------------------------------------------------------------
labels = [-1, 0, 1]
cm     = confusion_matrix(y_test, y_pred, labels=labels)

fig, ax = plt.subplots(figsize=(5,5))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('XGBoost Confusion Matrix')

for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, cm[i, j],
                ha='center', va='center',
                color='white' if cm[i, j] > cm.max()/2 else 'black')

plt.colorbar(im)
plt.tight_layout()
os.makedirs(os.path.dirname(OUTPUT_PLOT), exist_ok=True)
plt.savefig(OUTPUT_PLOT)
plt.show()
print(f"\nConfusion matrix plot saved to {OUTPUT_PLOT}")

# -----------------------------------------------------------------------------
# 7) Prepare Simulation Inputs
# -----------------------------------------------------------------------------

# Ensure mid_price is present in test set
if 'mid_price' not in df.columns:
    raise ValueError("mid_price column missing from dataset — required for simulation.")

# Match index with y_test
df_sim = pd.DataFrame({
    'timestamp': df.iloc[y_test.index]['timestamp'].values,
    'item': item_series.iloc[y_test.index].values,
    'mid_price': df.iloc[y_test.index]['mid_price'].values,
    'true_label': y_test.values,
    'pred_label': y_pred,
    'pred_proba_buy': y_pred_prob[:, le.transform([1])[0]],
    'pred_proba_sell': y_pred_prob[:, le.transform([-1])[0]],
    'pred_proba_hold': y_pred_prob[:, le.transform([0])[0]],
})

# Optional: add highest class confidence and predicted action
df_sim['pred_class_confidence'] = y_pred_prob.max(axis=1)
df_sim['pred_class_name'] = y_pred

# Preview
print(f"\n Simulator Input Preview Saved to {OUTPUT_PREDICTIONS}")

df_sim.to_csv(OUTPUT_PREDICTIONS, index=False)
