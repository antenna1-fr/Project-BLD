import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

# Location to insert in your pipeline right after you create labels:
# Load engineered features+labels and precomputed embeddings
LABELS_PARQUET = "data/labels.parquet"   # must contain (item, timestamp, label)
EMB_PARQUET    = "data/embeddings_12h_latent128.parquet"
FEAT_PARQUET   = "data/bazaar_features.parquet"

def time_split(df, cut_ts):
    tr = df[df["timestamp"] <= cut_ts].copy()
    te = df[df["timestamp"] >  cut_ts].copy()
    return tr, te

def run():
    feats = pd.read_parquet(FEAT_PARQUET)
    embs  = pd.read_parquet(EMB_PARQUET)
    ylab  = pd.read_parquet(LABELS_PARQUET)
    df = (embs.merge(ylab, on=["item","timestamp"])
              .merge(feats[["item","timestamp","spread","vol_1m","rv_60m"]], on=["item","timestamp"]))
    # simple time split at 80%
    cut = df["timestamp"].quantile(0.8)
    tr, te = time_split(df, cut)

    X_emb_tr = tr.filter(like="enc_");  X_emb_te = te.filter(like="enc_")
    X_man_tr = tr[["spread","vol_1m","rv_60m"]]; X_man_te = te[["spread","vol_1m","rv_60m"]]
    X_hyb_tr = pd.concat([X_emb_tr, X_man_tr], axis=1)
    X_hyb_te = pd.concat([X_emb_te, X_man_te], axis=1)
    y_tr, y_te = tr["label"], te["label"]

    # Logistic probe on embeddings
    lr = LogisticRegression(max_iter=1000, n_jobs=8)
    lr.fit(X_emb_tr, y_tr)
    print("\n=== Linear Probe (Embeddings) ===")
    print(classification_report(y_te, lr.predict(X_emb_te), digits=3))

    # XGB manual vs hybrid
    xgb_m = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", n_jobs=8)
    xgb_m.fit(X_man_tr, y_tr)
    print("\n=== XGB (Manual) ===")
    print(classification_report(y_te, xgb_m.predict(X_man_te), digits=3))

    xgb_h = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", n_jobs=8)
    xgb_h.fit(X_hyb_tr, y_tr)
    print("\n=== XGB (Hybrid) ===")
    print(classification_report(y_te, xgb_h.predict(X_hyb_te), digits=3))

if __name__ == "__main__":
    run()
