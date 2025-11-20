from dataclasses import dataclass

@dataclass
class DataCfg:
    parquet_path: str = "data/bazaar_minutely.parquet"  # your minutely table
    # columns present in raw data (rename here if needed)
    price_cols: tuple = ("best_bid","best_ask","mid_price")
    volume_col: str = "volume_1m"        # trades or filled volume per minute
    trades_col: str = "trades_1m"        # optional
    item_col: str = "item"
    ts_col: str = "timestamp"            # epoch seconds
    # engineered features saved to:
    features_parquet: str = "data/bazaar_features.parquet"

@dataclass
class FeatCfg:
    window_T: int = 720            # 12h @ 1-min
    patch_size: int = 8
    feature_list: tuple = (
        "mid_price","best_bid","best_ask","spread",
        "log_ret_1m","log_ret_5m","log_ret_15m",
        "vol_1m","trades_1m","oi_proxy","rv_15m","rv_60m",
    )
    # time encodings will be auto-added: sin_tod, cos_tod, dow_0..dow_6

@dataclass
class ModelCfg:
    feat_dim: int = 12            # will be validated against feature_list
    d_model: int = 256
    depth: int = 6
    heads: int = 8
    mlp_ratio: int = 4
    latent_dim: int = 128
    item_emb: int = 48
    time_emb: int = 16
    dropout: float = 0.1

@dataclass
class TrainCfg:
    epochs: int = 8
    batch_size: int = 128          # effective; use grad_accum if VRAM-limited
    lr: float = 3e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 2000
    cosine_final_lr: float = 3e-5
    grad_clip: float = 1.0
    tau: float = 0.1               # InfoNCE temperature
    mask_pct: float = 0.2
    mask_span: int = 2
    num_workers: int = 8
    fp16: bool = True
    log_dir: str = "runs/encoder_pretrain"

@dataclass
class Paths:
    ckpt_out: str = "checkpoints/bazaar_encoder.pt"
    id_map_json: str = "data/item2id.json"
    embeddings_out: str = "data/embeddings_12h_latent128.parquet"


