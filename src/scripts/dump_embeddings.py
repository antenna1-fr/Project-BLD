import torch, pandas as pd, json
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.configs.encoder_config import DataCfg, FeatCfg, ModelCfg, Paths
from src.data.ssl_window_dataset import SSLWindowDataset
from src.models.bazaar_encoder import BazaarEncoder

@torch.no_grad()
def dump():
    cfgD, cfgF, cfgM, paths = DataCfg(), FeatCfg(), ModelCfg(), Paths()
    ds = SSLWindowDataset(parquet_path=cfgD.features_parquet, T=cfgF.window_T, feat_cols=list(cfgF.feature_list),
                          id_map_path=paths.id_map_json if True else None)
    item_vocab = len(ds.item2id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BazaarEncoder(cfgM, cfgF, item_vocab=item_vocab).to(device).eval()
    model.load_state_dict(torch.load(paths.ckpt_out, map_location=device))

    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)
    Z, meta = [], []
    for x, item_id, tfeat in tqdm(dl, desc="embeddings"):
        z, _, _, _, _, _ = model(x.to(device), item_id.to(device), tfeat.to(device), mask_bool=None)
        Z.append(z.cpu())
    Z = torch.cat(Z).numpy()

    # reconstruct metadata (item, end_timestamp) from dataset indices
    items, tss = [], []
    df = pd.read_parquet(cfgD.features_parquet)
    for it, idxs in ds.index:
        items.append(it)
        tss.append(df.loc[idxs[-1], cfgD.ts_col])
    meta = pd.DataFrame({"item": items, "timestamp": tss})
    emb = pd.DataFrame(Z, columns=[f"enc_{i}" for i in range(Z.shape[1])])
    out = pd.concat([meta.reset_index(drop=True), emb.reset_index(drop=True)], axis=1)
    out.to_parquet(Paths().embeddings_out, index=False)
    print(f"[dump_embeddings] wrote {Paths().embeddings_out} shape={out.shape}")

if __name__ == "__main__":
    dump()
