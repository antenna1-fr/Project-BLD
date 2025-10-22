import torch, math
import torch.nn as nn
import torch.nn.functional as F
from src.configs.encoder_config import ModelCfg, FeatCfg

class Patchify(nn.Module):
    def __init__(self, patch_size: int): super().__init__(); self.p = patch_size
    def forward(self, x):  # x: [B, T, F]
        B, T, F = x.shape
        assert T % self.p == 0
        L = T // self.p
        return x.view(B, L, self.p, F)  # [B, L, P, F]

class BazaarEncoder(nn.Module):
    def __init__(self, cfgM: ModelCfg, cfgF: FeatCfg, item_vocab=2000):
        super().__init__()
        self.patcher = Patchify(cfgF.patch_size)
        self.item_emb = nn.Embedding(item_vocab, cfgM.item_emb)
        self.time_mlp = nn.Sequential(nn.Linear(cfgM.time_emb, cfgM.d_model), nn.ReLU(), nn.Dropout(cfgM.dropout))
        self.patch_proj = nn.Linear(cfgF.patch_size*cfgM.feat_dim + cfgM.item_emb, cfgM.d_model)
        self.cls = nn.Parameter(torch.zeros(1,1,cfgM.d_model))
        self.pos = None
        enc_layer = nn.TransformerEncoderLayer(d_model=cfgM.d_model, nhead=cfgM.heads,
                                               dim_feedforward=cfgM.d_model*cfgM.mlp_ratio, dropout=cfgM.dropout,
                                               batch_first=True, activation="gelu", norm_first=True)
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=cfgM.depth)
        self.norm = nn.LayerNorm(cfgM.d_model)
        self.proj = nn.Linear(cfgM.d_model, cfgM.latent_dim)        # global embedding (contrastive)
        self.recon = nn.Sequential(                                 # per-patch reconstruction head
            nn.Linear(cfgM.d_model, cfgM.d_model), nn.GELU(),
            nn.Linear(cfgM.d_model, cfgF.patch_size*cfgM.feat_dim)
        )
        # Optional auxiliary forecasting head for k-step Δmid (plug-in later if desired)
        self.delta_head = nn.Sequential(nn.Linear(cfgM.d_model, cfgM.d_model), nn.GELU(),
                                        nn.Linear(cfgM.d_model, 4))  # k in {1,5,15,60}

        self.cfgM, self.cfgF = cfgM, cfgF

    def _pos(self, L, d, device):
        pe = torch.zeros(1, L+1, d, device=device)
        pos = torch.arange(L+1, device=device).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2, device=device) * (-math.log(10000.0) / d))
        pe[:,:,0::2] = torch.sin(pos*div)
        pe[:,:,1::2] = torch.cos(pos*div)
        return pe

    def forward(self, x, item_ids, time_feats, mask_bool=None):
        """
        x: [B,T,F], item_ids: [B], time_feats: [B,time_emb], mask_bool: [B,L] True=masked
        """
        B, T, F = x.shape
        patches = self.patcher(x)                       # [B,L,P,F]
        B, L, P, F = patches.shape
        item_e = self.item_emb(item_ids).unsqueeze(1).expand(B, L, -1)  # [B,L,E]
        tok = patches.reshape(B, L, P*F)
        tok = torch.cat([tok, item_e], dim=-1)          # [B,L,P*F+E]
        tok = self.patch_proj(tok)                      # [B,L,d]

        if mask_bool is not None:
            tok = tok.masked_fill(mask_bool.unsqueeze(-1), 0.0)

        cls = self.cls.expand(B, -1, -1)                # [B,1,d]
        seq = torch.cat([cls, tok], dim=1)              # [B,L+1,d]
        if (self.pos is None) or (self.pos.size(1)!=(L+1)) or (self.pos.size(2)!=seq.size(-1)):
            self.pos = self._pos(L, seq.size(-1), seq.device)
        seq = seq + self.pos
        seq[:,0,:] += self.time_mlp(time_feats)         # context at CLS

        h = self.tr(seq)                                # [B,L+1,d]
        h = self.norm(h)
        cls_h, patch_h = h[:,0], h[:,1:]

        z_global = F.normalize(self.proj(cls_h), dim=-1)    # [B,latent]
        recon = self.recon(patch_h)                          # [B,L,P*F]
        delta = self.delta_head(cls_h)                       # [B,4] optional
        return z_global, recon, delta, L, P, F
