import json, math, random
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from src.configs.encoder_config import DataCfg, FeatCfg, ModelCfg, TrainCfg, Paths
from src.data.ssl_window_dataset import SSLWindowDataset
from src.models.bazaar_encoder import BazaarEncoder
from src.utils.masking import span_mask
from src.utils.losses import info_nce
from src.utils.schedule import cosine_warmup

def make_aug(x):
    # weak augmentations: gaussian noise + contiguous time dropout
    noise = 0.01 * torch.randn_like(x)
    x_aug = x + noise
    # time dropout: blank a small span to encourage temporal reasoning
    B,T,F = x_aug.shape
    span = max(1, int(0.1*T))
    for b in range(B):
        s = random.randint(0, max(0, T-span))
        x_aug[b, s:s+span, :] = 0.0
    return x_aug

def pretrain():
    cfgD, cfgF, cfgM, cfgT, paths = DataCfg(), FeatCfg(), ModelCfg(), TrainCfg(), Paths()
    # sanity check feat_dim
    assert cfgM.feat_dim == len(cfgF.feature_list), f"ModelCfg.feat_dim={cfgM.feat_dim} but feature_list has {len(cfgF.feature_list)}."

    dataset = SSLWindowDataset(parquet_path=cfgD.features_parquet, T=cfgF.window_T, feat_cols=list(cfgF.feature_list),
                               id_map_path=paths.id_map_json if False else None)
    item_vocab = len(set(dataset.item2id.values()))
    model = BazaarEncoder(cfgM, cfgF, item_vocab=item_vocab)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    writer = SummaryWriter(cfgT.log_dir)

    dl = DataLoader(dataset, batch_size=cfgT.batch_size//8, shuffle=True, num_workers=cfgT.num_workers, pin_memory=True)
    # gradient accumulation to reach effective batch size
    accum = 8
    steps_per_epoch = math.ceil(len(dataset) / (len(dl.dataset) if hasattr(dl,'dataset') else 1))  # not used; TB uses global_step
    opt = torch.optim.AdamW(model.parameters(), lr=cfgT.lr, weight_decay=cfgT.weight_decay)
    scaler = GradScaler(enabled=cfgT.fp16)

    global_step = 0
    total_steps = cfgT.epochs * len(dl) // 1

    for ep in range(cfgT.epochs):
        model.train()
        pbar = tqdm(dl, desc=f"epoch {ep+1}/{cfgT.epochs}")
        accum_loss = 0.0

        for step, batch in enumerate(pbar):
            x, item_ids, tfeat = [b.to(device) for b in batch]
            x1, x2 = make_aug(x), make_aug(x)
            L = cfgF.window_T // cfgF.patch_size
            mask = span_mask(x.size(0), L, pct=cfgT.mask_pct, span=cfgT.mask_span, device=device)

            lr = cosine_warmup(global_step, cfgT.warmup_steps, total_steps, cfgT.lr, cfgT.cosine_final_lr)
            for pg in opt.param_groups: pg["lr"] = lr

            with autocast(enabled=cfgT.fp16):
                z1, recon1, delta1, L_, P, F = model(x1, item_ids, tfeat, mask_bool=mask)
                z2, recon2, delta2, _, _, _   = model(x2, item_ids, tfeat, mask_bool=mask)

                # contrastive on globals
                l_con = info_nce(z1, z2, tau=cfgT.tau)

                # masked reconstruction (only where mask=True)
                target = x.view(x.size(0), L_, P, F)
                rec1 = recon1.view(x.size(0), L_, P, F)
                rec2 = recon2.view(x.size(0), L_, P, F)
                m4 = mask.unsqueeze(-1).unsqueeze(-1)
                l_rec = nn.functional.smooth_l1_loss(rec1[m4], target[m4]) + \
                        nn.functional.smooth_l1_loss(rec2[m4], target[m4])

                # optional tiny forecasting regularizer on CLS (use Δ mid proxies if you later add them)
                l_for = 0.0

                loss = 1.0*l_con + 0.5*l_rec + 0.25*l_for

            scaler.scale(loss/accum).backward()
            accum_loss += loss.item()

            if (step+1) % accum == 0:
                if cfgT.grad_clip:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), cfgT.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

                # logging
                writer.add_scalar("loss/total", accum_loss, global_step)
                writer.add_scalar("loss/contrastive", l_con.item(), global_step)
                writer.add_scalar("loss/recon", l_rec.item(), global_step)
                writer.add_scalar("train/lr", lr, global_step)
                accum_loss = 0.0
                global_step += 1

            pbar.set_postfix({"l_con": f"{l_con.item():.3f}", "l_rec": f"{l_rec.item():.3f}", "lr": f"{lr:.2e}"})

        # checkpoint per epoch
        torch.save(model.state_dict(), paths.ckpt_out)
        print(f"[pretrain] saved: {paths.ckpt_out}")

    writer.close()
    return paths.ckpt_out

if __name__ == "__main__":
    pretrain()
