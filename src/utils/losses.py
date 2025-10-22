import torch
import torch.nn.functional as F

def info_nce(z1, z2, tau=0.1):
    z = torch.cat([z1, z2], dim=0)                # [2B,D], assume normalized
    sim = torch.matmul(z, z.t()) / tau            # cosine sim if normalized
    B = z1.size(0)
    labels = torch.arange(B, device=z.device)
    labels = torch.cat([labels, labels], dim=0)
    mask = torch.eye(2*B, device=z.device).bool()
    sim.masked_fill_(mask, -1e9)                  # remove self-similarity
    loss = F.cross_entropy(sim, labels)
    return loss
