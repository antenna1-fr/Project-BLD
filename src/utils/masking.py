import torch, random

def span_mask(B, L, pct=0.2, span=2, device="cuda"):
    mask = torch.zeros(B, L, dtype=torch.bool, device=device)
    m = int(L * pct)
    for b in range(B):
        i = 0
        while i < m:
            s = random.randint(0, max(0, L-span))
            mask[b, s:s+span] = True
            i += span
    return mask
