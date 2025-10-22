import math

def cosine_warmup(step, warmup, total, base_lr, final_lr):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return final_lr + (base_lr - final_lr) * cosine
