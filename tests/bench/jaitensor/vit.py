"""One transformer block forward, the peer for vit.jai. Same shapes, same warm-up rule."""
import os
import time

import torch
import torch.nn.functional as F

WARMUP = 24


def block(x, qw, ow, w1, w2, g1, b1, g2, b2, heads):
    normed = F.layer_norm(x, (x.shape[-1],), g1, b1)
    q = normed @ qw
    seq, dim = q.shape
    hd = dim // heads
    qh = q.view(seq, heads, hd).permute(1, 0, 2).unsqueeze(0).contiguous()
    ctx = F.scaled_dot_product_attention(qh, qh, qh)
    attended = ctx.squeeze(0).permute(1, 0, 2).reshape(seq, dim)
    residual = x + attended @ ow
    normed2 = F.layer_norm(residual, (residual.shape[-1],), g2, b2)
    return residual + F.gelu(normed2 @ w1) @ w2


def main() -> None:
    level = os.environ.get("BENCH_LEVEL", "hard")
    tokens = 256 if level == "easy" else 512 if level == "medium" else 1024
    dim = 768
    heads = 12
    repeats = 8 if level == "easy" else 16 if level == "medium" else 24
    d = "mps"
    x = torch.randn(tokens, dim, device=d) * 0.05
    qw = torch.randn(dim, dim, device=d) * 0.05
    ow = torch.randn(dim, dim, device=d) * 0.05
    w1 = torch.randn(dim, dim * 4, device=d) * 0.05
    w2 = torch.randn(dim * 4, dim, device=d) * 0.05
    g1 = torch.ones(dim, device=d)
    b1 = torch.zeros(dim, device=d)
    g2 = torch.ones(dim, device=d)
    b2 = torch.zeros(dim, device=d)

    with torch.autocast(device_type=d, dtype=torch.float16, cache_enabled=False):
        for _ in range(WARMUP):
            block(x, qw, ow, w1, w2, g1, b1, g2, b2, heads)
    torch.mps.synchronize()
    started = time.perf_counter()
    with torch.autocast(device_type=d, dtype=torch.float16, cache_enabled=False):
        for _ in range(repeats):
            out = block(x, qw, ow, w1, w2, g1, b1, g2, b2, heads)
    torch.mps.synchronize()
    seconds = time.perf_counter() - started
    flops = repeats * (12.0 * tokens * dim * dim + 4.0 * tokens * tokens * dim)
    check = out.flatten()[0].item()
    sane = bool(torch.isfinite(out.flatten()[:1]).all().item()) and abs(check) < 1000.0
    print(f"jtb\tvit-block\t{seconds:.9f}\t{int(flops)}\t0\t{'ok' if sane else 'invalid'}\tamp-f16-f32")


main()
