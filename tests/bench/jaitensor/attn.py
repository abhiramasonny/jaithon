import os
import time

import torch
import torch.nn.functional as F


def packed_mha(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int) -> torch.Tensor:
    seq, dim = q.shape
    hd = dim // heads
    qh = q.view(seq, heads, hd).permute(1, 0, 2).unsqueeze(0).contiguous()
    kh = k.view(seq, heads, hd).permute(1, 0, 2).unsqueeze(0).contiguous()
    vh = v.view(seq, heads, hd).permute(1, 0, 2).unsqueeze(0).contiguous()
    ctx = F.scaled_dot_product_attention(qh, kh, vh)
    return ctx.squeeze(0).permute(1, 0, 2).reshape(seq, dim)


def time_attn(seq: int, dim: int, heads: int, repeats: int):
    q = torch.randn(seq, dim, device="mps")
    k = torch.randn(seq, dim, device="mps")
    v = torch.randn(seq, dim, device="mps")
    torch.mps.synchronize()
    with torch.autocast(device_type="mps", dtype=torch.float16, cache_enabled=False):
        packed_mha(q, k, v, heads)
    torch.mps.synchronize()
    started = time.perf_counter()
    live = []
    with torch.autocast(device_type="mps", dtype=torch.float16, cache_enabled=False):
        for _ in range(repeats):
            live.append(packed_mha(q, k, v, heads))
    torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    output = live[-1]
    values = output.flatten()[[0, output.numel() // 2, output.numel() - 1]]
    finite = bool(torch.isfinite(values).all().item())
    bounded = bool((values.abs() <= 1.01).all().item())
    nonzero = bool((values.abs() > 1e-6).any().item())
    del live
    return elapsed, finite and bounded and nonzero


def main() -> None:
    if not torch.backends.mps.is_available():
        raise SystemExit("bench needs Apple MPS")
    level = os.environ.get("BENCH_LEVEL", "hard")
    verbose = os.environ.get("JAITENSOR_VERBOSE", "0") != "0"
    repeats = 8 if level == "easy" else 16 if level == "medium" else 24
    shapes = [
        ("attn-256", 256, 256, 8),
        ("attn-512", 512, 512, 8),
        ("attn-1024", 1024, 512, 8),
    ]
    if level != "easy":
        shapes.extend(
            [
                ("attn-1024-wide", 1024, 1024, 16),
                ("attn-2048", 2048, 1024, 16),
                ("attn-4096", 4096, 1024, 16),
            ]
        )
    for name, seq, dim, heads in shapes:
        seconds, valid = time_attn(seq, dim, heads, repeats)
        if verbose:
            print(f"{name}: {seconds / repeats:.6f}s")
        flops = 4 * seq * seq * dim * repeats
        check = "ok" if valid else "invalid"
        print(f"jtb\t{name}\t{seconds:.9f}\t{flops}\t0\t{check}\tamp-f16-f32")


if __name__ == "__main__":
    main()
