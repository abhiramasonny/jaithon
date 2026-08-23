import os
import time

import torch
import torch.nn.functional as F


WARMUP = 24


def packed_mha(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int) -> torch.Tensor:
    seq, dim = q.shape
    hd = dim // heads
    qh = q.view(seq, heads, hd).permute(1, 0, 2).unsqueeze(0).contiguous()
    kh = k.view(seq, heads, hd).permute(1, 0, 2).unsqueeze(0).contiguous()
    vh = v.view(seq, heads, hd).permute(1, 0, 2).unsqueeze(0).contiguous()
    ctx = F.scaled_dot_product_attention(qh, kh, vh)
    return ctx.squeeze(0).permute(1, 0, 2).reshape(seq, dim)


# How long each shape should be timed for, in seconds. See the note in
# attn.jai: a fixed repeat count times the small shapes for almost no time and
# their run-to-run spread reached 28%. Both sides calibrate to this same
# target, so each measures itself for a sixth of a second.
TARGET_SECONDS = 0.15
MIN_REPEATS = 24
MAX_REPEATS = 4096


def repeats_for(seq: int, dim: int, heads: int, floor: int) -> int:
    q = torch.randn(seq, dim, device="mps")
    k = torch.randn(seq, dim, device="mps")
    v = torch.randn(seq, dim, device="mps")
    with torch.autocast(device_type="mps", dtype=torch.float16, cache_enabled=False):
        for _ in range(WARMUP):
            packed_mha(q, k, v, heads)
    torch.mps.synchronize()
    started = time.perf_counter()
    with torch.autocast(device_type="mps", dtype=torch.float16, cache_enabled=False):
        for _ in range(8):
            packed_mha(q, k, v, heads)
    torch.mps.synchronize()
    each = (time.perf_counter() - started) / 8
    if each <= 0.0:
        return MAX_REPEATS
    return max(floor, min(MAX_REPEATS, int(TARGET_SECONDS / each)))


def time_attn(seq: int, dim: int, heads: int, repeats: int):
    q = torch.randn(seq, dim, device="mps")
    k = torch.randn(seq, dim, device="mps")
    v = torch.randn(seq, dim, device="mps")
    torch.mps.synchronize()
    # See the note in gemm.jai: the GPU needs to have been busy for a while
    # before it runs at its working speed, and both sides warm up by the same
    # rule so the two are comparable.
    with torch.autocast(device_type="mps", dtype=torch.float16, cache_enabled=False):
        for _ in range(max(repeats, WARMUP)):
            packed_mha(q, k, v, heads)
    torch.mps.synchronize()
    started = time.perf_counter()
    # Exactly one result stays alive, which is what the jai side does and what
    # the note there says this side does. It appended every one of them to a
    # list instead: at attn-4096 that is twenty-four results of sixteen
    # megabytes held at once, so this side was being timed with four hundred
    # megabytes outstanding that the other side never had. Rebinding drops the
    # previous one to torch's caching allocator, same as `free()` does here.
    output = None
    with torch.autocast(device_type="mps", dtype=torch.float16, cache_enabled=False):
        for _ in range(repeats):
            output = packed_mha(q, k, v, heads)
    torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    values = output.flatten()[[0, output.numel() // 2, output.numel() - 1]]
    finite = bool(torch.isfinite(values).all().item())
    bounded = bool((values.abs() <= 1.01).all().item())
    nonzero = bool((values.abs() > 1e-6).any().item())
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
        counted = repeats_for(seq, dim, heads, max(repeats, MIN_REPEATS))
        seconds, valid = time_attn(seq, dim, heads, counted)
        if verbose:
            print(f"{name}: {seconds / counted:.6f}s")
        # Reported as though it had run `repeats` times; see gemm.jai.
        scaled = seconds / counted * repeats
        flops = 4 * seq * seq * dim * repeats
        check = "ok" if valid else "invalid"
        print(f"jtb\t{name}\t{scaled:.9f}\t{flops}\t0\t{check}\tamp-f16-f32")


if __name__ == "__main__":
    main()
