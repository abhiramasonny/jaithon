import os
import time

import torch


def gemm_batch(level: str) -> int:
    if level == "easy":
        return 512
    if level == "medium":
        return 2048
    return 4096


# Dispatches to run before the clock starts. See the note in gemm.jai: the
# GPU needs to have been busy for a while before it runs at its working speed,
# and both sides have to warm up by the same rule to be comparable.
WARMUP = 24


def time_shape(rows: int, inner: int, columns: int, repeats: int, trans_a=False, trans_b=False):
    if trans_a:
        a = torch.randn(inner, rows, device="mps")
        b = torch.randn(inner, columns, device="mps")
        mul = lambda: torch.matmul(a.T, b).float()
    elif trans_b:
        a = torch.randn(rows, inner, device="mps")
        b = torch.randn(columns, inner, device="mps")
        mul = lambda: torch.matmul(a, b.T).float()
    else:
        a = torch.randn(rows, inner, device="mps")
        b = torch.randn(inner, columns, device="mps")
        mul = lambda: torch.matmul(a, b).float()
    wave = 8
    outs = [None] * wave
    torch.mps.synchronize()
    with torch.autocast(device_type="mps", dtype=torch.float16, cache_enabled=False):
        for i in range(max(repeats, WARMUP)):
            outs[i % wave] = mul()
    torch.mps.synchronize()
    started = time.perf_counter()
    # Jaithon's float32 buffers are cast for every graph dispatch. Disable
    # PyTorch's cross-call autocast cache so repeated inputs have the same
    # storage/compute contract instead of becoming cached half tensors.
    with torch.autocast(device_type="mps", dtype=torch.float16, cache_enabled=False):
        for i in range(repeats):
            outs[i % wave] = mul()
    torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    product = outs[(repeats - 1) % wave]
    assert product is not None
    values = product.flatten()[[0, product.numel() // 2, product.numel() - 1]]
    finite = bool(torch.isfinite(values).all().item())
    bounded = bool((values.abs() <= inner * 1.05).all().item())
    nonzero = bool((values.abs() > 1e-6).any().item())
    return elapsed, finite and bounded and nonzero


def main() -> None:
    if not torch.backends.mps.is_available():
        raise SystemExit("bench needs Apple MPS")
    level = os.environ.get("BENCH_LEVEL", "hard")
    repeats = 8 if level == "easy" else 20 if level == "medium" else 40
    batch = gemm_batch(level)
    for name, rows, inner, columns, trans_a, trans_b in [
        ("fashion-L1", batch, 784, 256, False, False),
        ("fashion-W-TN", 784, batch, 256, True, False),
        ("cifar-L1", batch, 3072, 512, False, False),
        ("cifar-W-TN", 3072, batch, 512, True, False),
        ("cifar-X-NT", batch, 512, 3072, False, True),
        ("cifar-wide-L1", batch, 3072, 1024, False, False),
        ("cifar-wide-W-TN", 3072, batch, 1024, True, False),
        ("cifar100-wide-L1", 1024, 3072, 2048, False, False),
        ("cifar100-wide-W-TN", 3072, 1024, 2048, True, False),
        ("fashion-wide-L1", batch, 784, 2048, False, False),
        ("fashion-wide-W-TN", 784, batch, 2048, True, False),
        ("fashion-xl-L1", batch, 784, 4096, False, False),
        ("fashion-xl-W-TN", 784, batch, 4096, True, False),
        ("mnist-xl-L1", batch, 784, 8192, False, False),
    ]:
        seconds, valid = time_shape(rows, inner, columns, repeats, trans_a, trans_b)
        flops = 2 * rows * inner * columns * repeats
        check = "ok" if valid else "invalid"
        print(f"jtb\t{name}\t{seconds:.9f}\t{flops}\t0\t{check}\tamp-f16-f32/mps")


if __name__ == "__main__":
    main()
