import time

import torch


def time_shape(rows: int, inner: int, columns: int, repeats: int, trans_a=False, trans_b=False) -> float:
    if trans_a:
        a = torch.randn(inner, rows, device="mps")
        b = torch.randn(inner, columns, device="mps")
        mul = lambda: a.T @ b
    elif trans_b:
        a = torch.randn(rows, inner, device="mps")
        b = torch.randn(columns, inner, device="mps")
        mul = lambda: a @ b.T
    else:
        a = torch.randn(rows, inner, device="mps")
        b = torch.randn(inner, columns, device="mps")
        mul = lambda: a @ b
    torch.mps.synchronize()
    mul()
    torch.mps.synchronize()
    started = time.perf_counter()
    for _ in range(repeats):
        c = mul()
        torch.mps.synchronize()
        del c
    return (time.perf_counter() - started) / repeats


def report(name: str, rows: int, inner: int, columns: int, seconds: float) -> None:
    flops = 2.0 * rows * inner * columns
    print(f"{name}: {rows}x{inner}x{columns}  {seconds * 1e6:.0f} us  {flops / seconds / 1e9:.1f} GFLOP/s")


def main() -> None:
    if not torch.backends.mps.is_available():
        raise SystemExit("bench needs Apple MPS")
    for name, rows, inner, columns, trans_a, trans_b in [
        ("fashion L1", 512, 784, 256, False, False),
        ("fashion W TN", 784, 512, 256, True, False),
        ("cifar L1", 512, 3072, 512, False, False),
        ("cifar W TN", 3072, 512, 512, True, False),
        ("cifar X NT", 512, 512, 3072, False, True),
        ("cifar-wide L1", 512, 3072, 1024, False, False),
        ("cifar-wide W TN", 3072, 512, 1024, True, False),
        ("fashion-wide L1", 512, 784, 2048, False, False),
        ("fashion-wide W TN", 784, 512, 2048, True, False),
    ]:
        report(name, rows, inner, columns, time_shape(rows, inner, columns, 40, trans_a, trans_b))


if __name__ == "__main__":
    main()
