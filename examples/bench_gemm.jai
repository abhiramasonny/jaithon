from jaitensor import Tensor, matmul
from jaitensor import is_available
import std.gpu as gpu
from std.time import Instant

fn time_shape(
    rows: int,
    inner: int,
    columns: int,
    repeats: int,
    trans_a: bool = false,
    trans_b: bool = false
) -> float {
    let a = if trans_a { Tensor([inner, rows]) } else { Tensor([rows, inner]) }
    let b = if trans_b { Tensor([columns, inner]) } else { Tensor([inner, columns]) }
    defer { a.free() }
    defer { b.free() }
    let first = matmul(a, b, trans_a: trans_a, trans_b: trans_b)
    gpu.synchronize()
    first.free()
    let started = Instant.now()
    for _i in 0..repeats {
        let product = matmul(a, b, trans_a: trans_a, trans_b: trans_b)
        gpu.synchronize()
        product.free()
    }
    return started.elapsed().to_seconds() / float(repeats)
}

fn report(name: str, rows: int, inner: int, columns: int, seconds: float) -> void {
    let flops = 2.0 * float(rows) * float(inner) * float(columns)
    let gflops = flops / seconds / 1.0e9
    print(f"{name}: {rows}x{inner}x{columns}  {seconds * 1.0e6:.0f} us  {gflops:.1f} GFLOP/s")
}

fn main() -> int {
    if not is_available() { throw RuntimeError("bench needs a GPU") }
    let warmup_a = Tensor([64, 1024])
    let warmup_b = Tensor([1024, 64])
    let warmup = matmul(warmup_a, warmup_b)
    gpu.synchronize()
    warmup.free()
    warmup_a.free()
    warmup_b.free()
    let shapes = [
        ("fashion L1", 512, 784, 256, false, false),
        ("fashion W TN", 784, 512, 256, true, false),
        ("cifar L1", 512, 3072, 512, false, false),
        ("cifar W TN", 3072, 512, 512, true, false),
        ("cifar X NT", 512, 512, 3072, false, true),
        ("cifar-wide L1", 512, 3072, 1024, false, false),
        ("cifar-wide W TN", 3072, 512, 1024, true, false),
        ("fashion-wide L1", 512, 784, 2048, false, false),
        ("fashion-wide W TN", 784, 512, 2048, true, false),
    ]
    for (name, rows, inner, columns, trans_a, trans_b) in shapes {
        let seconds = time_shape(rows, inner, columns, 40, trans_a: trans_a, trans_b: trans_b)
        report(name, rows, inner, columns, seconds)
    }
    return 0
}
