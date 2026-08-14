#: The device paths agree with the arithmetic they claim to perform.
#:
#: Every operation in `std.gpu` has a CPU implementation and chooses between the
#: two by size, so a test that stays under `MIN_GPU_ELEMENTS` never reaches the
#: device at all and proves nothing about it. Every case here is deliberately
#: larger than that, and checks the device answer against a value this file
#: computes for itself.
#:
#: A device buffer holds 32-bit floats and a Jaithon float is a double, so the
#: comparisons are tolerant, and relative rather than absolute: a sum of a
#: hundred thousand terms carries far more rounding than one product does, and a
#: single fixed epsilon would either fail on the sum or prove nothing about the
#: product.
#:
#: The sizes are chosen to land on the edges the kernels actually have. The
#: elementwise kernels do four elements per thread, so a length that is not a
#: multiple of four exercises the tail. The matrix kernel works in 16x16 tiles,
#: so 70 exercises a ragged tile and 64 exercises the exact multiple.
import std.gpu
import std.math
from std.test import assert_eq, assert_true, assert_len

let N = 100_000  # comfortably past MIN_GPU_ELEMENTS

let ASYNC_SOURCE = """#include <metal_stdlib>
using namespace metal;
kernel void add_one(device float* values [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    values[i] += 1.0f;
}
kernel void twice(device float* values [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    values[i] *= 2.0f;
}
"""

fn ramp(n: int, scale: float) -> list[float] {
    var out: list[float] = []
    var i = 0
    while i < n {
        out.push(i % 97 * scale)
        i += 1
    }
    return out
}

#: Relative closeness, so the tolerance means the same thing for a product of
#: two small numbers and for a sum of a hundred thousand of them.
fn close(a: float, b: float, tolerance: float) -> bool {
    var scale = math.fabs(a)
    if math.fabs(b) > scale { scale = math.fabs(b) }
    if scale < 1.0 { scale = 1.0 }
    return math.fabs(a - b) <= tolerance * scale
}

fn test_vector_add_matches_elementwise_addition() -> void {
    let a = ramp(N, 1.0)
    let b = ramp(N, 0.5)
    let got = gpu.vector_add(a, b)
    assert_len(got, N)
    for i in [0, 1, 2, 3, 4, 5, 1000, 65535, N - 3, N - 2, N - 1] {
        assert_true(close(got[i], a[i] + b[i], 1e-6), f"add at {i}")
    }
}

fn test_vector_mul_matches_elementwise_multiplication() -> void {
    let a = ramp(N, 1.0)
    let b = ramp(N, 0.25)
    let got = gpu.vector_mul(a, b)
    assert_len(got, N)
    for i in [0, 1, 2, 3, 7, 4095, 4096, N - 4, N - 1] {
        assert_true(close(got[i], a[i] * b[i], 1e-6), f"mul at {i}")
    }
}

fn test_the_vector_kernels_handle_a_ragged_tail() -> void {
    let n = 4099  # not a multiple of the four-lane stride
    let a = ramp(n, 1.0)
    let b = ramp(n, 2.0)
    let sum = gpu.vector_add(a, b)
    let product = gpu.vector_mul(a, b)
    assert_len(sum, n)
    assert_len(product, n)
    for i in [n - 4, n - 3, n - 2, n - 1] {
        assert_true(close(sum[i], a[i] + b[i], 1e-6), f"tail add at {i}")
        assert_true(close(product[i], a[i] * b[i], 1e-6), f"tail mul at {i}")
    }
}

fn test_reduce_sum_matches_a_running_total() -> void {
    let a = ramp(N, 1.0)
    var want = 0.0
    for v in a { want = want + v }
    assert_true(close(gpu.reduce_sum(a), want, 1e-4), "reduce_sum")
}

fn test_dot_matches_the_sum_of_products() -> void {
    let a = ramp(N, 1.0)
    let b = ramp(N, 0.5)
    var want = 0.0
    var i = 0
    while i < N {
        want = want + a[i] * b[i]
        i += 1
    }
    assert_true(close(gpu.dot(a, b), want, 1e-4), "dot")
}

fn test_matmul_matches_the_triple_loop_on_a_ragged_tile() -> void {
    let n = 70  # deliberately not a multiple of the 16-wide tile
    let a = ramp(n * n, 0.5)
    let b = ramp(n * n, 0.25)
    let got = gpu.matmul(a, b, n, n, n)
    assert_len(got, n * n)
    for probe in [[0, 0], [0, 69], [69, 0], [69, 69], [15, 16], [16, 15], [33, 47]] {
        let r = probe[0]
        let c = probe[1]
        var want = 0.0
        var k = 0
        while k < n {
            want = want + a[r * n + k] * b[k * n + c]
            k += 1
        }
        assert_true(close(got[r * n + c], want, 1e-5), f"matmul at {r},{c}")
    }
}

fn test_matmul_handles_an_exact_multiple_of_the_tile() -> void {
    let n = 64
    let a = ramp(n * n, 0.5)
    let b = ramp(n * n, 0.25)
    let got = gpu.matmul(a, b, n, n, n)
    for probe in [[0, 0], [63, 63], [31, 32]] {
        let r = probe[0]
        let c = probe[1]
        var want = 0.0
        var k = 0
        while k < n {
            want = want + a[r * n + k] * b[k * n + c]
            k += 1
        }
        assert_true(close(got[r * n + c], want, 1e-5), f"square matmul at {r},{c}")
    }
}

#: A non-square shape, because rows, inner and columns are three different
#: numbers in the kernel and a tiled version can transpose two of them without
#: any square case noticing.
fn test_matmul_handles_a_non_square_shape() -> void {
    let rows = 40
    let inner = 70
    let cols = 33
    let a = ramp(rows * inner, 0.5)
    let b = ramp(inner * cols, 0.25)
    let got = gpu.matmul(a, b, rows, inner, cols)
    assert_len(got, rows * cols)
    for probe in [[0, 0], [0, 32], [39, 0], [39, 32], [17, 20]] {
        let r = probe[0]
        let c = probe[1]
        var want = 0.0
        var k = 0
        while k < inner {
            want = want + a[r * inner + k] * b[k * cols + c]
            k += 1
        }
        assert_true(close(got[r * cols + c], want, 1e-5), f"rect matmul at {r},{c}")
    }
}

fn test_upload_u8_expands_and_scales_without_a_boxed_host_list() -> void {
    if not gpu.is_available() { return }
    let buffer = gpu.Buffer(3)
    defer { buffer.free() }
    buffer.upload_u8(bytes([0, 127, 255]), scale: 0.5)
    assert_eq(buffer.download(), [0.0, 63.5, 127.5])
}

fn test_async_dispatches_are_ordered_and_synchronize_once() -> void {
    if not gpu.is_available() { return }
    let add_one = gpu.Kernel.compile(ASYNC_SOURCE, "add_one")
    let twice = gpu.Kernel.compile(ASYNC_SOURCE, "twice")
    let values = gpu.Buffer.of([1.0, 2.0, 3.0])
    defer { gpu.synchronize() }
    defer { add_one.free() }
    defer { twice.free() }
    defer { values.free() }

    add_one.dispatch_async([values], 3)
    twice.dispatch_async([values], 3)
    gpu.synchronize()
    assert_eq(values.download(), [4.0, 6.0, 8.0])
}
