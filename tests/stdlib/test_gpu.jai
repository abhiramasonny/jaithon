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
from std.test import assert_eq, assert_true, assert_len, assert_throws

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

fn test_buffer_view_reads_a_window_without_freeing_the_parent() -> void {
    if not gpu.is_available() { return }
    let buffer = gpu.Buffer.of([1.0, 2.0, 3.0, 4.0])
    defer { buffer.free() }
    let viewed = buffer.view(1, 2)
    defer { viewed.free() }
    assert_eq(viewed.download(), [2.0, 3.0])
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

fn test_flush_commits_async_work_before_synchronize() -> void {
    if not gpu.is_available() { return }
    let add_one = gpu.Kernel.compile(ASYNC_SOURCE, "add_one")
    let values = gpu.Buffer.of([1.0, 2.0, 3.0])
    defer { gpu.synchronize() }
    defer { add_one.free() }
    defer { values.free() }
    add_one.dispatch_async([values], 3)
    gpu.flush()
    gpu.synchronize()
    assert_eq(values.download(), [2.0, 3.0, 4.0])
}

fn test_fill_uniform_stays_in_range_on_the_device() -> void {
    if not gpu.is_available() { return }
    let values = gpu.Buffer(N)
    defer { values.free() }
    values.fill_uniform(-1.0, 1.0, 42)
    gpu.synchronize()
    let got = values.download()
    assert_len(got, N)
    var saw_low = false
    var saw_high = false
    for i in [0, 1, 2, 3, 1000, 65535, N - 1] {
        assert_true(got[i] >= -1.0 and got[i] < 1.0, f"uniform at {i}")
        if got[i] < 0.0 { saw_low = true }
        if got[i] > 0.0 { saw_high = true }
    }
    assert_true(saw_low and saw_high)
    let again = gpu.Buffer(N)
    defer { again.free() }
    again.fill_uniform(-1.0, 1.0, 42)
    gpu.synchronize()
    let copied = again.download()
    assert_true(close(copied[0], got[0], 1e-6))
    assert_true(close(copied[N - 1], got[N - 1], 1e-6))
}

#: A host write lands after the work already queued against the same buffer,
#: not underneath it.
#:
#: The queued kernel adds one and the host then uploads zeros, so the buffer
#: must read back as zeros. With no barrier the upload lands first and the
#: kernel adds one to it afterwards, reading back as ones -- the same
#: shape of mistake that had an optimizer's momentum refilled from the previous
#: parameter's update.
fn test_a_host_write_waits_for_work_already_queued_against_the_buffer() -> void {
    if not gpu.is_available() { return }
    let add_one = gpu.Kernel.compile(ASYNC_SOURCE, "add_one")
    let values = gpu.Buffer.of([1.0, 2.0, 3.0])
    defer { gpu.synchronize() }
    defer { add_one.free() }
    defer { values.free() }

    add_one.dispatch_async([values], 3)
    values.upload([0.0, 0.0, 0.0])
    gpu.synchronize()
    assert_eq(values.download(), [0.0, 0.0, 0.0])
}

#: Reading one buffer waits for the work that wrote it and leaves the rest of
#: the queue alone, so a later read of another buffer still gets its own work.
#:
#: The flush puts the two dispatches in separate batches, which is the case
#: where a read has to retire part of the queue rather than all or none of it.
fn test_reading_one_buffer_does_not_lose_work_queued_for_another() -> void {
    if not gpu.is_available() { return }
    let add_one = gpu.Kernel.compile(ASYNC_SOURCE, "add_one")
    let first = gpu.Buffer.of([1.0, 2.0, 3.0])
    let second = gpu.Buffer.of([10.0, 20.0, 30.0])
    defer { gpu.synchronize() }
    defer { add_one.free() }
    defer { first.free() }
    defer { second.free() }

    add_one.dispatch_async([first], 3)
    gpu.flush()
    add_one.dispatch_async([second], 3)
    assert_eq(first.download(), [2.0, 3.0, 4.0])
    assert_eq(second.download(), [11.0, 21.0, 31.0])
}

#: Packing on the way off the device agrees with narrowing to bytes and packing
#: those, which is the two-pass route it replaces.
fn test_pack_argb_matches_narrowing_then_packing() -> void {
    if not gpu.is_available() { return }
    let pixels = N // 3
    let values = gpu.Buffer(pixels * 3)
    defer { values.free() }
    var raw: list[float] = []
    for index in 0..pixels * 3 {
        #: Past both ends as well as between them, so the clamping is exercised.
        raw.push(float(index % 301) - 20.0)
    }
    values.upload(raw)
    let one_pass = values.pack_argb(0, pixels * 3, 3)
    let two_pass = __prim__.list_pack_argb(values.download_u8(0, pixels * 3), 3)
    assert_len(one_pass, pixels)
    assert_eq(one_pass, two_pass)
}

fn test_pack_argb_refuses_a_ragged_pixel_count() -> void {
    if not gpu.is_available() { return }
    let values = gpu.Buffer(10)
    defer { values.free() }
    assert_throws(ValueError, || values.pack_argb(0, 10, 3))
    assert_throws(ValueError, || values.pack_argb(0, 9, 2))
}
