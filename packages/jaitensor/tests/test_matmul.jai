#: The matmul routes: tile plans, the staged and MPS kernels, and the fused affine.
import jaitensor as jt
import std.gpu as gpu
from jaitensor.gemm import Plan, bucket_extent, cache_key, candidates, default_plan, dispatch_plan
from jaitensor.gemm import last_tune_counts, plan_from_text, reload_plans
import std.math as math
from jaitensor.ops import matmul_backend
from jaitensor.tensor import Tensor
from std.io import Path
from std.os import env, pid, set_env, temp_dir, unset_env
from std.random import Random
from std.test import assert_eq, assert_near, assert_throws, assert_true

fn device_tensor(shape: list[int], seed: int) -> Tensor {
    let tensor = jt.Tensor(shape)
    tensor.buffer().fill_uniform(-1.0, 1.0, seed)
    return tensor
}

fn assert_gpu_matmul_probes(
    product: Tensor,
    a: Tensor,
    b: Tensor,
    rows: int,
    inner: int,
    columns: int,
    trans_a: bool = false,
    trans_b: bool = false,
    tol: float = 5e-3
) -> void {
    let got = product.download()
    let a_vals = a.download()
    let b_vals = b.download()
    let probes = [
        [0, 0],
        [0, columns - 1],
        [rows - 1, 0],
        [rows - 1, columns - 1],
        [rows // 2, columns // 2],
    ]
    for probe in probes {
        let row = probe[0]
        let column = probe[1]
        var want = 0.0
        for k in 0..inner {
            let av = if trans_a { a_vals[k * rows + row] } else { a_vals[row * inner + k] }
            let bv = if trans_b { b_vals[column * inner + k] } else { b_vals[k * columns + column] }
            want += av * bv
        }
        assert_near(got[row * columns + column], want, tol)
    }
}

fn assert_gpu_matmul(
    rows: int,
    inner: int,
    columns: int,
    trans_a: bool = false,
    trans_b: bool = false,
    tol: float = 5e-3
) -> void {
    let a = if trans_a { device_tensor([inner, rows], 1) } else { device_tensor([rows, inner], 1) }
    let b = if trans_b {
        device_tensor([columns, inner], 2)
    } else {
        device_tensor([inner, columns], 2)
    }
    defer { a.free() }
    defer { b.free() }
    gpu.flush()
    let product = jt.matmul(a, b, trans_a: trans_a, trans_b: trans_b)
    defer { product.free() }
    assert_gpu_matmul_probes(
        product,
        a,
        b,
        rows,
        inner,
        columns,
        trans_a: trans_a,
        trans_b: trans_b,
        tol: tol
    )
}

fn reduce_to_scalar_for_test(t: Tensor) -> Tensor {
    return jt.sum(t)
}

fn test_tiled_matmul_matches_cpu_on_ragged_tiles() -> void {
    if not jt.is_available() { return }
    let a = jt.Tensor([17, 19], [float(i * 3 % 11) for i in 0..17 * 19])
    let b = jt.Tensor([19, 18], [float(i * 5 % 7) for i in 0..19 * 18])
    defer { a.free() }
    defer { b.free() }
    let product = jt.matmul(a, b)
    defer { product.free() }
    let got = product.download()
    let a_vals = a.download()
    let b_vals = b.download()
    for row in 0..17 {
        for column in 0..18 {
            var want = 0.0
            for k in 0..19 {
                want += a_vals[row * 19 + k] * b_vals[k * 18 + column]
            }
            assert_near(got[row * 18 + column], want, 1e-3)
        }
    }
}

fn test_default_plan_prefers_direct_kernels_then_splits() -> void {
    #: Small aligned products take a direct kernel, not the staged one that
    #: cost three times the GPU time at 64 cubed.
    assert_eq(default_plan(64, 64, 64).to_str(), "direct32x32:1")
    #: Unsplit while a split would keep under 256 steps of K: summing the
    #: partials is a second dispatch.
    assert_eq(default_plan(128, 128, 128).to_str(), "direct32x32:1")
    assert_eq(default_plan(512, 784, 256).to_str(), "direct32x32:2")
    assert_eq(default_plan(1024, 384, 384).to_str(), "direct32x32:1")
    #: Ragged shapes keep a staged kernel: `gemm64x32` where it still makes a
    #: few threadgroups, split to fill the device, else `gemm32x32`.
    assert_eq(default_plan(8, 7, 5).to_str(), "gemm32x32:1")
    assert_eq(default_plan(100, 100, 100).to_str(), "gemm32x32:1")
    #: 270 GFLOPS unsplit, 625 split four ways; and never the graph below the
    #: large threshold, whose first call per shape costs 3 ms.
    assert_eq(default_plan(300, 300, 300).to_str(), "gemm64x32:4")
    assert_eq(default_plan(1000, 1000, 96).to_str(), "gemm64x32:4")
    #: A one-wide output was given `gemm128x128` and computed mostly padding:
    #: 6 GFLOPS against 20 for `gemm64x32`.
    assert_eq(default_plan(100000, 50, 1, false, true).to_str(), "gemm64x32:1")
    #: Large products go to the graph wherever a device can run it.
    if gpu.is_available() {
        assert_eq(default_plan(512, 784, 8192).to_str(), "mpsgraph:1")
        assert_eq(default_plan(1024, 768, 768).to_str(), "mpsgraph:1")
    }
}

fn test_default_plan_splits_a_tall_skinny_product_past_sixteen() -> void {
    #: X^T X and X^T y over 10k samples: four and two output tiles, so
    #: sixteen splits left the device mostly idle. 151 us at 64 splits
    #: against 224 at 32 and 383 at 16, and X^T y 103 us at 128.
    assert_eq(default_plan(50, 10000, 50).to_str(), "gemm32x32:64")
    assert_eq(default_plan(50, 10000, 1).to_str(), "gemm32x32:128")
    #: A product with tiles enough of its own is never split that far.
    assert_eq(default_plan(8, 7, 5).to_str(), "gemm32x32:1")
    #: Deeper still, the graph: it splits K itself, and beat every kernel
    #: plan at 100k samples.
    if gpu.is_available() {
        assert_eq(default_plan(50, 100000, 50).to_str(), "mpsgraph:1")
        assert_eq(default_plan(50, 100000, 1).to_str(), "mpsgraph:1")
        assert_eq(default_plan(50, 100000, 50, true).to_str(), "mpsgraph:1")
    }
}

fn test_a_small_output_keeps_its_direct_kernel_past_the_graph_thresholds() -> void {
    #: Deep or large, an output too small to fill the device keeps a direct
    #: kernel where one divides it and splits K. The graph was static for
    #: these and ran the width-96 gram 4.1x and the width-160 one 3.5x slower
    #: than `direct32x32:16`; 256x5000x256 1.36x slower.
    assert_eq(default_plan(96, 40000, 96, true).to_str(), "direct32x32:16")
    assert_eq(default_plan(160, 40000, 160, true).to_str(), "direct32x32:8")
    assert_eq(default_plan(256, 5000, 256, true).to_str(), "direct32x32:4")
    assert_eq(default_plan(256, 8192, 128).to_str(), "direct32x32:8")
    #: Large, but padding the 500000-row operand for the graph cost more than
    #: the graph saved: 29.5 ms against 24.9 on `gemm64x32`.
    assert_eq(default_plan(500000, 130, 10).to_str(), "gemm64x32:1")
}

fn has_kernel_plan(plans: list[Plan]) -> bool {
    for plan in plans {
        if plan.config >= 0 { return true }
    }
    return false
}

fn test_kernels_are_timed_against_a_static_graph_plan_unless_the_output_is_large() -> void {
    if not gpu.is_available() { return }
    #: Statically the graph, but only a guess there: a tuning pass must be
    #: allowed to find a kernel that beats it, or a wrong guess is permanent.
    assert_true(has_kernel_plan(candidates(50, 100000, 50, true)), "a deep gram times kernels")
    assert_true(has_kernel_plan(candidates(50, 100000, 1, true)), "a deep X^T y times kernels")
    #: A large output is the graph's alone.
    assert_true(not has_kernel_plan(candidates(1024, 768, 768)), "1024x768x768 is the graph's")
    assert_true(not has_kernel_plan(candidates(4096, 4096, 4096)), "4096 cubed is the graph's")
    #: So is one it reads through padded copies, which the static plan only
    #: gives it where the copies are cheap: a staged kernel would take a
    #: second a sample at 8191 cubed.
    assert_true(not has_kernel_plan(candidates(4001, 1001, 999)), "4001x1001x999 is the graph's")
}

#: Point the planner at a fresh plan cache holding `rows`, run `body`, and put
#: everything back.
fn with_plan_cache(name: str, rows: list[str], body: fn() -> void) -> void {
    let path = Path(temp_dir()).join(f"jaitensor-{name}-{pid()}.tsv")
    let previous = env("JAITENSOR_GEMM_CACHE")
    var text = ""
    for row in rows { text = text + row + "\n" }
    path.write_text(text)
    set_env("JAITENSOR_GEMM_CACHE", path.to_str())
    reload_plans()
    defer {
        if previous is not null {
            set_env("JAITENSOR_GEMM_CACHE", previous)
        } else {
            unset_env("JAITENSOR_GEMM_CACHE")
        }
        reload_plans()
        if path.exists() { path.remove() }
    }
    body()
}

fn test_an_empty_inner_axis_is_zeros_whatever_its_neighbours_were_tuned_to() -> void {
    if not jt.is_available() { return }
    if env("JAITENSOR_GEMM_PLAN") is not null { return }
    #: An empty inner axis shared the one-deep bucket, and a bucket's tuned
    #: plan is borrowed by every member it is legal for. Tuned to the graph,
    #: it was handed to the empty product too, and MPS refuses `k == 0`:
    #: `gpu_matmul_buffers(): the kernel failed`. A row for the empty bucket
    #: itself -- a stale or hand-edited file -- must not reach MPS either.
    let rows = 2048
    let columns = 2048
    let graph = [
        cache_key(rows, 1, columns) + "\tmpsgraph:1",
        cache_key(rows, 0, columns) + "\tmpsgraph:1",
    ]
    with_plan_cache("empty-inner", graph, fn() {
        let a = jt.Tensor([rows, 0])
        let b = jt.Tensor([0, columns])
        let out = jt.Tensor([rows, columns])
        defer {
            a.free()
        }
        defer {
            b.free()
        }
        defer {
            out.free()
        }
        out.buffer().fill(7.0)
        jt.matmul(a, b, out: out)
        let got = out.download()
        for at in [0, 1, columns, rows * columns - 1, rows * columns // 2 + 17] {
            assert_eq(got[at], 0.0)
        }
    })
}

fn test_a_bad_forced_plan_raises_on_every_product() -> void {
    if not jt.is_available() { return }
    let previous = env("JAITENSOR_GEMM_PLAN")
    defer {
        if previous is not null {
            set_env("JAITENSOR_GEMM_PLAN", previous)
        } else {
            unset_env("JAITENSOR_GEMM_PLAN")
        }
        reload_plans()
    }
    let a = device_tensor([64, 64], 51)
    defer { a.free() }
    set_env("JAITENSOR_GEMM_PLAN", "bogus")
    reload_plans()
    #: It used to be marked read before it was parsed, so the first product
    #: raised and every later one ran unforced.
    for _attempt in 0..3 { assert_throws(ValueError, || jt.matmul(a, a).free()) }
    #: Set but empty is unset, as it was before bad values raised.
    set_env("JAITENSOR_GEMM_PLAN", "")
    reload_plans()
    let product = jt.matmul(a, a)
    product.free()
}

fn test_a_tune_behind_a_queued_backlog_still_times_its_candidates() -> void {
    if not jt.is_available() { return }
    if (env("JAITENSOR_AUTOTUNE") ?? "1") == "0" { return }
    if (env("JAITENSOR_GEMM_LAZY_TUNE") ?? "1") == "0" { return }
    if env("JAITENSOR_GEMM_PLAN") is not null { return }
    #: A bucket can earn its pass in the middle of an asynchronous loop. The
    #: pass's budget clock used to start before its first wait, which was for
    #: everything the caller had queued, so behind half a second of work no
    #: candidate was timed and the static plan was written down as tuned.
    with_plan_cache("backlog", [], fn() {
        let side = 2048
        let big = device_tensor([side, side], 61)
        let big_out = jt.Tensor([side, side])
        let rows = 256
        let inner = 8192
        let columns = 256
        let a = device_tensor([rows, inner], 62)
        let b = device_tensor([inner, columns], 63)
        let out = jt.Tensor([rows, columns])
        defer { big.free() }
        defer { big_out.free() }
        defer { a.free() }
        defer { b.free() }
        defer { out.free() }
        gpu.synchronize()
        #: About half a second of GPU work, queued and not waited on; the
        #: planner never sees it, so it cannot tune anything itself.
        let slow = plan_named("gemm32x32:1")
        for _call in 0..16 {
            dispatch_plan(slow, big.buffer(), big.buffer(), big_out.buffer(), big_out.buffer(), side, side, side)
        }
        var calls = 0
        while calls < 2000 and last_tune_counts()[0] == 0 {
            jt.matmul(a, b, out: out)
            calls += 1
        }
        gpu.synchronize()
        let (offered, timed) = last_tune_counts()
        assert_true(offered > 1, f"256x8192x256 offers kernels and the graph, got {offered} after {calls} calls")
        assert_true(timed >= 2, f"a pass behind a backlog timed {timed} of {offered} candidates")
    })
}

fn test_plan_from_text_reads_a_bare_name_as_one_split() -> void {
    let graph = plan_from_text("mpsgraph")
    assert_true(graph is not null)
    assert_eq(graph.to_str(), "mpsgraph:1")
    let kernel = plan_from_text("direct32x32")
    assert_true(kernel is not null)
    assert_eq(kernel.to_str(), "direct32x32:1")
    assert_eq(plan_from_text("gemm32x32:64").to_str(), "gemm32x32:64")
    assert_true(plan_from_text("no-such-kernel") is null)
    assert_true(plan_from_text("gemm32x32:many") is null)
    assert_true(plan_from_text("gemm32x32:0") is null)
}

fn test_a_shape_is_tuned_only_once_it_has_earned_it() -> void {
    if not jt.is_available() { return }
    #: The switches that turn tuning off or make it eager are a different
    #: policy, not this one.
    if (env("JAITENSOR_AUTOTUNE") ?? "1") == "0" { return }
    if (env("JAITENSOR_GEMM_LAZY_TUNE") ?? "1") == "0" { return }
    if env("JAITENSOR_GEMM_PLAN") is not null { return }
    #: Twice, each time against a fresh cache file. The second pass is the
    #: check that `reload_plans` is a fresh start: the planner keeps what it
    #: read from the first file it opened and memoises every route, so
    #: without it the second file was never tuned into -- and on a machine
    #: whose own cache held this bucket, neither was the first.
    for round in 0..2 {
        tuned_once_earned(Path(temp_dir()).join(f"jaitensor-lazy-plans-{pid()}-{round}.tsv"))
    }
}

fn tuned_once_earned(path: Path) -> void {
    #: Every new shape used to be tuned on its first call -- 0.5 s for each
    #: row count of a generation, 8-19 s for a first Ridge fit. Now the
    #: first calls run the static plan and write nothing; a tuning pass, and
    #: a cache row keyed on the shape's BUCKET, come only once the shape has
    #: done work worth tuning for.
    let previous = env("JAITENSOR_GEMM_CACHE")
    if path.exists() { path.remove() }
    set_env("JAITENSOR_GEMM_CACHE", path.to_str())
    reload_plans()
    defer {
        if previous is not null {
            set_env("JAITENSOR_GEMM_CACHE", previous)
        } else {
            unset_env("JAITENSOR_GEMM_CACHE")
        }
        reload_plans()
        if path.exists() { path.remove() }
    }
    let rows = 1000
    let inner = 1000
    let columns = 96
    let a = device_tensor([rows, inner], 41)
    let b = device_tensor([inner, columns], 42)
    let out = jt.Tensor([rows, columns])
    defer { a.free() }
    defer { b.free() }
    defer { out.free() }
    for _call in 0..3 { jt.matmul(a, b, out: out) }
    gpu.synchronize()
    assert_true(not path.exists(), "three calls are not worth a tuning pass")
    for _call in 0..2000 { jt.matmul(a, b, out: out) }
    gpu.synchronize()
    assert_true(path.exists(), "two thousand calls are")
    assert_true(path.read_text().contains("\t1024\t1024\t96\t0\t0\t"), "keyed on the bucket")
    assert_gpu_matmul_probes(out, a, b, rows, inner, columns)
}

fn test_bucket_extent_rounds_up_by_quarter_octaves() -> void {
    assert_eq(bucket_extent(1), 1)
    assert_eq(bucket_extent(29), 32)
    assert_eq(bucket_extent(64), 64)
    assert_eq(bucket_extent(65), 80)
    assert_eq(bucket_extent(1024), 1024)
    assert_eq(bucket_extent(1025), 1280)
    #: Neighbouring sample counts share a plan, so a new fold is not a new
    #: tuning pass.
    assert_eq(bucket_extent(77777), bucket_extent(80000))
}

fn plan_named(text: str) -> Plan {
    let plan = plan_from_text(text)
    if plan is null { throw AssertionError(f"{text} names no plan") }
    return plan
}

#: Host reference for `op(a) x op(b)` in float64.
fn host_reference(
    a: list[float],
    b: list[float],
    rows: int,
    inner: int,
    columns: int,
    trans_a: bool,
    trans_b: bool
) -> list[float] {
    var out = [0.0 for _index in 0..rows * columns]
    for row in 0..rows {
        for column in 0..columns {
            var total = 0.0
            for k in 0..inner {
                let av = if trans_a { a[k * rows + row] } else { a[row * inner + k] }
                let bv = if trans_b { b[column * inner + k] } else { b[k * columns + column] }
                total += av * bv
            }
            out[row * columns + column] = total
        }
    }
    return out
}

fn assert_plan_matches_host(
    plan: Plan,
    rows: int,
    inner: int,
    columns: int,
    trans_a: bool,
    trans_b: bool,
    same_operand: bool = false
) -> void {
    let a = if trans_a {
        device_tensor([inner, rows], 11)
    } else {
        device_tensor([rows, inner], 11)
    }
    let b = if same_operand {
        a
    } elif trans_b {
        device_tensor([columns, inner], 12)
    } else {
        device_tensor([inner, columns], 12)
    }
    let out = jt.Tensor([rows, columns])
    defer { a.free() }
    defer {
        if not same_operand { b.free() }
    }
    defer { out.free() }
    out.buffer().fill(1.0e9)
    dispatch_plan(
        plan,
        a.buffer(),
        b.buffer(),
        out.buffer(),
        out.buffer(),
        rows,
        inner,
        columns,
        trans_a: trans_a,
        trans_b: trans_b
    )
    let got = out.download()
    let left = a.download()
    let right = b.download()
    let want = host_reference(left, right, rows, inner, columns, trans_a, trans_b)
    let scale = host_reference(
        [abs(value) for value in left],
        [abs(value) for value in right],
        rows,
        inner,
        columns,
        trans_a,
        trans_b
    )
    let what = f"{plan.to_str()} {rows}x{inner}x{columns} ta={trans_a} tb={trans_b}"
    #: Relative to the sum of |a b|, so a lost K slice cannot hide: dropping
    #: one of 256 slices of a 20000-deep product moves an element by about
    #: 4e-4 of it, where float32 rounding is under 1e-6. The old bound, 1e-3
    #: per unit of depth absolute, let a missing slice pass.
    for at in 0..want.len() {
        let bound = 1.0e-4 * scale[at] + 1.0e-6
        if not (got[at] - want[at] < bound and want[at] - got[at] < bound) {
            throw AssertionError(f"{what}: element {at} is {got[at]}, want {want[at]}")
        }
    }
}

fn test_padded_graph_route_matches_host_on_ragged_widths() -> void {
    if not jt.is_available() { return }
    let graph = plan_named("mpsgraph")
    let samples = 300
    for width in [1, 2, 3, 5, 49, 50, 51, 63] {
        #: The gram, both operands one buffer: padded once.
        assert_plan_matches_host(graph, width, samples, width, true, false, same_operand: true)
        #: X^T y: a one-wide result.
        assert_plan_matches_host(graph, width, samples, 1, true, false)
        #: X W and A B^T: padding the inner axis of both operands.
        assert_plan_matches_host(graph, samples, width, 3, false, false)
        assert_plan_matches_host(graph, 7, width, 5, false, true)
        assert_plan_matches_host(graph, 6, width, 9, true, true)
    }
}

fn test_padded_graph_route_applies_the_epilogue() -> void {
    if not jt.is_available() { return }
    let rows = 9
    let inner = 50
    let columns = 7
    let x = device_tensor([rows, inner], 21)
    let w = device_tensor([inner, columns], 22)
    let bias = device_tensor([columns], 23)
    let out = jt.Tensor([rows, columns])
    defer { x.free() }
    defer { w.free() }
    defer { bias.free() }
    defer { out.free() }
    dispatch_plan(
        plan_named("mpsgraph"),
        x.buffer(),
        w.buffer(),
        out.buffer(),
        bias.buffer(),
        rows,
        inner,
        columns,
        epilogue: 2
    )
    let got = out.download()
    let plain = host_reference(x.download(), w.download(), rows, inner, columns, false, false)
    let shift = bias.download()
    for row in 0..rows {
        for column in 0..columns {
            let want = max(plain[row * columns + column] + shift[column], 0.0)
            assert_near(got[row * columns + column], want, 1e-3)
        }
    }
}

fn host_activation(value: float, epilogue: int) -> float {
    if epilogue == 2 { return max(value, 0.0) }
    if epilogue == 3 { return 1.0 / (1.0 + math.exp(-value)) }
    if epilogue == 4 {
        if value > 15.0 { return 1.0 }
        if value < -15.0 { return -1.0 }
        return math.tanh(value)
    }
    if epilogue == 5 {
        let inner = 0.7978845608 * (value + 0.044715 * value * value * value)
        let t = if inner > 15.0 { 1.0 } elif inner < -15.0 { -1.0 } else { math.tanh(inner) }
        return 0.5 * value * (1.0 + t)
    }
    return value
}

#: `out = act(op(a) x op(b) + bias)` on `plan`, against the host.
fn assert_epilogue_matches_host(
    plan: Plan,
    rows: int,
    inner: int,
    columns: int,
    trans_a: bool,
    trans_b: bool,
    epilogue: int,
    scale: float = 1.0
) -> void {
    let a = jt.Tensor(if trans_a { [inner, rows] } else { [rows, inner] })
    a.buffer().fill_uniform(-scale, scale, 41)
    let b = if trans_b { device_tensor([columns, inner], 42) } else { device_tensor([inner, columns], 42) }
    let bias = device_tensor([rows * columns], 43)
    let out = jt.Tensor([rows, columns])
    defer { a.free() }
    defer { b.free() }
    defer { bias.free() }
    defer { out.free() }
    out.buffer().fill(1.0e9)
    dispatch_plan(
        plan,
        a.buffer(),
        b.buffer(),
        out.buffer(),
        bias.buffer(),
        rows,
        inner,
        columns,
        trans_a: trans_a,
        trans_b: trans_b,
        epilogue: epilogue
    )
    let got = out.download()
    let plain = host_reference(a.download(), b.download(), rows, inner, columns, trans_a, trans_b)
    let shift = bias.download()
    for row in 0..rows {
        for column in 0..columns {
            var value = plain[row * columns + column]
            if epilogue == 6 {
                value += shift[row * columns + column]
            } elif epilogue != 0 {
                value += shift[column]
            }
            let want = host_activation(value, epilogue)
            let have = got[row * columns + column]
            assert_true(math.is_finite(have), f"{plan.to_str()} epilogue {epilogue} gave {have}")
            assert_near(have, want, 1e-3 * (1.0 + math.fabs(want)))
        }
    }
}

#: Kernels compile in fast-math mode, where `tanh` overflows an exponential:
#: it gave 0 at 44 and NaN from 45, and GELU NaN from an input of 20. A 1x2000
#: row of values around one sums to a pre-activation in the tens.
fn test_tanh_and_gelu_epilogues_stay_finite_on_large_pre_activations() -> void {
    if not jt.is_available() { return }
    for text in ["gemm32x32", "gemm64x32"] {
        for epilogue in [4, 5] {
            assert_epilogue_matches_host(plan_named(text), 3, 2000, 5, false, false, epilogue, scale: 40.0)
        }
    }
    assert_epilogue_matches_host(plan_named("direct32x32"), 32, 2000, 32, false, false, 5, scale: 40.0)
}

fn test_deep_split_matches_host() -> void {
    if not jt.is_available() { return }
    for text in ["gemm32x32:64", "gemm32x32:256", "gemm64x64:128"] {
        assert_plan_matches_host(plan_named(text), 50, 20000, 50, true, false, same_operand: true)
        assert_plan_matches_host(plan_named(text), 50, 20000, 1, true, false)
    }
}

fn test_ragged_tall_skinny_matmul_matches_host() -> void {
    if not jt.is_available() { return }
    #: Whatever the planner picks for it, statically or tuned.
    let x = device_tensor([20000, 50], 31)
    defer { x.free() }
    let gram = jt.matmul(x, x, trans_a: true)
    defer { gram.free() }
    let got = gram.download()
    let values = x.download()
    let want = host_reference(values, values, 50, 20000, 50, true, false)
    for at in 0..want.len() { assert_near(got[at], want[at], 0.1) }
}

fn test_matmul_backend_reports_a_known_plan() -> void {
    #: Whatever the tuner has decided for this machine, the reported route has
    #: to name a plan the dispatcher can read back.
    let reported = matmul_backend(512, 784, 256)
    let parsed = plan_from_text(reported)
    assert_true(parsed is not null)
    assert_eq(parsed.to_str(), reported)
}

fn test_staged_64_matmul_matches_cpu() -> void {
    if not jt.is_available() { return }
    assert_gpu_matmul(64, 1024, 64)
}

fn test_wide_nn2_matmul_matches_cpu() -> void {
    if not jt.is_available() { return }
    assert_gpu_matmul(128, 80, 2048)
}

fn test_mps_scale_matmul_matches_cpu() -> void {
    if not jt.is_available() { return }
    assert_gpu_matmul(512, 2048, 1024, tol: 2e-2)
}

fn test_mps_scale_transposed_matmul_matches_cpu() -> void {
    if not jt.is_available() { return }
    let rows = 784
    let inner = 512
    let columns = 4096
    let a = jt.Tensor([inner, rows], [float(i * 3 % 11) for i in 0..inner * rows])
    let b = jt.Tensor([inner, columns], [float(i * 5 % 7) for i in 0..inner * columns])
    defer { a.free() }
    defer { b.free() }
    let product = jt.matmul(a, b, trans_a: true)
    defer { product.free() }
    let got = product.download()
    let a_vals = a.download()
    let b_vals = b.download()
    for row in [0, 100, 783] {
        for column in [0, 1, 64, 2048, 4095] {
            var want = 0.0
            for k in 0..inner {
                want += a_vals[k * rows + row] * b_vals[k * columns + column]
            }
            assert_near(got[row * columns + column], want, 2e-2)
        }
    }
}

fn test_mps_fashion_weight_grad_matmul_matches_cpu() -> void {
    if not jt.is_available() { return }
    let rows = 784
    let inner = 512
    let columns = 256
    let a = jt.Tensor([inner, rows], [float(i * 3 % 11) for i in 0..inner * rows])
    let b = jt.Tensor([inner, columns], [float(i * 5 % 7) for i in 0..inner * columns])
    defer { a.free() }
    defer { b.free() }
    let product = jt.matmul(a, b, trans_a: true)
    defer { product.free() }
    let got = product.download()
    let a_vals = a.download()
    let b_vals = b.download()
    for row in [0, 100, 783] {
        for column in [0, 1, 64, 255] {
            var want = 0.0
            for k in 0..inner {
                want += a_vals[k * rows + row] * b_vals[k * columns + column]
            }
            assert_near(got[row * columns + column], want, 2e-2)
        }
    }
}

fn test_mps_cifar_layer_matmul_matches_cpu() -> void {
    if not jt.is_available() { return }
    assert_gpu_matmul(512, 3072, 512, tol: 2e-2)
}

fn test_staged_64_transposed_matmul_matches_cpu() -> void {
    if not jt.is_available() { return }
    let m = 64
    let k = 1024
    let n = 64
    let a_tn = jt.Tensor([k, m], [float(i * 3 % 11) for i in 0..k * m])
    let b_tn = jt.Tensor([k, n], [float(i * 5 % 7) for i in 0..k * n])
    defer { a_tn.free() }
    defer { b_tn.free() }
    let tn = jt.matmul(a_tn, b_tn, trans_a: true)
    defer { tn.free() }
    let a_nt = jt.Tensor([m, k], [float(i * 3 % 11) for i in 0..m * k])
    let b_nt = jt.Tensor([n, k], [float(i * 5 % 7) for i in 0..n * k])
    defer { a_nt.free() }
    defer { b_nt.free() }
    let nt = jt.matmul(a_nt, b_nt, trans_b: true)
    defer { nt.free() }
    let a_tn_vals = a_tn.download()
    let b_tn_vals = b_tn.download()
    let a_nt_vals = a_nt.download()
    let b_nt_vals = b_nt.download()
    let tn_got = tn.download()
    let nt_got = nt.download()
    for row in 0..m {
        for column in 0..n {
            var want_tn = 0.0
            var want_nt = 0.0
            for inner in 0..k {
                want_tn += a_tn_vals[inner * m + row] * b_tn_vals[inner * n + column]
                want_nt += a_nt_vals[row * k + inner] * b_nt_vals[column * k + inner]
            }
            assert_near(tn_got[row * n + column], want_tn, 5e-3)
            assert_near(nt_got[row * n + column], want_nt, 5e-3)
        }
    }
}

fn test_staged_64_matmul_covers_row_remainder() -> void {
    if not jt.is_available() { return }
    assert_gpu_matmul(80, 1024, 64)
}

fn test_staged_64_matmul_covers_l_remainder() -> void {
    if not jt.is_available() { return }
    assert_gpu_matmul(80, 1024, 100)
}

fn test_staged_64_matmul_ragged_k_matches_cpu() -> void {
    if not jt.is_available() { return }
    assert_gpu_matmul(64, 784, 1024)
}

fn test_mma_ragged_k_transposed_matmul_matches_cpu() -> void {
    if not jt.is_available() { return }
    let m = 64
    let k = 100
    let n = 64
    let a_nt = jt.Tensor([m, k], [float(i * 3 % 11) for i in 0..m * k])
    let b_nt = jt.Tensor([n, k], [float(i * 5 % 7) for i in 0..n * k])
    defer { a_nt.free() }
    defer { b_nt.free() }
    let nt = jt.matmul(a_nt, b_nt, trans_b: true)
    defer { nt.free() }
    let a_vals = a_nt.download()
    let b_vals = b_nt.download()
    let got = nt.download()
    for row in 0..m {
        for column in 0..n {
            var want = 0.0
            for inner in 0..k {
                want += a_vals[row * k + inner] * b_vals[column * k + inner]
            }
            assert_near(got[row * n + column], want, 5e-3)
        }
    }
}

fn test_aligned_256_matmul_matches_cpu() -> void {
    if not jt.is_available() { return }
    assert_gpu_matmul(64, 256, 256)
}

fn test_fused_64_relu_affine_backward_matches_cpu() -> void {
    if not jt.is_available() { return }
    let batch = 64
    let inputs = 1024
    let units = 64
    let x_vals = [float(i * 3 % 11) / 11.0 for i in 0..batch * inputs]
    let w_vals = [float(i * 5 % 7) / 7.0 - 0.5 for i in 0..inputs * units]
    let b_vals = [float(i % 5) / 5.0 - 0.4 for i in 0..units]
    let layer = jt.Dense(units, activation: jt.Activation.Relu)
    let rng = Random.with_seed(1)
    layer.build(inputs, rng)
    layer.set_weights(w_vals, b_vals)
    defer { layer.free() }
    let x = jt.Tensor([batch, inputs], x_vals, requires_grad: true)
    defer { x.free() }
    let y = layer.forward(x)
    defer { y.free() }
    let total = reduce_to_scalar_for_test(y)
    defer { total.free() }
    let intermediates = total.backward()
    defer {
        for tensor in intermediates { tensor.free() }
    }
    let params = layer.parameters()
    let grad_w = params[0].grad
    let grad_b = params[1].grad
    let grad_x = x.grad
    if grad_w is null or grad_b is null or grad_x is null {
        throw AssertionError("expected fused relu affine grads")
    }
    let gw = grad_w.download()
    let gb = grad_b.download()
    let gx = grad_x.download()
    var d_aff = [0.0 for _i in 0..batch * units]
    for row in 0..batch {
        for unit in 0..units {
            var aff = b_vals[unit]
            for k in 0..inputs {
                aff += x_vals[row * inputs + k] * w_vals[k * units + unit]
            }
            d_aff[row * units + unit] = if aff > 0.0 { 1.0 } else { 0.0 }
        }
    }
    for unit in 0..units {
        var want_b = 0.0
        for row in 0..batch { want_b += d_aff[row * units + unit] }
        assert_near(gb[unit], want_b, 5e-3)
    }
    for k in 0..inputs {
        for unit in 0..units {
            var want_w = 0.0
            for row in 0..batch {
                want_w += x_vals[row * inputs + k] * d_aff[row * units + unit]
            }
            assert_near(gw[k * units + unit], want_w, 5e-3)
        }
    }
    for row in 0..batch {
        for k in 0..inputs {
            var want_x = 0.0
            for unit in 0..units {
                want_x += d_aff[row * units + unit] * w_vals[k * units + unit]
            }
            assert_near(gx[row * inputs + k], want_x, 5e-3)
        }
    }
}

fn test_staged_128_matmul_matches_cpu() -> void {
    if not jt.is_available() { return }
    assert_gpu_matmul(128, 1024, 128)
}

fn test_staged_128_transposed_matmul_matches_cpu() -> void {
    if not jt.is_available() { return }
    let m = 128
    let k = 1024
    let n = 128
    let a_tn = jt.Tensor([k, m], [float(i * 3 % 11) for i in 0..k * m])
    let b_tn = jt.Tensor([k, n], [float(i * 5 % 7) for i in 0..k * n])
    defer { a_tn.free() }
    defer { b_tn.free() }
    let tn = jt.matmul(a_tn, b_tn, trans_a: true)
    defer { tn.free() }
    let a_nt = jt.Tensor([m, k], [float(i * 3 % 11) for i in 0..m * k])
    let b_nt = jt.Tensor([n, k], [float(i * 5 % 7) for i in 0..n * k])
    defer { a_nt.free() }
    defer { b_nt.free() }
    let nt = jt.matmul(a_nt, b_nt, trans_b: true)
    defer { nt.free() }
    let a_tn_vals = a_tn.download()
    let b_tn_vals = b_tn.download()
    let a_nt_vals = a_nt.download()
    let b_nt_vals = b_nt.download()
    let tn_got = tn.download()
    let nt_got = nt.download()
    for row in 0..m {
        for column in 0..n {
            var want_tn = 0.0
            var want_nt = 0.0
            for inner in 0..k {
                want_tn += a_tn_vals[inner * m + row] * b_tn_vals[inner * n + column]
                want_nt += a_nt_vals[row * k + inner] * b_nt_vals[column * k + inner]
            }
            assert_near(tn_got[row * n + column], want_tn, 5e-3)
            assert_near(nt_got[row * n + column], want_nt, 5e-3)
        }
    }
}

fn test_mixed_precision_matmul_stays_close() -> void {
    if not jt.is_available() { return }
    let a = jt.Tensor(
        [4, 4],
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    )
    let b = jt.Tensor(
        [4, 4],
        [2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 5.0]
    )
    jt.set_mixed_precision(true)
    let y = jt.matmul(a, b)
    jt.set_mixed_precision(false)
    defer { a.free() }
    defer { b.free() }
    defer { y.free() }
    let values = y.download()
    assert_near(values[0], 2.0, 1e-3)
    assert_near(values[5], 3.0, 1e-3)
    assert_near(values[10], 4.0, 1e-3)
    assert_near(values[15], 5.0, 1e-3)
}
