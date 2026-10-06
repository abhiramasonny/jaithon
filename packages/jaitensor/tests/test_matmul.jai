#: The matmul routes: tile plans, the staged and MPS kernels, and the fused affine.
import jaitensor as jt
import std.gpu as gpu
from jaitensor.gemm import Plan, bucket_extent, default_plan, dispatch_plan, plan_from_text, reload_plans
from jaitensor.ops import matmul_backend
from jaitensor.tensor import Tensor
from std.io import Path
from std.os import env, pid, set_env, temp_dir, unset_env
from std.random import Random
from std.test import assert_eq, assert_near, assert_true

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
    #: Ragged shapes keep the staged kernel, the largest tile that still makes
    #: a few threadgroups, then splits.
    assert_eq(default_plan(8, 7, 5).to_str(), "gemm32x32:1")
    assert_eq(default_plan(100, 100, 100).to_str(), "gemm32x32:1")
    #: Large products go to the graph wherever a device can run it.
    if gpu.is_available() {
        assert_eq(default_plan(512, 784, 8192).to_str(), "mpsgraph:1")
        assert_eq(default_plan(1024, 768, 768).to_str(), "mpsgraph:1")
    }
}

fn test_default_plan_splits_a_tall_skinny_product_past_sixteen() -> void {
    #: X^T X and X^T y over 10k samples: four and two output tiles, so
    #: sixteen splits left the device mostly idle.
    assert_eq(default_plan(50, 10000, 50).to_str(), "gemm32x32:32")
    assert_eq(default_plan(50, 10000, 1).to_str(), "gemm32x32:32")
    #: A product with tiles enough of its own is never split that far.
    assert_eq(default_plan(8, 7, 5).to_str(), "gemm32x32:1")
    #: Deeper still, the graph: it splits K itself, and beat every kernel
    #: plan at 100k samples.
    if gpu.is_available() {
        assert_eq(default_plan(50, 100000, 50).to_str(), "mpsgraph:1")
        assert_eq(default_plan(50, 100000, 1).to_str(), "mpsgraph:1")
    }
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
    let want = host_reference(a.download(), b.download(), rows, inner, columns, trans_a, trans_b)
    let what = f"{plan.to_str()} {rows}x{inner}x{columns} ta={trans_a} tb={trans_b}"
    for at in 0..want.len() {
        if not (got[at] - want[at] < 1e-3 * (1.0 + inner) and want[at] - got[at] < 1e-3 * (1.0 + inner)) {
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
