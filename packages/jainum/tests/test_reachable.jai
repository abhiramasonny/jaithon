#: Every name `import jainum` exports, called once, through the facade.
#:
#: The per-module tests ask what a function computes. This one asks the
#: cheaper question first, and of the surface rather than the implementation:
#: is the name exported at all, and does it survive being called with
#: arguments it should accept. `mod.jai` is the only file in the package with
#: no logic in it, so its bugs are all of one shape — the export list and the
#: module it re-exports from have drifted apart. A name that moved, a
#: signature that grew a required argument, a re-export nobody ever reached.
#: `jaithon check` catches the first of those and neither of the other two,
#: which is what this file is for.
#:
#: The contract is deliberately weak and deliberately wide, the way jaicv's
#: file of the same name is: every export is reached, and what is asserted is
#: usually only that the result carries the shape and the dtype it promised.
#: Where a second, independent computation is cheap — a host fold over the
#: same values, a solution substituted back into its own equation, a direct
#: DFT sum — it is written out and compared against, because a smoke test that
#: a kernel returning its own input would pass is not worth the seconds it
#: costs.
#:
#: `linalg`, `random` and `fft` are namespaces rather than flat exports, so
#: what is checked of them here is that the namespace is reachable and that
#: one representative call through it works. Their contents have files of
#: their own.
#:
#: Everything that touches storage returns early when there is no device.
import std.gpu as gpu
import std.math
import jainum as nn
from std.test import assert_eq, assert_near, assert_ne, assert_sorted
from std.test import assert_throws, assert_true
from jainum import BOOL, COMPLEX64, FLOAT32, INT32, NDArray

fn have_gpu() -> bool { return nn.is_available() }

#: An array holding exactly these values, so the host reference can be written
#: from the same list rather than read back off the device.
fn holding(shape: list[int], values: list[float], dtype: int = FLOAT32) -> NDArray {
    let a = NDArray(shape, dtype)
    a.upload(values)
    return a
}

fn ramp(shape: list[int], dtype: int = FLOAT32) -> NDArray {
    return holding(shape, [float(index) + 1.0 for index in 0..nn.element_count(shape)], dtype)
}

#: The one question every call below is asked: it returned, and what came back
#: is the shape it promised.
fn ran(a: NDArray, shape: list[int]) -> void { assert_eq(a.shape, shape) }

#: Elementwise agreement with a host reference, at float32's tolerance rather
#: than `assert_near`'s float64 default.
fn matches(a: NDArray, expected: list[float], tolerance: float = 1.0e-5) -> void {
    let got = a.to_list()
    assert_eq(got.len(), expected.len())
    for index in 0..expected.len() {
        assert_near(got[index], expected[index], tolerance, f"at index {index}")
    }
}

fn test_the_facade_answers_for_the_package() -> void {
    assert_eq(nn.VERSION, "0.1.0")
    assert_eq(nn.is_available(), gpu.is_available())
    assert_true(nn.device_count() >= 0)
    if not have_gpu() {
        assert_eq(nn.device_name(), "none")
        return
    }
    assert_ne(nn.device_name(), "none")
    assert_true(nn.device_count() >= 1)

    #: `set_device` is the one device call that changes something, and it only
    #: works before the first allocation exists — which, in a suite, it does.
    #: Reaching it through the index it refuses is the honest way to call it.
    assert_throws(RuntimeError, || nn.set_device(nn.device_count() + 8))
}

fn test_the_pure_surface_needs_no_device() -> void {
    assert_eq(nn.element_count([2, 3, 4]), 24)
    assert_eq(nn.normalize_axis(-1, 3), 2)
    assert_eq(nn.normalize_axes([-1, 0], 3), [2, 0])
    assert_eq(nn.contiguous_strides([2, 3], 1), [3, 1])
    assert_true(nn.is_contiguous([2, 3], [3, 1], 1))
    assert_eq(nn.broadcast_shapes([3, 1], [4]), [3, 4])
    assert_eq(nn.broadcast_strides([3, 1], [1, 1], [2, 3, 4]), [0, 1, 0])

    assert_eq(nn.promote(INT32, FLOAT32), FLOAT32)
    assert_eq(nn.promote(nn.FLOAT64, COMPLEX64), COMPLEX64)
    assert_eq(nn.dtype_name(nn.INT64), "int64")
    assert_eq(nn.dtype_lanes(COMPLEX64), 2)
    assert_eq(nn.dtype_lanes(nn.FLOAT64), 1)
    assert_true(nn.dtype_is_integer(INT32))
    assert_true(nn.dtype_is_bool(BOOL))
    assert_true(nn.dtype_is_complex(COMPLEX64))
    assert_eq(nn.dtype_min(BOOL), 0.0)
    assert_eq(nn.dtype_max(BOOL), 1.0)
    assert_eq(nn.MAX_EXACT_INT, 16777216)
    assert_eq(nn.GROUP, 256)

    #: The saturating bound has to be a float32 the integer dtype can hold, or
    #: a clamp against it lets through a value the type cannot.
    assert_true(nn.dtype_max(INT32) < 2147483648.0)
    assert_eq(nn.dtype_min(INT32), -nn.dtype_min(INT32) * -1.0)

    assert_eq(nn.f32_from_bits(nn.f32_bits(-2.5)), -2.5)
    assert_eq(nn.f32_from_bits(nn.f32_bits(0.0)), 0.0)

    let (first, count) = nn.Slice(1, 4).resolve(10)
    assert_eq(first, 1)
    assert_eq(count, 3)
    assert_eq(nn.sl(null, null, -1).step, -1)
    assert_eq(nn.NEWAXIS.name, "newaxis")
    assert_eq(nn.ELLIPSIS.name, "Ellipsis")
    assert_eq(nn.IndexToken("newaxis").name, nn.NEWAXIS.name)

    #: The selector families are only useful if their members differ, which is
    #: worth asserting where the values themselves are not.
    assert_eq({nn.OP_ADD, nn.OP_SUB, nn.OP_MUL, nn.OP_DIV}.len(), 4)
    assert_eq({nn.CAST_NONE, nn.CAST_ROUND, nn.CAST_BOOL}.len(), 3)
    assert_eq({nn.KIND_KEEP, nn.KIND_FLOAT, nn.KIND_BOOL, nn.KIND_INT}.len(), 4)
    assert_eq({nn.PAD_EDGE, nn.PAD_WRAP, nn.PAD_REFLECT}.len(), 3)

    assert_true(nn.HOST_SORT_LIMIT > 0)
    assert_true(nn.FLOAT32_MAX > 3.0e38)
    assert_true(nn.PRELUDE.starts_with("#include <metal_stdlib>"))
    assert_eq(nn.linspace_step(0.0, 1.0, 5), 0.25)
    assert_eq(nn.linspace_step(0.0, 1.0, 5, false), 0.2)
}

fn test_every_creation_export_runs() -> void {
    if not have_gpu() { return }
    ran(nn.empty([2, 3]), [2, 3])
    assert_eq(nn.zeros([4]).to_list(), [0.0, 0.0, 0.0, 0.0])
    assert_eq(nn.ones([3]).to_list(), [1.0, 1.0, 1.0])
    assert_eq(nn.full([2], 2.5).to_list(), [2.5, 2.5])

    let seed = ramp([2, 3])
    ran(nn.empty_like(seed), [2, 3])
    assert_eq(nn.zeros_like(seed).to_list(), [0.0 for _index in 0..6])
    assert_eq(nn.ones_like(seed).to_list(), [1.0 for _index in 0..6])
    assert_eq(nn.full_like(seed, 4.0).to_list(), [4.0 for _index in 0..6])

    assert_eq(nn.arange(4).to_list(), [0.0, 1.0, 2.0, 3.0])
    assert_eq(nn.arange(1.0, 2.0, 0.25).to_list(), [1.0, 1.25, 1.5, 1.75])
    matches(nn.linspace(0.0, 1.0, 5), [0.0, 0.25, 0.5, 0.75, 1.0])
    matches(nn.logspace(0.0, 2.0, 3), [1.0, 10.0, 100.0], 1.0e-3)
    matches(nn.geomspace(1.0, 8.0, 4), [1.0, 2.0, 4.0, 8.0], 1.0e-4)

    matches(nn.eye(2), [1.0, 0.0, 0.0, 1.0])
    matches(nn.eye(2, 3, 1), [0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    assert_eq(nn.identity(2).to_list(), nn.eye(2).to_list())
    matches(nn.tri(2), [1.0, 0.0, 1.0, 1.0])
    matches(nn.tril(ramp([2, 2])), [1.0, 0.0, 3.0, 4.0])
    matches(nn.triu(ramp([2, 2])), [1.0, 2.0, 0.0, 4.0])
    matches(nn.diag(holding([2], [5.0, 7.0])), [5.0, 0.0, 0.0, 7.0])
    matches(nn.diag(ramp([2, 2])), [1.0, 4.0])
    ran(nn.diagflat(ramp([2, 2])), [4, 4])

    matches(nn.array([[1.0, 2.0], [3.0, 4.0]]), [1.0, 2.0, 3.0, 4.0])
    ran(nn.array([[1.0, 2.0], [3.0, 4.0]]), [2, 2])
    ran(nn.asarray([1.0, 2.0, 3.0]), [3])
    matches(nn.from_list([1.0, 2.0, 3.0, 4.0], [2, 2]), [1.0, 2.0, 3.0, 4.0])
    matches(nn.from_bytes(bytes([0, 255]), [2], FLOAT32, 0, 1.0 / 255.0), [0.0, 1.0])
    ran(nn.frombuffer(bytes([1, 2, 3])), [3])

    let grids = nn.meshgrid([holding([2], [0.0, 1.0]), holding([3], [0.0, 1.0, 2.0])])
    assert_eq(grids.len(), 2)
    ran(grids[0], [3, 2])
    ran(nn.indices([2, 3]), [2, 2, 3])

    #: The expression is Metal, and its answer is one this test can write down
    #: without running a kernel.
    matches(nn.fromfunction("i0 * 10.0 + i1", [2, 3]), [0.0, 1.0, 2.0, 10.0, 11.0, 12.0])
}

fn test_every_unary_ufunc_export_runs() -> void {
    if not have_gpu() { return }
    let values = [0.125, 0.25, 0.5, 0.75]
    let unit = holding([4], values)
    let signed = holding([4], [-1.75, -0.5, 0.5, 1.75])
    let above = holding([4], [1.25, 2.0, 3.0, 4.0])

    for result in [
        nn.negative(unit),
        nn.positive(unit),
        nn.absolute(signed),
        nn.sign(signed),
        nn.reciprocal(unit),
        nn.sqrt(unit),
        nn.cbrt(signed),
        nn.square(signed),
        nn.exp(unit),
        nn.exp2(unit),
        nn.expm1(unit),
        nn.log(unit),
        nn.log2(unit),
        nn.log10(unit),
        nn.log1p(unit),
        nn.sin(unit),
        nn.cos(unit),
        nn.tan(unit),
        nn.arcsin(unit),
        nn.arccos(unit),
        nn.arctan(unit),
        nn.sinh(unit),
        nn.cosh(unit),
        nn.tanh(unit),
        nn.arcsinh(unit),
        nn.arccosh(above),
        nn.arctanh(unit),
        nn.deg2rad(unit),
        nn.rad2deg(unit),
        nn.radians(unit),
        nn.degrees(unit),
        nn.floor(signed),
        nn.ceil(signed),
        nn.trunc(signed),
        nn.rint(signed),
        nn.round(signed),
        nn.round(signed, 1),
        nn.isnan(unit),
        nn.isinf(unit),
        nn.isfinite(unit),
        nn.logical_not(unit),
        nn.nan_to_num(unit),
        nn.angle(signed),
        nn.angle(signed, true),
        nn.clip(signed, -1.0, 1.0),
        nn.clip(signed, -1.0),
        nn.clip(signed, null, 1.0),
    ] {
        ran(result, [4])
    }

    let integral = holding([4], [1.0, 2.0, 3.0, 4.0], INT32)
    ran(nn.bitwise_invert(integral), [4])
    ran(nn.invert(integral), [4])

    #: Six of them checked against the host library, on the same four values
    #: the device was handed.
    matches(nn.exp(unit), [math.exp(value) for value in values])
    matches(nn.log(unit), [math.log(value) for value in values])
    matches(nn.sqrt(unit), [math.sqrt(value) for value in values])
    matches(nn.sin(unit), [math.sin(value) for value in values])
    matches(nn.cos(unit), [math.cos(value) for value in values])
    matches(nn.cbrt(unit), [math.cbrt(value) for value in values])

    #: Rounding is to nearest even, which is the one place a "round" that
    #: agrees with intuition disagrees with numpy.
    matches(nn.rint(holding([4], [-1.5, -0.5, 0.5, 1.5])), [-2.0, -0.0, 0.0, 2.0])

    #: An infinity is a value these three have to recognise, and one a device
    #: division produces. NaN is left to `test_ufunc.jai`, which owns the
    #: question of what the Metal compiler's fast paths do with it.
    let infinite = holding([2], [1.0, 1.0]) / holding([2], [0.0, 1.0])
    matches(nn.isinf(infinite), [1.0, 0.0])
    matches(nn.isfinite(infinite), [0.0, 1.0])
    matches(nn.nan_to_num(infinite), [nn.FLOAT32_MAX, 1.0], 1.0e32)
}

fn test_every_binary_ufunc_export_runs() -> void {
    if not have_gpu() { return }
    let left_values = [3.0, -4.0, 2.5, -1.5]
    let right_values = [2.0, 3.0, -2.0, 4.0]
    let left = holding([4], left_values)
    let right = holding([4], right_values)
    let positive = holding([4], [1.0, 2.0, 3.0, 4.0])

    for result in [
        nn.add(left, right),
        nn.subtract(left, right),
        nn.multiply(left, right),
        nn.divide(left, right),
        nn.true_divide(left, right),
        nn.add(left, 2.0),
        nn.power(positive, right),
        nn.float_power(positive, right),
        nn.remainder(left, right),
        nn.mod(left, right),
        nn.fmod(left, right),
        nn.floor_divide(left, right),
        nn.hypot(left, right),
        nn.arctan2(left, right),
        nn.logaddexp(left, right),
        nn.maximum(left, right),
        nn.minimum(left, right),
        nn.fmax(left, right),
        nn.fmin(left, right),
        nn.copysign(left, right),
        nn.greater(left, right),
        nn.greater_equal(left, right),
        nn.less(left, right),
        nn.less_equal(left, right),
        nn.equal(left, right),
        nn.not_equal(left, right),
        nn.logical_and(left, right),
        nn.logical_or(left, right),
        nn.logical_xor(left, right),
        nn.isclose(left, right),
        nn.where(nn.greater(left, right), left, right),
    ] {
        ran(result, [4])
    }

    let keys = holding([4], [6.0, 3.0, 12.0, 5.0], INT32)
    let masks = holding([4], [3.0, 1.0, 10.0, 2.0], INT32)
    for result in [
        nn.bitwise_and(keys, masks),
        nn.bitwise_or(keys, masks),
        nn.bitwise_xor(keys, masks),
        nn.left_shift(keys, masks),
        nn.right_shift(keys, masks),
    ] {
        ran(result, [4])
    }

    #: Four of them against their definitions, computed a second time on the
    #: host from the same two lists.
    matches(
        nn.hypot(left, right),
        [math.sqrt(left_values[index] * left_values[index] + right_values[index] * right_values[index]) for index in 0..4]
    )
    matches(
        nn.arctan2(left, right),
        [math.atan2(left_values[index], right_values[index]) for index in 0..4]
    )
    matches(
        nn.logaddexp(left, right),
        [math.log(math.exp(left_values[index]) + math.exp(right_values[index])) for index in 0..4],
        1.0e-4
    )
    matches(nn.greater(left, right), [if left_values[index] > right_values[index] {
        1.0
    } else {
        0.0
    } for index in 0..4])

    #: A comparison is `BOOL` whatever it compared, which is what makes it a
    #: mask rather than a number.
    assert_eq(nn.greater(left, right).dtype, BOOL)
    assert_eq(nn.equal(left, left).dtype, BOOL)

    #: The one binary that broadcasts a shape rather than a scalar, which is
    #: the whole point of a ufunc.
    ran(nn.multiply(ramp([2, 3]), holding([3], [1.0, 10.0, 100.0])), [2, 3])
}

fn test_every_reduction_export_runs() -> void {
    if not have_gpu() { return }
    let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    let block = holding([2, 3], values)

    assert_near(nn.sum(block).item(), 21.0, 1.0e-5)
    assert_near(nn.prod(block).item(), 720.0, 1.0e-3)
    assert_near(nn.amin(block).item(), 1.0)
    assert_near(nn.amax(block).item(), 6.0)
    assert_near(nn.min(block).item(), 1.0)
    assert_near(nn.max(block).item(), 6.0)
    assert_near(nn.mean(block).item(), 3.5, 1.0e-5)
    assert_near(nn.ptp(block).item(), 5.0)
    assert_near(nn.count_nonzero(block).item(), 6.0)
    assert_near(nn.any(block).item(), 1.0)
    assert_near(nn.all(block).item(), 1.0)
    assert_near(nn.argmin(block).item(), 0.0)
    assert_near(nn.argmax(block).item(), 5.0)
    assert_near(nn.trace(ramp([2, 2])).item(), 5.0)

    for result in [
        nn.nansum(block),
        nn.nanmin(block),
        nn.nanmax(block),
        nn.nanmean(block),
        nn.nanvar(block),
        nn.nanstd(block),
        nn.variance(block),
        nn.std(block),
        nn.nanargmin(block),
        nn.nanargmax(block),
    ] {
        ran(result, [])
    }
    #: A scan with no axis flattens first, which is numpy's rule and not an
    #: accident of the kernel.
    for result in [nn.cumsum(block), nn.cumprod(block), nn.cummax(block), nn.cummin(block)] {
        ran(result, [6])
    }
    ran(nn.cumsum(block, axis: 1), [2, 3])
    ran(nn.sum(block, axis: 1, keepdims: true), [2, 1])
    ran(nn.variance(block, ddof: 1), [])

    assert_true(nn.allclose(block, block))
    assert_true(nn.array_equal(block, block))
    assert_true(not nn.array_equal(block, nn.add(block, 1.0)))
    assert_true(not nn.allclose(block, nn.add(block, 1.0)))

    #: An axis fold against the same fold run on the host, over the list the
    #: array was built from.
    var rows: list[float] = []
    for row in 0..2 {
        var total = 0.0
        for column in 0..3 { total += values[row * 3 + column] }
        rows.push(total)
    }
    matches(nn.sum(block, axis: 1), rows)

    #: Variance is the mean of the squared deviations, and saying so a second
    #: time costs four lines.
    var spread = 0.0
    for value in values { spread += (value - 3.5) * (value - 3.5) }
    assert_near(nn.variance(block).item(), spread / 6.0, 1.0e-5)
    assert_near(nn.std(block).item(), math.sqrt(spread / 6.0), 1.0e-5)
}

fn test_every_manipulation_export_runs() -> void {
    if not have_gpu() { return }
    let block = ramp([2, 3])
    let pair = ramp([2, 2])

    ran(nn.reshape(block, [3, 2]), [3, 2])
    ran(nn.ravel(block), [6])
    ran(nn.flatten(block), [6])
    ran(nn.resize(block, [3, 3]), [3, 3])
    ran(nn.squeeze(ramp([1, 3])), [3])
    ran(nn.expand_dims(block, 0), [1, 2, 3])
    ran(nn.transpose(block), [3, 2])
    ran(nn.permute_dims(block, [1, 0]), [3, 2])
    ran(nn.swapaxes(block, 0, 1), [3, 2])
    ran(nn.moveaxis(block, 0, 1), [3, 2])
    ran(nn.rollaxis(block, 1), [3, 2])
    ran(nn.atleast_1d(ramp([])), [1])
    ran(nn.atleast_2d(ramp([3])), [1, 3])
    ran(nn.atleast_3d(ramp([3])), [1, 3, 1])
    ran(nn.broadcast_to(ramp([3]), [2, 3]), [2, 3])
    ran(nn.diagonal(pair), [2])
    ran(nn.concatenate([block, block]), [4, 3])
    ran(nn.stack([block, block]), [2, 2, 3])
    ran(nn.vstack([block, block]), [4, 3])
    ran(nn.hstack([block, block]), [2, 6])
    ran(nn.dstack([block, block]), [2, 3, 2])
    ran(nn.column_stack([ramp([3]), ramp([3])]), [3, 2])
    ran(nn.tile(ramp([2]), [2]), [4])
    ran(nn.repeat(ramp([2]), 3), [6])
    ran(nn.flip(block), [2, 3])
    ran(nn.fliplr(block), [2, 3])
    ran(nn.flipud(block), [2, 3])
    ran(nn.roll(block, 1), [2, 3])
    ran(nn.rot90(block), [3, 2])
    ran(nn.pad(block, [[1, 1], [0, 0]]), [4, 3])
    ran(nn.pad(block, [[1, 1], [0, 0]], "edge"), [4, 3])

    assert_eq(nn.broadcast_arrays([ramp([3]), ramp([2, 1])])[0].shape, [2, 3])
    assert_eq(nn.split(block, 3, 1).len(), 3)
    assert_eq(nn.array_split(block, 2, 1).len(), 2)
    assert_eq(nn.vsplit(block, 2).len(), 2)
    assert_eq(nn.hsplit(block, 3).len(), 3)
    assert_eq(nn.dsplit(ramp([1, 1, 4]), 2).len(), 2)

    #: A reshape and a ravel are each other's inverse, and a flip is the host
    #: list read backwards. Both are cheap to say twice.
    assert_eq(nn.ravel(nn.reshape(block, [3, 2])).to_list(), block.to_list())
    let straight = ramp([4]).to_list()
    matches(nn.flip(ramp([4])), [straight[3 - index] for index in 0..4])
}

fn test_every_sorting_export_runs() -> void {
    if not have_gpu() { return }
    let scattered = holding([6], [4.0, 1.0, 5.0, 1.0, 3.0, 2.0])

    ran(nn.sort(scattered), [6])
    ran(nn.argsort(scattered), [6])
    ran(nn.partition(scattered, 2), [6])
    ran(nn.argpartition(scattered, 2), [6])
    ran(nn.searchsorted(nn.sort(scattered), holding([2], [2.5, 4.5])), [2])
    ran(nn.searchsorted(nn.sort(scattered), holding([2], [1.0, 3.0]), "right"), [2])
    ran(nn.isin(scattered, holding([2], [1.0, 5.0])), [6])
    ran(nn.in1d(scattered, holding([2], [1.0, 5.0])), [6])
    ran(nn.lexsort([scattered, holding([6], [1.0, 1.0, 2.0, 2.0, 3.0, 3.0])]), [6])

    let distinct = nn.unique(scattered, true, true, true)
    assert_true(distinct is nn.UniqueResult)
    assert_eq(distinct.values.to_list(), [1.0, 2.0, 3.0, 4.0, 5.0])
    assert_true(distinct.indices is not null)
    assert_true(distinct.inverse is not null)
    assert_eq(distinct.counts.to_list(), [2.0, 1.0, 1.0, 1.0, 1.0])

    let other = holding([3], [3.0, 5.0, 9.0])
    assert_eq(nn.union1d(scattered, other).to_list(), [1.0, 2.0, 3.0, 4.0, 5.0, 9.0])
    assert_eq(nn.intersect1d(scattered, other).to_list(), [3.0, 5.0])
    assert_eq(nn.setdiff1d(scattered, other).to_list(), [1.0, 2.0, 4.0])
    assert_eq(nn.setxor1d(scattered, other).to_list(), [1.0, 2.0, 4.0, 9.0])

    #: A sort is ascending and holds the same multiset it started with, which
    #: is the definition and not a recorded answer.
    let ordered = nn.sort(scattered).to_list()
    assert_sorted([value for value in ordered])
    var before = scattered.to_list()
    before.sort()
    assert_eq(ordered, before)
}

fn test_every_indexing_export_runs() -> void {
    if not have_gpu() { return }
    let flat = holding([6], [4.0, 0.0, 5.0, 0.0, 3.0, 2.0])
    let block = ramp([2, 3])
    let keys = holding([2], [0.0, 2.0], INT32)
    let mask = holding([6], [1.0, 0.0, 1.0, 0.0, 1.0, 0.0], BOOL)

    assert_eq(nn.take(flat, keys).to_list(), [4.0, 5.0])
    assert_eq(nn.take(block, keys, 1).shape, [2, 2])
    assert_eq(nn.take(flat, holding([1], [9.0], INT32), null, "clip").to_list(), [2.0])
    ran(nn.take_along_axis(block, holding([2, 1], [0.0, 2.0], INT32), 1), [2, 1])
    ran(nn.mask_select(flat, mask), [3])
    ran(nn.extract(mask, flat), [3])
    ran(nn.compress(holding([2], [1.0, 0.0], BOOL), block, 0), [1, 3])
    ran(nn.choose(holding([3], [0.0, 1.0, 0.0], INT32), [ramp([3]), nn.zeros([3])]), [3])
    ran(nn.select([nn.greater(flat, 2.0)], [flat], -1.0), [6])
    ran(nn.repeat_by_counts(ramp([3]), holding([3], [1.0, 0.0, 2.0], INT32)), [3])
    ran(nn.fancy_index(flat, keys), [2])
    assert_eq(nn.ix_([holding([2], [0.0, 1.0], INT32), holding([2], [0.0, 2.0], INT32)]).len(), 2)
    assert_eq(nn.index_exp((0, nn.sl(1, 3))).len(), 2)
    assert_eq(nn.nonzero(flat).len(), 1)
    assert_eq({nn.MODE_RAISE, nn.MODE_WRAP, nn.MODE_CLIP}.len(), 3)

    #: `nonzero`, `flatnonzero` and `argwhere` are three spellings of one
    #: question, so the host answers it too and all four have to agree.
    var expected: list[float] = []
    let values = flat.to_list()
    for index in 0..6 { if values[index] != 0.0 { expected.push(float(index)) } }
    assert_eq(nn.flatnonzero(flat).to_list(), expected)
    assert_eq(nn.nonzero(flat)[0].to_list(), expected)
    assert_eq(nn.argwhere(flat).to_list(), expected)
    assert_eq(nn.mask_select(flat, mask).to_list(), [4.0, 5.0, 3.0])

    #: The four writing entry points, each read back.
    let target = nn.zeros([6])
    nn.put(target, keys, holding([2], [7.0, 8.0]))
    assert_eq(target.to_list(), [7.0, 0.0, 8.0, 0.0, 0.0, 0.0])
    nn.place(target, mask, holding([1], [1.0]))
    assert_eq(target.to_list(), [1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    nn.putmask(target, mask, holding([1], [2.0]))
    assert_eq(target.to_list(), [2.0, 0.0, 2.0, 0.0, 2.0, 0.0])
    let rows = nn.zeros([2, 3])
    nn.put_along_axis(rows, holding([2, 1], [1.0, 2.0], INT32), holding([2, 1], [5.0, 6.0]), 1)
    assert_eq(rows.to_list(), [0.0, 5.0, 0.0, 0.0, 0.0, 6.0])
    let square = nn.zeros([3, 3])
    nn.fill_diagonal(square, 4.0)
    assert_eq(square.to_list(), [4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.0])
}

fn test_every_matmul_export_runs() -> void {
    if not have_gpu() { return }
    let a_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    let b_values = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    let a = holding([2, 3], a_values)
    let b = holding([3, 2], b_values)

    ran(nn.matmul(a, b), [2, 2])
    ran(nn.dot(a, b), [2, 2])
    ran(nn.inner(ramp([2, 3]), ramp([4, 3])), [2, 4])
    ran(nn.outer(ramp([3]), ramp([4])), [3, 4])
    ran(nn.tensordot(ramp([2, 3]), ramp([3, 4]), 1), [2, 4])
    ran(nn.kron(ramp([2, 2]), ramp([2, 2])), [4, 4])
    ran(nn.cross(ramp([2, 3]), ramp([2, 3])), [2, 3])
    ran(nn.matrix_power(ramp([2, 2]), 3), [2, 2])
    ran(nn.einsum("ij,jk->ik", a, b), [2, 2])
    ran(nn.vdot(ramp([2, 3]), ramp([6])), [])

    #: The product against the same triple loop run on the host, which is what
    #: a matrix product is.
    var expected = [0.0 for _index in 0..4]
    for row in 0..2 {
        for column in 0..2 {
            var total = 0.0
            for shared in 0..3 {
                total += a_values[row * 3 + shared] * b_values[shared * 2 + column]
            }
            expected[row * 2 + column] = total
        }
    }
    matches(nn.matmul(a, b), expected, 1.0e-3)
    matches(nn.einsum("ij,jk->ik", a, b), expected, 1.0e-3)
}

fn test_the_array_core_is_usable_from_outside() -> void {
    if not have_gpu() { return }
    let source = ramp([2, 3])

    let target = NDArray([2, 3], FLOAT32)
    nn.copy_into(target, source)
    assert_eq(target.to_list(), source.to_list())

    let painted = NDArray([4], FLOAT32)
    nn.fill(painted, 2.0)
    assert_eq(painted.to_list(), [2.0, 2.0, 2.0, 2.0])
    nn.fill(painted, 9.0, holding([4], [1.0, 0.0, 1.0, 0.0], BOOL))
    assert_eq(painted.to_list(), [9.0, 2.0, 9.0, 2.0])

    let flat = holding([4], [10.0, 20.0, 30.0, 40.0])
    assert_eq(nn.gather_flat(flat, holding([2], [3.0, 0.0], INT32)).to_list(), [40.0, 10.0])

    let landing = NDArray([4], FLOAT32)
    nn.fill(landing, 0.0)
    nn.scatter_flat(landing, holding([2], [1.0, 2.0], INT32), holding([2], [5.0, 6.0]))
    assert_eq(landing.to_list(), [0.0, 5.0, 6.0, 0.0])

    let (offsets, total) = nn.exclusive_scan(holding([4], [1.0, 0.0, 1.0, 1.0], BOOL))
    assert_eq(total, 3)
    assert_eq(offsets.to_list(), [0.0, 1.0, 1.0, 2.0])

    matches(nn.binary_arith(flat, 2.0, nn.OP_DIV), [5.0, 10.0, 15.0, 20.0])
    matches(nn.binary_arith(flat, flat, nn.OP_SUB), [0.0, 0.0, 0.0, 0.0])

    ran(nn.diagonal_view(ramp([3, 3]), 1), [2])
    let (wide, tall) = nn.broadcast_pair(ramp([3]), ramp([2, 1]))
    assert_eq(wide.shape, [2, 3])
    assert_eq(tall.shape, [2, 3])

    let complex = NDArray([2], COMPLEX64)
    assert_throws(TypeError, || nn.require_real(complex, "this test"))
    assert_eq(nn.conj(complex).dtype, COMPLEX64)
    assert_eq(nn.conjugate(complex).dtype, COMPLEX64)

    #: `PRELUDE`, `kernel_of` and `layout_buffer` are exported so that a
    #: caller outside the package can write a pass of their own, so the test
    #: for them is to write one: a strided scale, reading the same metadata
    #: block every kernel in the package reads.
    let scaled = NDArray([2, 3], FLOAT32)
    let meta = nn.layout_buffer([scaled, source])
    nn.kernel_of(SCALE_SOURCE, "tr_scale").dispatch_async(
        [scaled.storage, source.storage, meta],
        scaled.size(),
        [scaled.size(), nn.f32_bits(3.0)]
    )
    matches(scaled, [value * 3.0 for value in source.to_list()])
    meta.free()
}

let SCALE_SOURCE = nn.PRELUDE + """

/* `out = src * factor`, walking both operands through the shared metadata
 * block so a non-contiguous source works the way it does inside the package.
 *
 * A test's kernel and not a package one: it exists to prove `PRELUDE` carries
 * everything a caller needs to write their own pass, which is the only thing
 * exporting it is for. */
kernel void tr_scale(device float* out [[buffer(0)]],
                     device const float* src [[buffer(1)]],
                     device const float* meta [[buffer(2)]],
                     constant uint& count [[buffer(3)]],
                     constant uint& factor_bits [[buffer(4)]],
                     uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    out[nd_offset(meta, 0u, id)] = src[nd_offset(meta, 1u, id)] * as_type<float>(factor_bits);
}
"""

fn test_the_three_namespaces_are_reachable() -> void {
    if not have_gpu() { return }

    #: A solution checked by substituting it back into the equation it solved,
    #: multiplied out on the host so nothing in jainum verifies itself.
    let coefficients = [3.0, 1.0, 1.0, 2.0]
    let constants = [9.0, 8.0]
    let x = nn.linalg.solve(holding([2, 2], coefficients), holding([2], constants)).to_list()
    for row in 0..2 {
        var total = 0.0
        for column in 0..2 { total += coefficients[row * 2 + column] * x[column] }
        assert_near(total, constants[row], 1.0e-4)
    }
    assert_near(nn.linalg.norm(holding([2], [3.0, 4.0])).item(), 5.0, 1.0e-5)
    assert_true(nn.linalg.FLOAT32_EPS > 0.0)

    #: Two generators given the same seed have to draw the same numbers —
    #: which is the property a counter-based generator exists for, and the one
    #: an accidental global would break.
    let first = nn.random.default_rng(20260824).random([8]).to_list()
    let second = nn.random.default_rng(20260824).random([8]).to_list()
    assert_eq(first, second)
    for value in first { assert_true(value >= 0.0 and value < 1.0) }
    assert_ne(nn.random.default_rng(7).random([8]).to_list(), first)

    #: A real transform of eight samples against the direct sum, which is the
    #: definition the stage chain is an optimisation of.
    let signal = [1.0, 3.0, -2.0, 0.5, 4.0, -1.5, 2.0, 0.0]
    let spectrum = nn.fft.rfft(holding([8], signal)).to_list()
    for bin in 0..5 {
        var real = 0.0
        var imaginary = 0.0
        for index in 0..8 {
            let angle = -2.0 * math.PI * float(bin) * float(index) / 8.0
            real += signal[index] * math.cos(angle)
            imaginary += signal[index] * math.sin(angle)
        }
        assert_near(spectrum[bin * 2], real, 1.0e-4)
        assert_near(spectrum[bin * 2 + 1], imaginary, 1.0e-4)
    }
}
