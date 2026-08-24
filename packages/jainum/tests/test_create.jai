#: Tests for jainum's array creation: the ramp generators, the index-predicate
#: ones, the host-list and byte paths, and the coordinate grids.
#:
#: Every generator is checked against the arithmetic that defines it — the
#: ceiling rule for `arange`, `start + i * step` for `linspace`, a constant
#: ratio for `geomspace`, `c - r` against `k` for `eye` and `tri` — and not
#: against numbers this module produced earlier. Anything that touches storage
#: returns early when there is no GPU, the way the rest of the suite does.
import std.gpu as gpu
import std.math
from std.test import assert_eq, assert_near, assert_throws, assert_true
from jainum.array import BOOL, COMPLEX64, ELLIPSIS, FLOAT32, INT32, NDArray
from jainum.create import arange, array, asarray, diag, diagflat, empty_like, eye
from jainum.create import from_bytes, from_list, frombuffer, fromfunction, full
from jainum.create import full_like, geomspace, identity, indices, linspace
from jainum.create import linspace_step, logspace, meshgrid, ones, ones_like, tri
from jainum.create import tril, triu, zeros, zeros_like

fn have_gpu() -> bool { return gpu.is_available() }

fn test_the_constructors_fill_on_the_device() -> void {
    if not have_gpu() { return }
    let z = zeros([2, 3])
    defer { z.free() }
    assert_eq(z.shape, [2, 3])
    assert_eq(z.to_list(), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    let o = ones([4], INT32)
    defer { o.free() }
    assert_eq(o.to_list(), [1.0, 1.0, 1.0, 1.0])
    assert_eq(o.dtype, INT32)

    let f = full([2], 2.5)
    defer { f.free() }
    assert_eq(f.to_list(), [2.5, 2.5])
}

fn test_the_like_variants_inherit_shape_and_dtype() -> void {
    if not have_gpu() { return }
    let a = NDArray([2, 3], INT32)
    defer { a.free() }

    let z = zeros_like(a)
    defer { z.free() }
    assert_eq(z.shape, [2, 3])
    assert_eq(z.dtype, INT32)

    let o = ones_like(a, FLOAT32)
    defer { o.free() }
    assert_eq(o.dtype, FLOAT32, "an explicit dtype overrides the source's")

    let f = full_like(a, 3.0)
    defer { f.free() }
    assert_eq(f.to_list(), [3.0, 3.0, 3.0, 3.0, 3.0, 3.0])

    let e = empty_like(a)
    defer { e.free() }
    assert_eq(e.shape, [2, 3])
}

#: `fill` writes the bits it is handed, so `full` has to round on the host.
#: The only defensible answer is the one the cast kernel would have given.
fn test_a_full_of_an_integer_dtype_rounds_the_way_a_cast_does() -> void {
    if not have_gpu() { return }
    for value in [2.5, 3.5, -2.5, -0.5, 1.4, 1.6, 0.0] {
        let direct = full([1], value, INT32)
        let loose = full([1], value)
        let cast = loose.astype(INT32)
        assert_eq(direct.item(), cast.item(), f"full and astype disagree on {value}")
        direct.free()
        loose.free()
        cast.free()
    }
}

fn test_arange_reads_its_arguments_the_way_numpy_does() -> void {
    if not have_gpu() { return }
    let bare = arange(5)
    defer { bare.free() }
    assert_eq(bare.to_list(), [0.0, 1.0, 2.0, 3.0, 4.0], "one argument is the stop")
    assert_eq(bare.dtype, INT32, "integer arguments give an integer array")

    let stepped = arange(2, 10, 3)
    defer { stepped.free() }
    assert_eq(stepped.to_list(), [2.0, 5.0, 8.0])

    let down = arange(5, 0, -2)
    defer { down.free() }
    assert_eq(down.to_list(), [5.0, 3.0, 1.0])

    let backwards = arange(5, 0, 1)
    defer { backwards.free() }
    assert_eq(backwards.size(), 0, "a step running the wrong way gives nothing")

    let promoted = arange(0, 3, 0.5)
    defer { promoted.free() }
    assert_eq(promoted.dtype, FLOAT32, "one float argument promotes the lot")

    assert_throws(ValueError, || arange(0, 5, 0))
    assert_throws(ValueError, || arange(0.0, 1.0, 1.0e-9), "a span past MAX_EXACT_INT")
}

#: The rule the module documents: `ceil((stop - start) / step)`, and not
#: "keep adding until you pass stop". The two disagree on the second case
#: below, which is exactly where numpy surprises people.
fn test_arange_with_a_float_step_takes_the_ceiling_of_the_quotient() -> void {
    if not have_gpu() { return }
    let tight = arange(0.0, 1.0, 0.1)
    defer { tight.free() }
    assert_eq(tight.size(), int(math.ceil((1.0 - 0.0) / 0.1)))
    assert_eq(tight.size(), 10)

    #: And the case where the two rules part company. `(1.3 - 1.0) / 0.1` is
    #: 3.0000000000000004 in the host's doubles, so the ceiling is four and
    #: the fourth sample is the stop itself — inside a half-open interval that
    #: was supposed to exclude it. That is numpy's answer too, and it is the
    #: whole reason the rule is written down rather than inferred.
    let over = arange(1.0, 1.3, 0.1)
    defer { over.free() }
    assert_eq(over.size(), int(math.ceil((1.3 - 1.0) / 0.1)))
    assert_eq(over.size(), 4)
    assert_near(over.to_list()[3], 1.3, 1e-6, "the last sample is the excluded stop")

    let seen = tight.to_list()
    for index in 0..seen.len() { assert_near(seen[index], float(index) * 0.1, 1e-6) }
}

#: The two ramp kernels: a count that divides by four takes the float4 path
#: and one that does not takes the scalar path. Neither the values nor the
#: length may depend on which ran.
fn test_a_long_ramp_is_generated_rather_than_uploaded() -> void {
    if not have_gpu() { return }
    let quad = arange(0.0, 100000.0, 1.0)
    let odd = arange(0.0, 100003.0, 1.0)
    defer { quad.free() }
    defer { odd.free() }
    assert_eq(quad.size() % 4, 0)
    assert_eq(odd.size() % 4, 3)

    let a = quad.to_list()
    let b = odd.to_list()
    for probe in [0, 1, 3, 4, 5, 99999] {
        assert_eq(a[probe], float(probe))
        assert_eq(b[probe], float(probe))
    }
    assert_eq(b[100002], 100002.0, "the tail of the scalar path")
}

fn test_linspace_puts_the_endpoint_in_exactly() -> void {
    if not have_gpu() { return }
    let closed = linspace(0.0, 1.0, 5)
    defer { closed.free() }
    let step = linspace_step(0.0, 1.0, 5)
    assert_near(step, 0.25)
    let seen = closed.to_list()
    assert_eq(seen[0], 0.0)
    assert_eq(seen[4], 1.0, "the endpoint is written in, not computed")
    for index in 0..5 { assert_near(seen[index], float(index) * step, 1e-6) }

    let open = linspace(0.0, 1.0, 5, false)
    defer { open.free() }
    assert_near(linspace_step(0.0, 1.0, 5, false), 0.2)
    assert_near(open.to_list()[4], 0.8, 1e-6, "without the endpoint the stop is absent")

    let single = linspace(3.0, 9.0, 1)
    defer { single.free() }
    assert_eq(single.to_list(), [3.0], "one sample is the start under either flag")

    let none = linspace(0.0, 1.0, 0)
    defer { none.free() }
    assert_eq(none.size(), 0)
    assert_throws(ValueError, || linspace(0.0, 1.0, -1))
}

#: A ramp that accumulated its samples would drift here; one computed from the
#: index does not, and the endpoint is pinned on top of that.
fn test_a_long_linspace_still_lands_on_its_endpoint() -> void {
    if not have_gpu() { return }
    let num = 100000
    let a = linspace(-1.0, 1.0, num)
    defer { a.free() }
    assert_eq(a[0].item(), -1.0)
    assert_eq(a[num - 1].item(), 1.0, "the last sample is the endpoint itself")
    let step = linspace_step(-1.0, 1.0, num)
    assert_near(a[num // 2].item(), -1.0 + float(num // 2) * step, 1e-5)
}

fn test_logspace_is_the_base_raised_to_a_linspace() -> void {
    if not have_gpu() { return }
    let num = 7
    let a = logspace(0.0, 3.0, num)
    defer { a.free() }
    let seen = a.to_list()
    let step = linspace_step(0.0, 3.0, num)
    for index in 0..num {
        let expected = 10.0 ** (float(index) * step)
        assert_near(seen[index], expected, expected * 1e-5)
    }
    assert_eq(seen[num - 1], 1000.0, "the endpoint is base ** stop, computed on the host")

    let two = logspace(0.0, 4.0, 5, true, 2.0)
    defer { two.free() }
    let powers = two.to_list()
    for index in 0..5 { assert_near(powers[index], 2.0 ** float(index), 1e-4) }
}

fn test_geomspace_keeps_a_constant_ratio() -> void {
    if not have_gpu() { return }
    let a = geomspace(1.0, 1000.0, 4)
    defer { a.free() }
    let seen = a.to_list()
    assert_near(seen[0], 1.0, 1e-5)
    assert_eq(seen[3], 1000.0, "the endpoint is written in")

    #: The definition: every step multiplies by the same factor, which is the
    #: cube root of stop over start for four samples.
    let ratio = 1000.0 ** (1.0 / 3.0)
    for index in 1..4 { assert_near(seen[index] / seen[index - 1], ratio, ratio * 1e-4) }

    let negative = geomspace(-1.0, -8.0, 4)
    defer { negative.free() }
    let below = negative.to_list()
    for index in 0..4 { assert_true(below[index] < 0.0, "the sign rides outside the exponent") }
    assert_near(below[1] / below[0], 2.0, 1e-4)

    assert_throws(ValueError, || geomspace(0.0, 1.0, 4))
    assert_throws(ValueError, || geomspace(-1.0, 1.0, 4))
}

#: Both column counts matter: one divides by four and takes the float4 kernel,
#: the other does not and takes the scalar one.
fn test_eye_and_tri_agree_with_the_predicate_that_defines_them() -> void {
    if not have_gpu() { return }
    for cols in [8, 5] {
        for k in [-2, 0, 1, 3] {
            let ones_on = eye(4, cols, k)
            let filled = tri(4, cols, k)
            let seen_eye = ones_on.to_list()
            let seen_tri = filled.to_list()
            for r in 0..4 {
                for c in 0..cols {
                    let at = r * cols + c
                    let on = if c - r == k { 1.0 } else { 0.0 }
                    let under = if c - r <= k { 1.0 } else { 0.0 }
                    assert_eq(seen_eye[at], on, f"eye({r}, {c}) with k={k}")
                    assert_eq(seen_tri[at], under, f"tri({r}, {c}) with k={k}")
                }
            }
            ones_on.free()
            filled.free()
        }
    }
}

fn test_a_band_past_the_corner_is_the_corner_answer() -> void {
    if not have_gpu() { return }
    let far = eye(3, 4, 99)
    defer { far.free() }
    assert_eq(far.to_list(), [0.0 for _index in 0..12])

    let nothing = tri(3, 4, -99)
    defer { nothing.free() }
    assert_eq(nothing.to_list(), [0.0 for _index in 0..12])

    let everything = tri(3, 4, 99)
    defer { everything.free() }
    assert_eq(everything.to_list(), [1.0 for _index in 0..12])

    let unit = identity(3, INT32)
    defer { unit.free() }
    assert_eq(unit.to_list(), [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    assert_eq(unit.dtype, INT32)

    assert_throws(ValueError, || eye(-1, 2))
}

#: Two checks on one pass: the values against the band predicate, and the pair
#: against each other — every element belongs to exactly one of `tril(k)` and
#: `triu(k + 1)`, so the two add back up to the array they came from.
fn test_tril_and_triu_split_a_stack_between_them() -> void {
    if not have_gpu() { return }
    let a = NDArray([2, 3, 4])
    a.upload([float(index) + 1.0 for index in 0..24])
    defer { a.free() }
    let source = a.to_list()

    for k in [-1, 0, 2] {
        let low = tril(a, k)
        let high = triu(a, k + 1)
        let seen_low = low.to_list()
        let seen_high = high.to_list()
        for index in 0..24 {
            let within = index % 12
            let r = within // 4
            let c = within % 4
            let kept = if c - r <= k { source[index] } else { 0.0 }
            assert_eq(seen_low[index], kept, f"tril at {index} with k={k}")
            assert_eq(seen_low[index] + seen_high[index], source[index])
        }
        low.free()
        high.free()
    }
    assert_throws(ValueError, || tril(a[0, 0]), "a 0-d array has no two axes")
}

fn test_diag_lays_a_vector_on_a_diagonal_and_reads_it_back() -> void {
    if not have_gpu() { return }
    let v = NDArray([3])
    v.upload([5.0, 6.0, 7.0])
    defer { v.free() }
    let values = [5.0, 6.0, 7.0]

    for k in [-2, 0, 1] {
        let square = diag(v, k)
        let side = 3 + (if k < 0 { -k } else { k })
        assert_eq(square.shape, [side, side])
        let seen = square.to_list()
        for r in 0..side {
            for c in 0..side {
                let expected = if c - r == k { values[if k >= 0 { r } else { c }] } else { 0.0 }
                assert_eq(seen[r * side + c], expected, f"diag({r}, {c}) with k={k}")
            }
        }
        #: The round trip: the diagonal read back out is the vector that went
        #: in, which pins the two directions against each other.
        let back = diag(square, k)
        assert_eq(back.to_list(), values)
        square.free()
        back.free()
    }
    assert_throws(ValueError, || diag(indices([2, 2]), 0))
}

fn test_diagflat_flattens_before_it_lays_out() -> void {
    if not have_gpu() { return }
    let a = NDArray([2, 2])
    a.upload([1.0, 2.0, 3.0, 4.0])
    defer { a.free() }

    let out = diagflat(a)
    defer { out.free() }
    assert_eq(out.shape, [4, 4])
    for r in 0..4 {
        for c in 0..4 {
            let expected = if r == c { float(r) + 1.0 } else { 0.0 }
            assert_eq(out.to_nested()[r][c], expected)
        }
    }

    #: A strided input has to be materialised first, and the values must come
    #: out in element order rather than in storage order.
    let column = a[ELLIPSIS, 0..1]
    let thin = diagflat(column)
    defer { thin.free() }
    assert_eq(thin.to_list(), [1.0, 0.0, 0.0, 3.0])
}

fn test_array_reads_shape_and_dtype_off_the_nesting() -> void {
    if not have_gpu() { return }
    let whole = array([[1, 2, 3], [4, 5, 6]])
    defer { whole.free() }
    assert_eq(whole.shape, [2, 3])
    assert_eq(whole.dtype, INT32, "every leaf was an int")
    assert_eq(whole.to_nested(), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    let mixed = array([1, 2.5])
    defer { mixed.free() }
    assert_eq(mixed.dtype, FLOAT32, "one float promotes the lot")
    assert_eq(mixed.to_list(), [1.0, 2.5])

    let flags = array([true, false])
    defer { flags.free() }
    assert_eq(flags.dtype, BOOL)
    assert_eq(flags.to_list(), [1.0, 0.0])

    let scalar = array(7)
    defer { scalar.free() }
    assert_eq(scalar.shape, [])
    assert_eq(scalar.item(), 7.0)

    let none = array([])
    defer { none.free() }
    assert_eq(none.shape, [0])
    assert_eq(none.dtype, FLOAT32, "an empty array is float, as numpy's is")

    assert_throws(ValueError, || array([[1, 2], [3]]), "a ragged nesting is refused")
    assert_throws(ValueError, || array([1, [2]]))
}

fn test_asarray_passes_an_array_through_and_array_copies_it() -> void {
    if not have_gpu() { return }
    let a = array([1, 2, 3])
    defer { a.free() }
    assert_true(asarray(a) == a, "asarray does not copy what is already an array")

    let copied = array(a)
    defer { copied.free() }
    assert_true(not copied == a)
    copied[0] = 9.0
    assert_eq(a[0].item(), 1.0, "array() hands back an allocation of its own")

    let widened = asarray(a, FLOAT32)
    defer { widened.free() }
    assert_eq(widened.dtype, FLOAT32)
}

fn test_from_list_and_from_bytes_take_the_two_host_layouts() -> void {
    if not have_gpu() { return }
    let flat = from_list([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
    defer { flat.free() }
    assert_eq(flat.to_nested(), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert_throws(ValueError, || from_list([1.0], [2, 3]), "the count must match exactly")

    let raw = bytes([0, 128, 255, 64])
    let levels = [0, 128, 255, 64]
    let image = from_bytes(raw, [2, 2], FLOAT32, 0, 1.0 / 255.0)
    defer { image.free() }
    let seen = image.to_list()
    for index in 0..4 { assert_near(seen[index], float(levels[index]) / 255.0, 1e-6) }

    let tail = frombuffer(raw, null, 1)
    defer { tail.free() }
    assert_eq(tail.to_list(), [128.0, 255.0, 64.0])

    #: `count` is elements and `offset` is bytes, which for a two-lane dtype
    #: are not the same number: four bytes make two complex values.
    let pairs = frombuffer(raw, null, 0, COMPLEX64)
    defer { pairs.free() }
    assert_eq(pairs.size(), 2)
    assert_eq(pairs.to_list(), [0.0, 128.0, 255.0, 64.0])

    assert_throws(ValueError, || from_bytes(raw, [3, 3]))
}

fn test_meshgrid_swaps_the_first_two_axes_for_xy() -> void {
    if not have_gpu() { return }
    let x = array([1.0, 2.0, 3.0])
    let y = array([10.0, 20.0])
    defer { x.free() }
    defer { y.free() }

    let cartesian = meshgrid([x, y])
    assert_eq(cartesian[0].shape, [2, 3], "xy is shaped [len(y), len(x)]")
    assert_eq(cartesian[0].to_nested(), [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    assert_eq(cartesian[1].to_nested(), [[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]])
    cartesian[0].free()
    cartesian[1].free()

    let matrix = meshgrid([x, y], "ij")
    assert_eq(matrix[0].shape, [3, 2], "ij has input i varying along axis i")
    assert_eq(matrix[0].to_nested(), [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    assert_eq(matrix[1].to_nested(), [[10.0, 20.0], [10.0, 20.0], [10.0, 20.0]])
    matrix[0].free()
    matrix[1].free()

    let thin = meshgrid([x, y], "xy", true)
    assert_eq(thin[0].shape, [1, 3], "sparse leaves every other axis at one")
    assert_eq(thin[1].shape, [2, 1])
    assert_eq(thin[0].to_list(), [1.0, 2.0, 3.0])
    thin[0].free()
    thin[1].free()

    assert_throws(ValueError, || meshgrid([x], "diagonal"))
}

fn test_indices_writes_one_coordinate_sheet_per_axis() -> void {
    if not have_gpu() { return }
    let grid = indices([2, 3, 4])
    defer { grid.free() }
    assert_eq(grid.shape, [3, 2, 3, 4])
    let seen = grid.to_list()
    let plane = 24
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                let at = (i * 3 + j) * 4 + k
                assert_eq(seen[at], float(i))
                assert_eq(seen[plane + at], float(j))
                assert_eq(seen[2 * plane + at], float(k))
            }
        }
    }
}

fn test_fromfunction_evaluates_its_expression_on_the_device() -> void {
    if not have_gpu() { return }
    let table = fromfunction("i0 * 10.0 + i1", [3, 4])
    defer { table.free() }
    let seen = table.to_list()
    for r in 0..3 {
        for c in 0..4 { assert_eq(seen[r * 4 + c], float(r) * 10.0 + float(c)) }
    }

    #: The Metal standard library is in scope, which is the point of taking an
    #: expression rather than a host callback.
    let ring = fromfunction("sqrt(i0 * i0 + i1 * i1)", [16, 16])
    defer { ring.free() }
    let radius = ring.to_list()
    for r in 0..16 {
        for c in 0..16 {
            let expected = math.sqrt(float(r) * float(r) + float(c) * float(c))
            assert_near(radius[r * 16 + c], expected, 1e-4)
        }
    }

    #: A second shape must reuse the compiled kernel rather than bake its
    #: extents in, which is only visible as the answer still being right.
    let cube = fromfunction("i0", [2, 2, 2])
    defer { cube.free() }
    assert_eq(cube.to_list(), [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])

    let counted = fromfunction("i0", [5], INT32)
    defer { counted.free() }
    assert_eq(counted.to_list(), [0.0, 1.0, 2.0, 3.0, 4.0])
}

fn test_a_complex_generator_goes_through_the_widening_copy() -> void {
    if not have_gpu() { return }
    let ramp = arange(0, 3, 1, COMPLEX64)
    defer { ramp.free() }
    assert_eq(ramp.dtype, COMPLEX64)
    assert_eq(ramp.to_list(), [0.0, 0.0, 1.0, 0.0, 2.0, 0.0], "the imaginary lane is zero")

    let unit = eye(2, 2, 0, COMPLEX64)
    defer { unit.free() }
    assert_eq(unit.to_list(), [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])

    assert_throws(TypeError, || fromfunction("i0", [2], COMPLEX64))
}
