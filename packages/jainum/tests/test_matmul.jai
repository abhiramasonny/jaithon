#: Tests for jainum's products.
#:
#: Every answer is checked against the definition of the operation computed a
#: second way on the host — a triple loop for a matrix product, a four-deep one
#: for a Kronecker product, the determinant formula for a cross product —
#: rather than against numbers this module produced earlier. The shapes are
#: chosen to reach all four dispatch routes, and each says which one it takes:
#: a single output folds through the two-pass dot, a 16-byte-aligned shape goes
#: to Metal Performance Shaders, a result no larger than one tile folds along
#: the inner axis, and everything else is the tiled kernel.
#:
#: Anything that touches storage returns early without a GPU, as the rest of
#: the suite does.
import std.gpu as gpu
from std.test import assert_eq, assert_near, assert_throws, assert_true
from jainum.array import BOOL, FLOAT32, INT32, NDArray, element_count, sl
from jainum.manipulate import transpose
from jainum.matmul import cross, dot, einsum, inner, kron, matmul, matrix_power
from jainum.matmul import outer, tensordot, vdot

fn have_gpu() -> bool { return gpu.is_available() }

#: Values that are not a ramp: a ramp makes a product whose rows differ by a
#: constant, which hides an index mistake in the middle of a tile.
fn spread(shape: list[int], seed: int) -> NDArray {
    let total = element_count(shape)
    let a = NDArray(shape, FLOAT32)
    var values: list[float] = []
    var state = seed
    for _index in 0..total {
        state = (state * 1103515245 + 12345) % 32768
        values.push(float(state % 17) - 8.0)
    }
    a.upload(values)
    return a
}

fn host_product(
    a: list[float],
    b: list[float],
    rows: int,
    inner: int,
    columns: int
) -> list[float] {
    var out = [0.0 for _index in 0..rows * columns]
    for row in 0..rows {
        for column in 0..columns {
            var total = 0.0
            for at in 0..inner { total += a[row * inner + at] * b[at * columns + column] }
            out[row * columns + column] = total
        }
    }
    return out
}

#: Element-scaled, because a float32 sum over a few thousand terms is not a
#: float64 one and an absolute tolerance would either fail or prove nothing.
fn assert_close(actual: list[float], expected: list[float], what: str) -> void {
    assert_eq(actual.len(), expected.len(), f"{what}: length")
    for at in 0..actual.len() {
        let magnitude = if expected[at] < 0.0 { -expected[at] } else { expected[at] }
        let tolerance = magnitude * 1.0e-5 + 1.0e-4
        assert_near(actual[at], expected[at], tolerance, f"{what}: slot {at}")
    }
}

#: One product against the definition, with either operand optionally stored
#: the other way round — which is the transpose flag the GEMM reads instead of
#: copying, and a different index walk in every kernel here.
fn check_layout(
    rows: int,
    inner: int,
    columns: int,
    flip_a: bool,
    flip_b: bool,
    what: str
) -> void {
    let a_source = if flip_a {
        spread([inner, rows], rows * 31 + inner)
    } else {
        spread([rows, inner], rows * 31 + inner)
    }
    let b_source = if flip_b {
        spread([columns, inner], columns * 17 + inner)
    } else {
        spread([inner, columns], columns * 17 + inner)
    }
    let a = if flip_a { transpose(a_source) } else { a_source }
    let b = if flip_b { transpose(b_source) } else { b_source }
    let product = matmul(a, b)
    assert_eq(product.shape, [rows, columns], what)
    assert_close(
        product.to_list(),
        host_product(a.to_list(), b.to_list(), rows, inner, columns),
        what
    )
    a_source.free()
    b_source.free()
    product.free()
}

fn check_product(rows: int, inner: int, columns: int, what: str) -> void {
    check_layout(rows, inner, columns, false, false, what)
}

fn test_every_dispatch_route_computes_the_same_product() -> void {
    if not have_gpu() { return }

    #: One output: the two-pass dot, which is the only route with a grid that
    #: does not follow the shape of the result.
    check_product(1, 5000, 1, "a long vector dot")
    check_product(1, 7, 1, "a short vector dot")

    #: Every row a multiple of four floats, so MPS takes it.
    check_product(64, 64, 64, "aligned, MPS")
    check_product(128, 32, 256, "aligned and rectangular, MPS")

    #: A ragged inner extent keeps MPS out, and a result inside one tile folds
    #: along the inner axis instead.
    check_product(3, 300, 5, "small result, long inner axis")
    check_product(9, 1000, 1, "a matrix against a column")
    check_product(1, 999, 6, "a row against a matrix")

    #: Bigger than one tile with nothing aligned: the tiled kernel, including
    #: its bounds-checked edges.
    check_product(70, 33, 66, "ragged tiles")
    check_product(65, 8, 67, "one element past a tile in each direction")
    check_product(200, 129, 130, "several tiles")

    #: Result rows a multiple of four but an operand's are not: the tiled
    #: kernel runs, and stores its micro-tiles four at a time.
    check_product(100, 33, 64, "tiles with an aligned store")
}

fn test_the_transpose_flags_agree_with_the_row_major_answer() -> void {
    if not have_gpu() { return }
    for flip_a in [false, true] {
        for flip_b in [false, true] {
            let tag = f"a={flip_a} b={flip_b}"

            #: One shape per route, so a wrong index walk cannot hide in the
            #: kernel that was not asked for.
            check_layout(1, 512, 1, flip_a, flip_b, f"dot, {tag}")
            check_layout(5, 400, 3, flip_a, flip_b, f"fold, {tag}")
            check_layout(64, 64, 64, flip_a, flip_b, f"aligned, {tag}")
            check_layout(100, 33, 70, flip_a, flip_b, f"tiled, {tag}")
        }
    }
}

fn test_a_flipped_operand_is_the_product_of_the_flipped_values() -> void {
    if not have_gpu() { return }
    let source = spread([4, 6], 40)
    let b = spread([6, 3], 41)
    defer { source.free() }
    defer { b.free() }

    #: A negative stride is a layout no GEMM can address, so the operand is
    #: copied dense first — and the answer must be the one the reversed rows
    #: give, not the one the underlying buffer would.
    let reversed = source[sl(null, null, -1)]
    let product = matmul(reversed, b)
    defer { product.free() }
    assert_true(reversed.strides[0] < 0, "the test only means something on a flip")
    assert_close(
        product.to_list(),
        host_product(reversed.to_list(), b.to_list(), 4, 6, 3),
        "a flipped left operand"
    )
}

fn test_a_transposed_operand_needs_no_copy_and_no_correction() -> void {
    if not have_gpu() { return }
    let source = spread([4, 5], 21)
    let b = spread([4, 6], 8)
    defer { source.free() }
    defer { b.free() }

    #: `source.T @ b`: the left operand is column-major, which the GEMM reads
    #: through its transpose flag rather than through a copy.
    let column_major = transpose(source)
    assert_eq(column_major.strides, [1, 5], "a transpose is column-major, which a GEMM addresses")
    let product = matmul(column_major, b)
    defer { product.free() }
    let sv = source.to_list()
    let bv = b.to_list()
    var expected = [0.0 for _index in 0..30]
    for row in 0..5 {
        for column in 0..6 {
            var total = 0.0
            for at in 0..4 { total += sv[at * 5 + row] * bv[at * 6 + column] }
            expected[row * 6 + column] = total
        }
    }
    assert_eq(product.shape, [5, 6])
    assert_close(product.to_list(), expected, "a transposed left operand")
}

fn test_an_operand_no_gemm_can_address_is_materialised() -> void {
    if not have_gpu() { return }
    let source = spread([6, 8], 5)
    defer { source.free() }

    #: A column block: the rows are `8` apart but only four wide, which is
    #: neither row-major nor column-major, so this goes through a copy.
    let block = source[0..6, 2..6]
    let b = spread([4, 5], 12)
    defer { b.free() }
    let product = matmul(block, b)
    defer { product.free() }
    assert_close(
        product.to_list(),
        host_product(block.to_list(), b.to_list(), 6, 4, 5),
        "a strided left operand"
    )

    #: A row block, by contrast, is still row-major — it only starts later in
    #: the buffer — so it is read where it lies, offset and all.
    let rows = spread([16, 8], 6)
    defer { rows.free() }
    let window = rows[2..10]
    let square = spread([8, 8], 7)
    defer { square.free() }
    let windowed = matmul(window, square)
    defer { windowed.free() }
    assert_eq(window.offset, 16, "a row block is an offset, not a copy")
    assert_close(
        windowed.to_list(),
        host_product(window.to_list(), square.to_list(), 8, 8, 8),
        "a row block as the left operand"
    )

    #: A broadcast inside the plane: stride zero down the rows. The copy has to
    #: happen before the product, not be papered over by the batch walk.
    let row = spread([1, 4], 30)
    defer { row.free() }
    let stretched = row.broadcast_to([7, 4])
    let stacked = matmul(stretched, b)
    defer { stacked.free() }
    assert_close(
        stacked.to_list(),
        host_product(stretched.to_list(), b.to_list(), 7, 4, 5),
        "a broadcast left operand"
    )
}

fn test_the_leading_axes_stack_and_broadcast() -> void {
    if not have_gpu() { return }
    let a = spread([3, 4, 5], 7)
    let b = spread([3, 5, 2], 11)
    defer { a.free() }
    defer { b.free() }

    let product = matmul(a, b)
    defer { product.free() }
    assert_eq(product.shape, [3, 4, 2])
    let av = a.to_list()
    let bv = b.to_list()
    let cv = product.to_list()
    for plane in 0..3 {
        var left: list[float] = []
        var right: list[float] = []
        for at in 0..20 { left.push(av[plane * 20 + at]) }
        for at in 0..10 { right.push(bv[plane * 10 + at]) }
        var got: list[float] = []
        for at in 0..8 { got.push(cv[plane * 8 + at]) }
        assert_close(got, host_product(left, right, 4, 5, 2), f"plane {plane}")
    }

    #: A stretched batch axis feeds one plane to every dispatch, so all three
    #: results must be the product of that one plane.
    let single = spread([1, 4, 5], 13)
    defer { single.free() }
    let shared = matmul(single, b)
    defer { shared.free() }
    assert_eq(shared.shape, [3, 4, 2])
    let sv = single.to_list()
    let hv = shared.to_list()
    for plane in 0..3 {
        var right: list[float] = []
        for at in 0..10 { right.push(bv[plane * 10 + at]) }
        var got: list[float] = []
        for at in 0..8 { got.push(hv[plane * 8 + at]) }
        assert_close(got, host_product(sv, right, 4, 5, 2), f"broadcast plane {plane}")
    }

    #: Both sides broadcast, as numpy's do.
    let wide = matmul(spread([2, 1, 3, 4], 2), spread([1, 5, 4, 3], 4))
    defer { wide.free() }
    assert_eq(wide.shape, [2, 5, 3, 3])
}

fn test_a_plane_past_the_first_reads_its_own_values() -> void {
    if not have_gpu() { return }

    #: Aligned, batched, and therefore the one path that hands MPS a *window*
    #: into a buffer rather than the whole of it. A window whose offset is
    #: ignored produces the first plane's answer for every plane, silently,
    #: which is a bug this repo has had before — so each plane is checked
    #: against its own operands.
    let a = spread([3, 8, 8], 44)
    let b = spread([3, 8, 8], 45)
    defer { a.free() }
    defer { b.free() }
    let product = matmul(a, b)
    defer { product.free() }
    assert_eq(product.shape, [3, 8, 8])

    let av = a.to_list()
    let bv = b.to_list()
    let cv = product.to_list()
    for plane in 0..3 {
        var left: list[float] = []
        var right: list[float] = []
        var got: list[float] = []
        for at in 0..64 {
            left.push(av[plane * 64 + at])
            right.push(bv[plane * 64 + at])
            got.push(cv[plane * 64 + at])
        }
        assert_close(got, host_product(left, right, 8, 8, 8), f"aligned plane {plane}")
    }

    #: The same again through the tiled kernel, whose plane offsets are
    #: scalars rather than buffer windows.
    let ragged = matmul(spread([3, 70, 33], 46), spread([3, 33, 66], 47))
    defer { ragged.free() }
    assert_eq(ragged.shape, [3, 70, 66])
    let heads = ragged.to_list()
    var same = true
    for at in 0..4620 {
        if heads[at] != heads[4620 + at] { same = false }
    }
    assert_true(not same, "two planes of different operands are not the same answer")
}

fn test_a_vector_operand_loses_the_axis_that_promoted_it() -> void {
    if not have_gpu() { return }
    let a = spread([3, 4], 6)
    let x = spread([4], 9)
    let y = spread([3], 10)
    defer { a.free() }
    defer { x.free() }
    defer { y.free() }

    let column = matmul(a, x)
    defer { column.free() }
    assert_eq(column.shape, [3], "a vector on the right is a column, then dropped")
    assert_close(column.to_list(), host_product(a.to_list(), x.to_list(), 3, 4, 1), "matrix-vector")

    let row = matmul(y, a)
    defer { row.free() }
    assert_eq(row.shape, [4], "a vector on the left is a row, then dropped")
    assert_close(row.to_list(), host_product(y.to_list(), a.to_list(), 1, 3, 4), "vector-matrix")

    let scalar = matmul(x, x)
    defer { scalar.free() }
    assert_eq(scalar.shape, [], "two vectors leave nothing behind")
    var total = 0.0
    for value in x.to_list() { total += value * value }
    assert_near(scalar.item(), total, 1.0e-3)

    #: A stacked left operand keeps its batch axes and still drops the
    #: promoted one.
    let batched = matmul(spread([6, 3, 4], 1), x)
    defer { batched.free() }
    assert_eq(batched.shape, [6, 3])
}

fn test_the_shapes_that_cannot_multiply_raise() -> void {
    if not have_gpu() { return }
    let a = spread([3, 4], 1)
    let b = spread([5, 2], 2)
    let scalar = NDArray([], FLOAT32)
    defer { a.free() }
    defer { b.free() }
    defer { scalar.free() }
    assert_throws(ValueError, || matmul(a, b), "4 against 5")
    assert_throws(ValueError, || matmul(a, scalar), "a 0-d operand has no reading")
    assert_throws(ValueError, || matmul(spread([2, 3, 4], 1), spread([3, 4, 2], 1)))
}

fn test_an_empty_extent_is_a_shape_and_not_an_error() -> void {
    if not have_gpu() { return }

    #: A contracted axis of nothing sums nothing, which is zero — the one case
    #: where the result has elements but no dispatch runs over them.
    let empty_inner = matmul(NDArray([2, 0], FLOAT32), NDArray([0, 3], FLOAT32))
    defer { empty_inner.free() }
    assert_eq(empty_inner.shape, [2, 3])
    assert_eq(empty_inner.to_list(), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    let empty_rows = matmul(NDArray([0, 3], FLOAT32), spread([3, 4], 1))
    defer { empty_rows.free() }
    assert_eq(empty_rows.shape, [0, 4])
    assert_eq(empty_rows.size(), 0)
}

fn test_a_product_of_whole_numbers_stays_whole() -> void {
    if not have_gpu() { return }
    let ints = NDArray([2, 2], INT32)
    ints.upload([1.0, 2.0, 3.0, 4.0])
    defer { ints.free() }
    let squared = matmul(ints, ints)
    defer { squared.free() }
    assert_eq(squared.dtype, INT32)
    assert_eq(squared.to_list(), [7.0, 10.0, 15.0, 22.0])

    #: Two booleans multiply and then *sum*, which is counting.
    let flags = NDArray([2, 2], BOOL)
    flags.upload([1.0, 0.0, 1.0, 1.0])
    defer { flags.free() }
    let counted = matmul(flags, flags)
    defer { counted.free() }
    assert_eq(counted.dtype, INT32, "a count is not a flag")
    assert_eq(counted.to_list(), [1.0, 0.0, 2.0, 1.0])

    let mixed = matmul(ints, spread([2, 2], 3))
    defer { mixed.free() }
    assert_eq(mixed.dtype, FLOAT32)
}

fn test_dot_follows_each_of_numpys_five_rules() -> void {
    if not have_gpu() { return }
    let x = spread([5], 2)
    let y = spread([5], 3)
    let a = spread([3, 5], 4)
    let b = spread([5, 2], 5)
    let scalar = NDArray([], FLOAT32)
    scalar.upload([3.0])
    defer { x.free() }
    defer { y.free() }
    defer { a.free() }
    defer { b.free() }
    defer { scalar.free() }

    let scaled = dot(a, scalar)
    defer { scaled.free() }
    assert_eq(scaled.shape, [3, 5], "a 0-d operand scales")
    assert_near(scaled.to_list()[0], a.to_list()[0] * 3.0, 1.0e-4)

    let paired = dot(x, y)
    defer { paired.free() }
    assert_eq(paired.shape, [])
    var total = 0.0
    let xv = x.to_list()
    let yv = y.to_list()
    for at in 0..5 { total += xv[at] * yv[at] }
    assert_near(paired.item(), total, 1.0e-3)

    let matrices = dot(a, b)
    defer { matrices.free() }
    assert_close(
        matrices.to_list(),
        host_product(a.to_list(), b.to_list(), 3, 5, 2),
        "two matrices"
    )

    let against_vector = dot(a, x)
    defer { against_vector.free() }
    assert_eq(against_vector.shape, [3])
    assert_close(
        against_vector.to_list(),
        host_product(a.to_list(), x.to_list(), 3, 5, 1),
        "the last axis against a vector"
    )

    #: The rule that separates `dot` from `matmul`: stacked operands take the
    #: outer product of their batches instead of broadcasting together.
    let stacked = dot(spread([2, 3, 5], 6), spread([4, 5, 2], 7))
    defer { stacked.free() }
    assert_eq(stacked.shape, [2, 3, 4, 2])
}

fn test_vdot_inner_and_outer() -> void {
    if not have_gpu() { return }
    let a = spread([2, 3], 8)
    let b = spread([3, 2], 9)
    defer { a.free() }
    defer { b.free() }

    let flat = vdot(a, b)
    defer { flat.free() }
    var total = 0.0
    let av = a.to_list()
    let bv = b.to_list()
    for at in 0..6 { total += av[at] * bv[at] }
    assert_near(flat.item(), total, 1.0e-3, "vdot flattens both sides first")
    assert_throws(ValueError, || vdot(a, spread([5], 1)))

    #: `inner` contracts the last axis of both, so it is `dot` against a
    #: transpose.
    let paired = inner(a, transpose(b))
    defer { paired.free() }
    assert_eq(paired.shape, [2, 2])
    assert_close(paired.to_list(), host_product(av, bv, 2, 3, 2), "inner against a transpose")

    let table = outer(a, b)
    defer { table.free() }
    assert_eq(table.shape, [6, 6], "outer flattens both operands")
    for row in 0..6 {
        for column in 0..6 {
            assert_near(
                table.to_list()[row * 6 + column],
                av[row] * bv[column],
                1.0e-3,
                f"outer {row},{column}"
            )
        }
    }
}

fn test_tensordot_contracts_the_axes_it_is_given() -> void {
    if not have_gpu() { return }
    let a = spread([2, 3, 4], 15)
    let b = spread([3, 4, 5], 16)
    defer { a.free() }
    defer { b.free() }

    let contracted = tensordot(a, b, [[1, 2], [0, 1]])
    defer { contracted.free() }
    assert_eq(contracted.shape, [2, 5])
    let av = a.to_list()
    let bv = b.to_list()
    for row in 0..2 {
        for column in 0..5 {
            var total = 0.0
            for p in 0..3 {
                for q in 0..4 {
                    total += av[row * 12 + p * 4 + q] * bv[p * 20 + q * 5 + column]
                }
            }
            assert_near(
                contracted.to_list()[row * 5 + column],
                total,
                1.0e-3,
                f"tensordot {row},{column}"
            )
        }
    }

    #: A count contracts the last that many axes of the left against the first
    #: that many of the right, which is what makes the default of two the
    #: double contraction numpy documents.
    let counted = tensordot(a, b, 2)
    defer { counted.free() }
    assert_close(counted.to_list(), contracted.to_list(), "a count of two")

    #: Zero axes contract nothing, which is the outer product.
    let none = tensordot(spread([2], 1), spread([3], 2), 0)
    defer { none.free() }
    assert_eq(none.shape, [2, 3])

    #: A single axis on each side, named negatively.
    let single = tensordot(a, b, [[-1], [1]])
    defer { single.free() }
    assert_eq(single.shape, [2, 3, 3, 5])
    assert_throws(ValueError, || tensordot(a, b, [[0], [0]]), "2 against 3")
    assert_throws(ValueError, || tensordot(a, b, [[0], [0, 1]]), "one axis against two")
}

fn test_kron_lays_a_scaled_copy_in_every_block() -> void {
    if not have_gpu() { return }
    let a = spread([2, 3], 18)
    let b = spread([3, 2], 19)
    defer { a.free() }
    defer { b.free() }

    let product = kron(a, b)
    defer { product.free() }
    assert_eq(product.shape, [6, 6])
    let av = a.to_list()
    let bv = b.to_list()
    let pv = product.to_list()
    for i in 0..2 {
        for j in 0..3 {
            for p in 0..3 {
                for q in 0..2 {
                    let row = i * 3 + p
                    let column = j * 2 + q
                    assert_near(
                        pv[row * 6 + column],
                        av[i * 3 + j] * bv[p * 2 + q],
                        1.0e-3,
                        f"kron {row},{column}"
                    )
                }
            }
        }
    }

    #: Unequal ranks: the shorter shape gains leading ones, so a vector against
    #: a matrix is a row of blocks.
    let mixed = kron(spread([2], 1), spread([2, 3], 2))
    defer { mixed.free() }
    assert_eq(mixed.shape, [2, 6])
}

fn test_cross_matches_the_determinant_and_is_orthogonal() -> void {
    if not have_gpu() { return }
    let a = spread([4, 3], 22)
    let b = spread([4, 3], 23)
    defer { a.free() }
    defer { b.free() }

    let product = cross(a, b)
    defer { product.free() }
    assert_eq(product.shape, [4, 3])
    let av = a.to_list()
    let bv = b.to_list()
    let cv = product.to_list()
    for row in 0..4 {
        let base = row * 3
        assert_near(cv[base], av[base + 1] * bv[base + 2] - av[base + 2] * bv[base + 1], 1.0e-3)
        assert_near(cv[base + 1], av[base + 2] * bv[base] - av[base] * bv[base + 2], 1.0e-3)
        assert_near(cv[base + 2], av[base] * bv[base + 1] - av[base + 1] * bv[base], 1.0e-3)

        #: The property, checked separately from the formula: the product is
        #: normal to both operands.
        var along_a = 0.0
        var along_b = 0.0
        for at in 0..3 {
            along_a += cv[base + at] * av[base + at]
            along_b += cv[base + at] * bv[base + at]
        }
        assert_near(along_a, 0.0, 1.0e-2, "orthogonal to the left operand")
        assert_near(along_b, 0.0, 1.0e-2, "orthogonal to the right operand")
    }

    #: Two planar vectors have only the component normal to the plane, and it
    #: comes back without an axis to hold it.
    let planar = cross(spread([4, 2], 24), spread([4, 2], 25))
    defer { planar.free() }
    assert_eq(planar.shape, [4], "two 2-vectors give a scalar per row")

    #: A mixed pair reads the missing component as zero.
    let flat = NDArray([2], FLOAT32)
    flat.upload([1.0, 0.0])
    let solid = NDArray([3], FLOAT32)
    solid.upload([0.0, 1.0, 0.0])
    defer { flat.free() }
    defer { solid.free() }
    let mixed = cross(flat, solid)
    defer { mixed.free() }
    assert_eq(mixed.to_list(), [0.0, 0.0, 1.0], "x cross y is z")
    assert_throws(ValueError, || cross(spread([4], 1), spread([4], 2)), "4 components")

    #: `axisc` decides where the new axis lands.
    let columns = cross(a, b, axisc: 0)
    defer { columns.free() }
    assert_eq(columns.shape, [3, 4])
}

fn test_matrix_power_agrees_with_repeated_multiplication() -> void {
    if not have_gpu() { return }
    let a = spread([3, 3], 26)
    defer { a.free() }

    var expected = a.copy()
    for _step in 0..4 {
        let next = matmul(expected, a)
        expected.free()
        expected = next
    }
    defer { expected.free() }
    let squared_up = matrix_power(a, 5)
    defer { squared_up.free() }
    assert_close(squared_up.to_list(), expected.to_list(), "five products by squaring")

    let identity = matrix_power(a, 0)
    defer { identity.free() }
    assert_eq(identity.to_list(), [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

    #: The first power is a copy and not the array itself: writing to the
    #: result must not reach back into the operand.
    let once = matrix_power(a, 1)
    defer { once.free() }
    assert_eq(once.to_list(), a.to_list())
    once[0, 0] = 99.0
    assert_true(a.to_list()[0] != 99.0, "matrix_power(a, 1) does not alias a")

    #: A batch of matrices, each raised on its own — which also exercises the
    #: batched identity the general path starts from.
    let stack_of = spread([2, 3, 3], 27)
    defer { stack_of.free() }
    let batched = matrix_power(stack_of, 3)
    defer { batched.free() }
    assert_eq(batched.shape, [2, 3, 3])
    let cubed = matmul(matmul(stack_of, stack_of), stack_of)
    defer { cubed.free() }
    assert_close(batched.to_list(), cubed.to_list(), "a batch cubed")

    assert_throws(ValueError, || matrix_power(a, -1), "an inverse belongs to linalg")
    assert_throws(ValueError, || matrix_power(spread([3, 4], 1), 2), "not square")
    assert_throws(ValueError, || matrix_power(spread([3], 1), 2), "not a matrix")
}

fn test_einsum_reproduces_the_operations_it_generalises() -> void {
    if not have_gpu() { return }
    let a = spread([3, 4], 28)
    let b = spread([4, 5], 29)
    defer { a.free() }
    defer { b.free() }

    let product = einsum("ij,jk->ik", a, b)
    defer { product.free() }
    assert_close(product.to_list(), host_product(a.to_list(), b.to_list(), 3, 4, 5), "a matmul")

    #: The output order is the arrow's, not the operands', so this is the
    #: transpose of the product and needs the permutation to actually run.
    let flipped = einsum("ij,jk->ki", a, b)
    defer { flipped.free() }
    assert_eq(flipped.shape, [5, 3])
    let pv = product.to_list()
    let fv = flipped.to_list()
    for row in 0..3 {
        for column in 0..5 {
            assert_near(fv[column * 3 + row], pv[row * 5 + column], 1.0e-4, "transposed output")
        }
    }

    let outer_product = einsum("i,j->ij", spread([3], 1), spread([4], 2))
    defer { outer_product.free() }
    assert_eq(outer_product.shape, [3, 4])

    let full_contraction = einsum("ij,ij->", a, spread([3, 4], 30))
    defer { full_contraction.free() }
    assert_eq(full_contraction.shape, [])

    #: A batch label appears in both operands and in the result.
    let batched = einsum("bij,bjk->bik", spread([2, 3, 4], 31), spread([2, 4, 5], 32))
    defer { batched.free() }
    assert_eq(batched.shape, [2, 3, 5])

    #: A label in one operand and nowhere else is summed away first, which is
    #: its own contraction against a vector of ones.
    let summed = einsum("ij,jk->k", a, b)
    defer { summed.free() }
    assert_eq(summed.shape, [5])
    var expected = [0.0 for _index in 0..5]
    for column in 0..5 {
        var total = 0.0
        for row in 0..3 { total += pv[row * 5 + column] }
        expected[column] = total
    }
    assert_close(summed.to_list(), expected, "a lone label is summed out")

    #: Every label of one operand summed away, so the contracted axis is empty
    #: and the product is a scaled copy of the other operand.
    let scaled = einsum("ij,kl->kl", a, b)
    defer { scaled.free() }
    assert_eq(scaled.shape, [4, 5])
    var weight = 0.0
    for value in a.to_list() { weight += value }
    let bv = b.to_list()
    for at in 0..20 {
        assert_near(scaled.to_list()[at], weight * bv[at], 1.0e-3, f"scaled {at}")
    }

    #: A matrix against a vector, and two vectors against each other.
    let x = spread([4], 34)
    defer { x.free() }
    let applied = einsum("ij,j->i", a, x)
    defer { applied.free() }
    assert_close(
        applied.to_list(),
        host_product(a.to_list(), x.to_list(), 3, 4, 1),
        "einsum as a matrix-vector product"
    )
    let paired = einsum("i,i->", x, x)
    defer { paired.free() }
    var total = 0.0
    for value in x.to_list() { total += value * value }
    assert_near(paired.item(), total, 1.0e-3)
}

fn test_einsum_refuses_what_it_does_not_implement() -> void {
    if not have_gpu() { return }
    let a = spread([3, 3], 33)
    defer { a.free() }
    assert_throws(ValueError, || einsum("ij,jk", a, a), "no implicit output")
    assert_throws(ValueError, || einsum("...j,jk->...k", a, a), "no ellipsis")
    assert_throws(ValueError, || einsum("ij,jk,kl->il", a, a), "two operands only")
    assert_throws(ValueError, || einsum("ii,jk->jk", a, a), "a diagonal is not a product")
    assert_throws(ValueError, || einsum("ijk,jk->i", a, a), "three subscripts for two axes")
    assert_throws(ValueError, || einsum("ij,jk->iz", a, a), "a result label from nowhere")
    assert_throws(ValueError, || einsum("ij,jk->ik", a, spread([4, 3], 1)), "j is 3 and then 4")
}
