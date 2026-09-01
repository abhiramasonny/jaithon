#: Reductions down a matrix's columns and along its rows, and the sorts underneath.
from jaitensor import Tensor

from .kernels import base_kernel
from .params import check_matrix

fn column_reduce(x: Tensor, mask: Tensor, mode: int, ddof: int) -> tuple[Tensor, Tensor] {
    check_matrix(x)
    let width = x.shape[1]
    let first = Tensor([width])
    let second = Tensor([width])
    base_kernel("base_column_reduce").dispatch_async(
        [x.buffer(), mask.buffer(), first.buffer(), second.buffer()],
        width * 256,
        [x.shape[0], width, mode, max(ddof, 0)],
        group_size: 256
    )
    return (first, second)
}

#: The mean and variance of every column, as two `[n_features]` vectors.
#:
#: `ddof` 0 is the population variance a scaler wants, 1 the sample variance a
#: statistic wants.
pub fn column_moments(x: Tensor, ddof: int = 0) -> tuple[Tensor, Tensor] {
    return column_reduce(x, x, 0, ddof)
}

#: The smallest and largest value in every column.
pub fn column_min_max(x: Tensor) -> tuple[Tensor, Tensor] { return column_reduce(x, x, 1, 0) }

#: The mean of every column over the entries `mask` leaves at zero.
#:
#: `mask` is the missingness, so it pairs with `nan_mask` directly.
pub fn masked_column_mean(x: Tensor, mask: Tensor) -> Tensor {
    if not x.same_shape(mask) { throw ValueError(f"mask {mask.shape} does not match X {x.shape}") }
    let (mean, unused) = column_reduce(x, mask, 2, 0)
    unused.free()
    return mean
}

fn next_power_of_two(n: int) -> int {
    var size = 1
    while size < n { size *= 2 }
    return size
}

#: Every column sorted ascending in a power-of-two tall scratch. The first
#: `rows` of each column are the answer and the pad above them is `+inf`.
fn bitonic_columns(x: Tensor, keep_order: bool) -> tuple[Tensor, Tensor] {
    let rows = x.shape[0]
    let width = x.shape[1]
    let padded = next_power_of_two(rows)
    let values = Tensor([padded, width])
    let order = if keep_order { Tensor([padded, width]) } else { Tensor([1]) }
    let tag = if keep_order { 1 } else { 0 }
    base_kernel("base_pad_columns").dispatch_async(
        [x.buffer(), values.buffer(), order.buffer()],
        padded * width,
        [rows, padded, width, tag]
    )
    let stage = base_kernel("base_bitonic_step")
    let threads = padded // 2 * width
    var span = 2
    while span <= padded {
        var step = span // 2
        while step > 0 {
            stage.dispatch_async(
                [values.buffer(), order.buffer()],
                threads,
                [padded, width, span, step, tag]
            )
            step = step // 2
        }
        span *= 2
    }
    return (values, order)
}

pub fn copy_into(source: Tensor, destination: Tensor, count: int, stride: int = 1) -> void {
    base_kernel("base_copy").dispatch_async(
        [source.buffer(), destination.buffer()],
        count,
        [count, stride]
    )
}

#: Every column of `x` sorted ascending, independently of the others.
pub fn sort_columns(x: Tensor) -> Tensor {
    check_matrix(x)
    let (values, order) = bitonic_columns(x, false)
    let out = Tensor(x.shape)
    copy_into(values, out, out.size)
    values.free()
    order.free()
    return out
}

#: Every column sorted, and the row each sorted value came from.
#:
#: Returns `(sorted, origin)`, both `[n_samples, n_features]`: the sort has to
#: happen either way, so the values come back beside the permutation instead of
#: making the caller gather them again. Equal values are not kept in row order.
pub fn argsort_columns(x: Tensor) -> tuple[Tensor, Tensor] {
    check_matrix(x)
    let (values, order) = bitonic_columns(x, true)
    let sorted = Tensor(x.shape)
    let origin = Tensor(x.shape)
    copy_into(values, sorted, sorted.size)
    copy_into(order, origin, origin.size)
    values.free()
    order.free()
    return (sorted, origin)
}

#: The given quantiles of every column, as `[len(qs), n_features]`.
#:
#: Linear interpolation between the two nearest order statistics, which is what
#: numpy and sklearn do. `KBinsDiscretizer`'s edges and the trees' `BinMapper`
#: are both this call, which is why neither of those modules imports the other.
pub fn column_quantiles(x: Tensor, qs: list[float]) -> Tensor {
    check_matrix(x)
    if qs.len() == 0 { throw ValueError("column_quantiles needs at least one quantile") }
    let (values, order) = bitonic_columns(x, false)
    let fractions = Tensor([qs.len()], qs)
    let out = Tensor([qs.len(), x.shape[1]])
    #: Read straight out of the padded buffer with `rows` as the height: the
    #: real values occupy exactly the first `rows` of every column, so nothing
    #: has to be compacted first.
    base_kernel("base_quantile_pick").dispatch_async(
        [values.buffer(), fractions.buffer(), out.buffer()],
        out.size,
        [x.shape[0], x.shape[1], qs.len()]
    )
    values.free()
    order.free()
    fractions.free()
    return out
}

#: The winning column of every row and the value that won it. Ties go to the
#: lowest column, matching every argmax in the package.
pub fn row_extreme(m: Tensor, largest: bool) -> tuple[Tensor, Tensor] {
    check_matrix(m, "m")
    let index = Tensor([m.shape[0]])
    let value = Tensor([m.shape[0]])
    let wanted = if largest { 1 } else { 0 }
    base_kernel("base_row_extreme").dispatch_async(
        [m.buffer(), index.buffer(), value.buffer()],
        m.shape[0],
        [m.shape[0], m.shape[1], wanted]
    )
    return (index, value)
}

#: The index of the largest entry in each row, as `[n_rows]` float codes.
pub fn row_argmax(m: Tensor) -> Tensor {
    let (index, value) = row_extreme(m, true)
    value.free()
    return index
}

#: The index of the smallest entry in each row.
pub fn row_argmin(m: Tensor) -> Tensor {
    let (index, value) = row_extreme(m, false)
    value.free()
    return index
}

#: The `k` best entries of every row: `(values, indices)`, both `[n_rows, k]`.
#: `largest` false picks the smallest, which is what a neighbour search wants
#: out of a distance matrix.
pub fn row_topk(m: Tensor, k: int, largest: bool = true) -> tuple[Tensor, Tensor] {
    check_matrix(m, "m")
    if k <= 0 or k > m.shape[1] { throw ValueError(f"cannot take {k} of {m.shape[1]} columns") }
    let values = Tensor([m.shape[0], k])
    let index = Tensor([m.shape[0], k])
    let wanted = if largest { 1 } else { 0 }
    base_kernel("base_row_topk").dispatch_async(
        [m.buffer(), values.buffer(), index.buffer()],
        m.shape[0],
        [m.shape[0], m.shape[1], k, wanted]
    )
    return (values, index)
}

#: `log(sum(exp(row)))` for every row, shifted by the row maximum.
pub fn log_sum_exp_rows(m: Tensor) -> Tensor {
    check_matrix(m, "m")
    let out = Tensor([m.shape[0]])
    base_kernel("base_log_sum_exp_rows").dispatch_async(
        [m.buffer(), out.buffer()],
        m.shape[0],
        [m.shape[0], m.shape[1]]
    )
    return out
}
