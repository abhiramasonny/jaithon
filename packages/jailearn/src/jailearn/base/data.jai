#: Matrices in and out of the host, and the columns and rows taken from them.
import jaitensor as jt

from jaitensor import Tensor

from .kernels import base_kernel
from .params import check_matrix

#: A `[n_samples, n_features]` tensor from host rows.
#:
#: For hand-written data: examples, tests, the odd small table. Real data
#: should reach the device through `jaitensor`'s loaders, because this walks
#: every element on the host.
pub fn matrix(rows: list[list[float]]) -> Tensor {
    if rows.len() == 0 { throw ValueError("a design matrix needs at least one row") }
    let width = rows[0].len()
    if width == 0 { throw ValueError("a design matrix needs at least one feature") }
    var flat: list[float] = []
    for (index, row) in rows.enumerate() {
        if row.len() != width {
            throw ValueError(f"row {index} has {row.len()} features, expected {width}")
        }
        for value in row { flat.push(value) }
    }
    return Tensor([rows.len(), width], flat)
}

#: A `[n_samples]` tensor from host values, for targets and label vectors.
pub fn vector(values: list[float]) -> Tensor {
    if values.len() == 0 { throw ValueError("a target vector needs at least one value") }
    return Tensor([values.len()], values)
}

#: A matrix back as host rows, with the same warning as `matrix`.
pub fn to_rows(x: Tensor) -> list[list[float]] {
    check_matrix(x)
    let width = x.shape[1]
    let flat = x.download()
    var rows: list[list[float]] = []
    for row in 0..x.shape[0] {
        var out: list[float] = []
        for column in 0..width { out.push(flat[row * width + column]) }
        rows.push(out)
    }
    return rows
}

#: A tensor back as host values, whatever its shape.
pub fn to_values(v: Tensor) -> list[float] { return v.download() }

#: Column `j` of `x`, as a `[n_samples]` vector.
pub fn column(x: Tensor, j: int) -> Tensor {
    check_matrix(x)
    if j < 0 or j >= x.shape[1] {
        throw IndexError(f"column {j} is outside a {x.shape[1]}-feature matrix")
    }
    return gather_columns(x, [j], [x.shape[0]])
}

#: The named columns of `x`, in the order given, as `[n_samples, len(indices)]`.
pub fn select_columns(x: Tensor, indices: list[int]) -> Tensor {
    check_matrix(x)
    if indices.len() == 0 { throw ValueError("select_columns needs at least one column") }
    for index in indices {
        if index < 0 or index >= x.shape[1] {
            throw IndexError(f"column {index} is outside a {x.shape[1]}-feature matrix")
        }
    }
    return gather_columns(x, indices, [x.shape[0], indices.len()])
}

fn gather_columns(x: Tensor, indices: list[int], shape: list[int]) -> Tensor {
    let table = index_tensor(indices)
    let out = Tensor(shape)
    base_kernel("base_gather_columns").dispatch_async(
        [x.buffer(), table.buffer(), out.buffer()],
        out.size,
        [x.shape[0], x.shape[1], indices.len()]
    )
    table.free()
    return out
}

#: Matrices laid side by side, sharing a sample count.
pub fn hstack(parts: list[Tensor]) -> Tensor {
    if parts.len() == 0 { throw ValueError("hstack needs at least one matrix") }
    let rows = parts[0].shape[0]
    var total = 0
    for part in parts {
        check_matrix(part)
        if part.shape[0] != rows {
            throw ValueError(f"hstack got {part.shape[0]} rows against {rows}")
        }
        total += part.shape[1]
    }
    let out = Tensor([rows, total])
    var offset = 0
    for part in parts {
        base_kernel("base_place_block").dispatch_async(
            [part.buffer(), out.buffer()],
            part.size,
            [part.size, part.shape[1], total, offset]
        )
        offset += part.shape[1]
    }
    return out
}

#: The rows `indices` names, in that order.
#:
#: Folds, bootstraps and shuffles all come out of this, so the index vector
#: stays on the device rather than being rebuilt on the host once per fold.
pub fn take_rows(x: Tensor, indices: Tensor) -> Tensor {
    check_matrix(x)
    return jt.gather_rows(x, indices)
}

#: A `[count]` index vector from host integers, for index lists a caller builds
#: itself and that are short by construction.
pub fn index_tensor(indices: list[int]) -> Tensor {
    if indices.len() == 0 { throw ValueError("an index vector needs at least one index") }
    var values: list[float] = []
    for index in indices { values.push(float(index)) }
    return Tensor([values.len()], values)
}
