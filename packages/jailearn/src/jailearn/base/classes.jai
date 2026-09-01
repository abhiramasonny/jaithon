#: Totals accumulated per class code, and the scatter-add underneath them.
from jaitensor import Tensor

from .kernels import base_kernel
from .params import check_matrix, check_pair

fn class_moment(values: Tensor, labels: Tensor, width: int, classes: int, power: int) -> Tensor {
    if classes <= 0 { throw ValueError(f"classes must be positive, got {classes}") }
    let out = Tensor(if power == 0 { [classes] } else { [classes, width] })
    base_kernel("base_class_moment").dispatch_async(
        [values.buffer(), labels.buffer(), out.buffer()],
        classes * width * 256,
        [labels.size, width, classes, power],
        group_size: 256
    )
    return out
}

#: How many samples carry each code `0..classes-1`, as `[classes]`.
pub fn bincount(labels: Tensor, classes: int) -> Tensor {
    return class_moment(labels, labels, 1, classes, 0)
}

#: The column totals of each class, `[classes, n_features]`.
pub fn class_sums(x: Tensor, labels: Tensor, classes: int) -> Tensor {
    check_pair(x, labels)
    return class_moment(x, labels, x.shape[1], classes, 1)
}

#: The column totals of each class's squares, `[classes, n_features]`.
pub fn class_sumsq(x: Tensor, labels: Tensor, classes: int) -> Tensor {
    check_pair(x, labels)
    return class_moment(x, labels, x.shape[1], classes, 2)
}

#: Each row of `source` added into the row of the result its index names. The
#: order the additions land in is not fixed, so the last bits of a total vary
#: between runs on the same data.
pub fn scatter_add_rows(source: Tensor, indices: Tensor, rows: int) -> Tensor {
    check_matrix(source, "source")
    if indices.size != source.shape[0] {
        throw ValueError(f"{indices.size} indices for {source.shape[0]} rows")
    }
    if rows <= 0 { throw ValueError(f"destination rows must be positive, got {rows}") }
    let out = Tensor.zeros([rows, source.shape[1]])
    base_kernel("base_scatter_add_rows").dispatch_async(
        [source.buffer(), indices.buffer(), out.buffer()],
        source.size,
        [source.size, source.shape[1], rows]
    )
    return out
}
