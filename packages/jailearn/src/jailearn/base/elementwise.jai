#: Elementwise passes over a tensor, and the scalars read back off one.
import jaitensor as jt

from jaitensor import Tensor

from .kernels import base_kernel
from .params import check_matrix

#: `x[i][j] * scale[j] + offset[j]`, which is every scaler's apply step.
pub fn affine_columns(x: Tensor, scale: Tensor, offset: Tensor) -> Tensor {
    check_matrix(x)
    let width = x.shape[1]
    if scale.size != width or offset.size != width {
        throw ValueError(
            f"scale and offset must be [{width}], got {scale.shape} and {offset.shape}"
        )
    }
    let out = Tensor(x.shape)
    base_kernel("base_affine_columns").dispatch_async(
        [x.buffer(), scale.buffer(), offset.buffer(), out.buffer()],
        x.size,
        [x.size, width]
    )
    return out
}

#: How many positions hold the same value in both tensors.
#:
#: Counted by summing a mask in float32, so the answer is exact up to about
#: sixteen million matches and then starts rounding. Accuracy over a test set
#: that size is not the measurement anybody is taking.
pub fn count_equal(a: Tensor, b: Tensor) -> int {
    if a.size != b.size { throw ValueError(f"{a.shape} and {b.shape} have different sizes") }
    let mask = Tensor([a.size])
    base_kernel("base_mask_op").dispatch_async(
        [a.buffer(), b.buffer(), mask.buffer()],
        a.size,
        [a.size, 0]
    )
    let total = sum_to_host(mask)
    mask.free()
    return int(total + 0.5)
}

#: 1 where `x` holds a NaN and 0 elsewhere, same shape as `x`.
pub fn nan_mask(x: Tensor) -> Tensor {
    let mask = Tensor(x.shape)
    base_kernel("base_mask_op").dispatch_async(
        [x.buffer(), x.buffer(), mask.buffer()],
        x.size,
        [x.size, 1]
    )
    return mask
}

pub fn sum_to_host(t: Tensor) -> float {
    let total = jt.sum(t)
    let value = total.download()[0]
    total.free()
    return value
}
