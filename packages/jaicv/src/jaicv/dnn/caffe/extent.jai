#: Caffe's per-axis parameters, and the output extents they work out to.
from .message import Message
from .reading import Step

#: Caffe writes one value for every spatial axis, or one per axis.
pub fn axis_pair(values: list[int], fallback: int) -> list[int] {
    if values.len() == 0 { return [fallback, fallback] }
    if values.len() == 1 { return [values[0], values[0]] }
    return [values[0], values[1]]
}

#: A convolution's per-axis value: the `_h`/`_w` pair wins over the repeated
#: field when the file sets either of them, which is Caffe's own order.
pub fn conv_pair(param: Message, plain: str, tall: str, wide: str, fallback: int) -> list[int] {
    if param.has(tall) or param.has(wide) {
        return [param.int_of(tall, fallback), param.int_of(wide, fallback)]
    }
    return axis_pair(param.ints_of(plain), fallback)
}

#: A pooling layer's, which Caffe reads by a different rule again: the square
#: field wins for the kernel, and only `_h` is tested for the other two.
pub fn pool_pair(
    param: Message,
    plain: str,
    tall: str,
    wide: str,
    fallback: int,
    square: bool
) -> list[int] {
    if square {
        if param.has(plain) {
            let value = param.int_of(plain, fallback)
            return [value, value]
        }
        return [param.int_of(tall, fallback), param.int_of(wide, fallback)]
    }
    if param.has(tall) { return [param.int_of(tall, fallback), param.int_of(wide, fallback)] }
    let value = param.int_of(plain, fallback)
    return [value, value]
}

#: A convolution's output extent along one axis.
pub fn convolved(size: int, window: int, stride: int, pad: int, dilation: int) -> int {
    return (size + 2 * pad - (dilation * (window - 1) + 1)) // stride + 1
}

#: Caffe's pooled extent, both axes at once because Caffe's correction is
#: guarded by whether *either* axis is padded.
#:
#: The ceiling is the easy half. The correction after it is the half that gets
#: missed: with padding, a window that would begin inside the trailing padding
#: is dropped, so an input that rounds up to one more window than it can
#: honestly fill produces one fewer.
pub fn pooled_pair(
    step: Step,
    size: list[int],
    window: list[int],
    stride: list[int],
    pad: list[int],
    ceiling: bool
) -> list[int] {
    var counts: list[int] = []
    for axis in 0..2 {
        if stride[axis] <= 0 {
            throw ValueError(f"{step.label()} pools with a stride of {stride[axis]}")
        }
        let extent = size[axis] + 2 * pad[axis] - window[axis]
        if extent < 0 {
            throw ValueError(
                f"{step.label()} slides a {window[0]}x{window[1]} window over a " +
                    f"{size[0]}x{size[1]} input padded by {pad[0]}x{pad[1]}, which does not fit"
            )
        }
        counts.push((if ceiling { -(-extent // stride[axis]) } else { extent // stride[axis] }) + 1)
    }
    if pad[0] > 0 or pad[1] > 0 {
        for axis in 0..2 {
            if (counts[axis] - 1) * stride[axis] >= size[axis] + pad[axis] { counts[axis] -= 1 }
        }
    }
    return counts
}
