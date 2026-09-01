#: Reductions over an axis or the whole tensor, and the scalar losses and
#: accumulators built on them.
from ..tensor import GradNode, Tensor
from .elementwise import scale, square, subtract
from .kernels import cached_kernel
from .shape import reshape

fn broadcast_scalar(source: Tensor, shape: list[int]) -> Tensor {
    let output = Tensor(shape)
    let kernel = cached_kernel("fill_from_scalar")
    kernel.dispatch_async([source.buffer(), output.buffer()], output.size, [output.size])
    return output
}

fn expand_last(reduced: Tensor, columns: int) -> Tensor {
    var shape: list[int] = []
    for dimension in reduced.shape { shape.push(dimension) }
    shape.push(columns)
    let output = Tensor(shape)
    let kernel = cached_kernel("broadcast_last")
    kernel.dispatch_async([reduced.buffer(), output.buffer()], output.size, [reduced.size, columns])
    return output
}

#: Sum of every element, or of the last axis when `axis` is `-1`.
pub fn sum(input: Tensor, axis: int? = null) -> Tensor {
    if axis is null {
        #: Enough groups to fill the machine and no more. Each one sweeps its
        #: share of the input in a register, so giving every element its own
        #: thread would only add threadgroups to combine.
        let ELEMENTS_PER_GROUP = 2048
        let groups = min(1024, max(1, (input.size + ELEMENTS_PER_GROUP - 1) // ELEMENTS_PER_GROUP))
        let partials = Tensor([groups])
        let kernel = cached_kernel("reduce_sum_all")
        kernel.dispatch_async(
            [input.buffer(), partials.buffer()],
            groups * 256,
            [input.size, groups * 256],
            group_size: 256
        )
        let output = if groups == 1 {
            partials
        } else {
            let total = sum(partials)
            partials.free()
            total
        }
        if input.requires_grad {
            output.attach_creator(GradNode([input], fn(upstream: Tensor) -> list[Tensor] {
                return [broadcast_scalar(upstream, input.shape)]
            }))
        }
        return output
    }
    let resolved = if axis < 0 { input.rank() + axis } else { axis }
    if resolved != input.rank() - 1 {
        throw ValueError("sum currently reduces all elements or the last axis")
    }
    let columns = input.shape[resolved]
    let rows = input.size // columns
    var out_shape: list[int] = []
    for i in 0..resolved { out_shape.push(input.shape[i]) }
    let output = Tensor(out_shape)
    let kernel = cached_kernel("reduce_sum_rows")
    kernel.dispatch_async(
        [input.buffer(), output.buffer()],
        rows * 256,
        [rows, columns],
        group_size: 256
    )
    if input.requires_grad {
        output.attach_creator(GradNode([input], fn(upstream: Tensor) -> list[Tensor] {
            return [expand_last(upstream, columns)]
        }))
    }
    return output
}

#: Mean of every element, or of the last axis.
pub fn mean(input: Tensor, axis: int? = null) -> Tensor {
    let reduced = sum(input, axis: axis)
    let count = if axis is null { float(input.size) } else { float(input.dim(-1)) }
    let scaled = scale(reduced, 1.0 / count)
    if not input.requires_grad { reduced.free() }
    return scaled
}

#: Mean squared error between equal-shaped tensors. Returns a scalar.
pub fn mse(prediction: Tensor, target: Tensor) -> Tensor {
    if not prediction.same_shape(target) {
        throw ValueError(f"mse shape mismatch: {prediction.shape} vs {target.shape}")
    }
    let diff = subtract(prediction, target)
    let sq = square(diff)
    let loss = mean(sq)
    if prediction.requires_grad or target.requires_grad { return loss }
    diff.free()
    sq.free()
    return loss
}

#: Mean binary cross-entropy between probabilities and 0/1 targets.
pub fn binary_cross_entropy(prediction: Tensor, target: Tensor) -> Tensor {
    if not prediction.same_shape(target) {
        throw ValueError(f"bce shape mismatch: {prediction.shape} vs {target.shape}")
    }
    let per = Tensor(prediction.shape)
    let kernel = cached_kernel("bce_sample")
    kernel.dispatch_async(
        [prediction.buffer(), target.buffer(), per.buffer()],
        prediction.size,
        [prediction.size]
    )
    if prediction.requires_grad {
        per.attach_creator(GradNode([prediction], fn(upstream: Tensor) -> list[Tensor] {
            let g = Tensor(prediction.shape)
            let backward_kernel = cached_kernel("bce_prob_backward")
            backward_kernel.dispatch_async([prediction.buffer(), target.buffer(), upstream.buffer(), g.buffer()], prediction.size, [prediction.size])
            return [g]
        }))
    }
    let loss = mean(per)
    if not prediction.requires_grad { per.free() }
    return loss
}

#: Mean binary cross-entropy from logits. Numerically stable; no sigmoid first.
pub fn binary_cross_entropy_with_logits(logits: Tensor, target: Tensor) -> Tensor {
    if not logits.same_shape(target) {
        throw ValueError(f"bce_with_logits shape mismatch: {logits.shape} vs {target.shape}")
    }
    let per = Tensor(logits.shape)
    let kernel = cached_kernel("bce_logits_sample")
    kernel.dispatch_async(
        [logits.buffer(), target.buffer(), per.buffer()],
        logits.size,
        [logits.size]
    )
    if logits.requires_grad {
        per.attach_creator(GradNode([logits], fn(upstream: Tensor) -> list[Tensor] {
            let g = Tensor(logits.shape)
            let backward_kernel = cached_kernel("bce_logits_backward")
            backward_kernel.dispatch_async([logits.buffer(), target.buffer(), upstream.buffer(), g.buffer()], logits.size, [logits.size])
            return [g]
        }))
    }
    let loss = mean(per)
    if not logits.requires_grad { per.free() }
    return loss
}

pub fn softmax(input: Tensor) -> Tensor {
    if input.rank() != 2 { throw ValueError(f"softmax needs rank two, got {input.shape}") }
    let rows = input.shape[0]
    let columns = input.shape[1]
    let output = Tensor(input.shape)
    let kernel = cached_kernel("row_softmax")
    kernel.dispatch_async(
        [input.buffer(), output.buffer()],
        rows * 256,
        [rows, columns],
        group_size: 256
    )
    if input.requires_grad {
        output.attach_creator(GradNode([input], fn(upstream: Tensor) -> list[Tensor] {
            let grad_input = Tensor(input.shape)
            let backward_kernel = cached_kernel("softmax_backward")
            backward_kernel.dispatch_async([output.buffer(), upstream.buffer(), grad_input.buffer()], rows * 256, [rows, columns], group_size: 256)
            return [grad_input]
        }))
    }
    return output
}

#: Index of the largest value in each row of a rank-two tensor.
#:
#: Indices use float storage because all Metal buffers in `std.gpu` are
#: float32. Downloaded values are exact for practical class counts.
#: The largest value along the last axis, or of the whole tensor.
#:
#: The companion to `argmax`, which gives where the largest is but not what it
#: is. A detector wants both -- the class and its score -- and computing the
#: value on the host afterwards means downloading everything the reduction was
#: supposed to save.
pub fn amax(input: Tensor, axis: int? = null) -> Tensor {
    if axis is null {
        let flat = reshape(input, [1, input.size])
        let whole = amax(flat, axis: 1)
        flat.free()
        return whole
    }
    let resolved = if axis < 0 { input.rank() + axis } else { axis }
    if resolved != input.rank() - 1 {
        throw ValueError("amax reduces all elements or the last axis")
    }
    let columns = input.shape[resolved]
    let rows = input.size // columns
    var shape: list[int] = []
    for index in 0..input.rank() - 1 { shape.push(input.shape[index]) }
    if shape.len() == 0 { shape.push(1) }
    let output = Tensor(shape)
    cached_kernel("row_max").dispatch_async(
        [input.buffer(), output.buffer()],
        rows * 256,
        [rows, columns],
        group_size: 256
    )
    return output
}

pub fn argmax(input: Tensor) -> Tensor {
    if input.rank() != 2 { throw ValueError(f"argmax needs rank two, got {input.shape}") }
    let rows = input.shape[0]
    let columns = input.shape[1]
    let output = Tensor([rows])
    let kernel = cached_kernel("row_argmax")
    kernel.dispatch_async(
        [input.buffer(), output.buffer()],
        rows * 256,
        [rows, columns],
        group_size: 256
    )
    return output
}

#: The largest value along an axis and where it sits, in one pass.
#:
#: `amax` and `argmax` each read the whole tensor to work out the same thing,
#: so asking for both costs twice what it needs to. Whoever wants only one of
#: them should still say so. On the 8400 by 80 class scores a detector produces
#: per frame the pair now costs 0.137 ms, against 0.65 ms when a row was one
#: thread's serial walk.
#:
#: Reducing down the columns rather than along the rows is a separate kernel
#: rather than a transpose and the other one. The transpose is a whole extra
#: pass over the tensor and an allocation the size of it -- 0.3 ms and 2.7 MB
#: for those same scores -- and reading a column is no slower: neighbouring
#: threads take neighbouring elements of each row in turn, which is the access
#: the device wants.
pub fn max_with_index(input: Tensor, axis: int = -1) -> tuple[Tensor, Tensor] {
    if input.rank() != 2 {
        throw ValueError(f"max_with_index needs rank two, got {input.shape}")
    }
    let resolved = if axis < 0 { input.rank() + axis } else { axis }
    if resolved != 0 and resolved != 1 {
        throw ValueError(f"max_with_index takes axis 0 or 1, got {axis}")
    }
    let rows = input.shape[0]
    let columns = input.shape[1]
    let kept = if resolved == 1 { rows } else { columns }
    let values = Tensor([kept])
    let indices = Tensor([kept])
    #: Down a row a threadgroup reduces together; across one, neighbouring
    #: threads already take neighbouring elements and a thread each is right.
    if resolved == 1 {
        cached_kernel("row_peak").dispatch_async(
            [input.buffer(), values.buffer(), indices.buffer()],
            kept * 256,
            [rows, columns],
            group_size: 256
        )
    } else {
        cached_kernel("column_peak").dispatch_async(
            [input.buffer(), values.buffer(), indices.buffer()],
            kept,
            [rows, columns]
        )
    }
    return (values, indices)
}

#: Add `value[0] * weight` into a scalar accumulator tensor.
pub fn add_scaled_into(value: Tensor, total: Tensor, weight: int) -> void {
    if value.size != 1 or total.size != 1 {
        throw ValueError("add_scaled_into needs scalar tensors")
    }
    if weight < 0 { throw ValueError(f"weight must be non-negative, got {weight}") }
    let kernel = cached_kernel("add_scaled_into")
    kernel.dispatch_async([value.buffer(), total.buffer()], 1, [weight])
}

#: Add the number of correct argmax classes into a scalar accumulator.
pub fn add_correct_into(logits: Tensor, labels: Tensor, total: Tensor) -> void {
    if logits.rank() != 2 {
        throw ValueError(f"add_correct_into needs rank-two logits, got {logits.shape}")
    }
    if total.size != 1 { throw ValueError("add_correct_into needs a scalar accumulator") }
    let batch = logits.shape[0]
    let classes = logits.shape[1]
    if labels.size != batch {
        throw ValueError(f"add_correct_into needs {batch} labels, got {labels.size}")
    }
    let kernel = cached_kernel("add_correct_into")
    kernel.dispatch_async(
        [logits.buffer(), labels.buffer(), total.buffer()],
        batch,
        [batch, classes]
    )
}
