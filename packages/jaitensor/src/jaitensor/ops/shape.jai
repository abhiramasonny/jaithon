#: Shape and layout: views, concatenation, padding, axis moves, and the
#: row gathers built on them.
from ..tensor import GradNode, Tensor, element_count
from .kernels import cached_attn_kernel, cached_kernel

#: Reinterpret a tensor as `shape` without copying device memory.
pub fn reshape(input: Tensor, shape: list[int]) -> Tensor {
    if element_count(shape) != input.size {
        throw ValueError(f"cannot reshape {input.shape} into {shape}")
    }
    let output = Tensor(shape, storage: input.buffer().view(0, input.size))
    if input.requires_grad {
        let original_shape = input.shape
        output.attach_creator(GradNode([input], |upstream| [reshape(upstream, original_shape)]))
    }
    return output
}

fn raw_transpose(input: Tensor) -> Tensor {
    let rows = input.shape[0]
    let columns = input.shape[1]
    let output = Tensor([columns, rows])
    let kernel = cached_kernel("matrix_transpose")
    let across = (columns + 31) // 32
    let down = (rows + 31) // 32
    kernel.dispatch_async(
        [input.buffer(), output.buffer()],
        across * down * 256,
        [rows, columns, across],
        group_size: 256
    )
    return output
}

#: Transpose a rank-two tensor.
pub fn transpose(input: Tensor) -> Tensor {
    if input.rank() != 2 { throw ValueError(f"transpose needs rank two, got {input.shape}") }
    let output = raw_transpose(input)
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [raw_transpose(upstream)]))
    }
    return output
}

#: Concatenate tensors along `axis` (0 or the last axis).
pub fn cat(parts: list[Tensor], axis: int = 0) -> Tensor {
    if parts.len() == 0 { throw ValueError("cat needs at least one tensor") }
    let rank = parts[0].rank()
    let resolved = if axis < 0 { rank + axis } else { axis }
    if resolved != 0 and resolved != rank - 1 {
        throw ValueError("cat currently supports axis 0 or the last axis")
    }
    var along = 0
    for part in parts {
        if part.rank() != rank { throw ValueError("cat parts must have the same rank") }
        for i in 0..rank {
            if i == resolved { continue }
            if part.shape[i] != parts[0].shape[i] {
                throw ValueError(
                    f"cat shape mismatch at axis {i}: {part.shape} vs {parts[0].shape}"
                )
            }
        }
        along += part.shape[resolved]
    }
    var out_shape: list[int] = []
    for i in 0..rank {
        out_shape.push(if i == resolved { along } else { parts[0].shape[i] })
    }
    let output = Tensor(out_shape)
    var cursor = 0
    let kernel = cached_kernel("copy_range")
    var wants_grad = false
    for part in parts {
        if part.requires_grad { wants_grad = true }
    }
    if resolved == 0 {
        var offsets: list[int] = []
        for part in parts {
            offsets.push(cursor)
            kernel.dispatch_async(
                [part.buffer(), output.buffer()],
                part.size,
                [0, cursor, part.size]
            )
            cursor += part.size
        }
        if wants_grad {
            let shapes = [part.shape for part in parts]
            output.attach_creator(GradNode(parts, fn(upstream: Tensor) -> list[Tensor] {
                var grads: list[Tensor] = []
                let back = cached_kernel("copy_range")
                for (index, shape) in shapes.enumerate() {
                    let grad = Tensor(shape)
                    back.dispatch_async([upstream.buffer(), grad.buffer()], grad.size, [offsets[index], 0, grad.size])
                    grads.push(grad)
                }
                return grads
            }))
        }
        return output
    }
    let outer = parts[0].size // parts[0].shape[resolved]
    var prefix = 0
    var starts: list[int] = []
    let strided = cached_kernel("copy_strided_rows")
    for part in parts {
        let width = part.shape[resolved]
        starts.push(prefix)
        strided.dispatch_async(
            [part.buffer(), output.buffer()],
            outer * width,
            [outer, width, along, prefix]
        )
        prefix += width
    }
    if wants_grad {
        let shapes = [part.shape for part in parts]
        output.attach_creator(GradNode(parts, fn(upstream: Tensor) -> list[Tensor] {
            var grads: list[Tensor] = []
            let take = cached_kernel("take_strided_rows")
            for (index, shape) in shapes.enumerate() {
                let grad = Tensor(shape)
                let width = shape[resolved]
                take.dispatch_async([upstream.buffer(), grad.buffer()], outer * width, [outer, width, along, starts[index]])
                grads.push(grad)
            }
            return grads
        }))
    }
    return output
}

#: Insert a size-1 axis at `axis`.
pub fn unsqueeze(input: Tensor, axis: int) -> Tensor {
    let rank = input.rank()
    let resolved = if axis < 0 { rank + 1 + axis } else { axis }
    if resolved < 0 or resolved > rank {
        throw ValueError(f"unsqueeze axis {axis} is outside rank {rank}")
    }
    var shape: list[int] = []
    for i in 0..resolved { shape.push(input.shape[i]) }
    shape.push(1)
    for i in resolved..rank { shape.push(input.shape[i]) }
    return reshape(input, shape)
}

#: Drop size-1 axes. `axis` null drops every size-1 axis.
pub fn squeeze(input: Tensor, axis: int? = null) -> Tensor {
    var shape: list[int] = []
    if axis is null {
        for d in input.shape {
            if d != 1 { shape.push(d) }
        }
        if shape.len() == 0 { shape.push(1) }
        return reshape(input, shape)
    }
    let resolved = if axis < 0 { input.rank() + axis } else { axis }
    if resolved < 0 or resolved >= input.rank() {
        throw ValueError(f"squeeze axis {axis} is outside rank {input.rank()}")
    }
    if input.shape[resolved] != 1 {
        throw ValueError(f"cannot squeeze axis {axis} of size {input.shape[resolved]}")
    }
    for i in 0..input.rank() {
        if i != resolved { shape.push(input.shape[i]) }
    }
    if shape.len() == 0 { shape.push(1) }
    return reshape(input, shape)
}

#: Stack tensors by inserting a new leading axis, then concatenating.
pub fn stack(parts: list[Tensor]) -> Tensor {
    if parts.len() == 0 { throw ValueError("stack needs at least one tensor") }
    var expanded: list[Tensor] = []
    for part in parts { expanded.push(unsqueeze(part, 0)) }
    return cat(expanded, axis: 0)
}

#: Zero-pad an NHWC tensor. `pad` is `(top, bottom, left, right)`.
pub fn pad2d(input: Tensor, pad: list[int]) -> Tensor {
    if input.rank() != 4 { throw ValueError(f"pad2d needs NHWC input, got {input.shape}") }
    if pad.len() != 4 { throw ValueError("pad2d pad must be [top, bottom, left, right]") }
    for p in pad {
        if p < 0 { throw ValueError("pad2d pad values must be non-negative") }
    }
    let n = input.shape[0]
    let h = input.shape[1]
    let w = input.shape[2]
    let c = input.shape[3]
    let output = Tensor([n, h + pad[0] + pad[1], w + pad[2] + pad[3], c])
    let wide = c % 4 == 0
    let entry = if wide { "pad2d_nhwc4" } else { "pad2d_nhwc" }
    let threads = if wide { output.size // 4 } else { output.size }
    cached_kernel(entry).dispatch_async(
        [input.buffer(), output.buffer()],
        threads,
        [n, h, w, c, pad[0], pad[1], pad[2], pad[3]]
    )
    if input.requires_grad {
        let padded_h = h + pad[0] + pad[1]
        let padded_w = w + pad[2] + pad[3]
        let top = pad[0]
        let left = pad[2]
        output.attach_creator(GradNode([input], fn(upstream: Tensor) -> list[Tensor] {
            let grad = Tensor([n, h, w, c])
            let crop = cached_kernel("crop2d_nhwc")
            crop.dispatch_async([upstream.buffer(), grad.buffer()], grad.size, [n, h, w, c, top, left, padded_h, padded_w])
            return [grad]
        }))
    }
    return output
}

#: NHWC `[N,H,W,C]` to NCHW `[N,C,H,W]`.
#: Tiles a side, and the group that walks one. Matched by `TR_TILE` and
#: `TR_ROWS` in the kernel source.
const TRANSPOSE_TILE = 32
const TRANSPOSE_GROUP = 256

#: `count` matrices of `rows` by `cols`, each written back the other way up.
fn dispatch_transpose(input: Tensor, output: Tensor, count: int, rows: int, cols: int) -> void {
    let across = (cols + TRANSPOSE_TILE - 1) // TRANSPOSE_TILE
    let down = (rows + TRANSPOSE_TILE - 1) // TRANSPOSE_TILE
    cached_kernel("transpose_batched").dispatch_async(
        [input.buffer(), output.buffer()],
        count * across * down * TRANSPOSE_GROUP,
        [rows, cols, across, down],
        group_size: TRANSPOSE_GROUP
    )
}

pub fn nchw(input: Tensor) -> Tensor {
    if input.rank() != 4 { throw ValueError(f"nchw needs rank four, got {input.shape}") }
    let n = input.shape[0]
    let h = input.shape[1]
    let w = input.shape[2]
    let c = input.shape[3]
    let output = Tensor([n, c, h, w])
    dispatch_transpose(input, output, n, h * w, c)
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [nhwc(upstream)]))
    }
    return output
}

#: NCHW `[N,C,H,W]` to NHWC `[N,H,W,C]`.
pub fn nhwc(input: Tensor) -> Tensor {
    if input.rank() != 4 { throw ValueError(f"nhwc needs rank four, got {input.shape}") }
    let n = input.shape[0]
    let c = input.shape[1]
    let h = input.shape[2]
    let w = input.shape[3]
    let output = Tensor([n, h, w, c])
    dispatch_transpose(input, output, n, c, h * w)
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [nchw(upstream)]))
    }
    return output
}

#: Permute axes. Rank two is transpose. Rank four supports NHWC<->NCHW layouts.
pub fn permute(input: Tensor, axes: list[int]) -> Tensor {
    if axes.len() != input.rank() {
        throw ValueError(f"permute axes length {axes.len()} does not match rank {input.rank()}")
    }
    if input.rank() == 2 {
        if axes[0] == 1 and axes[1] == 0 { return transpose(input) }
        if axes[0] == 0 and axes[1] == 1 { return reshape(input, input.shape) }
        throw ValueError(f"unsupported permute {axes}")
    }
    if input.rank() == 4 {
        if axes == [0, 3, 1, 2] { return nchw(input) }
        if axes == [0, 2, 3, 1] { return nhwc(input) }
        if axes == [0, 1, 2, 3] { return reshape(input, input.shape) }
        throw ValueError(f"unsupported permute {axes}")
    }
    throw ValueError("permute currently supports rank two and four")
}

#: Repeat the last axis `times` times.
pub fn repeat(input: Tensor, times: int, axis: int = -1) -> Tensor {
    if times <= 0 { throw ValueError(f"repeat times must be positive, got {times}") }
    let resolved = if axis < 0 { input.rank() + axis } else { axis }
    if resolved != input.rank() - 1 {
        throw ValueError("repeat currently supports the last axis")
    }
    let inner = input.shape[resolved]
    let rows = input.size // inner
    var shape: list[int] = []
    for i in 0..input.rank() {
        shape.push(if i == resolved { inner * times } else { input.shape[i] })
    }
    let output = Tensor(shape)
    let wide = inner % 4 == 0
    let entry = if wide { "repeat_last4" } else { "repeat_last" }
    let threads = if wide { output.size // 4 } else { output.size }
    cached_kernel(entry).dispatch_async(
        [input.buffer(), output.buffer()],
        threads,
        [rows, inner, times]
    )
    if input.requires_grad {
        let original = input.shape
        output.attach_creator(GradNode([input], fn(upstream: Tensor) -> list[Tensor] {
            let grad = Tensor(original)
            let back = cached_kernel("repeat_last_backward")
            back.dispatch_async([upstream.buffer(), grad.buffer()], grad.size, [rows, inner, times])
            return [grad]
        }))
    }
    return output
}

#: Keep the lower triangle of a rank-two tensor, including the diagonal.
pub fn tril(input: Tensor) -> Tensor {
    if input.rank() != 2 { throw ValueError(f"tril needs rank two, got {input.shape}") }
    let rows = input.shape[0]
    let cols = input.shape[1]
    let output = Tensor(input.shape)
    let kernel = cached_attn_kernel("tril_square")
    kernel.dispatch_async([input.buffer(), output.buffer()], output.size, [rows, cols])
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [tril(upstream)]))
    }
    return output
}

#: Identity matrix of size `n` on the GPU.
pub fn eye(n: int) -> Tensor {
    if n <= 0 { throw ValueError(f"eye needs a positive size, got {n}") }
    let output = Tensor([n, n])
    let kernel = cached_kernel("eye_fill")
    kernel.dispatch_async([output.buffer()], output.size, [n])
    return output
}

#: Reverse the last axis. Other axes are not yet supported.
pub fn flip(input: Tensor, axis: int = -1) -> Tensor {
    let rank = input.rank()
    let resolved = if axis < 0 { rank + axis } else { axis }
    if resolved != rank - 1 {
        throw ValueError("flip currently supports the last axis")
    }
    let inner = input.shape[resolved]
    let rows = input.size // inner
    let output = Tensor(input.shape)
    let kernel = cached_kernel("flip_last")
    kernel.dispatch_async([input.buffer(), output.buffer()], output.size, [rows, inner])
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [flip(upstream, axis: axis)]))
    }
    return output
}

#: Nearest-neighbor upsample of NHWC images by an integer `scale`.
pub fn upsample_nearest2d(input: Tensor, scale: int) -> Tensor {
    if input.rank() != 4 {
        throw ValueError(f"upsample_nearest2d needs NHWC input, got {input.shape}")
    }
    if scale <= 0 { throw ValueError(f"upsample scale must be positive, got {scale}") }
    let n = input.shape[0]
    let h = input.shape[1]
    let w = input.shape[2]
    let c = input.shape[3]
    let out_h = h * scale
    let out_w = w * scale
    let output = Tensor([n, out_h, out_w, c])
    let wide = c % 4 == 0
    let entry = if wide { "upsample_nearest2d_nhwc4" } else { "upsample_nearest2d_nhwc" }
    let threads = if wide { output.size // 4 } else { output.size }
    cached_kernel(entry).dispatch_async(
        [input.buffer(), output.buffer()],
        threads,
        [n, h, w, c, scale, scale, out_h, out_w]
    )
    if input.requires_grad {
        output.attach_creator(GradNode([input], fn(upstream: Tensor) -> list[Tensor] {
            let grad = Tensor([n, h, w, c])
            let back = cached_kernel("upsample_nearest2d_backward_nhwc")
            back.dispatch_async([upstream.buffer(), grad.buffer()], grad.size, [n, h, w, c, scale, scale, out_h, out_w])
            return [grad]
        }))
    }
    return output
}

#: Pack `[seq, dim]` into `[heads * seq, head_dim]` with each head's rows contiguous.
pub fn split_heads(input: Tensor, heads: int) -> Tensor {
    if input.rank() != 2 {
        throw ValueError(f"split_heads needs rank two, got {input.shape}")
    }
    if heads <= 0 { throw ValueError(f"split_heads needs a positive head count, got {heads}") }
    let seq = input.shape[0]
    let dim = input.shape[1]
    if dim % heads != 0 {
        throw ValueError(f"split_heads dim {dim} must divide heads {heads}")
    }
    let hd = dim // heads
    let output = Tensor([heads * seq, hd])
    let kernel = cached_attn_kernel("split_heads")
    kernel.dispatch_async([input.buffer(), output.buffer()], output.size, [seq, heads, hd])
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [merge_heads(upstream, heads, seq)]))
    }
    return output
}

#: Inverse of `split_heads`: `[heads * seq, head_dim]` back to `[seq, dim]`.
pub fn merge_heads(packed: Tensor, heads: int, seq: int) -> Tensor {
    if packed.rank() != 2 {
        throw ValueError(f"merge_heads needs rank two, got {packed.shape}")
    }
    if heads <= 0 or seq <= 0 {
        throw ValueError("merge_heads needs positive heads and sequence length")
    }
    if packed.shape[0] != heads * seq {
        throw ValueError(f"merge_heads packed rows {packed.shape[0]} != {heads} * {seq}")
    }
    let hd = packed.shape[1]
    let output = Tensor([seq, heads * hd])
    let kernel = cached_attn_kernel("merge_heads")
    kernel.dispatch_async([packed.buffer(), output.buffer()], packed.size, [seq, heads, hd])
    if packed.requires_grad {
        output.attach_creator(GradNode([packed], |upstream| [split_heads(upstream, heads)]))
    }
    return output
}

#: Expand a rank-one tensor of integer-valued indices into one-hot rows.
pub fn one_hot(indices: Tensor, depth: int) -> Tensor {
    if indices.rank() != 1 { throw ValueError(f"one_hot needs rank one, got {indices.shape}") }
    if depth <= 0 { throw ValueError(f"depth must be positive, got {depth}") }
    let rows = indices.shape[0]
    let output = Tensor([rows, depth])
    let kernel = cached_kernel("one_hot_rows")
    kernel.dispatch_async([indices.buffer(), output.buffer()], rows * depth, [rows, depth])
    return output
}

#: Copy a contiguous range of `source` as a view of the same device memory.
#:
#: `source_offset` is in elements. The result is a leaf: it does not attach a
#: gradient graph. Freeing the view does not free `source`.
pub fn slice_flat(source: Tensor, source_offset: int, shape: list[int]) -> Tensor {
    let n = element_count(shape)
    if source_offset < 0 or source_offset + n > source.size {
        throw ValueError(
            f"slice_flat offset {source_offset} and size {n} " +
                f"run past source size {source.size}"
        )
    }
    let output = source.view(source_offset, shape)
    if source.requires_grad {
        let whole = source.shape
        output.attach_creator(GradNode([source], fn(upstream: Tensor) -> list[Tensor] {
            let grad = Tensor.zeros(whole)
            let kernel = cached_kernel("copy_range")
            kernel.dispatch_async([upstream.buffer(), grad.buffer()], n, [0, source_offset, n])
            return [grad]
        }))
    }
    return output
}

#: Copy `source` into `dest`. Both must have the same number of elements.
pub fn copy_into(source: Tensor, dest: Tensor) -> void {
    if source.size != dest.size {
        throw ValueError(f"copy_into size mismatch: {source.size} vs {dest.size}")
    }
    let kernel = cached_kernel("copy_offset")
    kernel.dispatch_async([source.buffer(), dest.buffer()], dest.size, [0, dest.size])
}

#: Gather `indices.len()` rows from a rank-two source.
#: Read whole rows of `source`, one per entry of `indices`.
#:
#: The gradient of this scatters back into the table with device atomics, and
#: a token that occurs twice in a batch has two threads adding to the same
#: slot. Float addition is not associative, so which order they arrive in
#: changes the last few bits: two runs from the same seed on the same data
#: give gradients that agree to about ten digits and not beyond. It is enough
#: to make a long training run diverge from its own repeat. Other libraries
#: are the same here for the same reason; making it exact means sorting the
#: indices and reducing each run of them in order, which costs more than the
#: reproducibility is usually worth.
pub fn gather_rows(source: Tensor, indices: Tensor, out_shape: list[int]? = null) -> Tensor {
    if source.rank() != 2 {
        throw ValueError(f"gather_rows needs a rank-two source, got {source.shape}")
    }
    #: The indices are read flat whatever shape they arrive in -- only how
    #: many there are matters, and a batch of index rows is the ordinary way
    #: an embedding is asked for. `out_shape` says what to call the result.
    if indices.rank() < 1 {
        throw ValueError(f"gather_rows needs an index tensor, got {indices.shape}")
    }
    let width = source.shape[1]
    var shape = [indices.size, width]
    if out_shape is not null {
        if element_count(out_shape) != indices.size * width {
            throw ValueError(
                f"gather_rows cannot reshape {indices.size}x{width} rows into {out_shape}"
            )
        }
        shape = out_shape
    }
    let output = Tensor(shape)
    let kernel = cached_kernel("gather_rows")
    kernel.dispatch_async(
        [source.buffer(), indices.buffer(), output.buffer()],
        output.size,
        [indices.size, width, source.shape[0]]
    )
    if source.requires_grad {
        output.attach_creator(GradNode([source], fn(upstream: Tensor) -> list[Tensor] {
            return [scatter_add_rows(upstream, indices, source.shape[0], width)]
        }))
    }
    return output
}

fn scatter_add_rows(upstream: Tensor, indices: Tensor, vocab: int, width: int) -> Tensor {
    let dest = Tensor.zeros([vocab, width])
    let kernel = cached_kernel("scatter_add_rows")
    kernel.dispatch_async(
        [upstream.buffer(), indices.buffer(), dest.buffer()],
        upstream.size,
        [indices.size, width, vocab]
    )
    return dest
}
