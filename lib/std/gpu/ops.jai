#: Buffer-level graph operations — GEMM, attention, convolution and the fused
#: MLP steps — queued on the device and never waited on.
from std.gpu.device import _require_device
from std.gpu.kernel import Buffer

#: Queue a device-buffer matrix product without waiting.
#:
#: `trans_a` / `trans_b` read `a` or `b` as transposed. Row bytes of each
#: physical matrix must be a multiple of 16 (width divisible by 4). Used by
#: jaitensor for GEMMs large enough that Metal Performance Shaders wins.
#:
#: `half` runs the product in half precision, converting on the way in and back
#: on the way out. Whether that is a gain depends on the shape — the conversion
#: touches the whole of the result, so a product with a large output and a
#: short inner dimension spends more on converting than it saves on
#: multiplying — which is why it is asked for per call rather than taken from
#: the device-wide mixed-precision setting.
pub fn matmul_async(
    a: Buffer,
    b: Buffer,
    output: Buffer,
    rows: int,
    inner: int,
    columns: int,
    trans_a: bool = false,
    trans_b: bool = false,
    half: bool = false
) -> void {
    _require_device("matmul_async")
    if rows < 0 or inner < 0 or columns < 0 {
        throw ValueError(
            f"matmul_async dimensions must be non-negative, got {rows}x{inner}x{columns}"
        )
    }
    __prim__.gpu_matmul_buffers(
        a.handle_of(),
        b.handle_of(),
        output.handle_of(),
        rows,
        inner,
        columns,
        trans_a,
        trans_b,
        half
    )
}

#: Packed multi-head attention. Q/K/V/out are `[seq, heads*hd]`.
#:
#: Returns false when the device has no fused SDPA path so callers can
#: fall back to a Metal kernel.
pub fn mha_async(
    query: Buffer,
    key: Buffer,
    value: Buffer,
    output: Buffer,
    seq: int,
    heads: int,
    hd: int,
    scale: float
) -> bool {
    _require_device("mha_async")
    if seq <= 0 or heads <= 0 or hd <= 0 {
        throw ValueError(f"mha_async dimensions must be positive, got {seq}x{heads}x{hd}")
    }
    if not scale > 0.0 {
        throw ValueError("mha_async scale must be positive")
    }
    return __prim__.gpu_mha_buffers(
        query.handle_of(),
        key.handle_of(),
        value.handle_of(),
        output.handle_of(),
        seq,
        heads,
        hd,
        scale
    )
}

#: Queue a 2D convolution without waiting.
#:
#: `layout` says how the caller has its data arranged: 0 is NHWC activations
#: with HWIO `[kh, kw, cin, cout]` weights, 1 is NCHW with OIHW
#: `[cout, cin, kh, kw]`. The output comes back in the same layout as the
#: input. `bias` is optional `cout` values either way.
#:
#: `activation` fuses an elementwise epilogue into the same dispatch: 0 none,
#: 1 ReLU, 2 SiLU. Fusing here means one graph execution covers conv, bias and
#: activation instead of a second full pass over the output.
pub fn conv2d_async(
    input: Buffer,
    weights: Buffer,
    output: Buffer,
    n: int,
    height: int,
    width: int,
    cin: int,
    cout: int,
    kh: int,
    kw: int,
    stride_h: int = 1,
    stride_w: int = 1,
    pad_h: int = 0,
    pad_w: int = 0,
    bias: Buffer? = null,
    activation: int = 0,
    layout: int = 0
) -> void {
    _require_device("conv2d_async")
    if n <= 0 or height <= 0 or width <= 0 or cin <= 0 or cout <= 0 or kh <= 0 or kw <= 0 {
        throw ValueError("conv2d_async dimensions must be positive")
    }
    if stride_h <= 0 or stride_w <= 0 {
        throw ValueError("conv2d_async stride must be positive")
    }
    if activation < 0 or activation > 2 {
        throw ValueError("conv2d_async activation must be 0, 1 or 2")
    }
    if layout < 0 or layout > 1 {
        throw ValueError("conv2d_async layout must be 0 or 1")
    }
    let bias_handle = if bias is null { null } else { bias.handle_of() }
    __prim__.gpu_conv2d_buffers(
        input.handle_of(),
        weights.handle_of(),
        bias_handle,
        output.handle_of(),
        n,
        height,
        width,
        cin,
        cout,
        kh,
        kw,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        activation,
        layout
    )
}

#: The convolution's gradient with respect to its input.
#:
#: `grad` is the gradient arriving at the output, `weights` the filter the
#: forward pass used, and `output` receives a tensor the shape of the input.
#: The dimensions are the FORWARD pass's, so the same numbers describe both
#: directions. False means this device or this shape has no such primitive and
#: the caller should fall back to its own.
pub fn conv2d_data_grad_async(
    grad: Buffer,
    weights: Buffer,
    output: Buffer,
    n: int,
    height: int,
    width: int,
    cin: int,
    cout: int,
    kh: int,
    kw: int,
    stride_h: int = 1,
    stride_w: int = 1,
    pad_h: int = 0,
    pad_w: int = 0
) -> bool {
    _require_device("conv2d_data_grad_async")
    if n <= 0 or height <= 0 or width <= 0 or cin <= 0 or cout <= 0 or kh <= 0 or kw <= 0 {
        throw ValueError("conv2d_data_grad_async dimensions must be positive")
    }
    if stride_h <= 0 or stride_w <= 0 {
        throw ValueError("conv2d_data_grad_async stride must be positive")
    }
    return __prim__.gpu_conv2d_data_grad(
        grad.handle_of(),
        weights.handle_of(),
        output.handle_of(),
        n,
        height,
        width,
        cin,
        cout,
        kh,
        kw,
        stride_h,
        stride_w,
        pad_h,
        pad_w
    )
}

#: The same convolution's gradient with respect to its weights. `output`
#: receives a tensor the shape of the filter.
pub fn conv2d_weights_grad_async(
    grad: Buffer,
    input: Buffer,
    output: Buffer,
    n: int,
    height: int,
    width: int,
    cin: int,
    cout: int,
    kh: int,
    kw: int,
    stride_h: int = 1,
    stride_w: int = 1,
    pad_h: int = 0,
    pad_w: int = 0
) -> bool {
    _require_device("conv2d_weights_grad_async")
    if n <= 0 or height <= 0 or width <= 0 or cin <= 0 or cout <= 0 or kh <= 0 or kw <= 0 {
        throw ValueError("conv2d_weights_grad_async dimensions must be positive")
    }
    if stride_h <= 0 or stride_w <= 0 {
        throw ValueError("conv2d_weights_grad_async stride must be positive")
    }
    return __prim__.gpu_conv2d_weights_grad(
        grad.handle_of(),
        input.handle_of(),
        output.handle_of(),
        n,
        height,
        width,
        cin,
        cout,
        kh,
        kw,
        stride_h,
        stride_w,
        pad_h,
        pad_w
    )
}

#: Fused two-layer ReLU MLP SGD step on device buffers.
#:
#: Updates `w1`/`b1`/`w2`/`b2` in place, adds the batch's total softmax
#: cross-entropy into `loss_acc`, and adds the correct-class count into
#: `correct_acc`.
pub fn mlp_sgd_step(
    x: Buffer,
    w1: Buffer,
    b1: Buffer,
    w2: Buffer,
    b2: Buffer,
    labels: Buffer,
    loss_acc: Buffer,
    batch: int,
    inputs: int,
    hidden: int,
    classes: int,
    learning_rate: float,
    correct_acc: Buffer? = null
) -> void {
    _require_device("mlp_sgd_step")
    if batch <= 0 or inputs <= 0 or hidden <= 0 or classes <= 0 {
        throw ValueError(
            f"mlp_sgd_step dimensions must be positive, got {batch}x{inputs}x{hidden}x{classes}"
        )
    }
    if learning_rate <= 0.0 {
        throw ValueError(f"mlp_sgd_step learning_rate must be positive, got {learning_rate}")
    }
    let correct_handle = if correct_acc is null { null } else { correct_acc.handle_of() }
    __prim__.gpu_mlp_sgd_step(
        x.handle_of(),
        w1.handle_of(),
        b1.handle_of(),
        w2.handle_of(),
        b2.handle_of(),
        labels.handle_of(),
        loss_acc.handle_of(),
        correct_handle,
        batch,
        inputs,
        hidden,
        classes,
        learning_rate
    )
}

#: Run every full batch of a two-layer fused SGD epoch without returning to
#: the VM between steps. Drops a ragged trailing batch. Returns samples used.
pub fn mlp_sgd_epoch(
    x: Buffer,
    w1: Buffer,
    b1: Buffer,
    w2: Buffer,
    b2: Buffer,
    labels: Buffer,
    loss_acc: Buffer,
    samples: int,
    batch: int,
    inputs: int,
    hidden: int,
    classes: int,
    learning_rate: float,
    flush_every: int = 4,
    correct_acc: Buffer? = null
) -> int {
    _require_device("mlp_sgd_epoch")
    if samples <= 0 or batch <= 0 or inputs <= 0 or hidden <= 0 or classes <= 0 or flush_every <= 0 {
        throw ValueError("mlp_sgd_epoch dimensions must be positive")
    }
    if learning_rate <= 0.0 {
        throw ValueError(f"mlp_sgd_epoch learning_rate must be positive, got {learning_rate}")
    }
    let correct_handle = if correct_acc is null { null } else { correct_acc.handle_of() }
    return __prim__.gpu_mlp_sgd_epoch(
        x.handle_of(),
        w1.handle_of(),
        b1.handle_of(),
        w2.handle_of(),
        b2.handle_of(),
        labels.handle_of(),
        loss_acc.handle_of(),
        correct_handle,
        samples,
        batch,
        inputs,
        hidden,
        classes,
        learning_rate,
        flush_every
    )
}

#: Fused two-layer ReLU MLP forward and backward into gradient buffers.
#:
#: Writes batch-mean parameter gradients into `g1`/`gb1`/`g2`/`gb2`, adds
#: the batch's total softmax cross-entropy into `loss_acc`, and optionally
#: adds the correct-class count into `correct_acc`.
pub fn mlp_bwd_step(
    x: Buffer,
    w1: Buffer,
    b1: Buffer,
    w2: Buffer,
    b2: Buffer,
    labels: Buffer,
    g1: Buffer,
    gb1: Buffer,
    g2: Buffer,
    gb2: Buffer,
    loss_acc: Buffer,
    batch: int,
    inputs: int,
    hidden: int,
    classes: int,
    correct_acc: Buffer? = null
) -> void {
    _require_device("mlp_bwd_step")
    if batch <= 0 or inputs <= 0 or hidden <= 0 or classes <= 0 {
        throw ValueError(
            f"mlp_bwd_step dimensions must be positive, got {batch}x{inputs}x{hidden}x{classes}"
        )
    }
    let correct_handle = if correct_acc is null { null } else { correct_acc.handle_of() }
    __prim__.gpu_mlp_bwd_step(
        x.handle_of(),
        w1.handle_of(),
        b1.handle_of(),
        w2.handle_of(),
        b2.handle_of(),
        labels.handle_of(),
        g1.handle_of(),
        gb1.handle_of(),
        g2.handle_of(),
        gb2.handle_of(),
        loss_acc.handle_of(),
        correct_handle,
        batch,
        inputs,
        hidden,
        classes
    )
}

#: Fused three-hidden-layer ReLU MLP SGD step on device buffers.
pub fn mlp3_sgd_step(
    x: Buffer,
    w1: Buffer,
    b1: Buffer,
    w2: Buffer,
    b2: Buffer,
    w3: Buffer,
    b3: Buffer,
    w4: Buffer,
    b4: Buffer,
    labels: Buffer,
    loss_acc: Buffer,
    batch: int,
    inputs: int,
    hidden: int,
    classes: int,
    learning_rate: float,
    correct_acc: Buffer? = null,
    hidden2: int = 0,
    hidden3: int = 0
) -> void {
    _require_device("mlp3_sgd_step")
    let h2 = if hidden2 == 0 { hidden } else { hidden2 }
    let h3 = if hidden3 == 0 { hidden } else { hidden3 }
    if batch <= 0 or inputs <= 0 or hidden <= 0 or h2 <= 0 or h3 <= 0 or classes <= 0 {
        throw ValueError(
            f"mlp3_sgd_step dimensions must be positive, got {batch}x{inputs}x{hidden}/{h2}/{h3}x{classes}"
        )
    }
    if learning_rate <= 0.0 {
        throw ValueError(f"mlp3_sgd_step learning_rate must be positive, got {learning_rate}")
    }
    let correct_handle = if correct_acc is null { null } else { correct_acc.handle_of() }
    __prim__.gpu_mlp3_sgd_step(
        x.handle_of(),
        w1.handle_of(),
        b1.handle_of(),
        w2.handle_of(),
        b2.handle_of(),
        w3.handle_of(),
        b3.handle_of(),
        w4.handle_of(),
        b4.handle_of(),
        labels.handle_of(),
        loss_acc.handle_of(),
        correct_handle,
        batch,
        inputs,
        hidden,
        h2,
        h3,
        classes,
        learning_rate
    )
}

#: Run every full batch of a three-hidden-layer fused SGD epoch in native code.
pub fn mlp3_sgd_epoch(
    x: Buffer,
    w1: Buffer,
    b1: Buffer,
    w2: Buffer,
    b2: Buffer,
    w3: Buffer,
    b3: Buffer,
    w4: Buffer,
    b4: Buffer,
    labels: Buffer,
    loss_acc: Buffer,
    samples: int,
    batch: int,
    inputs: int,
    hidden: int,
    classes: int,
    learning_rate: float,
    flush_every: int = 4,
    correct_acc: Buffer? = null,
    hidden2: int = 0,
    hidden3: int = 0
) -> int {
    _require_device("mlp3_sgd_epoch")
    let h2 = if hidden2 == 0 { hidden } else { hidden2 }
    let h3 = if hidden3 == 0 { hidden } else { hidden3 }
    if samples <= 0 or
        batch <= 0 or
        inputs <= 0 or
        hidden <= 0 or
        h2 <= 0 or
        h3 <= 0 or
        classes <= 0 or
        flush_every <= 0 {
        throw ValueError("mlp3_sgd_epoch dimensions must be positive")
    }
    if learning_rate <= 0.0 {
        throw ValueError(f"mlp3_sgd_epoch learning_rate must be positive, got {learning_rate}")
    }
    let correct_handle = if correct_acc is null { null } else { correct_acc.handle_of() }
    return __prim__.gpu_mlp3_sgd_epoch(
        x.handle_of(),
        w1.handle_of(),
        b1.handle_of(),
        w2.handle_of(),
        b2.handle_of(),
        w3.handle_of(),
        b3.handle_of(),
        w4.handle_of(),
        b4.handle_of(),
        labels.handle_of(),
        loss_acc.handle_of(),
        correct_handle,
        samples,
        batch,
        inputs,
        hidden,
        h2,
        h3,
        classes,
        learning_rate,
        flush_every
    )
}

#: Fused three-hidden-layer ReLU MLP forward and backward into gradient buffers.
pub fn mlp3_bwd_step(
    x: Buffer,
    w1: Buffer,
    b1: Buffer,
    w2: Buffer,
    b2: Buffer,
    w3: Buffer,
    b3: Buffer,
    w4: Buffer,
    b4: Buffer,
    labels: Buffer,
    g1: Buffer,
    gb1: Buffer,
    g2: Buffer,
    gb2: Buffer,
    g3: Buffer,
    gb3: Buffer,
    g4: Buffer,
    gb4: Buffer,
    loss_acc: Buffer,
    batch: int,
    inputs: int,
    hidden: int,
    classes: int,
    correct_acc: Buffer? = null,
    hidden2: int = 0,
    hidden3: int = 0
) -> void {
    _require_device("mlp3_bwd_step")
    let h2 = if hidden2 == 0 { hidden } else { hidden2 }
    let h3 = if hidden3 == 0 { hidden } else { hidden3 }
    if batch <= 0 or inputs <= 0 or hidden <= 0 or h2 <= 0 or h3 <= 0 or classes <= 0 {
        throw ValueError(
            f"mlp3_bwd_step dimensions must be positive, got {batch}x{inputs}x{hidden}/{h2}/{h3}x{classes}"
        )
    }
    let correct_handle = if correct_acc is null { null } else { correct_acc.handle_of() }
    __prim__.gpu_mlp3_bwd_step(
        x.handle_of(),
        w1.handle_of(),
        b1.handle_of(),
        w2.handle_of(),
        b2.handle_of(),
        w3.handle_of(),
        b3.handle_of(),
        w4.handle_of(),
        b4.handle_of(),
        labels.handle_of(),
        g1.handle_of(),
        gb1.handle_of(),
        g2.handle_of(),
        gb2.handle_of(),
        g3.handle_of(),
        gb3.handle_of(),
        g4.handle_of(),
        gb4.handle_of(),
        loss_acc.handle_of(),
        correct_handle,
        batch,
        inputs,
        hidden,
        h2,
        h3,
        classes
    )
}

#: True when every class-index label in the buffer is an integer in `[0, classes)`.
pub fn labels_valid(labels: Buffer, count: int, classes: int) -> bool {
    _require_device("labels_valid")
    if count < 0 or classes <= 0 {
        throw ValueError(f"labels_valid needs a non-negative count and positive classes")
    }
    return __prim__.gpu_labels_valid(labels.handle_of(), count, classes)
}
