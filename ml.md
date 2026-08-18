# Machine learning in Jaithon

Jaithon is a general language whose checker, JIT, and GPU runtime are built so
training is ordinary `.jai`. `jaitensor` is a package on that surface. This
file is the contract those pieces have to keep.

## Shape types

An integer literal is a type. The parser produces `TypeKind.Const`; the checker
interns it as `TyKind.Int` whose `name` is the decimal digits.

```jai
Tensor[float, 2, 3]
```

is a generic instantiation. User class generics are otherwise erased to the
bare class. The checker keeps arguments when the name is `Tensor` or ends with
`.Tensor`, so `jaitensor.Tensor` and an import alias both work.

Assignability (`lib/jaithon/compile/check/assign.jai`):

- Bare `Tensor` (no args) and a shaped `Tensor[...]` assign both ways.
- Two shaped tensors assign when they have the same argument count and each
  argument is the same type (dtype and every dim).
- A rank mismatch or a dim mismatch is rejected.

The first argument is the dtype. Storage is float32, so write `float`. A
conflicting annotation is a type error at the assignment or return that
carries it; it is not a runtime shape check. Runtime `Tensor.shape` can still
disagree with the annotation if the value arrived through `any`.

Integer type arguments are not symbolic. `Tensor[float, N, N]` is not a
language form unless `N` is a declared type name.

## Decorators

`@name` may precede a function or method. The parser stores a `list[str]` on
the declaration. Emit maps names to function flags:

| Decorator | Flag | Value |
| --- | --- | --- |
| `trace` | `FN_TRACE` | `1 << 9` (512) |
| `gpu_kernel` | `FN_GPU_KERNEL` | `1 << 10` (1024) |
| `mps_graph` | `FN_GPU_KERNEL` | same |

`@gpu_kernel` and `@mps_graph` are one flag. `std.trace.is_traced` and
`std.gpu.is_gpu_kernel` test them. Unknown names are stored and currently
ignored; they should become an `E06xx` error so a misspelling cannot silently
do nothing.

`.jaic` serializes both flags (`JAIC_KNOWN_FN_FLAGS`). Do not bump
`JAI_COMPILER_VERSION` in C before `make reseed`.

## Tracing

`@trace` does two things.

1. **JIT threshold.** Ordinary functions compile after 64 entries
   (`JAI_JIT_THRESHOLD`). Traced functions compile after 8
   (`JAI_JIT_TRACE_THRESHOLD`). The arm64 JIT takes up to 8 positional
   arguments (`x0`–`x7`). Defaults, `*args`, and `**kwargs` keep a function
   off that path; a traced kernel that should compile must not use them.

2. **Graph session.** Entering a traced function opens a session
   (`src/vm/trace/`). `std.trace.record_op(name, shape)` appends one op.
   A later invocation that records the same names and shapes in the same
   order sets `std.trace.is_replay()`. A mismatch starts a new graph.
   `is_replay()` is a flag. It does not skip interpreter work; a library
   skips work by testing it (jaitensor's static / fused path).

The outermost traced function owns the session (`gDepth` in `trace.c`).
Nested `@trace` callees record into that session. A tail call from a traced
function must not reuse the frame: that would skip `jaiTraceLeave` while
`return trace.is_replay()` still needs the session.

## GPU

`std.gpu` is the language GPU. Apple builds use Metal (`src/native/apple/gpu.m`);
other platforms stub the primitives.

- `Buffer` is float32 device memory. It is not garbage collected. `free` it.
- `set_device(i)` must run before the first buffer or kernel. `ensureDevice`
  latches the choice; later calls do not move existing allocations.
- `device_count()` is the number of Metal devices. `set_device` picks which
  GPU later allocations use. There is no built-in all-reduce.
- `Kernel.compile(source, entry)` compiles MSL and interns by source+entry.
  If the interned kernel was `free`d, compile again. Do not dispatch a
  `freed` interned handle.
- Mixed precision (`set_mixed_precision(true)`): GEMM and fused MLP graphs
  compute in float16 and store float32. There is no bf16 path and no
  float16 `Buffer`.
- Tiny MPSNDArray allocations abort the process. Native conv must refuse
  buffers under 512 bytes and let Jaithon fall back (im2col + GEMM). Never
  let a small conv hit MPS.

`@gpu_kernel` does not lower the Jaithon body to MSL. The function is a
marker; `Kernel.compile` still takes a Metal string. Lowering a Jaithon
subset to MSL or MPSGraph is the next language step, not a hidden behaviour
of the flag.

## jaitensor

Workspace package, written in Jaithon, on `std.gpu`.

Tensors are row-major float32, GPU-resident, explicit `free`. Autograd is a
library (`GradNode`), not a language form. `Sequential` stacks `Layer`.
Images are NHWC. Conv uses MPS when the buffers are large enough, otherwise
im2col.

`compile(..., mixed_precision: true)` sets the process AMP flag. The static
(no-GradNode) step runs when every layer `supports_static()`, not only when
`compile` was given a matching `batch_size`.

`DataLoader` shuffles by building a permutation and `gather_rows` on the GPU.
The VM is single-threaded; Jaithon closures cannot run on `std.thread`
workers. Overlap is the Metal queue, not a host prefetch thread.

`MultiHeadAttention` is Q/K/V/Wo plus per-head scaled-dot-product attention.
`heads > 1` splits the last axis into contiguous `[head_dim]` blocks, scores
each head at `1 / sqrt(head_dim)`, and concatenates the context. Sequences of
length 16+ with `head_dim <= 64` use a tiled flash kernel (no `seq x seq`
buffer). Other shapes use the MMA/MPS GEMM path.

`gather_rows` records a scatter-add into the source so `Embedding` can train.
`group_norm` is NHWC, one mean/variance per sample per group. `tril` zeros the
strict upper triangle of a rank-two tensor. `binary_cross_entropy_with_logits`
is the stable logits form; probability BCE still exists.

Tensor ops also include `abs`, `neg`, `sqrt`, `rsqrt`, `square`, `gelu`,
`silu`, `leaky_relu`, `clamp`, `sum`/`mean`, `cat`, `max_pool2d`/`avg_pool2d`,
`dropout`, and `mse`. `sum`/`mean`/`mse` record autograd without a host
round-trip. `reshape` and `slice_flat` are views (no device copy).
`scale` passes the multiplier as float32 bits, not a 1-element buffer.
`Buffer.fill_uniform` is a device kernel, so dropout noise stays on the GPU.

## What this is aimed at

The language should stay the place models are written, compiled, and run.

- Shapes and dtypes in the type (`Tensor[f16, 32, 128]`), with storage that
  matches, not float32 with a comment.
- `@gpu_kernel` bodies that are Jaithon, lowered to MSL or an MPSGraph, so a
  fused op does not require an ObjC file.
- Symbolic dims (`Tensor[float, N, N]`) once const generics exist for names,
  not only literals.
- Unknown decorator names as errors.
- Data-parallel training as a library on top of `set_device`, when more than
  one GPU is actually used.
