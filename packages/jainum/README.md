# jainum

N-dimensional arrays with numpy's API, on the GPU.

```jai
import jainum as nn

let a = nn.NDArray([3, 4])
defer { a.free() }
a.upload([float(index) for index in 0..12])

let block = a[0..2, 1..3]
print(block.to_nested())
print((a * 2.0 + 1.0).to_nested())
```

Names, argument order, and constants follow numpy, in snake case: `zeros_like`
for `zeros_like`, `NDArray` for `ndarray`, `FLOAT32` for `np.float32`. A
program written against numpy translates line for line, with one exception that
is worth reading before anything else.

## A GPU is not optional

Storage is one float32 Metal buffer per allocation and every operation below
is a kernel, so constructing an `NDArray` on a machine without a device raises
rather than falling back to the host. `nn.is_available()` is the question to
ask first; `nn.device_name()`, `nn.device_count()` and `nn.set_device(index)`
are the rest of that surface, and `set_device` has to be called before the
first array exists.

## Namespaces

Three parts of the library are namespaces rather than flat names, exactly as
they are in numpy — `nn.linalg.solve`, `nn.random.default_rng`, `nn.fft.rfft`.
Everything else is flat: `nn.zeros`, `nn.argmax`, `nn.concatenate`. Each
module is also importable on its own, so `from jainum.reduce import argmax`
keeps working; the facade in `mod.jai` is for callers who want the package
rather than one file.

## Indexing does not use slice syntax

`a[1:3]` does not work, and it does not fail in a way that suggests an
alternative. Slice syntax reaches `OP_GET_SLICE` in the VM, which throws for
anything that is not a list, a str or a tuple; the checker's advice that "a
user type needs `__getitem__`" is not backed by the runtime. So an index is:

| Written | Means |
| --- | --- |
| `a[2]` | element 2 of the leading axis, that axis dropped |
| `a[1, 3]` | one element of a 2-d array, as a 0-d array |
| `a[0..3]` | a `range` — the nicest spelling when the bounds are known and positive |
| `a[0..=3]` | inclusive, and the array can tell the two apart |
| `a[sl(null, null, -1)]` | a `Slice` — the open ends and negative steps a `range` cannot write |
| `a[nn.ELLIPSIS, 0]` | as many full axes as the rest of the key leaves over |
| `a[nn.NEWAXIS]` | a new axis of length one |
| `a[mask]` | a full-shape `BOOL` array, which compacts into a 1-d result |
| `a[keys]` | a 1-d integer array, which gathers from a 1-d array |

Everything but the last two returns a view. Commas make a tuple, which is what
the VM hands over, so `a[1, 0..3, nn.NEWAXIS]` is one key and not three.

An int index drops its axis and leaves a 0-d array rather than a float:
`a[1, 2].item()` is the float, and keeping the intermediate on the device is
what stops every index expression from being a synchronise.

There is also deliberately no `__eq__`. An elementwise `==` returning an array
would break `assert_eq`, every `is not null` guard, and membership in a dict or
a set, all of which need a bool. `nn.equal` and `nn.array_equal` are the
spellings that work.

## The dtype model

Storage is float32, full stop. A dtype is a semantic tag over those bits that
governs rounding, saturation and result type — jaicv's `Mat` depth model
applied to n dimensions, for the same reason and with the same consequences:

- `INT32` and `INT64` round to nearest even and clamp on the way in, and are
  exact to 2^24 (`nn.MAX_EXACT_INT`). Past that a count silently stops being a
  count. This is documented rather than hidden.
- `FLOAT64` is a source-compatible alias for float32, not a wider element.
- `BOOL` is 0.0 and 1.0, nothing else.
- `COMPLEX64` is interleaved real and imaginary float32, two float slots per
  element — jaicv's `CV_32FC2`. `.real()` and `.imag()` are views of the same
  slots, because a stride counts float slots and a complex stride therefore
  already steps both lanes.
- Promotion is the lattice `BOOL < INT32 < INT64 < FLOAT32 < FLOAT64 <
  COMPLEX64`. numpy's value-based promotion is not reproduced: with float32
  underneath, the only distinctions this can honour are integer-versus-float
  and real-versus-complex.

Complex is carried by `fft` and by `add`/`subtract`/`multiply`/`divide`/
`absolute`/`angle`. Every other entry point calls `require_real` at the door
and raises rather than quietly reading one lane of two.

## Views and strides

An `NDArray` is a window onto one float32 device buffer: an element offset and
one stride per axis, counted in float slots. Strides may be negative, which is
what lets a flip and a negative-step slice be views. A slice, a transpose, a
broadcast, a diagonal and `real`/`imag` all alias their parent, and a write
through one is visible in the other. `copy()` and `contiguous()` are where a
materialisation happens, and `contiguous()` hands back the array itself when it
already is one.

`free()` releases the allocation. A view does not own its storage, so freeing a
view leaves the parent alone.

## Modules

The package is layered, and the layering is enforced: a module imports strictly
downward and an import cycle is a compile error.

| Layer | Module | What it covers |
| --- | --- | --- |
| 0 | `array` | dtypes, strides, broadcasting, views, indexing, the five operators, and the Metal prelude every other module compiles against |
| 1 | `create` | `zeros`, `ones`, `full`, `arange`, `linspace`, `eye`, `from_list` |
| 1 | `ufunc` | the named elementwise functions, comparisons, and `where` |
| 1 | `reduce` | `sum`, `prod`, `min`, `max`, `argmin`, `argmax`, `all`, `any`, along an axis or over everything |
| 1 | `manipulate` | `reshape`, `transpose`, `concatenate`, `stack`, `split`, `flip`, `pad` |
| 2 | `index` | `take`, `put`, `nonzero`, `compress`, `choose`, `diag` |
| 2 | `sorting` | `sort`, `argsort`, `partition`, `searchsorted`, `unique` |
| 2 | `matmul` | `dot`, `matmul`, `tensordot`, `outer`, `trace` |
| 3 | `linalg` | `solve`, `inv`, `det`, `qr`, `svd`, `eig`, `cholesky`, `lstsq`, `norm` |
| 3 | `random` | a counter-based generator, uniforms, normals, integers, `shuffle`, `choice` |
| 3 | `fft` | `fft`, `ifft`, `fft2`, `rfft`, `fftfreq` |
| 3 | `stats` | `mean`, `var`, `std`, `median`, `percentile`, `histogram`, `corrcoef` |
| 4 | `mod` | the package surface, re-exporting the rest |

`array` owns the four arithmetic kernels because a class method cannot live in
another file; `ufunc` re-exports them under numpy's names rather than compiling
the arithmetic a second time.

## Writing a kernel here

`PRELUDE` holds the shared index math and the universal passes — strided copy
and cast, fill, gather, scatter, and the exclusive scan every compaction is
built on. A module writes `let SOURCE = PRELUDE + """ ... """` and asks
`kernel_of(SOURCE, entry)` for a pipeline.

Two rules the whole package follows:

- **Every bulk op has two dispatch paths.** A contiguous same-shape fast path
  taking a flat count, and a general path through `layout_buffer`. The
  divmod-per-axis walk is the real cost of the general one, and most calls take
  the other.
- **A host loop over a million elements costs seconds in this language.** Any
  elementwise, reduction, gather, scatter, sort, scan or transform pass is a
  Metal kernel. Where a host implementation is genuinely right — a small dense
  factorization, a sort below its threshold — the comment says why.

Shapes and strides travel in a small float buffer built fresh per dispatch by
`layout_buffer` and freed after encoding, because `Kernel.dispatch` binds its
trailing scalars as unsigned and a stride is signed. Those buffers are
deliberately not pooled: a recycled GPU buffer needs a host-write fence, and
this repo has been bitten by that before. A float scalar that carries no layout
travels as `constant uint&` through `f32_bits`.

## What is not here

Named so that nobody has to find out by trying:

- **File formats**: `.npy` and `.npz` save and load, `savetxt` and `loadtxt`.
- **Masked arrays** (`numpy.ma`) and **structured or record dtypes**.
- **`numpy.polynomial`**.
- **General n-operand `einsum`** with contraction ordering. The two-operand
  contractions `matmul` covers are here; the optimiser is not.
- **`datetime64` and `timedelta64`**.

Each of those is a plan of its own rather than a stub.

## Tests

```bash
JAITHON_PATH=lib ./jaithon test packages/jainum/tests/
```

One test file per module, named after it. The layout helpers are pure and run
without a device; anything that touches storage returns early when
`nn.is_available()` is false.

`test_reachable.jai` is the exception: it calls every name the facade exports,
once, through `import jainum as nn`. A module test asks what a function
computes, and cannot see a name that was renamed in one file and not the
other, or an export whose signature nobody has called since it grew an
argument. That is the failure this file exists to catch, and it is the one
jaicv and jaitensor keep a file of the same name for.
