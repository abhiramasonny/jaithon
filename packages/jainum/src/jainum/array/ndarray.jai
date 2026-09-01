#: The array itself: its layout, its views, its indexing, and the elementwise
#: work that has to see all three.
from std.core import Printable
from std.gpu import Buffer
from .bits import f32_bits
from .dtype import BOOL, CAST_BOOL, CAST_NONE, CAST_ROUND, COMPLEX64, FLOAT32
from .dtype import GROUP, INT32, OP_ADD, OP_DIV, OP_MUL, OP_SUB
from .dtype import dtype_is_bool, dtype_is_complex, dtype_is_integer, dtype_lanes
from .dtype import dtype_max, dtype_min, dtype_name, promote
from .metal import SOURCE, kernel_of, require_device
from .shape import broadcast_shapes, broadcast_strides, contiguous_strides
from .shape import element_count, is_contiguous, normalize_axis
from .slicing import ELLIPSIS, IndexToken, Slice, _sign

#: Reject a complex array at the door of a kernel that has no lanes.
#:
#: Complex is supported by `fft` and by the arithmetic here; every other
#: module calls this first and is then free to assume one float per element.
pub fn require_real(a: NDArray, op: str) -> void {
    if dtype_is_complex(a.dtype) {
        throw TypeError(f"{op} does not accept a complex array; take .real() or .imag() first")
    }
}

#: Two views of one common shape, neither of them a copy.
pub fn broadcast_pair(a: NDArray, b: NDArray) -> (NDArray, NDArray) {
    let shape = broadcast_shapes(a.shape, b.shape)
    return (a.broadcast_to(shape), b.broadcast_to(shape))
}

#: A view of the `offset`-th diagonal of the plane `axis1` x `axis2`.
#:
#: The diagonal is an axis whose stride is the sum of the two it replaces,
#: which is why this is a view and not a gather. The other axes keep their
#: order and the diagonal is appended last, as numpy's `diagonal` does.
pub fn diagonal_view(a: NDArray, offset: int = 0, axis1: int = 0, axis2: int = 1) -> NDArray {
    let rank = a.ndim()
    let first = normalize_axis(axis1, rank, "axis1")
    let second = normalize_axis(axis2, rank, "axis2")
    if first == second { throw ValueError(f"axis1 and axis2 are both {first}") }
    var start = a.offset
    var length = 0
    if offset >= 0 {
        length = max(0, min(a.shape[first], a.shape[second] - offset))
        start += offset * a.strides[second]
    } else {
        length = max(0, min(a.shape[first] + offset, a.shape[second]))
        start += -offset * a.strides[first]
    }
    var shape: list[int] = []
    var strides: list[int] = []
    for axis in 0..rank {
        if axis == first or axis == second { continue }
        shape.push(a.shape[axis])
        strides.push(a.strides[axis])
    }
    shape.push(length)
    strides.push(a.strides[first] + a.strides[second])
    return a.view(shape, strides, start, a.dtype)
}

#: The metadata block a strided dispatch reads: rank, iteration shape, then
#: each operand's offset and strides. Operand 0 is the destination.
#:
#: Every array must already have the iteration rank — call `broadcast_to`
#: first. Allocated fresh for each dispatch and freed straight after encoding:
#: an in-flight command buffer retains it, and a recycled pool of these would
#: need a host-write fence that this repo has been bitten by before.
pub fn layout_buffer(arrays: list[NDArray]) -> Buffer {
    if arrays.len() == 0 { throw ValueError("a layout needs at least one operand") }
    let rank = arrays[0].ndim()
    var values = [float(rank)]
    for extent in arrays[0].shape { values.push(float(extent)) }
    for operand in arrays {
        if operand.ndim() != rank {
            let got = operand.ndim()
            throw ValueError(f"a layout of rank {rank} got rank {got}; broadcast_to it first")
        }
        values.push(float(operand.offset))
        for stride in operand.strides { values.push(float(stride)) }
    }
    let meta = Buffer(values.len())
    meta.upload(values)
    return meta
}

#: A dense n-dimensional array of float32 slots, with a dtype that says how to
#: read them.
pub class NDArray: Printable {
    #: Elements along each axis. Empty for a 0-d array.
    pub let shape: list[int]
    #: Float slots between neighbouring elements along each axis. May be
    #: negative, and is zero on a broadcast axis.
    pub let strides: list[int]
    #: Float slot of the first element inside `storage`.
    pub let offset: int
    #: Semantic tag: rounding, saturation and result type. Not a width.
    pub let dtype: int
    #: The allocation this array is a window onto, shared with every other
    #: window of it.
    pub let storage: Buffer
    #: The array that owns `storage`, or null when this one does.
    pub let base: NDArray?
    var freed: bool

    #: Allocate `shape` elements of `dtype`, uninitialised.
    #:
    #: Pass `storage`, `strides` and `offset` to build a window onto an
    #: existing allocation instead; `view` and `__getitem__` are the callers
    #: that do.
    pub fn init(
        self,
        shape: list[int],
        dtype: int = FLOAT32,
        storage: Buffer? = null,
        strides: list[int]? = null,
        offset: int = 0,
        base: NDArray? = null
    ) {
        if dtype < BOOL or dtype > COMPLEX64 { throw ValueError(f"unknown dtype {dtype}") }
        let lanes = dtype_lanes(dtype)
        self.shape = [extent for extent in shape]
        self.strides = strides ?? contiguous_strides(shape, lanes)
        if self.strides.len() != self.shape.len() {
            throw ValueError(f"{self.strides.len()} strides for {self.shape.len()} axes")
        }
        self.dtype = dtype
        self.offset = offset
        self.base = base
        self.freed = false
        if storage is not null {
            self.storage = storage
            return
        }
        require_device("allocate an NDArray")
        self.storage = Buffer(max(1, element_count(self.shape) * lanes))
    }

    #: A window onto the same allocation with a shape of its own.
    #:
    #: `dtype` of -1 keeps this array's. The window points at whichever array
    #: owns the storage, so a view of a view still frees correctly.
    pub fn view(
        self,
        shape: list[int],
        strides: list[int],
        offset: int,
        dtype: int = -1
    ) -> NDArray {
        let tag = if dtype < 0 { self.dtype } else { dtype }
        return NDArray(shape, tag, self.storage, strides, offset, self.base ?? self)
    }

    #: Elements the array holds.
    pub fn size(self) -> int { return element_count(self.shape) }

    #: Number of axes.
    pub fn ndim(self) -> int { return self.shape.len() }

    #: Float slots one element occupies.
    pub fn lanes(self) -> int { return dtype_lanes(self.dtype) }

    #: Bytes one element occupies, as numpy's `itemsize` reports it. Storage is
    #: float32, so this is 4 per lane whatever the dtype claims to be.
    pub fn itemsize(self) -> int { return 4 * self.lanes() }

    #: The device allocation. Shared with every other window of it.
    pub fn buffer(self) -> Buffer { return self.storage }

    #: Whether the elements run in row-major order with nothing between them.
    pub fn is_contiguous(self) -> bool {
        return is_contiguous(self.shape, self.strides, self.lanes())
    }

    #: A view of the real parts of a complex array.
    #:
    #: No stride arithmetic: strides are counted in float slots, so a complex
    #: array's strides already step over both lanes and the real view keeps
    #: them unchanged. That is most of why they are counted that way.
    pub fn real(self) -> NDArray {
        if not dtype_is_complex(self.dtype) { return self }
        return self.view(self.shape, self.strides, self.offset, FLOAT32)
    }

    #: A view of the imaginary parts of a complex array, or zeros for a real
    #: one — which, being an allocation, is not a view.
    pub fn imag(self) -> NDArray {
        if not dtype_is_complex(self.dtype) {
            let zeros = NDArray(self.shape, self.dtype)
            fill(zeros, 0.0)
            return zeros
        }
        return self.view(self.shape, self.strides, self.offset + 1, FLOAT32)
    }

    #: A view reading this array as if it had `shape`, stretching every
    #: length-one and every missing leading axis.
    pub fn broadcast_to(self, shape: list[int]) -> NDArray {
        if shape == self.shape { return self }
        return self.view(shape, broadcast_strides(self.shape, self.strides, shape), self.offset)
    }

    #: The one value a one-element array holds.
    pub fn item(self) -> float {
        if self.size() != 1 { throw ValueError(f"item() needs one element, not {self.size()}") }
        if dtype_is_complex(self.dtype) {
            throw TypeError("item() has no complex reading; take .real() or .imag()")
        }
        let (dense, owned) = _dense(self)
        defer { if owned { dense.free() } }
        return dense.storage.download(dense.offset, 1)[0]
    }

    #: Every float slot in element order, materialising first when strided.
    #:
    #: A complex array hands back interleaved re/im, twice as many floats as
    #: elements.
    pub fn to_list(self) -> list[float] {
        let (dense, owned) = _dense(self)
        defer { if owned { dense.free() } }
        let count = dense.size() * dense.lanes()
        if count == 0 { return [] }
        return dense.storage.download(dense.offset, count)
    }

    #: The values as nested lists, one level per axis.
    pub fn to_nested(self) -> any {
        return _nest(self.to_list(), self.shape, 0, 0, self.lanes())[0]
    }

    #: Copy `values` in, in element order. The array must be contiguous and the
    #: count must match exactly, lanes included.
    pub fn upload(self, values: list[float]) -> void {
        self._check_live()
        if not self.is_contiguous() {
            throw ValueError("upload needs a contiguous array; call contiguous() first")
        }
        let wanted = self.size() * self.lanes()
        if values.len() != wanted {
            throw ValueError(f"{values.len()} values cannot fill {wanted} slots")
        }
        if wanted == 0 { return }
        self.storage.upload(values, self.offset)
    }

    #: A fresh contiguous array holding the same values.
    pub fn copy(self) -> NDArray {
        let out = NDArray(self.shape, self.dtype)
        copy_into(out, self)
        return out
    }

    #: This array if it is already dense at offset zero, a copy otherwise.
    #:
    #: The offset matters: a caller that reached for this wants to hand
    #: `storage` straight to a kernel that indexes from slot zero.
    pub fn contiguous(self) -> NDArray {
        if self.offset == 0 and self.is_contiguous() { return self }
        return self.copy()
    }

    #: A copy read as `dtype`, rounding and saturating on the way if the target
    #: is an integer dtype, and flattening to 0 and 1 if it is `BOOL`.
    pub fn astype(self, dtype: int) -> NDArray {
        let out = NDArray(self.shape, dtype)
        copy_into(out, self)
        return out
    }

    #: Release the device memory. A view does not own its allocation, so
    #: releasing one leaves the parent alone.
    pub fn free(self) -> void {
        if self.freed { return }
        self.freed = true
        if self.base is null { self.storage.free() }
    }

    #: Whether `free` has been called on this array.
    pub fn is_freed(self) -> bool { return self.freed }

    fn _check_live(self) -> void {
        if self.freed { throw RuntimeError("this NDArray has already been freed") }
    }

    fn __len__(self) -> int {
        if self.ndim() == 0 { throw TypeError("a 0-d array has no length") }
        return self.shape[0]
    }

    #: Shape, dtype, and as much of the values as is cheap to look at.
    #:
    #: A big array shows its leading corner and says so. Printing is a debug
    #: aid and a debug aid that downloads a hundred megabytes is a trap; the
    #: corner costs one small strided copy.
    pub fn to_str(self) -> str {
        if self.freed { return f"NDArray(freed, dtype={dtype_name(self.dtype)})" }
        let head = f"dtype={dtype_name(self.dtype)}"
        if self.size() <= _PRINT_LIMIT {
            return f"array({_render(self.to_nested())}, {head})"
        }
        var corner = self
        for axis in 0..self.ndim() {
            if corner.shape[axis] > _PRINT_EDGE {
                corner = _slice_axis(corner, axis, 0, _PRINT_EDGE, 1)
            }
        }
        return f"array(shape={self.shape}, {head}, corner={_render(corner.to_nested())})"
    }

    fn __str__(self) -> str { return self.to_str() }

    #: A view selected by a basic key, or a packed copy for the two flat fancy
    #: keys.
    #:
    #: Basic keys are an int, a `Slice`, a `range`, `NEWAXIS`, `ELLIPSIS`, or
    #: several of those comma-separated -- which arrives as one tuple -- and
    #: always produce a view. An int drops its axis, so
    #: `a[0]` on a 1-d array is a 0-d array rather than a float — `.item()`
    #: turns it into one, and keeping it an array is what stops every index
    #: expression from being a device sync.
    #:
    #: Fancy keys are an `NDArray` of `BOOL` shaped exactly like this one,
    #: which selects into a 1-d array, and a 1-d integer `NDArray`, which
    #: gathers from a 1-d array. `take` in `index` covers the axis-wise cases.
    fn __getitem__(self, key: any) -> any {
        self._check_live()
        if key is NDArray { return _fancy_get(self, key) }
        return _basic_view(self, if key is tuple { [item for item in key] } else { [key] })
    }

    #: Write through a key.
    #:
    #: A basic key writes through the view it selects, broadcasting an array
    #: value and filling a scalar one. A `BOOL` mask with a scalar value is a
    #: masked fill, and a 1-d integer key scatters.
    fn __setitem__(self, key: any, value: any) -> void {
        self._check_live()
        if key is NDArray { return _fancy_set(self, key, value) }
        let target = _basic_view(self, if key is tuple { [item for item in key] } else { [key] })
        if value is NDArray { copy_into(target, value) } else { fill(target, _as_float(value)) }
    }

    #: Elementwise sum with an array or a scalar.
    #:
    #: There is deliberately no `__eq__`. An elementwise `==` returning an
    #: array breaks `assert_eq`, every `is not null` guard that follows it, and
    #: membership in a dict or a set — all of which need a bool. `equal` and
    #: `array_equal` in `ufunc` are the spellings that work; do not "fix" this.
    #:
    #: A scalar only reads on the right: the VM has no reflected dunder, so
    #: `2.0 - a` raises where `a - 2.0` works. `subtract` in `ufunc` takes both
    #: orders.
    fn __add__(self, other: any) -> NDArray { return binary_arith(self, other, OP_ADD) }

    fn __sub__(self, other: any) -> NDArray { return binary_arith(self, other, OP_SUB) }

    fn __mul__(self, other: any) -> NDArray { return binary_arith(self, other, OP_MUL) }

    fn __div__(self, other: any) -> NDArray { return binary_arith(self, other, OP_DIV) }

    fn __neg__(self) -> NDArray { return _binary_scalar(self, -1.0, self.dtype, OP_MUL) }
}

const _PRINT_LIMIT = 256
const _PRINT_EDGE = 3

fn _render(value: any) -> str {
    if value is list {
        return "[" + ", ".join([_render(item) for item in value]) + "]"
    }
    return f"{value}"
}

fn _nest(values: list[float], shape: list[int], axis: int, at: int, lanes: int) -> (any, int) {
    if axis == shape.len() {
        if lanes == 2 { return (f"{values[at]}{_signed(values[at + 1])}j", at + 2) }
        return (values[at], at + 1)
    }
    var out: list[any] = []
    var cursor = at
    for _index in 0..shape[axis] {
        let (item, next) = _nest(values, shape, axis + 1, cursor, lanes)
        out.push(item)
        cursor = next
    }
    return (out, cursor)
}

fn _signed(value: float) -> str { return if value < 0.0 { f"{value}" } else { f"+{value}" } }

fn _as_float(value: any) -> float {
    if value is float { return value }
    if value is int { return float(value) }
    if value is bool { return if value { 1.0 } else { 0.0 } }
    throw TypeError(f"expected a number, got {type_of(value)}")
}

#: A view of one axis narrowed to `count` elements from `first`, stepping.
fn _slice_axis(a: NDArray, axis: int, first: int, count: int, step: int) -> NDArray {
    var shape = [extent for extent in a.shape]
    var strides = [stride for stride in a.strides]
    shape[axis] = count
    strides[axis] = a.strides[axis] * step
    return a.view(shape, strides, a.offset + first * a.strides[axis])
}

#: A `range` as a `Slice`.
#:
#: `range` exposes start, stop and step but not whether it was written `..` or
#: `..=`, and the two disagree by one. Its length does know, so an exclusive
#: length that differs from the real one is exactly the inclusive case.
fn _slice_of_range(key: range) -> Slice {
    let start = key.start()
    let step = key.step()
    var stop = key.stop()
    let span = stop - start
    var exclusive = 0
    if (span > 0) == (step > 0) and span != 0 { exclusive = (span + step - _sign(step)) // step }
    if key.len() != exclusive { stop += _sign(step) }
    return Slice(start, stop, step)
}

fn _basic_view(a: NDArray, items: list[any]) -> NDArray {
    var expanded: list[any] = []
    var ellipses = 0
    var consumed = 0
    for item in items {
        if item is IndexToken and item.name == ELLIPSIS.name {
            ellipses += 1
        } elif not item is IndexToken {
            consumed += 1
        }
    }
    if ellipses > 1 { throw ValueError("a key may hold one ELLIPSIS at most") }
    for item in items {
        if item is IndexToken and item.name == ELLIPSIS.name {
            for _axis in 0..a.ndim() - consumed { expanded.push(Slice()) }
        } else {
            expanded.push(item)
        }
    }
    if consumed > a.ndim() {
        throw IndexError(f"a key of {consumed} axes on an array of {a.ndim()}")
    }

    var shape: list[int] = []
    var strides: list[int] = []
    var offset = a.offset
    var axis = 0
    for item in expanded {
        if item is IndexToken {
            shape.push(1)
            strides.push(0)
            continue
        }
        let extent = a.shape[axis]
        let stride = a.strides[axis]
        axis += 1
        if item is int {
            let at = if item < 0 { item + extent } else { item }
            if at < 0 or at >= extent {
                throw IndexError(f"index {item} is out of bounds for an axis of {extent}")
            }
            offset += at * stride
            continue
        }
        if not item is Slice and not item is range {
            throw TypeError(f"an index may not be a {type_of(item)}")
        }
        let cut = if item is range { _slice_of_range(item) } else { item }
        let (first, count) = cut.resolve(extent)
        offset += first * stride
        shape.push(count)
        strides.push(stride * cut.step)
    }
    while axis < a.ndim() {
        shape.push(a.shape[axis])
        strides.push(a.strides[axis])
        axis += 1
    }
    return a.view(shape, strides, offset)
}

fn _fancy_get(a: NDArray, key: NDArray) -> NDArray {
    if dtype_is_bool(key.dtype) {
        if key.shape != a.shape {
            throw IndexError(f"a mask of {key.shape} on an array of {a.shape}")
        }
        let (offsets, total) = exclusive_scan(key)
        defer { offsets.free() }
        let out = NDArray([total], a.dtype)
        scatter_flat(out, offsets, a, key)
        return out
    }
    if a.ndim() != 1 {
        throw IndexError(
            f"an integer array indexes a 1-d array; use take() for {a.ndim()} dimensions"
        )
    }
    return gather_flat(a, key)
}

fn _fancy_set(a: NDArray, key: NDArray, value: any) -> void {
    if dtype_is_bool(key.dtype) {
        if value is NDArray {
            throw TypeError("a masked write takes a scalar; use where() for an array")
        }
        fill(a, _as_float(value), key)
        return
    }
    if a.ndim() != 1 { throw IndexError("an integer array writes into a 1-d array") }
    let source = if value is NDArray { value } else { _scalar_array(_as_float(value), a.dtype) }
    scatter_flat(a, key, source)
    if not value is NDArray { source.free() }
}

fn _scalar_array(value: float, dtype: int) -> NDArray {
    let out = NDArray([], dtype)
    fill(out, value)
    return out
}

#: `a` as a dense array starting at slot zero, and whether that cost an
#: allocation the caller now owns.
#:
#: `contiguous()` returns the array itself when it can, so freeing its result
#: unconditionally would free the caller's array. The flag is the only honest
#: way to say which of the two happened.
fn _dense(a: NDArray) -> (NDArray, bool) {
    if a.offset == 0 and a.is_contiguous() { return (a, false) }
    return (a.copy(), true)
}

#: The conversion mode a destination dtype asks of the copy kernels.
fn _cast_mode(dtype: int) -> int {
    if dtype == BOOL { return CAST_BOOL }
    if dtype_is_integer(dtype) { return CAST_ROUND }
    return CAST_NONE
}

#: Copy `src` into `dst`, broadcasting and converting as the shapes and dtypes
#: require.
pub fn copy_into(dst: NDArray, src: NDArray) -> void {
    let count = dst.size()
    if count == 0 { return }
    let source = src.broadcast_to(dst.shape)
    let mode = _cast_mode(dst.dtype)
    let low = f32_bits(dtype_min(dst.dtype))
    let high = f32_bits(dtype_max(dst.dtype))
    let lanes = dst.lanes()
    let dense_pair = dst.offset == 0 and dst.is_contiguous() and source.offset == 0
    if dense_pair and source.is_contiguous() and lanes == source.lanes() {
        kernel_of(SOURCE, "nd_copy").dispatch_async(
            [dst.storage, source.storage],
            count * lanes,
            [count * lanes, mode, low, high]
        )
        return
    }
    let meta = layout_buffer([dst, source])
    defer { meta.free() }
    kernel_of(SOURCE, "nd_copy_strided").dispatch_async(
        [dst.storage, source.storage, meta],
        count,
        [count, lanes, source.lanes(), mode, low, high]
    )
}

#: Set every element of `a` to `value`, or only those where `mask` is non-zero.
#:
#: A complex array takes the value on its real lane and zero on its imaginary
#: one, which is what assigning a real number to a complex array means.
pub fn fill(a: NDArray, value: float, mask: NDArray? = null) -> void {
    let count = a.size()
    if count == 0 { return }
    let lanes = a.lanes()
    let bits = f32_bits(value)
    if mask is not null {
        let gate = mask.broadcast_to(a.shape)
        let meta = layout_buffer([a, gate])
        defer { meta.free() }
        kernel_of(SOURCE, "nd_fill_masked").dispatch_async(
            [a.storage, gate.storage, meta],
            count,
            [count, lanes, bits]
        )
        return
    }
    if a.offset == 0 and a.is_contiguous() {
        kernel_of(SOURCE, "nd_fill").dispatch_async([a.storage], count, [count, lanes, bits])
        return
    }
    let meta = layout_buffer([a])
    defer { meta.free() }
    kernel_of(SOURCE, "nd_fill_strided").dispatch_async(
        [a.storage, meta],
        count,
        [count, lanes, bits]
    )
}

#: `left` op `right`, where `right` is an array or a scalar.
#:
#: The five operators on `NDArray` are the only callers that matter; `ufunc`
#: re-exports them under numpy's names rather than compiling the arithmetic a
#: second time. A class method cannot live in another file and the import DAG
#: forbids a cycle, so this is where they have to be.
#: Whether `sub` is exactly the trailing axes of `shape`.
#:
#: Exactly: no size-one axis standing in for a longer one. That is what makes
#: element `i` of the broadcast result element `i % size` of the operand, which
#: is what lets a repeating operand stay on a dense kernel instead of going
#: through the strided walk.
pub fn is_trailing(sub: list[int], shape: list[int]) -> bool {
    if sub.len() == 0 or sub.len() >= shape.len() { return false }
    let lead = shape.len() - sub.len()
    for axis in 0..sub.len() {
        if sub[axis] != shape[lead + axis] { return false }
    }
    return true
}

pub fn binary_arith(left: NDArray, right: any, op: int) -> NDArray {
    if right is NDArray { return _binary_arrays(left, right, op) }
    return _binary_scalar(left, _as_float(right), _scalar_dtype(right), op)
}

#: The dtype a bare Jaithon number contributes to a promotion.
#:
#: Its own type and not its value: numpy's value-based rule would make
#: `ints + 1.0` a float array and `ints + 1` an int one, and reading that off
#: `1.0 == trunc(1.0)` instead gets the first case wrong.
fn _scalar_dtype(value: any) -> int {
    if value is bool { return BOOL }
    if value is int { return INT32 }
    return FLOAT32
}

#: The dtype an arithmetic result carries.
#:
#: Two bools give an int rather than numpy's or-for-add, because half of these
#: operators have no boolean reading at all and one rule beats four. True
#: division always leaves the integers, as it does in numpy.
#:
#: An integer result saturates on the way out, the same rule `astype` and the
#: `ufunc` kernels follow: 2000000000 + 2000000000 in an int32 array answers
#: the int32 ceiling. numpy wraps there instead, which is a reading of a
#: 32-bit register this package does not have -- the value lives in float32
#: storage, where the wrapped pattern was never computed.
fn _result_dtype(a: int, b: int, op: int) -> int {
    var out = promote(a, b)
    if out == BOOL { out = INT32 }
    if op == OP_DIV and dtype_is_integer(out) { out = FLOAT32 }
    return out
}

#: 0 when the operand is dense and the full shape, its length when it repeats,
#: and -1 when neither and the strided walk has to run.
fn _wrap_of(a: NDArray, shape: list[int]) -> int {
    if a.offset != 0 or not a.is_contiguous() { return -1 }
    if a.shape == shape { return 0 }
    if is_trailing(a.shape, shape) { return a.size() }
    return -1
}

fn _binary_arrays(left: NDArray, right: NDArray, op: int) -> NDArray {
    let dtype = _result_dtype(left.dtype, right.dtype, op)
    let (a, b) = broadcast_pair(left, right)
    let lanes = dtype_lanes(dtype)

    #: One side is complex and the other is not. Widening both first is a real
    #: materialisation, but it happens only on that mix, and it keeps every
    #: kernel below on a single lane count.
    if lanes != a.lanes() or lanes != b.lanes() {
        let widened_a = a.astype(dtype)
        let widened_b = b.astype(dtype)
        defer { widened_a.free() }
        defer { widened_b.free() }
        return _binary_arrays(widened_a, widened_b, op)
    }
    let out = NDArray(a.shape, dtype)
    let count = out.size()
    if count == 0 { return out }
    let mode = _cast_mode(dtype)
    let low = f32_bits(dtype_min(dtype))
    let high = f32_bits(dtype_max(dtype))
    #: `broadcast_pair` has already made stride-zero views, so ask the ORIGINAL
    #: operands whether one of them merely repeats.
    let a_wrap = _wrap_of(left, a.shape)
    let b_wrap = _wrap_of(right, a.shape)
    if a_wrap >= 0 and b_wrap >= 0 {
        kernel_of(SOURCE, "nd_binary").dispatch_async(
            [out.storage, left.storage, right.storage],
            count,
            [count, lanes, op, mode, low, high, a_wrap, b_wrap]
        )
        return out
    }
    let meta = layout_buffer([out, a, b])
    defer { meta.free() }
    kernel_of(SOURCE, "nd_binary_strided").dispatch_async(
        [out.storage, a.storage, b.storage, meta],
        count,
        [count, lanes, op, mode, low, high]
    )
    return out
}

fn _binary_scalar(a: NDArray, value: float, scalar_dtype: int, op: int) -> NDArray {
    let dtype = _result_dtype(a.dtype, scalar_dtype, op)
    let out = NDArray(a.shape, dtype)
    let count = out.size()
    if count == 0 { return out }
    let lanes = out.lanes()
    let bits = f32_bits(value)
    let mode = _cast_mode(dtype)
    let low = f32_bits(dtype_min(dtype))
    let high = f32_bits(dtype_max(dtype))
    if a.offset == 0 and a.is_contiguous() and lanes == a.lanes() {
        kernel_of(SOURCE, "nd_binary_scalar").dispatch_async(
            [out.storage, a.storage],
            count,
            [count, lanes, op, bits, mode, low, high]
        )
        return out
    }
    let meta = layout_buffer([out, a.broadcast_to(out.shape)])
    defer { meta.free() }
    kernel_of(SOURCE, "nd_binary_scalar_strided").dispatch_async(
        [out.storage, a.storage, meta],
        count,
        [count, lanes, op, bits, mode, low, high]
    )
    return out
}

#: `source` read at each flat index in `indices`, shaped like `indices`.
#:
#: A negative index counts from the end. One outside the array is clamped
#: rather than raised on — a kernel cannot throw — so a caller that owes numpy
#: an `IndexError` checks the extremes first.
pub fn gather_flat(source: NDArray, indices: NDArray) -> NDArray {
    require_real(indices, "gather_flat")
    let (dense, owns_dense) = _dense(source)
    let (keys, owns_keys) = _dense(indices)
    defer { if owns_dense { dense.free() } }
    defer { if owns_keys { keys.free() } }
    let out = NDArray(indices.shape, source.dtype)
    let count = out.size()
    if count == 0 { return out }
    kernel_of(SOURCE, "nd_gather").dispatch_async(
        [out.storage, dense.storage, keys.storage],
        count,
        [count, out.lanes(), max(1, dense.size())]
    )
    return out
}

#: Write `values` into `dst` at each flat index in `indices`.
#:
#: `values` is either as long as `indices` or a single element broadcast over
#: all of them. Two indices naming the same slot race, and which one lands is
#: undefined — numpy's "the last one wins" is a promise a thread grid cannot
#: keep. A non-null `mask`, as long as `indices`, drops the writes where
#: it is zero — which is what makes a compaction out of `exclusive_scan`:
#: scan a mask into offsets, then scatter through those offsets under the same
#: mask, and the survivors land packed. `index`, `sorting` and `stats` all
#: build on that pair.
pub fn scatter_flat(
    dst: NDArray,
    indices: NDArray,
    values: NDArray,
    mask: NDArray? = null
) -> void {
    if not dst.is_contiguous() or dst.offset != 0 {
        throw ValueError("scatter_flat writes into a contiguous array")
    }
    let (keys, owns_keys) = _dense(indices)
    let (source, owns_source) = _dense(values)
    defer { if owns_keys { keys.free() } }
    defer { if owns_source { source.free() } }
    let count = keys.size()
    if count == 0 { return }
    if source.size() != count and source.size() != 1 {
        throw ValueError(f"{source.size()} values for {count} indices")
    }
    if mask is not null and mask.size() != count {
        throw ValueError(f"a mask of {mask.size()} for {count} indices")
    }
    let lanes = dst.lanes()
    let step = if source.size() == 1 { 0 } else { lanes }
    let gated = if mask is null { 0 } else { 1 }
    #: An unused gate still has to be bound, and the index buffer is the one
    #: read-only allocation already resident for this dispatch.
    let (gate, owns_gate) = _dense(mask ?? keys)
    defer { if owns_gate { gate.free() } }
    kernel_of(SOURCE, "nd_scatter").dispatch_async(
        [dst.storage, keys.storage, source.storage, gate.storage],
        count,
        [count, lanes, max(1, dst.size()), gated, step]
    )
}

#: Where each element would land if the non-zeros of `mask` were packed, plus
#: how many there are.
#:
#: Three dispatches over a fixed grid: per-group totals, one group turning
#: those into bases, then the per-element offsets. The grid is capped so the
#: middle pass stays serial in a number of groups it can hold, and the outer
#: two partition their chunks identically so the prefix is built in element
#: order.
#:
#: The returned count is the one host-visible value, and reading it costs a
#: synchronise — compaction cannot size its output without it.
pub fn exclusive_scan(mask: NDArray) -> (NDArray, int) {
    require_real(mask, "exclusive_scan")
    let (dense, owned) = _dense(mask)
    defer { if owned { dense.free() } }
    let count = dense.size()
    let offsets = NDArray([count], INT32)
    if count == 0 { return (offsets, 0) }

    let per = max(1, (count + _SCAN_GROUPS * GROUP - 1) // (_SCAN_GROUPS * GROUP))
    let span = per * GROUP
    let groups = (count + span - 1) // span
    let partials = Buffer(groups)
    let totals = Buffer(1)
    defer { partials.free() }
    defer { totals.free() }

    kernel_of(SOURCE, "nd_scan_totals").dispatch_async(
        [dense.storage, partials],
        groups * GROUP,
        [count, span, per],
        GROUP
    )
    kernel_of(SOURCE, "nd_scan_bases").dispatch_async(
        [partials, totals],
        GROUP,
        [groups, (groups + GROUP - 1) // GROUP],
        GROUP
    )
    kernel_of(SOURCE, "nd_scan_write").dispatch_async(
        [offsets.storage, dense.storage, partials],
        groups * GROUP,
        [count, span, per],
        GROUP
    )
    #: The download is the sync point -- it waits on the batch that wrote this
    #: buffer, and only on that -- so no global synchronise is needed here.
    return (offsets, int(totals.download(0, 1)[0]))
}

#: Groups the scan runs over at most. 1024 of them is a million elements at
#: one per thread before the per-thread chunk grows, and 1024 partials is four
#: per thread in the single group that scans them.
const _SCAN_GROUPS = 1024
