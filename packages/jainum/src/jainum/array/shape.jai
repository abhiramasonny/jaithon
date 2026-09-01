#: Element counts, axis normalisation, strides and broadcast shapes: the list
#: arithmetic an array's layout is made of, with no array in sight.

#: Elements a shape holds. An empty shape is one element, not none: a 0-d
#: array is a scalar living on the device.
pub fn element_count(shape: list[int]) -> int {
    var total = 1
    for extent in shape {
        if extent < 0 { throw ValueError(f"a shape cannot contain {extent}") }
        total *= extent
    }
    return total
}

#: Turn a possibly negative axis into an index into `0..ndim`.
pub fn normalize_axis(axis: int, ndim: int, name: str = "axis") -> int {
    let at = if axis < 0 { axis + ndim } else { axis }
    if at < 0 or at >= ndim {
        throw ValueError(f"{name} {axis} is out of bounds for an array of {ndim} dimensions")
    }
    return at
}

#: `normalize_axis` over a list, rejecting a repeat.
pub fn normalize_axes(axes: list[int], ndim: int, name: str = "axis") -> list[int] {
    var seen: dict[int, bool] = {}
    var out: list[int] = []
    for axis in axes {
        let at = normalize_axis(axis, ndim, name)
        if seen.get(at, false) { throw ValueError(f"{name} {axis} appears twice") }
        seen[at] = true
        out.push(at)
    }
    return out
}

#: Row-major strides for `shape`, in float slots.
pub fn contiguous_strides(shape: list[int], lanes: int) -> list[int] {
    var strides = [0 for _axis in shape]
    var running = lanes
    var axis = shape.len() - 1
    while axis >= 0 {
        strides[axis] = running
        running *= shape[axis]
        axis -= 1
    }
    return strides
}

#: Whether `strides` walks `shape` in row-major order with nothing between the
#: elements.
#:
#: An axis of one element is skipped: its stride is never used, so a view that
#: kept a stale one is still as dense as a fresh allocation, and refusing it
#: would send `reshape` through a copy for nothing.
pub fn is_contiguous(shape: list[int], strides: list[int], lanes: int) -> bool {
    if element_count(shape) == 0 { return true }
    var expected = lanes
    var axis = shape.len() - 1
    while axis >= 0 {
        let extent = shape[axis]
        if extent != 1 {
            if strides[axis] != expected { return false }
            expected *= extent
        }
        axis -= 1
    }
    return true
}

#: The shape `a` and `b` broadcast to, aligned from the right.
#:
#: The surviving extent is the one that is not 1, which is `max` everywhere
#: except against a zero-length axis: numpy answers `(3, 0)` for `(3, 0)`
#: against `(3, 1)` and against `()`, because a length-one axis stretches to
#: whatever the other side is and an empty one stays empty. `max` answered 1
#: there and handed the caller an output with three elements it then could
#: not fill from an empty operand.
pub fn broadcast_shapes(a: list[int], b: list[int]) -> list[int] {
    let rank = max(a.len(), b.len())
    var out = [0 for _axis in 0..rank]
    for axis in 0..rank {
        let left = if axis < rank - a.len() { 1 } else { a[axis - (rank - a.len())] }
        let right = if axis < rank - b.len() { 1 } else { b[axis - (rank - b.len())] }
        if left != right and left != 1 and right != 1 {
            throw ValueError(f"shapes {a} and {b} do not broadcast together")
        }
        out[axis] = if left == 1 { right } else { left }
    }
    return out
}

#: Strides that read `shape` as if it were `target`.
#:
#: A new leading axis and a stretched length-one axis both get stride 0, so
#: every thread of the broadcast axis reads the same slot and no memory is
#: copied to make the shapes agree.
pub fn broadcast_strides(shape: list[int], strides: list[int], target: list[int]) -> list[int] {
    let pad = target.len() - shape.len()
    if pad < 0 { throw ValueError(f"shape {shape} cannot broadcast to the shorter {target}") }
    var out = [0 for _axis in target]
    for axis in 0..target.len() {
        if axis < pad { continue }
        let extent = shape[axis - pad]
        if extent == target[axis] {
            out[axis] = strides[axis - pad]
        } elif extent != 1 {
            throw ValueError(f"shape {shape} does not broadcast to {target}")
        }
    }
    return out
}
