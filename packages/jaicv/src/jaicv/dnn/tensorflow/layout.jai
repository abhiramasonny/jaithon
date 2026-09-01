#: The invariant: permuting a tensor between NHWC and NCHW, and renumbering an axis.

#: Axis `k` of an imported rank-four tensor holds TensorFlow's axis
#: `TF_AXIS[k]`: batch, then channels, then the two spatial axes. Read the
#: other way round it is also the permutation that turns an `NHWC` tensor into
#: an `NCHW` one, which is what a `Transpose` node gets given.
pub let TF_AXIS = [0, 3, 1, 2]

#: The inverse. TensorFlow's axis `a` is axis `NCHW_AXIS[a]` here, and the
#: list is equally the permutation back to `NHWC`.
pub let NCHW_AXIS = [0, 2, 3, 1]

#: `values` laid out row-major for `shape`, reordered so that the source's
#: axis `perm[k]` becomes axis `k`.
pub fn permuted(values: list[float], shape: list[int], perm: list[int]) -> list[float] {
    let rank = shape.len()
    var strides = [1 for _axis in 0..rank]
    var axis = rank - 2
    while axis >= 0 {
        strides[axis] = strides[axis + 1] * shape[axis + 1]
        axis -= 1
    }
    let dims = [shape[perm[slot]] for slot in 0..rank]
    let steps = [strides[perm[slot]] for slot in 0..rank]
    var total = 1
    for dimension in dims { total *= dimension }
    var index = [0 for _axis in 0..rank]
    var out: list[float] = []
    for _slot in 0..total {
        var offset = 0
        for k in 0..rank { offset += index[k] * steps[k] }
        out.push(values[offset])
        var k = rank - 1
        while k >= 0 {
            index[k] += 1
            if index[k] < dims[k] { break }
            index[k] = 0
            k -= 1
        }
    }
    return out
}

pub fn reordered(shape: list[int], perm: list[int]) -> list[int] {
    return [shape[perm[slot]] for slot in 0..perm.len()]
}

#: Whether dropping TensorFlow's axes `dropped` out of a rank-four tensor
#: leaves the survivors in the same order here as it does in TensorFlow.
#:
#: Reducing or squeezing away `H` and `W` together leaves `NC` both ways, and
#: so the operator runs on the stored tensor with nothing but its axis numbers
#: changed. Dropping `H` alone leaves `NCW` here and `NWC` there, and then the
#: tensor has to be put back into TensorFlow's order first. Getting this wrong
#: produces a correctly shaped output with its values shuffled, which is the
#: hardest kind of wrong to notice.
pub fn survives_in_order(dropped: list[int]) -> bool {
    var stored: list[int] = []
    for axis in TF_AXIS {
        if not axis in dropped { stored.push(axis) }
    }
    var native: list[int] = []
    for axis in 0..4 {
        if not axis in dropped { native.push(axis) }
    }
    return stored == native
}

#: Whether a value that started at `from_rank` and had axes dropped down to
#: `to_rank` needs a transpose into `NCHW` before its new rank is recorded.
#:
#: Only a starting rank of four is ever stored `NCHW`; every other rank is
#: exactly TensorFlow's own order by the invariant this module keeps, and
#: dropping axes never reorders the survivors. So a result landing at rank
#: four out of a starting rank that was NOT four is still in TensorFlow's
#: order -- NHWC in everything but name -- and has to be turned into `NCHW`
#: before anything downstream treats it as an image, the same as `Reshape`
#: already does. A starting rank of four is handled by its own caller, by
#: renumbering axes or transposing the input first, so this only ever answers
#: yes for the other case.
pub fn lands_in_nhwc(from_rank: int, to_rank: int) -> bool {
    return from_rank != 4 and to_rank == 4
}
