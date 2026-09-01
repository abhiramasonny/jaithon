#: The whitelist a pickled global is resolved against, and the rebuilding of a
#: tensor from the storage and strides it was saved as.
from .bits import dtype_size, STORAGE_DTYPES
from .pickle import PickleTuple, Storage
from .tensor import TorchTensor

#: Whether a pickle may name this global.
#:
#: Everything on this list either builds a container or rebuilds a tensor, and
#: none of it reaches outside the file. Nothing else is reachable, and there is
#: no way for a caller to add to it -- a whitelist a caller can widen is not a
#: whitelist.
pub fn allowed_global(module_name: str, name: str) -> bool {
    if module_name == "torch._utils" {
        return name == "_rebuild_tensor" or
            name == "_rebuild_tensor_v2" or
            name == "_rebuild_parameter"
    }
    if module_name == "torch" { return STORAGE_DTYPES.contains(name) }
    if module_name == "collections" { return name == "OrderedDict" }
    if module_name == "builtins" or module_name == "__builtin__" {
        return name == "list" or
            name == "dict" or
            name == "tuple" or
            name == "set" or
            name == "frozenset"
    }
    return false
}

pub fn refuse_global(module_name: str, name: str) -> void {
    throw ValueError(
        f"refusing the pickled global {module_name}.{name}: a checkpoint may name only " +
            "torch._utils._rebuild_tensor, _rebuild_tensor_v2 and _rebuild_parameter, the " +
            "torch storage classes, collections.OrderedDict, and the container builtins. " +
            "Resolving anything else would let the file choose what this program runs."
    )
}

#: The items of a pickle tuple, or `null` if this is not one.
pub fn tuple_items(value: any) -> list[any]? {
    if value is PickleTuple {
        let pair: PickleTuple = value
        return pair.items
    }
    return null
}

#: A tuple or list of ints, as a list of ints.
fn int_items(value: any, what: str) -> list[int] {
    var source = tuple_items(value)
    if source is null and value is list {
        let listed: list[any] = value
        source = listed
    }
    if source is null {
        throw ValueError(f"a tensor's {what} should be a tuple of ints, and is not")
    }
    let items: list[any] = source
    var out: list[int] = []
    for item in items {
        if not item is int {
            throw ValueError(f"a tensor's {what} holds something that is not an int")
        }
        let whole: int = item
        out.push(whole)
    }
    return out
}

#: Whether `stride` is the row-major packing of `shape`.
#:
#: A dimension of extent one is skipped, because its stride never enters an
#: index and torch does not promise anything about what it holds there.
fn is_packed(shape: list[int], stride: list[int]) -> bool {
    if shape.len() != stride.len() { return false }
    var expected = 1
    var index = shape.len() - 1
    while index >= 0 {
        if shape[index] != 1 and stride[index] != expected { return false }
        expected *= shape[index]
        index -= 1
    }
    return true
}

#: Rebuild `_rebuild_tensor_v2`'s tensor: a storage, a first element, and the
#: shape and strides of a view onto it.
#:
#: The packed case is a slice of the storage, which is one copy. Anything else
#: -- a transpose, a strided slice -- is walked element by element with an
#: odometer over the shape, so a view arrives already laid out the way its own
#: shape says and no caller downstream has to carry strides around.
pub fn rebuild_tensor(args: list[any], named: str) -> TorchTensor {
    if args.len() < 4 {
        throw ValueError(f"{named} wants at least four arguments, and got {args.len()}")
    }
    if not args[0] is Storage {
        throw ValueError(f"{named} was not handed a storage to build from")
    }
    let storage: Storage = args[0]
    if not args[1] is int {
        throw ValueError(f"{named} was handed a storage offset that is not an int")
    }
    let offset: int = args[1]
    let shape = int_items(args[2], "shape")
    let stride = int_items(args[3], "stride")
    if shape.len() != stride.len() {
        throw ValueError(f"{named} was handed {shape.len()} dimensions and {stride.len()} strides")
    }
    let width = dtype_size(storage.dtype)
    var count = 1
    for extent in shape {
        if extent < 0 { throw ValueError(f"{named} was handed a negative extent in {shape}") }
        count *= extent
    }
    if count == 0 { return TorchTensor(shape, storage.dtype, bytes([])) }
    if offset < 0 { throw ValueError(f"{named} was handed a negative storage offset") }

    var last = offset
    for index in 0..shape.len() {
        if stride[index] < 0 {
            throw ValueError(f"{named} was handed a negative stride, which cannot be saved")
        }
        last += (shape[index] - 1) * stride[index]
    }
    if last >= storage.count {
        throw ValueError(f"{named} reaches element {last} of a storage holding {storage.count}")
    }

    if is_packed(shape, stride) {
        return TorchTensor(
            shape,
            storage.dtype,
            storage.data.slice(offset * width, (offset + count) * width)
        )
    }

    var out: list[int] = []
    var odometer: list[int] = []
    for _index in 0..shape.len() { odometer.push(0) }
    var element = offset
    for _step in 0..count {
        let at = element * width
        for byte in 0..width { out.push(storage.data[at + byte]) }
        var axis = shape.len() - 1
        while axis >= 0 {
            odometer[axis] += 1
            element += stride[axis]
            if odometer[axis] < shape[axis] { break }
            element -= stride[axis] * shape[axis]
            odometer[axis] = 0
            axis -= 1
        }
    }
    return TorchTensor(shape, storage.dtype, bytes(out))
}
