#: Opening a checkpoint and flattening what it holds onto dotted paths.
from .machine import Unpickler
from .pickle import Checkpoint
from .rebuild import tuple_items
from .tensor import TorchTensor
from .zip import Source, ZipArchive

#: The magic number a pre-1.6 checkpoint opens with, little-endian, as it sits
#: behind that container's `PROTO 2` and `LONG1 10`.
let LEGACY_MAGIC = [0x6C, 0xFC, 0x9C, 0x46, 0xF9, 0x20, 0x6A, 0xA8, 0x50, 0x19]

fn check_container(source: Source) -> void {
    if source.total < 4 { throw ValueError("this file is too short to be a torch checkpoint") }
    let data = source.at(0, min(source.total, 14))
    if data[0] == 0x50 and data[1] == 0x4B { return }
    if data[0] == 0x80 and data[2] == 0x8A and data[3] == 0x0A and data.len() >= 14 {
        var legacy = true
        for index in 0..10 {
            if data[4 + index] != LEGACY_MAGIC[index] { legacy = false }
        }
        if legacy {
            throw ValueError(
                "this is a pre-1.6 torch checkpoint, whose container is a bare pickle stream " +
                    "with the storages appended rather than the zip archive this reads. " +
                    "Load it under torch and re-save it: torch.save(state_dict, path) on " +
                    "torch 1.6 or later writes the zip form."
            )
        }
    }
    throw ValueError("this file is not a torch checkpoint: it is not a zip archive")
}

#: Where in the archive the checkpoint lives.
#:
#: torch names every member `<stem>/...`, and the stem is the file's own name
#: at the time it was saved, so it has to be read off the archive rather than
#: assumed.
fn find_root(archive: ZipArchive) -> str {
    var best = ""
    var found = false
    for name in archive.names {
        if name == "data.pkl" or name.ends_with("/data.pkl") {
            let root = name[0:name.len() - 8]
            if not found or root.len() < best.len() {
                best = root
                found = true
            }
        }
    }
    if not found {
        throw ValueError("this zip archive holds no data.pkl, so it is not a torch checkpoint")
    }
    return best
}

fn check_byte_order(archive: ZipArchive, root: str) -> void {
    let member = root + "byteorder"
    if not archive.has(member) { return }
    let order = archive.read(member).decode()
    if order != "little" {
        throw ValueError(
            f"this checkpoint was written on a {order}-endian machine, and this reader " +
                "only takes the little-endian form torch writes everywhere in practice"
        )
    }
}

#: Refuse a pickled `nn.Module` before running any of it.
#:
#: A module pickle carries the module's own `__dict__`, and every `nn.Module`
#: has these three keys in it; a state dict, whose keys are parameter paths,
#: has none of them. Catching the shape here rather than at the first refused
#: global is what lets the error say what to do about it, instead of naming
#: some class the reader has never heard of.
fn reject_module_pickle(pickled: bytes) -> void {
    if pickled.contains("_parameters".to_bytes()) and
        pickled.contains("_buffers".to_bytes()) and
        pickled.contains("_modules".to_bytes()) {
        throw ValueError(
            "this checkpoint holds a pickled nn.Module rather than a state dict. Rebuilding " +
                "the module needs its Python class, which the file does not carry and this " +
                "reader will not run. Re-save it as torch.save(model.state_dict(), path)."
        )
    }
}

fn joined(prefix: str, part: str) -> str {
    return prefix == "" ? part : prefix + "." + part
}

#: Walk the object graph, keeping the tensors and their paths.
fn flatten(prefix: str, value: any, out: dict[str, TorchTensor]) -> void {
    if value is TorchTensor {
        let tensor: TorchTensor = value
        if out.contains(prefix) {
            throw ValueError(f"two tensors in this checkpoint both arrive at the path '{prefix}'")
        }
        out[prefix] = tensor
        return
    }
    if value is dict {
        let mapping: dict[any, any] = value
        for (key, item) in mapping.items() {
            var text = ""
            if key is str { text = key } else { text = str(key) }
            flatten(joined(prefix, text), item, out)
        }
        return
    }
    if value is list {
        let items: list[any] = value
        for (index, item) in items.enumerate() {
            flatten(joined(prefix, str(index)), item, out)
        }
        return
    }
    let tupled = tuple_items(value)
    if tupled is not null {
        let items: list[any] = tupled
        for (index, item) in items.enumerate() {
            flatten(joined(prefix, str(index)), item, out)
        }
    }
}

fn read_source(source: Source) -> dict[str, TorchTensor] {
    check_container(source)
    let archive = ZipArchive(source)
    let root = find_root(archive)
    check_byte_order(archive, root)
    let pickled = archive.read(root + "data.pkl")
    reject_module_pickle(pickled)
    let machine = Unpickler(pickled, Checkpoint(archive, root))
    var out: dict[str, TorchTensor] = {}
    flatten("", machine.run(), out)
    return out
}

#: Every tensor in a checkpoint already in memory, by its dotted path.
pub fn read_torch_state_dict_bytes(data: bytes) -> dict[str, TorchTensor] {
    return read_source(Source.of_bytes(data))
}

#: Every tensor in a checkpoint on disk, by its dotted path.
#:
#: A `torch.save(model.state_dict(), path)` arrives with the keys torch used.
#: A checkpoint holding more than the weights — the usual
#: `{"model": ..., "optimizer": ..., "epoch": 12}` — arrives with the nesting
#: flattened onto those keys, and the epoch dropped. A bare
#: `torch.save(tensor, path)` has no path to build, so its one tensor arrives
#: under the empty key.
#:
#: Which device the checkpoint was saved from does not matter: the storages are
#: the same bytes either way, and the `cuda:0` a persistent id carries is read
#: past rather than acted on.
pub fn read_torch_state_dict(path: str) -> dict[str, TorchTensor] {
    let source = Source.of_file(path)
    defer { source.close() }
    return read_source(source)
}
