#: Binary tensor checkpoints for hand-written models.
#:
#: `Sequential.save_weights` writes JSON, which is right for a small model
#: and wrong for a large one: every float becomes about twenty characters of
#: text, built one boxed number at a time. A 3.3-million-parameter GPT was
#: estimated at ten seconds a save that way. Here the values travel as their
#: own bits: the device turns each float into its four bytes, one
#: `download_u8` brings the lot to the host, and loading runs the same two
#: steps backwards. Nothing is rounded, so a round trip is bit-exact,
#: infinities, NaN payloads and signed zeros included.
#:
#: A file is a magic line, the length of a JSON header in decimal on a line of
#: its own, the header, spaces up to a multiple of 64 bytes, and then every
#: tensor's float32 values in little-endian order, back to back. The header
#: names each tensor, gives its shape and its offset in floats into that data,
#: and carries whatever `meta` the caller saved, so a model's own settings --
#: its width, its step count -- can travel with its weights:
#:
#: ```
#: {"format": 1, "dtype": "f32", "tensors": [{"name": "wte", "shape": [256, 256], "offset": 0}, ...], "meta": {...}}
#: ```
#:
#: Any rank, any order; the names are the caller's. `save_checkpoint` writes
#: a list of named tensors, `load_checkpoint` reads them back as new tensors,
#: and `restore_checkpoint` copies them into tensors a model already holds.
import std.json as json
from std.gpu import Buffer, Kernel
from std.io import open, read_bytes
from .tensor import Tensor

let MAGIC = "JAITENSOR-CHECKPOINT 1\n"

#: Bytes the header and its preamble are padded to, so the values start
#: aligned however long the names are.
let ALIGN = 64

let _BYTES_SOURCE = """#include <metal_stdlib>
using namespace metal;

kernel void f32_to_bytes(device const uint* source [[buffer(0)]],
                         device float* dest [[buffer(1)]],
                         constant uint& count [[buffer(2)]],
                         constant uint& offset [[buffer(3)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint bits = source[id];
    uint at = (offset + id) * 4u;
    dest[at] = float(bits & 0xFFu);
    dest[at + 1u] = float((bits >> 8) & 0xFFu);
    dest[at + 2u] = float((bits >> 16) & 0xFFu);
    dest[at + 3u] = float(bits >> 24);
}

kernel void bytes_to_f32(device const float* source [[buffer(0)]],
                         device uint* dest [[buffer(1)]],
                         constant uint& count [[buffer(2)]],
                         constant uint& offset [[buffer(3)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint at = (offset + id) * 4u;
    dest[id] = uint(source[at]) | (uint(source[at + 1u]) << 8)
             | (uint(source[at + 2u]) << 16) | (uint(source[at + 3u]) << 24);
}
"""

#: Tensors read back from a checkpoint, with the metadata saved beside them.
#: The tensors are new and belong to the caller: `free` releases them all.
pub class Checkpoint {
    pub let names: list[str]
    pub let tensors: list[Tensor]
    pub let meta: dict[str, any]

    pub fn init(self, names: list[str], tensors: list[Tensor], meta: dict[str, any]) {
        self.names = names
        self.tensors = tensors
        self.meta = meta
    }

    #: The tensor saved as `name`.
    pub fn get(self, name: str) -> Tensor {
        for (index, known) in self.names.enumerate() {
            if known == name { return self.tensors[index] }
        }
        throw KeyError(f"checkpoint has no tensor named {repr(name)}")
    }

    #: Whether a tensor was saved as `name`.
    pub fn has(self, name: str) -> bool { return self.names.contains(name) }

    #: Release every tensor this checkpoint loaded.
    pub fn free(self) -> void {
        for tensor in self.tensors { tensor.free() }
    }
}

#: Write `tensors` to `path` under `names`, with `meta` in the header.
#:
#: Waits for the work that produces the tensors, and no more.
pub fn save_checkpoint(
    path: str,
    names: list[str],
    tensors: list[Tensor],
    meta: dict[str, any]? = null
) -> void {
    if names.len() != tensors.len() {
        throw ValueError(f"save_checkpoint got {names.len()} names for {tensors.len()} tensors")
    }
    var entries: list[any] = []
    var total = 0
    for (index, name) in names.enumerate() {
        if names[0:index].contains(name) {
            throw ValueError(f"save_checkpoint got {repr(name)} twice")
        }
        let tensor = tensors[index]
        entries.push({"name": name, "shape": tensor.shape, "offset": total})
        total += tensor.size
    }
    var data = bytes([])
    if total > 0 {
        let staging = Buffer(4 * total)
        let kernel = Kernel.compile(_BYTES_SOURCE, "f32_to_bytes")
        var offset = 0
        for tensor in tensors {
            kernel.dispatch_async([tensor.buffer(), staging], tensor.size, [tensor.size, offset])
            offset += tensor.size
        }
        data = staging.download_u8(0, 4 * total)
        staging.free()
    }
    let document: dict[str, any] = {
        "format": 1,
        "dtype": "f32",
        "tensors": entries,
        "meta": meta ?? {},
    }
    let header = bytes(json.stringify(document))
    let preamble = bytes(MAGIC + f"{header.len()}\n")
    let used = preamble.len() + header.len()
    let pad = (ALIGN - used % ALIGN) % ALIGN
    let file = open(path, "w")
    defer { file.close() }
    let _a = file.write_bytes(preamble)
    let _b = file.write_bytes(header)
    let _c = file.write_bytes(bytes(" ").repeat(pad))
    let _d = file.write_bytes(data)
}

#: The parsed header of a checkpoint and where its values start.
class Layout {
    pub let names: list[str]
    pub let shapes: list[list[int]]
    pub let offsets: list[int]
    pub let total: int
    pub let data_start: int
    pub let meta: dict[str, any]

    pub fn init(self, raw: bytes, path: str) {
        let magic = bytes(MAGIC)
        if raw.len() < magic.len() or raw.slice(0, magic.len()) != magic {
            throw ValueError(f"{path} is not a jaitensor checkpoint")
        }
        let line_end = raw.find(10, magic.len())
        if line_end < 0 { throw ValueError(f"{path}: checkpoint header is truncated") }
        let length = int(raw.slice(magic.len(), line_end).decode())
        let start = line_end + 1
        if start + length > raw.len() {
            throw ValueError(f"{path}: checkpoint header is truncated")
        }
        let document = json.parse(raw.slice(start, start + length).decode())
        if document["format"] != 1 or document["dtype"] != "f32" {
            throw ValueError(f"{path}: unsupported checkpoint format")
        }
        let used = start + length
        self.data_start = used + (ALIGN - used % ALIGN) % ALIGN
        var names: list[str] = []
        var shapes: list[list[int]] = []
        var offsets: list[int] = []
        var total = 0
        for entry in document["tensors"] {
            let name: str = entry["name"]
            var shape: list[int] = []
            var size = 1
            for dimension in entry["shape"] {
                let d: int = dimension
                shape.push(d)
                size *= d
            }
            let offset: int = entry["offset"]
            if offset != total { throw ValueError(f"{path}: tensor {repr(name)} is out of place") }
            names.push(name)
            shapes.push(shape)
            offsets.push(offset)
            total += size
        }
        if self.data_start + 4 * total > raw.len() {
            throw ValueError(f"{path}: checkpoint holds fewer values than its header lists")
        }
        self.names = names
        self.shapes = shapes
        self.offsets = offsets
        self.total = total
        let meta: dict[str, any] = document["meta"]
        self.meta = meta
    }

    pub fn index_of(self, name: str) -> int {
        for (index, known) in self.names.enumerate() {
            if known == name { return index }
        }
        return -1
    }

    #: Every value, as bytes widened to floats on the device.
    pub fn staging(self, raw: bytes) -> Buffer {
        let staging = Buffer(max(4 * self.total, 1))
        if self.total > 0 {
            staging.upload_u8(raw, source_offset: self.data_start, count: 4 * self.total)
        }
        return staging
    }
}

fn unpack_into(staging: Buffer, target: Tensor, offset: int) -> void {
    let kernel = Kernel.compile(_BYTES_SOURCE, "bytes_to_f32")
    kernel.dispatch_async([staging, target.buffer()], target.size, [target.size, offset])
}

#: Read every tensor in the checkpoint at `path` into new tensors.
pub fn load_checkpoint(path: str) -> Checkpoint {
    let raw = read_bytes(path)
    let layout = Layout(raw, path)
    let staging = layout.staging(raw)
    var tensors: list[Tensor] = []
    for (index, shape) in layout.shapes.enumerate() {
        let tensor = Tensor(shape)
        unpack_into(staging, tensor, layout.offsets[index])
        tensors.push(tensor)
    }
    #: Queued work keeps its buffers alive, so the staging buffer can go now.
    staging.free()
    return Checkpoint(layout.names, tensors, layout.meta)
}

#: Copy the checkpoint at `path` into tensors that already exist -- a model's
#: parameters -- by name, and return the metadata saved with it.
#:
#: Every tensor named must be in the file with the same shape; the file may
#: hold others, which are left alone. The writes are queued on the device, so
#: they land after any work already reading the old values.
pub fn restore_checkpoint(path: str, names: list[str], tensors: list[Tensor]) -> dict[str, any] {
    if names.len() != tensors.len() {
        throw ValueError(f"restore_checkpoint got {names.len()} names for {tensors.len()} tensors")
    }
    let raw = read_bytes(path)
    let layout = Layout(raw, path)
    var slots: list[int] = []
    for (index, name) in names.enumerate() {
        let slot = layout.index_of(name)
        if slot < 0 { throw KeyError(f"{path} has no tensor named {repr(name)}") }
        if layout.shapes[slot] != tensors[index].shape {
            throw ValueError(
                f"{path}: {repr(name)} is {layout.shapes[slot]} in the file and {tensors[index].shape} here"
            )
        }
        slots.push(slot)
    }
    let staging = layout.staging(raw)
    for (index, tensor) in tensors.enumerate() {
        unpack_into(staging, tensor, layout.offsets[slots[index]])
    }
    staging.free()
    return layout.meta
}
