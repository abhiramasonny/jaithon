#: Reading a PyTorch checkpoint.
#:
#: A `.pt` or `.pth` written by torch 1.6 or later is a ZIP archive holding one
#: pickled object graph in `data.pkl` and every tensor's raw storage as a
#: separate member under `data/`. Reading one therefore needs two things that
#: are not otherwise in this tree — a ZIP reader and a pickle interpreter — and
#: both are here. The deflate itself is zlib's, through `__prim__.inflate`.
#:
#: Reading from a path reads through the handle rather than pulling the file in
#: first: the directory, then each member as it is wanted. A checkpoint big
#: enough to need the zip64 fields is one that must not be held twice over, and
#: `read_torch_state_dict_bytes` — which takes the file already in memory — is
#: for the callers that already have it.
#:
#: The pickle interpreter is deliberately crippled, and that is the whole point
#: of it. A pickle names callables and asks for them to be called, so a general
#: interpreter is arbitrary code execution on a file that came from somewhere
#: else: the standard demonstration is a checkpoint whose pickle names
#: `os.system`. This one resolves globals against a fixed whitelist —
#: `torch._utils._rebuild_tensor`, `_rebuild_tensor_v2` and `_rebuild_parameter`,
#: the ten `torch.*Storage` classes, `collections.OrderedDict`, and the plain
#: container builtins — and throws on anything else, naming what it refused.
#: There is no flag to widen that list. A checkpoint needing more than this is
#: a checkpoint that should be re-saved.
#:
#: What comes back is the object graph flattened onto dotted paths, so a nested
#: `{"model": {"conv": {"weight": t}}}` arrives as `model.conv.weight` and a
#: list of tensors as `0`, `1`, and so on. Leaves that are not tensors — an
#: epoch counter, a learning rate, a `None` — are dropped: this reads weights.
#:
#: A `TorchTensor` is deliberately not the `Tensor` the executor in
#: `dnn.graph` runs on. That one is jaitensor's: on the GPU, float32, and every
#: dimension positive. A checkpoint is none of those things — it holds `int64`
#: counters, `bool` masks, 0-dimensional scalars and empty tensors, and an
#: `int64` past 2^53 does not survive a float32 — so a reader that handed back
#: device tensors would be deciding what to lose before the caller had seen it.
#: Going the other way is one line where a caller actually wants it:
#: `jt.Tensor(weight.shape, weight.floats())`.
#:
#: ```
#: let weights = read_torch_state_dict("resnet18.pth")
#: weights["fc.weight"].shape      # [1000, 512]
#: weights["fc.weight"].floats()   # 512000 values, row-major
#: ```
import std.io as io
from std.io import File
import std.math
from std.core import Printable

#: What one element of each torch storage class is, in this module's names.
#:
#: This doubles as the whitelist of storage classes: a `torch.` global is
#: allowed exactly when it is a key here.
let STORAGE_DTYPES: dict[str, str] = {
    "FloatStorage": "float32",
    "DoubleStorage": "float64",
    "HalfStorage": "float16",
    "BFloat16Storage": "bfloat16",
    "LongStorage": "int64",
    "IntStorage": "int32",
    "ShortStorage": "int16",
    "CharStorage": "int8",
    "ByteStorage": "uint8",
    "BoolStorage": "bool",
}

#: Bytes per element.
let DTYPE_SIZES: dict[str, int] = {
    "float32": 4,
    "float64": 8,
    "float16": 2,
    "bfloat16": 2,
    "int64": 8,
    "int32": 4,
    "int16": 2,
    "int8": 1,
    "uint8": 1,
    "bool": 1,
}

#: How wide one element of `dtype` is, in bytes.
pub fn dtype_size(dtype: str) -> int {
    if not DTYPE_SIZES.contains(dtype) {
        throw ValueError(f"'{dtype}' is not a dtype this reader knows")
    }
    return DTYPE_SIZES[dtype]
}

fn le_u16(data: bytes, at: int) -> int {
    return data[at] | data[at + 1] << 8
}

fn le_u32(data: bytes, at: int) -> int {
    return data[at] | data[at + 1] << 8 | data[at + 2] << 16 | data[at + 3] << 24
}

fn be_u32(data: bytes, at: int) -> int {
    return data[at] << 24 | data[at + 1] << 16 | data[at + 2] << 8 | data[at + 3]
}

#: A little-endian integer `width` bytes wide, read as two's complement when
#: `signed`.
#:
#: The top byte is folded in by multiplication rather than by a shift, because
#: an unsigned `0xFF` in the eighth byte is past `INT_MAX` and shifting it
#: there raises an overflow before the sign can be applied.
fn le_int(data: bytes, at: int, width: int, signed: bool) -> int {
    if width == 0 { return 0 }
    var value = 0
    var scale = 1
    for index in 0..width - 1 {
        value += data[at + index] * scale
        scale *= 256
    }
    var top = data[at + width - 1]
    if signed and top >= 128 { top -= 256 }
    if not signed and width >= 8 {
        throw ValueError("an unsigned 64-bit field does not fit in an int")
    }
    return value + top * scale
}

fn le_u64(data: bytes, at: int) -> int {
    if data[at + 7] >= 128 {
        throw ValueError("a ZIP field claims a size past 2^63, which cannot be a real file")
    }
    return le_int(data, at, 8, true)
}

#: An IEEE-754 binary32 out of its 32 bits.
#:
#: Assembled with `ldexp` rather than by any arithmetic on the value, so a
#: subnormal, a zero of either sign, and the exact mantissa of a large value
#: all survive: every step is exact in a double.
fn f32_from_bits(bits: int) -> float {
    let exponent = bits >> 23 & 0xFF
    let mantissa = bits & 0x7FFFFF
    var value = 0.0
    if exponent == 0 {
        value = math.ldexp(float(mantissa), -149)
    } elif exponent == 0xFF {
        value = mantissa == 0 ? math.INF : math.NAN
    } else {
        value = math.ldexp(float(mantissa + 0x800000), exponent - 150)
    }
    return bits >> 31 & 1 == 1 ? -value : value
}

#: An IEEE-754 binary64 out of its two halves.
#:
#: The 53-bit significand is built as `high * 2^32 + low`, which is at most
#: `2^53 - 1` and so still exact in a double before `ldexp` scales it.
fn f64_from_halves(high: int, low: int) -> float {
    let exponent = high >> 20 & 0x7FF
    let fraction = high & 0xFFFFF
    var value = 0.0
    if exponent == 0 {
        value = math.ldexp(float(fraction) * 4294967296.0 + float(low), -1074)
    } elif exponent == 0x7FF {
        value = fraction == 0 and low == 0 ? math.INF : math.NAN
    } else {
        value = math.ldexp(
            (float(fraction) + 1048576.0) * 4294967296.0 + float(low),
            exponent - 1075
        )
    }
    return high >> 31 & 1 == 1 ? -value : value
}

#: An IEEE-754 binary16 out of its 16 bits.
fn f16_from_bits(bits: int) -> float {
    let exponent = bits >> 10 & 0x1F
    let mantissa = bits & 0x3FF
    var value = 0.0
    if exponent == 0 {
        value = math.ldexp(float(mantissa), -24)
    } elif exponent == 0x1F {
        value = mantissa == 0 ? math.INF : math.NAN
    } else {
        value = math.ldexp(float(mantissa + 1024), exponent - 25)
    }
    return bits >> 15 & 1 == 1 ? -value : value
}

#: One member of a ZIP archive, as the central directory describes it.
class ZipEntry {
    pub let name: str
    pub let method: int
    pub let stored_size: int
    pub let size: int
    pub let crc: int
    pub let header_at: int

    pub fn init(
        self,
        name: str,
        method: int,
        stored_size: int,
        size: int,
        crc: int,
        header_at: int
    ) {
        self.name = name
        self.method = method
        self.stored_size = stored_size
        self.size = size
        self.crc = crc
        self.header_at = header_at
    }
}

#: Where an archive's bytes come from.
#:
#: A block already in memory is the easy case and the one a caller who already
#: has the file gets. The other case is why this exists: a checkpoint big
#: enough to need zip64 is a checkpoint that must not be pulled into memory
#: whole and then copied storage by storage back out of itself, which wants
#: twice the file in RAM before a single weight is available. Reading through
#: the handle costs one seek per member instead.
class Source {
    pub let block: bytes?
    pub let handle: File?
    pub let total: int

    pub fn init(self, block: bytes?, handle: File?, total: int) {
        self.block = block
        self.handle = handle
        self.total = total
    }

    #: An archive that is already a `bytes`.
    pub static fn of_bytes(data: bytes) -> Source { return Source(data, null, data.len()) }

    #: An archive on disk, read a window at a time.
    pub static fn of_file(path: str) -> Source {
        let handle = io.open(path, "r")
        return Source(null, handle, handle.seek(0, io.SEEK_END))
    }

    #: `count` bytes starting at `offset`.
    pub fn at(self, offset: int, count: int) -> bytes {
        if offset < 0 or count < 0 or offset + count > self.total {
            throw ValueError(
                f"this archive is {self.total} bytes and something in it points at " +
                    f"{offset}..{offset + count}, which is past the end"
            )
        }
        let held = self.block
        if held is not null {
            let data: bytes = held
            return data.slice(offset, offset + count)
        }
        let opened = self.handle
        if opened is not null {
            let file: File = opened
            let _moved = file.seek(offset, io.SEEK_SET)
            var taken = file.read_bytes(count)
            #: A short read is not an error on a stream, only an incomplete
            #: one, so the rest is asked for again rather than handed back as
            #: though it were the answer.
            while taken.len() < count {
                let more = file.read_bytes(count - taken.len())
                if more.len() == 0 {
                    throw ValueError(f"this archive ends {count - taken.len()} bytes early")
                }
                taken = taken.concat(more)
            }
            return taken
        }
        throw ValueError("this archive has neither a block nor a handle to read from")
    }

    #: Release the handle, if this source has one.
    pub fn close(self) -> void {
        let opened = self.handle
        if opened is not null {
            let file: File = opened
            file.close()
        }
    }
}

#: A ZIP archive, read through its central directory.
#:
#: The central directory is the authority on every member's size, method and
#: checksum, and it has to be: torch's writer sets the streaming flag in each
#: local header and leaves the sizes and the CRC there zeroed. The local header
#: is still consulted, but only for its own name and extra-field lengths, which
#: differ from the central directory's — torch pads the local extra field to
#: align the storage that follows it.
class ZipArchive {
    pub let source: Source
    pub let names: list[str]
    pub let entries: dict[str, ZipEntry]

    pub fn init(self, source: Source) {
        self.source = source
        self.names = []
        self.entries = {}
        let (end_at, tail, tail_at) = self._find_end_record()
        var count = le_u16(tail, end_at - tail_at + 8)
        var directory_at = le_u32(tail, end_at - tail_at + 16)
        var directory_size = le_u32(tail, end_at - tail_at + 12)
        if count == 0xFFFF or directory_at == 0xFFFFFFFF or directory_size == 0xFFFFFFFF {
            let record = self._read_zip64_end(end_at, tail, tail_at)
            count = record[0]
            directory_size = record[1]
            directory_at = record[2]
        }
        let directory = source.at(directory_at, directory_size)
        var at = 0
        for _index in 0..count {
            at = self._read_directory_entry(directory, at)
        }
    }

    #: Whether the archive holds a member under this exact name.
    pub fn has(self, name: str) -> bool { return self.entries.contains(name) }

    #: One member's bytes, decompressed if it was deflated.
    pub fn read(self, name: str) -> bytes {
        if not self.entries.contains(name) {
            throw ValueError(f"the archive has no member named '{name}'")
        }
        let entry = self.entries[name]
        let header = self.source.at(entry.header_at, 30)
        if le_u32(header, 0) != 0x04034B50 {
            throw ValueError(f"member '{name}' does not start with a local file header")
        }
        let begin = entry.header_at + 30 + le_u16(header, 26) + le_u16(header, 28)
        if entry.method == 0 {
            let payload = self.source.at(begin, entry.size)
            if __prim__.crc32(payload) != entry.crc {
                throw ValueError(f"member '{name}' does not match its recorded checksum")
            }
            return payload
        }
        if entry.method == 8 {
            return inflate_member(
                self.source.at(begin, entry.stored_size),
                entry.size,
                entry.crc,
                name
            )
        }
        throw ValueError(
            f"member '{name}' uses compression method {entry.method}; this reader " +
                "handles stored (0) and deflate (8)"
        )
    }

    #: The end-of-central-directory record: where it starts, the tail it was
    #: found in, and where that tail starts.
    #:
    #: The record is the last thing in the file except for an optional comment
    #: of up to 64 KiB, so the only way to it is backwards from the end looking
    #: for the signature. The tail is handed back with it because everything
    #: else the end of the archive holds is inside the same window.
    fn _find_end_record(self) -> tuple[int, bytes, int] {
        if self.source.total < 22 {
            throw ValueError("this file is too short to be a ZIP archive")
        }
        let span = min(self.source.total, 22 + 65535 + 20)
        let tail_at = self.source.total - span
        let tail = self.source.at(tail_at, span)
        var at = span - 22
        while at >= 0 {
            if le_u32(tail, at) == 0x06054B50 { return (tail_at + at, tail, tail_at) }
            at -= 1
        }
        throw ValueError("this file has no ZIP end-of-central-directory record")
    }

    #: The entry count, directory size and directory offset from the zip64 end
    #: record, which is where an archive past 4 GiB or 65535 members keeps them.
    fn _read_zip64_end(self, end_at: int, tail: bytes, tail_at: int) -> list[int] {
        let locator_at = end_at - 20
        if locator_at < tail_at or le_u32(tail, locator_at - tail_at) != 0x07064B50 {
            throw ValueError("this archive claims zip64 sizes but has no zip64 locator")
        }
        let record_at = le_u64(tail, locator_at - tail_at + 8)
        if record_at + 56 > self.source.total {
            throw ValueError("this archive's zip64 locator points past the end of the file")
        }
        let record = self.source.at(record_at, 56)
        if le_u32(record, 0) != 0x06064B50 {
            throw ValueError("this archive's zip64 locator does not point at a zip64 end record")
        }
        return [le_u64(record, 32), le_u64(record, 40), le_u64(record, 48)]
    }

    #: Read one central directory entry out of the directory block and return
    #: where the next one starts in it.
    fn _read_directory_entry(self, directory: bytes, at: int) -> int {
        if at + 46 > directory.len() or le_u32(directory, at) != 0x02014B50 {
            throw ValueError("this archive's central directory is truncated or misaligned")
        }
        let name_length = le_u16(directory, at + 28)
        let extra_length = le_u16(directory, at + 30)
        let comment_length = le_u16(directory, at + 32)
        if at + 46 + name_length + extra_length + comment_length > directory.len() {
            throw ValueError("this archive's central directory is truncated")
        }
        let name = directory.slice(at + 46, at + 46 + name_length).decode()
        var stored_size = le_u32(directory, at + 20)
        var size = le_u32(directory, at + 24)
        var header_at = le_u32(directory, at + 42)
        #: The zip64 extra field carries replacements only for the fields that
        #: were parked at the sentinel, in this fixed order and no other, so
        #: which ones are present has to be inferred from the 32-bit slots.
        if size == 0xFFFFFFFF or stored_size == 0xFFFFFFFF or header_at == 0xFFFFFFFF {
            var walk = at + 46 + name_length
            let extra_end = walk + extra_length
            while walk + 4 <= extra_end {
                let tag = le_u16(directory, walk)
                let length = le_u16(directory, walk + 2)
                if tag == 0x0001 {
                    var field = walk + 4
                    if size == 0xFFFFFFFF {
                        size = le_u64(directory, field)
                        field += 8
                    }
                    if stored_size == 0xFFFFFFFF {
                        stored_size = le_u64(directory, field)
                        field += 8
                    }
                    if header_at == 0xFFFFFFFF { header_at = le_u64(directory, field) }
                }
                walk += 4 + length
            }
        }
        let entry = ZipEntry(
            name,
            le_u16(directory, at + 10),
            stored_size,
            size,
            le_u32(directory, at + 16),
            header_at
        )
        self.names.push(name)
        self.entries[name] = entry
        return at + 46 + name_length + extra_length + comment_length
    }
}

#: A deflated ZIP member, through zlib.
#:
#: `__prim__.inflate` autodetects a zlib or a gzip header but will not take the
#: bare deflate stream a ZIP member holds, so the stream is dressed as gzip on
#: the way in. That is not a workaround so much as a free integrity check: a
#: gzip trailer is a CRC and a length, the central directory has just supplied
#: both, and zlib refuses the member if either disagrees.
fn inflate_member(payload: bytes, size: int, crc: int, name: str) -> bytes {
    var wrapper = [0x1F, 0x8B, 0x08, 0, 0, 0, 0, 0, 0, 0xFF]
    var trailer: list[int] = []
    for shift in 0..4 { trailer.push(crc >> 8 * shift & 0xFF) }
    for shift in 0..4 { trailer.push(size >> 8 * shift & 0xFF) }
    let expanded = __prim__.inflate(bytes(wrapper).concat(payload).concat(bytes(trailer)), size)
    if expanded.len() != size {
        throw ValueError(
            f"member '{name}' expanded to {expanded.len()} bytes, and the directory says {size}"
        )
    }
    return expanded
}

#: One tensor read out of a checkpoint.
#:
#: The elements are kept as the raw little-endian bytes torch wrote rather than
#: as decoded numbers, because nothing narrower holds all of them: an `int64`
#: near its limits does not survive a double, and a `float64` does not survive
#: a float. `floats` and `ints` decode on demand, and `data` is there for a
#: caller that would rather reinterpret the bytes than go round through either.
#:
#: The bytes are this tensor's own row-major order, not the storage's. A
#: transposed or sliced tensor is gathered through its strides when it is read,
#: so nothing downstream has to know it was ever a view.
pub class TorchTensor: Printable {
    #: Dimensions, outermost first. Empty for a 0-dimensional tensor.
    pub let shape: list[int]
    #: One of `float32`, `float64`, `float16`, `bfloat16`, `int64`, `int32`,
    #: `int16`, `int8`, `uint8`, `bool`.
    pub let dtype: str
    #: `numel() * item_size()` bytes, row-major, little-endian.
    pub let data: bytes

    pub fn init(self, shape: list[int], dtype: str, data: bytes) {
        let width = dtype_size(dtype)
        var count = 1
        for dimension in shape {
            if dimension < 0 {
                throw ValueError(f"a tensor cannot have shape {shape}")
            }
            count *= dimension
        }
        if data.len() != count * width {
            throw ValueError(
                f"a {dtype}{shape} tensor holds {count * width} bytes, and {data.len()} arrived"
            )
        }
        self.shape = shape
        self.dtype = dtype
        self.data = data
    }

    #: How many elements the tensor holds. A 0-dimensional tensor holds one.
    pub fn numel(self) -> int {
        var count = 1
        for dimension in self.shape { count *= dimension }
        return count
    }

    #: Bytes per element.
    pub fn item_size(self) -> int { return dtype_size(self.dtype) }

    #: Every element as a float, row-major.
    #:
    #: Integer dtypes widen on the way out, which is exact up to 2^53 and not
    #: past it; `ints` is the exact reading of those.
    pub fn floats(self) -> list[float] {
        let count = self.numel()
        var out: list[float] = []
        if self.dtype == "float32" {
            for index in 0..count { out.push(f32_from_bits(le_u32(self.data, index * 4))) }
            return out
        }
        if self.dtype == "float64" {
            for index in 0..count {
                out.push(
                    f64_from_halves(le_u32(self.data, index * 8 + 4), le_u32(self.data, index * 8))
                )
            }
            return out
        }
        if self.dtype == "float16" {
            for index in 0..count { out.push(f16_from_bits(le_u16(self.data, index * 2))) }
            return out
        }
        if self.dtype == "bfloat16" {
            #: A bfloat16 is the top half of a float32 and nothing else, so
            #: putting the halves back is the whole conversion.
            for index in 0..count { out.push(f32_from_bits(le_u16(self.data, index * 2) << 16)) }
            return out
        }
        for value in self.ints() { out.push(float(value)) }
        return out
    }

    #: Every element as an int, row-major. Booleans come back as 0 and 1.
    pub fn ints(self) -> list[int] {
        let count = self.numel()
        var out: list[int] = []
        if self.dtype == "bool" {
            for index in 0..count { out.push(self.data[index] == 0 ? 0 : 1) }
            return out
        }
        if self.dtype == "uint8" {
            for index in 0..count { out.push(self.data[index]) }
            return out
        }
        if self.dtype == "int8" or
            self.dtype == "int16" or
            self.dtype == "int32" or
            self.dtype == "int64" {
            let width = self.item_size()
            for index in 0..count { out.push(le_int(self.data, index * width, width, true)) }
            return out
        }
        throw ValueError(f"a {self.dtype} tensor has no exact integer reading; call floats()")
    }

    #: Every element of a `bool` tensor.
    pub fn bools(self) -> list[bool] {
        if self.dtype != "bool" {
            throw ValueError(f"a {self.dtype} tensor is not a mask; call floats() or ints()")
        }
        var out: list[bool] = []
        for index in 0..self.numel() { out.push(self.data[index] != 0) }
        return out
    }

    pub fn to_str(self) -> str { return f"TorchTensor({self.dtype}, {self.shape})" }
}

#: One tensor storage: a flat run of elements that any number of tensors may
#: be views onto.
class Storage {
    pub let dtype: str
    pub let data: bytes
    pub let count: int

    pub fn init(self, dtype: str, data: bytes, count: int) {
        self.dtype = dtype
        self.data = data
        self.count = count
    }
}

#: A global the pickle named and the whitelist allowed.
class PickleGlobal {
    pub let module_name: str
    pub let name: str

    pub fn init(self, module_name: str, name: str) {
        self.module_name = module_name
        self.name = name
    }

    pub fn full_name(self) -> str { return self.module_name + "." + self.name }
}

#: A pickle tuple.
#:
#: Jaithon's own tuples are fixed-arity and a pickle's are not, so they travel
#: as their items behind a marker class. Keeping them apart from a list is not
#: pedantry: a persistent id is a tuple, and telling one from a list of two
#: tensors is how the reader knows which it is looking at.
class PickleTuple {
    pub let items: list[any]

    pub fn init(self, items: list[any]) { self.items = items }
}

#: The archive plus the prefix every member of it sits under.
class Checkpoint {
    pub let archive: ZipArchive
    pub let root: str
    pub var loaded: dict[str, Storage]

    pub fn init(self, archive: ZipArchive, root: str) {
        self.archive = archive
        self.root = root
        self.loaded = {}
    }

    #: The storage a persistent id names, read once however many tensors are
    #: views onto it. Caching is not only for speed: a shared storage that was
    #: deflated would otherwise be expanded again for every view of it.
    pub fn storage(self, key: str, dtype: str, count: int) -> Storage {
        if self.loaded.contains(key) {
            let known = self.loaded[key]
            if known.dtype != dtype {
                throw ValueError(
                    f"storage '{key}' is read as {known.dtype} in one place and {dtype} in another"
                )
            }
            return known
        }
        let member = self.root + "data/" + key
        if not self.archive.has(member) {
            throw ValueError(f"the checkpoint names a storage '{key}' that it does not carry")
        }
        let raw = self.archive.read(member)
        let wanted = count * dtype_size(dtype)
        if raw.len() != wanted {
            throw ValueError(
                f"storage '{key}' should hold {wanted} bytes for {count} {dtype} elements, " +
                    f"and holds {raw.len()}"
            )
        }
        let made = Storage(dtype, raw, count)
        self.loaded[key] = made
        return made
    }
}

const OP_MARK = 0x28
const OP_EMPTY_TUPLE = 0x29
const OP_STOP = 0x2E
const OP_POP = 0x30
const OP_POP_MARK = 0x31
const OP_BINBYTES = 0x42
const OP_SHORT_BINBYTES = 0x43
const OP_BINFLOAT = 0x47
const OP_BININT = 0x4A
const OP_BININT1 = 0x4B
const OP_BININT2 = 0x4D
const OP_NONE = 0x4E
const OP_PERSID = 0x50
const OP_BINPERSID = 0x51
const OP_REDUCE = 0x52
const OP_BINUNICODE = 0x58
const OP_EMPTY_LIST = 0x5D
const OP_APPEND = 0x61
const OP_BUILD = 0x62
const OP_GLOBAL = 0x63
const OP_DICT = 0x64
const OP_APPENDS = 0x65
const OP_BINGET = 0x68
const OP_LONG_BINGET = 0x6A
const OP_LIST = 0x6C
const OP_BINPUT = 0x71
const OP_LONG_BINPUT = 0x72
const OP_SETITEM = 0x73
const OP_TUPLE = 0x74
const OP_SETITEMS = 0x75
const OP_EMPTY_DICT = 0x7D
const OP_PROTO = 0x80
const OP_NEWOBJ = 0x81
const OP_TUPLE1 = 0x85
const OP_TUPLE2 = 0x86
const OP_TUPLE3 = 0x87
const OP_NEWTRUE = 0x88
const OP_NEWFALSE = 0x89
const OP_LONG1 = 0x8A
const OP_LONG4 = 0x8B
const OP_SHORT_BINUNICODE = 0x8C
const OP_BINUNICODE8 = 0x8D
const OP_BINBYTES8 = 0x8E
const OP_EMPTY_SET = 0x8F
const OP_ADDITEMS = 0x90
const OP_FROZENSET = 0x91
const OP_NEWOBJ_EX = 0x92
const OP_STACK_GLOBAL = 0x93
const OP_MEMOIZE = 0x94
const OP_FRAME = 0x95

#: Whether a pickle may name this global.
#:
#: Everything on this list either builds a container or rebuilds a tensor, and
#: none of it reaches outside the file. Nothing else is reachable, and there is
#: no way for a caller to add to it -- a whitelist a caller can widen is not a
#: whitelist.
fn allowed_global(module_name: str, name: str) -> bool {
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

fn refuse_global(module_name: str, name: str) -> void {
    throw ValueError(
        f"refusing the pickled global {module_name}.{name}: a checkpoint may name only " +
            "torch._utils._rebuild_tensor, _rebuild_tensor_v2 and _rebuild_parameter, the " +
            "torch storage classes, collections.OrderedDict, and the container builtins. " +
            "Resolving anything else would let the file choose what this program runs."
    )
}

#: The items of a pickle tuple, or `null` if this is not one.
fn tuple_items(value: any) -> list[any]? {
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
fn rebuild_tensor(args: list[any], named: str) -> TorchTensor {
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

#: The restricted pickle machine.
#:
#: It is a stack machine like the real one, with the same memo and the same
#: marks, and it stops at the same `STOP`. What it does not have is a way to
#: reach a name the whitelist did not allow, and that is the only difference
#: that matters.
class Unpickler {
    pub let data: bytes
    pub let source: Checkpoint
    pub var at: int
    pub var stack: list[any]
    pub var marks: list[int]
    pub var memo: dict[int, any]

    pub fn init(self, data: bytes, source: Checkpoint) {
        self.data = data
        self.source = source
        self.at = 0
        self.stack = []
        self.marks = []
        self.memo = {}
    }

    #: Interpret the stream and return whatever `STOP` left on the stack.
    pub fn run(self) -> any {
        loop {
            if self.at >= self.data.len() {
                throw ValueError("the pickle ended before its STOP")
            }
            let opcode = self.data[self.at]
            self.at += 1
            if opcode == OP_STOP { return self._pop() }
            self._step(opcode)
        }
    }

    fn _step(self, opcode: int) -> void {
        if opcode == OP_BINPUT {
            self.memo[self._take_u8()] = self._peek()
            return
        }
        if opcode == OP_BINGET {
            self.stack.push(self._recall(self._take_u8()))
            return
        }
        if opcode == OP_BININT1 {
            self.stack.push(self._take_u8())
            return
        }
        if opcode == OP_BINUNICODE {
            self.stack.push(self._take_text(self._take_u32()))
            return
        }
        if opcode == OP_MARK {
            self.marks.push(self.stack.len())
            return
        }
        if opcode == OP_TUPLE {
            self.stack.push(PickleTuple(self._pop_to_mark()))
            return
        }
        if opcode == OP_REDUCE {
            self._reduce()
            return
        }
        if opcode == OP_BINPERSID {
            self.stack.push(self._resolve_persistent_id(self._pop()))
            return
        }
        if opcode == OP_TUPLE1 {
            self._tuple_of(1)
            return
        }
        if opcode == OP_TUPLE2 {
            self._tuple_of(2)
            return
        }
        if opcode == OP_TUPLE3 {
            self._tuple_of(3)
            return
        }
        if opcode == OP_EMPTY_DICT {
            self.stack.push(self._new_dict())
            return
        }
        if opcode == OP_EMPTY_LIST {
            self.stack.push(self._new_list())
            return
        }
        if opcode == OP_EMPTY_TUPLE {
            self.stack.push(PickleTuple([]))
            return
        }
        if opcode == OP_NEWFALSE {
            self.stack.push(false)
            return
        }
        if opcode == OP_NEWTRUE {
            self.stack.push(true)
            return
        }
        if opcode == OP_NONE {
            self.stack.push(null)
            return
        }
        if opcode == OP_SETITEMS {
            self._set_items(self._pop_to_mark())
            return
        }
        if opcode == OP_SETITEM {
            let value = self._pop()
            let key = self._pop()
            self._set_items([key, value])
            return
        }
        if opcode == OP_APPENDS {
            self._append_all(self._pop_to_mark())
            return
        }
        if opcode == OP_APPEND {
            self._append_all([self._pop()])
            return
        }
        if opcode == OP_GLOBAL {
            self._global(self._take_line(), self._take_line())
            return
        }
        if opcode == OP_STACK_GLOBAL {
            let name = self._pop()
            let module_name = self._pop()
            if not name is str or not module_name is str {
                throw ValueError("STACK_GLOBAL was not handed two strings")
            }
            self._global(module_name, name)
            return
        }
        if opcode == OP_MEMOIZE {
            self.memo[self.memo.len()] = self._peek()
            return
        }
        if opcode == OP_LONG_BINPUT {
            self.memo[self._take_u32()] = self._peek()
            return
        }
        if opcode == OP_LONG_BINGET {
            self.stack.push(self._recall(self._take_u32()))
            return
        }
        if opcode == OP_BININT2 {
            self.stack.push(self._take_u16())
            return
        }
        if opcode == OP_BININT {
            self.stack.push(self._take_i32())
            return
        }
        if opcode == OP_SHORT_BINUNICODE {
            self.stack.push(self._take_text(self._take_u8()))
            return
        }
        if opcode == OP_BINUNICODE8 {
            self.stack.push(self._take_text(self._take_u64()))
            return
        }
        if opcode == OP_SHORT_BINBYTES {
            self.stack.push(self._take_bytes(self._take_u8()))
            return
        }
        if opcode == OP_BINBYTES {
            self.stack.push(self._take_bytes(self._take_u32()))
            return
        }
        if opcode == OP_BINBYTES8 {
            self.stack.push(self._take_bytes(self._take_u64()))
            return
        }
        if opcode == OP_BINFLOAT {
            #: Alone among the binary opcodes, this one is big-endian.
            let at = self._advance(8)
            self.stack.push(f64_from_halves(be_u32(self.data, at), be_u32(self.data, at + 4)))
            return
        }
        if opcode == OP_LONG1 {
            self._long(self._take_u8())
            return
        }
        if opcode == OP_LONG4 {
            self._long(self._take_i32())
            return
        }
        if opcode == OP_DICT {
            self._dict_from_mark()
            return
        }
        if opcode == OP_LIST {
            let items = self._pop_to_mark()
            var made = self._new_list()
            for item in items { made.push(item) }
            self.stack.push(made)
            return
        }
        if opcode == OP_EMPTY_SET {
            self.stack.push(set([]))
            return
        }
        if opcode == OP_FROZENSET {
            self.stack.push(set(self._pop_to_mark()))
            return
        }
        if opcode == OP_ADDITEMS {
            #: Sets never carry weights, so the items are taken and dropped
            #: rather than folded into a set that nothing downstream reads.
            let _items = self._pop_to_mark()
            return
        }
        if opcode == OP_BUILD {
            self._build()
            return
        }
        if opcode == OP_PROTO {
            let version = self._take_u8()
            if version > 5 {
                throw ValueError(
                    f"this pickle is protocol {version}, and 5 is the highest there is"
                )
            }
            return
        }
        if opcode == OP_FRAME {
            let _length = self._take_u64()
            return
        }
        if opcode == OP_POP {
            let _dropped = self._pop()
            return
        }
        if opcode == OP_POP_MARK {
            let _dropped = self._pop_to_mark()
            return
        }
        if opcode == OP_PERSID {
            self.stack.push(self._resolve_persistent_id(self._take_line()))
            return
        }
        if opcode == OP_NEWOBJ or opcode == OP_NEWOBJ_EX {
            let _arguments = self._pop()
            if opcode == OP_NEWOBJ_EX { let _keywords = self._pop() }
            let target = self._pop()
            var named = "an object"
            if target is PickleGlobal {
                let global_target: PickleGlobal = target
                named = global_target.full_name()
            }
            throw ValueError(
                f"this pickle constructs {named} directly, which this reader will not do; " +
                    "only tensors and containers are rebuilt"
            )
        }
        throw ValueError(
            f"pickle opcode 0x{opcode} at byte {self.at - 1} is one this reader does not implement"
        )
    }

    fn _advance(self, count: int) -> int {
        if self.at + count > self.data.len() {
            throw ValueError("the pickle ends in the middle of an operand")
        }
        let begin = self.at
        self.at += count
        return begin
    }

    fn _take_u8(self) -> int { return self.data[self._advance(1)] }

    fn _take_u16(self) -> int { return le_u16(self.data, self._advance(2)) }

    fn _take_u32(self) -> int { return le_u32(self.data, self._advance(4)) }

    fn _take_i32(self) -> int { return le_int(self.data, self._advance(4), 4, true) }

    fn _take_u64(self) -> int { return le_u64(self.data, self._advance(8)) }

    fn _take_bytes(self, length: int) -> bytes {
        let begin = self._advance(length)
        return self.data.slice(begin, begin + length)
    }

    fn _take_text(self, length: int) -> str { return self._take_bytes(length).decode() }

    #: Everything up to the next newline, which is how the text opcodes end.
    fn _take_line(self) -> str {
        var end = self.at
        while end < self.data.len() and self.data[end] != 0x0A { end += 1 }
        if end >= self.data.len() {
            throw ValueError("the pickle ends in the middle of a line-terminated operand")
        }
        let text = self.data.slice(self.at, end).decode()
        self.at = end + 1
        return text
    }

    #: `LONG1` and `LONG4`: a little-endian two's complement integer of any
    #: width. Past eight bytes there is nothing here to hold it, and saying so
    #: is better than folding it to something wrong.
    fn _long(self, length: int) -> void {
        if length > 8 {
            throw ValueError(
                f"this pickle holds a {length}-byte integer, which is wider than an int"
            )
        }
        self.stack.push(le_int(self.data, self._advance(length), length, true))
    }

    fn _new_dict(self) -> dict[any, any] {
        var made: dict[any, any] = {}
        return made
    }

    fn _new_list(self) -> list[any] {
        var made: list[any] = []
        return made
    }

    fn _peek(self) -> any {
        if self.stack.len() == 0 { throw ValueError("the pickle read an empty stack") }
        return self.stack[self.stack.len() - 1]
    }

    fn _pop(self) -> any {
        if self.stack.len() == 0 { throw ValueError("the pickle popped an empty stack") }
        return self.stack.pop()
    }

    fn _recall(self, slot: int) -> any {
        if not self.memo.contains(slot) {
            throw ValueError(f"the pickle read memo slot {slot}, which nothing has filled")
        }
        return self.memo[slot]
    }

    fn _pop_to_mark(self) -> list[any] {
        if self.marks.len() == 0 { throw ValueError("the pickle closed a mark it never opened") }
        let floor = self.marks.pop()
        if floor > self.stack.len() { throw ValueError("the pickle's stack fell below its mark") }
        var taken: list[any] = []
        for index in floor..self.stack.len() { taken.push(self.stack[index]) }
        while self.stack.len() > floor { let _dropped = self.stack.pop() }
        return taken
    }

    fn _tuple_of(self, count: int) -> void {
        var items: list[any] = []
        for _index in 0..count { items.push(self._pop()) }
        var ordered: list[any] = []
        var index = items.len() - 1
        while index >= 0 {
            ordered.push(items[index])
            index -= 1
        }
        self.stack.push(PickleTuple(ordered))
    }

    fn _dict_from_mark(self) -> void {
        let flat = self._pop_to_mark()
        var made = self._new_dict()
        self.stack.push(made)
        self._set_items(flat)
    }

    fn _set_items(self, flat: list[any]) -> void {
        if flat.len() % 2 != 0 {
            throw ValueError("the pickle set an odd number of dictionary halves")
        }
        let target = self._peek()
        if not target is dict {
            throw ValueError("the pickle set items on something that is not a dictionary")
        }
        var into: dict[any, any] = target
        var index = 0
        while index < flat.len() {
            into[flat[index]] = flat[index + 1]
            index += 2
        }
    }

    fn _append_all(self, items: list[any]) -> void {
        let target = self._peek()
        if not target is list {
            throw ValueError("the pickle appended to something that is not a list")
        }
        var into: list[any] = target
        for item in items { into.push(item) }
    }

    fn _global(self, module_name: str, name: str) -> void {
        if not allowed_global(module_name, name) { refuse_global(module_name, name) }
        self.stack.push(PickleGlobal(module_name, name))
    }

    #: `BUILD` restores an object's state, which for pickle means setting its
    #: attributes -- not its items.
    #:
    #: The one thing here that ever carries state is the `OrderedDict` a state
    #: dict is, and the state it carries is its `_metadata` attribute, holding
    #: the module versions torch writes beside the weights. Those are
    #: attributes of the mapping and not entries in it, so folding them in
    #: would invent keys the checkpoint does not have. There is nowhere here to
    #: keep an attribute, so they are dropped; anything else with state is a
    #: real object, and rebuilding one of those is what this reader will not do.
    fn _build(self) -> void {
        let _state = self._pop()
        if self._peek() is dict { return }
        throw ValueError("this pickle restores object state this reader cannot rebuild")
    }

    fn _reduce(self) -> void {
        let arguments = self._pop()
        let target = self._pop()
        if not target is PickleGlobal {
            throw ValueError("the pickle called something that is not a whitelisted global")
        }
        let called: PickleGlobal = target
        let items = tuple_items(arguments)
        if items is null {
            throw ValueError(f"{called.full_name()} was called with something that is not a tuple")
        }
        let args: list[any] = items
        self.stack.push(self._apply(called, args))
    }

    fn _apply(self, called: PickleGlobal, args: list[any]) -> any {
        let full = called.full_name()
        if full == "torch._utils._rebuild_tensor_v2" or full == "torch._utils._rebuild_tensor" {
            return rebuild_tensor(args, full)
        }
        if full == "torch._utils._rebuild_parameter" {
            if args.len() == 0 or not args[0] is TorchTensor {
                throw ValueError("_rebuild_parameter was not handed a tensor to wrap")
            }
            return args[0]
        }
        if full == "collections.OrderedDict" or
            full == "builtins.dict" or
            full == "__builtin__.dict" {
            if args.len() != 0 {
                throw ValueError(
                    f"{full} was called with arguments, which this reader cannot apply"
                )
            }
            return self._new_dict()
        }
        if full == "builtins.list" or full == "__builtin__.list" {
            if args.len() == 0 { return self._new_list() }
            return self._sequence_of(args[0], full)
        }
        if full == "builtins.tuple" or full == "__builtin__.tuple" {
            if args.len() == 0 { return PickleTuple([]) }
            return PickleTuple(self._sequence_of(args[0], full))
        }
        if full == "builtins.set" or
            full == "builtins.frozenset" or
            full == "__builtin__.set" or
            full == "__builtin__.frozenset" {
            if args.len() == 0 { return set([]) }
            return set(self._sequence_of(args[0], full))
        }
        throw ValueError(f"{full} is on the whitelist as a value, and this pickle calls it")
    }

    fn _sequence_of(self, value: any, full: str) -> list[any] {
        let items = tuple_items(value)
        if items is not null {
            let taken: list[any] = items
            return taken
        }
        if value is list {
            let listed: list[any] = value
            return listed
        }
        throw ValueError(f"{full} was handed something that is not a sequence")
    }

    #: Resolve a persistent id, which for torch is always a storage.
    #:
    #: The id is the tuple `("storage", storage_type, key, location, numel)`,
    #: and it stands in for the archive member `data/<key>`. Nothing else is a
    #: persistent id in a checkpoint, so nothing else is accepted.
    fn _resolve_persistent_id(self, ident: any) -> any {
        let items = tuple_items(ident)
        if items is null {
            throw ValueError("this checkpoint's persistent ids are not the tuples torch writes")
        }
        let parts: list[any] = items
        if parts.len() < 5 or not parts[0] is str {
            throw ValueError("this checkpoint's persistent ids are not the tuples torch writes")
        }
        let kind: str = parts[0]
        if kind != "storage" {
            throw ValueError(
                f"this checkpoint holds a '{kind}' persistent id, and only storages are read"
            )
        }
        if not parts[1] is PickleGlobal {
            throw ValueError("a persistent id names a storage class this reader did not resolve")
        }
        let storage_class: PickleGlobal = parts[1]
        if storage_class.module_name != "torch" or not STORAGE_DTYPES.contains(storage_class.name) {
            throw ValueError(
                f"a persistent id names {storage_class.full_name()}, which is not a torch storage class"
            )
        }
        var key = ""
        if parts[2] is str {
            key = parts[2]
        } elif parts[2] is int {
            key = str(parts[2])
        } else {
            throw ValueError(
                "a persistent id names a storage key that is neither text nor a number"
            )
        }
        if not parts[4] is int {
            throw ValueError(f"the persistent id for storage '{key}' has no element count")
        }
        let count: int = parts[4]
        return self.source.storage(key, STORAGE_DTYPES[storage_class.name], count)
    }
}

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
