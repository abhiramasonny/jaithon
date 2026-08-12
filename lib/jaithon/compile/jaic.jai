#: Read and write the `.jaic` container of `spec/BYTECODE.md` §4-§7, and
#: disassemble what comes out of it.
#:
#: This is the Jaithon half of `src/vm/bytecode/serialize.c` and must agree with it byte
#: for byte, so the layout below is normative rather than convenient:
#:
#: - Every integer is little-endian; a function record's pool is the chunk's
#:   pool plus whatever the record names by index that the chunk didn't
#:   already hold, appended after the chunk's own constants so a `u24` operand
#:   already in the code keeps pointing at the same value.
#: - A class spec's `superNameIndex` of 0 means "no superclass" (`OP_INHERIT`
#:   links the subclass).
#: - The line table is a *byte* length followed by 12-byte
#:   `{offset, span, spanEnd}` rows; it is debug data, ignored when comparing.
#: - Nothing is trusted: a short read, an out-of-range index or a bad CRC all
#:   mean "recompile" — `read_module` returns null, never a half-built mod.
#:
#: ```
#: crc32([0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39])   # 3421780262
#: float_bits(1.0)                                                 # 4607182418800017408
#: bits_to_float(float_bits(-2.5))                                 # -2.5
#: ```
from jaithon.compile.emit import Chunk, ClassSpec, Const, ConstTag, ExceptionEntry
from jaithon.compile.emit import FieldSpec, FuncProto, LineEntry, MethodSpec
from jaithon.compile.emit import op_name, op_operand_bytes
from std.str import repeat

#: The four bytes every `.jaic` file starts with.
pub let MAGIC: list[int] = [0x4A, 0x41, 0x49, 0x43]

#: Container format version; a mismatch means recompile.
#:
#: Must equal `JAIC_VERSION` in `src/vm/bytecode/serialize.h` — one number split across
#: two languages.
pub const VERSION: int = 12

#: Oldest container this reader accepts. Must equal `JAIC_VERSION_MIN` in
#: `src/vm/bytecode/serialize.h`.
#:
#: A range, not an equality: across a format bump the seed's images were written
#: by the previous generation, and refusing them leaves the tree with no front
#: end and no way to build one. `make reseed` twice closes the gap — generation
#: one still runs the old compiler out of the old seed.
pub const VERSION_MIN: int = 11

#: First container version whose line table is LTV1 (delta + LEB128). Below it,
#: the table is `_LINE_ROW`-sized absolute rows.
pub const VERSION_LTV1: int = 12

#: `JAI_COMPILER_VERSION`, recorded so that a new compiler ignores old caches.
#:
#: Must equal `JAI_COMPILER_VERSION` in `src/common/common.h`.
pub const COMPILER_VERSION: int = 19

#: The `buildId` the running binary stamps into a `.jaic` and demands back.
#:
#: Keyed on a hash of the C sources (`build/jai_build_id.h`) rather than
#: written down here, since this compiler never reads C. An image missing it
#: is rejected by `src/vm/bytecode/serialize.c` with a header that otherwise validates
#: in full.
#:
#: ```
#: build_id() == build_id()      # true
#: build_id() >= 0               # true
#: ```
pub fn build_id() -> int { return __prim__.jaic_build_id() }

#: Header flag: debug information is present.
pub const FLAG_DEBUG: int = 1

#: Header flag: built with `--release`, so assertions were stripped.
pub const FLAG_RELEASE: int = 2

#: Where the optimisation level lives in the container flags, matching
#: `JAIC_FLAG_OPT_SHIFT`/`JAIC_FLAG_OPT_MASK` in `src/vm/bytecode/serialize.h`.
#:
#: Must match what `cacheFlagsFor` expects exactly: an image missing these
#: bits silently fails the cache check and falls back to the seed, so a
#: compiler edit can appear to never take effect.
pub const FLAG_OPT_SHIFT: int = 3
pub const FLAG_OPT_MASK: int = 3 << 3

#: `flags` for a build at `opt_level`, as the C loader expects them.
pub fn flags_for(release: bool, opt_level: int) -> int {
    var level = opt_level
    if level < 0 { level = 0 }
    if level > 3 { level = 3 }
    let base = if release { FLAG_RELEASE } else { FLAG_DEBUG }
    return base | level << FLAG_OPT_SHIFT & FLAG_OPT_MASK
}

#: The `nameIndex` that means "this record has no name".
pub const NO_NAME: int = 4294967295

#: Deepest nesting of function and tuple records the reader will follow.
pub const MAX_DEPTH: int = 64

#: Smallest file that could hold a header and a checksum.
pub const MIN_SIZE: int = 32

#: Bytes per line-table row before `VERSION_LTV1`: three absolute u32s.
const _LINE_ROW: int = 12
const _EXC_ROW: int = 16
const _CONST_LIMIT: int = 16777216
let _SIGN_BIT: int = -9223372036854775807 - 1
let _CRC_TABLE: list[int] = _build_crc_table()

fn _build_crc_table() -> list[int] {
    var table: list[int] = []
    for index in range(256) {
        var value = index
        for _bit in range(8) {
            value = if value & 1 == 1 { 0xEDB88320 ^ value >> 1 } else { value >> 1 }
        }
        table.push(value)
    }
    return table
}

#: CRC-32 (IEEE 802.3, reflected, polynomial `0xEDB88320`) of `data` — the
#: checksum held in the container's last four bytes, computed over everything
#: before them.
#:
#: ```
#: crc32([])          # 0
#: crc32([0x61])      # 3904355907
#: ```
pub fn crc32(data: list[int], length: int = -1) -> int {
    let stop = if length < 0 or length > data.len() { data.len() } else { length }
    var crc = 0xFFFFFFFF
    for index in range(stop) {
        crc = _CRC_TABLE[(crc ^ data[index]) & 0xFF] ^ crc >> 8
    }
    return crc ^ 0xFFFFFFFF
}

#: FNV-1a 64 of `text`, which is the container's `srcHash`.
#:
#: Signed 64-bit with the same bit pattern the C side writes, since the
#: multiply wraps.
#:
#: ```
#: source_hash("") == source_hash("")     # true
#: source_hash("a") != source_hash("b")   # true
#: ```
pub fn source_hash(text: str) -> int {
    var hash = -3750763034362895579  # 14695981039346656037 as i64
    for byte in utf8_bytes(text) {
        hash = (hash ^ byte) *% 1099511628211
    }
    return hash
}

#: The UTF-8 bytes of `text`, as integers in 0..255.
#:
#: ```
#: utf8_bytes("hi")      # [104, 105]
#: ```
pub fn utf8_bytes(text: str) -> list[int] {
    let encoded = __prim__.str_encode(text)
    var out: list[int] = []
    for index in range(encoded.len()) { out.push(encoded[index]) }
    return out
}

#: Decode UTF-8 `data` back into a string.
#:
#: Raises `ValueError` on invalid UTF-8: a pool that can't decode is corrupt,
#: not lossy.
pub fn utf8_text(data: list[int]) -> str {
    return __prim__.str_decode(__prim__.bytes_new(data))
}

#: The IEEE-754 binary64 bit pattern of `value`, as a signed 64-bit integer.
#:
#: ```
#: float_bits(0.0)       # 0
#: float_bits(2.0)       # 4611686018427387904
#: ```
pub fn float_bits(value: float) -> int {
    if value != value { return 0x7FF8000000000000 }  # a quiet NaN

    var magnitude = value
    var sign = 0
    if _is_negative(value) {
        sign = 1
        magnitude = -value
    }
    if magnitude > 1.7976931348623157e308 {
        return _assemble(sign, 2047, 0)  # infinity
    }
    if magnitude == 0.0 { return _assemble(sign, 0, 0) }

    let split = __prim__.f64_frexp(magnitude)  # (m in [0.5,1), exp)
    let exponent = split[1] + 1022
    if exponent <= 0 {
        # Subnormal: the value is fraction * 2**-1074 exactly.
        return _assemble(sign, 0, int(__prim__.f64_ldexp(magnitude, 1074)))
    }
    return _assemble(sign, exponent, int((split[0] * 2.0 - 1.0) * 4503599627370496.0))
}

#: The binary64 value whose bit pattern is `bits`.
#:
#: ```
#: bits_to_float(4607182418800017408)      # 1.0
#: bits_to_float(0)                        # 0.0
#: ```
pub fn bits_to_float(bits: int) -> float {
    let exponent = bits >> 52 & 2047
    let fraction = bits & 4503599627370495
    var value = 0.0
    if exponent == 0 {
        value = __prim__.f64_ldexp(float(fraction), -1074)
    } elif exponent == 2047 {
        value = if fraction == 0 { 1.0e308 * 10.0 } else { _nan() }
    } else {
        value = __prim__.f64_ldexp(float(fraction) + 4503599627370496.0, exponent - 1075)
    }
    if bits < 0 { return -value }
    return value
}

fn _nan() -> float {
    let infinity = 1.0e308 * 10.0
    return infinity - infinity
}

# atan2 is the only primitive that distinguishes -0.0 from 0.0: atan2(-0, -1)
# is -pi where atan2(+0, -1) is +pi.
fn _is_negative(value: float) -> bool {
    if value < 0.0 { return true }
    if value > 0.0 { return false }
    return __prim__.f64_atan2(value, -1.0) < 0.0
}

fn _assemble(sign: int, exponent: int, fraction: int) -> int {
    let low = (exponent & 2047) << 52 | fraction & 4503599627370495
    if sign == 1 { return low | _SIGN_BIT }
    return low
}

#: A growable little-endian byte buffer.
pub class Writer {
    #: The bytes written so far.
    pub var data: list[int]

    #: Create an empty buffer.
    pub fn init(self) { self.data = [] }

    #: Append one byte, keeping only its low eight bits.
    pub fn u8(self, value: int) -> void { self.data.push(value & 0xFF) }

    #: Append a little-endian `u16`.
    pub fn u16(self, value: int) -> void {
        self.data.push(value & 0xFF)
        self.data.push(value >> 8 & 0xFF)
    }

    #: Append a little-endian `u32`.
    pub fn u32(self, value: int) -> void {
        for shift in range(4) { self.data.push(value >> shift * 8 & 0xFF) }
    }

    #: Append a little-endian `u64`, sign included.
    pub fn u64(self, value: int) -> void {
        for shift in range(8) { self.data.push(value >> shift * 8 & 0xFF) }
    }

    #: Append the IEEE-754 bits of `value`.
    pub fn f64(self, value: float) -> void { self.u64(float_bits(value)) }

    #: Append raw bytes.
    pub fn bytes(self, values: list[int]) -> void {
        for value in values { self.data.push(value & 0xFF) }
    }
}

#: A bounds-checked cursor over a byte list.
#:
#: Every accessor checks bytes remaining first and sets `bad` instead of
#: raising, so a truncated file yields null once, not an exception per field.
pub class Reader {
    #: The buffer being read.
    pub let data: list[int]
    #: Index of the next byte.
    pub var position: int
    #: One past the last readable byte.
    pub var limit: int
    #: Sticky: set by the first read that ran short.
    pub var bad: bool
    #: Container version of the image being read.
    #:
    #: Carried here rather than passed down because exactly one section reads
    #: it — the line table, whose encoding changed at `VERSION_LTV1` — and
    #: threading it through `_read_function`, `_read_constants` and
    #: `_read_tuple` would touch several signatures to reach one branch.
    #: `VERSION` until the container header says otherwise.
    pub var version: int

    #: Read `data` up to `limit`, which defaults to its length.
    pub fn init(self, data: list[int], limit: int = -1) {
        self.data = data
        self.position = 0
        self.limit = if limit < 0 { data.len() } else { limit }
        self.bad = false
        self.version = VERSION
    }

    #: Bytes still readable.
    pub fn left(self) -> int {
        if self.bad or self.position > self.limit { return 0 }
        return self.limit - self.position
    }

    #: Skip `count` bytes, or mark the cursor bad.
    pub fn skip(self, count: int) -> bool {
        if self.bad or count < 0 or count > self.left() {
            self.bad = true
            return false
        }
        self.position += count
        return true
    }

    #: True when `count` records of `each` bytes still fit.
    pub fn fits(self, count: int, each: int) -> bool {
        if self.bad or count < 0 { return false }
        return count <= self.left() // (if each > 0 { each } else { 1 })
    }

    #: Read one byte, or 0 with `bad` set.
    pub fn u8(self) -> int {
        if not self.skip(1) { return 0 }
        return self.data[self.position - 1] & 0xFF
    }

    #: Read a little-endian `u16`.
    pub fn u16(self) -> int {
        let low = self.u8()
        return low | self.u8() << 8
    }

    #: Read a little-endian `u32`.
    pub fn u32(self) -> int {
        var value = 0
        for shift in range(4) {
            let byte = self.u8()
            value = value | byte << shift * 8
        }
        return value
    }

    #: Read a little-endian `u64` as a signed 64-bit integer.
    pub fn u64(self) -> int {
        var value = 0
        for shift in range(7) {
            let byte = self.u8()
            value = value | byte << shift * 8
        }
        let top = self.u8()
        value = value | (top & 0x7F) << 56
        if top >= 128 { value = value | _SIGN_BIT }
        return value
    }

    #: Read an IEEE-754 binary64.
    pub fn f64(self) -> float { return bits_to_float(self.u64()) }

    #: Read `count` raw bytes.
    pub fn bytes(self, count: int) -> list[int] {
        if not self.fits(count, 1) or not self.skip(count) {
            self.bad = true
            return []
        }
        var out: list[int] = []
        for index in range(count) { out.push(self.data[self.position - count + index]) }
        return out
    }
}

#: One `.jaic` file: the header fields, the mod pool and the mod body.
pub class Module {
    #: Source path, recorded for tracebacks.
    pub var path: str
    #: FNV-1a 64 of the source text; the cache is used only when it matches.
    pub var source_hash: int
    #: `FLAG_DEBUG` and `FLAG_RELEASE`.
    pub var flags: int
    #: Last-modified time of the source, informational only.
    pub var mtime: int
    #: Names this mod exports.
    pub var exports: list[str]
    #: Imported mod paths, with the alias each was bound to or null.
    pub var imports: list[Import]
    #: The mod body's function record.
    pub var body: FuncProto

    #: Describe a mod.
    pub fn init(self, path: str, source_hash: int, body: FuncProto, flags: int = FLAG_DEBUG) {
        self.path = path
        self.source_hash = source_hash
        self.flags = flags
        self.mtime = 0
        self.exports = []
        self.imports = []
        self.body = body
    }
}

#: One row of the container's import section.
pub class Import {
    #: The dotted mod path.
    pub let path: str
    #: The name it was bound to, or null when it kept its own.
    pub let alias: str?

    #: Record an import.
    pub fn init(self, path: str, alias: str? = null) {
        self.path = path
        self.alias = alias
    }
}

# The pool a record writes: chunk constants plus whatever the record names by
# index; appending never disturbs an existing u24 operand.
class Pool {
    pub var entries: list[Const]

    #: Content index over the string constants.
    #:
    #: Only strings can be content-indexed this way — `_same_constant`
    #: compares every other tag by object identity, which a content-keyed map
    #: can't express.
    var _strings: dict[str, int]

    pub fn init(self, entries: list[Const]) {
        self.entries = entries
        self._strings = {}
        #: A pool read back from a file starts non-empty; first occurrence
        #: wins, matching the linear scan this index replaces.
        for (index, existing) in entries.enumerate() {
            if existing.tag == ConstTag.Str {
                if self._strings.get(existing.value, -1) < 0 {
                    self._strings[existing.value] = index
                }
            }
        }
    }

    pub fn index_of(self, value: Const) -> int {
        if value.tag == ConstTag.Str {
            let found = self._strings.get(value.value, -1)
            if found >= 0 { return found }
            self.entries.push(value)
            let added = self.entries.len() - 1
            self._strings[value.value] = added
            return added
        }
        for (index, existing) in self.entries.enumerate() {
            if _same_constant(existing, value) { return index }
        }
        self.entries.push(value)
        return self.entries.len() - 1
    }

    pub fn index_of_name(self, name: str?) -> int {
        if name == null { return NO_NAME }
        return self.index_of(Const(ConstTag.Str, name))
    }
}

# Strings compare by content, everything else by object identity — never
# numeric, which would conflate int with float.
fn _same_constant(left: Const, right: Const) -> bool {
    if left is right { return true }
    if left.tag != right.tag { return false }
    if left.tag == ConstTag.Str { return left.value == right.value }
    return left.value is right.value
}

#: Serialise `mod` into the bytes of a `.jaic` file.
#:
#: The trailing CRC-32 covers everything before it, so the result is exactly
#: what `read_module` will accept.
pub fn write_module(mod: Module) -> list[int] {
    let out = Writer()
    out.bytes(MAGIC)
    out.u16(VERSION)
    out.u16(mod.flags)
    out.u32(COMPILER_VERSION)
    out.u32(build_id())
    out.u64(mod.source_hash)

    let path = utf8_bytes(mod.path)
    if path.len() > 65535 { throw ValueError(f"source path is too long: {path.len()}") }
    out.u16(path.len())
    out.bytes(path)
    out.u64(mod.mtime)

    # The mod-level pool holds export/import names; the body's own constants
    # live in its function record.
    let pool = Pool([])
    var names: list[str] = []
    for name in mod.exports { names.push(name) }
    for entry in mod.imports {
        names.push(entry.path)
        if entry.alias != null { names.push(entry.alias) }
    }
    out.u32(names.len())
    for name in names {
        out.u8(4)  # K_STR
        let encoded = utf8_bytes(name)
        out.u32(encoded.len())
        out.bytes(encoded)
        pool.index_of(Const(ConstTag.Str, name))
    }

    _write_function(out, mod.body, 0)

    out.u32(mod.exports.len())
    for name in mod.exports { out.u32(pool.index_of_name(name)) }
    out.u32(mod.imports.len())
    for entry in mod.imports {
        out.u32(pool.index_of_name(entry.path))
        out.u32(pool.index_of_name(entry.alias))
    }

    out.u32(crc32(out.data))
    return out.data
}

fn _write_function(out: Writer, proto: FuncProto, depth: int) -> void {
    if depth > MAX_DEPTH {
        throw ValueError(f"function records nest deeper than {MAX_DEPTH}")
    }
    let pool = Pool(_copy_constants(proto.chunk.constants))

    # Every index this record names must resolve before the pool is written —
    # resolving can append to it.
    let name_index = pool.index_of_name(proto.name)
    var param_indices: list[int] = []
    for name in proto.param_names { param_indices.push(pool.index_of_name(name)) }

    # Writing a class spec appends the names it refers to, so the loop
    # re-reads the length every time round.
    let body = Writer()
    var index = 0
    while index < pool.entries.len() {
        _write_constant(body, pool, pool.entries[index], true, depth + 1)
        index += 1
    }
    if pool.entries.len() >= _CONST_LIMIT {
        throw ValueError(f"constant pool overflow: {pool.entries.len()} entries")
    }

    out.u32(name_index)
    out.u8(proto.arity)
    out.u8(proto.flags & 0xFF)
    out.u16(proto.max_slots)
    out.u16(proto.upvalue_count)

    out.u32(proto.chunk.code.len())
    out.bytes(proto.chunk.code)

    out.u32(pool.entries.len())
    out.bytes(body.data)

    let line_stream = _ltv1_encode(proto.chunk.lines)
    out.u32(line_stream.len())
    out.bytes(line_stream)

    out.u16(proto.chunk.exceptions.len())
    for entry in proto.chunk.exceptions {
        out.u32(entry.start)
        out.u32(entry.end)
        out.u32(entry.handler)
        out.u32(entry.type_const)
    }

    out.u16(proto.default_offsets.len())
    for offset in proto.default_offsets { out.u32(offset) }

    out.u16(param_indices.len())
    for value in param_indices { out.u32(value) }
}

fn _copy_constants(constants: list[Const]) -> list[Const] {
    var out: list[Const] = []
    for value in constants { out.push(value) }
    return out
}

fn _write_constant(out: Writer, pool: Pool, value: Const, top_level: bool, depth: int) -> void {
    if depth > MAX_DEPTH { throw ValueError("constant nests too deeply") }
    match value.tag {
        ConstTag.Null => { out.u8(0) }
        ConstTag.Bool => {
            out.u8(1)
            out.u8(if bool(value.value) { 1 } else { 0 })
        }
        ConstTag.Int => {
            out.u8(2)
            out.u64(value.value)
        }
        ConstTag.Float => {
            out.u8(3)
            out.f64(value.value)
        }
        ConstTag.Str => {
            out.u8(4)
            let encoded = utf8_bytes(value.value)
            out.u32(encoded.len())
            out.bytes(encoded)
        }
        ConstTag.Bytes => {
            out.u8(5)
            out.u32(value.value.len())
            out.bytes(value.value)
        }
        ConstTag.Func => {
            out.u8(6)
            _write_function(out, value.value, depth)
        }
        ConstTag.ClassSpec => {
            # A nested class spec's method indices would name a pool still
            # being written, so it can't be rebuilt.
            if not top_level { throw ValueError("a class spec cannot nest inside a tuple") }
            out.u8(7)
            _write_class_spec(out, pool, value.value)
        }
        ConstTag.Tuple => {
            out.u8(8)
            out.u32(value.value.len())
            for item in value.value { _write_constant(out, pool, item, false, depth + 1) }
        }
        ConstTag.Type => {
            out.u8(9)
            let encoded = utf8_bytes(value.value)
            out.u32(encoded.len())
            out.bytes(encoded)
        }
    }
}

fn _write_class_spec(out: Writer, pool: Pool, spec: ClassSpec) -> void {
    out.u32(pool.index_of_name(spec.name))
    if spec.super_name == null {
        out.u32(0)
    } else {
        out.u32(pool.index_of_name(spec.super_name))
    }

    out.u16(spec.traits.len())
    for name in spec.traits { out.u32(pool.index_of_name(name)) }

    out.u16(spec.fields.len())
    for field in spec.fields {
        out.u32(pool.index_of_name(field.name))
        out.u8(field.visibility)
        out.u8(if field.is_static { 1 } else { 0 })
        out.u32(field.type_index)
    }

    out.u16(spec.methods.len())
    for method in spec.methods {
        out.u32(pool.index_of_name(method.name))
        out.u8(method.visibility)
        out.u8(method.flags & 0xFF)
        out.u32(method.func_index)
    }
}

#: Parse the bytes of a `.jaic` file, or return null when they cannot be used.
#:
#: Null means "recompile": bad magic, version/compiler mismatch, a source hash
#: other than `expected_hash`, a bad CRC, or a truncated/trailing byte — never
#: reported, since a stale cache is normal, not an error.
pub fn read_module(data: list[int], expected_hash: int? = null) -> Module? {
    if data.len() < MIN_SIZE { return null }
    for (index, byte) in MAGIC.enumerate() {
        if data[index] != byte { return null }
    }
    # Integrity first: nothing below parses a byte the checksum hasn't already
    # vouched for.
    let body_length = data.len() - 4
    var stored = 0
    for shift in range(4) { stored = stored | data[body_length + shift] << shift * 8 }
    if crc32(data, body_length) != stored { return null }

    let reader = Reader(data, body_length)
    reader.skip(4)
    let version = reader.u16()
    if version < VERSION_MIN or version > VERSION { return null }
    reader.version = version
    let flags = reader.u16()
    if reader.u32() != COMPILER_VERSION { return null }
    if reader.u32() != build_id() { return null }
    let hash = reader.u64()
    if expected_hash != null and hash != expected_hash { return null }

    let path_length = reader.u16()
    let path_bytes = reader.bytes(path_length)
    if reader.bad { return null }
    let mtime = reader.u64()

    let pool_count = reader.u32()
    if reader.bad or pool_count >= _CONST_LIMIT or not reader.fits(pool_count, 1) {
        return null
    }
    let pool = _read_constants(reader, pool_count, 0)
    if pool == null { return null }

    let body = _read_function(reader, 0)
    if body == null { return null }

    let mod = Module(utf8_text(path_bytes), hash, body, flags)
    mod.mtime = mtime

    let export_count = reader.u32()
    if reader.bad or not reader.fits(export_count, 4) { return null }
    for _index in range(export_count) {
        let name = _pool_string(pool, reader.u32())
        if name == null { return null }
        mod.exports.push(name)
    }

    let import_count = reader.u32()
    if reader.bad or not reader.fits(import_count, 8) { return null }
    for _index in range(import_count) {
        let path = _pool_string(pool, reader.u32())
        let alias_index = reader.u32()
        if path == null { return null }
        if alias_index == NO_NAME {
            mod.imports.push(Import(path, null))
        } else {
            let alias = _pool_string(pool, alias_index)
            if alias == null { return null }
            mod.imports.push(Import(path, alias))
        }
    }

    # A byte between the last section and the checksum means this is not the
    # file this reader thinks it is.
    if reader.bad or reader.position != reader.limit { return null }
    return mod
}

#: LTV1: the line table's encoding from `VERSION_LTV1` on.
#:
#: Must stay byte-identical to `jaiLtv1EncodeEntry` in `src/vm/bytecode/chunk.c`
#: — the C runtime reads what this writes, and nothing in the tree compares the
#: two implementations, so a disagreement surfaces as a wrong source span in a
#: diagnostic and nowhere else. `tests/vm/linetable_ltv1.c` pins the C side and
#: `scripts/linetable_golden.sh` pins the pair end to end.
#:
#: Per entry, deltas against the previous one, first against zero:
#:     uleb128  offset - prev_offset      (offsets are non-decreasing)
#:     sleb128  span - prev_span
#:     sleb128  span_end - span           (against its OWN span)
fn _uleb(out: list[int], value: int) -> void {
    var v = value
    while true {
        var byte = v & 0x7F
        v = v >> 7
        if v != 0 { byte = byte | 0x80 }
        out.push(byte)
        if v == 0 { break }
    }
}

#: Zigzag, so a small negative delta costs one byte like a small positive one.
#: Spans move backwards often — an operand is emitted after the expression that
#: owns it — so this is not a theoretical case.
fn _sleb(out: list[int], value: int) -> void {
    _uleb(out, value << 1 ^ value >> 63)
}

fn _ltv1_encode(lines: list[LineEntry]) -> list[int] {
    var out: list[int] = []
    var prev_offset = 0
    var prev_span = 0
    for line in lines {
        _uleb(out, line.offset - prev_offset)
        _sleb(out, line.span_start - prev_span)
        _sleb(out, line.span_end - line.span_start)
        prev_offset = line.offset
        prev_span = line.span_start
    }
    return out
}

#: Reads exactly `byte_count` bytes. Null on a malformed stream, which is a
#: refused image and therefore a recompile, never a wrong span.
fn _ltv1_decode(reader: Reader, byte_count: int) -> list[LineEntry]? {
    let stop = reader.position + byte_count
    var lines: list[LineEntry] = []
    var offset = 0
    var span = 0

    while reader.position < stop {
        let d_offset = _read_uleb(reader, stop)
        if d_offset == null { return null }
        let d_span = _read_sleb(reader, stop)
        if d_span == null { return null }
        let d_end = _read_sleb(reader, stop)
        if d_end == null { return null }

        offset = offset + (d_offset ?? 0)
        span = span + (d_span ?? 0)
        lines.push(LineEntry(offset, span, span + (d_end ?? 0)))
    }
    if reader.position != stop { return null }
    return lines
}

fn _read_uleb(reader: Reader, stop: int) -> int? {
    var value = 0
    var shift = 0
    while reader.position < stop {
        let byte = reader.u8()
        if reader.bad { return null }
        value = value | (byte & 0x7F) << shift
        shift += 7
        if byte & 0x80 == 0 { return value }
        if shift > 63 { return null }
    }
    return null
}

fn _read_sleb(reader: Reader, stop: int) -> int? {
    let zig = _read_uleb(reader, stop)
    if zig == null { return null }
    let z = zig ?? 0
    return z >> 1 ^ -(z & 1)
}

fn _read_function(reader: Reader, depth: int) -> FuncProto? {
    if depth > MAX_DEPTH { return null }
    let name_index = reader.u32()
    let arity = reader.u8()
    let flags = reader.u8()
    let max_slots = reader.u16()
    let upvalue_count = reader.u16()
    if reader.bad or upvalue_count > 255 { return null }

    let code_length = reader.u32()
    if reader.bad or code_length > reader.left() { return null }
    let code = reader.bytes(code_length)
    if reader.bad { return null }

    let const_count = reader.u32()
    if reader.bad or const_count >= _CONST_LIMIT or not reader.fits(const_count, 1) {
        return null
    }
    let constants = _read_constants(reader, const_count, depth + 1)
    if constants == null { return null }
    _link_class_specs(constants)

    let line_bytes = reader.u32()
    if reader.bad or line_bytes > reader.left() { return null }
    var lines: list[LineEntry] = []
    if reader.version >= VERSION_LTV1 {
        let decoded = _ltv1_decode(reader, line_bytes)
        if decoded == null { return null }
        lines = decoded ?? []
    } else {
        # Pre-LTV1: three absolute u32s per row.
        if line_bytes % _LINE_ROW != 0 { return null }
        for _row in range(line_bytes // _LINE_ROW) {
            lines.push(LineEntry(reader.u32(), reader.u32(), reader.u32()))
        }
    }

    let exception_count = reader.u16()
    if reader.bad or not reader.fits(exception_count, _EXC_ROW) { return null }
    var exceptions: list[ExceptionEntry] = []
    for _row in range(exception_count) {
        exceptions.push(ExceptionEntry(reader.u32(), reader.u32(), reader.u32(), reader.u32()))
    }

    let default_count = reader.u16()
    if reader.bad or default_count > 255 or not reader.fits(default_count, 4) {
        return null
    }
    var defaults: list[int] = []
    for _row in range(default_count) { defaults.push(reader.u32()) }

    let param_count = reader.u16()
    if reader.bad or not reader.fits(param_count, 4) { return null }
    var params: list[str] = []
    for _row in range(param_count) {
        let index = reader.u32()
        if index == NO_NAME {
            params.push("")
            continue
        }
        let name = _pool_string(constants, index)
        if name == null { return null }
        params.push(name)
    }
    if reader.bad { return null }

    var name: str? = null
    if name_index != NO_NAME {
        name = _pool_string(constants, name_index)
        if name == null { return null }
    }

    let proto = FuncProto(name, arity, flags)
    proto.max_slots = max_slots
    proto.upvalue_count = upvalue_count
    proto.default_offsets = defaults
    proto.param_names = params
    proto.chunk.code = code
    proto.chunk.constants = constants
    proto.chunk.lines = lines
    proto.chunk.exceptions = exceptions
    proto.chunk.cache_count = _cache_slots(code, constants)
    return proto
}

fn _read_constants(reader: Reader, count: int, depth: int) -> list[Const]? {
    var out: list[Const] = []
    for _index in range(count) {
        let value = _read_constant(reader, depth)
        if value == null { return null }
        out.push(value)
    }
    if reader.bad { return null }
    return out
}

fn _read_constant(reader: Reader, depth: int) -> Const? {
    if depth > MAX_DEPTH { return null }
    let tag = reader.u8()
    if reader.bad { return null }

    if tag == 0 { return Const(ConstTag.Null, null) }
    if tag == 1 { return Const(ConstTag.Bool, reader.u8() != 0) }
    if tag == 2 { return Const(ConstTag.Int, reader.u64()) }
    if tag == 3 { return Const(ConstTag.Float, reader.f64()) }
    if tag == 4 or tag == 9 {
        let length = reader.u32()
        if reader.bad or length > reader.left() { return null }
        let text = utf8_text(reader.bytes(length))
        return Const(if tag == 4 { ConstTag.Str } else { ConstTag.Type }, text)
    }
    if tag == 5 {
        let length = reader.u32()
        if reader.bad or length > reader.left() { return null }
        return Const(ConstTag.Bytes, reader.bytes(length))
    }
    if tag == 6 {
        let nested = _read_function(reader, depth + 1)
        if nested == null { return null }
        return Const(ConstTag.Func, nested)
    }
    if tag == 7 {
        let spec = _read_class_spec(reader)
        if spec == null { return null }
        return Const(ConstTag.ClassSpec, spec)
    }
    if tag == 8 {
        let count = reader.u32()
        if reader.bad or not reader.fits(count, 1) { return null }
        var items: list[Const] = []
        for _index in range(count) {
            let item = _read_constant(reader, depth + 1)
            if item == null { return null }
            items.push(item)
        }
        return Const(ConstTag.Tuple, items)
    }
    return null
}

#: Names inside a class spec are pool indices, and the pool is still being
#: read when the spec arrives, so they are parked as `#index` placeholders
#: until `_link_class_specs` runs over the finished pool.
fn _read_class_spec(reader: Reader) -> ClassSpec? {
    let name_index = reader.u32()
    let super_index = reader.u32()
    let trait_count = reader.u16()
    if reader.bad or not reader.fits(trait_count, 4) { return null }
    var traits: list[str] = []
    for _index in range(trait_count) { traits.push(f"#{reader.u32()}") }

    let field_count = reader.u16()
    if reader.bad or not reader.fits(field_count, 10) { return null }
    var fields: list[FieldSpec] = []
    for _index in range(field_count) {
        let field_name = reader.u32()
        let visibility = reader.u8()
        let is_static = reader.u8() != 0
        fields.push(FieldSpec(f"#{field_name}", visibility, is_static, reader.u32()))
    }

    let method_count = reader.u16()
    if reader.bad or not reader.fits(method_count, 10) { return null }
    var methods: list[MethodSpec] = []
    for _index in range(method_count) {
        let method_name = reader.u32()
        let visibility = reader.u8()
        let flags = reader.u8()
        methods.push(MethodSpec(f"#{method_name}", visibility, flags, reader.u32()))
    }
    if reader.bad { return null }

    let parent = if super_index == 0 { null } else { f"#{super_index}" }
    return ClassSpec(f"#{name_index}", parent, traits, fields, methods)
}

# Resolve the placeholders once the pool they index into is complete. Nested
# function records link their own pools as they are read.
fn _link_class_specs(constants: list[Const]) -> void {
    for value in constants {
        if value.tag == ConstTag.ClassSpec {
            let spec = value.value
            var fields: list[FieldSpec] = []
            for field in spec.fields {
                fields.push(
                    FieldSpec(
                        _linked(constants, field.name),
                        field.visibility,
                        field.is_static,
                        field.type_index
                    )
                )
            }
            var methods: list[MethodSpec] = []
            for method in spec.methods {
                methods.push(
                    MethodSpec(
                        _linked(constants, method.name),
                        method.visibility,
                        method.flags,
                        method.func_index
                    )
                )
            }
            var traits: list[str] = []
            for name in spec.traits { traits.push(_linked(constants, name)) }
            value.value = ClassSpec(
                _linked(constants, spec.name),
                _linked_optional(constants, spec.super_name),
                traits,
                fields,
                methods
            )
        }
    }
}

fn _linked(constants: list[Const], placeholder: str) -> str {
    if not placeholder.starts_with("#") { return placeholder }
    let index = int(placeholder[1:])
    return _pool_string(constants, index) ?? placeholder
}

fn _linked_optional(constants: list[Const], placeholder: str?) -> str? {
    if placeholder == null { return null }
    return _linked(constants, placeholder)
}

fn _pool_string(constants: list[Const], index: int) -> str? {
    if index < 0 or index >= constants.len() { return null }
    let value = constants[index]
    if value.tag != ConstTag.Str { return null }
    return value.value
}

# Cache slots are never stored — sizes come from scanning the code, which
# doubles as a decodability check. The per-opcode cache-operand offset must
# track `jaiOpCacheOperand` in src/vm/bytecode/chunk.c, or a loaded chunk silently gets
# too few cache lines and every site degrades to the slow path.
fn _cache_slots(code: list[int], constants: list[Const]) -> int {
    var highest = -1
    var offset = 0
    while offset < code.len() {
        let size = _instruction_size(code, offset, constants)
        if size <= 0 { return highest + 1 }
        let at = _cache_operand(op_name(code[offset]))
        if at >= 0 { highest = max(highest, _read_u16(code, offset + 1 + at)) }
        offset += size
    }
    return highest + 1
}

fn _cache_operand(name: str) -> int {
    if name == "OP_GET_GLOBAL" or
        name == "OP_SET_GLOBAL" or
        name == "OP_GET_FIELD" or
        name == "OP_SET_FIELD" {
        return 3
    }
    if name == "OP_INVOKE" { return 4 }
    if name == "OP_GET_FIELD_LOCAL" { return 5 }
    if name == "OP_FORMAT" { return 7 }
    return -1
}

# OP_CLOSURE is the one variable-length instruction: a u24 constant index and
# three bytes per capture, with the capture count coming from the function
# record that index names.
fn _instruction_size(code: list[int], offset: int, constants: list[Const]) -> int {
    let operands = op_operand_bytes(code[offset])
    if operands >= 0 { return 1 + operands }
    let index = _read_u24(code, offset + 1)
    if index >= constants.len() or constants[index].tag != ConstTag.Func { return -1 }
    return 4 + 3 * constants[index].value.upvalue_count
}

fn _read_u16(code: list[int], offset: int) -> int {
    if offset + 1 >= code.len() { return 0 }
    return code[offset] | code[offset + 1] << 8
}

#: Disassemble one function record, nested functions included.
#:
#: The layout matches `jaiDisassembleChunk`: offset, source position,
#: mnemonic, operands, and a note naming the constant or jump target.
pub fn disassemble(proto: FuncProto, title: str = "") -> str {
    var out: list[str] = []
    let name = if title != "" { title } else { proto.name ?? "<mod>" }
    out.push(f"== {name} ==")
    out.push(
        f"   arity {proto.arity}  flags {proto.flags}  slots {proto.max_slots}" +
            f"  upvalues {proto.upvalue_count}"
    )

    var offset = 0
    let chunk = proto.chunk
    while offset < chunk.code.len() {
        let step = disassemble_instruction(chunk, offset)
        out.push(step[0])
        if step[1] <= offset { break }
        offset = step[1]
    }
    for entry in chunk.exceptions {
        out.push(
            f"   handler [{entry.start},{entry.end}) -> {entry.handler}" +
                f" type {entry.type_const}"
        )
    }
    for value in chunk.constants {
        if value.tag == ConstTag.Func {
            out.push("")
            out.push(disassemble(value.value))
        }
    }
    return "\n".join(out)
}

#: Render the instruction at `offset` and return it with the next offset.
pub fn disassemble_instruction(chunk: Chunk, offset: int) -> tuple[str, int] {
    let code = chunk.code
    let raw = code[offset]
    let name = op_name(raw)
    let where = _position(chunk, offset)
    let head = _pad_left(str(offset), 4, "0") + "  " + _pad_left(where, 4, " ") + "  "

    if name == "OP_CLOSURE" {
        let index = _read_u24(code, offset + 1)
        var captures = 0
        if index < chunk.constants.len() and chunk.constants[index].tag == ConstTag.Func {
            captures = chunk.constants[index].value.upvalue_count
        }
        let text = head +
            _pad_right(name, 22) +
            _pad_right(f"{index}", 18) +
            f" ; {_render(chunk, index)}"
        return (text, offset + 4 + 3 * captures)
    }

    let operands = op_operand_bytes(raw)
    if operands < 0 or offset + 1 + operands > code.len() {
        return (head + _pad_right(name, 22) + " ; <truncated>", code.len())
    }
    let next = offset + 1 + operands
    var shown = ""
    var note = ""

    if operands == 3 {
        let index = _read_u24(code, offset + 1)
        shown = f"{index}"
        note = _render(chunk, index)
    } elif name == "OP_GET_GLOBAL" or
        name == "OP_SET_GLOBAL" or
        name == "OP_GET_FIELD" or
        name == "OP_SET_FIELD" {
        let index = _read_u24(code, offset + 1)
        shown = f"{index} c{_read_u16(code, offset + 4)}"
        note = _render(chunk, index)
    } elif name == "OP_INVOKE" {
        let index = _read_u24(code, offset + 1)
        shown = f"{index} ({code[offset + 4]} args) c{_read_u16(code, offset + 5)}"
        note = _render(chunk, index)
    } elif _is_jump(name) {
        let target = next + _read_i16(code, offset + 1 + operands - 2)
        shown = f"{_read_i16(code, offset + 1 + operands - 2)}"
        note = f"-> {target}"
    } elif name == "OP_LOOP" {
        let target = next + _read_i16(code, offset + 1)
        shown = f"{_read_i16(code, offset + 1)}"
        note = f"-> {target}"
    } elif operands == 1 {
        shown = f"{code[offset + 1]}"
    } elif operands == 2 {
        shown = f"{_read_u16(code, offset + 1)}"
    } elif operands > 0 {
        var parts: list[str] = []
        for index in range(operands) { parts.push(f"{code[offset + 1 + index]}") }
        shown = " ".join(parts)
    }

    if note != "" {
        return (head + _pad_right(name, 22) + _pad_right(shown, 18) + f" ; {note}", next)
    }
    if shown != "" { return (head + _pad_right(name, 22) + shown, next) }
    return (head + name, next)
}

fn _is_jump(name: str) -> bool {
    return name.starts_with("OP_JUMP") or
        name == "OP_FOR_ITER" or
        name == "OP_PUSH_FINALLY" or
        name.starts_with("OP_MATCH_")
}

fn _position(chunk: Chunk, offset: int) -> str {
    var found = -1
    for line in chunk.lines {
        if line.offset <= offset { found = line.span_start }
    }
    if found < 0 { return "?" }
    for line in chunk.lines {
        if line.offset == offset { return f"{found}" }
    }
    return "|"
}

fn _render(chunk: Chunk, index: int) -> str {
    if index < 0 or index >= chunk.constants.len() { return "<bad constant>" }
    let value = chunk.constants[index]
    match value.tag {
        ConstTag.Str => { return f"\"{value.value}\"" }
        ConstTag.Func => {
            let name = value.value.name ?? "anonymous"
            return f"<fn {name}>"
        }
        ConstTag.ClassSpec => { return f"<class {value.value.name}>" }
        ConstTag.Tuple => { return f"<tuple of {value.value.len()}>" }
        ConstTag.Null => { return "null" }
        _ => { return f"{value.value}" }
    }
}

fn _read_u24(code: list[int], offset: int) -> int {
    if offset + 2 >= code.len() { return 0 }
    return code[offset] | code[offset + 1] << 8 | code[offset + 2] << 16
}

fn _read_i16(code: list[int], offset: int) -> int {
    let value = _read_u16(code, offset)
    if value >= 32768 { return value - 65536 }
    return value
}

fn _pad_right(text: str, width: int) -> str {
    if text.len() >= width { return text }
    return text + repeat(" ", width - text.len())
}

fn _pad_left(text: str, width: int, fill: str) -> str {
    if text.len() >= width { return text }
    return repeat(fill, width - text.len()) + text
}
