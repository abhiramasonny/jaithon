#: The bytes of a `.jaic` file back to a `Module`, or null.
#:
#: Nothing is trusted. Bad magic, a version outside the accepted window, a
#: compiler version or build id that is not this binary's, a source hash other
#: than the caller's, a bad CRC, a short read, an out-of-range index, or a
#: byte left over before the checksum all mean the same thing -- recompile --
#: and all of them return null rather than raising or reporting, because a
#: stale cache is normal and not an error. The checksum is verified before a
#: single byte below the magic is parsed.
#:
#: Every null is returned before the object it would have gone into, so no
#: caller can be handed a half-built `Module`.
#:
#: What is NOT here: the writer, and the instruction walk that recovers
#: `cache_count` -- that is shared with the disassembler and lives in
#: `code.jai`.
from jaithon.compile.emit import ClassSpec, Const, ConstTag, ExceptionEntry
from jaithon.compile.emit import FieldSpec, FuncProto, MethodSpec
from jaithon.compile.jaic.bits import crc32, utf8_text
from jaithon.compile.jaic.code import _cache_slots
from jaithon.compile.jaic.format import COMPILER_VERSION, MAGIC, MAX_DEPTH
from jaithon.compile.jaic.format import MIN_SIZE, NO_NAME, VERSION
from jaithon.compile.jaic.format import VERSION_FNFLAGS16, VERSION_MIN
from jaithon.compile.jaic.format import VERSION_STRTAB, _CONST_LIMIT, _EXC_ROW
from jaithon.compile.jaic.format import build_id
from jaithon.compile.jaic.linetable import _ltv1_decode
from jaithon.compile.jaic.model import Import, Module
from jaithon.compile.jaic.stream import Reader

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

    # The string table comes before the pools, because every tag-10 constant in
    # them indexes it.
    if reader.version >= VERSION_STRTAB {
        let table = _read_strtab(reader)
        if table == null { return null }
        reader.strings = table ?? []
    }

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

#: The module string table: count, then each string as {uleb128 length, bytes}.
#: Null on a malformed table, which refuses the image and recompiles.
fn _read_strtab(reader: Reader) -> list[str]? {
    let count = reader.uleb()
    if count == null { return null }
    let n = count ?? 0
    if n < 0 or n >= _CONST_LIMIT { return null }
    # One byte minimum per entry, so a corrupt count cannot outrun the file.
    if not reader.fits(n, 1) { return null }

    var out: list[str] = []
    for _index in range(n) {
        let length = reader.uleb()
        if length == null { return null }
        let len = length ?? 0
        if len < 0 or len > reader.left() { return null }
        let encoded = reader.bytes(len)
        if reader.bad { return null }
        out.push(utf8_text(encoded))
    }
    return out
}

fn _read_function(reader: Reader, depth: int) -> FuncProto? {
    if depth > MAX_DEPTH { return null }
    let name_index = reader.u32()
    let arity = reader.u8()
    let flags = if reader.version >= VERSION_FNFLAGS16 { reader.u16() } else { reader.u8() }
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
    let decoded = _ltv1_decode(reader, line_bytes)
    if decoded == null { return null }
    let lines = decoded ?? []

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
    # Tag 4 (inline string bytes) is gone: from `VERSION_STRTAB` on every string
    # is a tag-10 index, and `VERSION_MIN` no longer admits an image that could
    # carry one.
    if tag == 9 {
        let length = reader.u32()
        if reader.bad or length > reader.left() { return null }
        return Const(ConstTag.Type, utf8_text(reader.bytes(length)))
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
    if tag == 10 {
        let index = reader.uleb()
        if index == null { return null }
        let position = index ?? 0
        if position < 0 or position >= reader.strings.len() { return null }
        return Const(ConstTag.Str, reader.strings[position])
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
