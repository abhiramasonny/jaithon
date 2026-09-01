#: A `Module` out to the bytes of a `.jaic` file, and every record inside it.
#:
#: Two things shape this direction and neither has a mirror in the reader.
#: The body is built into a second `Writer` before anything reaches the first,
#: because the string table has to be complete before it can be written and it
#: is only complete once every nested function record has been walked. And
#: `Pool` appends rather than rewrites: a `u24` operand already sitting in the
#: code names a constant by index, so an index that exists must never move.
#:
#: What is NOT here: the reader. The two directions share the format
#: constants, the byte cursors and LTV1, and deliberately nothing else --
#: `read.jai` re-derives the layout from the same specification rather than
#: from this file, which is what makes disagreeing with it a caught bug.
from jaithon.compile.emit import ClassSpec, Const, ConstTag, FuncProto
from jaithon.compile.jaic.bits import crc32, utf8_bytes
from jaithon.compile.jaic.format import COMPILER_VERSION, MAGIC, MAX_DEPTH
from jaithon.compile.jaic.format import NO_NAME, VERSION, VERSION_FNFLAGS16
from jaithon.compile.jaic.format import _CONST_LIMIT, build_id
from jaithon.compile.jaic.linetable import _ltv1_encode
from jaithon.compile.jaic.model import Module, StrTab
from jaithon.compile.jaic.stream import Writer

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

    # The body goes into its own buffer first: the string table has to be
    # complete before it can be written, and it is only complete once every
    # nested function record has been walked. The table then goes ahead of the
    # body, so the reader has it before the first tag-10 constant.
    let strings = StrTab()
    let body = Writer()

    # The mod-level pool holds export/import names; the body's own constants
    # live in its function record.
    let pool = Pool([])
    var names: list[str] = []
    for name in mod.exports { names.push(name) }
    for entry in mod.imports {
        names.push(entry.path)
        if entry.alias != null { names.push(entry.alias) }
    }
    body.u32(names.len())
    for name in names {
        body.u8(10)  # K_STRREF
        body.uleb(strings.intern(name))
        pool.index_of(Const(ConstTag.Str, name))
    }

    _write_function(body, strings, mod.body, 0)

    body.u32(mod.exports.len())
    for name in mod.exports { body.u32(pool.index_of_name(name)) }
    body.u32(mod.imports.len())
    for entry in mod.imports {
        body.u32(pool.index_of_name(entry.path))
        body.u32(pool.index_of_name(entry.alias))
    }

    strings.write(out)
    out.bytes(body.data)

    out.u32(crc32(out.data))
    return out.data
}

fn _write_function(out: Writer, strings: StrTab, proto: FuncProto, depth: int) -> void {
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
        _write_constant(body, pool, strings, pool.entries[index], true, depth + 1)
        index += 1
    }
    if pool.entries.len() >= _CONST_LIMIT {
        throw ValueError(f"constant pool overflow: {pool.entries.len()} entries")
    }

    out.u32(name_index)
    out.u8(proto.arity)
    if VERSION >= VERSION_FNFLAGS16 {
        out.u16(proto.flags & 0xFFFF)
    } else {
        out.u8(proto.flags & 0xFF)
    }
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

fn _write_constant(
    out: Writer,
    pool: Pool,
    strings: StrTab,
    value: Const,
    top_level: bool,
    depth: int
) -> void {
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
            # Tag 10, an index into the module string table — not tag 4 and not
            # the bytes. See `StrTab` in `src/vm/bytecode/serialize.c`.
            out.u8(10)
            out.uleb(strings.intern(value.value))
        }
        ConstTag.Bytes => {
            out.u8(5)
            out.u32(value.value.len())
            out.bytes(value.value)
        }
        ConstTag.Func => {
            out.u8(6)
            _write_function(out, strings, value.value, depth)
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
            for item in value.value { _write_constant(out, pool, strings, item, false, depth + 1) }
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
