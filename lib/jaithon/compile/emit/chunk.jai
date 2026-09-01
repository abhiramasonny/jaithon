#: What the emitter writes into: one function's chunk, and the record that
#: wraps it.
from jaithon.ast import Span
from jaithon.compile.emit.opcode import Op, STACK_EFFECT_VARIABLE, op_stack_effect
from jaithon.compile.emit.opcode import opcode
from jaithon.compile.emit.spec import Const, ConstTag, ExceptionEntry, LineEntry
from jaithon.compile.emit.spec import MAX_CONSTANTS

#: One function's code, constants, line table, exception table and cache count.
pub class Chunk {
    #: Instruction bytes.
    pub var code: list[int]
    #: The constant pool, deduplicated for everything but functions and classes.
    pub var constants: list[Const]
    #: Source spans, one row per span change.
    pub var lines: list[LineEntry]
    #: Guarded regions, in the order the emitter closed them.
    pub var exceptions: list[ExceptionEntry]
    #: Number of inline-cache slots this chunk hands out.
    pub var cache_count: int
    #: Operand-stack depth at the cursor. Temporaries live in frame slots, so
    #: this counts only values genuinely in flight.
    pub var depth: int
    #: High-water mark of `depth` (`max_slots = max_slot + max_depth`).
    pub var max_depth: int

    var const_index: dict[str, int]

    #: Create an empty chunk.
    pub fn init(self) {
        self.code = []
        self.constants = []
        self.lines = []
        self.exceptions = []
        self.cache_count = 0
        self.depth = 0
        self.max_depth = 0
        self.const_index = {}
    }

    #: Apply a stack effect the opcode table could not.
    pub fn adjust(self, delta: int) -> void {
        self.depth += delta
        if self.depth < 0 { self.depth = 0 }
        if self.depth > self.max_depth { self.max_depth = self.depth }
    }

    #: Re-synchronise the tracked depth at a merge point, such as a patched label.
    pub fn set_depth(self, value: int) -> void {
        self.depth = value
        if value > self.max_depth { self.max_depth = value }
    }

    #: Raise the high-water mark without touching the cursor. A `defer` thunk
    #: runs on its defining frame, so whatever it needs the frame must reserve.
    pub fn raise_max_depth(self, value: int) -> void {
        if value > self.max_depth { self.max_depth = value }
    }

    #: Offset the next byte will be written at.
    pub fn here(self) -> int { return self.code.len() }

    #: Append one byte.
    pub fn write_byte(self, value: int) -> void {
        self.code.push(value & 0xFF)
    }

    #: Append a little-endian `u16`.
    pub fn write_u16(self, value: int) -> void {
        self.code.push(value & 0xFF)
        self.code.push(value >> 8 & 0xFF)
    }

    #: Append a little-endian `u24`, the width of every constant index.
    pub fn write_u24(self, value: int) -> void {
        self.code.push(value & 0xFF)
        self.code.push(value >> 8 & 0xFF)
        self.code.push(value >> 16 & 0xFF)
    }

    #: Append a little-endian `i16`. Raises `ValueError` when `value` does not
    #: fit, which is how an over-long jump is caught before it silently wraps.
    pub fn write_i16(self, value: int) -> void {
        if value < -32768 or value > 32767 {
            throw ValueError(f"i16 operand out of range: {value}")
        }
        self.write_u16(value & 0xFFFF)
    }

    #: Overwrite the `u16` at `offset`.
    pub fn patch_u16(self, offset: int, value: int) -> void {
        self.code[offset] = value & 0xFF
        self.code[offset + 1] = value >> 8 & 0xFF
    }

    #: Overwrite the `i16` at `offset`.
    pub fn patch_i16(self, offset: int, value: int) -> void {
        if value < -32768 or value > 32767 {
            throw ValueError(f"jump distance out of range: {value}")
        }
        self.patch_u16(offset, value & 0xFFFF)
    }

    #: Write the opcode byte for `op`, opening a new line-table row when the
    #: source span differs from the previous instruction's, and applying the
    #: opcode's fixed stack effect. Returns its offset. Every instruction goes
    #: through here; only the `STACK_EFFECT_VARIABLE` opcodes are left for the
    #: caller to adjust.
    pub fn write_op(self, op: Op, span: Span?) -> int {
        let offset = self.here()
        if span != null and span.is_known() {
            let rows = self.lines.len()
            let same = rows > 0 and
                self.lines[rows - 1].span_start == span.start and
                self.lines[rows - 1].span_end == span.end
            if not same {
                self.lines.push(LineEntry(offset, span.start, span.end))
            }
        }
        let code = opcode(op)
        self.code.push(code)
        let effect = op_stack_effect(code)
        if effect != STACK_EFFECT_VARIABLE { self.adjust(effect) }
        return offset
    }

    #: Add `value` to the pool and return its index, reusing an equal entry.
    #: Functions and class specs are never shared, which is the line
    #: `constDedupable` in `src/vm/bytecode/chunk.c` draws; a tuple is on the sharing
    #: side of it.
    pub fn add_constant(self, value: Const) -> int {
        let key = _const_key(value)
        if key != null {
            let existing = self.const_index.get(key, -1)
            if existing >= 0 { return existing }
        }
        if self.constants.len() >= MAX_CONSTANTS {
            throw ValueError(f"constant pool overflow: {MAX_CONSTANTS} entries")
        }
        let index = self.constants.len()
        self.constants.push(value)
        if key != null { self.const_index[key] = index }
        return index
    }

    #: Reserve an inline-cache slot and return its index.
    pub fn add_cache(self) -> int {
        let index = self.cache_count
        self.cache_count += 1
        return index
    }
}

fn _const_key(value: Const) -> str? {
    match value.tag {
        ConstTag.Null => { return "n" }
        ConstTag.Bool => { return if bool(value.value) { "b1" } else { "b0" } }
        ConstTag.Int => { return f"i{value.value}" }
        ConstTag.Float => { return f"f{repr(value.value)}" }
        ConstTag.Str => { return f"s{value.value}" }
        ConstTag.Type => { return f"t{value.value}" }
        ConstTag.Bytes => { return f"y{value.value}" }
        ConstTag.Tuple => {
            var parts: list[str] = []
            for item in value.value {
                let key = _const_key(item)
                if key is null { return null }
                parts.push(f"{key.len()}:{key}")
            }
            return "p" + "".join(parts)
        }
        _ => { return null }
    }
}

#: One compiled function: the record `spec/BYTECODE.md` §5 serialises.
pub class FuncProto {
    #: The function name, or null for the module body and for lambdas.
    pub var name: str?
    #: Declared parameters, excluding `self`, the variadic and the keyword rest.
    pub var arity: int
    #: `FN_*` bits.
    pub var flags: int
    #: Frame window size, which is the resolver's peak slot count.
    pub var max_slots: int
    #: Number of captured variables.
    pub var upvalue_count: int
    #: The code and its pool.
    pub var chunk: Chunk
    #: Code offsets of the default-value thunks, for the last parameters.
    pub var default_offsets: list[int]
    #: Parameter names, used to resolve keyword arguments at call time.
    pub var param_names: list[str]

    #: Create an empty proto.
    pub fn init(self, name: str?, arity: int, flags: int) {
        self.name = name
        self.arity = arity
        self.flags = flags
        self.max_slots = 1
        self.upvalue_count = 0
        self.chunk = Chunk()
        self.default_offsets = []
        self.param_names = []
    }
}
