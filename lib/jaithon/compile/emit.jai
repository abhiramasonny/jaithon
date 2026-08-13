#: Lower a resolved `jaithon.ast` tree to bytecode: the opcode set, the chunk, and
#: the emitter.
#:
#: `Op` lists the opcodes of `src/vm/bytecode/chunk.h` in exactly that order: that order
#: *is* the wire format, so the enum's position is what a `.jaic` file stores
#: and a new opcode may only be appended.
from jaithon.ast import CompKind, Node, NodeKind, OpKind, Span, TypeExpr, TypeKind
from jaithon.ast import Visibility
from jaithon.ast import is_statement
from jaithon.compile.diag import DiagnosticBag
from jaithon.compile.symbol import Resolution, SemaCode, Symbol, SymbolKind

#: `ObjFunction` flag bits, as `spec/BYTECODE.md` §5 stores them.
pub const FN_VARIADIC: int = 1
pub const FN_KWREST: int = 2
pub const FN_METHOD: int = 4
pub const FN_STATIC: int = 8
pub const FN_GENERATOR: int = 16
pub const FN_ASYNC: int = 32
pub const FN_GETTER: int = 64
pub const FN_SETTER: int = 128

#: Largest constant index a `u24` operand can hold.
pub const MAX_CONSTANTS: int = 16777216

#: `kind` of the class-spec constant `OP_CLASS` reads (`classSpecInstantiate`).
pub const SPEC_CLASS: int = 0
pub const SPEC_TRAIT: int = 1
pub const SPEC_ENUM: int = 2

#: `typeConst` of an exception-table entry that catches everything.
#: `JAI_HANDLER_CATCH_ALL` in `src/vm/vm.c`; no pool index can reach it.
pub const CATCH_ALL: int = 4294967295

#: The opcode set of `src/vm/bytecode/chunk.h`, in wire order.
pub enum Op {
    Nop,
    Const,
    Null,
    True,
    False,
    Int,
    Pop,
    PopN,
    Dup,
    Dup2,
    Swap,
    Rot3,
    GetLocal,
    SetLocal,
    GetUpvalue,
    SetUpvalue,
    CloseUpvalue,
    GetGlobal,
    SetGlobal,
    DefGlobal,
    GetModule,
    Add,
    Sub,
    Mul,
    Div,
    FloorDiv,
    Mod,
    Pow,
    AddWrap,
    SubWrap,
    MulWrap,
    Neg,
    Pos,
    BAnd,
    BOr,
    BXor,
    Shl,
    Shr,
    BNot,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Is,
    IsNot,
    In,
    NotIn,
    Not,
    Concat,
    AddIntConst,
    IncLocal,
    CmpLocalConstLt,
    GetLocal2,
    AddLocals,
    Jump,
    JumpIfFalse,
    JumpIfTrue,
    JumpIfFalseKeep,
    JumpIfTrueKeep,
    JumpIfNull,
    Loop,
    GetIter,
    ForIter,
    Call,
    CallKw,
    CallSpread,
    Invoke,
    SuperInvoke,
    TailCall,
    Return,
    ReturnNull,
    Closure,
    BuildList,
    BuildDict,
    BuildSet,
    BuildTuple,
    BuildRange,
    ListAppend,
    DictInsert,
    SetAdd,
    GetIndex,
    SetIndex,
    GetSlice,
    SetSlice,
    Unpack,
    Class,
    Inherit,
    ImplTrait,
    Method,
    FieldDef,
    GetField,
    SetField,
    GetSuper,
    New,
    EnumNew,
    EnumTag,
    EnumField,
    IsInstance,
    Throw,
    Reraise,
    PushHandler,
    PopHandler,
    PushFinally,
    EndFinally,
    PushDefer,
    RunDefers,
    MatchExc,
    GetExc,
    MatchConst,
    MatchRange,
    MatchType,
    MatchSeq,
    MatchFields,
    Bind,
    Import,
    ImportFrom,
    Export,
    AssertFail,
    TypeGuard,
    Halt,
    # This enum's order is the on-disk encoding; new opcodes append at the end,
    # never insert, or old `.jaic` files misdecode.
    GetFieldLocal,
    ForIterBind,
    JumpIfCmpFalse,
    Format,
    JumpIfCmpLocalK,
    ToFloat,
    ModIntConst,
    AddBind,
    SubBind,
    MulBind,
    ElemKind,
    GetIterItems,
    ForIterPair,
    IterRange,
    ForRangeBind,
}

#: Stack effect of an instruction whose net change depends on its operands.
#: `SE_VAR` in `src/vm/bytecode/chunk.c`; the emitter adjusts those by hand.
pub const STACK_EFFECT_VARIABLE: int = -1000000

class OpSpec {
    pub let op: Op
    pub let name: str
    pub let operands: int
    pub let effect: int

    pub fn init(self, op: Op, name: str, operands: int, effect: int) {
        self.op = op
        self.name = name
        self.operands = operands
        self.effect = effect
    }
}

# From the JAI_OPCODES table in src/vm/bytecode/chunk.c. OP_CLOSURE is variable-length,
# recorded as -1 operand bytes.
let _OPS: list[OpSpec] = [
    OpSpec(Op.Nop, "OP_NOP", 0, 0),
    OpSpec(Op.Const, "OP_CONST", 3, 1),
    OpSpec(Op.Null, "OP_NULL", 0, 1),
    OpSpec(Op.True, "OP_TRUE", 0, 1),
    OpSpec(Op.False, "OP_FALSE", 0, 1),
    OpSpec(Op.Int, "OP_INT", 2, 1),
    OpSpec(Op.Pop, "OP_POP", 0, -1),
    OpSpec(Op.PopN, "OP_POPN", 1, STACK_EFFECT_VARIABLE),
    OpSpec(Op.Dup, "OP_DUP", 0, 1),
    OpSpec(Op.Dup2, "OP_DUP2", 0, 2),
    OpSpec(Op.Swap, "OP_SWAP", 0, 0),
    OpSpec(Op.Rot3, "OP_ROT3", 0, 0),
    OpSpec(Op.GetLocal, "OP_GET_LOCAL", 2, 1),
    OpSpec(Op.SetLocal, "OP_SET_LOCAL", 2, 0),
    OpSpec(Op.GetUpvalue, "OP_GET_UPVALUE", 1, 1),
    OpSpec(Op.SetUpvalue, "OP_SET_UPVALUE", 1, 0),
    OpSpec(Op.CloseUpvalue, "OP_CLOSE_UPVALUE", 2, 0),
    OpSpec(Op.GetGlobal, "OP_GET_GLOBAL", 5, 1),
    OpSpec(Op.SetGlobal, "OP_SET_GLOBAL", 5, 0),
    OpSpec(Op.DefGlobal, "OP_DEF_GLOBAL", 3, -1),
    OpSpec(Op.GetModule, "OP_GET_MODULE", 6, 1),
    OpSpec(Op.Add, "OP_ADD", 0, -1),
    OpSpec(Op.Sub, "OP_SUB", 0, -1),
    OpSpec(Op.Mul, "OP_MUL", 0, -1),
    OpSpec(Op.Div, "OP_DIV", 0, -1),
    OpSpec(Op.FloorDiv, "OP_FLOORDIV", 0, -1),
    OpSpec(Op.Mod, "OP_MOD", 0, -1),
    OpSpec(Op.Pow, "OP_POW", 0, -1),
    OpSpec(Op.AddWrap, "OP_ADD_WRAP", 0, -1),
    OpSpec(Op.SubWrap, "OP_SUB_WRAP", 0, -1),
    OpSpec(Op.MulWrap, "OP_MUL_WRAP", 0, -1),
    OpSpec(Op.Neg, "OP_NEG", 0, 0),
    OpSpec(Op.Pos, "OP_POS", 0, 0),
    OpSpec(Op.BAnd, "OP_BAND", 0, -1),
    OpSpec(Op.BOr, "OP_BOR", 0, -1),
    OpSpec(Op.BXor, "OP_BXOR", 0, -1),
    OpSpec(Op.Shl, "OP_SHL", 0, -1),
    OpSpec(Op.Shr, "OP_SHR", 0, -1),
    OpSpec(Op.BNot, "OP_BNOT", 0, 0),
    OpSpec(Op.Eq, "OP_EQ", 0, -1),
    OpSpec(Op.Ne, "OP_NE", 0, -1),
    OpSpec(Op.Lt, "OP_LT", 0, -1),
    OpSpec(Op.Le, "OP_LE", 0, -1),
    OpSpec(Op.Gt, "OP_GT", 0, -1),
    OpSpec(Op.Ge, "OP_GE", 0, -1),
    OpSpec(Op.Is, "OP_IS", 0, -1),
    OpSpec(Op.IsNot, "OP_IS_NOT", 0, -1),
    OpSpec(Op.In, "OP_IN", 0, -1),
    OpSpec(Op.NotIn, "OP_NOT_IN", 0, -1),
    OpSpec(Op.Not, "OP_NOT", 0, 0),
    OpSpec(Op.Concat, "OP_CONCAT", 0, -1),
    OpSpec(Op.AddIntConst, "OP_ADD_INT_CONST", 4, 1),
    OpSpec(Op.IncLocal, "OP_INC_LOCAL", 3, 0),
    OpSpec(Op.CmpLocalConstLt, "OP_CMP_LOCAL_CONST_LT", 4, 1),
    OpSpec(Op.GetLocal2, "OP_GET_LOCAL2", 4, 2),
    OpSpec(Op.AddLocals, "OP_ADD_LOCALS", 4, 1),
    OpSpec(Op.Jump, "OP_JUMP", 2, 0),
    OpSpec(Op.JumpIfFalse, "OP_JUMP_IF_FALSE", 2, -1),
    OpSpec(Op.JumpIfTrue, "OP_JUMP_IF_TRUE", 2, -1),
    OpSpec(Op.JumpIfFalseKeep, "OP_JUMP_IF_FALSE_KEEP", 2, 0),
    OpSpec(Op.JumpIfTrueKeep, "OP_JUMP_IF_TRUE_KEEP", 2, 0),
    OpSpec(Op.JumpIfNull, "OP_JUMP_IF_NULL", 2, 0),
    OpSpec(Op.Loop, "OP_LOOP", 2, 0),
    OpSpec(Op.GetIter, "OP_GET_ITER", 0, 0),
    OpSpec(Op.ForIter, "OP_FOR_ITER", 2, STACK_EFFECT_VARIABLE),
    OpSpec(Op.Call, "OP_CALL", 1, STACK_EFFECT_VARIABLE),
    OpSpec(Op.CallKw, "OP_CALL_KW", 4, STACK_EFFECT_VARIABLE),
    OpSpec(Op.CallSpread, "OP_CALL_SPREAD", 1, STACK_EFFECT_VARIABLE),
    OpSpec(Op.Invoke, "OP_INVOKE", 6, STACK_EFFECT_VARIABLE),
    OpSpec(Op.SuperInvoke, "OP_SUPER_INVOKE", 4, STACK_EFFECT_VARIABLE),
    OpSpec(Op.TailCall, "OP_TAIL_CALL", 1, STACK_EFFECT_VARIABLE),
    OpSpec(Op.Return, "OP_RETURN", 0, -1),
    OpSpec(Op.ReturnNull, "OP_RETURN_NULL", 0, 0),
    OpSpec(Op.Closure, "OP_CLOSURE", -1, 1),
    OpSpec(Op.BuildList, "OP_BUILD_LIST", 2, STACK_EFFECT_VARIABLE),
    OpSpec(Op.BuildDict, "OP_BUILD_DICT", 2, STACK_EFFECT_VARIABLE),
    OpSpec(Op.BuildSet, "OP_BUILD_SET", 2, STACK_EFFECT_VARIABLE),
    OpSpec(Op.BuildTuple, "OP_BUILD_TUPLE", 2, STACK_EFFECT_VARIABLE),
    OpSpec(Op.BuildRange, "OP_BUILD_RANGE", 1, -1),
    OpSpec(Op.ListAppend, "OP_LIST_APPEND", 2, -1),
    OpSpec(Op.DictInsert, "OP_DICT_INSERT", 2, -2),
    OpSpec(Op.SetAdd, "OP_SET_ADD", 2, -1),
    OpSpec(Op.GetIndex, "OP_GET_INDEX", 0, -1),
    OpSpec(Op.SetIndex, "OP_SET_INDEX", 0, -3),
    OpSpec(Op.GetSlice, "OP_GET_SLICE", 1, STACK_EFFECT_VARIABLE),
    OpSpec(Op.SetSlice, "OP_SET_SLICE", 1, STACK_EFFECT_VARIABLE),
    OpSpec(Op.Unpack, "OP_UNPACK", 2, STACK_EFFECT_VARIABLE),
    OpSpec(Op.Class, "OP_CLASS", 3, 1),
    OpSpec(Op.Inherit, "OP_INHERIT", 0, 0),
    OpSpec(Op.ImplTrait, "OP_IMPL_TRAIT", 3, 0),
    OpSpec(Op.Method, "OP_METHOD", 5, -1),
    OpSpec(Op.FieldDef, "OP_FIELD_DEF", 4, 0),
    OpSpec(Op.GetField, "OP_GET_FIELD", 5, 0),
    OpSpec(Op.SetField, "OP_SET_FIELD", 5, -2),
    OpSpec(Op.GetSuper, "OP_GET_SUPER", 3, 0),
    OpSpec(Op.New, "OP_NEW", 4, STACK_EFFECT_VARIABLE),
    OpSpec(Op.EnumNew, "OP_ENUM_NEW", 5, STACK_EFFECT_VARIABLE),
    OpSpec(Op.EnumTag, "OP_ENUM_TAG", 0, 1),
    OpSpec(Op.EnumField, "OP_ENUM_FIELD", 1, 1),
    OpSpec(Op.IsInstance, "OP_IS_INSTANCE", 3, 0),
    OpSpec(Op.Throw, "OP_THROW", 0, -1),
    OpSpec(Op.Reraise, "OP_RERAISE", 0, 0),
    OpSpec(Op.PushHandler, "OP_PUSH_HANDLER", 5, 0),
    OpSpec(Op.PopHandler, "OP_POP_HANDLER", 0, 0),
    OpSpec(Op.PushFinally, "OP_PUSH_FINALLY", 2, 0),
    OpSpec(Op.EndFinally, "OP_END_FINALLY", 0, 0),
    OpSpec(Op.PushDefer, "OP_PUSH_DEFER", 3, 0),
    OpSpec(Op.RunDefers, "OP_RUN_DEFERS", 0, 0),
    OpSpec(Op.MatchExc, "OP_MATCH_EXC", 3, 1),
    OpSpec(Op.GetExc, "OP_GET_EXC", 0, 1),
    OpSpec(Op.MatchConst, "OP_MATCH_CONST", 5, 0),
    OpSpec(Op.MatchRange, "OP_MATCH_RANGE", 9, 0),
    OpSpec(Op.MatchType, "OP_MATCH_TYPE", 5, 0),
    OpSpec(Op.MatchSeq, "OP_MATCH_SEQ", 4, 0),
    OpSpec(Op.MatchFields, "OP_MATCH_FIELDS", 5, STACK_EFFECT_VARIABLE),
    OpSpec(Op.Bind, "OP_BIND", 2, -1),
    OpSpec(Op.Import, "OP_IMPORT", 3, 1),
    OpSpec(Op.ImportFrom, "OP_IMPORT_FROM", 3, 1),
    OpSpec(Op.Export, "OP_EXPORT", 3, 0),
    OpSpec(Op.AssertFail, "OP_ASSERT_FAIL", 3, 0),
    OpSpec(Op.TypeGuard, "OP_TYPE_GUARD", 3, 0),
    OpSpec(Op.Halt, "OP_HALT", 0, 0),
    OpSpec(Op.GetFieldLocal, "OP_GET_FIELD_LOCAL", 7, 1),
    OpSpec(Op.ForIterBind, "OP_FOR_ITER_BIND", 4, 0),
    OpSpec(Op.JumpIfCmpFalse, "OP_JUMP_IF_CMP_FALSE", 3, -2),
    OpSpec(Op.Format, "OP_FORMAT", 9, STACK_EFFECT_VARIABLE),
    OpSpec(Op.JumpIfCmpLocalK, "OP_JUMP_IF_CMP_LOCAL_K", 8, 0),
    OpSpec(Op.ToFloat, "OP_TO_FLOAT", 0, 0),
    OpSpec(Op.ModIntConst, "OP_MOD_INT_CONST", 2, 0),
    OpSpec(Op.AddBind, "OP_ADD_BIND", 2, -2),
    OpSpec(Op.SubBind, "OP_SUB_BIND", 2, -2),
    OpSpec(Op.MulBind, "OP_MUL_BIND", 2, -2),
    OpSpec(Op.ElemKind, "OP_ELEM_KIND", 1, 0),
    OpSpec(Op.GetIterItems, "OP_GET_ITER_ITEMS", 2, 0),
    OpSpec(Op.ForIterPair, "OP_FOR_ITER_PAIR", 6, 0),
    OpSpec(Op.IterRange, "OP_ITER_RANGE", 5, -2),
    OpSpec(Op.ForRangeBind, "OP_FOR_RANGE_BIND", 8, 0),
]

#: Most parts one `OP_FORMAT` can join; `JAI_FMT_MAX_PARTS` in `src/vm/value.h`,
#: and also the width of its literal mask.
pub const FMT_MAX_PARTS: int = 24

let _OP_BYTES: dict[Op, int] = _build_op_bytes()

fn _build_op_bytes() -> dict[Op, int] {
    var table: dict[Op, int] = {}
    for (index, spec) in _OPS.enumerate() { table[spec.op] = index }
    return table
}

#: The byte that encodes `op`.
pub fn opcode(op: Op) -> int {
    return _OP_BYTES[op]
}

#: The `OP_`-prefixed name of the opcode byte `code`, as the C disassembler
#: prints it.
pub fn op_name(code: int) -> str {
    if code < 0 or code >= _OPS.len() { return "OP_UNKNOWN" }
    return _OPS[code].name
}

#: Bytes of operand that follow the opcode byte `code`, or -1 for `OP_CLOSURE`,
#: whose operand run depends on the capture count of the function it names.
pub fn op_operand_bytes(code: int) -> int {
    if code < 0 or code >= _OPS.len() { return 0 }
    return _OPS[code].operands
}

#: Net change `code` makes to the operand stack, or `STACK_EFFECT_VARIABLE`
#: when it depends on the operands.
pub fn op_stack_effect(code: int) -> int {
    if code < 0 or code >= _OPS.len() { return STACK_EFFECT_VARIABLE }
    return _OPS[code].effect
}

#: Number of opcodes in the instruction set.
pub fn op_count() -> int {
    return _OPS.len()
}

#: The constant-pool tags of `spec/BYTECODE.md` §4, in tag order.
pub enum ConstTag { Null, Bool, Int, Float, Str, Bytes, Func, ClassSpec, Tuple, Type }

#: One constant-pool entry: a tag and the payload that tag implies.
pub class Const {
    #: Which of the ten payload shapes this is.
    pub let tag: ConstTag
    #: `bool`, `int`, `float`, `str`, `list[int]` of bytes, `FuncProto`,
    #: `ClassSpec`, `list[Const]` for a tuple, or a type name. Mutable because a
    #: class spec read back from a file names its own pool by index, and those
    #: indices become strings only once the pool is complete.
    pub var value: any

    #: Wrap `value` with its tag.
    pub fn init(self, tag: ConstTag, value: any) {
        self.tag = tag
        self.value = value
    }
}

#: One line-table row: where an instruction starts and the source it came from.
pub class LineEntry {
    #: Code offset of the first instruction covered.
    pub let offset: int
    #: Byte offset of the start of the source span.
    pub let span_start: int
    #: Byte offset one past the end of the source span.
    pub let span_end: int

    #: Record a row.
    pub fn init(self, offset: int, span_start: int, span_end: int) {
        self.offset = offset
        self.span_start = span_start
        self.span_end = span_end
    }
}

#: One row of a function's exception table: the guarded region and its handler.
pub class ExceptionEntry {
    #: First code offset covered.
    pub var start: int
    #: One past the last code offset covered.
    pub var end: int
    #: Where the handler begins.
    pub var handler: int
    #: Constant index of the caught class, or 0 for catch-all.
    pub var type_const: int

    #: Record a region.
    pub fn init(self, start: int, end: int, handler: int, type_const: int) {
        self.start = start
        self.end = end
        self.handler = handler
        self.type_const = type_const
    }
}

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
        if self.depth < 0 { self.depth = 0 }  # only reachable after an error
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
            # Compares both ends, matching src/vm/bytecode/chunk.c's recordSpan: keying on
            # start alone previously made the line table come out too short.
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

# Null when two equal values must stay distinct: two functions with identical
# code are still two functions.
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
            # Null the moment a member has no key of its own: `constEqual` falls
            # back to object identity for a function, so a tuple holding one is
            # shared only with itself.
            var parts: list[str] = []
            for item in value.value {
                let key = _const_key(item)
                if key is null { return null }
                # Length-prefixed, so that `("a", "b")` and `("ab",)` cannot
                # land on one entry: a member's key is arbitrary text.
                parts.push(f"{key.len()}:{key}")
            }
            return "p" + "".join(parts)
        }
        _ => { return null }
    }
}

#: One field of a class spec (`spec/BYTECODE.md` §6).
pub class FieldSpec {
    #: The field name.
    pub let name: str
    #: 0 private, 1 protected, 2 public.
    pub let visibility: int
    #: True for a class-level field, which occupies no instance slot.
    pub let is_static: bool
    #: Constant index of the declared type, or 0 when untyped.
    pub let type_index: int

    #: Describe one field.
    pub fn init(self, name: str, visibility: int, is_static: bool, type_index: int) {
        self.name = name
        self.visibility = visibility
        self.is_static = is_static
        self.type_index = type_index
    }
}

#: One method of a class spec (`spec/BYTECODE.md` §6).
pub class MethodSpec {
    #: The method name.
    pub let name: str
    #: 0 private, 1 protected, 2 public.
    pub let visibility: int
    #: `FN_*` bits: method, static, getter, setter, variadic, kwrest.
    pub let flags: int
    #: Index of the method's function record in the enclosing pool.
    pub let func_index: int

    #: Describe one method.
    pub fn init(self, name: str, visibility: int, flags: int, func_index: int) {
        self.name = name
        self.visibility = visibility
        self.flags = flags
        self.func_index = func_index
    }
}

#: A class as the constant pool stores it, from which `OP_CLASS` builds one.
pub class ClassSpec {
    #: The class name.
    pub let name: str
    #: The superclass name, or null.
    pub let super_name: str?
    #: Names of the implemented traits.
    pub let traits: list[str]
    #: Fields in declaration order; slots are assigned parents first.
    pub let fields: list[FieldSpec]
    #: Methods, static methods, getters and setters.
    pub let methods: list[MethodSpec]

    #: Describe one class.
    pub fn init(
        self,
        name: str,
        super_name: str?,
        traits: list[str],
        fields: list[FieldSpec],
        methods: list[MethodSpec]
    ) {
        self.name = name
        self.super_name = super_name
        self.traits = traits
        self.fields = fields
        self.methods = methods
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

class LoopFrame {
    pub let label: str?
    pub let continue_target: int
    #: Lowest slot to close when an iteration ends, or -1. See `_close_iteration`.
    pub let capture_base: int
    #: Operand-stack depth `continue` and `break` unwind to. A `for` leaves its
    #: iterator pushed across the body, so `continue` keeps it and `break` drops it.
    pub let continue_depth: int
    pub let break_depth: int
    pub var breaks: list[int]

    pub fn init(
        self,
        label: str?,
        continue_target: int,
        capture_base: int,
        continue_depth: int,
        break_depth: int
    ) {
        self.label = label
        self.continue_target = continue_target
        self.capture_base = capture_base
        self.continue_depth = continue_depth
        self.break_depth = break_depth
        self.breaks = []
    }
}

class FnCtx {
    pub let proto: FuncProto
    pub var loops: list[LoopFrame]
    #: Lowest capture base of the blocks open around the cursor, or -1.
    #: `return` leaves all of them at once and closes at this.
    pub var close_base: int
    #: Watermark, not a stack: the lowest base of any block or loop *opened*
    #: since it was last reset. `_try` resets it around the guarded region,
    #: since an exception skips every block close inside that region.
    pub var deep_close_base: int
    #: Next free temporary slot, handed out above the resolver's peak. A
    #: temporary lives in the frame window, not on the operand stack, so a
    #: pattern test can fail without unwinding partially destructured values.
    pub var next_temp: int
    #: High-water mark of `next_temp`; the frame reserves this many slots
    #: before the operand stack starts.
    pub var max_slot: int
    #: The `finally` blocks open around the cursor, innermost last. An explicit
    #: `return`, `break` or `continue` leaving them inlines a copy of each; the
    #: exception path uses the exception table instead.
    pub var finallys: list[Node]
    #: Number of loop frames open when each `finally` was entered, so that a
    #: `break` knows which of them it is leaving.
    pub var finally_loop_depth: list[int]
    #: Above zero inside a `try`, `catch`, `finally` or deferred block. A
    #: `return` there may not reuse the frame: see `_can_tail_call`.
    pub var protect_depth: int
    #: True once this frame has registered a `defer`. Set where PUSH_DEFER is
    #: emitted, so a `return` above the `defer` sees false, exactly as the VM
    #: would find no defer registered yet.
    pub var has_defer: bool

    pub fn init(self, proto: FuncProto, local_count: int) {
        self.proto = proto
        self.loops = []
        self.close_base = -1
        self.deep_close_base = -1
        self.next_temp = if local_count < 1 { 1 } else { local_count }
        self.max_slot = self.next_temp
        self.finallys = []
        self.finally_loop_depth = []
        self.protect_depth = 0
        self.has_defer = false
    }
}

#: Lower a resolved program to a `FuncProto` for the module body.
#:
#: Run `jaithon.compile.resolve` first: the emitter reads the symbols, slots and
#: capture lists that pass left behind and looks nothing up itself.
pub class Emitter {
    #: The resolver's output: symbols by node identity, and function scopes.
    pub let resolution: Resolution
    #: Where a construct the instruction set cannot express is reported.
    pub let bag: DiagnosticBag
    #: When true `assert` compiles to nothing, as `--release` requires.
    pub let strip_asserts: bool
    #: The `--O` level handed to `optimize`: 0 disables passes 3-7 (spec §8).
    pub let opt_level: int
    #: The optimiser run over every function this emitter finishes; see
    #: `_no_optimize` for why it is a parameter.
    pub let optimize: fn(FuncProto, int) -> void

    #: The file's source text, so an assert can bake in its condition. Null
    #: falls back to the bare `"assertion failed"`, as the C does.
    pub var source: str?
    #: Enum declarations this file imported, by the name they are bound to here.
    #: An enum pattern needs the variant's tag, which is the declaration's
    #: variant order and for an imported enum lives in another file. The C reads
    #: it out of the checker's registry (`jaiTypeDeclFind` in
    #: `src/codegen/codegen_pattern.c`); here the checker hands the enums over.
    pub var imported_enums: dict[str, Node]
    var stack: list[FnCtx]
    var enums: dict[str, Node]

    #: Create an emitter over `resolution`, reporting into `bag`.
    pub fn init(
        self,
        resolution: Resolution,
        bag: DiagnosticBag,
        strip_asserts: bool = false,
        opt_level: int = 2,
        optimize: fn(FuncProto, int) -> void = _no_optimize
    ) {
        self.resolution = resolution
        self.bag = bag
        self.strip_asserts = strip_asserts
        self.opt_level = opt_level
        self.optimize = optimize
        self.source = null
        self.imported_enums = {}
        self.stack = []
        self.enums = {}
    }

    #: Lower `program` and return the module body's function record.
    pub fn compile_program(self, program: Node, module_name: str? = null) -> FuncProto {
        let proto = FuncProto(module_name ?? "__main__", 0, 0)
        # Imported enums go in first so that a name this file also declares
        # wins, which is the order the C's registry has them in.
        for name in self.imported_enums.keys() { self.enums[name] = self.imported_enums[name] }
        self._index_enums(program.child_list("stmts"))
        self.stack.push(FnCtx(proto, self.resolution.slot_count(program)))
        # Top-level `fn` declarations bind ahead of the rest (STYLE §8 ordering),
        # so a statement may call a function declared below it.
        let stmts = program.child_list("stmts")
        for statement in stmts {
            if statement.kind == NodeKind.FnDecl { self._stmt(statement) }
        }
        for statement in stmts {
            if statement.kind != NodeKind.FnDecl { self._stmt(statement) }
        }
        self._run_defers(program.span)
        self._op(Op.ReturnNull, program.span)
        self._finish()
        return proto
    }

    # Frame size is locals + stack high-water mark. This is where the C's
    # jaiCgFinishFunction calls jaiOptimize: once per function, after frame size is final.
    fn _finish(self) -> void {
        let ctx = self._ctx()
        ctx.proto.max_slots = ctx.max_slot + ctx.proto.chunk.max_depth
        self.optimize(ctx.proto, self.opt_level)
        self.stack.pop()
    }

    fn _index_enums(self, stmts: list[Node]) -> void {
        for node in stmts {
            if node.kind == NodeKind.EnumDecl {
                self.enums[node.text("name")] = node
            } elif node.kind == NodeKind.Block {
                self._index_enums(node.child_list("stmts"))
            }
        }
    }

    fn _chunk(self) -> Chunk { return self.stack[self.stack.len() - 1].proto.chunk }

    fn _ctx(self) -> FnCtx { return self.stack[self.stack.len() - 1] }

    fn _op(self, op: Op, span: Span) -> int {
        return self._chunk().write_op(op, span)
    }

    fn _op_u8(self, op: Op, value: int, span: Span) -> void {
        self._chunk().write_op(op, span)
        self._chunk().write_byte(value)
    }

    fn _op_u16(self, op: Op, value: int, span: Span) -> void {
        self._chunk().write_op(op, span)
        self._chunk().write_u16(value)
    }

    fn _op_k(self, op: Op, index: int, span: Span) -> void {
        self._chunk().write_op(op, span)
        self._chunk().write_u24(index)
    }

    fn _op_cached(self, op: Op, index: int, span: Span) -> void {
        self._chunk().write_op(op, span)
        self._chunk().write_u24(index)
        self._chunk().write_u16(self._chunk().add_cache())
    }

    fn _string(self, text: str) -> int {
        return self._chunk().add_constant(Const(ConstTag.Str, text))
    }

    fn _depth(self) -> int { return self._chunk().depth }

    fn _adjust(self, delta: int) -> void { self._chunk().adjust(delta) }

    fn _set_depth(self, value: int) -> void { self._chunk().set_depth(value) }

    # POPN carries a u8, so a single value is POP instead.
    fn _pop_n(self, count: int, span: Span) -> void {
        var left = count
        while left > 0 {
            if left == 1 {
                self._op(Op.Pop, span)
                return
            }
            let step = if left > 255 { 255 } else { left }
            self._op_u8(Op.PopN, step, span)
            self._adjust(-step)
            left -= step
        }
    }

    fn _alloc_temp(self) -> int {
        let ctx = self._ctx()
        let slot = ctx.next_temp
        ctx.next_temp += 1
        if ctx.next_temp > ctx.max_slot { ctx.max_slot = ctx.next_temp }
        return slot
    }

    fn _free_temps(self, count: int) -> void {
        let ctx = self._ctx()
        ctx.next_temp -= count
        if ctx.next_temp < 0 { ctx.next_temp = 0 }
    }

    fn _get_local(self, slot: int, span: Span) -> void {
        self._op_u16(Op.GetLocal, if slot < 0 { 0 } else { slot }, span)
    }

    fn _emit_int(self, value: int, span: Span) -> void {
        if value >= -32768 and value <= 32767 {
            self._chunk().write_op(Op.Int, span)
            self._chunk().write_i16(value)
            return
        }
        self._op_k(Op.Const, self._chunk().add_constant(Const(ConstTag.Int, value)), span)
    }

    fn _bind_local(self, slot: int, span: Span) -> void {
        self._op_u16(Op.Bind, if slot < 0 { 0 } else { slot }, span)
    }

    fn _emit_jump(self, op: Op, span: Span) -> int {
        self._chunk().write_op(op, span)
        let operand = self._chunk().here()
        self._chunk().write_u16(0)
        return operand
    }

    # A jump offset is measured from the end of its instruction: the byte after
    # the operand for every jump but PUSH_HANDLER, whose i16 comes before a u24.
    fn _patch(self, operand: int) -> void {
        self._patch_from(operand, operand + 2)
    }

    fn _patch_from(self, operand: int, base: int) -> void {
        self._chunk().patch_i16(operand, self._chunk().here() - base)
    }

    fn _patch_all(self, operands: list[int]) -> void {
        for operand in operands { self._patch(operand) }
    }

    fn _loop_back(self, target: int, span: Span) -> void {
        self._chunk().write_op(Op.Loop, span)
        let operand = self._chunk().here()
        self._chunk().write_i16(target - (operand + 2))
    }

    fn _unsupported(self, span: Span, what: str) -> void {
        self.bag.error(
            SemaCode.E0902_INTERNAL_ERROR,
            span,
            f"the self-hosted back end cannot lower {what}"
        )
    }

    fn _load_name(self, node: Node, name: str) -> void {
        self._load_symbol(self.resolution.symbol_of(node), name, node.span)
    }

    fn _load_symbol(self, symbol: Symbol?, name: str, span: Span) -> void {
        # A missing symbol is the recovery path: a late-bound global, as in a REPL.
        if symbol == null {
            self._op_cached(Op.GetGlobal, self._string(name), span)
            return
        }
        if symbol.is_slotted() {
            self._get_local(symbol.slot, span)
        } elif symbol.kind == SymbolKind.Upvalue {
            self._op_u8(Op.GetUpvalue, if symbol.slot < 0 { 0 } else { symbol.slot }, span)
        } elif symbol.kind == SymbolKind.Field {
            # An unqualified field name inside a method is `self.name`.
            self._op_u16(Op.GetLocal, 0, span)
            self._op_cached(Op.GetField, self._string(symbol.name), span)
        } else {
            self._op_cached(Op.GetGlobal, self._string(symbol.name), span)
        }
    }

    fn _store_symbol(self, symbol: Symbol?, name: str, span: Span, declaring: bool) -> void {
        if symbol == null or _symbol_is_global_like(symbol) {
            let bound = if symbol == null { name } else { symbol.name }
            if declaring {
                self._op_k(Op.DefGlobal, self._string(bound), span)
            } else {
                self._op_cached(Op.SetGlobal, self._string(bound), span)
                self._op(Op.Pop, span)
            }
            return
        }
        if symbol.is_slotted() {
            self._bind_local(symbol.slot, span)
        } elif symbol.kind == SymbolKind.Upvalue {
            self._op_u8(Op.SetUpvalue, if symbol.slot < 0 { 0 } else { symbol.slot }, span)
            self._op(Op.Pop, span)
        } else {
            # A field: SET_FIELD wants (object, value) and the value is on top.
            self._op_u16(Op.GetLocal, 0, span)
            self._op(Op.Swap, span)
            self._op_cached(Op.SetField, self._string(symbol.name), span)
        }
    }

    fn _bind_pattern(self, pattern: Node?, span: Span) -> void {
        if pattern == null {
            self._op(Op.Pop, span)
            return
        }
        let kind = pattern.kind
        if kind == NodeKind.PatWildcard {
            self._op(Op.Pop, span)
        } elif kind == NodeKind.PatBind {
            let symbol = self.resolution.symbol_of(pattern)
            if symbol == null or symbol.kind == SymbolKind.Global {
                let name = if symbol == null { pattern.text("name") } else { symbol.name }
                self._op_k(Op.DefGlobal, self._string(name), span)
            } elif symbol.kind == SymbolKind.Upvalue {
                self._op_u8(Op.SetUpvalue, symbol.slot, span)
                self._op(Op.Pop, span)
            } else {
                self._op_u16(Op.Bind, symbol.slot, span)
            }
        } elif kind == NodeKind.PatTuple or kind == NodeKind.PatList {
            let elements = pattern.child_list("elems")
            let rest = pattern.number("restIndex")
            self._chunk().write_op(Op.Unpack, span)
            self._chunk().write_byte(elements.len())
            self._chunk().write_byte(if rest < 0 { 255 } else { rest })
            self._adjust(elements.len() - 1)
            # UNPACK pushes right to left, so element 0 is on top.
            for element in elements { self._bind_pattern(element, span) }
        } else {
            self._unsupported(pattern.span, "this pattern in a binding position")
            self._op(Op.Pop, span)
        }
    }

    fn _each_expr(self, nodes: list[Node]) -> void {
        for node in nodes { self._expr(node) }
    }

    fn _expr(self, node: Node?) -> void {
        if node == null {
            self._op(Op.Null, Span.none())
            return
        }
        let span = node.span
        match node.kind {
            NodeKind.IntLit => { self._emit_int(node.number("intLit"), span) }
            NodeKind.FloatLit => {
                self._op_k(
                    Op.Const,
                    self._chunk().add_constant(Const(ConstTag.Float, node.get("floatLit"))),
                    span
                )
            }
            NodeKind.StrLit => { self._op_k(Op.Const, self._string(node.text("chars")), span) }
            NodeKind.BoolLit => {
                self._op(if node.flag("boolLit") { Op.True } else { Op.False }, span)
            }
            NodeKind.NullLit => { self._op(Op.Null, span) }
            NodeKind.FString => { self._fstring(node) }
            NodeKind.Ident => { self._load_name(node, node.text("name")) }
            # Slot 0 is this frame's receiver; a nested function has none, so the
            # resolver binds this node to the enclosing method's captured `self`.
            NodeKind.Self | NodeKind.Super => {
                let captured = self.resolution.symbol_of(node)
                if captured == null {
                    self._op_u16(Op.GetLocal, 0, span)
                } else {
                    self._load_symbol(captured, "self", span)
                }
            }
            NodeKind.ListLit => { self._sequence_literal(node, Op.BuildList) }
            NodeKind.SetLit => { self._sequence_literal(node, Op.BuildSet) }
            NodeKind.TupleLit => { self._sequence_literal(node, Op.BuildTuple) }
            NodeKind.DictLit => {
                let keys = node.child_list("keys")
                let values = node.child_list("values")
                for (index, key) in keys.enumerate() {
                    self._expr(key)
                    self._expr(values[index])
                }
                self._op_u16(Op.BuildDict, keys.len(), span)
                self._adjust(-2 * keys.len() + 1)
            }
            NodeKind.Unary => {
                self._expr(node.child("operand"))
                self._op(_unary_op(node.op()), span)
            }
            NodeKind.Binary => {
                self._expr(node.child("left"))
                self._expr(node.child("right"))
                self._op(_binary_op(node.op()), span)
            }
            NodeKind.Logical => { self._logical(node) }
            NodeKind.CompareChain => { self._chain(node) }
            NodeKind.Ternary | NodeKind.IfExpr => {
                self._expr(node.child("cond"))
                let otherwise = self._emit_jump(Op.JumpIfFalse, span)
                let base = self._depth()
                self._branch_value(_then_of(node), span)
                let done = self._emit_jump(Op.Jump, span)
                self._set_depth(base)
                self._patch(otherwise)
                self._branch_value(_else_of(node), span)
                self._set_depth(base + 1)
                self._patch(done)
            }
            NodeKind.Coalesce => {
                self._expr(node.child("left"))
                let absent = self._emit_jump(Op.JumpIfNull, span)
                let done = self._emit_jump(Op.Jump, span)
                let base = self._depth() - 1
                self._set_depth(base + 1)
                self._patch(absent)
                self._op(Op.Pop, span)
                self._expr(node.child("right"))
                self._set_depth(base + 1)
                self._patch(done)
            }
            NodeKind.Call => { self._call(node) }
            NodeKind.Index => {
                self._expr(node.child("object"))
                # Generics are erased (spec §6.1): `Box[int]` *is* `Box`, so
                # there is nothing to evaluate and nothing to subscript.
                if not self.resolution.is_type_args(node) {
                    self._expr(node.child("index"))
                    self._op(Op.GetIndex, span)
                }
            }
            NodeKind.Slice => {
                self._expr(node.child("object"))
                let before = self._depth()
                self._op_u8(Op.GetSlice, self._slice_parts(node), span)
                self._set_depth(before)  # object out, one slice in
            }
            NodeKind.Member => {
                let object = node.require("object")
                if object.kind == NodeKind.Super {
                    self._op_u16(Op.GetLocal, 0, span)
                    self._op_k(Op.GetSuper, self._string(node.text("name")), span)
                } else {
                    self._expr(object)
                    self._op_cached(Op.GetField, self._string(node.text("name")), span)
                }
            }
            NodeKind.OptMember => {
                self._expr(node.child("object"))
                let absent = self._emit_jump(Op.JumpIfNull, span)
                self._op_cached(Op.GetField, self._string(node.text("name")), span)
                self._patch(absent)
            }
            NodeKind.Lambda | NodeKind.AnonFn => { self._closure(node, null, 0) }
            NodeKind.Comprehension => { self._comprehension(node) }
            NodeKind.Range => {
                self._expr(node.child("start"))
                self._expr(node.child("stop"))
                self._op_u8(Op.BuildRange, if node.flag("inclusive") { 1 } else { 0 }, span)
            }
            NodeKind.MatchExpr => { self._match(node, true) }
            NodeKind.Cast => {
                self._expr(node.child("operand"))
                if node.flag("widen") {
                    # An int the checker proved is an int: one conversion, no
                    # test (spec §2.2).
                    self._op(Op.ToFloat, span)
                } else {
                    let target = node.type_field("target")
                    self._op_k(Op.TypeGuard, self._type_guard_const(target), span)
                }
            }
            NodeKind.ThrowExpr => {
                self._expr(node.child("operand"))
                self._op(Op.Throw, span)
            }
            NodeKind.Block => { self._block_expr(node) }
            NodeKind.Yield => {
                self._unsupported(span, "`yield`: the instruction set has no generators")
                self._op(Op.Null, span)
            }
            NodeKind.Await => {
                self._unsupported(span, "`await`: the instruction set has no coroutines")
                self._op(Op.Null, span)
            }
            _ => {
                self._unsupported(span, f"{node.kind} in expression position")
                self._op(Op.Null, span)
            }
        }
    }

    # Never a runtime eval (spec §1.7). The literal mask keeps a program that
    # rebinds global `str` from getting it called on literal parts, only on holes.
    fn _fstring(self, node: Node) -> void {
        let span = node.span
        let parts = node.child_list("parts")
        if parts.len() == 0 {
            self._op_k(Op.Const, self._string(""), span)
            return
        }
        if parts.len() == 1 and parts[0].kind == NodeKind.StrLit {
            self._expr(parts[0])
            return
        }

        if parts.len() <= FMT_MAX_PARTS {
            var litmask = 0
            for (index, part) in parts.enumerate() {
                if part.kind == NodeKind.StrLit { litmask = litmask | 1 << index }
                self._expr(part)
            }
            self._chunk().write_op(Op.Format, span)
            self._chunk().write_byte(parts.len())
            self._chunk().write_u24(litmask)
            self._chunk().write_u24(self._string("str"))
            self._chunk().write_u16(self._chunk().add_cache())
            self._adjust(-parts.len() + 1)
            return
        }

        # More holes than the mask holds: fall back to one join.
        self._op_k(Op.Const, self._string(""), span)
        for part in parts { self._fstring_part(part) }
        self._op_u16(Op.BuildList, parts.len(), span)
        self._adjust(-parts.len() + 1)
        self._invoke_name("join", 1, span)
    }

    fn _fstring_part(self, part: Node) -> void {
        if part.kind == NodeKind.StrLit {
            self._expr(part)
            return
        }
        self._op_cached(Op.GetGlobal, self._string("str"), part.span)
        self._expr(part)
        self._op_u8(Op.Call, 1, part.span)
        self._adjust(-1)
    }

    fn _logical(self, node: Node) -> void {
        self._expr(node.child("left"))
        var keep = Op.JumpIfTrueKeep
        if node.op() == OpKind.And { keep = Op.JumpIfFalseKeep }
        let short_circuit = self._emit_jump(keep, node.span)
        self._op(Op.Pop, node.span)
        self._expr(node.child("right"))
        self._patch(short_circuit)
    }

    # `a < b <= c` evaluates `b` once: it is stashed in a temporary between the
    # two comparisons, which keeps every short-circuit edge at one bool deep.
    fn _chain(self, node: Node) -> void {
        let span = node.span
        let operands = node.child_list("operands")
        let ops: list[OpKind] = node.get("ops")
        if ops.len() == 0 {
            self._expr(operands[0])
            return
        }

        let temp = if ops.len() > 1 { self._alloc_temp() } else { -1 }
        var short_circuit: list[int] = []
        let base = self._depth()

        self._expr(operands[0])
        for (index, op) in ops.enumerate() {
            self._expr(operands[index + 1])
            if index + 1 < ops.len() {
                self._op_u16(Op.SetLocal, temp, span)  # peeks: keeps the operand
            }
            self._op(_binary_op(op), span)
            if index + 1 < ops.len() {
                short_circuit.push(self._emit_jump(Op.JumpIfFalseKeep, span))
                self._op(Op.Pop, span)
                self._get_local(temp, span)
            }
        }

        self._set_depth(base + 1)
        self._patch_all(short_circuit)
        if temp >= 0 { self._free_temps(1) }
    }

    fn _sequence_literal(self, node: Node, build: Op) -> void {
        let items = node.child_list("items")
        self._each_expr(items)
        self._op_u16(build, items.len(), node.span)
        self._adjust(-items.len() + 1)
    }

    fn _slice_parts(self, node: Node) -> int {
        var flags = 0
        if node.child("start") != null {
            self._expr(node.child("start"))
            flags = flags | 1
        }
        if node.child("stop") != null {
            self._expr(node.child("stop"))
            flags = flags | 2
        }
        if node.child("step") != null {
            self._expr(node.child("step"))
            flags = flags | 4
        }
        return flags
    }

    # `tail` selects TAIL_CALL for the plain positional case. It also suppresses
    # the INVOKE fusion, because INVOKE has no tail form.
    fn _call(self, node: Node, tail: bool = false) -> void {
        let span = node.span
        let callee = node.require("callee")
        var keywords: list[str] = []
        var positional: list[Node] = []
        var spread: Node? = null

        for argument in node.records("args") {
            let value: Node = argument["value"]
            if argument["isSpread"] {
                spread = value
            } elif argument["name"] != null {
                keywords.push(argument["name"])
            } else {
                positional.push(value)
            }
        }

        # `super(a)` and `super.m(a)` fuse into SUPER_INVOKE, receiver in slot 0.
        if callee.kind == NodeKind.Super {
            self._super_call(node, "init", positional, span)
            return
        }
        if callee.kind == NodeKind.Member and callee.require("object").kind == NodeKind.Super {
            self._super_call(node, callee.text("name"), positional, span)
            return
        }

        # `obj.m(a)` fuses into INVOKE, which carries the method cache. An
        # `Enum.Variant(a)` is a static member, so this reaches it too.
        if callee.kind == NodeKind.Member and keywords.len() == 0 and spread == null and not tail {
            self._expr(callee.require("object"))
            self._each_expr(positional)
            self._invoke_name(callee.text("name"), positional.len(), span)
            return
        }

        # `a?.m(x)` yields null *without calling* when `a` is null (spec §4.2),
        # so the guard skips the arguments and the call, not just the load.
        var opt_skip = -1
        var is_tail = tail
        if callee.kind == NodeKind.OptMember {
            self._expr(callee.require("object"))
            opt_skip = self._emit_jump(Op.JumpIfNull, span)
            self._op_cached(Op.GetField, self._string(callee.text("name")), span)
            # The frame has to survive to the patch target below.
            is_tail = false
        } else {
            self._expr(callee)
        }

        self._each_expr(positional)
        if spread != null {
            self._expr(spread)
            # The operand counts every slot the call consumes, the splat list
            # included (BYTECODE.md §3.5).
            self._op_u8(Op.CallSpread, positional.len() + 1, span)
            self._adjust(-(positional.len() + 1))
            if opt_skip >= 0 { self._patch(opt_skip) }
            return
        }
        if keywords.len() > 0 {
            var names: list[Const] = []
            for name in keywords { names.push(Const(ConstTag.Str, name)) }
            var keyword_count = 0
            for argument in node.records("args") {
                if argument["name"] != null {
                    self._expr(argument["value"])
                    keyword_count += 1
                }
            }
            self._chunk().write_op(Op.CallKw, span)
            self._chunk().write_byte(positional.len())
            self._chunk().write_u24(self._chunk().add_constant(Const(ConstTag.Tuple, names)))
            self._adjust(-(positional.len() + keyword_count))
            if opt_skip >= 0 { self._patch(opt_skip) }
            return
        }
        self._op_u8(if is_tail { Op.TailCall } else { Op.Call }, positional.len(), span)
        self._adjust(-positional.len())
        if opt_skip >= 0 { self._patch(opt_skip) }
    }

    fn _super_call(self, node: Node, method: str, positional: list[Node], span: Span) -> void {
        self._op_u16(Op.GetLocal, 0, span)
        self._each_expr(positional)
        self._chunk().write_op(Op.SuperInvoke, span)
        self._chunk().write_u24(self._string(method))
        self._chunk().write_byte(positional.len())
        self._adjust(-positional.len())
    }

    fn _invoke_name(self, name: str, argc: int, span: Span) -> void {
        let index = self._string(name)
        let cache = self._chunk().add_cache()
        self._chunk().write_op(Op.Invoke, span)
        self._chunk().write_u24(index)
        self._chunk().write_byte(argc)
        self._chunk().write_u16(cache)
        self._adjust(-argc)
    }

    # A block in expression position yields its final expression statement. The
    # exit close comes after the value: it may be a closure over a block binding.
    fn _block_expr(self, node: Node) -> void {
        let close_at = self.resolution.capture_base_of(node)
        let saved = self._enter_close_scope(close_at)
        self._block_expr_contents(node)
        self._close_scope(close_at, node.span)
        self._ctx().close_base = saved
    }

    fn _block_expr_contents(self, node: Node) -> void {
        let stmts = node.child_list("stmts")
        if stmts.len() == 0 {
            self._op(Op.Null, node.span)
            return
        }
        var index = 0
        while index < stmts.len() - 1 {
            self._stmt(stmts[index])
            index += 1
        }
        let last = stmts[stmts.len() - 1]
        if last.kind == NodeKind.ExprStmt {
            self._expr(last.child("expr"))
            return
        }
        # `if`/`match` at a block's tail are its value: emitting them as
        # statements plus a null push made every such block evaluate to null. A
        # bare `if` is left alone — no value on the untaken path.
        if last.kind == NodeKind.Match or
            last.kind == NodeKind.If and last.child("elseBranch") != null {
            self._branch_value(last, last.span)
            return
        }
        self._stmt(last)
        self._op(Op.Null, node.span)
    }

    # A branch isn't always an expression node: a block yields its tail, and an
    # `elif` chain built in statement position arrives as `If`.
    fn _branch_value(self, branch: Node?, span: Span) -> void {
        if branch == null {
            self._op(Op.Null, span)
            return
        }
        if branch.kind == NodeKind.Block {
            self._block_expr(branch)
            return
        }
        if branch.kind == NodeKind.Match {
            self._match(branch, true)
            return
        }
        if branch.kind == NodeKind.If {
            let base = self._depth()
            self._expr(branch.child("cond"))
            let otherwise = self._emit_jump(Op.JumpIfFalse, span)
            self._branch_value(branch.child("thenBranch"), span)
            let done = self._emit_jump(Op.Jump, span)
            self._set_depth(base)
            self._patch(otherwise)
            self._branch_value(branch.child("elseBranch"), span)
            self._set_depth(base + 1)
            self._patch(done)
            return
        }
        self._expr(branch)
    }

    # The accumulator is built first; the element lands `clauses + 1` above it.
    fn _comprehension(self, node: Node) -> void {
        let span = node.span
        let clauses = node.records("clauses")
        let kind: CompKind = node.get("compKind")
        if kind == CompKind.Generator {
            self._unsupported(span, "a generator expression: it needs a coroutine")
        }
        if kind == CompKind.Dict {
            self._op_u16(Op.BuildDict, 0, span)
        } elif kind == CompKind.Set {
            self._op_u16(Op.BuildSet, 0, span)
        } else {
            self._op_u16(Op.BuildList, 0, span)
        }
        self._adjust(1)
        let container_depth = self._depth() - 1

        var starts: list[int] = []
        var exits: list[int] = []
        var before_iter: list[int] = []
        var skips: list[list[int]] = []
        # Which clauses used the range lowering below: their exit is patched
        # like `_for_range`'s (`_patch_from(., . + 8)`, not the ordinary
        # two-byte `_patch`), and they own temps that must be freed on the
        # way out, in the same reverse order this loop already closes in.
        var is_range: list[bool] = []
        var range_temps: list[int] = []
        for clause in clauses {
            before_iter.push(self._depth())
            let iterable: Node = clause["iterable"]
            let pattern: Node = clause["pattern"]
            # `for x in a..b`: the same lowering `_for_range` gives the
            # statement form, and it declines for exactly the same reasons —
            # reused via `_range_bind_slot` rather than re-derived, since a
            # range reached through a variable, a destructuring pattern or a
            # global loop variable all fall back to the general path below.
            if iterable.kind == NodeKind.Range and self._range_bind_slot(pattern, false) >= 0 {
                is_range.push(true)
                let wildcard = pattern.kind == NodeKind.PatWildcard
                let counter = self._alloc_temp()
                let end = self._alloc_temp()
                let slot = self._range_bind_slot(pattern, wildcard)
                self._expr(iterable.child("start"))
                self._expr(iterable.child("stop"))
                self._chunk().write_op(Op.IterRange, span)
                self._chunk().write_byte(if iterable.flag("inclusive") { 1 } else { 0 })
                self._chunk().write_u16(counter)
                self._chunk().write_u16(end)
                starts.push(self._chunk().here())
                self._chunk().write_op(Op.ForRangeBind, span)
                exits.push(self._chunk().here())
                self._chunk().write_u16(0)
                self._chunk().write_u16(slot)
                self._chunk().write_u16(counter)
                self._chunk().write_u16(end)
                range_temps.push(if wildcard { 3 } else { 2 })
            } else {
                is_range.push(false)
                range_temps.push(0)
                # Plain `for` only, not comprehension clauses: a clause's jumps are
                # tracked by the comprehension itself, and threading another branch
                # through that bookkeeping broke `{k: v for (v, k) in d.items()}`.
                self._expr(iterable)
                self._op(Op.GetIter, span)
                starts.push(self._chunk().here())
                exits.push(self._emit_jump(Op.ForIter, span))
                self._adjust(1)  # the loop value, on the fall-through
                self._bind_pattern(pattern, span)
            }
            var failures: list[int] = []
            let conditions: list[Node] = clause["conditions"]
            for condition in conditions {
                self._expr(condition)
                failures.push(self._emit_jump(Op.JumpIfFalse, span))
            }
            skips.push(failures)
        }

        # The accumulator's peek index, below one iterator per clause that
        # pushed one — a range clause pushed nothing, so `self._depth()`
        # already reflects only the clauses that did.
        if kind == CompKind.Dict {
            self._expr(node.child("keyExpr"))
            self._expr(node.child("element"))
            self._op_u16(Op.DictInsert, self._depth() - 1 - container_depth, span)
        } elif kind == CompKind.Set {
            self._expr(node.child("element"))
            self._op_u16(Op.SetAdd, self._depth() - 1 - container_depth, span)
        } else {
            self._expr(node.child("element"))
            self._op_u16(Op.ListAppend, self._depth() - 1 - container_depth, span)
        }

        var index = clauses.len() - 1
        while index >= 0 {
            # A filter that fails continues its own clause, not an inner one.
            self._patch_all(skips[index])
            self._loop_back(starts[index], span)
            # Both paths land back at this clause's own depth: FOR_ITER drops
            # the exhausted iterator; a range clause never put anything on the
            # operand stack to begin with, so this is a no-op there rather
            # than an assumed one.
            self._set_depth(before_iter[index])
            if is_range[index] {
                # The displacement is measured from the end of the
                # instruction, six bytes past its own operand rather than the
                # usual zero — same as `_for_range`'s `OP_FOR_RANGE_BIND` patch.
                self._patch_from(exits[index], exits[index] + 8)
                self._free_temps(range_temps[index])
            } else {
                self._patch(exits[index])
            }
            index -= 1
        }
    }

    fn _closure(self, node: Node, name: str?, extra_flags: int) -> int {
        let index = self._function_constant(node, name, extra_flags)
        self._emit_closure(node, index)
        return index
    }

    fn _emit_closure(self, node: Node, index: int) -> void {
        self._op_k(Op.Closure, index, node.span)
        let scope = self.resolution.scope_of(node)
        if scope == null { return }
        for upvalue in scope.upvalues {
            # bit 0: the source is a local of the enclosing frame rather than
            # one of its upvalues. bit 1: capture by value (spec §6's `let`).
            var how = if upvalue.is_local { 1 } else { 0 }
            if upvalue.by_value { how = how | 2 }
            self._chunk().write_byte(how)
            self._chunk().write_u16(upvalue.index)
        }
    }

    fn _function_constant(
        self,
        node: Node,
        name: str?,
        extra_flags: int,
        init_fields: list[any]? = null
    ) -> int {
        let scope = self.resolution.scope_of(node)
        var arity = 0
        var flags = extra_flags
        if node.flag("isStatic") { flags = flags | FN_STATIC }
        if node.flag("isGenerator") { flags = flags | FN_GENERATOR }
        if node.flag("isAsync") { flags = flags | FN_ASYNC }

        var names: list[str] = []
        var defaults: list[Node] = []
        for param in node.records("params") {
            if param["name"] == "self" { continue }
            names.push(param["name"])
            if param["isVariadic"] {
                flags = flags | FN_VARIADIC
            } elif param["isKwRest"] {
                flags = flags | FN_KWREST
            } else {
                arity += 1
                if param["defaultValue"] != null { defaults.push(param["defaultValue"]) }
            }
        }

        # A lambda still gets a name (`<anonymous>`, matching the C's
        # setFunctionNames): the name is a pool constant (§5), so null diverges.
        let proto = FuncProto((name ?? node.opt_text("name")) ?? "<anonymous>", arity, flags)
        proto.upvalue_count = if scope == null { 0 } else { scope.upvalues.len() }
        proto.param_names = names

        # The frame holds slot 0 and every parameter even with no body resolved.
        var locals = self.resolution.slot_count(node)
        if locals < node.records("params").len() + 1 {
            locals = node.records("params").len() + 1
        }
        self.stack.push(FnCtx(proto, locals))

        self._default_thunks(proto, defaults, node)
        if init_fields != null { self._field_defaults(init_fields) }

        let body = node.child("body")
        if body == null {
            self._op(Op.ReturnNull, node.span)
        } elif node.flag("isExprBody") or body.kind != NodeKind.Block {
            self._expr(body)
            self._run_defers(node.span)
            self._op(Op.Return, node.span)
        } else {
            self._stmt(body)
            self._run_defers(node.span)
            self._op(Op.ReturnNull, node.span)
        }

        self._finish()
        return self._chunk().add_constant(Const(ConstTag.Func, proto))
    }

    # Thunks (spec §5 `defaultOffsets`) can't fall through into each other: a
    # keyword call may default any hole (`f(a, c: 3)` defaults only `b`), so the
    # VM runs each thunk on its own.
    fn _default_thunks(self, proto: FuncProto, defaults: list[Node], node: Node) -> void {
        if defaults.len() == 0 { return }
        let to_body = self._emit_jump(Op.Jump, node.span)
        for default_value in defaults {
            proto.default_offsets.push(self._chunk().here())
            let depth = self._depth()
            self._expr(default_value)
            self._op(Op.Return, default_value.span)
            # A thunk is entered directly, so its stack effect is not the body's.
            self._set_depth(depth)
        }
        self._patch(to_body)
    }

    # `self.field = <default>` for every instance field that declares one, run
    # before the body of `init` so an explicit assignment can override it.
    fn _field_defaults(self, fields: list[any]) -> void {
        for field in fields {
            if field["defaultValue"] == null or field["isStatic"] { continue }
            let field_span: Span = field["span"]
            self._op_u16(Op.GetLocal, 0, field_span)
            self._expr(field["defaultValue"])
            self._op_cached(Op.SetField, self._string(field["name"]), field_span)
        }
    }

    # The spec constant OP_CLASS reads is the 6-tuple the VM's
    # `classSpecInstantiate` decodes:
    #     (INT kind, STR name, STR|NULL superName, BOOL isAbstract,
    #      TUPLE required, TUPLE variants)
    # `required` holds a trait's (name, arity) pairs flattened and `variants` an
    # enum's (name, TUPLE fieldNames) pairs.
    fn _class_spec(
        self,
        kind: int,
        name: str,
        super_name: str?,
        is_abstract: bool,
        required: list[Const],
        variants: list[Const]
    ) -> int {
        var items: list[Const] = [Const(ConstTag.Int, kind), Const(ConstTag.Str, name), if super_name == null {
            Const(ConstTag.Null, null)
        } else {
            Const(ConstTag.Str, super_name)
        }, Const(ConstTag.Bool, is_abstract), Const(ConstTag.Tuple, required), Const(ConstTag.Tuple, variants)]
        return self._chunk().add_constant(Const(ConstTag.Tuple, items))
    }

    fn _class_decl(self, node: Node) -> void {
        let span = node.span
        let name = node.text("name")
        let superclass = node.type_field("superclass")
        let super_name = superclass?.name

        if super_name != null {
            self._op_cached(Op.GetGlobal, self._string(super_name), span)
        }
        self._op_k(
            Op.Class,
            self._class_spec(SPEC_CLASS, name, super_name, node.flag("isAbstract"), [], []),
            span
        )
        if super_name != null {
            self._op(Op.Inherit, span)  # peek(1) = super, peek(0) = subclass
            self._op(Op.Swap, span)
            self._op(Op.Pop, span)
        }

        for trait_type in node.type_list("traits") {
            self._op_k(Op.ImplTrait, self._string(trait_type.name ?? ""), span)
        }

        # A class's own type parameters are ordinary `Name` types here, so the
        # kind mapping has to be told which names are not concrete.
        var type_params: list[str] = []
        for generic in node.records("generics") {
            let generic_name = generic["name"]
            if generic_name != null { type_params.push(generic_name) }
        }

        let fields = node.records("fields")
        for field in fields {
            # bits 0-1 visibility, bit 2 static, bit 3 immutable,
            # bits 4-7 the declared kind
            var bits = _visibility_code(field["visibility"]) & 0x3
            if field["isStatic"] { bits = bits | 4 }
            if field["isLet"] { bits = bits | 8 }
            bits = bits | (_field_kind_code(field["type"], type_params) & 0xF) << 4
            self._chunk().write_op(Op.FieldDef, field["span"])
            self._chunk().write_u24(self._string(field["name"]))
            self._chunk().write_byte(bits)
        }

        # A static initialiser runs once, here, with the class still on the
        # stack; an instance field's default belongs to the constructor.
        for field in fields {
            if not field["isStatic"] or field["defaultValue"] == null { continue }
            let field_span: Span = field["span"]
            self._op(Op.Dup, field_span)
            self._expr(field["defaultValue"])
            self._op_cached(Op.SetField, self._string(field["name"]), field_span)
        }

        var init: Node? = null
        for method in node.child_list("methods") {
            if method.opt_text("name") == "init" {
                init = method
                break
            }
        }
        var needs_init = init == null
        if needs_init {
            needs_init = false
            for field in fields {
                if field["defaultValue"] != null and not field["isStatic"] {
                    needs_init = true
                    break
                }
            }
        }

        for method in node.child_list("methods") {
            let own = if method is init { fields } else { null }
            self._method(method, FN_METHOD, own)
        }
        for getter in node.child_list("getters") {
            self._method(getter, FN_METHOD | FN_GETTER, null)
        }
        for setter in node.child_list("setters") {
            self._method(setter, FN_METHOD | FN_SETTER, null)
        }

        if needs_init {
            # No declared `init` but a field has a default: synthesise one whose
            # only body is the default prologue.
            let synthetic = Node(NodeKind.FnDecl, {"name": "init"}, span)
            let index = self._function_constant(synthetic, "init", FN_METHOD, fields)
            self._op_k(Op.Closure, index, span)
            self._chunk().write_op(Op.Method, span)
            self._chunk().write_u24(self._string("init"))
            self._chunk().write_byte(FN_METHOD)
            self._chunk().write_byte(_visibility_code(Visibility.Public))
        }

        self._define(node, name, span)
    }

    fn _method(self, method: Node, flags: int, init_fields: list[any]?) -> void {
        let span = method.span
        var combined = flags
        if method.flag("isStatic") { combined = combined | FN_STATIC }
        if method.flag("isGenerator") { combined = combined | FN_GENERATOR }
        if method.flag("isAsync") { combined = combined | FN_ASYNC }
        let name = method.text("name")
        let index = self._function_constant(method, name, combined, init_fields)
        self._emit_closure(method, index)
        self._chunk().write_op(Op.Method, span)
        self._chunk().write_u24(self._string(name))
        # FN_INIT is bit 8 and does not survive the u8 operand; the VM recovers
        # it from the method name.
        self._chunk().write_byte(combined & 0xFF)
        # Visibility has a byte of its own: the flags byte is full.
        self._chunk().write_byte(_visibility_code(method.visibility()))
    }

    fn _enum_decl(self, node: Node) -> void {
        let span = node.span
        let name = node.text("name")
        self.enums[name] = node

        var variants: list[Const] = []
        for variant in node.records("variants") {
            variants.push(Const(ConstTag.Str, variant["name"]))
            var field_names: list[Const] = []
            for param in variant["params"] {
                field_names.push(Const(ConstTag.Str, param["name"] ?? ""))
            }
            variants.push(Const(ConstTag.Tuple, field_names))
        }
        self._op_k(Op.Class, self._class_spec(SPEC_ENUM, name, null, false, [], variants), span)
        for method in node.child_list("methods") { self._method(method, FN_METHOD, null) }
        self._define(node, name, span)
    }

    fn _trait_decl(self, node: Node) -> void {
        let span = node.span
        let name = node.text("name")

        # A required method has no body, so it cannot be a METHOD: it travels in
        # the spec constant as a (name, arity) pair.
        var required: list[Const] = []
        for method in node.child_list("methods") {
            if method.child("body") != null { continue }
            required.push(Const(ConstTag.Str, method.opt_text("name") ?? ""))
            required.push(Const(ConstTag.Int, method.records("params").len()))
        }
        self._op_k(Op.Class, self._class_spec(SPEC_TRAIT, name, null, false, required, []), span)
        for super_trait in node.type_list("supers") {
            self._op_k(Op.ImplTrait, self._string(super_trait.name ?? ""), span)
        }
        for method in node.child_list("methods") {
            if method.child("body") == null { continue }  # defaults only
            self._method(method, FN_METHOD, null)
        }
        self._define(node, name, span)
    }

    fn _define(self, node: Node, name: str?, span: Span) -> void {
        let symbol = self.resolution.symbol_of(node)
        self._store_symbol(symbol, name ?? "_", span, true)
        if symbol != null and symbol.visibility == Visibility.Public {
            self._op_k(Op.Export, self._string(symbol.name), span)
        }
    }

    fn _stmt(self, node: Node?) -> void {
        if node == null { return }
        let span = node.span
        match node.kind {
            NodeKind.ExprStmt => {
                self._expr(node.child("expr"))
                self._op(Op.Pop, span)
            }
            NodeKind.VarDecl => {
                self._expr(node.child("init"))
                let pattern = node.child("pattern")
                # `list[int]` is otherwise a compile-time fact only: a container
                # reaching a function through `any` can be mutated with no
                # diagnostic, because with an `any` receiver the checker does
                # not know what it was declared to hold. Stamp it on the value
                # so the mutation sites can check.
                if pattern != null and pattern.kind == NodeKind.PatBind {
                    let packed = _container_kind_code(node.type_field("declaredType"))
                    if packed != 0 {
                        self._chunk().write_op(Op.ElemKind, span)
                        self._chunk().write_byte(packed)
                    }
                }
                self._bind_pattern(pattern, span)
                # A `pub let` inside a function body is a mistake, but the C
                # exports it anyway. A destructuring pattern exports nothing: no single name to carry.
                let symbol = self.resolution.symbol_of(node)
                if symbol != null and
                    symbol.visibility == Visibility.Public and
                    pattern != null and
                    pattern.kind == NodeKind.PatBind {
                    self._op_k(Op.Export, self._string(symbol.name), span)
                }
            }
            NodeKind.Assign => { self._assign(node) }
            # A block's bindings die here so the next sibling reuses their
            # slots; closures over them are detached on the way out (spec §5.2).
            NodeKind.Block => {
                let close_at = self.resolution.capture_base_of(node)
                let saved = self._enter_close_scope(close_at)
                for statement in node.child_list("stmts") { self._stmt(statement) }
                self._close_scope(close_at, span)
                self._ctx().close_base = saved
            }
            NodeKind.If => {
                let base = self._depth()
                self._expr(node.child("cond"))
                let otherwise = self._emit_jump(Op.JumpIfFalse, span)
                self._stmt(node.child("thenBranch"))
                if node.child("elseBranch") == null {
                    self._set_depth(base)
                    self._patch(otherwise)
                } else {
                    let done = self._emit_jump(Op.Jump, span)
                    self._set_depth(base)
                    self._patch(otherwise)
                    self._stmt(node.child("elseBranch"))
                    self._set_depth(base)
                    self._patch(done)
                }
            }
            NodeKind.While => {
                let base = self._depth()
                let start = self._chunk().here()
                self._expr(node.child("cond"))
                let exit = self._emit_jump(Op.JumpIfFalse, span)
                self._body_of_loop(node, start, base, base)
                self._loop_back(start, span)
                self._set_depth(base)
                self._patch(exit)
                self._close_loop()
            }
            NodeKind.Loop => {
                let base = self._depth()
                let start = self._chunk().here()
                self._body_of_loop(node, start, base, base)
                self._loop_back(start, span)
                self._close_loop()
            }
            NodeKind.For => { self._for(node) }
            NodeKind.Match => { self._match(node, false) }
            NodeKind.Break | NodeKind.Continue => { self._jump(node) }
            # Close comes after the value (which may itself close over the slot)
            # and before RETURN's frame-wide close, which runs after defers that
            # still borrow these slots.
            NodeKind.Return => {
                let value = node.child("value")
                if value == null {
                    self._close_scope(self._return_close_base(), span)
                    self._finally_chain_for_return()
                    self._run_defers(span)
                    self._op(Op.ReturnNull, span)
                } elif self._can_tail_call(value) {
                    # No close: `_can_tail_call` rules out a finally chain and
                    # defers, and TAIL_CALL closes the frame before reusing it.
                    self._call(value, true)
                    # TAIL_CALL reuses the frame for a closure callee; a native
                    # one leaves its result behind, so RETURN is still needed.
                    self._op(Op.Return, span)
                } else {
                    self._expr(value)
                    self._close_scope(self._return_close_base(), span)
                    self._finally_chain_for_return()
                    self._run_defers(span)
                    self._op(Op.Return, span)
                }
            }
            NodeKind.Throw => {
                self._expr(node.child("value"))
                self._op(Op.Throw, span)
            }
            NodeKind.Try => { self._try(node) }
            NodeKind.Defer => {
                # The deferred block runs on this frame, so it needs no capture
                # list: its GET_LOCALs address the parent's slots.
                let parent = self._ctx()
                # `defer` is the name jaiFunctionIsDeferThunk recognizes a thunk
                # by, keeping it out of the "defer requires a callable" check and slot coalescing.
                let proto = FuncProto("defer", 0, 0)
                proto.upvalue_count = parent.proto.upvalue_count
                self.stack.push(FnCtx(proto, parent.next_temp))
                self._ctx().max_slot = parent.max_slot
                # Never tail-call out of a deferred block: its frame is already
                # being torn down.
                self._ctx().protect_depth = 1
                self._stmt(node.child("body"))
                self._op(Op.ReturnNull, span)
                parent.proto.chunk.raise_max_depth(proto.chunk.max_depth)
                self._finish()
                self._op_k(Op.PushDefer, self._chunk().add_constant(Const(ConstTag.Func, proto)), span)
                parent.has_defer = true
            }
            NodeKind.Assert => {
                if self.strip_asserts { return }
                self._expr(node.child("cond"))
                let ok = self._emit_jump(Op.JumpIfTrue, span)
                self._op_k(Op.AssertFail, self._string(_assert_message(node, self.source)), span)
                self._patch(ok)
            }
            NodeKind.FnDecl => {
                self._closure(node, node.opt_text("name"), 0)
                self._define(node, node.opt_text("name"), span)
            }
            NodeKind.ClassDecl => { self._class_decl(node) }
            NodeKind.EnumDecl => { self._enum_decl(node) }
            NodeKind.TraitDecl => { self._trait_decl(node) }
            NodeKind.Import => {
                self._op_k(Op.Import, self._string(node.text("path")), span)
                self._define(node, _last_segment(node.opt_text("alias") ?? node.text("path")), span)
            }
            NodeKind.FromImport => { self._from_import(node) }
            NodeKind.Export => {
                let names: list[str?] = node.get("names")
                for name in names {
                    if name != null { self._op_k(Op.Export, self._string(name), span) }
                }
            }
            # Aliases erase at use sites (spec §2.3), but `from M import Alias`
            # resolves against M's globals/exports at run time, so a public
            # alias still needs a binding here or import fails with E0802.
            NodeKind.TypeDecl => {
                if node.visibility() == Visibility.Public {
                    let index = self._string(node.text("name"))
                    self._op(Op.Null, span)
                    self._op_k(Op.DefGlobal, index, span)
                    self._op_k(Op.Export, index, span)
                }
            }
            NodeKind.ModuleDecl => { }
            _ => {
                # Every remaining kind is an expression used for its effect.
                self._expr(node)
                self._op(Op.Pop, span)
            }
        }
    }

    fn _assign(self, node: Node) -> void {
        let span = node.span
        let target = node.require("target")
        let value = node.child("value")
        let compound = node.flag("isCompound")

        if target.kind == NodeKind.Ident {
            if compound {
                self._load_name(target, target.text("name"))
                self._expr(value)
                self._op(_binary_op(node.op()), span)
            } else {
                self._expr(value)
            }
            self._store_symbol(self.resolution.symbol_of(target), target.text("name"), span, false)
        } elif target.kind == NodeKind.Member {
            # Intern the field name before the object: otherwise `p.chunk.a = v`
            # pools `chunk` first here vs `a` first in the C — a `make bootstrap` divergence.
            let field = self._string(target.text("name"))
            self._expr(target.child("object"))
            if compound {
                self._op(Op.Dup, span)
                self._op_cached(Op.GetField, field, span)
                self._expr(value)
                self._op(_binary_op(node.op()), span)
            } else {
                self._expr(value)
            }
            self._op_cached(Op.SetField, field, span)
        } elif target.kind == NodeKind.Index {
            self._expr(target.child("object"))
            self._expr(target.child("index"))
            if compound {
                self._op(Op.Dup2, span)
                self._op(Op.GetIndex, span)
                self._expr(value)
                self._op(_binary_op(node.op()), span)
            } else {
                self._expr(value)
            }
            self._op(Op.SetIndex, span)
        } elif target.kind == NodeKind.Slice {
            self._expr(target.child("object"))
            let before = self._depth()
            let flags = self._slice_parts(target)
            self._expr(value)
            self._op_u8(Op.SetSlice, flags, span)
            self._set_depth(before - 1)  # object, bounds and value all go
        } else {
            self._expr(value)
            self._bind_pattern(target, span)
        }
    }

    # `while` and `loop`: the iteration-ending close belongs at the body's end.
    fn _body_of_loop(
        self,
        node: Node,
        continue_target: int,
        continue_depth: int,
        break_depth: int
    ) -> void {
        let base = self.resolution.capture_base_of(node)
        self._enter_loop(node.opt_text("label"), continue_target, base, continue_depth, break_depth)
        self._loop_body(node.child("body"))
        self._close_iteration(base, node.span)
    }

    # A loop frame, plus the watermark update an enclosing `try` needs.
    fn _enter_loop(
        self,
        label: str?,
        continue_target: int,
        base: int,
        continue_depth: int,
        break_depth: int
    ) -> void {
        let ctx = self._ctx()
        ctx.loops.push(LoopFrame(label, continue_target, base, continue_depth, break_depth))
        if base >= 0 and (ctx.deep_close_base < 0 or base < ctx.deep_close_base) {
            ctx.deep_close_base = base
        }
    }

    # The statements of a loop body, without the block's own exit close: every
    # path out closes at the loop's base, never higher than the body's.
    fn _loop_body(self, body: Node?) -> void {
        if body == null { return }
        if body.kind != NodeKind.Block {
            self._stmt(body)
            return
        }
        for statement in body.child_list("stmts") { self._stmt(statement) }
    }

    fn _close_loop(self) -> void {
        let frame = self._ctx().loops.pop()
        self._set_depth(frame.break_depth)
        self._patch_all(frame.breaks)
    }

    # Detaches open upvalues at/above `base` (spec §5.2, §6) so closures over
    # those slots keep their values before the slots are reused. -1 emits
    # nothing, the common case.
    fn _close_scope(self, base: int, span: Span) -> void {
        if base < 0 { return }
        self._op_u16(Op.CloseUpvalue, base, span)
    }

    # Per-iteration bindings (spec §5.2): closing at the lowest captured slot
    # makes the next iteration's OP_CLOSURE allocate fresh cells.
    fn _close_iteration(self, base: int, span: Span) -> void {
        self._close_scope(base, span)
    }

    # Returns the previous `close_base`; the caller restores it when the scope ends.
    fn _enter_close_scope(self, base: int) -> int {
        let ctx = self._ctx()
        let saved = ctx.close_base
        if base >= 0 {
            if ctx.close_base < 0 or base < ctx.close_base { ctx.close_base = base }
            if ctx.deep_close_base < 0 or base < ctx.deep_close_base {
                ctx.deep_close_base = base
            }
        }
        return saved
    }

    # `return` leaves every block and iteration at once: close at the lowest base.
    fn _return_close_base(self) -> int {
        var base = self._ctx().close_base
        let loops = self._close_base_through(0)
        if loops >= 0 and (base < 0 or loops < base) { base = loops }
        return base
    }

    # Inline copies of every `finally` entered inside the loop at `index`,
    # innermost first. Only the explicit break/continue/return edges need them.
    fn _finally_chain_for_loop(self, index: int) -> void {
        let ctx = self._ctx()
        var position = ctx.finallys.len() - 1
        while position >= 0 and ctx.finally_loop_depth[position] > index + 1 {
            self._stmt(ctx.finallys[position])
            position -= 1
        }
    }

    # Reuses the frame only when nothing else must run after: defers, an
    # enclosing `try`/`finally`, and generators rule it out. `super`, spread and
    # keyword calls are out too — SUPER_INVOKE, CALL_SPREAD and CALL_KW have no tail form.
    fn _can_tail_call(self, value: Node) -> bool {
        if value.kind != NodeKind.Call { return false }
        let ctx = self._ctx()
        if ctx.protect_depth > 0 or ctx.has_defer { return false }
        if ctx.proto.flags & FN_GENERATOR != 0 { return false }
        if ctx.finallys.len() > 0 { return false }

        let callee = value.child("callee")
        if callee == null { return false }
        if callee.kind == NodeKind.Super { return false }
        if callee.kind == NodeKind.Member and callee.require("object").kind == NodeKind.Super {
            return false
        }
        for argument in value.records("args") {
            if argument["isSpread"] { return false }
            if argument["name"] != null { return false }
        }
        return true
    }

    # OP_RETURN runs this frame's defers itself; RUN_DEFERS is what makes them
    # run *here*, before an enclosing `finally` copy (the order of spec §3.8).
    fn _run_defers(self, span: Span) -> void {
        if self._ctx().has_defer { self._op(Op.RunDefers, span) }
    }

    fn _finally_chain_for_return(self) -> void {
        let ctx = self._ctx()
        var position = ctx.finallys.len() - 1
        while position >= 0 {
            self._stmt(ctx.finallys[position])
            position -= 1
        }
    }

    # One close at the lowest base covers every loop level left at once.
    fn _close_base_through(self, target: int) -> int {
        let loops = self._ctx().loops
        var base = -1
        var index = loops.len() - 1
        while index >= target {
            let candidate = loops[index].capture_base
            if candidate >= 0 and (base < 0 or candidate < base) { base = candidate }
            index -= 1
        }
        return base
    }

    #: `X.items()` as a loop iterable: take the lazy path when X turns out to
    #: be a dict, and otherwise fall through to the ordinary call so a user
    #: class's own `items()` still runs. The branch is the opcode's, because
    #: this file cannot know X's type — see `_is_items_call`.
    fn _items_iterable(self, iterable: Node?, span: Span) -> void {
        self._expr(_items_receiver(iterable))
        let skip = self._emit_jump(Op.GetIterItems, span)
        self._invoke_name("items", 0, span)
        self._op(Op.GetIter, span)
        self._patch(skip)
    }

    #: The frame slot `for <pattern> in <range>` binds each value into, or -1
    #: when the pattern is not one `OP_FOR_RANGE_BIND` can carry.
    #:
    #: It carries a slot number and nothing else, so a global, an upvalue and a
    #: destructuring pattern all fall back to the general loop — a range never
    #: yields anything unpackable anyway, so the tuple case is an error program
    #: either way and belongs on the path that reports it.
    #:
    #: A wildcard has no symbol and needs a destination nothing will read, so it
    #: takes a temporary — but only when `allocate` says this call is the one
    #: about to emit the loop, not the one deciding whether to.
    fn _range_bind_slot(self, pattern: Node?, allocate: bool) -> int {
        if pattern == null { return -1 }
        if pattern.kind == NodeKind.PatWildcard {
            return if allocate { self._alloc_temp() } else { 0 }
        }
        if pattern.kind != NodeKind.PatBind { return -1 }
        let symbol = self.resolution.symbol_of(pattern)
        if symbol == null or not symbol.is_slotted() { return -1 }
        return symbol.slot
    }

    #: `for x in a..b` with no ObjRange and no ObjIter: two int temporaries hold
    #: the counter and the end, and nothing is pushed at all.
    #:
    #: The saving is per loop ENTRY, which is why it is worth an opcode pair: an
    #: allocation census found the two objects a range loop's header builds were
    #: 99.95% of everything `tests/bench/nbody` allocated and 98.5% of
    #: `matrix_mul`'s. Measured at ~15.7ns per entry on an M2 Max, two thirds of
    #: it the allocation itself and a third the collections it forces.
    #:
    #: Emitted here rather than fused from bytecode — unlike `FOR_ITER_PAIR`,
    #: which is a peephole rule — because the loop stops occupying an operand
    #: stack slot, and `break` unwinds to a depth only this function knows.
    #:
    #: Answers false, having emitted nothing, for a loop that is not this shape:
    #: the range must be written out at the loop (one reaching it through a
    #: variable already built its object, so there is nothing left to save) and
    #: it must bind to a single frame slot.
    fn _for_range(self, node: Node, iterable: Node?) -> bool {
        if iterable == null or iterable.kind != NodeKind.Range { return false }
        let pattern = node.child("pattern")
        if self._range_bind_slot(pattern, false) < 0 { return false }
        let span = node.span
        let base = self._depth()
        # Two temporaries, and a third only for `for _ in a..b`, which has no
        # symbol to bind into. Every range loop is one nesting level of frame,
        # and the frame is what the compiled tier spends its registers on.
        let wildcard = pattern != null and pattern.kind == NodeKind.PatWildcard
        let counter = self._alloc_temp()
        let end = self._alloc_temp()
        let slot = self._range_bind_slot(pattern, wildcard)
        self._expr(iterable.child("start"))
        self._expr(iterable.child("stop"))
        self._chunk().write_op(Op.IterRange, span)
        self._chunk().write_byte(if iterable.flag("inclusive") { 1 } else { 0 })
        self._chunk().write_u16(counter)
        self._chunk().write_u16(end)

        let start = self._chunk().here()
        self._chunk().write_op(Op.ForRangeBind, span)
        let exit = self._chunk().here()
        self._chunk().write_u16(0)
        self._chunk().write_u16(slot)
        self._chunk().write_u16(counter)
        self._chunk().write_u16(end)

        # Nothing of this loop's is on the operand stack, so `break` and
        # `continue` both unwind to `base` and neither has anything to drop.
        let capture_base = self.resolution.capture_base_of(node)
        self._enter_loop(node.opt_text("label"), start, capture_base, base, base)
        self._loop_body(node.child("body"))
        self._close_iteration(capture_base, span)
        self._loop_back(start, span)
        self._set_depth(base)
        # The displacement is measured from the end of the instruction, which is
        # six bytes past its own operand rather than the usual zero.
        self._patch_from(exit, exit + 8)
        self._close_loop()
        self._free_temps(if wildcard { 3 } else { 2 })
        return true
    }

    fn _for(self, node: Node) -> void {
        let span = node.span
        let base = self._depth()
        let iterable = node.child("iterable")
        if self._for_range(node, iterable) { return }
        if _is_items_call(iterable) {
            self._items_iterable(iterable, span)
        } else {
            self._expr(iterable)
            self._op(Op.GetIter, span)
        }
        let start = self._chunk().here()
        let exit = self._emit_jump(Op.ForIter, span)
        self._adjust(1)  # the element, on the fall-through path
        # `continue` keeps the iterator (depth base+1); `break` drops it (base).
        let capture_base = self.resolution.capture_base_of(node)
        self._enter_loop(node.opt_text("label"), start, capture_base, base + 1, base)
        self._bind_pattern(node.child("pattern"), span)
        self._loop_body(node.child("body"))
        self._close_iteration(capture_base, span)
        self._loop_back(start, span)
        self._set_depth(base)  # FOR_ITER pops the exhausted iterator
        self._patch(exit)
        self._close_loop()
    }

    fn _jump(self, node: Node) -> void {
        let loops = self._ctx().loops
        let label = node.opt_text("label")
        var index = loops.len() - 1
        if label != null {
            while index >= 0 and loops[index].label != label { index -= 1 }
        }
        if index < 0 {
            self._unsupported(node.span, f"`break`/`continue` to `{label}`")
            return
        }
        let frame = loops[index]
        self._finally_chain_for_loop(index)
        let saved = self._depth()
        # The inner loop's iterator and any enclosing temporaries go here.
        var want = frame.continue_depth
        if node.kind == NodeKind.Break { want = frame.break_depth }
        self._pop_n(saved - want, node.span)
        # A closure made in this iteration must not go on aliasing the slot.
        self._close_iteration(self._close_base_through(index), node.span)
        if node.kind == NodeKind.Break {
            frame.breaks.push(self._emit_jump(Op.Jump, node.span))
        } else {
            self._loop_back(frame.continue_target, node.span)
        }
        # Code after the jump is unreachable but still has to be emitted.
        self._set_depth(saved)
    }

    # `PUSH_HANDLER` carries the branch offset *and* a type constant, so its
    # jump site ends after both operands, which is the base `_patch_from` gets.
    fn _push_handler(self, type_const: int, span: Span) -> int {
        self._chunk().write_op(Op.PushHandler, span)
        let operand = self._chunk().here()
        self._chunk().write_u16(0)
        self._chunk().write_u24(type_const)
        return operand
    }

    # Matches on the type *name* as a plain STR: the tag-9 TYPE descriptor has
    # no value representation and the reader refuses it. A union carries a
    # tuple of names (matches `jaiCgTypeGuardConst`, src/codegen/codegen.c); a
    # str there was wrong and `make bootstrap` caught it.
    fn _type_guard_const(self, caught: TypeExpr?) -> int {
        if caught is not null and caught.kind == TypeKind.Union {
            var names: list[Const] = []
            for member in caught.args {
                names.push(Const(ConstTag.Str, member.name ?? "any"))
            }
            return self._chunk().add_constant(Const(ConstTag.Tuple, names))
        }
        return self._string(caught?.name ?? "any")
    }

    fn _try(self, node: Node) -> void {
        let span = node.span
        let ctx = self._ctx()
        let finally_block = node.child("finallyBlock")
        let catches = node.records("catches")
        let base = self._depth()

        # A table handler assumes depth 0 (temporaries discarded). A live `for`
        # iterator or a `finally` forces PUSH_HANDLER instead — a `finally`
        # because the unwinder tries dynamic handlers before the table, so a
        # table clause there would never fire.
        let dynamic = catches.len() > 0 and (base > 0 or finally_block != null)

        var finally_entry = -1
        if finally_block != null {
            finally_entry = self._emit_jump(Op.PushFinally, span)
            # A `return`, `break` or `continue` leaving the region jumps past
            # END_FINALLY, so each of those edges inlines its own copy.
            ctx.finallys.push(finally_block)
            ctx.finally_loop_depth.push(ctx.loops.len())
        }
        var dispatch_entry = -1
        if dynamic { dispatch_entry = self._push_handler(self._string("any"), span) }

        # An exception skips every block close in the guarded region: the
        # unwinder jumps straight to the handler, reusing the body's binding slots.
        let saved_deep = ctx.deep_close_base
        ctx.deep_close_base = -1

        let guarded_start = self._chunk().here()
        ctx.protect_depth += 1
        self._stmt(node.child("body"))
        ctx.protect_depth -= 1
        if dynamic { self._op(Op.PopHandler, span) }
        let guarded_end = self._chunk().here()
        let body_deep = ctx.deep_close_base

        var done: list[int] = [self._emit_jump(Op.Jump, span)]

        if dynamic {
            self._patch_from(dispatch_entry, dispatch_entry + 5)
            self._close_scope(body_deep, span)
            self._catch_dispatch(catches, base, done, span)
        } else {
            for handler in catches {
                let handler_span: Span = handler["span"]
                let entry = self._chunk().here()
                self._set_depth(base)
                self._close_scope(body_deep, handler_span)

                let types: list[TypeExpr] = handler["types"]
                if types.len() == 0 {
                    self._chunk().exceptions.push(
                        ExceptionEntry(guarded_start, guarded_end, entry, CATCH_ALL)
                    )
                } else {
                    for caught in types {
                        self._chunk().exceptions.push(
                            ExceptionEntry(
                                guarded_start,
                                guarded_end,
                                entry,
                                self._type_guard_const(caught)
                            )
                        )
                    }
                }
                self._catch_body(handler, handler_span)
                done.push(self._emit_jump(Op.Jump, handler_span))
            }
        }

        let catches_end = self._chunk().here()

        if finally_block != null {
            ctx.finallys.pop()
            ctx.finally_loop_depth.pop()

            self._set_depth(base)
            self._patch_all(done)
            self._patch(finally_entry)

            # One catch-all entry spans body and handlers, registered after the
            # typed entries so the VM's first-match scan still prefers those.
            self._chunk().exceptions.push(
                ExceptionEntry(guarded_start, catches_end, self._chunk().here(), CATCH_ALL)
            )
            # The entry covers the handlers too, so the close is the whole
            # region's. On the normal edge it is a no-op.
            self._close_scope(ctx.deep_close_base, span)

            # Both the normal edge above and the unwinder land here. END_FINALLY
            # resumes a pending unwind and otherwise falls through.
            ctx.protect_depth += 1
            self._stmt(finally_block)
            ctx.protect_depth -= 1
            self._op(Op.EndFinally, span)
        } else {
            self._set_depth(base)
            self._patch_all(done)
        }

        # An outer `try` has to close what this one opened as well.
        if saved_deep >= 0 and (ctx.deep_close_base < 0 or saved_deep < ctx.deep_close_base) {
            ctx.deep_close_base = saved_deep
        }
        self._set_depth(base)
    }

    # Entered by the unwinder with the stack already restored to `base`.
    fn _catch_dispatch(self, catches: list[any], base: int, done: list[int], span: Span) -> void {
        var entries: list[list[int]] = []
        self._set_depth(base)
        for handler in catches {
            let handler_span: Span = handler["span"]
            var hits: list[int] = []
            let types: list[TypeExpr] = handler["types"]
            if types.len() == 0 {
                hits.push(self._emit_jump(Op.Jump, handler_span))
            } else {
                for caught in types {
                    self._op_k(Op.MatchExc, self._type_guard_const(caught), handler_span)
                    hits.push(self._emit_jump(Op.JumpIfTrue, handler_span))
                }
            }
            entries.push(hits)
        }
        self._op(Op.Reraise, span)

        for (index, handler) in catches.enumerate() {
            let handler_span: Span = handler["span"]
            self._set_depth(base)
            self._patch_all(entries[index])
            self._catch_body(handler, handler_span)
            done.push(self._emit_jump(Op.Jump, handler_span))
        }
    }

    fn _catch_body(self, handler: any, handler_span: Span) -> void {
        self._op(Op.GetExc, handler_span)
        let symbol = self.resolution.symbol_of(handler)
        if symbol != null or handler["name"] != null {
            self._store_symbol(symbol, handler["name"] ?? "_", handler_span, symbol == null)
        } else {
            self._op(Op.Pop, handler_span)
        }
        let ctx = self._ctx()
        ctx.protect_depth += 1
        self._stmt(handler["body"])
        ctx.protect_depth -= 1
    }

    fn _from_import(self, node: Node) -> void {
        let span = node.span
        self._op_k(Op.Import, self._string(node.text("path")), span)
        if node.flag("isWildcard") {
            self._op(Op.Pop, span)
            self._unsupported(span, "`import *`: the loader needs an explicit name list")
            return
        }
        for item in node.records("items") {
            let item_span: Span = item["span"]
            self._op_k(Op.ImportFrom, self._string(item["name"]), item_span)
            let bound: str = item["alias"] ?? item["name"]
            self._op_k(Op.DefGlobal, self._string(bound), item_span)
        }
        self._op(Op.Pop, span)
    }

    # The subject is bound to a temp that every test re-reads, which is what
    # makes the miss invariant hold: a failing test always leaves exactly one value pushed.
    fn _match(self, node: Node, as_expression: bool) -> void {
        let span = node.span
        let base = self._depth()

        self._expr(node.child("subject"))
        let subject = self._alloc_temp()
        self._bind_local(subject, span)

        var done: list[int] = []
        for arm in node.records("arms") {
            let arm_span: Span = arm["span"]
            var misses: list[int] = []
            var guard_miss: list[int] = []

            self._set_depth(base)
            self._pattern_in_slot(arm["pattern"], subject, misses)

            if arm["guard"] != null {
                let guard: Node = arm["guard"]
                self._expr(guard)
                guard_miss.push(self._emit_jump(Op.JumpIfFalse, guard.span))
            }

            let body: Node? = arm["body"]
            if as_expression {
                if body != null and body.kind == NodeKind.Block {
                    self._block_expr(body)
                } else {
                    self._expr(body)
                }
            } elif body != null {
                if body.kind == NodeKind.Block or is_statement(body.kind) {
                    self._stmt(body)
                } else {
                    self._expr(body)
                    self._op(Op.Pop, arm_span)
                }
            }
            done.push(self._emit_jump(Op.Jump, arm_span))

            # A pattern miss arrives with the failed candidate still pushed.
            self._set_depth(base + 1)
            self._patch_all(misses)
            self._op(Op.Pop, arm_span)
            self._set_depth(base)
            self._patch_all(guard_miss)
        }

        self._set_depth(base)
        if as_expression {
            # The checker proves exhaustiveness; this covers `any` subjects and
            # keeps the join depth honest.
            self._op_k(Op.AssertFail, self._string("no match arm matched"), span)
            self._op(Op.Null, span)
        }

        self._set_depth(base + (if as_expression { 1 } else { 0 }))
        self._patch_all(done)
        self._free_temps(1)
    }

    # Test the value in `slot`, leaving nothing behind on success. Every failure
    # path jumps to an offset appended to `misses` with one value pushed.
    fn _pattern_in_slot(self, pattern: Node?, slot: int, misses: list[int]) -> void {
        if pattern == null { return }
        let span = pattern.span
        let kind = pattern.kind

        if kind == NodeKind.PatWildcard {
            return
        } elif kind == NodeKind.PatBind {
            let declared = pattern.type_field("type")
            if declared != null {
                self._get_local(slot, span)
                self._chunk().write_op(Op.MatchType, span)
                self._chunk().write_u24(self._string(declared.name ?? "any"))
                misses.push(self._reserve_jump())
                self._op(Op.Pop, span)
            }
            let symbol = self.resolution.symbol_of(pattern)
            if symbol == null and pattern.opt_text("name") == null { return }
            self._get_local(slot, span)
            self._store_symbol(symbol, pattern.opt_text("name") ?? "_", span, symbol == null)
        } elif kind == NodeKind.PatLiteral {
            self._get_local(slot, span)
            self._chunk().write_op(Op.MatchConst, span)
            self._chunk().write_u24(self._literal_constant(pattern.child("value")))
            misses.push(self._reserve_jump())
            self._op(Op.Pop, span)
        } elif kind == NodeKind.PatRange {
            self._get_local(slot, span)
            self._chunk().write_op(Op.MatchRange, span)
            self._chunk().write_u24(self._literal_constant(pattern.child("lo")))
            self._chunk().write_u24(self._literal_constant(pattern.child("hi")))
            self._chunk().write_byte(if pattern.flag("inclusive") { 1 } else { 0 })
            misses.push(self._reserve_jump())
            self._op(Op.Pop, span)
        } elif kind == NodeKind.PatOr {
            self._test_or(pattern, slot, misses)
        } elif kind == NodeKind.PatTuple or kind == NodeKind.PatList {
            self._test_sequence(pattern, slot, misses)
        } elif kind == NodeKind.PatEnum {
            self._test_enum(pattern, slot, misses)
        } elif kind == NodeKind.PatClass {
            self._test_class(pattern, slot, misses)
        } else {
            self._unsupported(span, f"the pattern {kind}")
        }
    }

    # A sub-pattern on the value on top: parked so every test addresses a slot.
    fn _sub_pattern(self, sub: Node?, misses: list[int], span: Span) -> void {
        if sub == null or sub.kind == NodeKind.PatWildcard {
            self._op(Op.Pop, span)
            return
        }
        if sub.kind == NodeKind.PatBind and sub.type_field("type") == null {
            self._bind_pattern(sub, span)
            return
        }
        let temp = self._alloc_temp()
        self._bind_local(temp, span)
        self._pattern_in_slot(sub, temp, misses)
        self._free_temps(1)
    }

    fn _reserve_jump(self) -> int {
        let operand = self._chunk().here()
        self._chunk().write_u16(0)
        return operand
    }

    fn _literal_constant(self, node: Node?) -> int {
        if node == null { return self._chunk().add_constant(Const(ConstTag.Null, null)) }
        match node.kind {
            NodeKind.IntLit => {
                return self._chunk().add_constant(Const(ConstTag.Int, node.number("intLit")))
            }
            NodeKind.FloatLit => {
                return self._chunk().add_constant(Const(ConstTag.Float, node.get("floatLit")))
            }
            NodeKind.StrLit => { return self._string(node.text("chars")) }
            NodeKind.BoolLit => {
                return self._chunk().add_constant(Const(ConstTag.Bool, node.flag("boolLit")))
            }
            NodeKind.NullLit => { return self._chunk().add_constant(Const(ConstTag.Null, null)) }
            _ => {
                self._unsupported(node.span, "a non-literal pattern constant")
                return self._chunk().add_constant(Const(ConstTag.Null, null))
            }
        }
    }

    # Only the last alternative's failures escape to the enclosing pattern.
    fn _test_or(self, pattern: Node, slot: int, misses: list[int]) -> void {
        let alternatives = pattern.child_list("elems")
        if alternatives.len() == 0 { return }
        var matched: list[int] = []
        let base = self._depth()

        for (index, alternative) in alternatives.enumerate() {
            if index == alternatives.len() - 1 {
                self._pattern_in_slot(alternative, slot, misses)
                break
            }
            var local: list[int] = []
            self._pattern_in_slot(alternative, slot, local)
            matched.push(self._emit_jump(Op.Jump, pattern.span))
            self._set_depth(base + 1)  # the failing candidate is still pushed
            self._patch_all(local)
            self._op(Op.Pop, pattern.span)
        }

        self._set_depth(base)
        self._patch_all(matched)
    }

    fn _test_sequence(self, pattern: Node, slot: int, misses: list[int]) -> void {
        let span = pattern.span
        let elements = pattern.child_list("elems")
        let rest = pattern.number("restIndex")
        let count = elements.len()
        let has_rest = rest >= 0
        let fixed = if has_rest { count - 1 } else { count }
        let tail = if has_rest { count - rest - 1 } else { 0 }

        # With a rest the count is a minimum; without one it is the exact length.
        self._get_local(slot, span)
        self._chunk().write_op(Op.MatchSeq, span)
        self._chunk().write_byte(if fixed > 255 { 255 } else { fixed })
        self._chunk().write_byte(if has_rest { 1 } else { 0 })
        misses.push(self._reserve_jump())
        self._op(Op.Pop, span)

        for (index, element) in elements.enumerate() {
            if has_rest and index == rest {
                # value[rest : -tail], or value[rest:] when the rest is last.
                self._get_local(slot, span)
                self._emit_int(rest, span)
                var flags = 1
                var pushed = 1
                if tail > 0 {
                    self._emit_int(-tail, span)
                    flags = flags | 2
                    pushed += 1
                }
                self._op_u8(Op.GetSlice, flags, span)
                self._adjust(-pushed)
            } else {
                let offset = if has_rest and index > rest { index - count } else { index }
                self._get_local(slot, span)
                self._emit_int(offset, span)
                self._op(Op.GetIndex, span)
            }
            self._sub_pattern(element, misses, span)
        }
    }

    fn _test_enum(self, pattern: Node, slot: int, misses: list[int]) -> void {
        let span = pattern.span
        let owner = pattern.opt_text("typeName")
        let variant = pattern.opt_text("variantName")
        let tag = self._variant_tag(pattern)

        self._get_local(slot, span)
        self._chunk().write_op(Op.MatchType, span)
        if tag < 0 {
            # Declared in another module, so its tag numbering isn't visible
            # here; MATCH_TYPE resolves the dotted name at run time instead (BYTECODE.md §3.9).
            self._chunk().write_u24(self._string(f"{owner ?? "?"}.{variant ?? "?"}"))
        } else {
            self._chunk().write_u24(self._string(owner ?? "any"))
        }
        misses.push(self._reserve_jump())
        self._op(Op.Pop, span)

        if tag >= 0 {
            # ENUM_TAG peeks, so drop the enum value and test the tag alone:
            # that keeps the one-extra-value miss invariant.
            self._get_local(slot, span)
            self._op(Op.EnumTag, span)
            self._op(Op.Swap, span)
            self._op(Op.Pop, span)
            self._chunk().write_op(Op.MatchConst, span)
            self._chunk().write_u24(self._chunk().add_constant(Const(ConstTag.Int, tag)))
            misses.push(self._reserve_jump())
            self._op(Op.Pop, span)
        }

        for (index, element) in pattern.child_list("subPatterns").enumerate() {
            self._get_local(slot, span)
            self._op_u8(Op.EnumField, index, span)
            self._op(Op.Swap, span)
            self._op(Op.Pop, span)
            self._sub_pattern(element, misses, span)
        }
    }

    fn _test_class(self, pattern: Node, slot: int, misses: list[int]) -> void {
        let span = pattern.span
        let sub_patterns = pattern.child_list("subPatterns")

        self._get_local(slot, span)
        self._chunk().write_op(Op.MatchType, span)
        self._chunk().write_u24(self._string(pattern.opt_text("typeName") ?? "any"))
        misses.push(self._reserve_jump())
        self._op(Op.Pop, span)
        if sub_patterns.len() == 0 { return }

        # MATCH_FIELDS destructures in one step: its constant is a tuple of field
        # names, it peeks the object, and on success pushes one value per name.
        var names: list[Const] = []
        for index in range(sub_patterns.len()) {
            names.push(Const(ConstTag.Str, _field_name(pattern, index)))
        }
        self._get_local(slot, span)
        self._chunk().write_op(Op.MatchFields, span)
        self._chunk().write_u24(self._chunk().add_constant(Const(ConstTag.Tuple, names)))
        misses.push(self._reserve_jump())
        self._adjust(sub_patterns.len())  # the success path pushes one per field

        # Fields come off last-first; park them so sub-patterns compile in order.
        var temps: list[int] = []
        for _index in range(sub_patterns.len()) { temps.push(0) }
        var index = sub_patterns.len() - 1
        while index >= 0 {
            temps[index] = self._alloc_temp()
            self._bind_local(temps[index], span)
            index -= 1
        }
        self._op(Op.Pop, span)  # the object MATCH_FIELDS left under

        for (position, element) in sub_patterns.enumerate() {
            self._pattern_in_slot(element, temps[position], misses)
        }
        self._free_temps(sub_patterns.len())
    }

    # Declaration order of `Enum.Variant` (the tag ENUM_TAG pushes), or -1 when
    # declared elsewhere — not an error: `_test_enum` falls back to dotted MATCH_TYPE.
    fn _variant_tag(self, pattern: Node) -> int {
        let name = pattern.opt_text("variantName")
        let owner = pattern.opt_text("typeName")
        if name == null or owner == null { return -1 }
        let decl = self.enums.get(owner, null)
        if decl == null { return -1 }
        for (tag, variant) in decl.records("variants").enumerate() {
            if variant["name"] == name { return tag }
        }
        return -1
    }
}

# Not a binding of this frame: a global, and every class, trait and enum.
fn _symbol_is_global_like(symbol: Symbol) -> bool {
    let kind = symbol.kind
    return not (kind == SymbolKind.Local or kind == SymbolKind.Param or kind == SymbolKind.Upvalue or kind == SymbolKind.Field)
}

fn _visibility_code(visibility: Visibility) -> int {
    if visibility == Visibility.Public { return 2 }
    if visibility == Visibility.Protected { return 1 }
    return 0
}

#: The declared kind of a field, for bits 4-7 of `OP_FIELD_DEF`'s info byte.
#: Mirrors `FieldKind` in `src/vm/object/object.h`; 0 is `any`.
#:
#: A HINT, not a guarantee, and the VM guards on it rather than trusting it. So
#: the safe direction is downward: anything this cannot name exactly comes out
#: `any`, which claims nothing. A wrong claim is a false `TypeError` on correct
#: code, which is far worse than a missed one.
#:
#: `T?` is `any` because a nullable holds `null`, and a bare type parameter is
#: `any` because its type is not known here. `list[T]` is still `list`: the
#: container is known even when its element is not.
fn _field_kind_code(declared: TypeExpr?, type_params: list[str]) -> int {
    if declared == null { return 0 }
    let t: TypeExpr = declared

    if t.kind == TypeKind.Name {
        let n = t.name ?? ""
        if type_params.contains(n) { return 0 }
        if n == "int" { return 1 }
        if n == "float" { return 2 }
        if n == "bool" { return 3 }
        if n == "str" { return 4 }
        if n == "list" { return 5 }
        if n == "dict" { return 6 }
        if n == "any" { return 0 }
        # A class or trait name. Not `instance` yet: telling a class name from
        # an alias or an enum needs the checker, which does not run here, and
        # an enum value is not an instance.
        return 0
    }

    if t.kind == TypeKind.Generic {
        let n = t.name ?? ""
        if n == "list" { return 5 }
        if n == "dict" { return 6 }
        return 0
    }

    # Optional, Union, Fn, Tuple, Infer.
    return 0
}

#: The packed `OP_ELEM_KIND` byte for a declared container type, or 0 when the
#: type is not a container or its element type cannot be named.
#:
#: Low nibble is the element (or dict value) kind, high nibble the dict key
#: kind, both `FieldKind` codes. Zero means "promise nothing", which is what an
#: untyped container carries and what makes the guard one compare on the common
#: path.
fn _container_kind_code(declared: TypeExpr?) -> int {
    if declared == null { return 0 }
    let t: TypeExpr = declared
    if t.kind != TypeKind.Generic { return 0 }

    let name = t.name ?? ""
    if name == "list" {
        if t.args.len() != 1 { return 0 }
        return _field_kind_code(t.args[0], []) & 0xF
    }
    if name == "dict" {
        if t.args.len() != 2 { return 0 }
        let key = _field_kind_code(t.args[0], []) & 0xF
        let value = _field_kind_code(t.args[1], []) & 0xF
        return key << 4 | value
    }
    return 0
}

#: Emit the iterable of a `for` loop, leaving an iterator on the stack.
#:
#: `for … in X.items()` gets `OP_GET_ITER_ITEMS` applied to X rather than a call
#: to `items()` followed by `OP_GET_ITER`. On a dict that iterates the table
#: lazily instead of building the whole list of pairs first — `jaiDictItems`
#: allocates an N-element list of N fresh 2-tuples per call and keeps all N
#: alive at once, so every collection during the loop marks the lot.
#:
#: Matched on SHAPE only, because this file has no checker types. The opcode
#: decides at run time what the receiver actually is, so a user class defining
#: its own `items()` still gets that method called and behaves exactly as
#: before.
fn _is_items_call(iterable: Node?) -> bool {
    if iterable == null { return false }
    let node: Node = iterable
    if node.kind != NodeKind.Call { return false }
    if node.records("args").len() != 0 { return false }
    let callee = node.child("callee")
    if callee == null { return false }
    let target: Node = callee
    if target.kind != NodeKind.Member { return false }
    if target.opt_text("name") != "items" { return false }
    # `super.items()` binds to slot 0 and has its own lowering.
    let receiver = target.child("object")
    if receiver == null { return false }
    let recv: Node = receiver
    return recv.kind != NodeKind.Super
}

#: The receiver of an `X.items()` call matched by `_is_items_call`.
fn _items_receiver(iterable: Node?) -> Node? {
    if iterable == null { return null }
    let node: Node = iterable
    return node.child("callee")?.child("object")
}

fn _then_of(node: Node) -> Node? {
    if node.kind == NodeKind.Ternary { return node.child("thenExpr") }
    return node.child("thenBranch")
}

fn _else_of(node: Node) -> Node? {
    if node.kind == NodeKind.Ternary { return node.child("elseExpr") }
    return node.child("elseBranch")
}

fn _field_name(pattern: Node, index: int) -> str {
    let names: list[str?] = pattern.get("fieldNames")
    if index >= names.len() { return f"{index}" }
    return names[index] ?? f"{index}"
}

fn _last_segment(path: str?) -> str? {
    if path == null { return null }
    let parts = path.split(".")
    return parts[parts.len() - 1]
}

#: 256-byte buffer minus 32 reserved for prefix and terminator (`assertMessage`,
#: src/codegen/codegen_stmt.c).
const _ASSERT_TEXT_MAX: int = 224

fn _assert_message(node: Node, source: str?) -> str {
    let message = node.child("message")
    if message != null and message.kind == NodeKind.StrLit { return message.text("chars") }

    let cond = node.child("cond")
    if source != null and cond != null and cond.span.is_known() {
        # Span is a BYTE offset (as in the C), but Jaithon slices strings by
        # scalar — wrong past a multi-byte char. Bytes have no slice either
        # (`raw[start:stop]` raises `TypeError`), so take bytes one at a time.
        let raw = __prim__.str_encode(source)
        let start = cond.span.start
        var stop = cond.span.end
        if stop > start and stop <= raw.len() {
            if stop - start > _ASSERT_TEXT_MAX { stop = start + _ASSERT_TEXT_MAX }
            var taken: list[int] = []
            var at = start
            while at < stop {
                taken.push(raw[at])
                at += 1
            }
            return f"assertion failed: {_decoded(taken)}"
        }
    }
    return "assertion failed"
}

# Drops the last scalar if the cut landed mid-character: `str_decode` refuses
# invalid UTF-8, so the longest valid prefix is the only faithful answer (the C
# truncates at a byte and prints mojibake).
fn _decoded(raw: list[int]) -> str {
    var length = raw.len()
    while length > 0 {
        try {
            return __prim__.str_decode(__prim__.bytes_new(raw[:length]))
        } catch _invalid: ValueError {
            length -= 1
        }
    }
    return ""
}

fn _unary_op(op: OpKind) -> Op {
    if op == OpKind.Neg { return Op.Neg }
    if op == OpKind.Pos { return Op.Pos }
    if op == OpKind.Not { return Op.Not }
    return Op.BNot
}

fn _binary_op(op: OpKind) -> Op {
    if op == OpKind.Add { return Op.Add }
    if op == OpKind.Sub { return Op.Sub }
    if op == OpKind.Mul { return Op.Mul }
    if op == OpKind.Div { return Op.Div }
    if op == OpKind.FloorDiv { return Op.FloorDiv }
    if op == OpKind.Mod { return Op.Mod }
    if op == OpKind.Pow { return Op.Pow }
    if op == OpKind.AddWrap { return Op.AddWrap }
    if op == OpKind.SubWrap { return Op.SubWrap }
    if op == OpKind.MulWrap { return Op.MulWrap }
    if op == OpKind.Band { return Op.BAnd }
    if op == OpKind.Bor { return Op.BOr }
    if op == OpKind.Bxor { return Op.BXor }
    if op == OpKind.Shl { return Op.Shl }
    if op == OpKind.Shr { return Op.Shr }
    if op == OpKind.Eq { return Op.Eq }
    if op == OpKind.Ne { return Op.Ne }
    if op == OpKind.Lt { return Op.Lt }
    if op == OpKind.Le { return Op.Le }
    if op == OpKind.Gt { return Op.Gt }
    if op == OpKind.Ge { return Op.Ge }
    if op == OpKind.Is { return Op.Is }
    if op == OpKind.IsNot { return Op.IsNot }
    if op == OpKind.In { return Op.In }
    if op == OpKind.NotIn { return Op.NotIn }
    return Op.Add
}

#: Lower a resolved program and return the module body's function record.
#:
#: The record comes back even when a diagnostic was reported, so a caller can
#: disassemble what was produced while reporting what was not.
pub fn emit(
    program: Node,
    resolution: Resolution,
    bag: DiagnosticBag,
    strip_asserts: bool = false,
    module_name: str? = null,
    opt_level: int = 2,
    optimize: fn(FuncProto, int) -> void = _no_optimize,
    source: str? = null,
    imported_enums: dict[str, Node] = {}
) -> FuncProto {
    let emitter = Emitter(resolution, bag, strip_asserts, opt_level, optimize)
    emitter.source = source
    emitter.imported_enums = imported_enums
    return emitter.compile_program(program, module_name)
}

# `jaithon.compile.opt` can't be imported here (circular import, E0801), so the
# driver is handed in and run where jaiCgFinishFunction calls jaiOptimize —
# once per function, not once over the module, since slot coalescing isn't idempotent.
fn _no_optimize(proto: FuncProto, level: int) -> void { }
