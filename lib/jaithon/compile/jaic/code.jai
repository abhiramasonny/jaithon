#: Walking the instruction stream: how long the instruction at an offset is,
#: which opcodes carry an inline-cache slot, and the fixed-width reads over a
#: `list[int]` of code that answering either of those needs.
#:
#: Its own file because both directions of the container walk the same bytes.
#: `read.jai` scans a freshly read chunk to recover `cache_count`, which is
#: never stored, and `disassemble.jai` walks it to print it. `OP_CLOSURE` is
#: the one variable-length instruction, and if the two of them measured it
#: differently one would desynchronise part-way through a function.
#:
#: What is NOT here: anything that renders. `_read_i16` is used only by the
#: disassembler and still lives here, because reading an integer out of a code
#: list is this file's job and splitting the three apart would say otherwise.
from jaithon.compile.emit import Const, ConstTag, op_name, op_operand_bytes

# Cache slots are never stored — sizes come from scanning the code, which
# doubles as a decodability check. The per-opcode cache-operand offset must
# track `jaiOpCacheOperand` in src/vm/bytecode/chunk.c, or a loaded chunk silently gets
# too few cache lines and every site degrades to the slow path.
pub fn _cache_slots(code: list[int], constants: list[Const]) -> int {
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

pub fn _read_u16(code: list[int], offset: int) -> int {
    if offset + 1 >= code.len() { return 0 }
    return code[offset] | code[offset + 1] << 8
}

pub fn _read_u24(code: list[int], offset: int) -> int {
    if offset + 2 >= code.len() { return 0 }
    return code[offset] | code[offset + 1] << 8 | code[offset + 2] << 16
}

pub fn _read_i16(code: list[int], offset: int) -> int {
    let value = _read_u16(code, offset)
    if value >= 32768 { return value - 65536 }
    return value
}
