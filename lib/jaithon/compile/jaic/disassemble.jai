#: Render a function record as text, in the layout `jaiDisassembleChunk`
#: prints: offset, source position, mnemonic, operands, and a note naming the
#: constant or the jump target.
#:
#: Debug output, and the only module of `jaic/` that is neither read nor
#: write. It is here rather than beside the emitter because what it renders is
#: a `FuncProto` shaped the way the container stores one, cache operands and
#: all, and because `mod.jai` reaches it through the same import it already
#: uses for `write_module`.
#:
#: What is NOT here: instruction lengths, which it shares with the reader and
#: which live in `code.jai`.
from jaithon.compile.emit import Chunk, ConstTag, FuncProto
from jaithon.compile.emit import op_name, op_operand_bytes
from jaithon.compile.jaic.code import _read_i16, _read_u16, _read_u24
from std.str import repeat

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

fn _pad_right(text: str, width: int) -> str {
    if text.len() >= width { return text }
    return text + repeat(" ", width - text.len())
}

fn _pad_left(text: str, width: int, fill: str) -> str {
    if text.len() >= width { return text }
    return repeat(fill, width - text.len()) + text
}
