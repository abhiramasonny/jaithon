#: The caffemodel half: the same messages, off the protobuf wire.
from .message import Message
from .schema import MAX_DEPTH, Spec, layout_of
from ..protobuf import WIRE_I32, WIRE_I64, WIRE_LEN, WIRE_VARINT, as_double, as_float, as_string, packed_doubles, packed_floats, packed_varints, parse_message

#: The same message out of the binary format.
pub fn decode_message(kind: str, data: bytes, depth: int) -> Message {
    if depth > MAX_DEPTH {
        throw ValueError(f"this caffemodel nests more than {MAX_DEPTH} deep")
    }
    let table = layout_of(kind)
    let message = Message(kind)
    for entry in parse_message(data) {
        let spec = table.by_number.get(entry.number, null)
        if spec is null { continue }
        let field: Spec = spec
        let where = f"{kind}.{field.name}"
        if field.kind == "msg" {
            if entry.wire != WIRE_LEN {
                throw ValueError(f"{where} is a message and arrived as wire type {entry.wire}")
            }
            message.add_child(field.name, decode_message(field.of, entry.data, depth + 1))
        } elif field.kind == "str" {
            message.add_str(field.name, as_string(entry))
        } elif field.kind == "float" {
            if entry.wire == WIRE_LEN {
                message.add_floats(field.name, packed_floats(entry))
            } elif entry.wire == WIRE_I32 {
                message.add_floats(field.name, [as_float(entry)])
            } else {
                throw ValueError(f"{where} is a float and arrived as wire type {entry.wire}")
            }
        } elif field.kind == "double" {
            if entry.wire == WIRE_LEN {
                message.add_floats(field.name, packed_doubles(entry))
            } elif entry.wire == WIRE_I64 {
                message.add_floats(field.name, [as_double(entry)])
            } else {
                throw ValueError(f"{where} is a double and arrived as wire type {entry.wire}")
            }
        } else {
            if entry.wire == WIRE_LEN {
                message.add_ints(field.name, packed_varints(entry))
            } elif entry.wire == WIRE_VARINT {
                message.add_ints(field.name, [entry.varint])
            } else {
                throw ValueError(f"{where} is an integer and arrived as wire type {entry.wire}")
            }
        }
    }
    return message
}
