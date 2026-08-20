#: The protobuf wire format, read without a schema.
#:
#: ONNX, Caffe and TensorFlow all ship their models as protobuf, and every
#: importer in this directory reads one through here. There is no generated
#: code and no `.proto` anywhere in the tree, because there needs to be none:
#: each field on the wire carries its own number and its own wire type, so a
#: reader recovers the whole structure and leaves it to the importer to say
#: which number means what. An unknown field then costs nothing -- it parses
#: like any other and the importer ignores it -- which is what keeps a model
#: written by a newer exporter readable at all.
#:
#: Three details of the format are easy to get wrong and are load-bearing here.
#: A negative `int32` or `int64` is stored sign-extended to ten bytes rather
#: than five, so the varint reader has to run to ten and let the tenth group's
#: high bits fall off the top exactly as a 64-bit accumulator does. `sint32`
#: and `sint64` use the zigzag encoding instead, and undoing zigzag needs a
#: logical right shift where `>>` is arithmetic. And a packed repeat is not
#: required to arrive in one piece: concatenating two encodings is protobuf's
#: own merge, so one field number can appear several times and the runs join --
#: which is why `fields_by_number` exists beside `field_by_number`.
#:
#: A model file is untrusted input. Every length is checked against what is
#: left in the buffer, every varint is capped at ten bytes, a wire type
#: protobuf does not define is an error rather than a guess, and a group that
#: never closes is an error rather than a walk off the end. Nothing here
#: recurses either: a nested message comes back as `bytes` for the caller to
#: parse when it wants it, and a group is skipped with an explicit stack, so
#: no arrangement of bytes can exhaust the call stack.
#:
#: ```
#: let fields = parse_message(raw)
#: let graph = field_by_number(fields, 7)
#: if graph is not null { let nodes = fields_by_number(parse_nested(graph), 1) }
#: ```
from std.core import Printable
from std.math import INF, NAN, ldexp

#: Wire type 0: a base-128 varint, holding every integer and boolean kind.
pub let WIRE_VARINT = 0
#: Wire type 1: eight little-endian bytes -- `fixed64`, `sfixed64`, `double`.
pub let WIRE_I64 = 1
#: Wire type 2: a varint length then that many bytes -- `string`, `bytes`, a
#: nested message, or a packed repeat.
pub let WIRE_LEN = 2
#: Wire type 3: the opening tag of a group, the encoding proto2 deprecated.
pub let WIRE_START_GROUP = 3
#: Wire type 4: the tag that closes the innermost open group.
pub let WIRE_END_GROUP = 4
#: Wire type 5: four little-endian bytes -- `fixed32`, `sfixed32`, `float`.
pub let WIRE_I32 = 5

#: The largest field number protobuf allows, `2**29 - 1`.
pub let MAX_FIELD_NUMBER = 536870911

#: A varint past ten bytes cannot be a 64-bit value, so it is a malformed one.
let MAX_VARINT_BYTES = 10

#: How deeply groups may nest before the buffer is called malformed. This is
#: protobuf's own default recursion limit; nothing legitimate approaches it,
#: and the cap keeps a hostile run of start-group tags from growing the open
#: stack without bound.
let MAX_GROUP_DEPTH = 100

#: `2**63 - 1`, the mask that turns the arithmetic `>>` into a logical one.
let LOW_63 = 0x7FFFFFFFFFFFFFFF

#: One field, as it sits on the wire.
#:
#: Which of the payload fields is meaningful is decided by `wire`, and the rest
#: are left at zero or empty: `varint` for wire type 0, `fixed64` for 1, `data`
#: for 2, `fixed32` for 5, and `data` again for 3, where it holds the group's
#: body -- everything between the opening and closing tags, which is itself a
#: message and parses with `parse_message`.
#:
#: `fixed32` is the four bytes read as an unsigned value, so it spans
#: `0..2**32`. `fixed64` and `varint` are the eight or ten bytes read into a
#: signed 64-bit integer, so a `uint64` or `fixed64` above `2**63 - 1` arrives
#: as the negative integer with the same bit pattern -- there is no wider
#: integer to put it in, and the pattern is what `as_double` needs anyway.
pub class Field: Printable {
    #: The field number the writer's schema gave this field.
    pub var number: int
    #: Which of the six wire types the payload is in.
    pub var wire: int
    #: Wire type 0's value, sign-extended as protobuf wrote it.
    pub var varint: int
    #: Wire type 2's bytes, or wire type 3's group body.
    pub var data: bytes
    #: Wire type 1's eight bytes, little-endian.
    pub var fixed64: int
    #: Wire type 5's four bytes, little-endian.
    pub var fixed32: int

    pub fn init(
        self,
        number: int,
        wire: int,
        varint: int = 0,
        data: bytes = bytes([]),
        fixed64: int = 0,
        fixed32: int = 0
    ) {
        self.number = number
        self.wire = wire
        self.varint = varint
        self.data = data
        self.fixed64 = fixed64
        self.fixed32 = fixed32
    }

    pub fn to_str(self) -> str {
        if self.wire == WIRE_VARINT { return f"Field({self.number}, varint {self.varint})" }
        if self.wire == WIRE_I64 { return f"Field({self.number}, fixed64 {self.fixed64})" }
        if self.wire == WIRE_I32 { return f"Field({self.number}, fixed32 {self.fixed32})" }
        if self.wire == WIRE_LEN { return f"Field({self.number}, {self.data.len()} bytes)" }
        if self.wire == WIRE_START_GROUP {
            return f"Field({self.number}, group of {self.data.len()} bytes)"
        }
        return f"Field({self.number}, wire type {self.wire})"
    }

    fn __str__(self) -> str { return self.to_str() }
}

#: Where an error happened, for a message a reader of the model file can act
#: on. Built only on the throwing path, so the parse loop pays nothing for it.
fn at_field(number: int) -> str {
    if number == 0 { return "reading a tag" }
    return f"reading field {number}"
}

#: What to call a wire type in an error.
fn wire_name(wire: int) -> str {
    if wire == WIRE_VARINT { return "varint" }
    if wire == WIRE_I64 { return "64-bit" }
    if wire == WIRE_LEN { return "length-delimited" }
    if wire == WIRE_START_GROUP { return "start-group" }
    if wire == WIRE_END_GROUP { return "end-group" }
    if wire == WIRE_I32 { return "32-bit" }
    return f"undefined wire type {wire}"
}

#: A position in a buffer, with every read bounds-checked against its end.
#:
#: Each read takes the field number it is reading for rather than a message,
#: because the parse loop runs once per field of a model that may hold millions
#: of them and an f-string built per field to be thrown away is real cost.
class Cursor {
    pub let data: bytes
    pub let end: int
    pub var at: int

    pub fn init(self, data: bytes) {
        self.data = data
        self.end = data.len()
        self.at = 0
    }

    pub fn done(self) -> bool { return self.at >= self.end }

    #: The next base-128 varint.
    #:
    #: The groups are little-endian, seven bits each, and the tenth is allowed
    #: to overflow: protobuf writes a negative `int32` sign-extended to sixty-
    #: four bits, so its tenth group carries bit 63 and nothing above it
    #: survives the shift. That truncation is the encoding, not a rounding of
    #: it -- reject the tenth group instead and every negative field in every
    #: model fails to read.
    pub fn varint(self, number: int) -> int {
        var value = 0
        for index in 0..MAX_VARINT_BYTES {
            if self.at >= self.end {
                throw ValueError(f"{at_field(number)}: a varint runs past the end of the message")
            }
            let byte = self.data[self.at]
            self.at += 1
            value = value | (byte & 0x7F) << 7 * index
            if byte < 0x80 { return value }
        }
        throw ValueError(
            f"{at_field(number)}: a varint is unterminated after {MAX_VARINT_BYTES} bytes"
        )
    }

    #: The next `width` bytes as a little-endian integer.
    pub fn fixed(self, width: int, number: int) -> int {
        if self.end - self.at < width {
            throw ValueError(
                f"{at_field(number)}: a {width}-byte field runs past the end of the message"
            )
        }
        let value = little_endian(self.data, self.at, width)
        self.at += width
        return value
    }

    #: The next `count` bytes, copied out.
    pub fn take(self, count: int, number: int) -> bytes {
        self.check(count, number)
        let out = bytes([self.data[self.at + index] for index in 0..count])
        self.at += count
        return out
    }

    #: Step over `count` bytes without copying them.
    pub fn skip(self, count: int, number: int) -> void {
        self.check(count, number)
        self.at += count
    }

    #: A length is checked as a distance from here rather than as an absolute
    #: offset, so a length near `2**63` cannot overflow its way back in range.
    #: A negative one is a length varint whose top bit was set, which no
    #: encoder writes.
    fn check(self, count: int, number: int) -> void {
        if count < 0 {
            throw ValueError(f"{at_field(number)}: a length of {count} is not a length")
        }
        let left = self.end - self.at
        if count > left {
            throw ValueError(
                f"{at_field(number)}: a length of {count} runs past the {left} bytes left"
            )
        }
    }
}

#: `width` bytes of `data` from `at`, little-endian.
#:
#: For eight bytes the top byte's shift lands on bit 63, so the result is the
#: signed reinterpretation of the pattern rather than a value above `2**63`.
fn little_endian(data: bytes, at: int, width: int) -> int {
    var value = 0
    for index in 0..width {
        value = value | data[at + index] << 8 * index
    }
    return value
}

#: Every field of a message, in the order it was written.
#:
#: Repeated fields keep their repeats and unknown numbers keep their place, so
#: what comes back is the message and not an interpretation of it. Nested
#: messages are left as bytes in `data`; `parse_nested` opens one when the
#: caller wants it.
pub fn parse_message(data: bytes) -> list[Field] {
    let cursor = Cursor(data)
    var fields: list[Field] = []
    while not cursor.done() {
        let tag = cursor.varint(0)
        let number = tag >> 3
        let wire = tag & 7
        if number < 1 or number > MAX_FIELD_NUMBER {
            throw ValueError(
                f"a tag names field {number}, outside the 1..{MAX_FIELD_NUMBER} protobuf allows"
            )
        }
        var field = Field(number, wire)
        if wire == WIRE_VARINT {
            field.varint = cursor.varint(number)
        } elif wire == WIRE_I64 {
            field.fixed64 = cursor.fixed(8, number)
        } elif wire == WIRE_LEN {
            let length = cursor.varint(number)
            field.data = cursor.take(length, number)
        } elif wire == WIRE_I32 {
            field.fixed32 = cursor.fixed(4, number)
        } elif wire == WIRE_START_GROUP {
            let start = cursor.at
            let stop = scan_group(cursor, number)
            field.data = bytes([data[index] for index in start..stop])
        } elif wire == WIRE_END_GROUP {
            throw ValueError(f"field {number}: an end-group tag with no group open")
        } else {
            throw ValueError(f"field {number}: {wire_name(wire)} is not one protobuf defines")
        }
        fields.push(field)
    }
    return fields
}

#: Walk to the tag that closes the group just opened on `number`, and return
#: the offset that tag starts at -- which is one past the group's last body
#: byte.
#:
#: The open numbers are kept on a stack rather than counted, because a group
#: is closed only by an end tag carrying its own number: count alone and a
#: truncated inner group silently ends the outer one instead.
fn scan_group(cursor: Cursor, number: int) -> int {
    var open: list[int] = [number]
    var body_end = cursor.at
    while open.len() > 0 {
        if cursor.done() {
            throw ValueError(f"group {number} is never closed before the message ends")
        }
        body_end = cursor.at
        let tag = cursor.varint(number)
        let inner = tag >> 3
        let wire = tag & 7
        if inner < 1 or inner > MAX_FIELD_NUMBER {
            throw ValueError(
                f"inside group {number}, a tag names field {inner}, which protobuf disallows"
            )
        }
        if wire == WIRE_END_GROUP {
            let expected = open.pop()
            if inner != expected {
                throw ValueError(f"group {expected} is closed by an end tag for field {inner}")
            }
            continue
        }
        if wire == WIRE_START_GROUP {
            if open.len() >= MAX_GROUP_DEPTH {
                throw ValueError(f"groups nest more than {MAX_GROUP_DEPTH} deep")
            }
            open.push(inner)
            continue
        }
        if wire == WIRE_VARINT {
            let _skipped = cursor.varint(inner)
        } elif wire == WIRE_I64 {
            cursor.skip(8, inner)
        } elif wire == WIRE_I32 {
            cursor.skip(4, inner)
        } elif wire == WIRE_LEN {
            cursor.skip(cursor.varint(inner), inner)
        } else {
            throw ValueError(f"field {inner}: {wire_name(wire)} is not one protobuf defines")
        }
    }
    return body_end
}

#: The last field with this number, or null when there is none.
#:
#: Last rather than first because that is protobuf's own rule: concatenating
#: two encodings is a merge, and for a singular field the later value wins.
#: Reading a repeated field this way would quietly drop all but one of it, so
#: repeats want `fields_by_number`.
pub fn field_by_number(fields: list[Field], number: int) -> Field? {
    var found: Field? = null
    for field in fields {
        if field.number == number { found = field }
    }
    return found
}

#: Every field with this number, in the order written.
#:
#: A repeated field lands here whether the writer packed it or not, and so does
#: a packed one split across several runs by a merge.
pub fn fields_by_number(fields: list[Field], number: int) -> list[Field] {
    return [field for field in fields if field.number == number]
}

#: The fields of a nested message.
pub fn parse_nested(field: Field) -> list[Field] {
    if field.wire != WIRE_LEN and field.wire != WIRE_START_GROUP {
        throw ValueError(
            f"field {field.number} is {wire_name(field.wire)}, so it holds no nested message"
        )
    }
    return parse_message(field.data)
}

fn require_wire(field: Field, wire: int, reading: str) -> void {
    if field.wire != wire {
        throw ValueError(
            f"field {field.number} is {wire_name(field.wire)}, so it does not hold {reading}"
        )
    }
}

#: A length-delimited field as text.
pub fn as_string(field: Field) -> str {
    require_wire(field, WIRE_LEN, "a string")
    try {
        return field.data.decode()
    } catch error: ValueError {
        throw ValueError(f"field {field.number} is not valid UTF-8: {error.message}")
    }
}

#: A length-delimited field as its bytes.
pub fn as_bytes(field: Field) -> bytes {
    require_wire(field, WIRE_LEN, "a bytes value")
    return field.data
}

#: A varint field as a boolean. Protobuf writes 1 and 0, but the reader takes
#: any non-zero as true, which is what every protobuf implementation does.
pub fn as_bool(field: Field) -> bool {
    require_wire(field, WIRE_VARINT, "a bool")
    return field.varint != 0
}

#: A `fixed32` field as the `float` it encodes.
pub fn as_float(field: Field) -> float {
    require_wire(field, WIRE_I32, "a float")
    return f32_from_bits(field.fixed32)
}

#: A `fixed64` field as the `double` it encodes.
pub fn as_double(field: Field) -> float {
    require_wire(field, WIRE_I64, "a double")
    return f64_from_bits(field.fixed64)
}

#: A varint field as the `sint32` or `sint64` it encodes.
pub fn as_signed(field: Field) -> int {
    require_wire(field, WIRE_VARINT, "a sint32 or sint64")
    return zigzag(field.varint)
}

#: Undo zigzag: the low bit is the sign and the rest is the magnitude, so a
#: small negative costs one byte instead of the ten a sign-extended `int64`
#: costs.
#:
#: The shift has to be logical. `>>` is arithmetic, so the mask puts the bit it
#: copied down from the top back to zero before the sign is applied -- without
#: it every negative input comes back one short of doubled.
fn zigzag(value: int) -> int {
    return value >> 1 & LOW_63 ^ -(value & 1)
}

#: Every varint in a packed repeat.
#:
#: Packed `bool` lands here too, as ones and zeroes, and so does packed `enum`.
pub fn packed_varints(field: Field) -> list[int] {
    require_wire(field, WIRE_LEN, "a packed varint repeat")
    let cursor = Cursor(field.data)
    var out: list[int] = []
    while not cursor.done() {
        out.push(cursor.varint(field.number))
    }
    return out
}

#: Every value of a packed `sint32` or `sint64` repeat.
pub fn packed_signed(field: Field) -> list[int] {
    return [zigzag(value) for value in packed_varints(field)]
}

fn packed_width(field: Field, width: int, reading: str) -> int {
    require_wire(field, WIRE_LEN, reading)
    let length = field.data.len()
    if length % width != 0 {
        throw ValueError(f"field {field.number}'s {length} bytes are not whole {width}-byte values")
    }
    return length // width
}

#: Every value of a packed `float` repeat.
pub fn packed_floats(field: Field) -> list[float] {
    let count = packed_width(field, 4, "a packed float repeat")
    let data = field.data
    var out: list[float] = []
    for index in 0..count {
        out.push(f32_from_bits(little_endian(data, index * 4, 4)))
    }
    return out
}

#: Every value of a packed `double` repeat.
pub fn packed_doubles(field: Field) -> list[float] {
    let count = packed_width(field, 8, "a packed double repeat")
    let data = field.data
    var out: list[float] = []
    for index in 0..count {
        out.push(f64_from_bits(little_endian(data, index * 8, 8)))
    }
    return out
}

#: Every value of a packed `fixed32` or `sfixed32` repeat, unsigned.
pub fn packed_fixed32(field: Field) -> list[int] {
    let count = packed_width(field, 4, "a packed fixed32 repeat")
    let data = field.data
    return [little_endian(data, index * 4, 4) for index in 0..count]
}

#: Every value of a packed `fixed64` or `sfixed64` repeat, as the signed
#: reinterpretation `fixed64` on a `Field` also carries.
pub fn packed_fixed64(field: Field) -> list[int] {
    let count = packed_width(field, 8, "a packed fixed64 repeat")
    let data = field.data
    return [little_endian(data, index * 8, 8) for index in 0..count]
}

#: The binary32 value whose bit pattern is the low thirty-two bits of `bits`.
#:
#: `ldexp` does the exponent, which is where a hand-rolled decoder goes wrong:
#: a biased exponent of zero means the leading one is *absent*, so the fraction
#: scales by `2**-149` outright instead of having the hidden bit added first.
#: Get that wrong and every denormal comes back as if it were `2**-126` larger.
fn f32_from_bits(bits: int) -> float {
    let exponent = bits >> 23 & 0xFF
    let fraction = bits & 0x7FFFFF
    var value = 0.0
    if exponent == 0 {
        value = ldexp(float(fraction), -149)
    } elif exponent == 255 {
        value = if fraction == 0 { INF } else { NAN }
    } else {
        value = ldexp(float(fraction) + 8388608.0, exponent - 150)
    }
    if bits >> 31 & 1 == 1 { return -value }
    return value
}

#: The binary64 value whose bit pattern is `bits`.
#:
#: `bits` is the signed reinterpretation the eight wire bytes read as, so the
#: sign bit is simply `bits < 0` and the exponent needs its mask to survive the
#: arithmetic shift.
fn f64_from_bits(bits: int) -> float {
    let exponent = bits >> 52 & 0x7FF
    let fraction = bits & 0xFFFFFFFFFFFFF
    var value = 0.0
    if exponent == 0 {
        value = ldexp(float(fraction), -1074)
    } elif exponent == 2047 {
        value = if fraction == 0 { INF } else { NAN }
    } else {
        value = ldexp(float(fraction) + 4503599627370496.0, exponent - 1075)
    }
    if bits < 0 { return -value }
    return value
}
