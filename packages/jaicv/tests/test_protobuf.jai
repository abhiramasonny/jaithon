#: The wire reader in `jaicv.dnn.protobuf`, against bytes protobuf itself
#: wrote.
#:
#: Every fixture below came out of `SerializeToString` on a real proto2 schema
#: -- `packages/jaicv/tools/protobuf_fixtures.py` holds the schema, encodes the
#: messages and prints these literals -- and every expected value below is what
#: that same script fed in or read back. Nothing here is a golden of jaicv's
#: own output: the point of the exercise is agreeing with protobuf, so hand-
#: written bytes would only prove the reader agrees with whoever wrote them.
#:
#: The malformed cases at the bottom are the exception, and are hand-built on
#: purpose: protobuf's encoder will not emit a truncated length or a group that
#: never closes, and those are exactly the shapes a model file downloaded from
#: somewhere might have.
#:
#: The fixtures are hex rather than lists of byte literals because `jaithon fmt`
#: breaks an over-long element list one element per line, which would spread a
#: hundred-byte message over a hundred lines.
from std.math import INF, copysign, is_nan
from std.test import assert_contains, assert_eq, assert_throws, assert_true
from jaicv.dnn.protobuf import Field, MAX_FIELD_NUMBER
from jaicv.dnn.protobuf import WIRE_END_GROUP, WIRE_I32, WIRE_I64, WIRE_LEN
from jaicv.dnn.protobuf import WIRE_START_GROUP, WIRE_VARINT
from jaicv.dnn.protobuf import as_bool, as_bytes, as_double, as_float, as_signed, as_string
from jaicv.dnn.protobuf import field_by_number, fields_by_number, packed_doubles
from jaicv.dnn.protobuf import packed_fixed32, packed_floats, packed_signed, packed_varints
from jaicv.dnn.protobuf import parse_message, parse_nested

#: One fixture, written as hex in chunks that stay inside the column limit.
fn from_hex(chunks: list[str]) -> bytes {
    var out: list[int] = []
    for chunk in chunks {
        let codes = chunk.bytes()
        if codes.len() % 2 != 0 { throw ValueError(f"a chunk of {codes.len()} hex digits") }
        for index in 0..codes.len() // 2 {
            out.push(nibble(codes[index * 2]) * 16 + nibble(codes[index * 2 + 1]))
        }
    }
    return bytes(out)
}

fn nibble(code: int) -> int {
    if code >= 48 and code <= 57 { return code - 48 }
    if code >= 97 and code <= 102 { return code - 87 }
    throw ValueError(f"'{chr(code)}' is not a lowercase hex digit")
}

let SCALARS = from_hex(
    [
        "08d4fdffffffffffffff0110cb89ec8ff72318ffffffff0f208180808080808010289501",
        "309593d89fee47380145efbeadde4921e81f2952b4b8a855efbeadde591132547698badc",
        "fe650000604069000000000000c4bf720e68c3a96c6c6f2c2077c3b6726c647a0500ff7f",
        "8001820108086312046e657374",
    ]
)

let REPEATS = from_hex(
    [
        "0a270001ffffffffffffffffff01ac02d4fdffffffffffffff01ffffffff0780808080f8",
        "ffffffff01100710f9ffffffffffffffff011080011a1400000000000000800000803f00",
        "0020c077cc2b3222200000000000000000555555555555d53f9c7500883ce437fe695714",
        "8b0abf05402a17000102ffffffffffffffffff01feffffffffffffffff01320401000101",
        "3a0c0000000001000000ffffffff4200420161420d6c6f6e67657220737472696e674a06",
        "080012026d304a06080a12026d314a06081412026d32",
    ]
)

let EXTREMES = from_hex(
    [
        "088080808080808080800110ffffffffffffffff7f18ffffffffffffffffff0120808080",
        "80f8ffffffff0128ffffffff0730ffffffff0f38ffffffffffffffffff0140feffffffff",
        "ffffffff0148015100000000000000805d01000000650000807f6d000080ff750000c07f",
        "7901000000000000008101000000000000f07f8901000000000000f87f9501ffff7f7f99",
        "01ffffffffffffef7f",
    ]
)

let EMPTIES = from_hex(["0a0012001a002005"])

let CHAIN = from_hex(
    [
        "0800122e0801122a08021226080312220804121e0805121a08061216080712120808120e",
        "0809120a080a1206080b1202080c",
    ]
)

let WITH_GROUP = from_hex(["080b130816120767726f757065641a0808211204646565702308372414182c"])

let BIT_PATTERNS = from_hex(
    [
        "0a3400000000000000800000803f000080bf01000000ffff7f0000008000ffff7f7f0000",
        "807f000080ff0000c07fdb0f4940abaaaa3e126800000000000000000000000000000080",
        "000000000000f03f000000000000f0bf0100000000000000ffffffffffff0f0000000000",
        "00001000ffffffffffffef7f000000000000f07f000000000000f0ff000000000000f87f",
        "182d4454fb210940555555555555d53f",
    ]
)

let GROUP_DECOYS = from_hex(["08011308021204140c0b1c141803"])

let HIGH_NUMBERS = from_hex(["0801807d02f8ffffff0f03"])

let EMPTY_MESSAGE = from_hex([""])

let DUPLICATES = from_hex(["080112056669727374080212067365636f6e64"])

let SPLIT_PACKED = from_hex(["0a030102030a020405"])

#: The field the test is about, unwrapped. A missing one is the test's own
#: failure rather than the reader's, so it fails here instead of somewhere the
#: optional finally gets dereferenced.
fn need(fields: list[Field], number: int) -> Field {
    let found = field_by_number(fields, number)
    if found is null { throw AssertionError(f"field {number} is not in the message") }
    return found
}

#: Compare decoded floats to what protobuf was given.
#:
#: Exactly, with no tolerance: reading a bit pattern back is not an
#: approximation and there is nothing for it to round. NaN needs its own arm
#: only because it is not equal to itself.
fn assert_floats(got: list[float], want: list[float], where: str) -> void {
    assert_eq(got.len(), want.len(), f"{where}: wrong count")
    for index in 0..want.len() {
        if is_nan(want[index]) {
            assert_true(is_nan(got[index]), f"{where}[{index}] should be NaN, got {got[index]}")
        } else {
            assert_eq(got[index], want[index], f"{where}[{index}]")
        }
    }
}

#: Whether `value` is a zero carrying a sign bit, which `== -0.0` cannot tell
#: apart from `0.0`.
fn is_negative_zero(value: float) -> bool {
    return value == 0.0 and copysign(1.0, value) == -1.0
}

fn test_every_scalar_kind() -> void {
    let fields = parse_message(SCALARS)
    assert_eq(fields.len(), 16, "one field per member of the schema")

    assert_eq(need(fields, 1).varint, -300, "int32")
    assert_eq(need(fields, 2).varint, 1234567890123, "int64")
    assert_eq(need(fields, 3).varint, 4294967295, "uint32")
    assert_eq(need(fields, 4).varint, 9007199254740993, "uint64")
    assert_eq(as_signed(need(fields, 5)), -75, "sint32")
    assert_eq(as_signed(need(fields, 6)), -1234567890123, "sint64")
    assert_eq(as_bool(need(fields, 7)), true, "bool")
    assert_eq(need(fields, 8).fixed32, 3735928559, "fixed32")

    #: `fixed64` and `sfixed64` are the same eight bytes; there is no unsigned
    #: 64-bit integer to put the first in, so both read as the signed value.
    assert_eq(need(fields, 9).fixed64, -6289078614652622815, "fixed64 above 2**63")
    assert_eq(need(fields, 10).fixed32, 3735928559, "sfixed32 arrives unsigned")
    assert_eq(need(fields, 11).fixed64, -81985529216486895, "sfixed64")

    assert_eq(as_float(need(fields, 12)), 3.5, "float")
    assert_eq(as_double(need(fields, 13)), -0.15625, "double")
    assert_eq(as_string(need(fields, 14)), "héllo, wörld", "a string outside ASCII")
    assert_eq(as_bytes(need(fields, 15)).to_list(), [0x00, 0xFF, 0x7F, 0x80, 0x01], "bytes")
}

fn test_wire_types_are_reported() -> void {
    let fields = parse_message(SCALARS)
    assert_eq(need(fields, 1).wire, WIRE_VARINT, "int32 is a varint")
    assert_eq(need(fields, 9).wire, WIRE_I64, "fixed64 is 64-bit")
    assert_eq(need(fields, 12).wire, WIRE_I32, "float is 32-bit")
    assert_eq(need(fields, 14).wire, WIRE_LEN, "a string is length-delimited")
    assert_eq(need(fields, 16).wire, WIRE_LEN, "a nested message is length-delimited")
}

fn test_nested_message() -> void {
    let inner = parse_nested(need(parse_message(SCALARS), 16))
    assert_eq(need(inner, 1).varint, 99, "the nested int")
    assert_eq(as_string(need(inner, 2)), "nest", "the nested string")
}

fn test_packed_repeats() -> void {
    let fields = parse_message(REPEATS)

    assert_eq(
        packed_varints(need(fields, 1)),
        [0, 1, -1, 300, -300, 2147483647, -2147483648],
        "packed int32, negatives sign-extended to ten bytes"
    )
    assert_floats(
        packed_floats(need(fields, 3)),
        [0.0, -0.0, 1.0, -2.5, 9.99999993922529e-09],
        "packed float"
    )
    assert_true(is_negative_zero(packed_floats(need(fields, 3))[1]), "a packed -0.0 keeps its sign")
    assert_floats(
        packed_doubles(need(fields, 4)),
        [0.0, 0.3333333333333333, -1e300, 2.718281828459045],
        "packed double"
    )
    assert_eq(
        packed_signed(need(fields, 5)),
        [0, -1, 1, (-9223372036854775808), 9223372036854775807],
        "packed sint64 over the whole range"
    )
    assert_eq(packed_varints(need(fields, 6)), [1, 0, 1, 1], "packed bool")
    assert_eq(packed_fixed32(need(fields, 7)), [0, 1, 4294967295], "packed fixed32")
}

fn test_unpacked_repeats() -> void {
    let repeated = fields_by_number(parse_message(REPEATS), 2)
    assert_eq(repeated.len(), 3, "an unpacked repeat is one field per value")
    assert_eq([field.varint for field in repeated], [7, -7, 128], "unpacked int32")
}

fn test_repeated_strings_and_messages() -> void {
    let fields = parse_message(REPEATS)
    let strings = fields_by_number(fields, 8)
    assert_eq([as_string(field) for field in strings], ["", "a", "longer string"], "strings")

    let messages = fields_by_number(fields, 9)
    assert_eq(messages.len(), 3, "three repeated messages")
    for index in 0..3 {
        let inner = parse_nested(messages[index])
        assert_eq(need(inner, 1).varint, index * 10, f"message {index}")
        assert_eq(as_string(need(inner, 2)), f"m{index}", f"message {index}")
    }
}

fn test_integer_extremes() -> void {
    let fields = parse_message(EXTREMES)
    assert_eq(need(fields, 1).varint, (-9223372036854775808), "the smallest int64")
    assert_eq(need(fields, 2).varint, 9223372036854775807, "the largest int64")

    #: 2**64 - 1 has no room left for a sign, so it comes back as the negative
    #: integer sharing its bit pattern. Nothing wider exists to hold it.
    assert_eq(need(fields, 3).varint, -1, "the largest uint64")

    assert_eq(need(fields, 4).varint, -2147483648, "the smallest int32")
    assert_eq(need(fields, 5).varint, 2147483647, "the largest int32")
    assert_eq(need(fields, 6).varint, 4294967295, "the largest uint32")
    assert_eq(as_signed(need(fields, 7)), (-9223372036854775808), "the smallest sint64")
    assert_eq(as_signed(need(fields, 8)), 9223372036854775807, "the largest sint64")
    assert_eq(as_signed(need(fields, 9)), -1, "sint32 -1, one byte rather than ten")
}

fn test_float_extremes() -> void {
    let fields = parse_message(EXTREMES)
    assert_true(is_negative_zero(as_double(need(fields, 10))), "-0.0 keeps its sign bit")
    assert_eq(as_float(need(fields, 11)), 1.401298464324817e-45, "the smallest float denormal")
    assert_eq(as_float(need(fields, 12)), INF, "float infinity")
    assert_eq(as_float(need(fields, 13)), -INF, "negative float infinity")
    assert_true(is_nan(as_float(need(fields, 14))), "float NaN")
    assert_eq(as_double(need(fields, 15)), 5e-324, "the smallest double denormal")
    assert_eq(as_double(need(fields, 16)), INF, "double infinity")
    assert_true(is_nan(as_double(need(fields, 17))), "double NaN")
    assert_eq(as_float(need(fields, 18)), 3.4028234663852886e38, "the largest finite float")
    assert_eq(as_double(need(fields, 19)), 1.7976931348623157e308, "the largest finite double")
}

fn test_float_bit_patterns() -> void {
    #: Both zeroes, the denormal boundary from either side, the largest finite
    #: value, both infinities and a quiet NaN -- the patterns a decoder that
    #: forgets the hidden mantissa bit gets wrong.
    let decoded = packed_floats(need(parse_message(BIT_PATTERNS), 1))
    assert_floats(
        decoded,
        [
            0.0,
            -0.0,
            1.0,
            -1.0,
            1.401298464324817e-45,
            1.1754942106924411e-38,
            1.1754943508222875e-38,
            3.4028234663852886e38,
            INF,
            -INF,
            0.0 * INF,
            3.1415927410125732,
            0.3333333432674408,
        ],
        "binary32 patterns"
    )
    assert_true(is_negative_zero(decoded[1]), "0x80000000 is -0.0, not 0.0")
    assert_true(is_nan(decoded[10]), "0x7FC00000 is a quiet NaN")
}

fn test_double_bit_patterns() -> void {
    let decoded = packed_doubles(need(parse_message(BIT_PATTERNS), 2))
    assert_floats(
        decoded,
        [
            0.0,
            -0.0,
            1.0,
            -1.0,
            5e-324,
            2.225073858507201e-308,
            2.2250738585072014e-308,
            1.7976931348623157e308,
            INF,
            -INF,
            0.0 * INF,
            3.141592653589793,
            0.3333333333333333,
        ],
        "binary64 patterns"
    )
    assert_true(is_negative_zero(decoded[1]), "0x8000000000000000 is -0.0, not 0.0")
    assert_true(is_nan(decoded[10]), "0x7FF8000000000000 is a quiet NaN")
}

fn test_empty_message() -> void {
    assert_eq(parse_message(EMPTY_MESSAGE).len(), 0, "no bytes means no fields")
}

fn test_empty_values() -> void {
    let fields = parse_message(EMPTIES)
    assert_eq(fields.len(), 4, "an empty value is still a field on the wire")
    assert_eq(as_string(need(fields, 1)), "", "an empty string")
    assert_eq(as_bytes(need(fields, 2)).len(), 0, "empty bytes")
    assert_eq(parse_nested(need(fields, 3)).len(), 0, "an empty nested message")
    assert_eq(need(fields, 4).varint, 5, "the field after the empty ones")
}

fn test_deep_nesting() -> void {
    var fields = parse_message(CHAIN)
    for depth in 0..12 {
        assert_eq(need(fields, 1).varint, depth, f"the link at depth {depth}")
        fields = parse_nested(need(fields, 2))
    }
    assert_eq(need(fields, 1).varint, 12, "the innermost link")
    assert_true(field_by_number(fields, 2) is null, "the innermost link has no next")
}

fn test_groups_keep_their_body_and_the_stream() -> void {
    let fields = parse_message(WITH_GROUP)
    assert_eq(fields.len(), 3, "the field before, the group, and the field after")
    assert_eq(need(fields, 1).varint, 11, "before the group")
    assert_eq(need(fields, 3).varint, 44, "after the group, so the group was skipped exactly")

    let group = need(fields, 2)
    assert_eq(group.wire, WIRE_START_GROUP, "field 2 is a group")

    #: A group body is a message, so it parses like one.
    let body = parse_message(group.data)
    assert_eq(need(body, 1).varint, 22, "the group's int")
    assert_eq(as_string(need(body, 2)), "grouped", "the group's string")

    let payload = parse_nested(need(body, 3))
    assert_eq(need(payload, 1).varint, 33, "a message inside the group")
    assert_eq(as_string(need(payload, 2)), "deep", "a message inside the group")

    let inner = need(body, 4)
    assert_eq(inner.wire, WIRE_START_GROUP, "a group inside a group")
    assert_eq(need(parse_message(inner.data), 1).varint, 55, "the inner group's leaf")
}

fn test_group_body_may_hold_bytes_that_read_as_tags() -> void {
    #: The string inside this group is `14 0C 0B 1C`, which reads as an end-
    #: group tag for field 2 followed by a start-group tag for field 1. Skip
    #: the group by scanning for a closing tag rather than by honouring each
    #: field's length and the group ends four bytes early.
    let fields = parse_message(GROUP_DECOYS)
    assert_eq(fields.len(), 3, "the decoy tags did not end the group early")
    assert_eq(need(fields, 3).varint, 3, "the field after the group")

    let body = parse_message(need(fields, 2).data)
    assert_eq(need(body, 1).varint, 2, "the group's int")
    assert_eq(as_bytes(need(body, 2)).to_list(), [0x14, 0x0C, 0x0B, 0x1C], "the decoys")
}

fn test_high_field_numbers() -> void {
    let fields = parse_message(HIGH_NUMBERS)
    assert_eq(need(fields, 1).varint, 1, "a one-byte tag")
    assert_eq(need(fields, 2000).varint, 2, "a two-byte tag")
    assert_eq(need(fields, MAX_FIELD_NUMBER).varint, 3, "the largest field number")
}

fn test_an_unwanted_field_does_not_break_the_stream() -> void {
    #: Reading only field 3 out of a message whose first two fields are a
    #: ten-byte varint and a two-byte one is the whole reason a schemaless
    #: reader works: what it does not recognise it still measures.
    let fields = parse_message(SCALARS)
    assert_eq(need(fields, 16).wire, WIRE_LEN, "the last field survived thirteen it skipped")
    assert_true(field_by_number(fields, 17) is null, "a number nobody wrote is absent")
}

fn test_duplicate_field_numbers_take_the_last() -> void {
    #: Two encodings concatenated is protobuf's own merge; protobuf reads this
    #: back as `inner: 2, label: "second"`, so the reader has to as well.
    let fields = parse_message(DUPLICATES)
    assert_eq(need(fields, 1).varint, 2, "the later value of a singular field wins")
    assert_eq(as_string(need(fields, 2)), "second", "the later string wins")
    assert_eq(
        [field.varint for field in fields_by_number(fields, 1)],
        [1, 2],
        "both are still there for a caller that wants the repeats"
    )
}

fn test_a_packed_repeat_split_across_runs() -> void {
    #: A merge can leave one packed repeat as several length-delimited runs.
    #: Protobuf reads this back as one list of five, so joining the runs has to
    #: give the same five.
    let runs = fields_by_number(parse_message(SPLIT_PACKED), 1)
    assert_eq(runs.len(), 2, "two runs of the same field number")
    var joined: list[int] = []
    for run in runs {
        for value in packed_varints(run) { joined.push(value) }
    }
    assert_eq(joined, [1, 2, 3, 4, 5], "the runs join into the list protobuf reads")
}

fn test_a_varint_running_past_the_end_throws() -> void {
    let error = assert_throws(ValueError, || parse_message(bytes([0x08, 0x80, 0x80])))
    assert_contains(error.message, "runs past the end", "says what went wrong")
}

fn test_an_unterminated_varint_throws() -> void {
    #: Eleven continuation bytes. Nothing here may spin: the reader stops at
    #: ten, which is as long as a 64-bit varint can be.
    let raw = bytes([0x08, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01])
    let error = assert_throws(ValueError, || parse_message(raw))
    assert_contains(error.message, "unterminated", "says what went wrong")
}

fn test_a_length_running_past_the_end_throws() -> void {
    let error = assert_throws(ValueError, || parse_message(bytes([0x12, 0x05, 0x61])))
    assert_contains(error.message, "runs past", "says what went wrong")
    assert_contains(error.message, "field 2", "says which field")
}

fn test_a_negative_length_throws() -> void {
    #: A length varint with its top bit set decodes to a negative integer.
    #: Left unchecked it walks the cursor backwards instead of forwards.
    let raw = bytes([0x0A, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01])
    let error = assert_throws(ValueError, || parse_message(raw))
    assert_contains(error.message, "is not a length", "says what went wrong")
}

fn test_a_truncated_fixed_field_throws() -> void {
    let short32 = assert_throws(ValueError, || parse_message(bytes([0x0D, 0x01, 0x02])))
    assert_contains(short32.message, "4-byte field runs past", "a truncated fixed32")
    let short64 = assert_throws(ValueError, || parse_message(bytes([0x09, 0x01])))
    assert_contains(short64.message, "8-byte field runs past", "a truncated fixed64")
}

fn test_an_undefined_wire_type_throws() -> void {
    for wire in [6, 7] {
        let error = assert_throws(ValueError, || parse_message(bytes([8 | wire])))
        assert_contains(error.message, "not one protobuf defines", f"wire type {wire}")
    }
}

fn test_a_field_number_outside_the_range_throws() -> void {
    let zero = assert_throws(ValueError, || parse_message(bytes([0x00])))
    assert_contains(zero.message, "field 0", "field numbers start at one")

    #: A tag of 2**32, which names field 2**29 -- one past the largest.
    let raw = bytes([0x80, 0x80, 0x80, 0x80, 0x10])
    let over = assert_throws(ValueError, || parse_message(raw))
    assert_contains(over.message, f"1..{MAX_FIELD_NUMBER}", "says what the range is")
}

fn test_a_stray_end_group_throws() -> void {
    let error = assert_throws(ValueError, || parse_message(bytes([0x0C])))
    assert_contains(error.message, "no group open", "says what went wrong")
}

fn test_a_group_that_never_closes_throws() -> void {
    let error = assert_throws(ValueError, || parse_message(bytes([0x0B, 0x08, 0x01])))
    assert_contains(error.message, "never closed", "says what went wrong")
}

fn test_a_mismatched_end_group_throws() -> void {
    #: Field 1 opens the group and field 2 closes it. Counting depth instead of
    #: tracking numbers would take this for a well-formed group.
    let error = assert_throws(ValueError, || parse_message(bytes([0x0B, 0x14])))
    assert_contains(error.message, "closed by an end tag for field 2", "says what went wrong")
}

fn test_groups_nested_too_deep_throw() -> void {
    let raw = bytes([0x0B for _index in 0..101])
    let error = assert_throws(ValueError, || parse_message(raw))
    assert_contains(error.message, "nest more than", "says what went wrong")
}

fn test_a_nested_message_that_does_not_parse_throws() -> void {
    #: The outer message is well formed; the two bytes inside field 1 are not.
    let error = assert_throws(
        ValueError,
        || parse_nested(need(parse_message(bytes([0x0A, 0x02, 0x08, 0x80])), 1))
    )
    assert_contains(error.message, "runs past the end", "says what went wrong")
}

fn test_a_packed_repeat_of_the_wrong_length_throws() -> void {
    let odd_floats = Field(1, WIRE_LEN, data: bytes([0x00, 0x00, 0x80]))
    let float_error = assert_throws(ValueError, || packed_floats(odd_floats))
    assert_contains(float_error.message, "not whole 4-byte values", "three bytes is not a float")

    let odd_doubles = Field(1, WIRE_LEN, data: bytes([0x00, 0x00, 0x00, 0x00]))
    let double_error = assert_throws(ValueError, || packed_doubles(odd_doubles))
    assert_contains(double_error.message, "not whole 8-byte values", "four bytes is not a double")
}

fn test_a_string_that_is_not_utf8_throws() -> void {
    let error = assert_throws(ValueError, || as_string(Field(4, WIRE_LEN, data: bytes([0xFF]))))
    assert_contains(error.message, "field 4", "says which field")
    assert_contains(error.message, "UTF-8", "says what went wrong")
}

fn test_reading_a_field_as_the_wrong_kind_throws() -> void {
    let varint = need(parse_message(SCALARS), 1)
    let as_a_float = assert_throws(ValueError, || as_float(varint))
    assert_contains(as_a_float.message, "does not hold a float", "a varint is not a float")

    let text = need(parse_message(SCALARS), 14)
    let as_a_double = assert_throws(ValueError, || as_double(text))
    assert_contains(as_a_double.message, "length-delimited", "names the wire type it found")

    let nested = assert_throws(ValueError, || parse_nested(varint))
    assert_contains(nested.message, "no nested message", "a varint holds no message")
}

fn test_an_end_group_wire_type_is_named_in_errors() -> void {
    let error = assert_throws(ValueError, || as_string(Field(1, WIRE_END_GROUP)))
    assert_contains(error.message, "end-group", "the wire type is named, not numbered")
}
