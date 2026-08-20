#!/usr/bin/env python3
"""Encode the protobuf fixtures `test_protobuf.jai` reads, with real protobuf.

The reader under test has no schema and no generated code: it walks the wire
format, so what it has to agree with is exactly what protobuf's own encoder
emits. This writes a proto2 schema covering every scalar kind, both packed and
unpacked repeats, nesting, the deprecated group encoding and the extremes of
each integer range, compiles it with the protoc that ships in grpcio-tools,
serialises one message of each and prints the bytes as Jaithon literals.

    cd /tmp && uv run --with protobuf --with grpcio-tools \\
        python ~/Developer/projects/jaithon/packages/jaicv/tools/protobuf_fixtures.py

Everything after the literals is a comment recording what protobuf put in and
what it reads back out, which is where the expected values in the test come
from. Paste the printed block over the fixture section of
`packages/jaicv/tests/test_protobuf.jai`.
"""
from __future__ import annotations

import importlib.util
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

SCHEMA = '''
syntax = "proto2";
package jaicvfix;

message Nested {
  optional int32 inner = 1;
  optional string label = 2;
}

message Chain {
  optional int32 depth = 1;
  optional Chain next = 2;
}

message Scalars {
  optional int32 f_int32 = 1;
  optional int64 f_int64 = 2;
  optional uint32 f_uint32 = 3;
  optional uint64 f_uint64 = 4;
  optional sint32 f_sint32 = 5;
  optional sint64 f_sint64 = 6;
  optional bool f_bool = 7;
  optional fixed32 f_fixed32 = 8;
  optional fixed64 f_fixed64 = 9;
  optional sfixed32 f_sfixed32 = 10;
  optional sfixed64 f_sfixed64 = 11;
  optional float f_float = 12;
  optional double f_double = 13;
  optional string f_string = 14;
  optional bytes f_bytes = 15;
  optional Nested f_nested = 16;
}

message Repeats {
  repeated int32 packed_ints = 1 [packed = true];
  repeated int32 unpacked_ints = 2 [packed = false];
  repeated float packed_floats = 3 [packed = true];
  repeated double packed_doubles = 4 [packed = true];
  repeated sint64 packed_zigzag = 5 [packed = true];
  repeated bool packed_bools = 6 [packed = true];
  repeated fixed32 packed_fixed32 = 7 [packed = true];
  repeated string strings = 8;
  repeated Nested messages = 9;
}

message Extremes {
  optional int64 min_int64 = 1;
  optional int64 max_int64 = 2;
  optional uint64 max_uint64 = 3;
  optional int32 min_int32 = 4;
  optional int32 max_int32 = 5;
  optional uint32 max_uint32 = 6;
  optional sint64 min_sint64 = 7;
  optional sint64 max_sint64 = 8;
  optional sint32 neg_one_sint32 = 9;
  optional double neg_zero = 10;
  optional float f_denormal = 11;
  optional float f_inf = 12;
  optional float f_neg_inf = 13;
  optional float f_nan = 14;
  optional double d_denormal = 15;
  optional double d_inf = 16;
  optional double d_nan = 17;
  optional float f_max = 18;
  optional double d_max = 19;
}

message Empties {
  optional string empty_string = 1;
  optional bytes empty_bytes = 2;
  optional Nested empty_nested = 3;
  optional int32 after = 4;
}

message WithGroup {
  optional int32 before = 1;
  optional group Body = 2 {
    optional int32 value = 1;
    optional string tag = 2;
    optional Nested payload = 3;
    optional group Inner = 4 {
      optional int32 leaf = 1;
    }
  }
  optional int32 after = 3;
}

message BitPatterns {
  repeated float f32 = 1 [packed = true];
  repeated double f64 = 2 [packed = true];
}

message HighNumbers {
  optional int32 low = 1;
  optional int32 mid = 2000;
  optional int32 high = 536870911;
}
'''


def compile_schema(work: Path):
    (work / "fixtures.proto").write_text(SCHEMA)
    subprocess.run(
        [sys.executable, "-m", "grpc_tools.protoc",
         f"-I{work}", f"--python_out={work}", "fixtures.proto"],
        check=True,
    )
    spec = importlib.util.spec_from_file_location("fixtures_pb2", work / "fixtures_pb2.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["fixtures_pb2"] = module
    spec.loader.exec_module(module)
    return module


#: Hex rather than a list of byte literals, in chunks that stay inside the
#: formatter's column limit: `jaithon fmt` breaks an over-long element list one
#: element per line, which turns a hundred-byte message into a hundred lines.
CHUNK = 72


def literal(name: str, blob: bytes) -> str:
    if not blob:
        return f'let {name} = from_hex([""])'
    text = blob.hex()
    chunks = [text[at:at + CHUNK] for at in range(0, len(text), CHUNK)]
    if len(chunks) == 1:
        return f'let {name} = from_hex(["{chunks[0]}"])'
    inner = ",\n    ".join(f'"{chunk}"' for chunk in chunks)
    return f"let {name} = from_hex([\n    {inner}\n])"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        pb = compile_schema(Path(tmp))
        emit(pb)
    return 0


def emit(pb) -> None:
    out = []
    facts = []

    scalars = pb.Scalars()
    scalars.f_int32 = -300
    scalars.f_int64 = 1234567890123
    scalars.f_uint32 = 4294967295
    scalars.f_uint64 = 9007199254740993
    scalars.f_sint32 = -75
    scalars.f_sint64 = -1234567890123
    scalars.f_bool = True
    scalars.f_fixed32 = 3735928559
    scalars.f_fixed64 = 12157665459056928801
    scalars.f_sfixed32 = -559038737
    scalars.f_sfixed64 = -81985529216486895
    scalars.f_float = 3.5
    scalars.f_double = -0.15625
    scalars.f_string = "héllo, wörld"
    scalars.f_bytes = bytes([0x00, 0xFF, 0x7F, 0x80, 0x01])
    scalars.f_nested.inner = 99
    scalars.f_nested.label = "nest"
    out.append(literal("SCALARS", scalars.SerializeToString()))
    facts.append(("SCALARS", scalars))

    repeats = pb.Repeats()
    repeats.packed_ints.extend([0, 1, -1, 300, -300, 2147483647, -2147483648])
    repeats.unpacked_ints.extend([7, -7, 128])
    repeats.packed_floats.extend([0.0, -0.0, 1.0, -2.5, 1e-8])
    repeats.packed_doubles.extend([0.0, 1.0 / 3.0, -1e300, 2.718281828459045])
    repeats.packed_zigzag.extend([0, -1, 1, -9223372036854775808, 9223372036854775807])
    repeats.packed_bools.extend([True, False, True, True])
    repeats.packed_fixed32.extend([0, 1, 4294967295])
    repeats.strings.extend(["", "a", "longer string"])
    for index in range(3):
        item = repeats.messages.add()
        item.inner = index * 10
        item.label = f"m{index}"
    out.append(literal("REPEATS", repeats.SerializeToString()))
    facts.append(("REPEATS", repeats))

    extremes = pb.Extremes()
    extremes.min_int64 = -9223372036854775808
    extremes.max_int64 = 9223372036854775807
    extremes.max_uint64 = 18446744073709551615
    extremes.min_int32 = -2147483648
    extremes.max_int32 = 2147483647
    extremes.max_uint32 = 4294967295
    extremes.min_sint64 = -9223372036854775808
    extremes.max_sint64 = 9223372036854775807
    extremes.neg_one_sint32 = -1
    extremes.neg_zero = -0.0
    extremes.f_denormal = struct.unpack("<f", struct.pack("<I", 1))[0]
    extremes.f_inf = math.inf
    extremes.f_neg_inf = -math.inf
    extremes.f_nan = math.nan
    extremes.d_denormal = struct.unpack("<d", struct.pack("<Q", 1))[0]
    extremes.d_inf = math.inf
    extremes.d_nan = math.nan
    extremes.f_max = struct.unpack("<f", struct.pack("<I", 0x7F7FFFFF))[0]
    extremes.d_max = struct.unpack("<d", struct.pack("<Q", 0x7FEFFFFFFFFFFFFF))[0]
    out.append(literal("EXTREMES", extremes.SerializeToString()))
    facts.append(("EXTREMES", extremes))

    empties = pb.Empties()
    empties.empty_string = ""
    empties.empty_bytes = b""
    empties.empty_nested.SetInParent()
    empties.after = 5
    out.append(literal("EMPTIES", empties.SerializeToString()))
    facts.append(("EMPTIES", empties))

    chain = pb.Chain()
    cursor = chain
    for depth in range(12):
        cursor.depth = depth
        cursor = cursor.next
    cursor.depth = 12
    out.append(literal("CHAIN", chain.SerializeToString()))
    facts.append(("CHAIN", chain))

    grouped = pb.WithGroup()
    grouped.before = 11
    grouped.body.value = 22
    grouped.body.tag = "grouped"
    grouped.body.payload.inner = 33
    grouped.body.payload.label = "deep"
    grouped.body.inner.leaf = 55
    grouped.after = 44
    out.append(literal("WITH_GROUP", grouped.SerializeToString()))
    facts.append(("WITH_GROUP", grouped))

    #: The patterns a float decoder gets wrong: both zeroes, the denormal
    #: boundary either side, the largest finite value, both infinities and a
    #: quiet NaN. Written as the values they stand for, so protobuf lays the
    #: patterns back down itself.
    f32_patterns = [0x00000000, 0x80000000, 0x3F800000, 0xBF800000, 0x00000001,
                    0x007FFFFF, 0x00800000, 0x7F7FFFFF, 0x7F800000, 0xFF800000,
                    0x7FC00000, 0x40490FDB, 0x3EAAAAAB]
    f64_patterns = [0x0000000000000000, 0x8000000000000000, 0x3FF0000000000000,
                    0xBFF0000000000000, 0x0000000000000001, 0x000FFFFFFFFFFFFF,
                    0x0010000000000000, 0x7FEFFFFFFFFFFFFF, 0x7FF0000000000000,
                    0xFFF0000000000000, 0x7FF8000000000000, 0x400921FB54442D18,
                    0x3FD5555555555555]
    patterns = pb.BitPatterns()
    patterns.f32.extend(struct.unpack("<f", struct.pack("<I", p))[0] for p in f32_patterns)
    patterns.f64.extend(struct.unpack("<d", struct.pack("<Q", p))[0] for p in f64_patterns)
    out.append(literal("BIT_PATTERNS", patterns.SerializeToString()))

    #: A group whose body holds bytes that read as end-group tags. The skipper
    #: has to step over that field by its length; scan for the tag instead and
    #: the group ends in the middle of a string.
    decoys = pb.WithGroup()
    decoys.before = 1
    decoys.body.value = 2
    decoys.body.tag = "\x14\x0c\x0b\x1c"
    decoys.after = 3
    out.append(literal("GROUP_DECOYS", decoys.SerializeToString()))
    facts.append(("GROUP_DECOYS", decoys))

    high = pb.HighNumbers()
    high.low = 1
    high.mid = 2
    high.high = 3
    out.append(literal("HIGH_NUMBERS", high.SerializeToString()))
    facts.append(("HIGH_NUMBERS", high))

    out.append(literal("EMPTY_MESSAGE", pb.Nested().SerializeToString()))

    #: Concatenating two encodings is protobuf's own merge, and the only way
    #: an encoder ever writes one field number twice for a singular field.
    first = pb.Nested(inner=1, label="first")
    second = pb.Nested(inner=2, label="second")
    duplicates = first.SerializeToString() + second.SerializeToString()
    out.append(literal("DUPLICATES", duplicates))
    facts.append(("DUPLICATES (parsed back by protobuf)", pb.Nested.FromString(duplicates)))

    #: A packed repeat may arrive as several length-delimited runs; the reader
    #: has to hand back every run so the caller can join them.
    left = pb.Repeats()
    left.packed_ints.extend([1, 2, 3])
    right = pb.Repeats()
    right.packed_ints.extend([4, 5])
    split = left.SerializeToString() + right.SerializeToString()
    out.append(literal("SPLIT_PACKED", split))
    facts.append(("SPLIT_PACKED (parsed back by protobuf)", pb.Repeats.FromString(split)))

    print("\n\n".join(out))
    print()
    print("# ---- what protobuf put in, for the assertions ----")
    for name, message in facts:
        print(f"# {name}")
        for line in str(message).splitlines():
            print(f"#   {line}")
        print(f"#   raw hex: {message.SerializeToString().hex()}")
    print("# BIT_PATTERNS f32, pattern then the value it stands for")
    for pattern in f32_patterns:
        print(f"#   {pattern:#010x} {struct.unpack('<f', struct.pack('<I', pattern))[0]!r}")
    print("# BIT_PATTERNS f64, pattern then the value it stands for")
    for pattern in f64_patterns:
        print(f"#   {pattern:#018x} {struct.unpack('<d', struct.pack('<Q', pattern))[0]!r}")
    print("# float bit patterns")
    for label, value in [("f_denormal", extremes.f_denormal), ("f_max", extremes.f_max)]:
        print(f"#   {label} = {struct.unpack('<I', struct.pack('<f', value))[0]:#010x} {value!r}")
    for label, value in [("d_denormal", extremes.d_denormal), ("d_max", extremes.d_max),
                         ("neg_zero", extremes.neg_zero)]:
        print(f"#   {label} = {struct.unpack('<Q', struct.pack('<d', value))[0]:#018x} {value!r}")


if __name__ == "__main__":
    raise SystemExit(main())
