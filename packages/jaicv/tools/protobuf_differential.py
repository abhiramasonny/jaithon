#!/usr/bin/env python3
"""Read random protobuf with jaicv and with protobuf, and diff the two.

`test_protobuf.jai` pins the cases worth naming. This covers the ones nobody
would think to name: a few thousand messages of random field numbers, wire
types and values, plus a few thousand corrupted and outright random buffers, on
each of which the only question is whether the reader accepts it and what it
says it holds.

Protobuf's own parser is the reference on both halves. A message with no fields
at all sends every field into the unknown-field set, which carries exactly what
`parse_message` returns -- number, wire type, and the value or the bytes -- so
the two decodings compare directly, groups and all.

    cd /tmp && uv run --with protobuf \\
        python ~/Developer/projects/jaithon/packages/jaicv/tools/protobuf_differential.py

Nothing is written into the tree: the cases and the Jaithon driver live in a
temporary directory for the length of the run.
"""
from __future__ import annotations

import random
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from google.protobuf.internal import encoder
from google.protobuf.unknown_fields import UnknownFieldSet

ROOT = Path(__file__).resolve().parents[3]

#: Enough draws to reach every branch of the reader many times over, and few
#: enough that the run stays inside a few seconds.
MESSAGES = 3000
FUZZ = 6000

#: A message with no fields, so protobuf files everything it reads under
#: unknown fields instead of interpreting any of it.
BLANK = """
name: "blank.proto"
package: "diff"
syntax: "proto2"
"""

DRIVER = '''
from std.io import read_file, write_file
from jaicv.dnn.protobuf import Field, WIRE_I32, WIRE_I64, WIRE_LEN, WIRE_START_GROUP
from jaicv.dnn.protobuf import WIRE_VARINT, parse_message

let DIGITS = "0123456789abcdef"

fn nibble(code: int) -> int {
    if code >= 48 and code <= 57 { return code - 48 }
    if code >= 97 and code <= 102 { return code - 87 }
    throw ValueError(f"not a hex digit: {code}")
}

fn from_hex(text: str) -> bytes {
    let codes = text.bytes()
    var out: list[int] = []
    for index in 0..codes.len() // 2 {
        out.push(nibble(codes[index * 2]) * 16 + nibble(codes[index * 2 + 1]))
    }
    return bytes(out)
}

fn to_hex(data: bytes) -> str {
    var out = ""
    for index in 0..data.len() {
        let byte = data[index]
        out += DIGITS[byte // 16] + DIGITS[byte % 16]
    }
    return out
}

fn flatten(fields: list[Field], prefix: str, rows: list[str]) -> void {
    for field in fields {
        let where = f"{prefix}{field.number}"
        if field.wire == WIRE_VARINT {
            rows.push(f"{where}:0:{field.varint}")
        } elif field.wire == WIRE_I64 {
            rows.push(f"{where}:1:{field.fixed64}")
        } elif field.wire == WIRE_LEN {
            rows.push(f"{where}:2:{to_hex(field.data)}")
        } elif field.wire == WIRE_I32 {
            rows.push(f"{where}:5:{field.fixed32}")
        } elif field.wire == WIRE_START_GROUP {
            rows.push(f"{where}:3:")
            flatten(parse_message(field.data), where + "/", rows)
        }
    }
}

fn main(args: list[str]) -> void {
    let dir = args[args.len() - 1]

    var decoded: list[str] = []
    for line in read_file(dir + "/cases.hex").split("\\n") {
        if line == "" { continue }
        var rows: list[str] = []
        flatten(parse_message(from_hex(line)), "", rows)
        decoded.push(" ".join(rows))
    }
    write_file(dir + "/got.txt", "\\n".join(decoded) + "\\n")

    var verdicts: list[str] = []
    for line in read_file(dir + "/fuzz.hex").split("\\n") {
        if line == "" { continue }
        try {
            let _fields = parse_message(from_hex(line))
            verdicts.push("accept")
        } catch _error: ValueError {
            verdicts.push("reject")
        }
    }
    write_file(dir + "/fuzz_got.txt", "\\n".join(verdicts) + "\\n")
}
'''


def blank_message():
    pool = descriptor_pool.DescriptorPool()
    proto = descriptor_pb2.FileDescriptorProto()
    proto.name = "blank.proto"
    proto.package = "diff"
    proto.syntax = "proto2"
    proto.message_type.add().name = "Blank"
    pool.Add(proto)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("diff.Blank"))


def tag(number: int, wire: int) -> bytes:
    return encoder._VarintBytes((number << 3) | wire)


def random_field(depth: int) -> bytes:
    number = random.choice([1, 2, 3, 7, 15, 16, 300, 2000, 100000, 536870911])
    wire = random.choice([0, 0, 0, 1, 2, 2, 5] + ([3] if depth < 3 else []))
    if wire == 0:
        #: Weighted at the boundaries, where a varint changes length or runs
        #: out of room for a sign.
        value = random.choice([
            0, 1, 127, 128, 300, 2**31 - 1, 2**31, 2**32 - 1, 2**63 - 1, 2**64 - 1,
            random.randrange(2**64),
        ])
        return tag(number, 0) + encoder._VarintBytes(value)
    if wire == 1:
        return tag(number, 1) + struct.pack("<Q", random.getrandbits(64))
    if wire == 5:
        return tag(number, 5) + struct.pack("<I", random.getrandbits(32))
    if wire == 2:
        blob = bytes(random.getrandbits(8) for _ in range(random.randint(0, 30)))
        return tag(number, 2) + encoder._VarintBytes(len(blob)) + blob
    body = b"".join(random_field(depth + 1) for _ in range(random.randint(0, 3)))
    return tag(number, 3) + body + tag(number, 4)


def flatten(unknowns, prefix: str = "") -> list[str]:
    rows = []
    for field in unknowns:
        where = f"{prefix}{field.field_number}"
        if field.wire_type in (0, 1):
            #: Jaithon has no unsigned 64-bit integer, so the reader hands back
            #: the signed value sharing the pattern and the oracle follows.
            value = field.data
            signed = value - 2**64 if value >= 2**63 else value
            rows.append(f"{where}:{field.wire_type}:{signed}")
        elif field.wire_type == 2:
            rows.append(f"{where}:2:{field.data.hex()}")
        elif field.wire_type == 5:
            rows.append(f"{where}:5:{field.data}")
        elif field.wire_type == 3:
            rows.append(f"{where}:3:")
            rows.extend(flatten(field.data, where + "/"))
        else:
            raise SystemExit(f"protobuf returned wire type {field.wire_type}")
    return rows


def write_cases(blank, work: Path) -> tuple[list[str], list[str]]:
    cases, oracle = [], []
    while len(cases) < MESSAGES:
        raw = b"".join(random_field(0) for _ in range(random.randint(1, 12)))
        try:
            rows = flatten(UnknownFieldSet(blank.FromString(raw)))
        except Exception:
            continue
        cases.append(raw.hex())
        oracle.append(" ".join(rows))
    work.joinpath("cases.hex").write_text("\n".join(cases) + "\n")
    return cases, oracle


def write_fuzz(blank, work: Path, valid: list[str]) -> list[str]:
    lines, verdicts = [], []
    raws = [bytes.fromhex(case) for case in valid]
    while len(lines) < FUZZ:
        kind = random.randrange(4)
        if kind == 0:
            raw = bytes(random.getrandbits(8) for _ in range(random.randint(1, 40)))
        elif kind == 1:
            base = random.choice(raws)
            raw = base[: random.randint(1, len(base))]
        elif kind == 2:
            base = bytearray(random.choice(raws))
            for _ in range(random.randint(1, 3)):
                base[random.randrange(len(base))] = random.getrandbits(8)
            raw = bytes(base)
        else:
            #: Groups that open far more often than they close, which is the
            #: shape that costs an unguarded reader its stack.
            opens = random.randint(1, 140)
            raw = bytes([0x0B]) * opens + bytes([0x0C]) * random.randint(0, opens)
        try:
            blank.FromString(raw)
            verdicts.append("accept")
        except Exception:
            verdicts.append("reject")
        lines.append(raw.hex())
    work.joinpath("fuzz.hex").write_text("\n".join(lines) + "\n")
    return verdicts


def main() -> int:
    random.seed(20260819)
    blank = blank_message()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        cases, oracle = write_cases(blank, work)
        verdicts = write_fuzz(blank, work, cases)

        driver = work / "differential.jai"
        driver.write_text(DRIVER)
        run = subprocess.run(
            [str(ROOT / "jaithon"), "run", str(driver), "--", str(work)],
            cwd=ROOT,
            env={"PATH": "/usr/bin:/bin", "JAITHON_PATH": "lib:packages/jaicv/src"},
            capture_output=True,
            text=True,
        )
        if run.returncode != 0:
            print(run.stdout + run.stderr, file=sys.stderr)
            return 1

        got = work.joinpath("got.txt").read_text().splitlines()
        fuzz_got = work.joinpath("fuzz_got.txt").read_text().splitlines()

    fields = sum(len(row.split()) for row in oracle)
    agree = sum(1 for want, mine in zip(oracle, got) if want == mine)
    print(f"well-formed: {len(oracle)} messages, {fields} fields, {agree} decoded identically")
    for index, (want, mine) in enumerate(zip(oracle, got)):
        if want != mine:
            print(f"  case {index}\n    protobuf {want[:200]}\n    jaicv    {mine[:200]}")

    same = sum(1 for want, mine in zip(verdicts, fuzz_got) if want == mine)
    rejected = sum(1 for verdict in verdicts if verdict == "reject")
    print(
        f"malformed:   {len(verdicts)} buffers, protobuf rejects {rejected}, "
        f"{same} verdicts agree"
    )
    return 0 if agree == len(oracle) and same == len(verdicts) else 1


if __name__ == "__main__":
    raise SystemExit(main())
