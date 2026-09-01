#: LTV1, the line table's encoding: the encoder, the decoder, and the two
#: varint primitives they share. The format itself is specified on `_uleb`
#: below, where it has been since the encoding landed.
#:
#: One file for both directions because they are one specification, and
#: because a disagreement between them is invisible -- nothing in the tree
#: compares the two, so it surfaces as a wrong source span in a diagnostic and
#: nowhere else.
#:
#: What is NOT here: the module string table, whose entries are uleb128-framed
#: too but whose codec belongs to the sections that carry it.
from jaithon.compile.emit import LineEntry
from jaithon.compile.jaic.stream import Reader

#: LTV1: the line table's encoding from `VERSION_LTV1` on.
#:
#: Must stay byte-identical to `jaiLtv1EncodeEntry` in `src/vm/bytecode/chunk.c`
#: — the C runtime reads what this writes, and nothing in the tree compares the
#: two implementations, so a disagreement surfaces as a wrong source span in a
#: diagnostic and nowhere else. `tests/vm/linetable_ltv1.c` pins the C side and
#: `scripts/linetable_golden.sh` pins the pair end to end.
#:
#: Per entry, deltas against the previous one, first against zero:
#:     uleb128  offset - prev_offset      (offsets are non-decreasing)
#:     sleb128  span - prev_span
#:     sleb128  span_end - span           (against its OWN span)
fn _uleb(out: list[int], value: int) -> void {
    var v = value
    loop {
        var byte = v & 0x7F
        v = v >> 7
        if v != 0 { byte = byte | 0x80 }
        out.push(byte)
        if v == 0 { break }
    }
}

#: Zigzag, so a small negative delta costs one byte like a small positive one.
#: Spans move backwards often — an operand is emitted after the expression that
#: owns it — so this is not a theoretical case.
fn _sleb(out: list[int], value: int) -> void {
    _uleb(out, value << 1 ^ value >> 63)
}

pub fn _ltv1_encode(lines: list[LineEntry]) -> list[int] {
    var out: list[int] = []
    var prev_offset = 0
    var prev_span = 0
    for line in lines {
        _uleb(out, line.offset - prev_offset)
        _sleb(out, line.span_start - prev_span)
        _sleb(out, line.span_end - line.span_start)
        prev_offset = line.offset
        prev_span = line.span_start
    }
    return out
}

#: Reads exactly `byte_count` bytes. Null on a malformed stream, which is a
#: refused image and therefore a recompile, never a wrong span.
pub fn _ltv1_decode(reader: Reader, byte_count: int) -> list[LineEntry]? {
    let stop = reader.position + byte_count
    var lines: list[LineEntry] = []
    var offset = 0
    var span = 0

    while reader.position < stop {
        let d_offset = _read_uleb(reader, stop)
        if d_offset == null { return null }
        let d_span = _read_sleb(reader, stop)
        if d_span == null { return null }
        let d_end = _read_sleb(reader, stop)
        if d_end == null { return null }

        offset = offset + (d_offset ?? 0)
        span = span + (d_span ?? 0)
        lines.push(LineEntry(offset, span, span + (d_end ?? 0)))
    }
    if reader.position != stop { return null }
    return lines
}

fn _read_uleb(reader: Reader, stop: int) -> int? {
    if reader.position >= stop { return null }
    return reader.uleb()
}

fn _read_sleb(reader: Reader, stop: int) -> int? {
    let zig = _read_uleb(reader, stop)
    if zig == null { return null }
    let z = zig ?? 0
    return z >> 1 ^ -(z & 1)
}
