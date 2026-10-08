#: Baseline JPEG, both directions.
#:
#: Decoding covers baseline sequential DCT with Huffman coding — the encoding
#: every camera and every web image uses — at any chroma subsampling, with
#: restart markers. Progressive JPEG is refused rather than half-decoded.
#:
#: Encoding writes baseline 4:2:0 for colour and 4:4:4 for greyscale, with the
#: standard Annex K tables scaled by a quality setting.
import std.math
from std.os import env_or
from ..core.mat import Mat
from ..core.types import CV_8U, CV_MAKETYPE

#: Read once. On, a scan goes through `scan_fast`: a lookahead Huffman decode,
#: a sparse IDCT and integer planes, with the same arithmetic in the same
#: order as the reference loop in `scan`, so the picture is the same byte for
#: byte. A stream the fast path cannot reproduce exactly -- one that reads past
#: a restart marker, an odd sampling layout -- is handed back to the reference
#: loop. Off, every scan goes through the reference loop.
let _FAST_JPEG = env_or("JAICV_FAST_JPEG", "1") != "0"

#: Zig-zag order: the sequence a JPEG stores an 8x8 block's coefficients in.
pub let ZIGZAG = [
    0,
    1,
    8,
    16,
    9,
    2,
    3,
    10,
    17,
    24,
    32,
    25,
    18,
    11,
    4,
    5,
    12,
    19,
    26,
    33,
    40,
    48,
    41,
    34,
    27,
    20,
    13,
    6,
    7,
    14,
    21,
    28,
    35,
    42,
    49,
    56,
    57,
    50,
    43,
    36,
    29,
    22,
    15,
    23,
    30,
    37,
    44,
    51,
    58,
    59,
    52,
    45,
    38,
    31,
    39,
    46,
    53,
    60,
    61,
    54,
    47,
    55,
    62,
    63,
]

#: The luminance quantisation table from the standard's Annex K.
pub let STANDARD_LUMA_QUANT = [
    16,
    11,
    10,
    16,
    24,
    40,
    51,
    61,
    12,
    12,
    14,
    19,
    26,
    58,
    60,
    55,
    14,
    13,
    16,
    24,
    40,
    57,
    69,
    56,
    14,
    17,
    22,
    29,
    51,
    87,
    80,
    62,
    18,
    22,
    37,
    56,
    68,
    109,
    103,
    77,
    24,
    35,
    55,
    64,
    81,
    104,
    113,
    92,
    49,
    64,
    78,
    87,
    103,
    121,
    120,
    101,
    72,
    92,
    95,
    98,
    112,
    100,
    103,
    99,
]

#: The chrominance quantisation table from the standard's Annex K.
pub let STANDARD_CHROMA_QUANT = [
    17,
    18,
    24,
    47,
    99,
    99,
    99,
    99,
    18,
    21,
    26,
    66,
    99,
    99,
    99,
    99,
    24,
    26,
    56,
    99,
    99,
    99,
    99,
    99,
    47,
    66,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
]

#: Whether `data` starts with a JPEG marker.
pub fn looks_like(data: bytes) -> bool {
    return data.len() >= 3 and data[0] == 0xFF and data[1] == 0xD8 and data[2] == 0xFF
}

#: One Huffman table, as a JPEG stores it: counts by length, then the values.
pub class HuffmanTable {
    #: Smallest code of each length, indexed by length minus one.
    pub let min_code: list[int]
    #: Largest code of each length, or -1 when that length is unused.
    pub let max_code: list[int]
    #: Index into `values` where each length's codes begin.
    pub let value_at: list[int]
    #: The symbols, in code order.
    pub let values: list[int]
    #: This table as `scan_fast` reads it (see `lookahead_for`), built the
    #: first time it is wanted.
    pub var lookahead: list[int]

    pub fn init(self, counts: list[int], values: list[int]) {
        self.values = values
        self.lookahead = []
        var min_code: list[int] = []
        var max_code: list[int] = []
        var value_at: list[int] = []
        var code = 0
        var index = 0
        for length in 0..16 {
            value_at.push(index)
            min_code.push(code)
            if counts[length] == 0 {
                max_code.push(-1)
            } else {
                max_code.push(code + counts[length] - 1)
                code += counts[length]
                index += counts[length]
            }
            code = code << 1
        }
        self.min_code = min_code
        self.max_code = max_code
        self.value_at = value_at
    }
}

#: Reads the entropy-coded segment one bit at a time, unstuffing the `FF 00`
#: pairs a JPEG inserts so a marker can never appear inside the data.
pub class BitReader {
    pub let data: bytes
    pub var at: int
    var bits: int
    var count: int

    pub fn init(self, data: bytes, at: int) {
        self.data = data
        self.at = at
        self.bits = 0
        self.count = 0
    }

    pub fn next_bit(self) -> int {
        if self.count == 0 {
            if self.at >= self.data.len() { return 0 }
            var byte = self.data[self.at]
            self.at += 1
            if byte == 0xFF {
                let following = if self.at < self.data.len() { self.data[self.at] } else { 0 }
                if following == 0x00 {
                    self.at += 1
                } elif following >= 0xD0 and following <= 0xD7 {
                    self.at += 1
                    byte = self.data[self.at]
                    self.at += 1
                } else {
                    return 0
                }
            }
            self.bits = byte
            self.count = 8
        }
        self.count -= 1
        return self.bits >> self.count & 1
    }

    pub fn receive(self, length: int) -> int {
        var value = 0
        for _index in 0..length { value = value << 1 | self.next_bit() }
        return value
    }

    #: Sign-extend a value the standard stores in magnitude form.
    pub fn extend(self, value: int, length: int) -> int {
        if length == 0 { return 0 }
        if value < 1 << length - 1 { return value - (1 << length) + 1 }
        return value
    }

    pub fn decode(self, table: HuffmanTable) -> int {
        var code = 0
        for length in 0..16 {
            code = code << 1 | self.next_bit()
            if table.max_code[length] >= 0 and code <= table.max_code[length] {
                let index = table.value_at[length] + code - table.min_code[length]
                if index < table.values.len() { return table.values[index] }
                return 0
            }
        }
        return 0
    }

    pub fn align(self) -> void { self.count = 0 }
}

#: One colour component of a frame.
pub class Component {
    pub let id: int
    pub var h: int
    pub var v: int
    pub let quant: int
    pub var dc_table: int
    pub var ac_table: int
    pub var predictor: int
    pub var samples: list[float]
    pub var width: int
    pub var height: int

    pub fn init(self, id: int, h: int, v: int, quant: int) {
        self.id = id
        self.h = h
        self.v = v
        self.quant = quant
        self.dc_table = 0
        self.ac_table = 0
        self.predictor = 0
        self.samples = []
        self.width = 0
        self.height = 0
    }
}

#: The separable DCT basis, `DCT_BASIS[u * 8 + x] = Cu * cos((2x+1)u*pi/16)`,
#: with the 1/sqrt(2) normalisation of the zeroth term folded in. Both
#: transforms below index this instead of calling `math.cos`, which is what
#: made them cost milliseconds a block.
let DCT_BASIS = [(if index // 8 == 0 {
    0.70710678118654752
} else {
    1.0
}) * math.cos(float(2 * (index % 8) + 1) * float(index // 8) * math.PI / 16.0) for index in 0..64]

#: `scratch` is the caller's, not a local: a fresh 64-element list an inner
#: transform allocates is one allocation a block -- thousands a frame -- and
#: the JIT cannot see the element kind of a list the body itself built, so a
#: local scratch also left the whole function running interpreted.
fn idct_8x8(block: list[float], out: list[float], scratch: list[float]) -> void {
    #: A separable float IDCT: rows first, then columns. Slower than the
    #: integer approximations a production encoder uses and exact enough that
    #: the result never differs from a reference decoder by more than a unit.
    for u in 0..8 {
        for x in 0..8 {
            var total = 0.0
            for v in 0..8 {
                let coefficient = block[v * 8 + u]
                if coefficient == 0.0 { continue }
                total += coefficient * DCT_BASIS[v * 8 + x]
            }
            scratch[x * 8 + u] = total * 0.5
        }
    }
    for y in 0..8 {
        for x in 0..8 {
            var total = 0.0
            for u in 0..8 {
                let coefficient = scratch[y * 8 + u]
                if coefficient == 0.0 { continue }
                total += coefficient * DCT_BASIS[u * 8 + x]
            }
            out[y * 8 + x] = total * 0.5
        }
    }
}

#: Decode a baseline JPEG into a BGR (or single-channel) `Mat`.
pub fn decode(data: bytes) -> Mat { return decode_with(data, _FAST_JPEG) }

#: `decode`, choosing the scan: `fast` takes `scan_fast` where it can
#: reproduce the reference exactly, and false always takes the reference loop.
#: The two give the same bytes; this is how a test holds them to it.
pub fn decode_with(data: bytes, fast: bool) -> Mat {
    if not looks_like(data) { throw ValueError("this is not a JPEG") }
    var quant: dict[int, list[int]] = {}
    #: Seeded with the tables of the standard's Annex K. A conforming JPEG
    #: sends its own DHT and overwrites all four before the scan, so this
    #: changes nothing for one; a motion-JPEG frame out of an AVI usually sends
    #: no DHT at all, because the tables are the same in every frame and the
    #: format leaves them implied. Without this such a frame decodes to
    #: whatever the missing table would have said, which is nothing.
    var dc_tables: dict[int, HuffmanTable] = {
        0: HuffmanTable(DC_LUMA_COUNTS, DC_LUMA_VALUES),
        1: HuffmanTable(DC_CHROMA_COUNTS, DC_CHROMA_VALUES),
    }
    var ac_tables: dict[int, HuffmanTable] = {
        0: HuffmanTable(AC_LUMA_COUNTS, AC_LUMA_VALUES),
        1: HuffmanTable(AC_CHROMA_COUNTS, AC_CHROMA_VALUES),
    }
    var components: list[Component] = []
    var frame_width = 0
    var frame_height = 0
    var restart_interval = 0
    var at = 2

    while at + 4 <= data.len() {
        if data[at] != 0xFF {
            at += 1
            continue
        }
        let marker = data[at + 1]
        at += 2
        if marker == 0xD8 or marker == 0x01 or marker >= 0xD0 and marker <= 0xD7 { continue }
        if marker == 0xD9 { break }
        if at + 2 > data.len() { break }
        let length = data[at] << 8 | data[at + 1]
        let body = at + 2
        let body_end = at + length

        if marker == 0xDB {
            var cursor = body
            while cursor < body_end {
                let precision = data[cursor] >> 4
                let slot = data[cursor] & 15
                cursor += 1
                var table: list[int] = []
                for index in 0..64 {
                    if precision == 0 {
                        table.push(data[cursor + index])
                    } else {
                        table.push(data[cursor + index * 2] << 8 | data[cursor + index * 2 + 1])
                    }
                }
                cursor += if precision == 0 { 64 } else { 128 }
                quant[slot] = table
            }
        } elif marker == 0xC4 {
            var cursor = body
            while cursor < body_end {
                let kind = data[cursor] >> 4
                let slot = data[cursor] & 15
                cursor += 1
                var counts: list[int] = []
                var total = 0
                for index in 0..16 {
                    counts.push(data[cursor + index])
                    total += data[cursor + index]
                }
                cursor += 16
                var values: list[int] = []
                for index in 0..total { values.push(data[cursor + index]) }
                cursor += total
                let table = HuffmanTable(counts, values)
                if kind == 0 { dc_tables[slot] = table } else { ac_tables[slot] = table }
            }
        } elif marker == 0xC0 or marker == 0xC1 {
            frame_height = data[body + 1] << 8 | data[body + 2]
            frame_width = data[body + 3] << 8 | data[body + 4]
            let count = data[body + 5]
            components = []
            for index in 0..count {
                let base = body + 6 + index * 3
                components.push(
                    Component(data[base], data[base + 1] >> 4, data[base + 1] & 15, data[base + 2])
                )
            }
        } elif marker == 0xC2 {
            throw ValueError("progressive JPEG is not supported")
        } elif marker == 0xC3 or marker >= 0xC5 and marker <= 0xCF and marker != 0xC8 {
            throw ValueError(f"JPEG frame type {marker} is not supported")
        } elif marker == 0xDD {
            restart_interval = data[body] << 8 | data[body + 1]
        } elif marker == 0xDA {
            let count = data[body]
            for index in 0..count {
                let id = data[body + 1 + index * 2]
                let tables = data[body + 2 + index * 2]
                for component in components {
                    if component.id == id {
                        component.dc_table = tables >> 4
                        component.ac_table = tables & 15
                    }
                }
            }
            return scan(
                data,
                body_end,
                components,
                quant,
                dc_tables,
                ac_tables,
                frame_width,
                frame_height,
                restart_interval,
                fast
            )
        }
        at = body_end
    }
    throw ValueError("this JPEG has no scan")
}

fn scan(
    data: bytes,
    start: int,
    components: list[Component],
    quant: dict[int, list[int]],
    dc_tables: dict[int, HuffmanTable],
    ac_tables: dict[int, HuffmanTable],
    frame_width: int,
    frame_height: int,
    restart_interval: int,
    fast: bool
) -> Mat {
    if components.len() == 0 or frame_width == 0 or frame_height == 0 {
        throw ValueError("this JPEG has no frame")
    }
    var max_h = 1
    var max_v = 1
    for component in components {
        if component.h > max_h { max_h = component.h }
        if component.v > max_v { max_v = component.v }
    }
    let mcu_width = max_h * 8
    let mcu_height = max_v * 8
    let mcus_across = (frame_width + mcu_width - 1) // mcu_width
    let mcus_down = (frame_height + mcu_height - 1) // mcu_height

    for component in components {
        component.width = mcus_across * component.h * 8
        component.height = mcus_down * component.v * 8
    }

    #: Resolved once, not once a block. The lookup and its null check ran for
    #: every 8x8 block of every component -- thousands a frame -- and the JIT
    #: cannot settle a local that is sometimes null, so doing it inside
    #: `decode_block` also left that whole function interpreted.
    var quant_tables: list[list[int]] = []
    var block_dc: list[HuffmanTable] = []
    var block_ac: list[HuffmanTable] = []
    for component in components {
        let table = quant.get(component.quant, null)
        if table is null {
            throw ValueError("this JPEG names a quantisation table it never defined")
        }
        let dc = dc_tables.get(component.dc_table, null)
        let ac = ac_tables.get(component.ac_table, null)
        if dc is null or ac is null {
            throw ValueError("this JPEG names a Huffman table it never defined")
        }
        quant_tables.push(table)
        block_dc.push(dc)
        block_ac.push(ac)
    }

    if fast {
        let quick = scan_fast(
            data,
            start,
            components,
            quant_tables,
            block_dc,
            block_ac,
            frame_width,
            frame_height,
            restart_interval,
            max_h,
            max_v,
            mcus_across,
            mcus_down
        )
        if quick is not null { return quick }
    }

    for component in components {
        component.samples = [0.0 for _index in 0..component.width * component.height]
        component.predictor = 0
    }
    let reader = BitReader(data, start)
    var block = [0.0 for _index in 0..64]
    var pixels = [0.0 for _index in 0..64]
    var dct_scratch = [0.0 for _index in 0..64]
    var since_restart = 0

    for mcu_y in 0..mcus_down {
        for mcu_x in 0..mcus_across {
            if restart_interval > 0 and since_restart == restart_interval {
                reader.align()
                #: A restart marker resets every predictor, which is what makes
                #: a corrupt stream recoverable from the next one.
                skip_restart(reader)
                for component in components { component.predictor = 0 }
                since_restart = 0
            }
            for slot in 0..components.len() {
                let component = components[slot]
                for by in 0..component.v {
                    for bx in 0..component.h {
                        decode_block(
                            reader,
                            component,
                            quant_tables[slot],
                            block_dc[slot],
                            block_ac[slot],
                            block
                        )
                        idct_8x8(block, pixels, dct_scratch)
                        let origin_x = (mcu_x * component.h + bx) * 8
                        let origin_y = (mcu_y * component.v + by) * 8
                        for y in 0..8 {
                            for x in 0..8 {
                                #: Rounded and clamped here, as a reference
                                #: decoder does before upsampling, so the
                                #: triangle filter below sees whole samples.
                                component.samples[(origin_y + y) * component.width + origin_x + x] = clamp255(
                                    pixels[y * 8 + x] +
                                        128.0
                                )
                            }
                        }
                    }
                }
            }
            since_restart += 1
        }
    }
    return assemble(components, frame_width, frame_height, max_h, max_v)
}

fn skip_restart(reader: BitReader) -> void {
    while reader.at + 1 < reader.data.len() {
        if reader.data[reader.at] == 0xFF {
            let marker = reader.data[reader.at + 1]
            if marker >= 0xD0 and marker <= 0xD7 {
                reader.at += 2
                return
            }
        }
        reader.at += 1
    }
}

#: The tables are resolved by the caller, once a scan rather than once a
#: block; see the note where they are built.
fn decode_block(
    reader: BitReader,
    component: Component,
    table: list[int],
    dc: HuffmanTable,
    ac: HuffmanTable,
    block: list[float]
) -> void {
    for index in 0..64 { block[index] = 0.0 }

    let length = reader.decode(dc)
    let difference = reader.extend(reader.receive(length), length)
    component.predictor += difference
    block[0] = float(component.predictor * table[0])

    var index = 1
    while index < 64 {
        let symbol = reader.decode(ac)
        if symbol == 0 { break }
        let run = symbol >> 4
        let size = symbol & 15
        if size == 0 {
            if run != 15 { break }
            index += 16
            continue
        }
        index += run
        if index >= 64 { break }
        let value = reader.extend(reader.receive(size), size)
        block[ZIGZAG[index]] = float(value * table[index])
        index += 1
    }
}

#: Expand a subsampled plane to full resolution with the triangle filter a
#: reference decoder uses.
#:
#: Repeating the nearest sample instead is visibly blockier and disagrees with
#: libjpeg by up to thirteen units on a 4:2:0 image; the 3:1 weighting matches
#: it to within rounding.
fn upsample_axis(
    samples: list[float],
    width: int,
    height: int,
    factor: int,
    horizontal: bool
) -> tuple[list[float], int, int] {
    if factor <= 1 { return (samples, width, height) }
    let out_width = if horizontal { width * factor } else { width }
    let out_height = if horizontal { height } else { height * factor }
    var out = [0.0 for _index in 0..out_width * out_height]
    let length = if horizontal { width } else { height }
    let lanes = if horizontal { height } else { width }
    for lane in 0..lanes {
        for index in 0..length {
            let previous = samples[pick(if index > 0 {
                index - 1
            } else {
                0
            }, lane, width, horizontal)]
            let here = samples[pick(index, lane, width, horizontal)]
            let following = samples[pick(if index + 1 < length {
                index + 1
            } else {
                length - 1
            }, lane, width, horizontal)]
            if factor == 2 {
                #: Every sample reaching here came out of `clamp255`, so both
                #: weighted sums are non-negative and truncation is the floor.
                let first = float(int((3.0 * here + previous + 2.0) * 0.25))
                let second = float(int((3.0 * here + following + 1.0) * 0.25))
                out[pick(index * 2, lane, out_width, horizontal)] = first
                out[pick(index * 2 + 1, lane, out_width, horizontal)] = second
                continue
            }
            for step in 0..factor {
                out[pick(index * factor + step, lane, out_width, horizontal)] = here
            }
        }
    }
    return (out, out_width, out_height)
}

fn pick(index: int, lane: int, width: int, horizontal: bool) -> int {
    if horizontal { return lane * width + index }
    return index * width + lane
}

#: The planes put back together as one picture.
#:
#: This was moved onto the device once, upsampling by indexing and converting
#: from YCbCr in one kernel per pixel, and it bought 1.5% of a 720p decode --
#: about five milliseconds of three hundred and thirty. The colour conversion
#: is not where the time is. The entropy decode above it is: Huffman coding is
#: serial by construction, and it is the whole cost. Anything that matters here
#: has to attack that, or hand the file to a decoder that already has.
fn assemble(components: list[Component], width: int, height: int, max_h: int, max_v: int) -> Mat {
    for component in components {
        if component.h < max_h {
            let (grown, w, h) = upsample_axis(
                component.samples,
                component.width,
                component.height,
                max_h // component.h,
                true
            )
            component.samples = grown
            component.width = w
            component.height = h
            component.h = max_h
        }
        if component.v < max_v {
            let (grown, w, h) = upsample_axis(
                component.samples,
                component.width,
                component.height,
                max_v // component.v,
                false
            )
            component.samples = grown
            component.width = w
            component.height = h
            component.v = max_v
        }
    }
    let channels = if components.len() == 1 { 1 } else { 3 }
    var values: list[float] = []
    for y in 0..height {
        for x in 0..width {
            if channels == 1 {
                values.push(clamp255(sample(components[0], x, y, max_h, max_v)))
                continue
            }
            let luma = sample(components[0], x, y, max_h, max_v)
            let cb = sample(components[1], x, y, max_h, max_v) - 128.0
            let cr = sample(components[2], x, y, max_h, max_v) - 128.0
            #: JPEG stores YCbCr; the rest of jaicv works in BGR.
            values.push(clamp255(luma + 1.772 * cb))
            values.push(clamp255(luma - 0.344136 * cb - 0.714136 * cr))
            values.push(clamp255(luma + 1.402 * cr))
        }
    }
    return Mat.from_list(height, width, CV_MAKETYPE(CV_8U, channels), values)
}

fn sample(component: Component, x: int, y: int, max_h: int, max_v: int) -> float {
    #: A subsampled plane is expanded by repeating, which is what a baseline
    #: decoder does and what OpenCV's does.
    let sx = x * component.h // max_h
    let sy = y * component.v // max_v
    let cx = min(sx, component.width - 1)
    let cy = min(sy, component.height - 1)
    return component.samples[cy * component.width + cx]
}

fn clamp255(value: float) -> float {
    #: Clamping first leaves only a non-negative argument to round, and for
    #: one of those `int()` truncates to the same whole number `math.floor`
    #: returns -- but as a single JIT instruction rather than a call, which
    #: this tier declines. Every decoded sample passes through here.
    if value <= 0.0 { return 0.0 }
    if value >= 255.0 { return 255.0 }
    return float(int(value + 0.5))
}

# ---------------------------------------------------------------------------
# The fast scan.
#
# Everything above this line is the reference decoder, and everything below
# reproduces it exactly: the same float operations in the same order, so a
# picture decoded either way is the same byte for byte. What changes is how the
# work is laid out, which is where the reference spent its time:
#
# - Bits come out of a 56-bit accumulator over the scan with its stuffing
#   already taken out, rather than one method call a bit, and a Huffman code of
#   nine bits or fewer -- nearly all of them -- is one table lookup.
# - The IDCT skips what is zero instead of testing it: a column with no
#   coefficient costs nothing, a coefficient feeds its eight outputs side by
#   side, and the zeroth basis row -- one number eight times -- is one product
#   rather than eight. A block that is only its DC term is one value written
#   sixty-four times. The reference skips the same zeros one test at a time
#   and adds the same terms in the same order, so the sums are the same sums.
# - Planes are int lists, not float ones. Every sample in them came out of a
#   clamp-and-round, so it is a whole number, and the triangle filter's float
#   arithmetic on whole numbers is exact -- `(3 * here + previous + 2) >> 2` is
#   the same number the reference gets with `int(... * 0.25)`. 4:2:0 chroma is
#   doubled down the columns inside the colour conversion, not stored twice.
# - The colour conversion is three table lookups and one subtraction a pixel:
#   each of its terms depends on one or two whole-number samples only.
# - The picture is packed straight into bytes and handed to `Mat.from_bytes`,
#   not pushed a float at a time into a list for `Mat.from_list`.
#
# Each loop sits in a function of its own, because the JIT records a fixed
# number of loop forms per function and the reference `scan` ran out of them.
# On a 640x426 COCO photograph this is about 16 ms against the reference's 60;
# see packages/jaicv/README.md for the numbers and what the rest of it is.
# ---------------------------------------------------------------------------

#: Every DCT basis value of the zeroth frequency is the same number, which is
#: what lets a DC-only block be computed once. Checked rather than assumed.
fn dc_row_is_flat() -> bool {
    for x in 1..8 {
        if DCT_BASIS[x] != DCT_BASIS[0] { return false }
    }
    return true
}

let DC_FLAT = dc_row_is_flat()

#: Where the parts of a decoding table start. The first 512 entries are the
#: nine-bit lookahead: `length << 8 | symbol` for each prefix that a code of
#: nine bits or fewer resolves, -1 where it takes a longer one. Then the table
#: itself, flat, for the longer codes: sixteen `max_code`, sixteen `min_code`,
#: sixteen `value_at`, the symbol count, and the symbols.
const LOOK_MAX = 512
const LOOK_MIN = 528
const LOOK_AT = 544
const LOOK_COUNT = 560
const LOOK_VALUES = 561

#: One Huffman table laid out for `FastScan.block`, built once a table.
#:
#: The lookahead is what `BitReader.decode`'s loop does to a prefix: it takes
#: the first length whose code is at or below that length's `max_code`, so a
#: prefix belongs to the shortest such length and lengths are filled shortest
#: first, each over only the prefixes no shorter length took. An index that
#: comes out negative is left at -1 too, and `block` hands that stream back to
#: the reference loop, which reads it the way it always has.
fn lookahead_for(table: HuffmanTable) -> list[int] {
    if table.lookahead.len() > 0 { return table.lookahead }
    let max_code = table.max_code
    let min_code = table.min_code
    let value_at = table.value_at
    let values = table.values
    let count = values.len()
    var look: list[int] = __prim__.list_filled(LOOK_VALUES + count, -2)
    var length = 0
    var code = 0
    var top = 0
    var index = 0
    var entry = 0
    var first = 0
    var last = 0
    var at = 0
    while length < 9 {
        top = max_code[length]
        if top >= 0 {
            if top > (1 << length + 1) - 1 { top = (1 << length + 1) - 1 }
            code = 0
            while code <= top {
                index = value_at[length] + code - min_code[length]
                entry = if index < 0 {
                    -1
                } elif index < count {
                    length + 1 << 8 | values[index]
                } else {
                    length + 1 << 8
                }
                first = code << 8 - length
                last = first + (1 << 8 - length)
                at = first
                while at < last {
                    if look[at] == -2 { look[at] = entry }
                    at += 1
                }
                code += 1
            }
        }
        length += 1
    }
    at = 0
    while at < LOOK_MAX {
        if look[at] == -2 { look[at] = -1 }
        at += 1
    }
    at = 0
    while at < 16 {
        look[LOOK_MAX + at] = max_code[at]
        look[LOOK_MIN + at] = min_code[at]
        look[LOOK_AT + at] = value_at[at]
        at += 1
    }
    look[LOOK_COUNT] = count
    at = 0
    while at < count {
        look[LOOK_VALUES + at] = values[at]
        at += 1
    }
    table.lookahead = look
    return look
}

#: The entropy-coded data from `start` with its stuffed `FF 00` pairs undone and
#: its restart markers taken out. `marks` gets the position in the result of
#: each restart marker, then one more entry: where the scan's data ends, which is
#: the first marker that is not a restart. The result is longer than the data by
#: eight zeros, so the accumulator can look ahead past the end without a test.
fn unstuff(data: bytes, start: int, marks: list[int]) -> list[int] {
    let end = data.len()
    var clean: list[int] = __prim__.list_filled(end - start + 8, 0)
    var count = 0
    var at = start
    var byte = 0
    var following = 0
    var going = true
    while going and at < end {
        byte = data[at]
        if byte != 0xFF {
            clean[count] = byte
            count += 1
            at += 1
        } elif at + 1 >= end {
            going = false
        } else {
            following = data[at + 1]
            if following == 0x00 {
                clean[count] = 0xFF
                count += 1
                at += 2
            } elif following >= 0xD0 and following <= 0xD7 {
                marks.push(count)
                at += 2
            } else {
                going = false
            }
        }
    }
    marks.push(count)
    return clean
}

#: The state of one fast scan, and the two things done to every block.
#:
#: A class rather than free functions because the function tier compiles a
#: function of at most eight arguments, and a block needs a dozen things: here
#: they are fields, read into locals once a call. Nothing hot calls anything
#: that might not have compiled, nothing returns from inside a loop, and no
#: local holds an instance -- each of those has left a whole body of this
#: decoder interpreted at one time or another.
class FastScan {
    #: The unstuffed data, and the accumulator over it: `nbits` bits of `acc`
    #: are unread, and `pos` is the next index of `clean` to load.
    var clean: list[int]
    var acc: int
    var nbits: int
    var pos: int
    #: Restart positions in `clean`, then where the scan's data ends.
    var marks: list[int]
    var marker: int
    pub var restart_interval: int
    var since_restart: int
    pub var mcus_across: int
    var planes: list[list[int]]
    var widths: list[int]
    var hs: list[int]
    var vs: list[int]
    var quant: list[list[int]]
    #: `lookahead_for` of each component's tables.
    var dc_tables: list[list[int]]
    var ac_tables: list[list[int]]
    var predictors: list[int]
    #: One block's coefficients at natural positions, zero between blocks.
    var coef: list[float]
    var scratch: list[float]
    #: A DC-only block's one value, from `rows` to `columns`.
    var flat: int
    var basis: list[float]
    var zigzag: list[int]

    pub fn init(
        self,
        clean: list[int],
        marks: list[int],
        planes: list[list[int]],
        widths: list[int],
        hs: list[int],
        vs: list[int],
        quant: list[list[int]],
        tables: list[list[int]]
    ) {
        self.clean = clean
        self.acc = 0
        self.nbits = 0
        self.pos = 0
        self.marks = marks
        self.marker = 0
        self.restart_interval = 0
        self.since_restart = 0
        self.mcus_across = 0
        self.planes = planes
        self.widths = widths
        self.hs = hs
        self.vs = vs
        self.quant = quant
        #: `tables` is each component's DC table then its AC table.
        var dc_tables: list[list[int]] = []
        var ac_tables: list[list[int]] = []
        for slot in 0..planes.len() {
            dc_tables.push(tables[slot * 2])
            ac_tables.push(tables[slot * 2 + 1])
        }
        self.dc_tables = dc_tables
        self.ac_tables = ac_tables
        self.predictors = __prim__.list_filled(planes.len(), 0)
        self.coef = __prim__.list_filled(64, 0.0)
        self.scratch = __prim__.list_filled(64, 0.0)
        self.flat = 0
        var basis: list[float] = __prim__.list_filled(64, 0.0)
        for index in 0..64 { basis[index] = DCT_BASIS[index] }
        self.basis = basis
        var zigzag: list[int] = __prim__.list_filled(64, 0)
        for index in 0..64 { zigzag[index] = ZIGZAG[index] }
        self.zigzag = zigzag
    }

    #: One MCU row, in the reference's order. False when the stream reads past
    #: the end of a restart interval, or uses a table the lookahead could not
    #: express, either of which the reference reads differently.
    pub fn row(self, mcu_y: int) -> bool {
        let count = self.planes.len()
        let interval = self.restart_interval
        let across = self.mcus_across
        let marks = self.marks
        let predictors = self.predictors
        let planes = self.planes
        let widths = self.widths
        let hs = self.hs
        let vs = self.vs
        var ok = true
        var mcu_x = 0
        var slot = 0
        var by = 0
        var bx = 0
        var top = 0
        var packed = 0
        var live = 0
        var edge = 0
        while ok and mcu_x < across {
            if interval > 0 and self.since_restart == interval {
                #: The reference aligns and scans forward from where it is to
                #: the next marker. That lands on this interval's own marker
                #: exactly when nothing past the interval was read, and only
                #: then does reading on from the marker here agree with it.
                if self.marker >= marks.len() - 1 {
                    ok = false
                } else {
                    edge = marks[self.marker]
                    if self.pos * 8 - self.nbits > edge * 8 { ok = false }
                    self.acc = 0
                    self.nbits = 0
                    self.pos = edge
                    self.marker += 1
                    slot = 0
                    while slot < count {
                        predictors[slot] = 0
                        slot += 1
                    }
                    self.since_restart = 0
                }
            }
            slot = 0
            while slot < count {
                let plane = planes[slot]
                let width = widths[slot]
                let h = hs[slot]
                let v = vs[slot]
                by = 0
                while by < v {
                    top = (mcu_y * v + by) * 8 * width + mcu_x * h * 8
                    bx = 0
                    while bx < h {
                        packed = self.block(slot)
                        if packed < 0 {
                            ok = false
                        } else {
                            live = self.rows(packed)
                            self.columns(live, plane, width, top + bx * 8)
                        }
                        bx += 1
                    }
                    by += 1
                }
                slot += 1
            }
            self.since_restart += 1
            mcu_x += 1
        }
        return ok
    }

    #: Whether the scan, now finished, stayed inside the data it had.
    pub fn finished(self) -> bool {
        let last = self.marks.len() - 1
        let at = if self.marker < last { self.marker } else { last }
        return self.pos * 8 - self.nbits <= self.marks[at] * 8
    }

    #: One block's coefficients into `coef`. Returns how many it set, the DC
    #: always among them, in the low byte and which columns they fell in
    #: above it -- or -1 for a stream the reference must read:
    #: a DC length no conforming stream has, or a code whose table index comes
    #: out negative.
    fn block(self, slot: int) -> int {
        let clean = self.clean
        let limit = clean.len()
        let coef = self.coef
        let zigzag = self.zigzag
        let table = self.quant[slot]
        let dc_table = self.dc_tables[slot]
        let ac_table = self.ac_tables[slot]
        var acc = self.acc
        var nbits = self.nbits
        var pos = self.pos
        var entry = 0
        var code = 0
        var length = 0
        var top = 0
        var index = 0
        var raw = 0
        var run = 0
        var size = 0
        var symbol = 0
        var count = 1
        var at = 0
        var going = true

        if nbits < 32 {
            acc = acc & (1 << nbits) - 1
            while nbits <= 48 {
                acc = acc << 8 | (if pos < limit { clean[pos] } else { 0 })
                pos += 1
                nbits += 8
            }
        }
        entry = dc_table[acc >> nbits - 9 & 511]
        if entry == -1 {
            #: `BitReader.decode`, written out for a code longer than nine
            #: bits. A call would do, except that a call to a function that
            #: has not compiled stops the compile of everything after it, and
            #: a stream whose codes are all short never calls it.
            entry = 16 << 8
            code = 0
            length = 0
            while length < 16 {
                code = code << 1 | acc >> nbits - 1 - length & 1
                top = dc_table[LOOK_MAX + length]
                if top >= 0 and code <= top {
                    index = dc_table[LOOK_AT + length] + code - dc_table[LOOK_MIN + length]
                    entry = if index < 0 {
                        -1
                    } elif index < dc_table[LOOK_COUNT] {
                        length + 1 << 8 | dc_table[LOOK_VALUES + index]
                    } else {
                        length + 1 << 8
                    }
                    length = 16
                }
                length += 1
            }
            if entry < 0 { return -1 }
        }
        nbits -= entry >> 8
        length = entry & 255
        if length > 16 { return -1 }
        var difference = 0
        if length > 0 {
            raw = acc >> nbits - length & (1 << length) - 1
            nbits -= length
            difference = if raw < 1 << length - 1 { raw - (1 << length) + 1 } else { raw }
        }
        let predictor = self.predictors[slot] + difference
        self.predictors[slot] = predictor
        coef[0] = float(predictor * table[0])
        var columns = 1

        index = 1
        while going and index < 64 {
            if nbits < 32 {
                acc = acc & (1 << nbits) - 1
                while nbits <= 48 {
                    acc = acc << 8 | (if pos < limit { clean[pos] } else { 0 })
                    pos += 1
                    nbits += 8
                }
            }
            entry = ac_table[acc >> nbits - 9 & 511]
            if entry == -1 {
                entry = 16 << 8
                code = 0
                length = 0
                while length < 16 {
                    code = code << 1 | acc >> nbits - 1 - length & 1
                    top = ac_table[LOOK_MAX + length]
                    if top >= 0 and code <= top {
                        at = ac_table[LOOK_AT + length] + code - ac_table[LOOK_MIN + length]
                        entry = if at < 0 {
                            -1
                        } elif at < ac_table[LOOK_COUNT] {
                            length + 1 << 8 | ac_table[LOOK_VALUES + at]
                        } else {
                            length + 1 << 8
                        }
                        length = 16
                    }
                    length += 1
                }
            }
            symbol = entry & 255
            run = symbol >> 4
            size = symbol & 15
            if entry < 0 {
                count = -1
                going = false
            } elif symbol == 0 {
                nbits -= entry >> 8
                going = false
            } elif size == 0 {
                nbits -= entry >> 8
                if run != 15 { going = false } else { index += 16 }
            } else {
                nbits -= entry >> 8
                index += run
                if index >= 64 {
                    going = false
                } else {
                    raw = acc >> nbits - size & (1 << size) - 1
                    nbits -= size
                    if raw < 1 << size - 1 { raw = raw - (1 << size) + 1 }
                    at = zigzag[index]
                    coef[at] = float(raw * table[index])
                    columns = columns | 1 << (at & 7)
                    count += 1
                    index += 1
                }
            }
        }
        if count < 0 { return -1 }
        self.acc = acc
        self.nbits = nbits
        self.pos = pos
        return columns << 8 | count
    }

    #: The first half of `idct_8x8`, for the block in `coef` that `block`
    #: described in `packed`: column u of the coefficients becomes column u of
    #: `scratch`. Returns which columns came out not all zero, and leaves
    #: `coef` all zeros again for the next block.
    #:
    #: A block that is only its DC term is every output the same sum, so the
    #: same number: it is worked out once here, left in `flat`, and -1 tells
    #: `columns` to fill the block with it.
    #:
    #: The reference adds `coefficient * basis` over every coefficient that is
    #: not zero, in order, then halves; this does the same over the same
    #: coefficients in the same order, with the eight outputs a coefficient
    #: feeds summed side by side instead of one after another.
    #:
    #: `rows` and `columns` are separate, and both called for every block,
    #: because the function tier holds a body to 20,000 instructions and will
    #: not plan a call to a method that has never run.
    fn rows(self, packed: int) -> int {
        let coef = self.coef
        let basis = self.basis
        let scratch = self.scratch
        let count = packed & 255
        var coefficient = 0.0
        var total = 0.0
        var k = 0
        var u = 0
        var v = 0
        var at = 0

        let b0 = basis[0]
        if count == 1 {
            coefficient = coef[0]
            coef[0] = 0.0
            total = 0.0
            if coefficient != 0.0 { total += coefficient * b0 }
            coefficient = total * 0.5
            total = 0.0
            if coefficient != 0.0 { total += coefficient * b0 }
            let level = total * 0.5 + 128.0
            self.flat = if level <= 0.0 { 0 } elif level >= 255.0 { 255 } else { int(level + 0.5) }
            return -1
        }

        let columns = packed >> 8
        var t0 = 0.0
        var t1 = 0.0
        var t2 = 0.0
        var t3 = 0.0
        var t4 = 0.0
        var t5 = 0.0
        var t6 = 0.0
        var t7 = 0.0
        var any = false
        var live = 0
        while u < 8 {
            if columns >> u & 1 != 0 {
                #: The v = 0 term comes first and is the same for all eight
                #: outputs, because that basis row is flat: one product, and
                #: `0.0 + p` is `p`.
                coefficient = coef[u]
                coef[u] = 0.0
                any = coefficient != 0.0
                total = if any { coefficient * b0 } else { 0.0 }
                t0 = total
                t1 = total
                t2 = total
                t3 = total
                t4 = total
                t5 = total
                t6 = total
                t7 = total
                v = 1
                while v < 8 {
                    at = v * 8
                    coefficient = coef[at + u]
                    if coefficient != 0.0 {
                        coef[at + u] = 0.0
                        t0 += coefficient * basis[at]
                        t1 += coefficient * basis[at + 1]
                        t2 += coefficient * basis[at + 2]
                        t3 += coefficient * basis[at + 3]
                        t4 += coefficient * basis[at + 4]
                        t5 += coefficient * basis[at + 5]
                        t6 += coefficient * basis[at + 6]
                        t7 += coefficient * basis[at + 7]
                        any = true
                    }
                    v += 1
                }
                if any {
                    live = live | 1 << u
                    scratch[u] = t0 * 0.5
                    scratch[8 + u] = t1 * 0.5
                    scratch[16 + u] = t2 * 0.5
                    scratch[24 + u] = t3 * 0.5
                    scratch[32 + u] = t4 * 0.5
                    scratch[40 + u] = t5 * 0.5
                    scratch[48 + u] = t6 * 0.5
                    scratch[56 + u] = t7 * 0.5
                }
            }
            u += 1
        }
        return live
    }

    #: The second half: the columns of `scratch` named by `live`, through the
    #: transform again and the clamp after it, into `plane` at `origin` with
    #: rows `width` apart. A `live` of -1 is a DC-only block, every sample of
    #: it `flat`.
    fn columns(self, live: int, plane: list[int], width: int, origin: int) -> void {
        var y = 0
        var x = 0
        var at = 0
        if live < 0 {
            let pixel = self.flat
            while y < 8 {
                at = origin + y * width
                x = 0
                while x < 8 {
                    plane[at + x] = pixel
                    x += 1
                }
                y += 1
            }
            return
        }
        let basis = self.basis
        let scratch = self.scratch
        var coefficient = 0.0
        var level = 0.0
        var u = 0
        var t0 = 0.0
        var t1 = 0.0
        var t2 = 0.0
        var t3 = 0.0
        var t4 = 0.0
        var t5 = 0.0
        var t6 = 0.0
        var t7 = 0.0
        let b0 = basis[0]
        let first = live & 1
        while y < 8 {
            coefficient = if first != 0 { scratch[y * 8] } else { 0.0 }
            level = if coefficient != 0.0 { coefficient * b0 } else { 0.0 }
            t0 = level
            t1 = level
            t2 = level
            t3 = level
            t4 = level
            t5 = level
            t6 = level
            t7 = level
            u = 1
            while u < 8 {
                if live >> u & 1 != 0 {
                    coefficient = scratch[y * 8 + u]
                    if coefficient != 0.0 {
                        at = u * 8
                        t0 += coefficient * basis[at]
                        t1 += coefficient * basis[at + 1]
                        t2 += coefficient * basis[at + 2]
                        t3 += coefficient * basis[at + 3]
                        t4 += coefficient * basis[at + 4]
                        t5 += coefficient * basis[at + 5]
                        t6 += coefficient * basis[at + 6]
                        t7 += coefficient * basis[at + 7]
                    }
                }
                u += 1
            }
            at = origin + y * width
            level = t0 * 0.5 + 128.0
            plane[at] = if level <= 0.0 { 0 } elif level >= 255.0 { 255 } else { int(level + 0.5) }
            level = t1 * 0.5 + 128.0
            plane[at + 1] = if level <= 0.0 {
                0
            } elif level >= 255.0 {
                255
            } else {
                int(level + 0.5)
            }
            level = t2 * 0.5 + 128.0
            plane[at + 2] = if level <= 0.0 {
                0
            } elif level >= 255.0 {
                255
            } else {
                int(level + 0.5)
            }
            level = t3 * 0.5 + 128.0
            plane[at + 3] = if level <= 0.0 {
                0
            } elif level >= 255.0 {
                255
            } else {
                int(level + 0.5)
            }
            level = t4 * 0.5 + 128.0
            plane[at + 4] = if level <= 0.0 {
                0
            } elif level >= 255.0 {
                255
            } else {
                int(level + 0.5)
            }
            level = t5 * 0.5 + 128.0
            plane[at + 5] = if level <= 0.0 {
                0
            } elif level >= 255.0 {
                255
            } else {
                int(level + 0.5)
            }
            level = t6 * 0.5 + 128.0
            plane[at + 6] = if level <= 0.0 {
                0
            } elif level >= 255.0 {
                255
            } else {
                int(level + 0.5)
            }
            level = t7 * 0.5 + 128.0
            plane[at + 7] = if level <= 0.0 {
                0
            } elif level >= 255.0 {
                255
            } else {
                int(level + 0.5)
            }
            y += 1
        }
    }
}

#: `upsample_axis` along rows, on whole-number samples, into `out`, which is
#: `factor` times as wide.
#:
#: These run once a plane, so they compile on stack replacement, and they are
#: written for it: every local declared before the loops (one declared in a
#: loop body is null on the way in, which the form will not take), and the
#: output passed in rather than made here (a list the body built itself has
#: no element kind the tier can see).
fn widen_rows(plane: list[int], width: int, height: int, factor: int, out: list[int]) -> void {
    let out_width = width * factor
    let last = width - 1
    var y = 0
    var index = 0
    var row = 0
    var target = 0
    var here = 0
    var previous = 0
    var following = 0
    var step = 0
    while y < height {
        row = y * width
        target = y * out_width
        if factor == 2 {
            previous = plane[row]
            index = 0
            while index < width {
                here = plane[row + index]
                following = if index < last { plane[row + index + 1] } else { here }
                out[target] = 3 * here + previous + 2 >> 2
                out[target + 1] = 3 * here + following + 1 >> 2
                previous = here
                target += 2
                index += 1
            }
        } else {
            index = 0
            while index < width {
                here = plane[row + index]
                step = 0
                while step < factor {
                    out[target] = here
                    target += 1
                    step += 1
                }
                index += 1
            }
        }
        y += 1
    }
}

#: `upsample_axis` down columns, a row at a time, into `out`, which is
#: `factor` times as tall.
fn widen_columns(plane: list[int], width: int, height: int, factor: int, out: list[int]) -> void {
    let last = height - 1
    var y = 0
    var x = 0
    var here_row = 0
    var previous_row = 0
    var following_row = 0
    var first = 0
    var second = 0
    var here = 0
    var step = 0
    while y < height {
        here_row = y * width
        previous_row = if y > 0 { here_row - width } else { here_row }
        following_row = if y < last { here_row + width } else { here_row }
        if factor == 2 {
            first = y * 2 * width
            second = first + width
            x = 0
            while x < width {
                here = plane[here_row + x]
                out[first + x] = 3 * here + plane[previous_row + x] + 2 >> 2
                out[second + x] = 3 * here + plane[following_row + x] + 1 >> 2
                x += 1
            }
        } else {
            step = 0
            while step < factor {
                first = (y * factor + step) * width
                x = 0
                while x < width {
                    out[first + x] = plane[here_row + x]
                    x += 1
                }
                step += 1
            }
        }
        y += 1
    }
}

#: `assemble`'s colour conversion as tables, built the first time a colour
#: JPEG takes the fast path. Every term of it is a function of one or two
#: whole-number samples, so it can be worked out for all of them once:
#:
#: - `_BLUE_RED[luma << 8 | cb]` is `clamp255(luma + 1.772 * cb)`, and
#:   `_BLUE_RED[65536 + (luma << 8 | cr)]` is `clamp255(luma + 1.402 * cr)`;
#: - `_GREEN[luma << 8 | cb]` is `luma - 0.344136 * cb`, and
#:   `_GREEN[65536 + cr]` is `0.714136 * cr`, so green is one subtraction and
#:   a clamp -- the same two operations, in the same order, as `assemble`.
#:
#: (`cb` and `cr` here are the samples less 128.0, exactly as there.)
var _BLUE_RED: list[int] = []
var _GREEN: list[float] = []

fn colour_tables() -> void {
    if _BLUE_RED.len() > 0 { return }
    var blue_red: list[int] = __prim__.list_filled(131072, 0)
    var green: list[float] = __prim__.list_filled(65536 + 256, 0.0)
    var luma = 0
    var chroma = 0
    var y = 0.0
    var c = 0.0
    var level = 0.0
    while luma < 256 {
        y = float(luma)
        chroma = 0
        while chroma < 256 {
            c = float(chroma) - 128.0
            level = y + 1.772 * c
            blue_red[luma << 8 | chroma] = if level <= 0.0 {
                0
            } elif level >= 255.0 {
                255
            } else {
                int(level + 0.5)
            }
            level = y + 1.402 * c
            blue_red[65536 + (luma << 8 | chroma)] = if level <= 0.0 {
                0
            } elif level >= 255.0 {
                255
            } else {
                int(level + 0.5)
            }
            green[luma << 8 | chroma] = y - 0.344136 * c
            chroma += 1
        }
        luma += 1
    }
    chroma = 0
    while chroma < 256 {
        green[65536 + chroma] = 0.714136 * (float(chroma) - 128.0)
        chroma += 1
    }
    _BLUE_RED = blue_red
    _GREEN = green
}

#: YCbCr planes, all `stride` wide, to packed BGR, by `colour_tables`.
fn ycc_to_bgr(
    luma_plane: list[int],
    cb_plane: list[int],
    cr_plane: list[int],
    stride: int,
    width: int,
    height: int,
    out: list[int]
) -> void {
    let blue_red = _BLUE_RED
    let green = _GREEN
    var o = 0
    var y = 0
    var x = 0
    var at = 0
    var luma = 0
    var cb = 0
    var cr = 0
    var level = 0.0
    while y < height {
        at = y * stride
        x = 0
        while x < width {
            luma = luma_plane[at] << 8
            cb = cb_plane[at]
            cr = cr_plane[at]
            out[o] = blue_red[luma | cb]
            level = green[luma | cb] - green[65536 + cr]
            out[o + 1] = if level <= 0.0 { 0 } elif level >= 255.0 { 255 } else { int(level + 0.5) }
            out[o + 2] = blue_red[65536 + (luma | cr)]
            o += 3
            at += 1
            x += 1
        }
        y += 1
    }
}

#: `ycc_to_bgr` for chroma planes still at half height, upsampled down the
#: columns as it goes: `widen_columns`' two rows from each chroma row, worked
#: out where they are wanted instead of stored. Output row y reads chroma row
#: y / 2 and the one above it (y even) or below it (y odd), as `widen_columns`
#: does, with its rounding.
fn ycc_to_bgr_tall(
    luma_plane: list[int],
    cb_plane: list[int],
    cr_plane: list[int],
    stride: int,
    width: int,
    height: int,
    chroma_rows: int,
    out: list[int]
) -> void {
    let blue_red = _BLUE_RED
    let green = _GREEN
    let last = chroma_rows - 1
    var o = 0
    var y = 0
    var x = 0
    var at = 0
    var half = 0
    var here = 0
    var other = 0
    var bias = 0
    var luma = 0
    var cb = 0
    var cr = 0
    var level = 0.0
    while y < height {
        half = y >> 1
        here = half * stride
        if y & 1 == 0 {
            other = if half > 0 { here - stride } else { here }
            bias = 2
        } else {
            other = if half < last { here + stride } else { here }
            bias = 1
        }
        at = y * stride
        x = 0
        while x < width {
            luma = luma_plane[at + x] << 8
            cb = 3 * cb_plane[here + x] + cb_plane[other + x] + bias >> 2
            cr = 3 * cr_plane[here + x] + cr_plane[other + x] + bias >> 2
            out[o] = blue_red[luma | cb]
            level = green[luma | cb] - green[65536 + cr]
            out[o + 1] = if level <= 0.0 { 0 } elif level >= 255.0 { 255 } else { int(level + 0.5) }
            out[o + 2] = blue_red[65536 + (luma | cr)]
            o += 3
            x += 1
        }
        y += 1
    }
}

#: One plane's top-left `width` by `height`, packed.
fn crop_plane(plane: list[int], stride: int, width: int, height: int, out: list[int]) -> void {
    var o = 0
    var y = 0
    var x = 0
    var row = 0
    while y < height {
        row = y * stride
        x = 0
        while x < width {
            out[o] = plane[row + x]
            o += 1
            x += 1
        }
        y += 1
    }
}

#: The working lists of the last fast decode, one a role: three planes, three
#: widened across, three widened down, and the packed picture.
#:
#: A 720p frame's lists come to about forty megabytes, and allocating them
#: fresh -- the allocator handing over pages and zeroing every one -- was
#: three of the thirty-one milliseconds a frame took (half of one of a COCO
#: photograph's seventeen, whose sizes vary more). Nothing needs them
#: zeroed: every element of a plane is written by exactly one block, every
#: element of a widened plane and of the picture by the loop that fills it,
#: before anything reads it. So the last decode's are kept and handed to the next.
#: That holds on to them between decodes; the pictures of a video, or of a
#: dataset, are mostly one size, which is the case this is for. A list over
#: `SPARE_LIMIT` elements -- a 1080p picture's is 6.2 million -- is not kept,
#: so one large photograph does not leave hundreds of megabytes behind it.
var _SPARE: list[list[int]] = [[], [], [], [], [], [], [], [], [], []]

const SPARE_OUT = 9
const SPARE_LIMIT = 8388608

#: A list of at least `count` elements for `role`, its contents whatever the
#: last decode left: the one kept for that role when it is big enough and not
#: much bigger, a new one (kept in its place) when not. The packed picture is
#: converted to bytes whole, so its slack is bounded to a quarter.
fn spare(role: int, count: int) -> list[int] {
    let kept = _SPARE[role]
    if kept.len() >= count and kept.len() <= count + count // 4 { return kept }
    var fresh: list[int] = __prim__.list_filled(count, 0)
    if count <= SPARE_LIMIT { _SPARE[role] = fresh }
    return fresh
}

#: Let go of the working lists `spare` keeps between decodes: about forty
#: megabytes after a 720p frame. Nothing decoded refers to them, so the next
#: decode only allocates afresh. `AviReader.release` calls this.
pub fn release_buffers() -> void {
    for role in 0.._SPARE.len() { _SPARE[role] = [] }
}

#: The scan, the fast way, or null to send it through the reference loop.
fn scan_fast(
    data: bytes,
    start: int,
    components: list[Component],
    quant_tables: list[list[int]],
    block_dc: list[HuffmanTable],
    block_ac: list[HuffmanTable],
    frame_width: int,
    frame_height: int,
    restart_interval: int,
    max_h: int,
    max_v: int,
    mcus_across: int,
    mcus_down: int
) -> Mat? {
    #: A scan header whose length runs past the end of the file leaves
    #: nothing to read. The reference reads that as zeros; `unstuff` cannot
    #: size a buffer for it.
    if start >= data.len() { return null }
    let count = components.len()
    if count != 1 and count != 3 { return null }
    #: `rows` and `columns` take the zeroth basis row as one number. It is,
    #: on every machine this has run on; if it ever were not, the reference
    #: decodes instead.
    if not DC_FLAT { return null }
    var planes: list[list[int]] = []
    var widths: list[int] = []
    var heights: list[int] = []
    var hs: list[int] = []
    var vs: list[int] = []
    var tables: list[list[int]] = []
    var usable = true
    for component in components {
        if component.h < 1 or component.h > 4 or component.v < 1 or component.v > 4 {
            usable = false
        } elif max_h % component.h != 0 or max_v % component.v != 0 {
            usable = false
        }
    }
    if not usable { return null }
    for slot in 0..count {
        let component = components[slot]
        planes.push(spare(slot, component.width * component.height))
        widths.push(component.width)
        heights.push(component.height)
        hs.push(component.h)
        vs.push(component.v)
        tables.push(lookahead_for(block_dc[slot]))
        tables.push(lookahead_for(block_ac[slot]))
    }

    var marks: list[int] = []
    let clean = unstuff(data, start, marks)
    let fast = FastScan(clean, marks, planes, widths, hs, vs, quant_tables, tables)
    fast.restart_interval = restart_interval
    fast.mcus_across = mcus_across
    var decoded = true
    var mcu_y = 0
    while decoded and mcu_y < mcus_down {
        decoded = fast.row(mcu_y)
        mcu_y += 1
    }
    if not decoded or not fast.finished() { return null }

    #: `assemble`'s upsampling: across first, then down, each by the whole
    #: factor that brings the plane to the frame's sampling. Chroma at half
    #: height under full-size luma -- 4:2:0, what cameras and video write --
    #: is left at half height and doubled inside the colour conversion.
    let tall = count == 3 and
        hs[0] == max_h and
        vs[0] == max_v and
        vs[1] == vs[2] and
        max_v // vs[1] == 2
    for slot in 0..count {
        if hs[slot] < max_h {
            let factor = max_h // hs[slot]
            let wider = spare(3 + slot, widths[slot] * factor * heights[slot])
            widen_rows(planes[slot], widths[slot], heights[slot], factor, wider)
            planes[slot] = wider
            widths[slot] = widths[slot] * factor
        }
        if vs[slot] < max_v and not tall {
            let factor = max_v // vs[slot]
            let taller = spare(6 + slot, widths[slot] * heights[slot] * factor)
            widen_columns(planes[slot], widths[slot], heights[slot], factor, taller)
            planes[slot] = taller
            heights[slot] = heights[slot] * factor
        }
    }

    let picture = frame_width * frame_height * count
    let out = spare(SPARE_OUT, picture)
    if count == 1 {
        crop_plane(planes[0], widths[0], frame_width, frame_height, out)
    } else {
        colour_tables()
        if tall {
            ycc_to_bgr_tall(
                planes[0],
                planes[1],
                planes[2],
                widths[0],
                frame_width,
                frame_height,
                heights[1],
                out
            )
        } else {
            ycc_to_bgr(planes[0], planes[1], planes[2], widths[0], frame_width, frame_height, out)
        }
    }
    #: `bytes_new` rather than `bytes`: the conversion builtin walks the list
    #: as a generic iterable, a third of the whole decode on its own.
    return Mat.from_bytes(
        frame_height,
        frame_width,
        CV_MAKETYPE(CV_8U, count),
        __prim__.bytes_new(out)
    )
}

fn scaled_table(base: list[int], quality: int) -> list[int] {
    #: The standard's quality scaling: below fifty the tables grow, above it
    #: they shrink, and every entry stays in 1..255.
    let clamped = max(1, min(100, quality))
    let factor = if clamped < 50 { 5000 // clamped } else { 200 - clamped * 2 }
    var out: list[int] = []
    for value in base {
        var scaled = (value * factor + 50) // 100
        if scaled < 1 { scaled = 1 }
        if scaled > 255 { scaled = 255 }
        out.push(scaled)
    }
    return out
}

#: The luminance DC code lengths and values from Annex K.
let DC_LUMA_COUNTS = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
let DC_LUMA_VALUES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
let DC_CHROMA_COUNTS = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
let DC_CHROMA_VALUES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
let AC_LUMA_COUNTS = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125]
let AC_LUMA_VALUES = [
    1,
    2,
    3,
    0,
    4,
    17,
    5,
    18,
    33,
    49,
    65,
    6,
    19,
    81,
    97,
    7,
    34,
    113,
    20,
    50,
    129,
    145,
    161,
    8,
    35,
    66,
    177,
    193,
    21,
    82,
    209,
    240,
    36,
    51,
    98,
    114,
    130,
    9,
    10,
    22,
    23,
    24,
    25,
    26,
    37,
    38,
    39,
    40,
    41,
    42,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    83,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
    99,
    100,
    101,
    102,
    103,
    104,
    105,
    106,
    115,
    116,
    117,
    118,
    119,
    120,
    121,
    122,
    131,
    132,
    133,
    134,
    135,
    136,
    137,
    138,
    146,
    147,
    148,
    149,
    150,
    151,
    152,
    153,
    154,
    162,
    163,
    164,
    165,
    166,
    167,
    168,
    169,
    170,
    178,
    179,
    180,
    181,
    182,
    183,
    184,
    185,
    186,
    194,
    195,
    196,
    197,
    198,
    199,
    200,
    201,
    202,
    210,
    211,
    212,
    213,
    214,
    215,
    216,
    217,
    218,
    225,
    226,
    227,
    228,
    229,
    230,
    231,
    232,
    233,
    234,
    241,
    242,
    243,
    244,
    245,
    246,
    247,
    248,
    249,
    250,
]
let AC_CHROMA_COUNTS = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 119]
let AC_CHROMA_VALUES = [
    0,
    1,
    2,
    3,
    17,
    4,
    5,
    33,
    49,
    6,
    18,
    65,
    81,
    7,
    97,
    113,
    19,
    34,
    50,
    129,
    8,
    20,
    66,
    145,
    161,
    177,
    193,
    9,
    35,
    51,
    82,
    240,
    21,
    98,
    114,
    209,
    10,
    22,
    36,
    52,
    225,
    37,
    241,
    23,
    24,
    25,
    26,
    38,
    39,
    40,
    41,
    42,
    53,
    54,
    55,
    56,
    57,
    58,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    83,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
    99,
    100,
    101,
    102,
    103,
    104,
    105,
    106,
    115,
    116,
    117,
    118,
    119,
    120,
    121,
    122,
    130,
    131,
    132,
    133,
    134,
    135,
    136,
    137,
    138,
    146,
    147,
    148,
    149,
    150,
    151,
    152,
    153,
    154,
    162,
    163,
    164,
    165,
    166,
    167,
    168,
    169,
    170,
    178,
    179,
    180,
    181,
    182,
    183,
    184,
    185,
    186,
    194,
    195,
    196,
    197,
    198,
    199,
    200,
    201,
    202,
    210,
    211,
    212,
    213,
    214,
    215,
    216,
    217,
    218,
    226,
    227,
    228,
    229,
    230,
    231,
    232,
    233,
    234,
    242,
    243,
    244,
    245,
    246,
    247,
    248,
    249,
    250,
]

#: Code and length for every symbol a canonical Huffman table defines.
#: The canonical Huffman codes as two flat lists indexed by symbol, rather than
#: a dict of tuples. A symbol is a byte, so 256 entries cover every table the
#: standard defines, and a lookup becomes one indexed read instead of a hash
#: probe and a tuple unpack -- which the coefficient loop below did for every
#: symbol it wrote. A length of zero means the table does not define that
#: symbol; `code` is meaningless there.
fn build_codes(counts: list[int], values: list[int]) -> tuple[list[int], list[int]] {
    var code_of = [0 for _index in 0..256]
    var length_of = [0 for _index in 0..256]
    var code = 0
    var index = 0
    for length in 0..16 {
        for _step in 0..counts[length] {
            code_of[values[index]] = code
            length_of[values[index]] = length + 1
            code += 1
            index += 1
        }
        code = code << 1
    }
    return (code_of, length_of)
}

class BitWriter {
    pub var out: list[int]
    var bits: int
    var count: int

    pub fn init(self) {
        self.out = []
        self.bits = 0
        self.count = 0
    }

    pub fn write(self, code: int, length: int) -> void {
        for index in 0..length {
            let bit = code >> length - 1 - index & 1
            self.bits = self.bits << 1 | bit
            self.count += 1
            if self.count == 8 {
                self.out.push(self.bits & 255)
                #: A literal FF has to be followed by a zero so it can never be
                #: mistaken for a marker.
                if self.bits & 255 == 0xFF { self.out.push(0) }
                self.bits = 0
                self.count = 0
            }
        }
    }

    pub fn flush(self) -> void {
        while self.count != 0 { self.write(1, 1) }
    }
}

fn magnitude_of(value: int) -> int {
    var size = 0
    var magnitude = abs(value)
    while magnitude > 0 {
        size += 1
        magnitude = magnitude >> 1
    }
    return size
}

#: `scratch` is the caller's, not a local: a fresh 64-element list an inner
#: transform allocates is one allocation a block -- thousands a frame -- and
#: the JIT cannot see the element kind of a list the body itself built, so a
#: local scratch also left the whole function running interpreted.
fn fdct_8x8(block: list[float], out: list[float], scratch: list[float]) -> void {
    #: Read once into a local. A global is a module lookup on every read, and
    #: the inner loops below read this 1024 times a block.
    let basis = DCT_BASIS
    #: Separable, matching `idct_8x8`: a column pass into a scratch block and
    #: then a row pass. Doing it as the literal double sum of the standard's
    #: definition costs 4096 multiply-adds and 8192 cosines a block; this
    #: costs 1024 multiply-adds and no cosines.
    for v in 0..8 {
        let row = v * 8
        for x in 0..8 {
            var total = 0.0
            for y in 0..8 { total += block[y * 8 + x] * basis[row + y] }
            scratch[row + x] = total
        }
    }
    for v in 0..8 {
        let row = v * 8
        for u in 0..8 {
            var total = 0.0
            let urow = u * 8
            for x in 0..8 { total += scratch[row + x] * basis[urow + x] }
            out[row + u] = total * 0.25
        }
    }
}

fn segment(out: list[int], marker: int, payload: list[int]) -> void {
    out.push(0xFF)
    out.push(marker)
    let length = payload.len() + 2
    out.push(length >> 8 & 255)
    out.push(length & 255)
    for byte in payload { out.push(byte) }
}

#: Encode a `Mat` as a baseline JPEG.
#:
#: Greyscale is written as a single component, colour as YCbCr at 4:4:4 —
#: subsampling would trade fidelity for size, and a codec used inside a vision
#: pipeline should not do that without being asked.
pub fn encode(image: Mat, quality: int = 95) -> bytes {
    let channels = image.channels()
    if channels != 1 and channels != 3 {
        throw ValueError(f"JPEG cannot hold {channels} channels")
    }
    let luma_quant = scaled_table(STANDARD_LUMA_QUANT, quality)
    let chroma_quant = scaled_table(STANDARD_CHROMA_QUANT, quality)
    #: The divisors in zigzag order, already inverted and already floats. The
    #: quantise loop below runs 64 times a block and thousands of times an
    #: image; leaving it to divide meant a division and an int-to-float
    #: conversion at every one of them, which measured as the single most
    #: expensive phase of encoding.
    let luma_recip = [1.0 / float(luma_quant[ZIGZAG[i]]) for i in 0..64]
    let chroma_recip = [1.0 / float(chroma_quant[ZIGZAG[i]]) for i in 0..64]
    let values = image.values()
    let width = image.cols
    let height = image.rows

    var out: list[int] = [0xFF, 0xD8]
    var quant_payload: list[int] = [0]
    for index in 0..64 { quant_payload.push(luma_quant[ZIGZAG[index]]) }
    segment(out, 0xDB, quant_payload)
    if channels == 3 {
        var chroma_payload: list[int] = [1]
        for index in 0..64 { chroma_payload.push(chroma_quant[ZIGZAG[index]]) }
        segment(out, 0xDB, chroma_payload)
    }

    var frame: list[int] = [
        8,
        height >> 8 & 255,
        height & 255,
        width >> 8 & 255,
        width & 255,
        channels,
    ]
    for index in 0..channels {
        frame.push(index + 1)
        frame.push(0x11)
        frame.push(if index == 0 { 0 } else { 1 })
    }
    segment(out, 0xC0, frame)

    let tables = [
        (0x00, DC_LUMA_COUNTS, DC_LUMA_VALUES),
        (0x10, AC_LUMA_COUNTS, AC_LUMA_VALUES),
        (0x01, DC_CHROMA_COUNTS, DC_CHROMA_VALUES),
        (0x11, AC_CHROMA_COUNTS, AC_CHROMA_VALUES),
    ]
    for (slot, counts, table_values) in tables {
        if channels == 1 and slot & 15 == 1 { continue }
        var payload: list[int] = [slot]
        for value in counts { payload.push(value) }
        for value in table_values { payload.push(value) }
        segment(out, 0xC4, payload)
    }

    var scan_header: list[int] = [channels]
    for index in 0..channels {
        scan_header.push(index + 1)
        scan_header.push(if index == 0 { 0x00 } else { 0x11 })
    }
    scan_header.push(0)
    scan_header.push(63)
    scan_header.push(0)
    segment(out, 0xDA, scan_header)

    let (dc_luma_code, dc_luma_len) = build_codes(DC_LUMA_COUNTS, DC_LUMA_VALUES)
    let (ac_luma_code, ac_luma_len) = build_codes(AC_LUMA_COUNTS, AC_LUMA_VALUES)
    let (dc_chroma_code, dc_chroma_len) = build_codes(DC_CHROMA_COUNTS, DC_CHROMA_VALUES)
    let (ac_chroma_code, ac_chroma_len) = build_codes(AC_CHROMA_COUNTS, AC_CHROMA_VALUES)

    let writer = BitWriter()
    var predictors = [0, 0, 0]
    #: One scratch block a component, filled by a single pass over the pixels.
    #: Converting inside the per-component loop instead ran the colour
    #: transform, the edge clamp and the address arithmetic three times for
    #: every pixel in the image.
    var planes = [[0.0 for _index in 0..64] for _component in 0..3]
    var coefficients = [0.0 for _index in 0..64]
    var dct_scratch = [0.0 for _index in 0..64]
    var levels = [0 for _index in 0..64]
    #: Locals, not globals and not a list of lists: each of these is read once
    #: a pixel or once a coefficient, and a module lookup or an outer index at
    #: that rate costs more than the arithmetic it feeds.
    let zigzag = ZIGZAG
    var luma_plane = planes[0]
    var cb_plane = planes[1]
    var cr_plane = planes[2]

    let blocks_across = (width + 7) // 8
    let blocks_down = (height + 7) // 8
    for by in 0..blocks_down {
        for bx in 0..blocks_across {
            for y in 0..8 {
                let sy = min(by * 8 + y, height - 1)
                let row = sy * width
                for x in 0..8 {
                    let sx = min(bx * 8 + x, width - 1)
                    let base = (row + sx) * channels
                    let cell = y * 8 + x
                    if channels == 1 {
                        luma_plane[cell] = values[base] - 128.0
                    } else {
                        let blue = values[base]
                        let green = values[base + 1]
                        let red = values[base + 2]
                        luma_plane[cell] = 0.299 * red + 0.587 * green + 0.114 * blue - 128.0
                        cb_plane[cell] = -0.168736 * red - 0.331264 * green + 0.5 * blue
                        cr_plane[cell] = 0.5 * red - 0.418688 * green - 0.081312 * blue
                    }
                }
            }
            for component in 0..channels {
                fdct_8x8(planes[component], coefficients, dct_scratch)
                let recip = if component == 0 { luma_recip } else { chroma_recip }
                let dc_code = if component == 0 { dc_luma_code } else { dc_chroma_code }
                let dc_len = if component == 0 { dc_luma_len } else { dc_chroma_len }
                let ac_code = if component == 0 { ac_luma_code } else { ac_chroma_code }
                let ac_len = if component == 0 { ac_luma_len } else { ac_chroma_len }

                for index in 0..64 {
                    let scaled = coefficients[zigzag[index]] * recip[index]
                    #: Round away from zero. `floor(x - 0.5)` for a negative
                    #: sends a coefficient of -1e-17 to -1, which turned every
                    #: flat block into noise and every file into ten times the
                    #: bytes it needed. `int()` truncates toward zero, so it is
                    #: already the floor on one side and the ceiling on the
                    #: other -- and unlike `math.floor` it is a single JIT
                    #: instruction rather than a call the tier declines.
                    levels[index] = if scaled >= 0.0 {
                        int(scaled + 0.5)
                    } else {
                        int(scaled - 0.5)
                    }
                }

                let difference = levels[0] - predictors[component]
                predictors[component] = levels[0]
                let dc_size = magnitude_of(difference)
                writer.write(dc_code[dc_size], dc_len[dc_size])
                if dc_size > 0 {
                    let encoded = if difference < 0 {
                        difference + (1 << dc_size) - 1
                    } else {
                        difference
                    }
                    writer.write(encoded, dc_size)
                }

                var run = 0
                for index in 1..64 {
                    let level = levels[index]
                    if level == 0 {
                        run += 1
                        continue
                    }
                    while run > 15 {
                        writer.write(ac_code[0xF0], ac_len[0xF0])
                        run -= 16
                    }
                    let size = magnitude_of(level)
                    let symbol = run * 16 + size
                    writer.write(ac_code[symbol], ac_len[symbol])
                    let encoded = if level < 0 { level + (1 << size) - 1 } else { level }
                    writer.write(encoded, size)
                    run = 0
                }
                if run > 0 { writer.write(ac_code[0], ac_len[0]) }
            }
        }
    }
    writer.flush()
    for byte in writer.out { out.push(byte) }
    out.push(0xFF)
    out.push(0xD9)
    return bytes(out)
}
