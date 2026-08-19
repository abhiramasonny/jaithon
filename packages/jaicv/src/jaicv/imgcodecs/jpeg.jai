#: Baseline JPEG, both directions.
#:
#: Decoding covers baseline sequential DCT with Huffman coding — the encoding
#: every camera and every web image uses — at any chroma subsampling, with
#: restart markers. Progressive JPEG is refused rather than half-decoded.
#:
#: Encoding writes baseline 4:2:0 for colour and 4:4:4 for greyscale, with the
#: standard Annex K tables scaled by a quality setting.
import std.math
from ..core.mat import Mat
from ..core.types import CV_8U, CV_MAKETYPE

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

    pub fn init(self, counts: list[int], values: list[int]) {
        self.values = values
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
pub fn decode(data: bytes) -> Mat {
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
                restart_interval
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
    restart_interval: int
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
        component.samples = [0.0 for _index in 0..component.width * component.height]
        component.predictor = 0
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
