#: Reading QR codes, with OpenCV's `QRCodeDetector` interface.
#:
#: ```
#: let detector = QRCodeDetector()
#: let (text, corners) = detector.detect_and_decode(frame)
#: ```
#:
#: The reader is the standard's, in the standard's order: find the three finder
#: patterns, take the perspective they imply, sample the module grid through
#: it, read the format information, unmask, de-interleave the blocks,
#: Reed-Solomon them (`galois.jai`), and decode the bit stream.
#:
#: What it covers: every version, one to forty; every error-correction level;
#: numeric, alphanumeric, byte and ECI segments, and structured-append headers,
#: which it steps over rather than reassembling. Kanji segments are decoded to
#: their Shift-JIS bytes rather than to text, because that mapping is a table
#: this does not carry.
#:
#: What it does not cover: micro QR, which is a different symbol with a
#: different layout, and inverted (light-on-dark) codes.
import std.math
from ..core.mat import Mat
from ..core.types import CV_8U, CV_MAKETYPE, Point2f
from ..imgproc.geometric import get_perspective_transform, perspective_transform
from ..imgproc.threshold import otsu_threshold
from .galois import gf_exp, gf_mul, rs_correct

#: Total codewords -- data plus error correction -- in each version.
let TOTAL_CODEWORDS = [
    26,
    44,
    70,
    100,
    134,
    172,
    196,
    242,
    292,
    346,
    404,
    466,
    532,
    581,
    655,
    733,
    815,
    901,
    991,
    1085,
    1156,
    1258,
    1364,
    1474,
    1588,
    1706,
    1828,
    1921,
    2051,
    2185,
    2323,
    2465,
    2611,
    2761,
    2876,
    3034,
    3196,
    3362,
    3532,
    3706,
]

#: Error-correction codewords per block, by version then level in the order
#: L, M, Q, H.
let EC_PER_BLOCK = [
    [7, 10, 13, 17],
    [10, 16, 22, 28],
    [15, 26, 18, 22],
    [20, 18, 26, 16],
    [26, 24, 18, 22],
    [18, 16, 24, 28],
    [20, 18, 18, 26],
    [24, 22, 22, 26],
    [30, 22, 20, 24],
    [18, 26, 24, 28],
    [20, 30, 28, 24],
    [24, 22, 26, 28],
    [26, 22, 24, 22],
    [30, 24, 20, 24],
    [22, 24, 30, 24],
    [24, 28, 24, 30],
    [28, 28, 28, 28],
    [30, 26, 28, 28],
    [28, 26, 26, 26],
    [28, 26, 30, 28],
    [28, 26, 28, 30],
    [28, 28, 30, 24],
    [30, 28, 30, 30],
    [30, 28, 30, 30],
    [26, 28, 30, 30],
    [28, 28, 28, 30],
    [30, 28, 30, 30],
    [30, 28, 30, 30],
    [30, 28, 30, 30],
    [30, 28, 30, 30],
    [30, 28, 30, 30],
    [30, 28, 30, 30],
    [30, 28, 30, 30],
    [30, 28, 30, 30],
    [30, 28, 30, 30],
    [30, 28, 30, 30],
    [30, 28, 30, 30],
    [30, 28, 30, 30],
    [30, 28, 30, 30],
    [30, 28, 30, 30],
]

#: How many blocks the codewords are split into, by version then level.
#:
#: Everything else about the split follows from these two tables and the total:
#: the data codewords are what is left after the parity, they divide as evenly
#: as they can, and the blocks that get one more are the last ones. Carrying
#: the per-group counts as well would be four more numbers a row to mistype
#: with nothing to check them against.
let BLOCK_COUNT = [
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 2, 2],
    [1, 2, 2, 4],
    [1, 2, 4, 4],
    [2, 4, 4, 4],
    [2, 4, 6, 5],
    [2, 4, 6, 6],
    [2, 5, 8, 8],
    [4, 5, 8, 8],
    [4, 5, 8, 11],
    [4, 8, 10, 11],
    [4, 9, 12, 16],
    [4, 9, 16, 16],
    [6, 10, 12, 18],
    [6, 10, 17, 16],
    [6, 11, 16, 19],
    [6, 13, 18, 21],
    [7, 14, 21, 25],
    [8, 16, 20, 25],
    [8, 17, 23, 25],
    [9, 17, 23, 34],
    [9, 18, 25, 30],
    [10, 20, 27, 32],
    [12, 21, 29, 35],
    [12, 23, 34, 37],
    [12, 25, 34, 40],
    [13, 26, 35, 42],
    [14, 28, 38, 45],
    [15, 29, 40, 48],
    [16, 31, 43, 51],
    [17, 33, 45, 54],
    [18, 35, 48, 57],
    [19, 37, 51, 60],
    [19, 38, 53, 63],
    [20, 40, 56, 66],
    [21, 43, 59, 70],
    [22, 45, 62, 74],
    [24, 47, 65, 77],
    [25, 49, 68, 81],
]

#: The two format bits, in the order the symbol stores them, mapped to the
#: index the tables above use. The standard numbers the levels by capacity and
#: encodes them by something else entirely, which is the whole reason for this.
let LEVEL_OF_BITS = [1, 0, 3, 2]

#: Names, for `QRCode.correction_level`.
pub let LEVEL_NAMES = ["L", "M", "Q", "H"]

#: How wide a symbol of this version is, in modules.
pub fn version_size(version: int) -> int { return 17 + 4 * version }

#: Where the alignment patterns' centres sit for a version.
#:
#: Generated rather than tabulated: the standard's table is forty rows of up to
#: seven numbers and this rule reproduces every one of them. The first centre is
#: always at six and the last always seven from the edge; the rest are spread
#: between them at an even spacing, rounded up.
pub fn alignment_centres(version: int) -> list[int] {
    if version < 2 { return [] }
    let count = version // 7 + 2
    #: The spacing is the gap divided by the number of gaps, rounded UP to an
    #: even number -- rounding down puts every centre after the first in the
    #: wrong column for versions 30, 36 and 39, and the symbol then reads its
    #: data out of the alignment patterns. Version 32 is the standard's one
    #: exception and does not follow the rule at all.
    let step = if version == 32 { 26 } else { (version * 4 + count * 2 + 1) // (count * 2 - 2) * 2 }
    var out = [6]
    var position = version * 4 + 10
    var rest: list[int] = []
    while rest.len() < count - 1 {
        rest.push(position)
        position -= step
    }
    for index in 0..rest.len() { out.push(rest[rest.len() - 1 - index]) }
    return out
}

fn compute_function_map(version: int) -> list[list[bool]] {
    let side = version_size(version)
    var map: list[list[bool]] = []
    for _y in 0..side {
        var row: list[bool] = []
        for _x in 0..side { row.push(false) }
        map.push(row)
    }

    #: The three finders with their separators, and the format information
    #: beside them.
    for y in 0..9 {
        for x in 0..9 { map[y][x] = true }
    }
    for y in 0..9 {
        for x in side - 8..side { map[y][x] = true }
    }
    for y in side - 8..side {
        for x in 0..9 { map[y][x] = true }
    }

    #: Timing: the sixth row and the sixth column, end to end.
    for index in 0..side {
        map[6][index] = true
        map[index][6] = true
    }

    #: Version information, for anything large enough to carry it.
    if version >= 7 {
        for y in 0..6 {
            for x in side - 11..side - 8 { map[y][x] = true }
        }
        for y in side - 11..side - 8 {
            for x in 0..6 { map[y][x] = true }
        }
    }

    let centres = alignment_centres(version)
    for cy in centres {
        for cx in centres {
            #: The three corners an alignment pattern would overlap a finder in
            #: do not carry one.
            if cy == 6 and cx == 6 { continue }
            if cy == 6 and cx == centres[centres.len() - 1] { continue }
            if cx == 6 and cy == centres[centres.len() - 1] { continue }
            for y in cy - 2..cy + 3 {
                for x in cx - 2..cx + 3 { map[y][x] = true }
            }
        }
    }
    return map
}

var _function_maps: dict[int, list[list[bool]]] = {}

#: Which modules of a version carry the finder, timing, alignment and
#: information patterns rather than data.
#:
#: Built once per version and kept. The predicate form below is asked for every
#: module of every symbol, eight times over while a mask is chosen, and
#: answering it by walking the alignment centres each time was two thirds of
#: the cost of writing a version 40 symbol.
pub fn function_map(version: int) -> list[list[bool]] {
    let existing = _function_maps.get(version, null)
    if existing is not null { return existing }
    let built = compute_function_map(version)
    _function_maps[version] = built
    return built
}

#: Whether a module is part of the finder, timing, alignment or information
#: patterns rather than of the data.
pub fn is_function_module(version: int, y: int, x: int) -> bool {
    return function_map(version)[y][x]
}

#: Whether the mask at `pattern` flips the module at `(y, x)`.
pub fn mask_bit(pattern: int, y: int, x: int) -> bool {
    if pattern == 0 { return (y + x) % 2 == 0 }
    if pattern == 1 { return y % 2 == 0 }
    if pattern == 2 { return x % 3 == 0 }
    if pattern == 3 { return (y + x) % 3 == 0 }
    if pattern == 4 { return (y // 2 + x // 3) % 2 == 0 }
    if pattern == 5 { return y * x % 2 + y * x % 3 == 0 }
    if pattern == 6 { return (y * x % 2 + y * x % 3) % 2 == 0 }
    return ((y + x) % 2 + y * x % 3) % 2 == 0
}

fn popcount(value: int) -> int {
    var bits = 0
    var rest = value
    while rest != 0 {
        bits += rest & 1
        rest = rest >> 1
    }
    return bits
}

#: The fifteen-bit format word closest to `bits`, as (level index, mask), or
#: null when nothing is within the code's correcting distance.
#:
#: The word is a BCH(15,5) code with a fixed mask applied, and there are only
#: thirty-two of them, so the nearest is found by trying all thirty-two rather
#: than by decoding: the same answer, and no syndrome arithmetic to get wrong.
pub fn decode_format(bits: int) -> tuple[int, int]? {
    var best = -1
    var best_distance = 16
    for candidate in 0..32 {
        var remainder = candidate << 10
        for shift in 0..5 {
            let bit = 14 - shift
            if remainder >> bit & 1 == 1 { remainder = remainder ^ 0x537 << bit - 10 }
        }
        let word = (candidate << 10 | remainder) ^ 0x5412
        let distance = popcount(word ^ bits)
        if distance < best_distance {
            best_distance = distance
            best = candidate
        }
    }
    if best < 0 or best_distance > 3 { return null }
    return (LEVEL_OF_BITS[best >> 3], best & 7)
}

#: The version an eighteen-bit version word names, or null.
#:
#: Same approach as the format word and for the same reason: thirty-four
#: candidates is fewer than a BCH(18,6) decoder.
pub fn decode_version(bits: int) -> int? {
    var best = -1
    var best_distance = 19
    for version in 7..41 {
        var remainder = version << 12
        for shift in 0..6 {
            let bit = 17 - shift
            if remainder >> bit & 1 == 1 { remainder = remainder ^ 0x1F25 << bit - 12 }
        }
        let word = version << 12 | remainder
        let distance = popcount(word ^ bits)
        if distance < best_distance {
            best_distance = distance
            best = version
        }
    }
    if best < 0 or best_distance > 3 { return null }
    return best
}

#: A symbol read off a module grid: what it says and how it was stored.
pub class QRCode {
    #: The decoded text.
    pub let text: str
    #: The symbol's version, one to forty.
    pub let version: int
    #: `"L"`, `"M"`, `"Q"` or `"H"`.
    pub let correction_level: str
    #: Which of the eight data masks was applied.
    pub let mask: int
    #: The raw bytes, before any character decoding.
    pub let payload: bytes

    pub fn init(self, text: str, version: int, level: int, mask: int, payload: bytes) {
        self.text = text
        self.version = version
        self.correction_level = LEVEL_NAMES[level]
        self.mask = mask
        self.payload = payload
    }

    pub fn to_str(self) -> str {
        return f"QRCode(v{self.version}{self.correction_level}, mask {self.mask}, {repr(self.text)})"
    }

    pub fn __str__(self) -> str { return self.to_str() }
}

fn grid_at(grid: list[list[bool]], y: int, x: int) -> bool { return grid[y][x] }

#: The two format words a symbol carries, read from their two placements.
fn read_format_words(grid: list[list[bool]], side: int) -> tuple[int, int] {
    #: Around the top-left finder, skipping the timing module at (6, 6).
    var first = 0
    for index in 0..15 {
        let x = if index < 6 { index } elif index == 6 { 7 } else { 8 }
        let y = if index < 6 { 8 } elif index < 8 { 8 } elif index == 8 { 7 } else { 14 - index }
        first = first << 1 | (if grid_at(grid, y, x) { 1 } else { 0 })
    }

    #: The copy split between the bottom-left and top-right finders. Read most
    #: significant bit first, which is the order the standard lays it out in
    #: going up the left column and then right along the top row.
    var second = 0
    for index in 0..15 {
        if index < 8 {
            second = second << 1 | (if grid_at(grid, side - 1 - index, 8) { 1 } else { 0 })
        } else {
            second = second << 1 | (if grid_at(grid, 8, side - 15 + index) { 1 } else { 0 })
        }
    }
    return (first, second)
}

#: The two version words, for a symbol large enough to carry them.
fn read_version_words(grid: list[list[bool]], side: int) -> tuple[int, int] {
    var bottom = 0
    var right = 0
    #: Bit `b` of the word sits at column `b / 3` and three rows starting at
    #: `side - 11`, offset by `b % 3`. Reading the three-module groups the other
    #: way round -- which is easy to do and produces a plausible number -- gives
    #: a word that decodes to some other version, and the symbol is then thrown
    #: out as damaged. Versions 7 to 20 hid it: their garbled words decoded to
    #: nothing at all, and a word that decodes to nothing is treated as absent.
    for index in 0..18 {
        let bit = 17 - index
        let column = bit // 3
        let offset = bit % 3
        bottom = bottom << 1 | (if grid_at(grid, side - 11 + offset, column) { 1 } else { 0 })
        right = right << 1 | (if grid_at(grid, column, side - 11 + offset) { 1 } else { 0 })
    }
    return (bottom, right)
}

#: Every data module, in the order the standard writes them: two columns at a
#: time from the bottom right, upwards then downwards, skipping the column the
#: vertical timing pattern occupies.
fn read_codewords(grid: list[list[bool]], version: int, mask: int) -> list[int] {
    let side = version_size(version)
    var out: list[int] = []
    var current = 0
    var filled = 0
    var column = side - 1
    var upward = true

    while column > 0 {
        if column == 6 { column -= 1 }
        for step in 0..side {
            let y = if upward { side - 1 - step } else { step }
            for offset in 0..2 {
                let x = column - offset
                if is_function_module(version, y, x) { continue }
                var bit = grid_at(grid, y, x)
                if mask_bit(mask, y, x) { bit = not bit }
                current = current << 1 | (if bit { 1 } else { 0 })
                filled += 1
                if filled == 8 {
                    out.push(current)
                    current = 0
                    filled = 0
                }
            }
        }
        upward = not upward
        column -= 2
    }
    return out
}

#: Undo the interleaving the encoder applies and correct each block.
#:
#: Returns the data codewords in order, or null when any block has more errors
#: than its parity can locate.
fn deinterleave(codewords: list[int], version: int, level: int) -> list[int]? {
    let total = TOTAL_CODEWORDS[version - 1]
    let ec_each = EC_PER_BLOCK[version - 1][level]
    let blocks = BLOCK_COUNT[version - 1][level]
    if blocks <= 0 or total <= 0 { return null }
    let data_total = total - ec_each * blocks
    if data_total <= 0 or codewords.len() < total { return null }

    let short_data = data_total // blocks
    let long_blocks = data_total % blocks
    var lengths: list[int] = []
    for index in 0..blocks {
        lengths.push(if index >= blocks - long_blocks { short_data + 1 } else { short_data })
    }

    var data: list[list[int]] = []
    for _index in 0..blocks { data.push([]) }
    var at = 0
    for column in 0..short_data + 1 {
        for block in 0..blocks {
            if column >= lengths[block] { continue }
            data[block].push(codewords[at])
            at += 1
        }
    }
    var parity: list[list[int]] = []
    for _index in 0..blocks { parity.push([]) }
    for column in 0..ec_each {
        for block in 0..blocks {
            parity[block].push(codewords[at])
            at += 1
        }
    }

    var out: list[int] = []
    for block in 0..blocks {
        var whole: list[int] = []
        for value in data[block] { whole.push(value) }
        for value in parity[block] { whole.push(value) }
        if not rs_correct(whole, ec_each) { return null }
        for index in 0..lengths[block] { out.push(whole[index]) }
    }
    return out
}

#: A cursor over a codeword stream, reading it as bits.
class BitReader {
    let data: list[int]
    var at: int

    pub fn init(self, data: list[int]) {
        self.data = data
        self.at = 0
    }

    pub fn remaining(self) -> int { return self.data.len() * 8 - self.at }

    pub fn read(self, count: int) -> int {
        if count > self.remaining() { throw ValueError("the bit stream ended early") }
        var value = 0
        for _index in 0..count {
            let byte = self.data[self.at // 8]
            let bit = byte >> 7 - self.at % 8 & 1
            value = value << 1 | bit
            self.at += 1
        }
        return value
    }
}

#: How many bits a segment's character count takes, which depends on both the
#: mode and how large the symbol is.
fn count_bits(mode: int, version: int) -> int {
    let band = if version <= 9 { 0 } elif version <= 26 { 1 } else { 2 }
    if mode == 1 { return [10, 12, 14][band] }
    if mode == 2 { return [9, 11, 13][band] }
    if mode == 4 { return [8, 16, 16][band] }
    if mode == 8 { return [8, 10, 12][band] }
    return 0
}

let ALPHANUMERIC = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"

#: Read the segments out of the corrected data codewords.
fn decode_segments(data: list[int], version: int) -> tuple[str, bytes] {
    let reader = BitReader(data)
    var text = ""
    var raw: list[int] = []

    while reader.remaining() >= 4 {
        let mode = reader.read(4)
        #: The terminator, or padding that reads as one.
        if mode == 0 { break }

        if mode == 7 {
            #: An ECI header names a character set. The assignment number is
            #: one, two or three bytes, flagged by its leading bits; the text
            #: after it is read as bytes either way, so the number is stepped
            #: over rather than acted on.
            let first = reader.read(8)
            if first & 0x80 == 0 {
                #: one byte, nothing more to read
            } elif first & 0xC0 == 0x80 {
                let _rest = reader.read(8)
            } else {
                let _rest = reader.read(16)
            }
            continue
        }
        if mode == 3 {
            #: A structured-append header: which part of how many, and a parity
            #: over the whole message. Reassembly is the caller's business, so
            #: the header is stepped over and this part decoded on its own.
            let _position = reader.read(16)
            continue
        }
        if mode != 1 and mode != 2 and mode != 4 and mode != 8 {
            throw ValueError(f"unsupported QR segment mode {mode}")
        }

        let count = reader.read(count_bits(mode, version))
        if mode == 1 {
            var index = 0
            while index < count {
                let left = count - index
                if left >= 3 {
                    let group = reader.read(10)
                    text += f"{group // 100}{group // 10 % 10}{group % 10}"
                    index += 3
                } elif left == 2 {
                    let group = reader.read(7)
                    text += f"{group // 10}{group % 10}"
                    index += 2
                } else {
                    text += f"{reader.read(4)}"
                    index += 1
                }
            }
        } elif mode == 2 {
            var index = 0
            while index < count {
                if count - index >= 2 {
                    let pair = reader.read(11)
                    text += ALPHANUMERIC[pair // 45]
                    text += ALPHANUMERIC[pair % 45]
                    index += 2
                } else {
                    text += ALPHANUMERIC[reader.read(6)]
                    index += 1
                }
            }
        } elif mode == 4 {
            var octets: list[int] = []
            for _index in 0..count { octets.push(reader.read(8)) }
            for value in octets { raw.push(value) }
            text += bytes(octets).decode()
        } else {
            #: Kanji, kept as its Shift-JIS bytes. Turning them into text needs
            #: a mapping table this does not carry, and inventing one would be
            #: worse than handing back what was stored.
            for _index in 0..count {
                let packed = reader.read(13)
                var value = packed // 192 << 8 | packed % 192
                value += if value < 0x1F00 { 0x8140 } else { 0xC140 }
                raw.push(value >> 8 & 255)
                raw.push(value & 255)
            }
        }
    }
    #: A code with no byte segment still has bytes: whatever its text is in
    #: UTF-8. `payload` is meant to be what the symbol carried, and for a
    #: numeric or alphanumeric segment that is the same thing.
    if raw.len() == 0 {
        for value in bytes(text).to_list() { raw.push(value) }
    }
    return (text, bytes(raw))
}

#: Read a symbol out of a module grid, where `true` is a dark module.
#:
#: Returns null when the grid is not a symbol: a size that names no version, a
#: format word too damaged to read, or a block the parity cannot recover.
pub fn decode_grid(grid: list[list[bool]]) -> QRCode? {
    let side = grid.len()
    if side < 21 or side > 177 or (side - 17) % 4 != 0 { return null }
    for row in grid {
        if row.len() != side { return null }
    }
    var version = (side - 17) // 4

    if version >= 7 {
        let (bottom, right) = read_version_words(grid, side)
        let named = decode_version(bottom) ?? decode_version(right)
        #: The words are a check, not the source: the size already says which
        #: version this is, and a symbol whose words disagree with its own size
        #: is damaged rather than a different version.
        if named is not null and named != version { return null }
    }

    let (first, second) = read_format_words(grid, side)
    let format = decode_format(first) ?? decode_format(second)
    if format is null { return null }
    let (level, mask) = format

    let codewords = read_codewords(grid, version, mask)
    let data = deinterleave(codewords, version, level)
    if data is null { return null }
    let (text, raw) = decode_segments(data, version)
    return QRCode(text, version, level, mask, raw)
}

#: --------------------------------------------------------------------------
#: Writing
#: --------------------------------------------------------------------------
#:
#: The encoder is here for the same reason a codec's writer sits beside its
#: reader: the two together are checkable without anything else, and a round
#: trip over every version and every level is what says the tables above are
#: right. It is also what `QRCodeEncoder` is for.
fn ec_generator(count: int) -> list[int] {
    var poly = [1]
    for index in 0..count {
        var next: list[int] = []
        for _slot in 0..poly.len() + 1 { next.push(0) }
        for slot in 0..poly.len() {
            next[slot] = next[slot] ^ poly[slot]
            next[slot + 1] = next[slot + 1] ^ gf_mul(poly[slot], gf_exp(index))
        }
        poly = next
    }
    return poly
}

fn ec_of(data: list[int], count: int) -> list[int] {
    let generator = ec_generator(count)
    var remainder: list[int] = []
    for value in data { remainder.push(value) }
    for _index in 0..count { remainder.push(0) }
    for index in 0..data.len() {
        let lead = remainder[index]
        if lead == 0 { continue }
        for slot in 0..generator.len() {
            remainder[index + slot] = remainder[index + slot] ^ gf_mul(generator[slot], lead)
        }
    }
    var out: list[int] = []
    for index in data.len()..remainder.len() { out.push(remainder[index]) }
    return out
}

fn alphanumeric_index(character: str) -> int {
    for index in 0..ALPHANUMERIC.len() {
        if ALPHANUMERIC[index] == character { return index }
    }
    return -1
}

fn is_all_digits(text: str) -> bool {
    if text.len() == 0 { return false }
    for character in text.chars() {
        if character < "0" or character > "9" { return false }
    }
    return true
}

fn is_all_alphanumeric(text: str) -> bool {
    if text.len() == 0 { return false }
    for character in text.chars() {
        if alphanumeric_index(character) < 0 { return false }
    }
    return true
}

class BitWriter {
    var bits: list[int]

    pub fn init(self) { self.bits = [] }

    pub fn write(self, value: int, count: int) -> void {
        for index in 0..count { self.bits.push(value >> count - 1 - index & 1) }
    }

    pub fn len(self) -> int { return self.bits.len() }

    pub fn codewords(self) -> list[int] {
        var out: list[int] = []
        var index = 0
        while index < self.bits.len() {
            var byte = 0
            for offset in 0..8 {
                let bit = if index + offset < self.bits.len() {
                    self.bits[index + offset]
                } else {
                    0
                }
                byte = byte << 1 | bit
            }
            out.push(byte)
            index += 8
        }
        return out
    }
}

fn encode_segment(writer: BitWriter, text: str, version: int) -> void {
    if is_all_digits(text) {
        writer.write(1, 4)
        writer.write(text.len(), count_bits(1, version))
        var index = 0
        while index < text.len() {
            let left = text.len() - index
            if left >= 3 {
                writer.write(int(text[index:index + 3]), 10)
                index += 3
            } elif left == 2 {
                writer.write(int(text[index:index + 2]), 7)
                index += 2
            } else {
                writer.write(int(text[index:index + 1]), 4)
                index += 1
            }
        }
        return
    }
    if is_all_alphanumeric(text) {
        writer.write(2, 4)
        writer.write(text.len(), count_bits(2, version))
        var index = 0
        while index < text.len() {
            if text.len() - index >= 2 {
                let first = alphanumeric_index(text[index:index + 1])
                let second = alphanumeric_index(text[index + 1:index + 2])
                writer.write(first * 45 + second, 11)
                index += 2
            } else {
                writer.write(alphanumeric_index(text[index:index + 1]), 6)
                index += 1
            }
        }
        return
    }
    let octets = bytes(text).to_list()
    writer.write(4, 4)
    writer.write(octets.len(), count_bits(4, version))
    for value in octets { writer.write(value, 8) }
}

#: The smallest version that holds `text` at `level`, or null when none does.
pub fn smallest_version(text: str, level: int) -> int? {
    for version in 1..41 {
        let capacity = TOTAL_CODEWORDS[version - 1] -
            EC_PER_BLOCK[version - 1][level] * BLOCK_COUNT[version - 1][level]
        let writer = BitWriter()
        encode_segment(writer, text, version)
        if writer.len() + 4 <= capacity * 8 { return version }
    }
    return null
}

fn penalty(grid: list[list[bool]]) -> int {
    let side = grid.len()
    var score = 0

    #: Runs of five or more of one shade, along both axes.
    for axis in 0..2 {
        for line in 0..side {
            var run = 1
            for step in 1..side {
                let previous = if axis == 0 { grid[line][step - 1] } else { grid[step - 1][line] }
                let current = if axis == 0 { grid[line][step] } else { grid[step][line] }
                if current == previous {
                    run += 1
                    continue
                }
                if run >= 5 { score += run - 2 }
                run = 1
            }
            if run >= 5 { score += run - 2 }
        }
    }

    #: Two-by-two blocks of one shade.
    for y in 0..side - 1 {
        for x in 0..side - 1 {
            let corner = grid[y][x]
            if grid[y][x + 1] == corner and
                grid[y + 1][x] == corner and
                grid[y + 1][x + 1] == corner {
                score += 3
            }
        }
    }

    #: The finder-like run, which a scanner could mistake for one.
    let pattern = [true, false, true, true, true, false, true, false, false, false, false]
    let reverse = [false, false, false, false, true, false, true, true, true, false, true]
    for axis in 0..2 {
        for line in 0..side {
            for start in 0..side - 10 {
                var forward = true
                var backward = true
                for offset in 0..11 {
                    let value = if axis == 0 {
                        grid[line][start + offset]
                    } else {
                        grid[start + offset][line]
                    }
                    if value != pattern[offset] { forward = false }
                    if value != reverse[offset] { backward = false }
                    #: Almost every start position disagrees with both within
                    #: the first module or two, and reading the other nine
                    #: answers nothing. This is the innermost loop of the mask
                    #: choice, which runs eight times over every symbol.
                    if not forward and not backward { break }
                }
                if forward { score += 40 }
                if backward { score += 40 }
            }
        }
    }

    #: How far the balance of dark to light is from even.
    var dark = 0
    for row in grid {
        for value in row {
            if value { dark += 1 }
        }
    }
    let percent = dark * 100 // (side * side)
    let low = math.abs(percent - 50) // 5
    score += low * 10
    return score
}

fn place_function_patterns(grid: list[list[bool]], version: int) -> void {
    let side = version_size(version)

    for corner in [[0, 0], [0, side - 7], [side - 7, 0]] {
        let top = corner[0]
        let left = corner[1]
        for y in 0..7 {
            for x in 0..7 {
                let edge = y == 0 or y == 6 or x == 0 or x == 6
                let core = y >= 2 and y <= 4 and x >= 2 and x <= 4
                grid[top + y][left + x] = edge or core
            }
        }
    }

    for index in 8..side - 8 {
        grid[6][index] = index % 2 == 0
        grid[index][6] = index % 2 == 0
    }

    let centres = alignment_centres(version)
    for cy in centres {
        for cx in centres {
            if cy == 6 and cx == 6 { continue }
            if cy == 6 and cx == centres[centres.len() - 1] { continue }
            if cx == 6 and cy == centres[centres.len() - 1] { continue }
            for y in 0..5 {
                for x in 0..5 {
                    let ring = y == 0 or y == 4 or x == 0 or x == 4
                    grid[cy - 2 + y][cx - 2 + x] = ring or y == 2 and x == 2
                }
            }
        }
    }

    #: The one module that is always dark, beside the bottom-left finder.
    grid[side - 8][8] = true
}

fn place_format(grid: list[list[bool]], side: int, level: int, mask: int) -> void {
    var candidate = 0
    for index in 0..4 {
        if LEVEL_OF_BITS[index] == level { candidate = index << 3 | mask }
    }
    var remainder = candidate << 10
    for shift in 0..5 {
        let bit = 14 - shift
        if remainder >> bit & 1 == 1 { remainder = remainder ^ 0x537 << bit - 10 }
    }
    let word = (candidate << 10 | remainder) ^ 0x5412

    for index in 0..15 {
        let bit = word >> 14 - index & 1 == 1
        let x = if index < 6 { index } elif index == 6 { 7 } else { 8 }
        let y = if index < 8 { 8 } elif index == 8 { 7 } else { 14 - index }
        grid[y][x] = bit
        if index < 8 {
            grid[side - 1 - index][8] = bit
        } else {
            grid[8][side - 15 + index] = bit
        }
    }
}

fn place_version(grid: list[list[bool]], side: int, version: int) -> void {
    if version < 7 { return }
    var remainder = version << 12
    for shift in 0..6 {
        let bit = 17 - shift
        if remainder >> bit & 1 == 1 { remainder = remainder ^ 0x1F25 << bit - 12 }
    }
    let word = version << 12 | remainder
    for index in 0..18 {
        let bit = 17 - index
        let column = bit // 3
        let offset = bit % 3
        let value = word >> bit & 1 == 1
        grid[side - 11 + offset][column] = value
        grid[column][side - 11 + offset] = value
    }
}

#: Build the module grid for `text`, `true` where a module is dark.
#:
#: `version` of zero picks the smallest that fits. The mask is chosen the way
#: the standard says: all eight are laid down and the one with the lowest
#: penalty wins.
pub fn encode_grid(text: str, level: int = 0, version: int = 0) -> list[list[bool]] {
    if level < 0 or level > 3 { throw ValueError("the correction level runs from 0 (L) to 3 (H)") }
    let wanted = if version > 0 { version } else { smallest_version(text, level) ?? 0 }
    if wanted < 1 or wanted > 40 { throw ValueError("this text does not fit in any QR version") }

    let capacity = TOTAL_CODEWORDS[wanted - 1] -
        EC_PER_BLOCK[wanted - 1][level] * BLOCK_COUNT[wanted - 1][level]
    let writer = BitWriter()
    encode_segment(writer, text, wanted)
    if writer.len() > capacity * 8 {
        throw ValueError(
            f"this text does not fit in a version {wanted} symbol at level {LEVEL_NAMES[level]}"
        )
    }
    #: The terminator, then whatever it takes to reach a whole codeword, then
    #: the two pad bytes the standard names, alternating.
    for _index in 0..min(4, capacity * 8 - writer.len()) { writer.write(0, 1) }
    while writer.len() % 8 != 0 { writer.write(0, 1) }
    var data = writer.codewords()
    var pad = 0
    while data.len() < capacity {
        data.push(if pad % 2 == 0 { 236 } else { 17 })
        pad += 1
    }

    let ec_each = EC_PER_BLOCK[wanted - 1][level]
    let blocks = BLOCK_COUNT[wanted - 1][level]
    let short_data = capacity // blocks
    let long_blocks = capacity % blocks
    var groups: list[list[int]] = []
    var parities: list[list[int]] = []
    var at = 0
    for block in 0..blocks {
        let length = if block >= blocks - long_blocks { short_data + 1 } else { short_data }
        var piece: list[int] = []
        for _index in 0..length {
            piece.push(data[at])
            at += 1
        }
        groups.push(piece)
        parities.push(ec_of(piece, ec_each))
    }

    var stream: list[int] = []
    for column in 0..short_data + 1 {
        for block in 0..blocks {
            if column < groups[block].len() { stream.push(groups[block][column]) }
        }
    }
    for column in 0..ec_each {
        for block in 0..blocks { stream.push(parities[block][column]) }
    }

    let side = version_size(wanted)
    let functions = function_map(wanted)

    #: The unmasked symbol, laid out once. The eight masks differ only in which
    #: data modules they flip and in the format word beside the finders, so
    #: everything else -- the patterns, the version word, the zigzag walk that
    #: places the stream -- is done here rather than eight times over.
    var base: list[list[bool]] = []
    for _y in 0..side {
        var row: list[bool] = []
        for _x in 0..side { row.push(false) }
        base.push(row)
    }
    place_function_patterns(base, wanted)
    place_version(base, side, wanted)

    var index = 0
    var column = side - 1
    var upward = true
    while column > 0 {
        if column == 6 { column -= 1 }
        for step in 0..side {
            let y = if upward { side - 1 - step } else { step }
            for offset in 0..2 {
                let x = column - offset
                if functions[y][x] { continue }
                var bit = false
                if index < stream.len() * 8 {
                    bit = stream[index // 8] >> 7 - index % 8 & 1 == 1
                }
                index += 1
                base[y][x] = bit
            }
        }
        upward = not upward
        column -= 2
    }

    var best: list[list[bool]] = []
    var best_score = -1
    for mask in 0..8 {
        var grid: list[list[bool]] = []
        for y in 0..side {
            var row: list[bool] = []
            for x in 0..side {
                let value = base[y][x]
                row.push(if not functions[y][x] and mask_bit(mask, y, x) {
                    not value
                } else {
                    value
                })
            }
            grid.push(row)
        }
        place_format(grid, side, level, mask)

        let score = penalty(grid)
        if best_score < 0 or score < best_score {
            best_score = score
            best = grid
        }
    }
    return best
}

#: --------------------------------------------------------------------------
#: Finding a symbol in a picture
#: --------------------------------------------------------------------------
pub class Finder {
    pub var y: float
    pub var x: float
    pub var width: float
    pub var count: int

    pub fn init(self, y: float, x: float, width: float) {
        self.y = y
        self.x = x
        self.width = width
        self.count = 1
    }
}

#: Whether five consecutive runs have the 1:1:3:1:1 shape a finder's centre
#: line always has, allowing each run half a module of slack.
fn finder_ratio(runs: list[int]) -> bool {
    var total = 0
    for value in runs { total += value }
    if total < 7 { return false }
    let unit = float(total) / 7.0
    let slack = unit / 2.0
    let wanted = [1.0, 1.0, 3.0, 1.0, 1.0]
    for index in 0..5 {
        if math.fabs(float(runs[index]) - wanted[index] * unit) >= wanted[index] * slack {
            return false
        }
    }
    return true
}

#: Walk one line of the image and report every 1:1:3:1:1 run of dark and light
#: it holds, as (centre along the line, module width).
fn scan_line(values: list[float], start: int, step: int, count: int) -> list[tuple[float, float]] {
    var out: list[tuple[float, float]] = []
    var runs = [0, 0, 0, 0, 0]
    var filled = 0
    var dark = values[start] < 128.0
    var length = 0

    for position in 0..count + 1 {
        let ended = position == count
        let value = if ended { not dark } else { values[start + position * step] < 128.0 }
        if not ended and value == dark {
            length += 1
            continue
        }
        #: A run just ended. It shifts into the window, and if the window now
        #: holds five runs ending on a dark one in the 1:1:3:1:1 proportion,
        #: the middle of the wide run is a finder's centre.
        for index in 0..4 { runs[index] = runs[index + 1] }
        runs[4] = length
        filled += 1
        if filled >= 5 and dark and finder_ratio(runs) {
            var total = 0
            for value_length in runs { total += value_length }
            let middle = float(position - runs[4] - runs[3]) - float(runs[2]) / 2.0
            out.push((middle, float(total) / 7.0))
        }
        dark = value
        length = 1
    }
    return out
}

#: Every finder pattern in a binarised image.
#:
#: The rows are scanned first, which finds each finder many times over; every
#: hit is then confirmed by scanning the column through it, and the confirmed
#: hits are merged. A finder seen on one row and no column is not a finder.
pub fn find_finders(values: list[float], rows: int, cols: int) -> list[Finder] {
    var found = scan_rows(values, rows, cols, 2)
    #: A finder is at least seven modules tall, so every other row still
    #: crosses one three times over and the pass costs half as much. Only a
    #: picture where that was not enough pays for the finer sweep.
    if found.len() < 3 { found = scan_rows(values, rows, cols, 1) }

    #: One row through a finder is a coincidence; several is a finder. A real
    #: finder is met on every row it spans, so it collects the most rows, and
    #: the dense middle of a large symbol throws up a scatter of shapes that
    #: pass the ratio test on two rows and no more.
    var best_count = 0
    for candidate in found {
        if candidate.count > best_count { best_count = candidate.count }
    }
    let floor = max(2, best_count // 2)

    var strong: list[Finder] = []
    #: Strongest first, and no more than sixteen. Every triple is considered,
    #: so the count matters cubically: thirty candidates is four thousand
    #: triples, and each one that passes the shape test is sampled over the
    #: whole symbol. Reading a version 40 symbol took twenty-seven seconds
    #: before this and under a second after.
    var taken: list[bool] = []
    for _index in 0..found.len() { taken.push(false) }
    while strong.len() < 16 {
        var pick = -1
        for index in 0..found.len() {
            if taken[index] or found[index].count < floor { continue }
            if pick < 0 or found[index].count > found[pick].count { pick = index }
        }
        if pick < 0 { break }
        taken[pick] = true
        strong.push(found[pick])
    }
    return strong
}

fn scan_rows(values: list[float], rows: int, cols: int, step: int) -> list[Finder] {
    var found: list[Finder] = []
    var y = 0
    while y < rows {
        for (centre, width) in scan_line(values, y * cols, 1, cols) {
            #: Merge into a finder already seen before confirming, not after.
            #: The confirmation walks a whole column, and a finder is met on
            #: every row it spans: confirming each time made reading a version
            #: 40 symbol thirty seconds' work, nearly all of it re-proving what
            #: the first row had already established.
            var merged = false
            for existing in found {
                if math.fabs(existing.y - float(y)) < existing.width * 4.0 and
                    math.fabs(existing.x - centre) < existing.width * 2.0 {
                    let weight = float(existing.count)
                    existing.x = (existing.x * weight + centre) / (weight + 1.0)
                    existing.width = (existing.width * weight + width) / (weight + 1.0)
                    existing.count += 1
                    merged = true
                    break
                }
            }
            if merged { continue }

            let x = int(centre)
            if x < 0 or x >= cols { continue }
            var confirmed = false
            var centre_y = 0.0
            var size = 0.0
            for hit in scan_line(values, x, cols, rows) {
                if math.fabs(hit[0] - float(y)) > hit[1] * 3.0 { continue }
                confirmed = true
                centre_y = hit[0]
                size = (width + hit[1]) / 2.0
                break
            }
            if not confirmed or size < 1.0 { continue }
            found.push(Finder(centre_y, centre, size))
        }
        y += step
    }
    return found
}

fn distance(ay: float, ax: float, by: float, bx: float) -> float {
    let dy = ay - by
    let dx = ax - bx
    return math.sqrt(dy * dy + dx * dx)
}

#: Order three finders as (corner, along the top, down the side).
#:
#: The corner is the one opposite the longest side, which is the diagonal; of
#: the other two, the one that turns clockwise from the corner is the top.
fn order_finders(a: Finder, b: Finder, c: Finder) -> tuple[Finder, Finder, Finder] {
    let ab = distance(a.y, a.x, b.y, b.x)
    let bc = distance(b.y, b.x, c.y, c.x)
    let ca = distance(c.y, c.x, a.y, a.x)
    var corner = a
    var first = b
    var second = c
    if ab >= bc and ab >= ca {
        corner = c
        first = a
        second = b
    } elif bc >= ab and bc >= ca {
        corner = a
        first = b
        second = c
    } else {
        corner = b
        first = c
        second = a
    }
    let cross = (first.x - corner.x) * (second.y - corner.y) -
        (first.y - corner.y) * (second.x - corner.x)
    if cross < 0.0 { return (corner, second, first) }
    return (corner, first, second)
}

#: Sample a symbol's modules through the perspective its three finders imply.
#:
#: The three finder centres sit three and a half modules in from their corners,
#: which fixes three of the four points a perspective needs. The fourth is the
#: opposite corner, estimated from the other three and then refined against the
#: alignment pattern that sits near it -- without that refinement a symbol
#: photographed at an angle samples progressively off-centre towards the far
#: corner, which is where every module of a large version is.
fn sample_grid(
    values: list[float],
    rows: int,
    cols: int,
    corner: Finder,
    top: Finder,
    side_finder: Finder,
    dimension: int
) -> list[list[bool]]? {
    let edge = float(dimension) - 3.5
    var source = [Point2f(3.5, 3.5), Point2f(edge, 3.5), Point2f(3.5, edge), Point2f(edge, edge)]
    var target = [
        Point2f(corner.x, corner.y),
        Point2f(top.x, top.y),
        Point2f(side_finder.x, side_finder.y),
        Point2f(top.x + side_finder.x - corner.x, top.y + side_finder.y - corner.y),
    ]

    #: The alignment pattern nearest the far corner, if this version has one.
    #: It is looked for where the three-finder estimate says it should be --
    #: mapped through that estimate rather than measured from the corner --
    #: and only within a couple of modules of there. A wider search finds a
    #: data module that happens to look like one, and a wrong fourth point is
    #: worse than none: it warps every sample towards the far corner, where
    #: most of a large symbol's modules are.
    let centres = alignment_centres((dimension - 17) // 4)
    if centres.len() >= 2 {
        let centre_module = float(centres[centres.len() - 1]) + 0.5
        let rough = get_perspective_transform(source, target)
        let guessed = perspective_transform([Point2f(centre_module, centre_module)], rough)[0]
        let scale = distance(target[0].y, target[0].x, target[1].y, target[1].x) / (edge - 3.5)
        let refined = refine_alignment(values, rows, cols, guessed, scale)
        if refined is not null {
            source[3] = Point2f(centre_module, centre_module)
            target[3] = refined
        }
    }

    let matrix = get_perspective_transform(source, target)
    var probes: list[Point2f] = []
    for y in 0..dimension {
        for x in 0..dimension { probes.push(Point2f(float(x) + 0.5, float(y) + 0.5)) }
    }
    let mapped = perspective_transform(probes, matrix)

    #: A module is decided by a vote over the pixels around its centre, not by
    #: the one pixel there. One flipped pixel at a centre corrupts a whole
    #: module, and a module is the unit the parity budget is spent in: at two
    #: per cent salt-and-pepper noise a single-pixel sample lost every symbol
    #: below level M, and a vote over a third of a module reads them all.
    let unit = distance(target[0].y, target[0].x, target[1].y, target[1].x) / (edge - 3.5)
    let reach = if unit >= 6.0 { 2 } elif unit >= 3.0 { 1 } else { 0 }

    var grid: list[list[bool]] = []
    var at = 0
    for _y in 0..dimension {
        var row: list[bool] = []
        for _x in 0..dimension {
            let point = mapped[at]
            at += 1
            let py = int(point.y + 0.5)
            let px = int(point.x + 0.5)
            if py < 0 or py >= rows or px < 0 or px >= cols { return null }
            var dark = 0
            var seen = 0
            for dy in -reach..reach + 1 {
                let sy = py + dy
                if sy < 0 or sy >= rows { continue }
                for dx in -reach..reach + 1 {
                    let sx = px + dx
                    if sx < 0 or sx >= cols { continue }
                    seen += 1
                    if values[sy * cols + sx] < 128.0 { dark += 1 }
                }
            }
            row.push(dark * 2 > seen)
        }
        grid.push(row)
    }
    return grid
}

#: The centre of the alignment pattern nearest `guess`, or null.
fn refine_alignment(
    values: list[float],
    rows: int,
    cols: int,
    guess: Point2f,
    unit: float
) -> Point2f? {
    if unit < 1.0 { return null }
    let reach = int(unit * 2.0) + 2
    let cy = int(guess.y + 0.5)
    let cx = int(guess.x + 0.5)
    var best: Point2f? = null
    var best_distance = 0.0

    for dy in -reach..reach + 1 {
        let y = cy + dy
        if y < 1 or y >= rows - 1 { continue }
        for dx in -reach..reach + 1 {
            let x = cx + dx
            if x < 1 or x >= cols - 1 { continue }
            if values[y * cols + x] >= 128.0 { continue }
            #: A one-unit dark centre inside a one-unit light ring: the
            #: pixel is dark, its neighbours a unit away are light, and its
            #: neighbours two modules away are dark again.
            let step = int(unit + 0.5)
            if step < 1 { continue }
            if y - 2 * step < 0 or y + 2 * step >= rows { continue }
            if x - 2 * step < 0 or x + 2 * step >= cols { continue }
            if values[(y - step) * cols + x] < 128.0 { continue }
            if values[(y + step) * cols + x] < 128.0 { continue }
            if values[y * cols + x - step] < 128.0 { continue }
            if values[y * cols + x + step] < 128.0 { continue }
            if values[(y - 2 * step) * cols + x] >= 128.0 { continue }
            if values[(y + 2 * step) * cols + x] >= 128.0 { continue }
            if values[y * cols + x - 2 * step] >= 128.0 { continue }
            if values[y * cols + x + 2 * step] >= 128.0 { continue }
            let away = distance(float(y), float(x), guess.y, guess.x)
            if best is null or away < best_distance {
                best = Point2f(float(x), float(y))
                best_distance = away
            }
        }
    }
    return best
}

#: How many modules across a symbol is, from the distance between its finder
#: centres. Snapped to the nearest legal dimension, since only sizes one more
#: than a multiple of four exist.
fn dimension_between(corner: Finder, top: Finder, side_finder: Finder) -> int {
    let across = distance(corner.y, corner.x, top.y, top.x)
    let down = distance(corner.y, corner.x, side_finder.y, side_finder.x)
    let unit = (corner.width + top.width + side_finder.width) / 3.0
    if unit <= 0.0 { return 0 }
    let estimate = (across + down) / 2.0 / unit + 7.0
    var dimension = int(estimate + 0.5)
    let remainder = (dimension - 17) % 4
    if remainder == 1 { dimension -= 1 }
    if remainder == 2 { dimension += 2 }
    if remainder == 3 { dimension += 1 }
    return dimension
}

#: Find and read QR codes, with OpenCV's method names.
#:
#: ```
#: let detector = QRCodeDetector()
#: let (text, corners) = detector.detect_and_decode(frame)
#: ```
pub class QRCodeDetector {
    #: How far the three finder centres may be from a right isoceles triangle,
    #: as a fraction of the shorter side, before the triple is refused.
    pub var tolerance: float

    pub fn init(self, tolerance: float = 0.4) { self.tolerance = tolerance }

    fn prepared(self, image: Mat) -> tuple[list[float], int, int] {
        image.flush()
        let gray = if image.channels() == 1 {
            image
        } else {
            var values: list[float] = []
            let flat = image.to_list()
            let cn = image.channels()
            for index in 0..image.rows * image.cols {
                let b = flat[index * cn]
                let g = flat[index * cn + 1]
                let r = flat[index * cn + 2]
                values.push(0.114 * b + 0.587 * g + 0.299 * r)
            }
            Mat.from_list(image.rows, image.cols, CV_MAKETYPE(CV_8U, 1), values)
        }
        gray.flush()
        let level = otsu_threshold(gray)
        let raw = gray.to_list()
        var binary: list[float] = []
        #: Rewritten to a hard nought or two hundred and fifty-five so every
        #: test below is the same comparison against the midpoint, whatever the
        #: picture's own contrast was.
        for value in raw { binary.push(if value <= level { 0.0 } else { 255.0 }) }
        return (binary, gray.rows, gray.cols)
    }

    #: Every symbol in the image, as its four corners in the order OpenCV
    #: reports them: top-left, top-right, bottom-right, bottom-left.
    pub fn detect_multi(self, image: Mat) -> list[list[Point2f]] {
        var out: list[list[Point2f]] = []
        for found in self.read(image, false) { out.push(found[1]) }
        return out
    }

    #: The corners of the first symbol found, or null.
    pub fn detect(self, image: Mat) -> list[Point2f]? {
        let all = self.detect_multi(image)
        if all.len() == 0 { return null }
        return all[0]
    }

    #: Every symbol in the image, decoded.
    pub fn detect_and_decode_multi(self, image: Mat) -> tuple[list[str], list[list[Point2f]]] {
        var texts: list[str] = []
        var corners: list[list[Point2f]] = []
        for found in self.read(image, true) {
            let code = found[0]
            if code is null { continue }
            texts.push(code.text)
            corners.push(found[1])
        }
        return (texts, corners)
    }

    #: The first symbol's text and corners; the text is empty when nothing was
    #: found or nothing could be read, which is what OpenCV does too.
    pub fn detect_and_decode(self, image: Mat) -> tuple[str, list[Point2f]] {
        for found in self.read(image, true) {
            let code = found[0]
            if code is not null { return (code.text, found[1]) }
        }
        return ("", [])
    }

    #: Everything the reader learned about one symbol, for a caller that wants
    #: the version and correction level as well as the text.
    pub fn read_codes(self, image: Mat) -> list[QRCode] {
        var out: list[QRCode] = []
        for found in self.read(image, true) {
            let code = found[0]
            if code is not null { out.push(code) }
        }
        return out
    }

    fn read(self, image: Mat, decode: bool) -> list[tuple[QRCode?, list[Point2f]]] {
        let (values, rows, cols) = self.prepared(image)
        let finders = find_finders(values, rows, cols)
        var out: list[tuple[QRCode?, list[Point2f]]] = []
        if finders.len() < 3 { return out }

        var used: list[bool] = []
        for _index in 0..finders.len() { used.push(false) }

        for a in 0..finders.len() {
            for b in a + 1..finders.len() {
                for c in b + 1..finders.len() {
                    if used[a] or used[b] or used[c] { continue }
                    let (corner, top, side_finder) = order_finders(
                        finders[a],
                        finders[b],
                        finders[c]
                    )
                    let across = distance(corner.y, corner.x, top.y, top.x)
                    let down = distance(corner.y, corner.x, side_finder.y, side_finder.x)
                    if across <= 0.0 or down <= 0.0 { continue }
                    #: Three finders of one symbol are the same size. Checking
                    #: that before the geometry costs three comparisons and
                    #: throws out most triples, which each cost a sampling of
                    #: the whole symbol otherwise.
                    let widest = max(corner.width, max(top.width, side_finder.width))
                    let narrowest = min(corner.width, min(top.width, side_finder.width))
                    if widest > narrowest * 2.0 { continue }
                    let shorter = min(across, down)
                    #: The two sides of a symbol are equal and meet at a right
                    #: angle. Both are checked, because three finders in a row
                    #: pass the first test on their own.
                    if math.fabs(across - down) > shorter * self.tolerance { continue }
                    let diagonal = distance(top.y, top.x, side_finder.y, side_finder.x)
                    let wanted = math.sqrt(across * across + down * down)
                    if math.fabs(diagonal - wanted) > shorter * self.tolerance { continue }

                    let estimate = dimension_between(corner, top, side_finder)
                    if estimate < 21 or estimate > 177 { continue }
                    #: The module width is measured from runs crossing the
                    #: finders, and a symbol photographed at an angle makes
                    #: every such run wider than a module by the secant of the
                    #: angle -- which is enough to name the version one step
                    #: wrong. Rather than correct the width, the two
                    #: neighbouring sizes are tried as well: a wrong dimension
                    #: fails the format word or the parity, so accepting the
                    #: first that reads is as safe as getting it right first
                    #: time and costs two more samplings on a symbol that would
                    #: otherwise not be read at all.
                    var code: QRCode? = null
                    var dimension = 0
                    for offset in [0, 4, -4, 8, -8] {
                        let trying = estimate + offset
                        if trying < 21 or trying > 177 { continue }
                        let grid = sample_grid(values, rows, cols, corner, top, side_finder, trying)
                        if grid is null { continue }
                        if not decode {
                            dimension = trying
                            break
                        }
                        code = decode_grid(grid)
                        if code is not null {
                            dimension = trying
                            break
                        }
                    }
                    if dimension == 0 { continue }
                    if decode and code is null { continue }

                    let far = Point2f(
                        top.x + side_finder.x - corner.x,
                        top.y + side_finder.y - corner.y
                    )
                    out.push(
                        (
                            code,
                            [
                                Point2f(corner.x, corner.y),
                                Point2f(top.x, top.y),
                                far,
                                Point2f(side_finder.x, side_finder.y),
                            ]
                        )
                    )
                    used[a] = true
                    used[b] = true
                    used[c] = true
                }
            }
        }
        return out
    }
}

#: Render a symbol as an image, `scale` pixels a module with a four-module
#: quiet zone, which is what the standard asks for and what a reader expects.
pub fn qr_encode(text: str, level: int = 0, version: int = 0, scale: int = 4) -> Mat {
    if scale < 1 { throw ValueError("a module has to be at least one pixel across") }
    let grid = encode_grid(text, level, version)
    let modules = grid.len()
    let quiet = 4
    let side = (modules + quiet * 2) * scale
    var values: list[float] = []
    for y in 0..side {
        for x in 0..side {
            let my = y // scale - quiet
            let mx = x // scale - quiet
            let dark = my >= 0 and my < modules and mx >= 0 and mx < modules and grid[my][mx]
            values.push(if dark { 0.0 } else { 255.0 })
        }
    }
    return Mat.from_list(side, side, CV_MAKETYPE(CV_8U, 1), values)
}
