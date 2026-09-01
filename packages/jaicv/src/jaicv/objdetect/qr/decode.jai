#: Reading a module grid: unmask, de-interleave, correct, and decode the stream.
from ..galois import rs_correct
from .symbol import decode_format, decode_version, is_function_module, mask_bit
from .symbol import version_size
from .tables import BLOCK_COUNT, EC_PER_BLOCK, LEVEL_NAMES, TOTAL_CODEWORDS

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
pub fn count_bits(mode: int, version: int) -> int {
    let band = if version <= 9 { 0 } elif version <= 26 { 1 } else { 2 }
    if mode == 1 { return [10, 12, 14][band] }
    if mode == 2 { return [9, 11, 13][band] }
    if mode == 4 { return [8, 16, 16][band] }
    if mode == 8 { return [8, 10, 12][band] }
    return 0
}

pub let ALPHANUMERIC = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"

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
