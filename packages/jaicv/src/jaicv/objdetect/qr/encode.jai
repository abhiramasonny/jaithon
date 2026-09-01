#: Writing a module grid, and rendering one as an image.
import std.math
from ...core.mat import Mat
from ...core.types import CV_8U, CV_MAKETYPE
from ..galois import gf_exp, gf_mul
from .decode import ALPHANUMERIC, count_bits
from .symbol import alignment_centres, function_map, mask_bit, version_size
from .tables import BLOCK_COUNT, EC_PER_BLOCK, LEVEL_NAMES, LEVEL_OF_BITS
from .tables import TOTAL_CODEWORDS

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
