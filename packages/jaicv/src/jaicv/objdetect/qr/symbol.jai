#: The symbol's fixed layout: the function patterns, the mask, and the two words
#: that say how the rest is stored.
from .tables import LEVEL_OF_BITS

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
