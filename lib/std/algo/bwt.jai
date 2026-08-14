#: The Burrows-Wheeler transform, and the run-length pair it exists to feed.
#:
#: `bwt_encode` sorts equal characters next to each other; `rle_encode` then
#: shortens the runs that creates. Pair the result with `std.algo.huffman` for
#: the rest of the pipeline — that is bzip2.
from std.algo.sort import sort_by, natural_order

#: Collapse `text` into (character, run length) pairs.
#:
#: O(n). Runs come back as pairs rather than as a packed string because a
#: packed string cannot be read back unambiguously once the text itself
#: contains digits. Pair it with `bwt_encode`, which is what makes long runs
#: appear in ordinary text in the first place.
#:
#: ```
#: rle_encode("aaabbc")   # [("a", 3), ("b", 2), ("c", 1)]
#: ```
pub fn rle_encode(text: str) -> list[tuple[str, int]] {
    let chars = _chars(text)
    let runs: list[tuple[str, int]] = []
    var start = 0
    while start < chars.len() {
        var end = start
        while end < chars.len() and chars[end] == chars[start] { end += 1 }
        runs.push((chars[start], end - start))
        start = end
    }
    return runs
}

#: Expand (character, run length) pairs back into a string.
#:
#: Raises `ValueError` on a symbol that is not exactly one character, or on a
#: run length below 1.
#:
#: ```
#: rle_decode([("a", 3), ("b", 2)])   # "aaabb"
#: ```
pub fn rle_decode(runs: list[tuple[str, int]]) -> str {
    let out: list[str] = []
    for run in runs {
        let symbol = run[0]
        let count = run[1]
        if symbol.len() != 1 {
            throw ValueError(f"run symbol must be one character, got '{symbol}'")
        }
        if count < 1 { throw ValueError(f"run length must be positive, got {count}") }
        for _ in 0..count { out.push(symbol) }
    }
    return "".join(out)
}

#: Return the Burrows-Wheeler transform of `text` and the row of the original.
#:
#: The last column of the sorted table of all rotations of `text`. Characters
#: that share a following context end up next to each other, which is what
#: makes `rle_encode` and then `HuffmanCode` pay off — this is the front of the
#: bzip2 pipeline. The transform is reversible from the returned row index
#: alone, so no sentinel character has to be reserved.
#:
#: The rotations are ordered by prefix doubling, O(n log^2 n), rather than by
#: comparing rotations character by character, O(n^2 log n).
#:
#: ```
#: bwt_encode("banana")   # ("nnbaaa", 3)
#: ```
pub fn bwt_encode(text: str) -> tuple[str, int] {
    let chars = _chars(text)
    let n = chars.len()
    if n == 0 { return ("", 0) }
    let order = _rotation_order(_codes(text))
    let column: list[str] = []
    var row = 0
    for (position, start) in order.enumerate() {
        column.push(chars[(start + n - 1) % n])
        if start == 0 { row = position }
    }
    return ("".join(column), row)
}

#: Invert `bwt_encode`, given the last column and the original's row.
#:
#: O(n): one pass to rank the column and one to walk the LF mapping backwards.
#: Raises `ValueError` when `row` is out of range.
#:
#: ```
#: bwt_decode("nnbaaa", 3)   # "banana"
#: ```
pub fn bwt_decode(last_column: str, row: int) -> str {
    let chars = _chars(last_column)
    let n = chars.len()
    if n == 0 {
        if row != 0 { throw ValueError(f"row must be 0 for an empty column, got {row}") }
        return ""
    }
    if row < 0 or row >= n { throw ValueError(f"row must be in 0..{n}, got {row}") }
    let counts: dict[str, int] = {}
    let rank = [0 for _ in 0..n]
    for i in 0..n {
        let ch = chars[i]
        rank[i] = counts.get(ch, 0)
        counts[ch] = rank[i] + 1
    }
    let first: dict[str, int] = {}
    var running = 0
    for symbol in sort_by(counts.keys(), natural_order) {
        first[symbol] = running
        running += counts[symbol]
    }
    let out = ["" for _ in 0..n]
    var p = row
    var index = n - 1
    while index >= 0 {
        out[index] = chars[p]
        p = first[chars[p]] + rank[p]
        index -= 1
    }
    return "".join(out)
}

fn _chars(text: str) -> list[str] {
    let out: list[str] = []
    for ch in text { out.push(ch) }
    return out
}

fn _codes(text: str) -> list[int] {
    let out: list[int] = []
    for ch in text { out.push(ord(ch)) }
    return out
}

#: Order the rotations of `codes` by prefix doubling.
#:
#: Round k orders every rotation by its first 2^k characters, using the ranks
#: from round k - 1 as the two halves of a pair. Each round is therefore one
#: sort of integer pairs rather than a character-by-character comparison, and
#: log n rounds suffice: O(n log^2 n).
fn _rotation_order(codes: list[int]) -> list[int] {
    let n = codes.len()
    var order = [index for index in 0..n]
    var rank = [code for code in codes]
    var step = 1
    while step < n {
        let current = rank
        let shift = step
        order = sort_by(order, |a, b| _by_rank_pair(current, shift, n, a, b))
        let updated = [0 for _ in 0..n]
        var index = 1
        while index < n {
            let earlier = order[index - 1]
            let here = order[index]
            let tied = _by_rank_pair(current, shift, n, earlier, here) == 0
            updated[here] = updated[earlier] + (tied ? 0 : 1)
            index += 1
        }
        rank = updated
        if rank[order[n - 1]] == n - 1 { break }
        step = step * 2
    }
    return order
}

fn _by_rank_pair(rank: list[int], shift: int, n: int, a: int, b: int) -> int {
    if rank[a] != rank[b] { return rank[a] < rank[b] ? -1 : 1 }
    let left = rank[(a + shift) % n]
    let right = rank[(b + shift) % n]
    if left != right { return left < right ? -1 : 1 }
    return 0
}
