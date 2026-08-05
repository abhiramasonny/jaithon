#: String algorithms that need a real construction rather than a scan.
#:
#: `std.str` covers what a program does to one string at a time — padding,
#: case, distance, escaping. This module holds the algorithms that build an
#: index, a code or a transform first, and it is where the linear-time
#: classics live:
#:
#:   palindrome_radii / longest_palindrome   Manacher, O(n).
#:   z_function / prefix_function            The two linear scans the rest of
#:                                           string matching is built from.
#:   SuffixAutomaton                         Every substring of a text, as a
#:                                           DFA with O(n) states.
#:   minimal_rotation                        Booth, O(n).
#:   diff                                    Myers, O((n + m) * d).
#:
#: The compression half is a working pipeline rather than a museum:
#: `bwt_encode` sorts equal characters next to each other, `rle_encode`
#: shortens the runs that creates, and `HuffmanCode` codes what is left —
#: which is bzip2. `lz77_compress` is the other classic answer to the same
#: problem. `base64_encode` and `hex_encode` then move the result through a
#: text-only channel, and `crc32` and `fnv1a` say whether it arrived intact.
#:
#: Byte-oriented functions take and return `list[int]` whose elements are all
#: in 0..=255; `utf8_encode` and `utf8_decode` convert between those and `str`.
#: Everything else treats a string as a sequence of Unicode scalar values, so
#: indices and lengths agree with `str.len()`.
#:
#: See `std.ds.text` for the indexes built by sorting instead: a trie, a suffix
#: array and a rolling hash.
from std.algo.sort import sort_by, natural_order

# Shortest match LZ77 will encode as a back-reference: a two-character match
# costs more to store than the literals it replaces.
const MIN_MATCH = 3

# Back-references examined per position. Bounding the hash chain is what keeps
# compression linear on input like "aaaa..." where every position is a
# candidate; raising it buys a slightly better ratio for a lot of time.
const MAX_CHAIN = 64

const BASE64_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
const HEX_LOWER = "0123456789abcdef"
const HEX_UPPER = "0123456789ABCDEF"
const CRC32_POLYNOMIAL = 0xEDB88320
const FNV_OFFSET_BASIS = 2166136261
const FNV_PRIME = 16777619

let _BASE64_VALUES: dict[str, int] = {digit: value for (digit, value) in zip(BASE64_ALPHABET, range(64))}

let _SOUNDEX_DIGITS: dict[str, str] = {
    "B": "1",
    "F": "1",
    "P": "1",
    "V": "1",
    "C": "2",
    "G": "2",
    "J": "2",
    "K": "2",
    "Q": "2",
    "S": "2",
    "X": "2",
    "Z": "2",
    "D": "3",
    "T": "3",
    "L": "4",
    "M": "5",
    "N": "5",
    "R": "6",
}

#: One step of an edit script, as produced by `diff`.
#:
#: `before_index` indexes the old sequence and `after_index` the new one. An
#: `Insert` names only the new position, a `Delete` only the old one, and a
#: `Keep` both — which is what lets a renderer print two aligned columns.
pub enum DiffOp {
    #: `line` is common to both sides and survives the edit untouched.
    Keep(before_index: int, after_index: int, line: str),
    #: `line` sits at `before_index` in the old sequence and is dropped. A
    #: line that changed is reported as a `Delete` and then an `Insert`, never
    #: as one replacement, so a renderer that wants to pair them must do so
    #: itself.
    Delete(before_index: int, line: str),
    #: `line` is new and lands at `after_index` in the new sequence. It has no
    #: counterpart in the old one, which is why there is no `before_index`.
    Insert(after_index: int, line: str),
}

#: A single LZ77 instruction: copy, then emit one literal.
#:
#: Copy `length` characters starting `distance` characters back from the end of
#: the output, then append `literal`. `length` is 0 when no back-reference was
#: found, and `literal` is empty when the input ended on a match. `distance`
#: may be smaller than `length`: the copy is character by character, so
#: `LZ77Token(1, 5, "")` repeats the previous character five times.
pub class LZ77Token {
    #: How far back the copy starts, in characters; 0 when there is no copy.
    pub let distance: int
    #: How many characters to copy; 0 when there is no copy.
    pub let length: int
    #: The character to append after the copy; empty at end of input.
    pub let literal: str

    #: Record one copy-then-literal step.
    pub fn init(self, distance: int, length: int, literal: str) {
        self.distance = distance
        self.length = length
        self.literal = literal
    }

    fn __str__(self) -> str {
        return f"LZ77Token({self.distance}, {self.length}, \"{self.literal}\")"
    }
}

#: Every substring of a text, as a deterministic automaton with O(n) states.
#:
#: Built by Blumer's online construction, one character at a time, in O(n) for
#: a fixed alphabet. A walk from the start state spells a substring, so
#: `contains` costs the length of the pattern with no preprocessing of the
#: pattern at all — the mirror image of KMP, and the right structure when one
#: text is searched for many patterns.
#:
#: The suffix links form a tree whose subtree sizes are occurrence counts,
#: which is what `count` uses; the state lengths give the number of distinct
#: substrings without enumerating them.
#:
#: ```
#: let sam = SuffixAutomaton("banana")
#: sam.contains("nan")                      # true
#: sam.count("ana")                         # 2
#: sam.distinct_substrings()                # 15
#: sam.longest_common_substring("ananas")   # "anana"
#: ```
pub class SuffixAutomaton {
    var transitions: list[dict[str, int]]
    var link: list[int]
    var length: list[int]
    # 1 for a state a character created, 0 for a clone: the occurrences a state
    # owns before anything is summed up the suffix-link tree.
    var owned: list[int]
    var counts: list[int]
    var last: int
    var counted: bool

    #: Build an automaton over `text`; the default builds an empty one to
    #: `extend`.
    pub fn init(self, text: str = "") {
        self.transitions = [{}]
        self.link = [-1]
        self.length = [0]
        self.owned = [0]
        self.counts = [0]
        self.last = 0
        self.counted = true
        for ch in text { self.extend(ch) }
    }

    #: Append one character to the text, in O(1) amortised time.
    pub fn extend(self, ch: str) -> void {
        if ch.len() != 1 {
            throw ValueError(f"extend takes one character, got {ch.len()}")
        }
        let current = self._new_state(self.length[self.last] + 1, 1)
        var p = self.last
        while p != -1 and ch not in self.transitions[p] {
            self.transitions[p][ch] = current
            p = self.link[p]
        }
        if p == -1 {
            self.link[current] = 0
        } else {
            let q = self.transitions[p][ch]
            if self.length[p] + 1 == self.length[q] {
                self.link[current] = q
            } else {
                # `q` stands for occurrences both longer and shorter than
                # length[p] + 1, and only the shorter ones may follow `p`. It
                # splits into a clone carrying those and the original keeping
                # the rest.
                let clone = self._new_state(self.length[p] + 1, 0)
                self.transitions[clone] = {k: v for (k, v) in self.transitions[q].items()}
                self.link[clone] = self.link[q]
                while p != -1 and self.transitions[p].get(ch, -1) == q {
                    self.transitions[p][ch] = clone
                    p = self.link[p]
                }
                self.link[q] = clone
                self.link[current] = clone
            }
        }
        self.last = current
        self.counted = false
    }

    #: Whether `pattern` occurs in the text. O(pattern length).
    pub fn contains(self, pattern: str) -> bool {
        return self._walk(pattern) >= 0
    }

    #: Number of occurrences of `pattern`, overlaps included.
    #:
    #: O(pattern length), after an O(n) propagation on the first call following
    #: a change. The empty string is reported as 0 rather than n + 1.
    pub fn count(self, pattern: str) -> int {
        if pattern.len() == 0 { return 0 }
        let state = self._walk(pattern)
        if state < 0 { return 0 }
        self._ensure_counts()
        return self.counts[state]
    }

    #: Number of distinct non-empty substrings of the text.
    #:
    #: Each state spells `length[state] - length[link[state]]` distinct
    #: substrings, so this is one O(n) sum rather than an enumeration.
    pub fn distinct_substrings(self) -> int {
        var total = 0
        for state in 1..self.link.len() {
            total += self.length[state] - self.length[self.link[state]]
        }
        return total
    }

    #: Return the longest string occurring in both the text and `other`.
    #:
    #: O(other.len()): walk `other` through the automaton, following suffix
    #: links whenever the next character has no transition, which shortens the
    #: current match by exactly as much as it must. Ties resolve to the
    #: earliest run in `other`.
    pub fn longest_common_substring(self, other: str) -> str {
        var state = 0
        var length = 0
        var best = 0
        var best_end = 0
        for (position, ch) in _chars(other).enumerate() {
            while state != 0 and ch not in self.transitions[state] {
                state = self.link[state]
                length = self.length[state]
            }
            if ch in self.transitions[state] {
                state = self.transitions[state][ch]
                length += 1
            } else {
                state = 0
                length = 0
            }
            if length > best {
                best = length
                best_end = position + 1
            }
        }
        return other[best_end - best:best_end]
    }

    fn _new_state(self, length: int, owned: int) -> int {
        self.transitions.push({})
        self.link.push(-1)
        self.length.push(length)
        self.owned.push(owned)
        self.counts.push(owned)
        return self.link.len() - 1
    }

    fn _walk(self, pattern: str) -> int {
        var state = 0
        for ch in pattern {
            if ch not in self.transitions[state] { return -1 }
            state = self.transitions[state][ch]
        }
        return state
    }

    #: Sum occurrence counts up the suffix-link tree.
    #:
    #: A state occurs wherever its children in that tree do, and a child is
    #: always longer than its parent. Ordering the states by length with a
    #: counting sort therefore puts every child before its parent, which is all
    #: the ordering the sum needs — no recursion over the tree.
    #:
    #: The sum starts from `owned` rather than from the previous totals, so
    #: extending the text after counting stays correct.
    fn _ensure_counts(self) -> void {
        if self.counted { return }
        let states = self.link.len()
        self.counts = [value for value in self.owned]
        var longest = 0
        for state in 0..states {
            if self.length[state] > longest { longest = self.length[state] }
        }
        let buckets = [0 for _ in 0..=longest]
        for state in 0..states { buckets[self.length[state]] += 1 }
        var running = 0
        for value in 0..=longest {
            let occurrences = buckets[value]
            buckets[value] = running
            running += occurrences
        }
        let order = [0 for _ in 0..states]
        for state in 0..states {
            order[buckets[self.length[state]]] = state
            buckets[self.length[state]] += 1
        }
        var index = states - 1
        while index > 0 {
            let state = order[index]
            self.counts[self.link[state]] += self.counts[state]
            index -= 1
        }
        self.counted = true
    }

    fn __str__(self) -> str {
        return f"SuffixAutomaton({self.link.len()} states)"
    }
}

#: A canonical Huffman code: an optimal prefix code over a known alphabet.
#:
#: The code lengths come from the two-queue construction, which is O(k log k)
#: for the initial sort and O(k) after it — no heap, because merged weights
#: come out in non-decreasing order and so stay ordered in a plain queue. The
#: codes themselves are then assigned canonically: shortest first, and
#: alphabetically within one length. That makes the table reproducible across
#: runs and means a decoder needs only the code lengths, never the tree.
#:
#: ```
#: let code = HuffmanCode.from_text("abracadabra")
#: code.table()["a"]                         # "0"
#: code.encode("abracadabra")                # "01001110101011001001110"
#: code.decode(code.encode("abracadabra"))   # "abracadabra"
#: ```
pub class HuffmanCode {
    var codes: dict[str, str]
    # Indexed by code length: the numeric value of a code of that length maps
    # back to its symbol. Decoding is then one dict probe per bit.
    var by_length: list[dict[int, str]]

    #: Build the code for a symbol-to-frequency table.
    #:
    #: Raises `ValueError` on an empty table, a symbol that is not exactly one
    #: character, or a frequency below 1.
    pub fn init(self, frequencies: dict[str, int]) {
        self.codes = {}
        self.by_length = [{}]
        let alphabet = sort_by(frequencies.keys(), natural_order)
        if alphabet.len() == 0 { throw ValueError("a Huffman code needs at least one symbol") }
        for symbol in alphabet {
            if symbol.len() != 1 {
                throw ValueError(f"symbol must be one character, got '{symbol}'")
            }
            if frequencies[symbol] < 1 {
                throw ValueError(
                    f"frequency of '{symbol}' must be positive, got {frequencies[symbol]}"
                )
            }
        }
        let lengths = _huffman_lengths(alphabet, frequencies)
        # Canonical assignment: add one, then shift left by the growth in
        # length. The code stays prefix-free because a longer code always
        # extends a value one past every shorter one.
        var value = 0
        var previous = 0
        for symbol in sort_by(alphabet, |a, b| _by_length_then_symbol(lengths, a, b)) {
            let length = lengths[symbol]
            if previous > 0 { value = value + 1 << length - previous }
            previous = length
            self.codes[symbol] = _to_bits(value, length)
            while self.by_length.len() <= length { self.by_length.push({}) }
            self.by_length[length][value] = symbol
        }
    }

    #: Build the code from the character frequencies of `text`.
    pub static fn from_text(text: str) -> HuffmanCode {
        let frequencies: dict[str, int] = {}
        for ch in text { frequencies[ch] = frequencies.get(ch, 0) + 1 }
        return HuffmanCode(frequencies)
    }

    #: The symbol-to-code table, as a copy.
    pub fn table(self) -> dict[str, str] {
        return {symbol: bits for (symbol, bits) in self.codes.items()}
    }

    #: Encode `text` as a string of "0" and "1".
    #:
    #: Raises `KeyError` on a character the code was not built for.
    pub fn encode(self, text: str) -> str {
        let out: list[str] = []
        for ch in text {
            if ch not in self.codes { throw KeyError(f"no code for '{ch}'") }
            out.push(self.codes[ch])
        }
        return "".join(out)
    }

    #: Decode a string of "0" and "1" produced by `encode`.
    #:
    #: Raises `ValueError` on a character other than "0" or "1", or on trailing
    #: bits that complete no code.
    pub fn decode(self, bits: str) -> str {
        let out: list[str] = []
        var value = 0
        var length = 0
        for bit in bits {
            if bit != "0" and bit != "1" { throw ValueError(f"expected a bit, got '{bit}'") }
            value = value * 2 + (bit == "1" ? 1 : 0)
            length += 1
            # The code is prefix-free, so the first value that is a code is the
            # only one it can be: no lookahead and no backtracking.
            if length < self.by_length.len() and value in self.by_length[length] {
                out.push(self.by_length[length][value])
                value = 0
                length = 0
            }
        }
        if length > 0 {
            throw ValueError(f"{length} trailing bits complete no code")
        }
        return "".join(out)
    }

    fn __str__(self) -> str { return f"HuffmanCode({self.codes.len()} symbols)" }
}

#: Return the radius of the longest palindrome centred at every position.
#:
#: Manacher's algorithm, O(n). The result has `2 * text.len() + 1` entries, one
#: per character and one per gap: index `2 * i + 1` centres on character `i`
#: and an even index centres on the gap before character `index / 2`. The value
#: is the **length in the original string** of the palindrome centred there, so
#: odd-length palindromes sit at odd indices and even-length ones at even
#: indices.
#:
#: Every question about the palindromes of a string is a scan of this array:
#: the longest one, how many there are, whether a given slice is one.
#:
#: ```
#: palindrome_radii("aba")   # [0, 1, 0, 3, 0, 1, 0]
#: ```
pub fn palindrome_radii(text: str) -> list[int] {
    # Interleaving a separator that cannot occur in the input makes every
    # palindrome odd-length, which collapses the two cases into one.
    let scan: list[int] = [-1]
    for ch in text {
        scan.push(ord(ch))
        scan.push(-1)
    }
    let width = scan.len()
    let radii = [0 for _ in 0..width]
    var center = 0
    var right = 0
    for i in 0..width {
        # Inside the rightmost palindrome found so far, the mirror position
        # already answers this one — up to the point where it would reach past
        # the edge, which is the only part that has to be tested.
        if i < right { radii[i] = min(right - i, radii[2 * center - i]) }
        while i - radii[i] - 1 >= 0 and
            i + radii[i] + 1 < width and
            scan[i - radii[i] - 1] == scan[i + radii[i] + 1] {
            radii[i] += 1
        }
        if i + radii[i] > right {
            center = i
            right = i + radii[i]
        }
    }
    return radii
}

#: Return the longest palindromic substring of `text`, earliest on a tie.
#:
#: O(n) time and memory through `palindrome_radii`, against O(n^2) for
#: expanding around every centre.
#:
#: ```
#: longest_palindrome("forgeeksskeegfor")   # "geeksskeeg"
#: longest_palindrome("cbbd")               # "bb"
#: ```
pub fn longest_palindrome(text: str) -> str {
    if text.len() == 0 { return "" }
    let radii = palindrome_radii(text)
    var best = 0
    var best_center = 0
    for (center, radius) in radii.enumerate() {
        if radius > best {
            best = radius
            best_center = center
        }
    }
    let start = (best_center - best) // 2
    return text[start:start + best]
}

#: Return, for every position of `text`, the length of the longest prefix of
#: `text` that starts there.
#:
#: The Z-array, in O(n). `z[0]` is the whole length by convention. To find
#: every occurrence of a pattern, run this over `pattern + "\0" + text` and
#: look for entries equal to `pattern.len()`.
#:
#: ```
#: z_function("aabxaab")   # [7, 1, 0, 0, 3, 1, 0]
#: ```
pub fn z_function(text: str) -> list[int] {
    let chars = _chars(text)
    let n = chars.len()
    let z = [0 for _ in 0..n]
    if n == 0 { return z }
    z[0] = n
    # [left, right) is the match with the furthest right edge seen so far; a
    # position inside it starts from its mirror instead of from zero.
    var left = 0
    var right = 0
    for i in 1..n {
        if i < right { z[i] = min(right - i, z[i - left]) }
        while i + z[i] < n and chars[z[i]] == chars[i + z[i]] { z[i] += 1 }
        if i + z[i] > right {
            left = i
            right = i + z[i]
        }
    }
    return z
}

#: Return, for every prefix of `text`, the length of its longest proper border.
#:
#: A border is a string that is both a prefix and a suffix of the prefix ending
#: at that position. This is the KMP failure function, in O(n): after a
#: mismatch at position `i` of a pattern, matching resumes at
#: `prefix_function(pattern)[i - 1]` without re-reading a character of the
#: text.
#:
#: `n - border[n - 1]` is also the shortest period of the string, which is how
#: to test in linear time whether a string is a repetition of a shorter one.
#:
#: ```
#: prefix_function("abcabcd")   # [0, 0, 0, 1, 2, 3, 0]
#: prefix_function("aabaaab")   # [0, 1, 0, 1, 2, 2, 3]
#: ```
pub fn prefix_function(text: str) -> list[int] {
    let chars = _chars(text)
    let n = chars.len()
    let border = [0 for _ in 0..n]
    for i in 1..n {
        # Fall back through shorter and shorter borders until one extends.
        var length = border[i - 1]
        while length > 0 and chars[i] != chars[length] { length = border[length - 1] }
        if chars[i] == chars[length] { length += 1 }
        border[i] = length
    }
    return border
}

#: Return the starting index of the lexicographically smallest rotation.
#:
#: Booth's algorithm: one failure function over `text + text`, carrying the
#: best candidate start as it scans, in O(n) time and O(n) memory. Building all
#: n rotations and taking the minimum is O(n^2).
#:
#: ```
#: least_rotation_index("cabbage")   # 1
#: ```
pub fn least_rotation_index(text: str) -> int {
    let n = text.len()
    if n == 0 { return 0 }
    let codes = _codes(text + text)
    let doubled = codes.len()
    let failure = [-1 for _ in 0..doubled]
    var best = 0
    for j in 1..doubled {
        let here = codes[j]
        var i = failure[j - best - 1]
        while i != -1 and here != codes[best + i + 1] {
            # The candidate starting at `best` has just lost to one starting
            # inside the part already matched; that later start takes over.
            if here < codes[best + i + 1] { best = j - i - 1 }
            i = failure[i]
        }
        if here != codes[best + i + 1] {
            if here < codes[best] { best = j }
            failure[j - best] = -1
        } else {
            failure[j - best] = i + 1
        }
    }
    return best % n
}

#: Return the lexicographically smallest rotation of `text`.
#:
#: The canonical form of a cyclic string: two strings are rotations of each
#: other exactly when their minimal rotations are equal, so this is how to key
#: a cycle in a dict.
#:
#: ```
#: minimal_rotation("cabbage")                          # "abbagec"
#: minimal_rotation("bca") == minimal_rotation("cab")   # true
#: ```
pub fn minimal_rotation(text: str) -> str {
    let start = least_rotation_index(text)
    if start == 0 { return text }
    return text[start:text.len()] + text[0:start]
}

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
    # rank[i] counts the copies of chars[i] before position i. The i-th copy of
    # a character in the last column is also the i-th in the sorted first
    # column, and that correspondence is the whole of the LF mapping.
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

#: Compress `text` into LZ77 copy-then-literal tokens.
#:
#: Matches are found through a hash chain over three-character prefixes: `head`
#: holds the most recent position of each trigram and `previous` chains the
#: older ones, so a position only ever examines places that really do share
#: three characters. Each walk is capped at `MAX_CHAIN` links and at `window`
#: characters, which bounds the whole compression at O(n) for repetitive input
#: instead of O(n^2).
#:
#: `window` is how far back a copy may reach and `max_length` how long one may
#: be; both trade ratio against the size of an encoded token.
#:
#: ```
#: let tokens = lz77_compress("abcabcabc")
#: tokens.len()              # 4
#: lz77_decompress(tokens)   # "abcabcabc"
#: ```
pub fn lz77_compress(text: str, window: int = 4096, max_length: int = 18) -> list[LZ77Token] {
    if window < 1 { throw ValueError(f"window must be positive, got {window}") }
    if max_length < MIN_MATCH {
        throw ValueError(f"max_length must be at least {MIN_MATCH}, got {max_length}")
    }
    let chars = _chars(text)
    let n = chars.len()
    let tokens: list[LZ77Token] = []
    let head: dict[str, int] = {}
    let previous = [-1 for _ in 0..n]
    var position = 0
    while position < n {
        var best_length = 0
        var best_distance = 0
        if position + MIN_MATCH <= n {
            let limit = min(max_length, n - position)
            var candidate = head.get(_trigram(chars, position), -1)
            var probes = 0
            while candidate >= 0 and probes < MAX_CHAIN and position - candidate <= window {
                var length = 0
                while length < limit and chars[candidate + length] == chars[position + length] {
                    length += 1
                }
                if length > best_length {
                    best_length = length
                    best_distance = position - candidate
                    if length == limit { break }
                }
                candidate = previous[candidate]
                probes += 1
            }
        }
        if best_length < MIN_MATCH {
            best_length = 0
            best_distance = 0
        }
        # Index every position the token covers, the one the literal consumes
        # included, or a later match would never see inside this copy.
        let covered = min(position + max(best_length, 1), n)
        for j in position..covered {
            if j + MIN_MATCH <= n {
                let key = _trigram(chars, j)
                previous[j] = head.get(key, -1)
                head[key] = j
            }
        }
        let literal = position + best_length < n ? chars[position + best_length] : ""
        tokens.push(LZ77Token(best_distance, best_length, literal))
        position += best_length + 1
    }
    return tokens
}

#: Rebuild the text that `lz77_compress` produced these tokens from.
#:
#: O(output length). Raises `ValueError` on a distance reaching before the
#: start of the output, which is the one way a token stream can be malformed.
#:
#: ```
#: lz77_decompress([LZ77Token(0, 0, "a"), LZ77Token(1, 3, "")])   # "aaaa"
#: ```
pub fn lz77_decompress(tokens: list[LZ77Token]) -> str {
    let out: list[str] = []
    for token in tokens {
        if token.length > 0 {
            if token.distance < 1 or token.distance > out.len() {
                throw ValueError(f"distance must be in 1..={out.len()}, got {token.distance}")
            }
            let start = out.len() - token.distance
            # Reading through the tail as it grows is deliberate: that is how a
            # distance shorter than the length encodes a repeating run.
            for offset in 0..token.length { out.push(out[start + offset]) }
        }
        if token.literal.len() > 0 { out.push(token.literal) }
    }
    return "".join(out)
}

#: Encode `text` as UTF-8 bytes.
#:
#: ```
#: utf8_encode("hé")   # [104, 195, 169]
#: ```
pub fn utf8_encode(text: str) -> list[int] {
    let out: list[int] = []
    for ch in text {
        let code = ord(ch)
        if code < 0x80 {
            out.push(code)
        } elif code < 0x800 {
            out.push(0xC0 | code >> 6)
            out.push(0x80 | code & 0x3F)
        } elif code < 0x10000 {
            out.push(0xE0 | code >> 12)
            out.push(0x80 | code >> 6 & 0x3F)
            out.push(0x80 | code & 0x3F)
        } else {
            out.push(0xF0 | code >> 18)
            out.push(0x80 | code >> 12 & 0x3F)
            out.push(0x80 | code >> 6 & 0x3F)
            out.push(0x80 | code & 0x3F)
        }
    }
    return out
}

#: Decode UTF-8 bytes into a string.
#:
#: Raises `ValueError` on a truncated sequence or on a byte that can neither
#: start nor continue one, rather than substituting a replacement character: a
#: decoder that silently repairs its input hides the corruption that produced
#: it.
#:
#: ```
#: utf8_decode([104, 195, 169])   # "hé"
#: ```
pub fn utf8_decode(data: list[int]) -> str {
    _check_bytes(data)
    let out: list[str] = []
    let n = data.len()
    var index = 0
    while index < n {
        let lead = data[index]
        var code = 0
        var extra = 0
        if lead < 0x80 {
            code = lead
        } elif lead >= 0xF0 {
            code = lead & 0x07
            extra = 3
        } elif lead >= 0xE0 {
            code = lead & 0x0F
            extra = 2
        } elif lead >= 0xC0 {
            code = lead & 0x1F
            extra = 1
        } else {
            throw ValueError(f"byte {lead} at index {index} cannot start a sequence")
        }
        if index + extra >= n { throw ValueError(f"sequence at index {index} is truncated") }
        for step in 1..=extra {
            let continuation = data[index + step]
            if continuation < 0x80 or continuation > 0xBF {
                throw ValueError(
                    f"byte {continuation} at index {index + step} is not a continuation"
                )
            }
            code = code << 6 | continuation & 0x3F
        }
        out.push(chr(code))
        index += extra + 1
    }
    return "".join(out)
}

#: Encode bytes as base64, standard alphabet, padded with `=`.
#:
#: ```
#: base64_encode(utf8_encode("hello"))   # "aGVsbG8="
#: ```
pub fn base64_encode(data: list[int]) -> str {
    _check_bytes(data)
    let out: list[str] = []
    let n = data.len()
    var index = 0
    # Three bytes are 24 bits, which is exactly four six-bit digits.
    while index + 2 < n {
        let chunk = data[index] << 16 | data[index + 1] << 8 | data[index + 2]
        out.push(_base64_digit(chunk >> 18))
        out.push(_base64_digit(chunk >> 12))
        out.push(_base64_digit(chunk >> 6))
        out.push(_base64_digit(chunk))
        index += 3
    }
    let rest = n - index
    if rest == 1 {
        let chunk = data[index] << 16
        out.push(_base64_digit(chunk >> 18))
        out.push(_base64_digit(chunk >> 12))
        out.push("==")
    } elif rest == 2 {
        let chunk = data[index] << 16 | data[index + 1] << 8
        out.push(_base64_digit(chunk >> 18))
        out.push(_base64_digit(chunk >> 12))
        out.push(_base64_digit(chunk >> 6))
        out.push("=")
    }
    return "".join(out)
}

#: Decode base64 back into bytes.
#:
#: ASCII whitespace is skipped, so text wrapped at 76 columns decodes as it
#: stands, and everything from the first `=` on is ignored. Raises `ValueError`
#: on a character outside the alphabet or on a final group of one digit, which
#: encodes no whole byte.
#:
#: ```
#: utf8_decode(base64_decode("aGVsbG8="))   # "hello"
#: ```
pub fn base64_decode(text: str) -> list[int] {
    let values: list[int] = []
    for ch in text {
        if ch in " \t\n\r" { continue }
        if ch == "=" { break }
        if ch not in _BASE64_VALUES { throw ValueError(f"'{ch}' is not a base64 character") }
        values.push(_BASE64_VALUES[ch])
    }
    let out: list[int] = []
    var index = 0
    while index + 3 < values.len() {
        let chunk = values[index] << 18 |
            values[index + 1] << 12 |
            values[index + 2] << 6 |
            values[index + 3]
        out.push(chunk >> 16 & 0xFF)
        out.push(chunk >> 8 & 0xFF)
        out.push(chunk & 0xFF)
        index += 4
    }
    let rest = values.len() - index
    if rest == 1 { throw ValueError("base64 input ends with one leftover character") }
    if rest == 2 {
        let chunk = values[index] << 18 | values[index + 1] << 12
        out.push(chunk >> 16 & 0xFF)
    } elif rest == 3 {
        let chunk = values[index] << 18 | values[index + 1] << 12 | values[index + 2] << 6
        out.push(chunk >> 16 & 0xFF)
        out.push(chunk >> 8 & 0xFF)
    }
    return out
}

#: Encode bytes as hexadecimal, two characters per byte.
#:
#: ```
#: hex_encode([255, 0, 16])   # "ff0010"
#: hex_encode([255], true)    # "FF"
#: ```
pub fn hex_encode(data: list[int], upper: bool = false) -> str {
    _check_bytes(data)
    let digits = upper ? HEX_UPPER : HEX_LOWER
    let out: list[str] = []
    for value in data {
        let high = value >> 4
        let low = value & 0xF
        out.push(digits[high:high + 1])
        out.push(digits[low:low + 1])
    }
    return "".join(out)
}

#: Decode hexadecimal, in either case, back into bytes.
#:
#: Raises `ValueError` on an odd number of digits or on a non-hex character.
#:
#: ```
#: hex_decode("ff0010")   # [255, 0, 16]
#: ```
pub fn hex_decode(text: str) -> list[int] {
    if text.len() % 2 != 0 {
        throw ValueError(f"hex input must have an even length, got {text.len()}")
    }
    let chars = _chars(text)
    let out: list[int] = []
    var index = 0
    while index < chars.len() {
        let high = _hex_value(chars[index])
        let low = _hex_value(chars[index + 1])
        if high < 0 or low < 0 {
            throw ValueError(f"'{chars[index]}{chars[index + 1]}' is not a hex byte")
        }
        out.push(high << 4 | low)
        index += 2
    }
    return out
}

#: Return the CRC-32 checksum of `data`, as used by zip, gzip and PNG.
#:
#: The reflected IEEE 802.3 polynomial, computed bit by bit: eight shifts per
#: byte and no 256-entry table to build or carry. A CRC detects accidental
#: corruption, burst errors especially; it is not a hash, and forging a
#: collision for one is arithmetic, not work.
#:
#: ```
#: crc32(utf8_encode("hello"))   # 907060870
#: ```
pub fn crc32(data: list[int]) -> int {
    _check_bytes(data)
    var crc = 0xFFFFFFFF
    for byte in data {
        crc = crc ^ byte
        for _ in 0..8 {
            let feedback = crc & 1
            crc = crc >> 1
            if feedback == 1 { crc = crc ^ CRC32_POLYNOMIAL }
        }
    }
    return crc ^ 0xFFFFFFFF
}

#: Return the 32-bit FNV-1a hash of `data`.
#:
#: One xor and one multiply per byte, which makes it the cheapest hash worth
#: using for short keys: dict keys, string interning, bloom filters. Like every
#: non-cryptographic hash it is easy to find collisions for, so never use it on
#: input an adversary chooses.
#:
#: ```
#: fnv1a(utf8_encode("hello"))   # 1335831723
#: fnv1a([])                     # 2166136261
#: ```
pub fn fnv1a(data: list[int]) -> int {
    _check_bytes(data)
    var hash = FNV_OFFSET_BASIS
    for byte in data {
        hash = (hash ^ byte) * FNV_PRIME & 0xFFFFFFFF
    }
    return hash
}

#: Return the four-character Soundex key of `name`.
#:
#: An initial letter and three digits standing for the consonant sounds after
#: it, so that names spelled differently but said alike collide. Vowels, `H`
#: and `W` drop out, but only a vowel resets the rule that collapses a repeated
#: digit — which is why the `2` and `1` of `Ashcraft` survive the `H` between
#: them. Non-letters are ignored, and a name with no letters returns "".
#:
#: Soundex is coarse and English-specific. `metaphone` is the better key for
#: anything beyond the surname matching this was designed for in 1918.
#:
#: ```
#: soundex("Robert")     # "R163"
#: soundex("Rupert")     # "R163"
#: soundex("Ashcraft")   # "A261"
#: ```
pub fn soundex(name: str) -> str {
    let letters = _letters(name)
    if letters.len() == 0 { return "" }
    let out: list[str] = [letters[0]]
    var previous = _SOUNDEX_DIGITS.get(letters[0], "0")
    var index = 1
    while index < letters.len() and out.len() < 4 {
        let letter = letters[index]
        let digit = _SOUNDEX_DIGITS.get(letter, "0")
        if digit != "0" {
            if digit != previous { out.push(digit) }
            previous = digit
        } elif letter != "H" and letter != "W" {
            previous = "0"
        }
        index += 1
    }
    while out.len() < 4 { out.push("0") }
    return "".join(out)
}

#: Return the Metaphone key of `word`.
#:
#: Lawrence Philips' original 1990 rules: the word is rewritten as the
#: consonant sounds of English, with `0` standing for the "th" sound and `X`
#: for "sh". Unlike Soundex the key has no fixed length and keeps the order of
#: the sounds, so it discriminates far better on ordinary words. Vowels survive
#: only in first position, and silent letters — the `K` of `knight`, the `GH`
#: of `night`, the `B` of `lamb` — go by rule rather than by table.
#:
#: Non-letters are ignored. This is the original algorithm, not Double
#: Metaphone: one key per word, and tuned for English.
#:
#: ```
#: metaphone("phone")                          # "FN"
#: metaphone("Smith")                          # "SM0"
#: metaphone("knight") == metaphone("night")   # true, both "NT"
#: ```
pub fn metaphone(word: str) -> str {
    let letters = _letters(word)
    let n = letters.len()
    if n == 0 { return "" }
    let out: list[str] = []
    var index = 0
    # Where the word's first sound begins. Dropping a silent first letter moves
    # it, so that the `E` of "aero" is still an initial vowel and survives.
    var first_sound = 0
    # Silent first letters, which no later rule can catch because they are
    # silent only at the very start.
    let opening = n >= 2 ? letters[0] + letters[1] : ""
    if opening == "AE" or opening == "GN" or opening == "KN" or opening == "PN" or opening == "WR" {
        index = 1
        first_sound = 1
    } elif letters[0] == "X" {
        out.push("S")
        index = 1
    } elif opening == "WH" {
        out.push("W")
        index = 2
    }
    while index < n {
        let ch = letters[index]
        let next = _at(letters, index + 1)
        let after = _at(letters, index + 2)
        let before = _at(letters, index - 1)
        # A doubled letter says nothing twice, except CC as in "accident".
        if ch != "C" and before == ch {
            index += 1
            continue
        }
        if _is_vowel(ch) {
            if index == first_sound { out.push(ch) }
        } elif ch == "B" {
            if index != n - 1 or before != "M" { out.push("B") }
        } elif ch == "C" {
            if next == "I" and after == "A" {
                out.push("X")
            } elif next == "H" {
                out.push("X")
            } elif next == "I" or next == "E" or next == "Y" {
                # SC before a front vowel is a single S, as in "science".
                if before != "S" { out.push("S") }
            } else {
                out.push("K")
            }
        } elif ch == "D" {
            if next == "G" and (after == "E" or after == "Y" or after == "I") {
                out.push("J")
                index += 2
            } else {
                out.push("T")
            }
        } elif ch == "G" {
            if next == "H" {
                # GH speaks only at the end of a word or before a vowel.
                if index + 1 == n - 1 or _is_vowel(after) { out.push("K") } else { index += 1 }
            } elif next == "I" or next == "E" or next == "Y" {
                out.push("J")
            } elif next != "N" {
                out.push("K")
            }
        } elif ch == "H" {
            let after_vowel_only = _is_vowel(before) and not _is_vowel(next)
            let part_of_digraph = before == "C" or
                before == "S" or
                before == "P" or
                before == "T" or
                before == "G"
            if not after_vowel_only and not part_of_digraph { out.push("H") }
        } elif ch == "K" {
            if before != "C" { out.push("K") }
        } elif ch == "P" {
            if next == "H" {
                out.push("F")
                index += 1
            } else {
                out.push("P")
            }
        } elif ch == "Q" {
            out.push("K")
        } elif ch == "S" {
            if next == "H" {
                out.push("X")
                index += 1
            } elif next == "I" and (after == "O" or after == "A") {
                out.push("X")
            } else {
                out.push("S")
            }
        } elif ch == "T" {
            if next == "I" and (after == "O" or after == "A") {
                out.push("X")
            } elif next == "H" {
                out.push("0")
                index += 1
            } elif not (next == "C" and after == "H") {
                out.push("T")
            }
        } elif ch == "V" {
            out.push("F")
        } elif ch == "W" or ch == "Y" {
            # A semivowel is only heard when a vowel follows it.
            if _is_vowel(next) { out.push(ch) }
        } elif ch == "X" {
            out.push("KS")
        } elif ch == "Z" {
            out.push("S")
        } else {
            out.push(ch)
        }
        index += 1
    }
    return "".join(out)
}

#: Split `text` into word tokens.
#:
#: A token is a maximal run of letters and digits, keeping an apostrophe that
#: stands between two of them, so `don't` is one token. Case is preserved;
#: `word_frequency` folds it.
#:
#: ```
#: tokenize("Don't stop, it's fine!")   # ["Don't", "stop", "it's", "fine"]
#: ```
pub fn tokenize(text: str) -> list[str] {
    let chars = _chars(text)
    let tokens: list[str] = []
    let current: list[str] = []
    for (index, ch) in chars.enumerate() {
        if _is_token_char(ch) {
            current.push(ch)
        } elif ch == "'" and
            current.len() > 0 and
            index + 1 < chars.len() and
            _is_token_char(chars[index + 1]) {
            current.push(ch)
        } elif current.len() > 0 {
            tokens.push("".join(current))
            current.clear()
        }
    }
    if current.len() > 0 { tokens.push("".join(current)) }
    return tokens
}

#: Count how often each lower-cased token occurs in `text`.
#:
#: ```
#: word_frequency("the cat the hat")   # {"the": 2, "cat": 1, "hat": 1}
#: ```
pub fn word_frequency(text: str) -> dict[str, int] {
    let counts: dict[str, int] = {}
    for token in tokenize(text) {
        let word = token.lower()
        counts[word] = counts.get(word, 0) + 1
    }
    return counts
}

#: Return the `count` most frequent words of `text`, most frequent first.
#:
#: Ties break alphabetically, so the answer is the same on every run.
#:
#: ```
#: top_words("the cat the hat the bat", 2)   # [("the", 3), ("bat", 1)]
#: ```
pub fn top_words(text: str, count: int) -> list[tuple[str, int]] {
    if count < 0 { throw ValueError(f"count must be non-negative, got {count}") }
    let counts = word_frequency(text)
    let out: list[tuple[str, int]] = []
    for word in sort_by(counts.keys(), |a, b| _by_count_then_word(counts, a, b)) {
        if out.len() >= count { break }
        out.push((word, counts[word]))
    }
    return out
}

#: Return every contiguous run of `size` characters in `text`.
#:
#: Character n-grams are the standard feature for language detection and fuzzy
#: matching: two strings sharing many of them are similar however their words
#: are ordered.
#:
#: ```
#: char_ngrams("hello", 2)   # ["he", "el", "ll", "lo"]
#: ```
pub fn char_ngrams(text: str, size: int) -> list[str] {
    if size < 1 { throw ValueError(f"size must be positive, got {size}") }
    let out: list[str] = []
    var start = 0
    while start + size <= text.len() {
        out.push(text[start:start + size])
        start += 1
    }
    return out
}

#: Return every run of `size` consecutive tokens of `text`, space-joined.
#:
#: Joining makes an n-gram usable as a dict key, which is what a phrase count
#: or a language model needs. Tokens never contain a space, so `split(" ")`
#: takes one apart again.
#:
#: ```
#: word_ngrams("a b c d", 2)   # ["a b", "b c", "c d"]
#: ```
pub fn word_ngrams(text: str, size: int) -> list[str] {
    if size < 1 { throw ValueError(f"size must be positive, got {size}") }
    let tokens = tokenize(text)
    let out: list[str] = []
    var start = 0
    while start + size <= tokens.len() {
        out.push(" ".join(tokens[start:start + size]))
        start += 1
    }
    return out
}

#: Return a shortest edit script turning `before` into `after`.
#:
#: Myers' algorithm: O((n + m) * d) time for an edit distance of d, so the cost
#: follows how different the two sequences are rather than how long they are.
#: That is why it is the diff every version control system uses — two revisions
#: of a file are mostly the same file. The trace kept to rebuild the script
#: costs O((n + m) * d) memory.
#:
#: The script is in output order and holds exactly one operation per element of
#: each side: applying `Keep` and `Insert` in order rebuilds `after`, and the
#: number of `Delete` and `Insert` operations is the edit distance. A changed
#: line appears as a `Delete` followed by an `Insert`.
#:
#: ```
#: diff(["a", "b", "c"], ["a", "x", "c"])
#: # [DiffOp.Keep(0, 0, "a"), DiffOp.Delete(1, "b"), DiffOp.Insert(1, "x"),
#: #  DiffOp.Keep(2, 2, "c")]
#: ```
pub fn diff(before: list[str], after: list[str]) -> list[DiffOp] {
    let n = before.len()
    let m = after.len()
    let bound = n + m
    # Diagonal k = x - y runs from -bound to bound; one spare slot at each end
    # keeps the k - 1 and k + 1 probes in range without a bounds test.
    let offset = bound + 1
    let furthest = [0 for _ in 0..2 * bound + 3]
    let trace: list[list[int]] = []
    var found = false
    var depth = 0
    while depth <= bound and not found {
        trace.push([value for value in furthest])
        var k = -depth
        while k <= depth {
            # Step from the better of the two neighbouring diagonals — down is
            # an insertion, right is a deletion — then run the diagonal as far
            # as the two sides agree. Everything free is taken greedily, which
            # is what makes the whole edit script optimal.
            var x = 0
            if k == -depth or k != depth and furthest[k - 1 + offset] < furthest[k + 1 + offset] {
                x = furthest[k + 1 + offset]
            } else {
                x = furthest[k - 1 + offset] + 1
            }
            var y = x - k
            while x < n and y < m and before[x] == after[y] {
                x += 1
                y += 1
            }
            furthest[k + offset] = x
            if x >= n and y >= m {
                found = true
                break
            }
            k += 2
        }
        depth += 1
    }
    return _backtrack(before, after, trace, offset)
}

#: Diff two texts line by line.
#:
#: A trailing newline produces a trailing empty line, exactly as `split` does,
#: so the script round-trips through `"\n".join`.
#:
#: ```
#: diff_lines("a\nb", "a\nc").len()   # 3
#: ```
pub fn diff_lines(before: str, after: str) -> list[DiffOp] {
    return diff(before.split("\n"), after.split("\n"))
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

fn _letters(text: str) -> list[str] {
    let out: list[str] = []
    for ch in text.upper() {
        let code = ord(ch)
        if code >= 65 and code <= 90 { out.push(ch) }
    }
    return out
}

fn _at(letters: list[str], index: int) -> str {
    if index < 0 or index >= letters.len() { return "" }
    return letters[index]
}

# Guards against the empty string that `_at` returns past either end: "" is a
# substring of every string, so `"" in "AEIOU"` would be true.
fn _is_vowel(ch: str) -> bool {
    return ch.len() == 1 and ch in "AEIOU"
}

fn _is_token_char(ch: str) -> bool {
    let code = ord(ch)
    if code >= 48 and code <= 57 { return true }
    # A character with two cases is a letter; CJK scalars have one case but are
    # still words.
    if ch.lower() != ch.upper() { return true }
    return code >= 0x3041
}

fn _trigram(chars: list[str], position: int) -> str {
    return chars[position] + chars[position + 1] + chars[position + 2]
}

fn _hex_value(ch: str) -> int {
    let code = ord(ch)
    if code >= 48 and code <= 57 { return code - 48 }
    if code >= 97 and code <= 102 { return code - 87 }
    if code >= 65 and code <= 70 { return code - 55 }
    return -1
}

fn _base64_digit(value: int) -> str {
    let index = value & 0x3F
    return BASE64_ALPHABET[index:index + 1]
}

fn _check_bytes(data: list[int]) -> void {
    for (index, value) in data.enumerate() {
        if value < 0 or value > 255 {
            throw ValueError(f"byte at index {index} must be in 0..=255, got {value}")
        }
    }
}

fn _to_bits(value: int, width: int) -> str {
    let out: list[str] = []
    var position = width - 1
    while position >= 0 {
        out.push(value >> position & 1 == 1 ? "1" : "0")
        position -= 1
    }
    return "".join(out)
}

fn _by_frequency(frequencies: dict[str, int], a: str, b: str) -> int {
    return natural_order(frequencies[a], frequencies[b])
}

fn _by_length_then_symbol(lengths: dict[str, int], a: str, b: str) -> int {
    let by_length = natural_order(lengths[a], lengths[b])
    return by_length != 0 ? by_length : natural_order(a, b)
}

fn _by_count_then_word(counts: dict[str, int], a: str, b: str) -> int {
    let by_count = natural_order(counts[b], counts[a])
    return by_count != 0 ? by_count : natural_order(a, b)
}

#: Return the code length of every symbol under an optimal prefix code.
#:
#: The two-queue construction: leaves in one queue in non-decreasing weight,
#: merged nodes in the other. A merge always produces a weight at least as
#: large as the last one, so merged nodes stay ordered without a heap and the
#: two smallest nodes are always at one head or the other. Every merge deepens
#: both groups by one, which yields the lengths — all a canonical code needs.
#:
#: `alphabet` must be sorted and `sort_by` is stable, so equal frequencies keep
#: alphabetical order and the code is the same on every run.
fn _huffman_lengths(alphabet: list[str], frequencies: dict[str, int]) -> dict[str, int] {
    let lengths = {symbol: 0 for symbol in alphabet}
    if alphabet.len() == 1 {
        lengths[alphabet[0]] = 1
        return lengths
    }
    let leaves = sort_by(alphabet, |a, b| _by_frequency(frequencies, a, b))
    let leaf_weights = [frequencies[symbol] for symbol in leaves]
    let merged_weights: list[int] = []
    let merged_groups: list[list[str]] = []
    var leaf_index = 0
    var merged_index = 0
    while leaves.len() - leaf_index + (merged_weights.len() - merged_index) > 1 {
        var weight = 0
        let combined: list[str] = []
        for _ in 0..2 {
            var take_leaf = merged_index >= merged_weights.len()
            if not take_leaf and leaf_index < leaf_weights.len() {
                take_leaf = leaf_weights[leaf_index] <= merged_weights[merged_index]
            }
            if take_leaf {
                weight += leaf_weights[leaf_index]
                combined.push(leaves[leaf_index])
                leaf_index += 1
            } else {
                weight += merged_weights[merged_index]
                for symbol in merged_groups[merged_index] { combined.push(symbol) }
                merged_index += 1
            }
        }
        for symbol in combined { lengths[symbol] += 1 }
        merged_weights.push(weight)
        merged_groups.push(combined)
    }
    return lengths
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
        # Every rotation now has a rank of its own: no later round can separate
        # them further.
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

#: Walk Myers' trace backwards, turning each round into one edit.
#:
#: Round d recorded how far every diagonal reached after d edits. Starting from
#: the end point, the diagonal that fed each round says whether its last move
#: was an insertion or a deletion, and the run along the diagonal before it is
#: a block of keeps. The script comes out backwards and is reversed once.
fn _backtrack(
    before: list[str],
    after: list[str],
    trace: list[list[int]],
    offset: int
) -> list[DiffOp] {
    let backwards: list[DiffOp] = []
    var x = before.len()
    var y = after.len()
    var depth = trace.len() - 1
    while depth >= 0 {
        let furthest = trace[depth]
        let k = x - y
        var previous_k = 0
        if k == -depth or k != depth and furthest[k - 1 + offset] < furthest[k + 1 + offset] {
            previous_k = k + 1
        } else {
            previous_k = k - 1
        }
        let previous_x = furthest[previous_k + offset]
        let previous_y = previous_x - previous_k
        while x > previous_x and y > previous_y {
            backwards.push(DiffOp.Keep(x - 1, y - 1, before[x - 1]))
            x -= 1
            y -= 1
        }
        if depth > 0 {
            if x == previous_x {
                backwards.push(DiffOp.Insert(previous_y, after[previous_y]))
            } else {
                backwards.push(DiffOp.Delete(previous_x, before[previous_x]))
            }
        }
        x = previous_x
        y = previous_y
        depth -= 1
    }
    let script: list[DiffOp] = []
    var index = backwards.len() - 1
    while index >= 0 {
        script.push(backwards[index])
        index -= 1
    }
    return script
}
