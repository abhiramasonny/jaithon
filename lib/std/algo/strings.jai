#: String algorithms that need a real construction rather than a scan.
#:
#: `std.str` covers what a program does to one string at a time — padding,
#: case, distance, escaping. This module started as everything else: every
#: algorithm here now lives in the file named for what it builds, and this
#: file kept the five that build nothing and hand back only an array of
#: indices or edit operations from a single pass over the raw text:
#:
#:   palindrome_radii / longest_palindrome   Manacher, O(n).
#:   z_function / prefix_function            The two linear scans the rest of
#:                                           string matching is built from.
#:   minimal_rotation                        Booth, O(n).
#:   diff                                    Myers, O((n + m) * d).
#:
#: Everything that first builds an object to query afterward moved out, one
#: file per thing built:
#:
#:   std.algo.suffix_automaton   SuffixAutomaton — every substring of a text,
#:                               as a DFA.
#:   std.algo.huffman            HuffmanCode — a canonical prefix code.
#:   std.algo.lz77                LZ77Token, lz77_compress/decompress — the
#:                                 dictionary half of the compression pair.
#:   std.algo.bwt                 bwt_encode/decode and their rle_encode/
#:                                 decode partner — the block-sort half.
#:   std.algo.codecs              utf8_encode/decode, base64_encode/decode,
#:                                 hex_encode/decode, crc32, fnv1a — bytes
#:                                 through a text-only channel, and the checks
#:                                 that they arrived intact.
#:   std.algo.phonetics           soundex, metaphone — phonetic keys.
#:   std.algo.tokenize            tokenize, word_frequency, top_words,
#:                                 char_ngrams, word_ngrams.
#:
#: Byte-oriented functions (now in `std.algo.codecs`) take and return
#: `list[int]` whose elements are all in 0..=255. Everything in this file
#: treats a string as a sequence of Unicode scalar values, so indices and
#: lengths agree with `str.len()`.
#:
#: See `std.ds.text` for the indexes built by sorting instead: a trie, a suffix
#: array and a rolling hash.

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
