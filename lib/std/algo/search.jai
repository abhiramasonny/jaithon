#: Searching: ordered lists, unordered lists, unimodal functions, and strings.
#:
#: Ordered search over a list sorted by the same comparator, n elements:
#:
#:   binary_search        O(log n). The default. Index of the first equal
#:                        element, or null.
#:   lower_bound          O(log n). Insertion point before equals.
#:   upper_bound          O(log n). Insertion point after equals.
#:   equal_range          O(log n). Both bounds, so the run of equals is
#:                        `xs[lower:upper]` and its length is the count.
#:   binary_search_by     O(log n). The general form: the partition point of
#:                        any predicate that is false then true.
#:   exponential_search   O(log i) for a hit at index i. Beats binary search
#:                        when the target is near the front of a long list.
#:   interpolation_search O(log log n) expected on evenly spread numeric keys
#:                        (timestamps, ids), O(n) worst. Guesses where the key
#:                        should be rather than always halving.
#:   ternary_search       O(log((hi - lo) / tolerance)) over a unimodal
#:                        function — a continuous domain, not a list.
#:
#: Unordered search: `linear_search` and `find_all`, both O(n). For anything
#: repeated, build a dict or a set instead — O(1) per lookup beats both.
#:
#: String search, text of length n, pattern of length m, alphabet size s:
#:
#:   kmp_search           O(n + m) guaranteed, O(m) memory, never backs up over
#:                        the text. The safe default and the one for streams.
#:   boyer_moore_search   O(n / m) best, O(n * m) worst, O(m + s) memory. Skips
#:                        ahead instead of stepping, so it is the fastest in
#:                        practice for long patterns over large alphabets.
#:   rabin_karp_search    O(n + m) expected, O(1) memory. Worth it when the
#:                        rolling hash generalises: many equal-length patterns
#:                        at once, or 2-D pattern matching.
#:   z_algorithm          O(n). Not a search but the structure underneath one;
#:                        also gives periods, borders, and longest prefixes.
#:   AhoCorasick          O(n + total pattern length + matches) for many
#:                        patterns in one pass, whatever their lengths.
#:
#: Positions are code point offsets, not byte offsets, so they stay correct for
#: non-ASCII text. An empty pattern matches nothing.
from std.algo.sort import Comparator, natural_order

const RABIN_KARP_BASE = 1_000_003
const RABIN_KARP_MOD = 1_000_000_007
const MAX_TERNARY_STEPS = 200

#: An Aho-Corasick automaton: find any number of patterns in one pass.
#:
#: Construction is O(total pattern length) and searching is O(n + matches),
#: independent of how many patterns there are — the reason to use it instead of
#: running `kmp_search` once per pattern, which costs O(k * n) for k patterns.
#:
#: The patterns form a trie; each node also carries a failure link to the
#: longest proper suffix of its path that is itself a trie node, so a mismatch
#: falls back without rescanning the text. Output sets are merged along those
#: links, so a node reports every pattern ending there, including the shorter
#: ones nested inside it.
#:
#: Build once and search many texts; `aho_corasick` is the one-shot form.
pub class AhoCorasick {
    var lengths: list[int]
    var edges: list[dict[int, int]]
    var fallback: list[int]
    var outputs: list[list[int]]

    #: Build the automaton for `patterns`.
    #:
    #: Raises `ValueError` if `patterns` is empty or any pattern is empty.
    pub fn init(self, patterns: list[str]) {
        if patterns.len() == 0 { throw ValueError("AhoCorasick needs at least one pattern") }
        self.lengths = []
        self.edges = [{}]
        self.fallback = [0]
        self.outputs = [[]]
        for (index, pattern) in patterns.enumerate() {
            let codes = _codes(pattern)
            if codes.len() == 0 { throw ValueError(f"pattern {index} is empty") }
            self.lengths.push(codes.len())
            var node = 0
            for code in codes {
                if code in self.edges[node] {
                    node = self.edges[node][code]
                } else {
                    let created = self.edges.len()
                    self.edges.push({})
                    self.fallback.push(0)
                    self.outputs.push([])
                    self.edges[node][code] = created
                    node = created
                }
            }
            self.outputs[node].push(index)
        }
        self._link()
    }

    #: Return every match as `(pattern_index, start_position)`.
    #:
    #: O(n + matches). Matches come out ordered by where they end, so nested
    #: and overlapping patterns are all reported.
    #:
    #: ```
    #: AhoCorasick(["he", "she", "his", "hers"]).search("ushers")
    #: # [(1, 1), (0, 2), (3, 2)]
    #: ```
    pub fn search(self, text: str) -> list[tuple[int, int]] {
        let found: list[tuple[int, int]] = []
        var node = 0
        for (position, code) in _codes(text).enumerate() {
            while node != 0 and code not in self.edges[node] {
                node = self.fallback[node]
            }
            if code in self.edges[node] { node = self.edges[node][code] }
            for index in self.outputs[node] {
                found.push((index, position - self.lengths[index] + 1))
            }
        }
        return found
    }

    #: Compute failure links breadth first, so that a node's link is always
    #: resolved before its children need it, and fold each target's outputs
    #: into the node itself to keep reporting O(1) per match.
    fn _link(self) -> void {
        let queue: list[int] = []
        for node in self.edges[0].values() {
            queue.push(node)
        }
        var head = 0
        while head < queue.len() {
            let node = queue[head]
            head += 1
            for (code, next) in self.edges[node].items() {
                var state = self.fallback[node]
                while state != 0 and code not in self.edges[state] {
                    state = self.fallback[state]
                }
                var target = 0
                if code in self.edges[state] { target = self.edges[state][code] }
                self.fallback[next] = target
                for index in self.outputs[target] {
                    self.outputs[next].push(index)
                }
                queue.push(next)
            }
        }
    }
}

#: Return the index of the first element equal to `target`, or null.
#:
#: O(log n) comparisons. `xs` must be sorted by `cmp`; the result is
#: unspecified otherwise. Returning the *first* equal element rather than an
#: arbitrary one makes the answer independent of the list's length.
#:
#: ```
#: binary_search([1, 3, 5], 3)   # 1
#: binary_search([1, 3, 5], 4)   # null
#: ```
pub fn binary_search[T](xs: list[T], target: T, cmp: Comparator[T] = natural_order) -> int? {
    let position = _lower_bound_range(xs, 0, xs.len(), target, cmp)
    if position < xs.len() and cmp(xs[position], target) == 0 { return position }
    return null
}

#: Return the first index where `target` could be inserted, before any equals.
#:
#: O(log n). Equivalently, the number of elements strictly less than `target`.
#:
#: ```
#: lower_bound([1, 2, 2, 3], 2)   # 1
#: lower_bound([1, 2, 2, 3], 9)   # 4
#: ```
pub fn lower_bound[T](xs: list[T], target: T, cmp: Comparator[T] = natural_order) -> int {
    return _lower_bound_range(xs, 0, xs.len(), target, cmp)
}

#: Return the last index where `target` could be inserted, after any equals.
#:
#: O(log n). Equivalently, the number of elements less than or equal to
#: `target`.
#:
#: ```
#: upper_bound([1, 2, 2, 3], 2)   # 3
#: ```
pub fn upper_bound[T](xs: list[T], target: T, cmp: Comparator[T] = natural_order) -> int {
    var lo = 0
    var hi = xs.len()
    while lo < hi {
        let mid = lo + (hi - lo) // 2
        if cmp(xs[mid], target) <= 0 { lo = mid + 1 } else { hi = mid }
    }
    return lo
}

#: Return `(lower_bound, upper_bound)` for `target`.
#:
#: O(log n). The equal elements are `xs[lower:upper]` and there are
#: `upper - lower` of them, which is how to count duplicates without scanning.
#:
#: ```
#: equal_range([1, 2, 2, 2, 3], 2)   # (1, 4)
#: equal_range([1, 2, 2, 2, 3], 9)   # (5, 5)
#: ```
pub fn equal_range[T](
    xs: list[T],
    target: T,
    cmp: Comparator[T] = natural_order
) -> tuple[int, int] {
    return (lower_bound(xs, target, cmp), upper_bound(xs, target, cmp))
}

#: Return the first index where `predicate` holds, or `xs.len()` if it never
#: does.
#:
#: O(log n) predicate calls. `predicate` must be monotone over `xs`: false for
#: a prefix, then true for the rest. This is the general binary search — every
#: other one here is a special case — and it is what to use when the ordering
#: key is computed rather than compared, or when searching for a boundary
#: rather than a value.
#:
#: ```
#: binary_search_by([1, 3, 5, 7], |x| x >= 4)   # 2
#: binary_search_by([1, 3, 5, 7], |x| x >= 9)   # 4
#: ```
pub fn binary_search_by[T](xs: list[T], predicate: fn(T) -> bool) -> int {
    var lo = 0
    var hi = xs.len()
    while lo < hi {
        let mid = lo + (hi - lo) // 2
        if predicate(xs[mid]) { hi = mid } else { lo = mid + 1 }
    }
    return lo
}

#: Return the index of the first element equal to `target`, or null, by
#: doubling a bound before searching inside it.
#:
#: O(log i) where i is the index found, against O(log n) for `binary_search`.
#: The probe indices double — 1, 2, 4, 8 — until they overshoot, and the
#: binary search then covers only the last doubling. Prefer it when the list is
#: very long and hits cluster near the front, or when reading far-away elements
#: is expensive (a paged or memory-mapped list).
#:
#: ```
#: exponential_search([1, 2, 3, 4, 5, 6, 7, 8], 7)   # 6
#: exponential_search([1, 2, 3], 9)                  # null
#: ```
pub fn exponential_search[T](xs: list[T], target: T, cmp: Comparator[T] = natural_order) -> int? {
    let n = xs.len()
    if n == 0 { return null }
    var bound = 1
    while bound < n and cmp(xs[bound], target) < 0 {
        bound = bound * 2
    }
    let lo = bound // 2
    let hi = min(bound + 1, n)
    let position = _lower_bound_range(xs, lo, hi, target, cmp)
    if position < n and cmp(xs[position], target) == 0 { return position }
    return null
}

#: Return the index of an element equal to `target`, or null, by estimating
#: where it should be.
#:
#: O(log log n) expected when the keys are spread evenly, because each probe
#: interpolates the target's position within the remaining range instead of
#: taking the midpoint; O(n) worst case when they are not (an exponential
#: distribution defeats it). Prefer it for large, dense, near-uniform integer
#: keys — auto-increment ids, second-resolution timestamps — and
#: `binary_search` for everything else, including short lists, where the extra
#: arithmetic per probe costs more than the probes it saves.
#:
#: With duplicates, which equal element is found is unspecified.
#:
#: ```
#: interpolation_search([0, 10, 20, 30, 40], 30)   # 3
#: interpolation_search([0, 10, 20, 30, 40], 35)   # null
#: ```
pub fn interpolation_search(xs: list[int], target: int) -> int? {
    var lo = 0
    var hi = xs.len() - 1
    while lo <= hi and xs[lo] <= target and target <= xs[hi] {
        if xs[hi] == xs[lo] {
            return xs[lo] == target ? lo : null
        }
        # Interpolate in floats: the exact ratio can overflow an int for keys
        # near the 64-bit limits, and only the probe's accuracy depends on it.
        let ratio = (float(target) - float(xs[lo])) / (float(xs[hi]) - float(xs[lo]))
        var probe = lo + int(ratio * float(hi - lo))
        if probe < lo { probe = lo }
        if probe > hi { probe = hi }
        if xs[probe] == target { return probe }
        if xs[probe] < target { lo = probe + 1 } else { hi = probe - 1 }
    }
    return null
}

#: Return the argument that minimises a unimodal `f` on `[lo, hi]`.
#:
#: O(log((hi - lo) / tolerance)) iterations, two evaluations of `f` each: the
#: range is cut into thirds and whichever outer third cannot hold the minimum
#: is discarded, shrinking it by 2/3 per step. `f` must be unimodal — strictly
#: decreasing then strictly increasing. To maximise instead, search `|x| -f(x)`.
#: Unlike the list searches this needs no data, only the function.
#:
#: Raises `ValueError` when `hi < lo` or `tolerance` is not positive.
#:
#: ```
#: let x = ternary_search(|v| (v - 2.0) * (v - 2.0), 0.0, 5.0)
#: x > 1.999 and x < 2.001   # true
#: ```
pub fn ternary_search(
    f: fn(float) -> float,
    lo: float,
    hi: float,
    tolerance: float = 1e-9
) -> float {
    if hi < lo { throw ValueError(f"empty interval: lo is {lo} but hi is {hi}") }
    if tolerance <= 0.0 { throw ValueError(f"tolerance must be positive, got {tolerance}") }
    var left = lo
    var right = hi
    var steps = 0
    while right - left > tolerance and steps < MAX_TERNARY_STEPS {
        let third = (right - left) / 3.0
        let probe_low = left + third
        let probe_high = right - third
        if f(probe_low) < f(probe_high) { right = probe_high } else { left = probe_low }
        steps += 1
    }
    return (left + right) / 2.0
}

#: Return the index of the first element equal to `target`, or null.
#:
#: O(n), no ordering required, one `==` per element. For a single lookup in an
#: unsorted list this is optimal; for repeated lookups build a dict once.
#:
#: ```
#: linear_search(["a", "b", "c"], "b")   # 1
#: linear_search(["a"], "z")             # null
#: ```
pub fn linear_search[T](xs: list[T], target: T) -> int? {
    for (position, value) in xs.enumerate() {
        if value == target { return position }
    }
    return null
}

#: Return every index at which `target` occurs.
#:
#: O(n) always, since every element must be examined.
#:
#: ```
#: find_all([1, 2, 1, 3, 1], 1)   # [0, 2, 4]
#: ```
pub fn find_all[T](xs: list[T], target: T) -> list[int] {
    let found: list[int] = []
    for (position, value) in xs.enumerate() {
        if value == target { found.push(position) }
    }
    return found
}

#: Return the start of every occurrence of `pattern` in `text`, using
#: Knuth-Morris-Pratt.
#:
#: O(n + m) time and O(m) memory in every case. The failure function records,
#: for each prefix of the pattern, the longest proper prefix that is also a
#: suffix of it; on a mismatch the pattern slides by that much and the text
#: cursor never moves backwards. Overlapping matches are all reported.
#:
#: This is the default: linear worst case, no dependence on alphabet size, and
#: the only one here that can run against a stream.
#:
#: ```
#: kmp_search("abababa", "aba")   # [0, 2, 4]
#: kmp_search("aaa", "b")         # []
#: ```
pub fn kmp_search(text: str, pattern: str) -> list[int] {
    let matches: list[int] = []
    let p = _codes(pattern)
    let t = _codes(text)
    let m = p.len()
    let n = t.len()
    if m == 0 or m > n { return matches }
    let failure = _kmp_failure(p)
    var matched = 0
    for position in 0..n {
        while matched > 0 and t[position] != p[matched] {
            matched = failure[matched - 1]
        }
        if t[position] == p[matched] { matched += 1 }
        if matched == m {
            matches.push(position - m + 1)
            matched = failure[matched - 1]
        }
    }
    return matches
}

#: Return the start of every occurrence of `pattern` in `text`, using
#: Boyer-Moore.
#:
#: O(n / m) at best, O(n * m) at worst, O(m + s) memory for an alphabet of s
#: distinct code points in the pattern. The pattern is compared right to left,
#: and a mismatch shifts it by the larger of two precomputed rules: the bad
#: character rule (align the mismatched text symbol with its last occurrence in
#: the pattern, or skip past it entirely if it never occurs) and the good
#: suffix rule (realign the part that already matched). Whole blocks of text
#: are therefore never examined — the longer the pattern and the larger the
#: alphabet, the faster it gets, which is why it is what `grep` reaches for.
#:
#: Prefer `kmp_search` when the worst case matters or the alphabet is tiny
#: (DNA, bits), where the skips shrink to nothing.
#:
#: ```
#: boyer_moore_search("abracadabra", "abra")   # [0, 7]
#: ```
pub fn boyer_moore_search(text: str, pattern: str) -> list[int] {
    let matches: list[int] = []
    let p = _codes(pattern)
    let t = _codes(text)
    let m = p.len()
    let n = t.len()
    if m == 0 or m > n { return matches }
    let last_seen: dict[int, int] = {}
    for position in 0..m {
        last_seen[p[position]] = position
    }
    let shifts = _good_suffix_shifts(p)
    var start = 0
    while start <= n - m {
        var scan = m - 1
        while scan >= 0 and p[scan] == t[start + scan] {
            scan -= 1
        }
        if scan < 0 {
            matches.push(start)
            start += shifts[0]
        } else {
            let bad_character = scan - last_seen.get(t[start + scan], -1)
            start += max(shifts[scan + 1], bad_character)
        }
    }
    return matches
}

#: Return the start of every occurrence of `pattern` in `text`, using
#: Rabin-Karp.
#:
#: O(n + m) expected, O(n * m) worst case, O(1) memory. One rolling hash of the
#: pattern is compared against the hash of each window of the text, which is
#: updated in constant time as the window slides; equal hashes are verified
#: character by character, so the result is exact rather than probabilistic.
#:
#: On its own it loses to `kmp_search`. It earns its place when the rolling
#: hash does more work: hash k patterns of the same length into a set and find
#: them all in one pass, detect duplicate substrings, or hash rows to search a
#: 2-D grid.
#:
#: ```
#: rabin_karp_search("abcabc", "abc")   # [0, 3]
#: ```
pub fn rabin_karp_search(text: str, pattern: str) -> list[int] {
    let matches: list[int] = []
    let p = _codes(pattern)
    let t = _codes(text)
    let m = p.len()
    let n = t.len()
    if m == 0 or m > n { return matches }
    var high_order = 1
    for _ in 0..m - 1 {
        high_order = high_order * RABIN_KARP_BASE % RABIN_KARP_MOD
    }
    var pattern_hash = 0
    var window_hash = 0
    for position in 0..m {
        pattern_hash = (pattern_hash * RABIN_KARP_BASE + p[position]) % RABIN_KARP_MOD
        window_hash = (window_hash * RABIN_KARP_BASE + t[position]) % RABIN_KARP_MOD
    }
    var start = 0
    loop {
        if window_hash == pattern_hash and _matches_at(t, p, start) {
            matches.push(start)
        }
        if start + m >= n { break }
        var rolled = (window_hash - t[start] * high_order) % RABIN_KARP_MOD
        if rolled < 0 { rolled += RABIN_KARP_MOD }
        window_hash = (rolled * RABIN_KARP_BASE + t[start + m]) % RABIN_KARP_MOD
        start += 1
    }
    return matches
}

#: Return the Z array of `s`: `z[i]` is the length of the longest substring
#: starting at `i` that is also a prefix of `s`.
#:
#: O(n) time and memory. The trick is the rightmost segment `[left, right)`
#: already known to match a prefix: inside it, the answer at `i` is read off
#: its mirror `i - left` for free, and only the part beyond `right` is ever
#: compared, so the text cursor only moves forward.
#:
#: This is the substrate for string search rather than a search itself. To find
#: `pattern` in `text`, run it over `pattern + separator + text` where the
#: separator occurs in neither, and every `i` with `z[i] == pattern.len()`
#: marks a match. It also yields periods and borders directly.
#:
#: ```
#: z_algorithm("aabaab")   # [6, 1, 0, 3, 1, 0]
#: ```
pub fn z_algorithm(s: str) -> list[int] {
    let a = _codes(s)
    let n = a.len()
    let z = [0 for _ in 0..n]
    if n == 0 { return z }
    z[0] = n
    var left = 0
    var right = 0
    var i = 1
    while i < n {
        if i < right {
            let mirror = z[i - left]
            let remaining = right - i
            z[i] = mirror < remaining ? mirror : remaining
        }
        while i + z[i] < n and a[z[i]] == a[i + z[i]] {
            z[i] += 1
        }
        if i + z[i] > right {
            left = i
            right = i + z[i]
        }
        i += 1
    }
    return z
}

#: Find every occurrence of every pattern in `text` in a single pass.
#:
#: O(n + total pattern length + matches). Each match is
#: `(pattern_index, start_position)`, ordered by where the match ends. Build an
#: `AhoCorasick` yourself instead when the same patterns are searched in more
#: than one text: this rebuilds the automaton on every call.
#:
#: ```
#: aho_corasick("ushers", ["he", "she", "his", "hers"])
#: # [(1, 1), (0, 2), (3, 2)]
#: ```
pub fn aho_corasick(text: str, patterns: list[str]) -> list[tuple[int, int]] {
    return AhoCorasick(patterns).search(text)
}

fn _codes(s: str) -> list[int] {
    return [ord(character) for character in s]
}

fn _lower_bound_range[T](
    xs: list[T],
    from_index: int,
    to_index: int,
    target: T,
    cmp: Comparator[T]
) -> int {
    var lo = from_index
    var hi = to_index
    while lo < hi {
        let mid = lo + (hi - lo) // 2
        if cmp(xs[mid], target) < 0 { lo = mid + 1 } else { hi = mid }
    }
    return lo
}

fn _matches_at(text: list[int], pattern: list[int], start: int) -> bool {
    for offset in 0..pattern.len() {
        if text[start + offset] != pattern[offset] { return false }
    }
    return true
}

#: Build the KMP failure function: `failure[i]` is the length of the longest
#: proper prefix of `pattern[0..=i]` that is also a suffix of it.
fn _kmp_failure(pattern: list[int]) -> list[int] {
    let m = pattern.len()
    let failure = [0 for _ in 0..m]
    var matched = 0
    var position = 1
    while position < m {
        while matched > 0 and pattern[position] != pattern[matched] {
            matched = failure[matched - 1]
        }
        if pattern[position] == pattern[matched] { matched += 1 }
        failure[position] = matched
        position += 1
    }
    return failure
}

#: Build the Boyer-Moore good suffix table.
#:
#: `shifts[j + 1]` is how far to slide the pattern after a mismatch at index
#: `j`, and `shifts[0]` after a full match. The first pass handles the case
#: where the matched suffix occurs again elsewhere in the pattern; the second
#: fills the rest with the longest pattern prefix that is also a suffix of that
#: matched part. Every entry is at least 1, so the search always advances.
fn _good_suffix_shifts(pattern: list[int]) -> list[int] {
    let m = pattern.len()
    let shifts = [0 for _ in 0..=m]
    let borders = [0 for _ in 0..=m]
    var i = m
    var j = m + 1
    borders[i] = j
    while i > 0 {
        while j <= m and pattern[i - 1] != pattern[j - 1] {
            if shifts[j] == 0 { shifts[j] = j - i }
            j = borders[j]
        }
        i -= 1
        j -= 1
        borders[i] = j
    }
    j = borders[0]
    for position in 0..=m {
        if shifts[position] == 0 { shifts[position] = j }
        if position == j { j = borders[j] }
    }
    return shifts
}
