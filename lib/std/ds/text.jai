#: String indexes: a prefix trie, a suffix array with its LCP table, and a
#: rolling hash.
const DEFAULT_BASE = 257
const DEFAULT_MODULUS = 1000000007

class _TrieNode {
    pub var children: dict[str, _TrieNode]
    pub var terminal: bool
    pub var keys_below: int

    pub fn init(self) {
        self.children = {}
        self.terminal = false
        self.keys_below = 0
    }
}

#: A prefix tree over strings.
#:
#: Every operation costs the length of the key or prefix involved, independent
#: of how many keys the trie holds. Results come out in lexicographic order.
#:
#: ```
#: let trie = Trie()
#: trie.insert("car")
#: trie.insert("cart")
#: trie.insert("dog")
#: trie.contains("car")             # true
#: trie.contains("ca")              # false
#: trie.starts_with("ca")           # true
#: trie.keys_with_prefix("car")     # ["car", "cart"]
#: trie.longest_prefix_of("carts")  # "cart"
#: trie.count("c")                  # 2
#: ```
pub class Trie {
    var root: _TrieNode
    var size: int

    #: Create an empty trie.
    pub fn init(self) {
        self.root = _TrieNode()
        self.size = 0
    }

    #: Build a trie holding every key in `keys`.
    pub static fn from_list(keys: list[str]) -> Trie {
        let trie = Trie()
        for key in keys { trie.insert(key) }
        return trie
    }

    #: Number of keys.
    pub fn len(self) -> int {
        return self.size
    }

    #: Report whether the trie holds no keys. O(1).
    #:
    #: Counts keys, not nodes — a trie holding only `""` is not empty.
    pub fn is_empty(self) -> bool {
        return self.size == 0
    }

    #: Add `key`, returning false when it was already present.
    pub fn insert(self, key: str) -> bool {
        if self._find(key) != null { return false }
        var node = self.root
        node.keys_below += 1
        for index in 0..key.len() {
            let character = key[index]
            if not node.children.has(character) {
                node.children[character] = _TrieNode()
            }
            node = node.children[character]
            node.keys_below += 1
        }
        node.terminal = true
        self.size += 1
        return true
    }

    #: Report whether `key` was inserted.
    pub fn contains(self, key: str) -> bool {
        return self._find(key) != null
    }

    #: Report whether any key begins with `prefix`.
    pub fn starts_with(self, prefix: str) -> bool {
        return self._walk(prefix) != null
    }

    #: Number of keys beginning with `prefix`; the whole trie for `""`.
    pub fn count(self, prefix: str = "") -> int {
        let node = self._walk(prefix)
        if node == null { return 0 }
        return node.keys_below
    }

    #: Every key beginning with `prefix`, in lexicographic order.
    pub fn keys_with_prefix(self, prefix: str) -> list[str] {
        let start = self._walk(prefix)
        let found: list[str] = []
        if start == null { return found }
        # Pushed in reverse so the smallest pops first, keeping output sorted.
        let stack: list[tuple[_TrieNode, str]] = [(start, prefix)]
        while stack.len() > 0 {
            let frame = stack.pop()
            let node = frame[0]
            let text = frame[1]
            if node.terminal { found.push(text) }
            let labels = sorted(node.children.keys())
            var position = labels.len() - 1
            while position >= 0 {
                let label = labels[position]
                stack.push((node.children[label], text + label))
                position -= 1
            }
        }
        return found
    }

    #: Every key, in lexicographic order.
    pub fn keys(self) -> list[str] {
        return self.keys_with_prefix("")
    }

    #: The longest key that is a prefix of `text`, or null when there is none.
    pub fn longest_prefix_of(self, text: str) -> str? {
        var node = self.root
        var best: str? = null
        if node.terminal { best = "" }
        for index in 0..text.len() {
            let character = text[index]
            if not node.children.has(character) { break }
            node = node.children[character]
            if node.terminal { best = text[0:index + 1] }
        }
        return best
    }

    #: Remove `key`, returning false when it was absent.
    pub fn delete(self, key: str) -> bool {
        if self._find(key) == null { return false }
        let path: list[tuple[_TrieNode, str]] = []
        var node = self.root
        node.keys_below -= 1
        for index in 0..key.len() {
            let character = key[index]
            path.push((node, character))
            node = node.children[character]
            node.keys_below -= 1
        }
        node.terminal = false
        self.size -= 1
        # A node with keys_below == 0 is on no remaining key's path, so it's
        # safe to prune here.
        var position = path.len() - 1
        var child = node
        while position >= 0 {
            if child.keys_below > 0 { break }
            let frame = path[position]
            let parent = frame[0]
            let dead = frame[1]
            let survivors: dict[str, _TrieNode] = {}
            for (label, branch) in parent.children.items() {
                if label != dead { survivors[label] = branch }
            }
            parent.children = survivors
            child = parent
            position -= 1
        }
        return true
    }

    #: Remove every key.
    pub fn clear(self) -> void {
        self.root = _TrieNode()
        self.size = 0
    }

    fn _walk(self, prefix: str) -> _TrieNode? {
        var node = self.root
        for index in 0..prefix.len() {
            let character = prefix[index]
            if not node.children.has(character) { return null }
            node = node.children[character]
        }
        return node
    }

    fn _find(self, key: str) -> _TrieNode? {
        let node = self._walk(key)
        if node == null { return null }
        if not node.terminal { return null }
        return node
    }

    fn __len__(self) -> int { return self.size }

    fn __contains__(self, key: str) -> bool { return self.contains(key) }

    fn __str__(self) -> str {
        return f"Trie({self.size} keys)"
    }
}

#: The sorted suffixes of a string, with the longest-common-prefix table.
#:
#: Construction is O(n log^2 n) by prefix doubling; the LCP table is filled by
#: Kasai's algorithm in O(n).
#:
#: ```
#: let index = SuffixArray("banana")
#: index.indices()                  # [5, 3, 1, 0, 4, 2]
#: index.suffix_at(0)               # "a"
#: index.lcp()                      # [0, 1, 3, 0, 0, 2]
#: index.count("ana")               # 2
#: index.positions("na")            # [2, 4]
#: index.longest_repeated()         # "ana"
#: ```
pub class SuffixArray {
    let text: str
    let suffixes: list[int]
    let heights: list[int]

    #: Index `text`.
    pub fn init(self, text: str) {
        self.text = text
        self.suffixes = _build_suffix_array(text)
        self.heights = _build_lcp(text, self.suffixes)
    }

    #: Length of the indexed text.
    pub fn len(self) -> int {
        return self.suffixes.len()
    }

    #: The indexed text.
    pub fn source(self) -> str {
        return self.text
    }

    #: The starting offsets of the suffixes, in sorted order.
    pub fn indices(self) -> list[int] {
        return [offset for offset in self.suffixes]
    }

    #: The `rank`-th smallest suffix.
    pub fn suffix_at(self, rank: int) -> str {
        if rank < 0 or rank >= self.suffixes.len() {
            throw IndexError(f"rank {rank} is not within [0, {self.suffixes.len()})")
        }
        return self.text[self.suffixes[rank]:self.text.len()]
    }

    #: The LCP table: entry i is the shared prefix length of suffixes i-1 and i.
    #:
    #: Entry 0 is always 0, so the table lines up with `indices()`.
    pub fn lcp(self) -> list[int] {
        return [height for height in self.heights]
    }

    #: Number of occurrences of `pattern` in the text.
    #:
    #: The empty pattern matches every suffix, so it counts the text length.
    pub fn count(self, pattern: str) -> int {
        if pattern.len() == 0 { return self.suffixes.len() }
        return self._upper_bound(pattern) - self._lower_bound(pattern)
    }

    #: Report whether `pattern` occurs in the text.
    pub fn contains(self, pattern: str) -> bool {
        return self.count(pattern) > 0
    }

    #: Every starting offset of `pattern`, ascending.
    pub fn positions(self, pattern: str) -> list[int] {
        let low = self._lower_bound(pattern)
        let high = self._upper_bound(pattern)
        let found: list[int] = []
        for rank in low..high { found.push(self.suffixes[rank]) }
        return sorted(found)
    }

    #: The longest substring that occurs at least twice, or `""` when none does.
    #:
    #: Two suffixes sharing a long prefix are adjacent in sorted order, so the
    #: answer is the largest entry of the LCP table.
    pub fn longest_repeated(self) -> str {
        var best = 0
        var at = 0
        for (rank, height) in self.heights.enumerate() {
            if height > best {
                best = height
                at = self.suffixes[rank]
            }
        }
        return self.text[at:at + best]
    }

    #: The first rank whose suffix is not less than `pattern`.
    fn _lower_bound(self, pattern: str) -> int {
        var low = 0
        var high = self.suffixes.len()
        while low < high {
            let middle = (low + high) // 2
            if self._prefix_compare(self.suffixes[middle], pattern) < 0 {
                low = middle + 1
            } else {
                high = middle
            }
        }
        return low
    }

    #: The first rank whose suffix does not begin with `pattern`.
    fn _upper_bound(self, pattern: str) -> int {
        var low = 0
        var high = self.suffixes.len()
        while low < high {
            let middle = (low + high) // 2
            if self._prefix_compare(self.suffixes[middle], pattern) <= 0 {
                low = middle + 1
            } else {
                high = middle
            }
        }
        return low
    }

    #: Compare the suffix at `offset` with `pattern`, treating a suffix that
    #: starts with `pattern` as equal to it so both bounds share this routine.
    fn _prefix_compare(self, offset: int, pattern: str) -> int {
        let limit = self.text.len()
        for index in 0..pattern.len() {
            if offset + index >= limit { return -1 }
            let here = self.text[offset + index]
            let there = pattern[index]
            if here < there { return -1 }
            if here > there { return 1 }
        }
        return 0
    }

    fn __len__(self) -> int { return self.suffixes.len() }

    fn __str__(self) -> str {
        return f"SuffixArray({self.suffixes.len()} suffixes)"
    }
}

#: Polynomial hashes of every prefix of a string, for O(1) substring hashing.
#:
#: The hash of `text[start:stop]` is
#: `prefix[stop] - prefix[start] * base ** (stop - start)` modulo a prime, so any
#: two substrings can be compared in constant time.
#:
#: Equal hashes are strong evidence, not proof: `equals` verifies a hash match
#: character by character, so it never reports a false positive.
#:
#: ```
#: let hash = RollingHash("abracadabra")
#: hash.hash_of(0, 4) == hash.hash_of(7, 11)   # true
#: hash.equals(0, 7, 4)                        # true
#: hash.find_all("abra")                       # [0, 7]
#: ```
pub class RollingHash {
    let text: str
    let base: int
    let modulus: int
    let prefix: list[int]
    let powers: list[int]

    #: Hash `text`, using `base` as the polynomial base and `modulus` as the prime.
    pub fn init(self, text: str, base: int = DEFAULT_BASE, modulus: int = DEFAULT_MODULUS) {
        if base < 2 {
            throw ValueError(f"base must be at least 2, got {base}")
        }
        if modulus <= base {
            throw ValueError(f"modulus must exceed base {base}, got {modulus}")
        }
        self.text = text
        self.base = base
        self.modulus = modulus
        let length = text.len()
        let prefix: list[int] = [0]
        let powers: list[int] = [1]
        for index in 0..length {
            prefix.push((prefix[index] * base + ord(text[index])) % modulus)
            powers.push(powers[index] * base % modulus)
        }
        self.prefix = prefix
        self.powers = powers
    }

    #: Length of the hashed text.
    pub fn len(self) -> int {
        return self.text.len()
    }

    #: The hash of the half-open range `text[start:stop]`.
    pub fn hash_of(self, start: int, stop: int) -> int {
        if start < 0 or stop > self.text.len() or start > stop {
            throw IndexError(f"range [{start}, {stop}) is not within [0, {self.text.len()})")
        }
        let span = self.prefix[start] * self.powers[stop - start] % self.modulus
        # Adding modulus back first keeps the result in [0, modulus) regardless
        # of `%`'s sign convention on negative operands.
        return (self.prefix[stop] - span + self.modulus) % self.modulus
    }

    #: The hash of the whole text.
    pub fn full(self) -> int {
        return self.prefix[self.text.len()]
    }

    #: Report whether the two substrings of `length` at `left` and `right` match.
    #:
    #: The hashes are compared first, and a hit is confirmed character by
    #: character, so a collision costs time but never correctness.
    pub fn equals(self, left: int, right: int, length: int) -> bool {
        if length < 0 {
            throw ValueError(f"length must be non-negative, got {length}")
        }
        if left + length > self.text.len() or right + length > self.text.len() {
            return false
        }
        if self.hash_of(left, left + length) != self.hash_of(right, right + length) {
            return false
        }
        for index in 0..length {
            if self.text[left + index] != self.text[right + index] { return false }
        }
        return true
    }

    #: The length of the longest common prefix of the suffixes at `left` and
    #: `right`, found by binary search over `equals` in O(log n) hash compares.
    pub fn common_prefix(self, left: int, right: int) -> int {
        let limit = min(self.text.len() - left, self.text.len() - right)
        var low = 0
        var high = limit
        while low < high {
            let middle = (low + high + 1) // 2
            if self.equals(left, right, middle) {
                low = middle
            } else {
                high = middle - 1
            }
        }
        return low
    }

    #: Every starting offset of `pattern`, ascending, by Rabin-Karp.
    #:
    #: O(n + m) hash work plus verification. An empty pattern matches at every
    #: position, including the end.
    pub fn find_all(self, pattern: str) -> list[int] {
        let length = pattern.len()
        let found: list[int] = []
        if length > self.text.len() { return found }
        if length == 0 {
            for index in 0..=self.text.len() { found.push(index) }
            return found
        }
        var wanted = 0
        for index in 0..length {
            wanted = (wanted * self.base + ord(pattern[index])) % self.modulus
        }
        for start in 0..=self.text.len() - length {
            if self.hash_of(start, start + length) != wanted { continue }
            if self._verify(start, pattern) { found.push(start) }
        }
        return found
    }

    fn _verify(self, start: int, pattern: str) -> bool {
        for index in 0..pattern.len() {
            if self.text[start + index] != pattern[index] { return false }
        }
        return true
    }

    fn __len__(self) -> int { return self.text.len() }

    fn __str__(self) -> str {
        return f"RollingHash({self.text.len()} characters, base {self.base})"
    }
}

#: Sort the suffix offsets of `text` by prefix doubling.
fn _build_suffix_array(text: str) -> list[int] {
    let length = text.len()
    let suffixes: list[int] = []
    if length == 0 { return suffixes }
    # `<` isn't defined for tuples, so pairs are ordered by two stable sorts
    # (minor key, then major) — a radix pass, which doubling wants anyway.
    let offsets: list[int] = []
    for index in 0..length { offsets.push(index) }
    let by_character = sorted(offsets, |index| ord(text[index]))
    let rank: list[int] = []
    for _index in 0..length { rank.push(0) }
    var distinct = 0
    rank[by_character[0]] = 0
    for position in 1..length {
        let here = ord(text[by_character[position]])
        if here != ord(text[by_character[position - 1]]) { distinct += 1 }
        rank[by_character[position]] = distinct
    }
    for offset in by_character { suffixes.push(offset) }
    var step = 1
    while step < length and distinct < length - 1 {
        # -1 for a pair that runs off the end: a shorter suffix must sort
        # before an otherwise identical longer one.
        let second: list[int] = []
        for index in 0..length {
            second.push(if index + step < length { rank[index + step] } else { -1 })
        }
        let ordered = sorted(sorted(offsets, |index| second[index]), |index| rank[index])
        let next_rank: list[int] = []
        for _index in 0..length { next_rank.push(0) }
        distinct = 0
        next_rank[ordered[0]] = 0
        for position in 1..length {
            let previous = ordered[position - 1]
            let current = ordered[position]
            if rank[current] != rank[previous] or second[current] != second[previous] {
                distinct += 1
            }
            next_rank[current] = distinct
        }
        for position in 0..length { rank[position] = next_rank[position] }
        for position in 0..length { suffixes[position] = ordered[position] }
        step *= 2
    }
    return suffixes
}

#: Fill the LCP table for an already sorted suffix array, by Kasai's algorithm.
fn _build_lcp(text: str, suffixes: list[int]) -> list[int] {
    let length = suffixes.len()
    let heights: list[int] = []
    for _index in 0..length { heights.push(0) }
    if length == 0 { return heights }
    let inverse: list[int] = []
    for _index in 0..length { inverse.push(0) }
    for (rank, offset) in suffixes.enumerate() { inverse[offset] = rank }
    # Walking the text left to right, the match length can only drop by one per
    # step, which is what keeps the whole pass linear.
    var matched = 0
    for offset in 0..length {
        let rank = inverse[offset]
        if rank == 0 {
            matched = 0
            continue
        }
        let previous = suffixes[rank - 1]
        let limit = min(length - offset, length - previous)
        while matched < limit and text[offset + matched] == text[previous + matched] {
            matched += 1
        }
        heights[rank] = matched
        if matched > 0 { matched -= 1 }
    }
    return heights
}
