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

fn _chars(text: str) -> list[str] {
    let out: list[str] = []
    for ch in text { out.push(ch) }
    return out
}
