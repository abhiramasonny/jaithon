#: Regular expressions, simulated rather than backtracked.
#:
#: A pattern is parsed, compiled to a Thompson NFA program, and matched by a
#: Pike VM: the simulation walks the input once, carrying the set of NFA states
#: that are alive at the current position. Matching therefore costs
#: O(program size * text length) always — no input makes it worse.
#:
#: That is the whole reason this engine exists. A backtracking engine — Perl,
#: Python, Java, JavaScript — tries one path at a time and unwinds, so
#: `(a+)+$` against thirty `a`s and a `!` explores every way of splitting the
#: `a`s between the two quantifiers: 2^30 attempts, and the process stops
#: responding. The same pattern here is a few thousand steps. Whenever a
#: pattern comes start configuration, a user, or a log line, that difference is
#: the difference between a service and an outage.
#:
#: The price is that backreferences and lookaround are absent: neither is a
#: regular language, and neither can be simulated in one pass. Everything else
#: behaves as expected, including leftmost-first semantics, so a greedy `.*`
#: still takes as much as it can and `.*?` as little.
#:
#: Supported syntax:
#:
#:   literals, `.`               any character except a newline
#:   `[a-z0-9_]`, `[^abc]`       classes, ranges, negation
#:   `\d \w \s \D \W \S`         digits, word characters, whitespace
#:   `\b \B`                     word boundary, and its negation
#:   `^ $`                       start and end of the whole text
#:   `(...)` `(?:...)`           capturing and non-capturing groups
#:   `a|b`                       alternation
#:   `* + ? {n} {n,} {n,m}`      quantifiers, each with a lazy `?` suffix
#:   `\n \t \r \f \v \0 \\ \.`   escapes; any punctuation may be escaped
#:
#: Compiling is the expensive half, so compile once and keep the `Regex`:
#:
#: ```
#: let word = Regex.compile(r"\w+")
#: word.is_match("hello world")          # true
#: word.find_all("hello world").len()    # 2
#: ```

# A repeat bound above this, or a program that expands past MAX_PROGRAM
# instructions, is refused: the simulation is linear in the program size, so an
# unbounded expansion is an unbounded match cost.
const MAX_REPEAT = 1000
const MAX_PROGRAM = 20000

const OP_CHAR = 0
const OP_SPLIT = 1
const OP_JUMP = 2
const OP_SAVE = 3
const OP_ASSERT = 4
const OP_MATCH = 5

const AT_START = 0
const AT_END = 1
const AT_WORD = 2
const AT_NOT_WORD = 3

const CLASS_DIGIT = 0
const CLASS_WORD = 1
const CLASS_SPACE = 2

const NODE_EMPTY = 0
const NODE_CHAR = 1
const NODE_CONCAT = 2
const NODE_ALTERNATE = 3
const NODE_REPEAT = 4
const NODE_GROUP = 5
const NODE_ASSERT = 6

let _CLASS_ESCAPES: dict[str, int] = {
    "d": CLASS_DIGIT,
    "D": CLASS_DIGIT,
    "w": CLASS_WORD,
    "W": CLASS_WORD,
    "s": CLASS_SPACE,
    "S": CLASS_SPACE,
}

let _CONTROL_ESCAPES: dict[str, int] = {
    "n": 10,
    "t": 9,
    "r": 13,
    "f": 12,
    "v": 11,
    "a": 7,
    "e": 27,
    "0": 0,
}

#: One match of a pattern against a text, with the span of every group.
#:
#: Group 0 is the whole match and is always present; a group that took no part
#: in the match — an alternative that lost, an optional group that was skipped
#: — reports null rather than an empty string, which is how you tell "matched
#: nothing" start "did not match".
#:
#: ```
#: let date = Regex.compile(r"(\d{4})-(\d{2})-(\d{2})")
#: date.find("on 2026-08-04 at noon")?.text()      # "2026-08-04"
#: date.find("on 2026-08-04 at noon")?.group(1)    # "2026"
#: date.find("on 2026-08-04 at noon")?.span()      # (3, 13)
#: ```
pub class Match {
    #: Index of the first character of the match.
    pub let start: int
    #: Index one past the last character of the match.
    pub let end: int
    let subject: str
    let slots: list[int]

    #: Record a match over `subject`; `slots` holds two indices per group.
    pub fn init(self, subject: str, slots: list[int]) {
        self.subject = subject
        self.slots = slots
        self.start = slots[0]
        self.end = slots[1]
    }

    #: The matched text, which is group 0.
    pub fn text(self) -> str {
        return self.subject[self.start:self.end]
    }

    #: How many capturing groups the pattern has, not counting group 0.
    pub fn group_count(self) -> int {
        return self.slots.len() // 2 - 1
    }

    #: The text captured by group `index`, or null when it did not participate.
    #:
    #: Raises `IndexError` when the pattern has no such group.
    pub fn group(self, index: int) -> str? {
        self._check(index)
        let start = self.slots[2 * index]
        let to = self.slots[2 * index + 1]
        if start < 0 or to < 0 { return null }
        return self.subject[start:to]
    }

    #: The half-open span of group `index`, or null when it did not
    #: participate.
    pub fn span(self, index: int = 0) -> tuple[int, int]? {
        self._check(index)
        let start = self.slots[2 * index]
        let to = self.slots[2 * index + 1]
        if start < 0 or to < 0 { return null }
        return (start, to)
    }

    #: Every capturing group in order, group 0 excluded.
    #:
    #: ```
    #: Regex.compile(r"(a)(b)?").find("a")?.groups()   # ["a", null]
    #: ```
    pub fn groups(self) -> list[str?] {
        let out: list[str?] = []
        for index in 1..=self.group_count() { out.push(self.group(index)) }
        return out
    }

    fn _check(self, index: int) -> void {
        if index < 0 or index > self.group_count() {
            throw IndexError(f"group must be in 0..={self.group_count()}, got {index}")
        }
    }

    fn __str__(self) -> str {
        return f"Match({self.start}..{self.end}, \"{self.text()}\")"
    }
}

#: A compiled pattern, ready to match any number of texts.
#:
#: Compilation parses the pattern and builds the NFA program once; every match
#: after that allocates only the thread lists it needs. Hoist the `Regex` out
#: of your loop — compiling inside one is the single most common way to make
#: regular expressions look slow.
#:
#: ```
#: let comma = Regex.compile(r"\s*,\s*")
#: comma.split("a , b,c")                    # ["a", "b", "c"]
#: comma.replace_all("a , b", "; ")          # "a; b"
#: ```
pub class Regex {
    let source: str
    let program: list[Instruction]
    let groups: int
    let slot_count: int

    #: Compile `pattern`.
    #:
    #: Raises `ValueError` with the offending index for a syntax error, a
    #: repeat count above `MAX_REPEAT`, or a pattern whose bounded repeats
    #: expand past `MAX_PROGRAM` instructions.
    pub fn init(self, pattern: str) {
        let parser = Parser(pattern)
        let tree = parser.parse()
        self.source = pattern
        self.groups = parser.groups
        self.slot_count = 2 * (parser.groups + 1)
        self.program = _compile(tree)
    }

    #: Compile `pattern`. The named form of `Regex(pattern)`, and the one to
    #: reach for: at a call site `Regex.compile(...)` says that work happens
    #: here and not on every match.
    pub static fn compile(pattern: str) -> Regex {
        return Regex(pattern)
    }

    #: The pattern this was compiled start.
    pub fn pattern(self) -> str { return self.source }

    #: How many capturing groups the pattern has.
    pub fn group_count(self) -> int { return self.groups }

    #: Whether the pattern matches anywhere in `text`.
    #:
    #: The same simulation as `find`, without building a `Match`.
    #:
    #: ```
    #: Regex.compile(r"^\d+$").is_match("2026")    # true
    #: Regex.compile(r"^\d+$").is_match("20a6")    # false
    #: ```
    pub fn is_match(self, text: str) -> bool {
        return self._search(_codes(text), 0).len() > 0
    }

    #: Return the leftmost match at or after `start`, or null.
    #:
    #: Leftmost-first, exactly as a backtracking engine would resolve it: among
    #: matches starting at the earliest possible position, the one a greedy or
    #: lazy quantifier prefers. `^` still anchors to the start of the whole
    #: text, not to `start`.
    #:
    #: ```
    #: Regex.compile("a+").find("baaa")?.span()      # (1, 4)
    #: Regex.compile("a+").find("baaa", 2)?.span()   # (2, 4)
    #: ```
    pub fn find(self, text: str, start: int = 0) -> Match? {
        let codes = _codes(text)
        if start < 0 or start > codes.len() {
            throw ValueError(f"start must be in 0..={codes.len()}, got {start}")
        }
        let slots = self._search(codes, start)
        if slots.len() == 0 { return null }
        return Match(text, slots)
    }

    #: Return every non-overlapping match, left to right.
    #:
    #: An empty match is reported, and then the search resumes one character
    #: later, so a pattern that can match nothing still terminates.
    #:
    #: ```
    #: let hits = Regex.compile(r"\d+").find_all("a1bb22ccc333")
    #: [hit.text() for hit in hits]   # ["1", "22", "333"]
    #: ```
    pub fn find_all(self, text: str) -> list[Match] {
        let codes = _codes(text)
        let out: list[Match] = []
        var position = 0
        while position <= codes.len() {
            let slots = self._search(codes, position)
            if slots.len() == 0 { break }
            out.push(Match(text, slots))
            position = slots[1] > slots[0] ? slots[1] : slots[1] + 1
        }
        return out
    }

    #: Replace the first match in `text` with `replacement`.
    #:
    #: ```
    #: Regex.compile(r"\d+").replace("a1b2", "#")   # "a#b2"
    #: ```
    pub fn replace(self, text: str, replacement: str) -> str {
        return self._substitute(text, replacement, 1)
    }

    #: Replace every match in `text` with `replacement`.
    #:
    #: In `replacement`, `$0` is the whole match, `$1` the first group and so
    #: on, and `$$` is a literal dollar sign. Digits are read as far as they
    #: name a group the pattern actually has, so `$12` is group 12 only when
    #: there are twelve; otherwise it is group 1 followed by "2". A `$n` that
    #: names no group at all is left as written, and a group that did not
    #: participate expands to nothing.
    #:
    #: ```
    #: let mail = Regex.compile(r"(\w+)@(\w+)")
    #: mail.replace_all("bob@example", "$2:$1")   # "example:bob"
    #: ```
    pub fn replace_all(self, text: str, replacement: str) -> str {
        return self._substitute(text, replacement, 0)
    }

    #: Split `text` around every match, dropping the matched separators.
    #:
    #: `max_splits` caps the number of splits, leaving the rest of the text in
    #: the final element; 0 means no limit.
    #:
    #: ```
    #: Regex.compile(r"\s*,\s*").split("a , b,c")     # ["a", "b", "c"]
    #: Regex.compile(",").split("a,b,c", 1)           # ["a", "b,c"]
    #: ```
    pub fn split(self, text: str, max_splits: int = 0) -> list[str] {
        if max_splits < 0 {
            throw ValueError(f"max_splits must be non-negative, got {max_splits}")
        }
        let out: list[str] = []
        var last = 0
        var count = 0
        for hit in self.find_all(text) {
            if max_splits > 0 and count >= max_splits { break }
            out.push(text[last:hit.start])
            last = hit.end
            count += 1
        }
        out.push(text[last:text.len()])
        return out
    }

    fn _substitute(self, text: str, replacement: str, limit: int) -> str {
        let out: list[str] = []
        var last = 0
        var count = 0
        for hit in self.find_all(text) {
            if limit > 0 and count >= limit { break }
            out.push(text[last:hit.start])
            out.push(_expand(hit, replacement))
            last = hit.end
            count += 1
        }
        out.push(text[last:text.len()])
        return "".join(out)
    }

    #: Run the simulation start `start`, returning the capture slots of the
    #: match, or an empty list when there is none.
    #:
    #: Two thread lists are kept: `current` holds the states alive at this
    #: position, `next` those reached by consuming one character. Both are in
    #: priority order, and a state is added at most once per position — that
    #: cap is what bounds the work at O(program size) per character however
    #: many ways there are to reach a state.
    fn _search(self, codes: list[int], start: int) -> list[int] {
        let length = codes.len()
        let size = self.program.len()
        var current: list[Thread] = []
        var next: list[Thread] = []
        var seen_current = [-1 for _ in 0..size]
        var seen_next = [-1 for _ in 0..size]
        var generation = 1
        var current_generation = 1
        var matched: list[int] = []
        var position = start
        loop {
            # A fresh thread joins at the back of the list at every position
            # until something matches: lowest priority, so a match that started
            # earlier always wins, which is what "leftmost" means. Injecting
            # here rather than restarting the search at each position is what
            # keeps an unanchored search linear.
            if matched.len() == 0 {
                let fresh = [-1 for _ in 0..self.slot_count]
                self._add_thread(
                    current,
                    seen_current,
                    current_generation,
                    0,
                    fresh,
                    codes,
                    position
                )
            }
            generation += 1
            let next_generation = generation
            var index = 0
            while index < current.len() {
                let thread = current[index]
                let instruction = self.program[thread.program_counter]
                if instruction.op == OP_CHAR and position < length {
                    let accepted = instruction.char_set?.matches(codes[position]) ?? false
                    if accepted {
                        self._add_thread(
                            next,
                            seen_next,
                            next_generation,
                            thread.program_counter + 1,
                            thread.slots,
                            codes,
                            position + 1
                        )
                    }
                } elif instruction.op == OP_MATCH {
                    matched = thread.slots
                    # Everything after this thread is a lower-priority
                    # alternative that a backtracking engine would never have
                    # reached either, so cutting them is what makes the result
                    # leftmost-*first* rather than leftmost-longest.
                    break
                }
                index += 1
            }
            let spare_threads = current
            current = next
            next = spare_threads
            next.clear()
            let spare_seen = seen_current
            seen_current = seen_next
            seen_next = spare_seen
            current_generation = next_generation
            if position >= length { break }
            if matched.len() > 0 and current.len() == 0 { break }
            position += 1
        }
        return matched
    }

    #: Follow every instruction that consumes no input, adding the states that
    #: do to `threads`.
    #:
    #: The stack makes this a depth-first walk with the preferred branch of
    #: each `Split` on top, so threads land in `threads` in exactly the order a
    #: backtracking engine would try them. `seen` stamps each program counter
    #: with the current generation, which both prevents duplicates and stops an
    #: empty loop body start spinning.
    fn _add_thread(
        self,
        threads: list[Thread],
        seen: list[int],
        generation: int,
        program_counter: int,
        slots: list[int],
        codes: list[int],
        position: int
    ) -> void {
        let stack: list[Thread] = [Thread(program_counter, slots)]
        while stack.len() > 0 {
            let thread = stack.pop()
            if seen[thread.program_counter] == generation { continue }
            seen[thread.program_counter] = generation
            let instruction = self.program[thread.program_counter]
            if instruction.op == OP_JUMP {
                stack.push(Thread(instruction.x, thread.slots))
            } elif instruction.op == OP_SPLIT {
                stack.push(Thread(instruction.y, thread.slots))
                stack.push(Thread(instruction.x, thread.slots))
            } elif instruction.op == OP_SAVE {
                # Copy before writing: several live threads share one slot
                # list, and only this one is recording a group boundary.
                let updated = [value for value in thread.slots]
                updated[instruction.x] = position
                stack.push(Thread(thread.program_counter + 1, updated))
            } elif instruction.op == OP_ASSERT {
                if _assertion_holds(instruction.x, codes, position) {
                    stack.push(Thread(thread.program_counter + 1, thread.slots))
                }
            } else {
                threads.push(thread)
            }
        }
    }

    fn __str__(self) -> str {
        return f"Regex(/{self.source}/, {self.program.len()} instructions)"
    }
}

#: One NFA instruction. `x` and `y` are program counters, except in `OP_SAVE`
#: where `x` is a capture slot and in `OP_ASSERT` where it is an assertion
#: kind.
class Instruction {
    pub var op: int
    pub var x: int
    pub var y: int
    pub var char_set: CharClass?

    pub fn init(self, op: int, x: int = 0, y: int = 0, char_set: CharClass? = null) {
        self.op = op
        self.x = x
        self.y = y
        self.char_set = char_set
    }
}

#: A live simulation state: where in the program, and the capture slots
#: recorded on the way there.
class Thread {
    pub var program_counter: int
    pub var slots: list[int]

    pub fn init(self, program_counter: int, slots: list[int]) {
        self.program_counter = program_counter
        self.slots = slots
    }
}

#: A set of characters: explicit ranges, built-in predicates, and one overall
#: negation start a leading `^`.
#:
#: Predicates are kept as predicates instead of being expanded into ranges so
#: that `\w` stays one test rather than several hundred, and so that `[\D\s]`
#: means what it says.
class CharClass {
    pub var ranges: list[int]
    pub var predicates: list[int]
    pub var negated: bool

    pub fn init(self, negated: bool = false) {
        self.ranges = []
        self.predicates = []
        self.negated = negated
    }

    pub fn add_range(self, low: int, high: int) -> void {
        self.ranges.push(low)
        self.ranges.push(high)
    }

    pub fn add_predicate(self, kind: int, negated: bool) -> void {
        self.predicates.push(kind)
        self.predicates.push(negated ? 1 : 0)
    }

    pub fn matches(self, code: int) -> bool {
        var hit = false
        var index = 0
        while index < self.ranges.len() {
            if code >= self.ranges[index] and code <= self.ranges[index + 1] {
                hit = true
                break
            }
            index += 2
        }
        var probe = 0
        while not hit and probe < self.predicates.len() {
            let wanted = self.predicates[probe + 1] == 0
            if _predicate_holds(self.predicates[probe], code) == wanted { hit = true }
            probe += 2
        }
        return hit != self.negated
    }
}

#: A parsed pattern. `children` holds the operands: one for a group or a
#: repeat, several for a concatenation or an alternation, none otherwise.
class Node {
    pub var kind: int
    pub var children: list[Node]
    pub var char_set: CharClass?
    pub var low: int
    pub var high: int
    pub var lazy: bool
    pub var index: int
    pub var assertion: int

    pub fn init(self, kind: int) {
        self.kind = kind
        self.children = []
        self.char_set = null
        self.low = 0
        self.high = -1
        self.lazy = false
        self.index = -1
        self.assertion = AT_START
    }
}

#: A recursive descent parser over the pattern, one method per precedence
#: level: alternation of concatenations of quantified atoms.
class Parser {
    pub var groups: int
    let pattern: str
    let chars: list[str]
    var index: int

    pub fn init(self, pattern: str) {
        self.pattern = pattern
        self.chars = _chars(pattern)
        self.index = 0
        self.groups = 0
    }

    pub fn parse(self) -> Node {
        let tree = self._alternation()
        if self.index < self.chars.len() { throw self._error("unmatched ')'") }
        return tree
    }

    fn _alternation(self) -> Node {
        let branches: list[Node] = [self._concatenation()]
        while self._peek() == "|" {
            self.index += 1
            branches.push(self._concatenation())
        }
        if branches.len() == 1 { return branches[0] }
        let node = Node(NODE_ALTERNATE)
        node.children = branches
        return node
    }

    fn _concatenation(self) -> Node {
        let items: list[Node] = []
        while self.index < self.chars.len() and self._peek() != "|" and self._peek() != ")" {
            items.push(self._quantified())
        }
        if items.len() == 0 { return Node(NODE_EMPTY) }
        if items.len() == 1 { return items[0] }
        let node = Node(NODE_CONCAT)
        node.children = items
        return node
    }

    fn _quantified(self) -> Node {
        var atom = self._atom()
        loop {
            var low = 0
            var high = -1
            let ch = self._peek()
            if ch == "*" {
                self.index += 1
            } elif ch == "+" {
                low = 1
                self.index += 1
            } elif ch == "?" {
                high = 1
                self.index += 1
            } elif ch == "{" and self._looks_like_bounds() {
                let bounds = self._bounds()
                low = bounds[0]
                high = bounds[1]
            } else {
                break
            }
            if atom.kind == NODE_ASSERT {
                throw self._error("an anchor cannot be repeated")
            }
            if atom.kind == NODE_REPEAT {
                throw self._error("nested quantifier: parenthesise the inner one")
            }
            let repeat = Node(NODE_REPEAT)
            repeat.children = [atom]
            repeat.low = low
            repeat.high = high
            # A trailing '?' makes the quantifier lazy: prefer the shortest
            # match rather than the longest.
            if self._peek() == "?" {
                repeat.lazy = true
                self.index += 1
            }
            atom = repeat
        }
        return atom
    }

    fn _atom(self) -> Node {
        let ch = self._peek()
        if ch == "" { throw self._error("pattern ends where an expression was expected") }
        if ch == "(" { return self._group() }
        if ch == "[" { return self._char_class() }
        if ch == "\\" { return self._escape() }
        if ch == "." {
            self.index += 1
            let node = Node(NODE_CHAR)
            let any = CharClass(true)
            any.add_range(10, 10)
            node.char_set = any
            return node
        }
        if ch == "^" or ch == "$" {
            self.index += 1
            let node = Node(NODE_ASSERT)
            node.assertion = ch == "^" ? AT_START : AT_END
            return node
        }
        if ch == "*" or ch == "+" or ch == "?" {
            throw self._error(f"'{ch}' has nothing to repeat")
        }
        if ch == ")" { throw self._error("unmatched ')'") }
        self.index += 1
        return _literal(ord(ch))
    }

    fn _group(self) -> Node {
        self.index += 1
        var capture = -1
        if self._peek() == "?" {
            if self._peek_at(1) != ":" {
                throw self._error("only '(?:' is supported; there is no lookaround")
            }
            self.index += 2
        } else {
            self.groups += 1
            capture = self.groups
        }
        let inner = self._alternation()
        if self._peek() != ")" { throw self._error("unmatched '('") }
        self.index += 1
        let node = Node(NODE_GROUP)
        node.children = [inner]
        node.index = capture
        return node
    }

    fn _escape(self) -> Node {
        self.index += 1
        let ch = self._peek()
        if ch == "" { throw self._error("pattern ends with a backslash") }
        self.index += 1
        if ch in _CLASS_ESCAPES {
            let node = Node(NODE_CHAR)
            let set = CharClass()
            # The uppercase spelling is the negated one: \D, \W, \S.
            set.add_predicate(_CLASS_ESCAPES[ch], ch != ch.lower())
            node.char_set = set
            return node
        }
        if ch == "b" or ch == "B" {
            let node = Node(NODE_ASSERT)
            node.assertion = ch == "b" ? AT_WORD : AT_NOT_WORD
            return node
        }
        return _literal(self._escape_code(ch))
    }

    fn _char_class(self) -> Node {
        self.index += 1
        let set = CharClass()
        if self._peek() == "^" {
            set.negated = true
            self.index += 1
        }
        var first = true
        loop {
            let ch = self._peek()
            if ch == "" { throw self._error("unterminated character class") }
            # A ']' in first position is the character, not the terminator.
            if ch == "]" and not first {
                self.index += 1
                break
            }
            first = false
            var low = 0
            if ch == "\\" {
                self.index += 1
                let escaped = self._peek()
                self.index += 1
                if escaped == "" { throw self._error("pattern ends with a backslash") }
                if escaped in _CLASS_ESCAPES {
                    set.add_predicate(_CLASS_ESCAPES[escaped], escaped != escaped.lower())
                    continue
                }
                # Inside a class \b is a backspace; a boundary would be
                # meaningless here.
                low = escaped == "b" ? 8 : self._escape_code(escaped)
            } else {
                self.index += 1
                low = ord(ch)
            }
            # A '-' before the closing bracket is itself a character.
            if self._peek() == "-" and self._peek_at(1) != "]" and self._peek_at(1) != "" {
                self.index += 1
                var high = 0
                let bound = self._peek()
                self.index += 1
                if bound == "\\" {
                    let escaped = self._peek()
                    self.index += 1
                    if escaped == "" { throw self._error("pattern ends with a backslash") }
                    high = self._escape_code(escaped)
                } else {
                    high = ord(bound)
                }
                if high < low {
                    throw self._error(f"range runs backwards, start {low} to {high}")
                }
                set.add_range(low, high)
            } else {
                set.add_range(low, low)
            }
        }
        let node = Node(NODE_CHAR)
        node.char_set = set
        return node
    }

    fn _escape_code(self, ch: str) -> int {
        if ch in _CONTROL_ESCAPES { return _CONTROL_ESCAPES[ch] }
        # Escaped punctuation is that punctuation, which is what makes
        # `std.str.escape_regex` work. An escaped letter or digit that means
        # nothing is a typo, and passing it through as a literal would hide it.
        if _is_alphanumeric(ch) { throw self._error(f"unknown escape '\\{ch}'") }
        return ord(ch)
    }

    fn _looks_like_bounds(self) -> bool {
        var scan = self.index + 1
        var digits = 0
        while scan < self.chars.len() and _digit_value(self.chars[scan]) >= 0 {
            scan += 1
            digits += 1
        }
        if digits == 0 { return false }
        if scan < self.chars.len() and self.chars[scan] == "," {
            scan += 1
            while scan < self.chars.len() and _digit_value(self.chars[scan]) >= 0 { scan += 1 }
        }
        return scan < self.chars.len() and self.chars[scan] == "}"
    }

    fn _bounds(self) -> tuple[int, int] {
        self.index += 1
        let low = self._count()
        var high = low
        if self._peek() == "," {
            self.index += 1
            high = self._peek() == "}" ? -1 : self._count()
        }
        self.index += 1
        if high >= 0 and high < low {
            throw self._error(f"repeat {{{low},{high}}} counts down")
        }
        return (low, high)
    }

    fn _count(self) -> int {
        var value = 0
        while _digit_value(self._peek()) >= 0 {
            value = value * 10 + _digit_value(self._peek())
            self.index += 1
            if value > MAX_REPEAT {
                throw self._error(f"repeat count must be at most {MAX_REPEAT}")
            }
        }
        return value
    }

    fn _peek(self) -> str { return self._peek_at(0) }

    fn _peek_at(self, offset: int) -> str {
        let at = self.index + offset
        if at >= self.chars.len() { return "" }
        return self.chars[at]
    }

    fn _error(self, message: str) -> ValueError {
        return ValueError(f"{message} at index {self.index} of pattern /{self.pattern}/")
    }
}

#: Compile a parse tree into an NFA program.
#:
#: `Save 0` and `Save 1` around the whole tree record the span of the match
#: itself, so group 0 needs no special case anywhere else.
fn _compile(tree: Node) -> list[Instruction] {
    let program: list[Instruction] = []
    _emit(program, Instruction(OP_SAVE, 0))
    _compile_node(program, tree)
    _emit(program, Instruction(OP_SAVE, 1))
    _emit(program, Instruction(OP_MATCH))
    return program
}

fn _emit(program: list[Instruction], instruction: Instruction) -> int {
    if program.len() >= MAX_PROGRAM {
        throw ValueError(
            f"pattern expands past {MAX_PROGRAM} instructions; lower its repeat counts"
        )
    }
    program.push(instruction)
    return program.len() - 1
}

fn _compile_node(program: list[Instruction], node: Node) -> void {
    if node.kind == NODE_EMPTY {
        return
    }
    if node.kind == NODE_CHAR {
        _emit(program, Instruction(OP_CHAR, 0, 0, node.char_set))
    } elif node.kind == NODE_ASSERT {
        _emit(program, Instruction(OP_ASSERT, node.assertion))
    } elif node.kind == NODE_CONCAT {
        for child in node.children { _compile_node(program, child) }
    } elif node.kind == NODE_GROUP {
        if node.index < 0 {
            _compile_node(program, node.children[0])
        } else {
            _emit(program, Instruction(OP_SAVE, 2 * node.index))
            _compile_node(program, node.children[0])
            _emit(program, Instruction(OP_SAVE, 2 * node.index + 1))
        }
    } elif node.kind == NODE_ALTERNATE {
        _compile_alternate(program, node)
    } else {
        _compile_repeat(program, node)
    }
}

#: Chain the branches: each one but the last is guarded by a `Split` that
#: prefers it and falls through to the next, and ends with a `Jump` past them
#: all. Preferring the earlier branch is what makes alternation ordered.
fn _compile_alternate(program: list[Instruction], node: Node) -> void {
    let jumps: list[int] = []
    let last = node.children.len() - 1
    for (position, branch) in node.children.enumerate() {
        if position == last {
            _compile_node(program, branch)
        } else {
            let split = _emit(program, Instruction(OP_SPLIT))
            program[split].x = program.len()
            _compile_node(program, branch)
            jumps.push(_emit(program, Instruction(OP_JUMP)))
            program[split].y = program.len()
        }
    }
    for jump in jumps { program[jump].x = program.len() }
}

#: Expand a repeat into copies of its body.
#:
#: `a{2,4}` becomes `aa` and then two optional copies, and `a{2,}` becomes `a`
#: followed by `a+`. Expansion is why a bound is capped: the program, and so
#: the cost of every match, grows with it.
fn _compile_repeat(program: list[Instruction], node: Node) -> void {
    let body = node.children[0]
    if node.high < 0 {
        # `a{3,}` is `aa` followed by `a+`: the last required copy is the one
        # the loop repeats, so only low - 1 copies are emitted ahead of it.
        let required = node.low > 0 ? node.low - 1 : 0
        for _ in 0..required { _compile_node(program, body) }
        _compile_loop(program, body, node.lazy, node.low == 0)
        return
    }
    for _ in 0..node.low { _compile_node(program, body) }
    let splits: list[int] = []
    let starts: list[int] = []
    for _ in 0..node.high - node.low {
        splits.push(_emit(program, Instruction(OP_SPLIT)))
        starts.push(program.len())
        _compile_node(program, body)
    }
    let end = program.len()
    for (position, split) in splits.enumerate() {
        _patch(program, split, starts[position], end, node.lazy)
    }
}

#: Emit `body*` when `optional`, `body+` otherwise.
#:
#: A star tests before the body and jumps back after it; a plus falls into the
#: body and tests after it. Either way the `Split` prefers the body when greedy
#: and the exit when lazy — that one bit is the whole difference between `.*`
#: and `.*?`.
fn _compile_loop(program: list[Instruction], body: Node, lazy: bool, optional: bool) -> void {
    if optional {
        let split = _emit(program, Instruction(OP_SPLIT))
        let start = program.len()
        _compile_node(program, body)
        _emit(program, Instruction(OP_JUMP, split))
        _patch(program, split, start, program.len(), lazy)
        return
    }
    let start = program.len()
    _compile_node(program, body)
    let split = _emit(program, Instruction(OP_SPLIT))
    _patch(program, split, start, program.len(), lazy)
}

fn _patch(program: list[Instruction], split: int, preferred: int, other: int, lazy: bool) -> void {
    program[split].x = lazy ? other : preferred
    program[split].y = lazy ? preferred : other
}

fn _assertion_holds(kind: int, codes: list[int], position: int) -> bool {
    if kind == AT_START { return position == 0 }
    if kind == AT_END { return position == codes.len() }
    let before = position > 0 and _is_word_code(codes[position - 1])
    let after = position < codes.len() and _is_word_code(codes[position])
    let boundary = before != after
    return kind == AT_WORD ? boundary : not boundary
}

#: Substitute `$0`..`$n` and `$$` in `replacement` start the groups of `hit`.
fn _expand(hit: Match, replacement: str) -> str {
    let chars = _chars(replacement)
    let out: list[str] = []
    var index = 0
    while index < chars.len() {
        let ch = chars[index]
        if ch != "$" or index + 1 >= chars.len() {
            out.push(ch)
            index += 1
            continue
        }
        if chars[index + 1] == "$" {
            out.push("$")
            index += 2
            continue
        }
        # Take digits only while they still name a group that exists, so "$12"
        # is group 12 in a pattern with twelve groups and group 1 in one with
        # fewer.
        var group = 0
        var scan = index + 1
        while scan < chars.len() and
            _digit_value(chars[scan]) >= 0 and
            group * 10 + _digit_value(chars[scan]) <= hit.group_count() {
            group = group * 10 + _digit_value(chars[scan])
            scan += 1
        }
        if scan == index + 1 {
            out.push(ch)
            index += 1
            continue
        }
        out.push(hit.group(group) ?? "")
        index = scan
    }
    return "".join(out)
}

fn _literal(code: int) -> Node {
    let node = Node(NODE_CHAR)
    let set = CharClass()
    set.add_range(code, code)
    node.char_set = set
    return node
}

fn _predicate_holds(kind: int, code: int) -> bool {
    if kind == CLASS_DIGIT { return code >= 48 and code <= 57 }
    if kind == CLASS_WORD { return _is_word_code(code) }
    return code == 32 or code >= 9 and code <= 13
}

fn _is_word_code(code: int) -> bool {
    if code >= 48 and code <= 57 { return true }
    if code >= 65 and code <= 90 { return true }
    if code >= 97 and code <= 122 { return true }
    return code == 95
}

fn _digit_value(ch: str) -> int {
    if ch.len() != 1 { return -1 }
    let code = ord(ch)
    if code < 48 or code > 57 { return -1 }
    return code - 48
}

fn _is_alphanumeric(ch: str) -> bool {
    if ch.len() != 1 { return false }
    let code = ord(ch)
    if code >= 48 and code <= 57 { return true }
    if code >= 65 and code <= 90 { return true }
    return code >= 97 and code <= 122
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
