#: The Pike VM: run an NFA program over a text, and the `Regex` and `Match`
#: values that surface it.
from .parse import AT_END, AT_START, AT_WORD, Parser
from .parse import _chars, _digit_value, _is_word_code
from .program import Instruction, _compile
from .program import OP_ASSERT, OP_CHAR, OP_JUMP, OP_MATCH, OP_SAVE, OP_SPLIT

#: One match of a pattern against a text, with the span of every group.
#:
#: Group 0 is the whole match and is always present. A group that took no
#: part in the match reports null, not an empty string — that's how you tell
#: "matched nothing" from "did not match".
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
#: Compilation parses the pattern and builds the NFA program once; matching
#: after that only allocates the thread lists it needs. Hoist the `Regex` out
#: of loops — compiling inside one is the most common way to make regular
#: expressions look slow.
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
    #: Leftmost-first, exactly as a backtracking engine would resolve it among
    #: matches starting at the earliest position. `^` still anchors to the
    #: start of the whole text, not to `start`.
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
    #: `$0` is the whole match, `$1` the first group, and `$$` a literal
    #: dollar sign. Digits are read only as far as they name a group that
    #: exists, so `$12` is group 12 only with twelve groups, else group 1
    #: followed by "2". An unmatched `$n` is left as written; a group that
    #: didn't participate expands to nothing.
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

    #: Run the simulation from `start`, returning the capture slots of the
    #: match, or an empty list when there is none.
    #:
    #: Two thread lists are kept: `current` holds states alive at this
    #: position, `next` those reached by consuming one character, both in
    #: priority order. A state is added at most once per position — that cap
    #: is what bounds the work to O(program size) per character.
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
    #: each `Split` on top, so threads land in priority order — the order a
    #: backtracking engine would try them. `seen` stamps each program counter
    #: with the current generation, which prevents duplicates and stops an
    #: empty loop body from spinning forever.
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

fn _codes(text: str) -> list[int] {
    let out: list[int] = []
    for ch in text { out.push(ord(ch)) }
    return out
}
