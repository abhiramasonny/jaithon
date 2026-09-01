#: The pattern syntax: character classes, the parse tree, and the recursive
#: descent parser that builds one.
const MAX_REPEAT = 1000

pub const AT_START = 0
pub const AT_END = 1
pub const AT_WORD = 2
pub const AT_NOT_WORD = 3

const CLASS_DIGIT = 0
const CLASS_WORD = 1
const CLASS_SPACE = 2

pub const NODE_EMPTY = 0
pub const NODE_CHAR = 1
pub const NODE_CONCAT = 2
pub const NODE_ALTERNATE = 3
pub const NODE_REPEAT = 4
pub const NODE_GROUP = 5
pub const NODE_ASSERT = 6

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

#: A set of characters: explicit ranges, built-in predicates, and one overall
#: negation start a leading `^`.
#:
#: Predicates are kept as predicates instead of being expanded into ranges so
#: that `\w` stays one test rather than several hundred, and so that `[\D\s]`
#: means what it says.
pub class CharClass {
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
pub class Node {
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
pub class Parser {
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
                low = escaped == "b" ? 8 : self._escape_code(escaped)
            } else {
                self.index += 1
                low = ord(ch)
            }
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

pub fn _is_word_code(code: int) -> bool {
    if code >= 48 and code <= 57 { return true }
    if code >= 65 and code <= 90 { return true }
    if code >= 97 and code <= 122 { return true }
    return code == 95
}

pub fn _digit_value(ch: str) -> int {
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

pub fn _chars(text: str) -> list[str] {
    let out: list[str] = []
    for ch in text { out.push(ch) }
    return out
}
