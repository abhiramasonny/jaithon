#: A hand-written JSON parser over a generated document.
#:
#: Holdout, and the closest thing in the suite to a whole program: it scans a
#: string a character at a time, cuts substrings out of it, allocates a node per
#: value, inserts into a dict per object, grows a list per array, and recurses.
#: Every other benchmark isolates one of those. Nothing here uses a library —
#: all four ports parse by hand, so the columns are comparable.
#:
#: The grammar is deliberately a subset: objects, arrays, non-negative integers,
#: strings without escapes, and the three literals. Escapes and floats would put
#: the ports' formatting rules, not their parsers, in the measurement.
from std.os import env_or

let LEVEL = env_or("BENCH_LEVEL", "hard")
let SCALE = if LEVEL == "easy" { 16 } elif LEVEL == "medium" { 4 } else { 1 }
# The level shrinks the document rather than the pass count, so every level
# still parses the same shape and the answer changes with it.
let RECORDS = max(2500 // SCALE, 1)
let REPS = 6

const K_NULL = 0
const K_BOOL = 1
const K_INT = 2
const K_STR = 3
const K_LIST = 4
const K_DICT = 5

class JNode {
    pub var kind: int
    pub var num: int
    pub var text: str
    pub var items: list[JNode]
    pub var fields: dict[str, JNode]

    fn init(self, kind: int) {
        self.kind = kind
        self.num = 0
        self.text = ""
        self.items = []
        self.fields = {}
    }
}

class Parser {
    pub var src: str
    pub var at: int
    pub var n: int

    fn init(self, src: str) {
        self.src = src
        self.at = 0
        self.n = src.len()
    }

    pub fn skip(self) -> void {
        while self.at < self.n and self.src[self.at] == " " { self.at += 1 }
    }

    pub fn value(self) -> JNode {
        self.skip()
        let c = self.src[self.at]
        if c == "{" { return self.object() }
        if c == "[" { return self.array() }
        if c == "\"" {
            var node = JNode(K_STR)
            node.text = self.text()
            return node
        }
        if c == "t" {
            self.at += 4
            var node = JNode(K_BOOL)
            node.num = 1
            return node
        }
        if c == "f" {
            self.at += 5
            var node = JNode(K_BOOL)
            node.num = 0
            return node
        }
        if c == "n" {
            self.at += 4
            return JNode(K_NULL)
        }
        var node = JNode(K_INT)
        node.num = self.integer()
        return node
    }

    pub fn text(self) -> str {
        self.at += 1
        let start = self.at
        while self.src[self.at] != "\"" { self.at += 1 }
        let out = self.src[start:self.at]
        self.at += 1
        return out
    }

    pub fn integer(self) -> int {
        var acc = 0
        while self.at < self.n {
            let c = self.src[self.at]
            if c < "0" or c > "9" { break }
            acc = acc * 10 + (ord(c) - 48)
            self.at += 1
        }
        return acc
    }

    pub fn array(self) -> JNode {
        var node = JNode(K_LIST)
        self.at += 1
        self.skip()
        if self.src[self.at] == "]" {
            self.at += 1
            return node
        }
        loop {
            node.items.push(self.value())
            self.skip()
            let c = self.src[self.at]
            self.at += 1
            if c == "]" { break }
        }
        return node
    }

    pub fn object(self) -> JNode {
        var node = JNode(K_DICT)
        self.at += 1
        self.skip()
        if self.src[self.at] == "}" {
            self.at += 1
            return node
        }
        loop {
            self.skip()
            let key = self.text()
            self.skip()
            self.at += 1  # the ':'
            node.fields[key] = self.value()
            self.skip()
            let c = self.src[self.at]
            self.at += 1
            if c == "}" { break }
        }
        return node
    }
}

#: Sums every integer reachable from a node and the length of every string, so
#: the answer depends on the whole tree rather than on its root.
fn walk(node: JNode) -> int {
    let k = node.kind
    if k == K_INT { return node.num }
    if k == K_BOOL { return node.num }
    if k == K_STR { return node.text.len() }
    if k == K_LIST {
        var total = 0
        for item in node.items { total += walk(item) }
        return total
    }
    if k == K_DICT {
        var total = 0
        for (key, val) in node.fields.items() { total += key.len() + walk(val) }
        return total
    }
    return 0
}

fn document(records: int) -> str {
    var parts = []
    parts.push("[")
    var s = 7
    var i = 0
    while i < records {
        if i > 0 { parts.push(", ") }
        s = (s * 1103515245 + 12345) % 2147483648
        let a = s // 65536 % 1000
        let b = s % 997
        let flag = if a % 2 == 0 { "true" } else { "false" }
        parts.push(
            f"{{\"id\": {i}, \"name\": \"item{a}\", \"tags\": [{a}, {b}, {a + b}], \"ok\": {flag}, \"note\": null}}"
        )
        i += 1
    }
    parts.push("]")
    return "".join(parts)
}

fn main() -> int {
    let src = document(RECORDS)
    var total = 0
    for _r in 0..REPS {
        var p = Parser(src)
        total = walk(p.value())
    }
    print(src.len())
    print(total)
    return 0
}
