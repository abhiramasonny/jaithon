class Span {
    pub var start: int
    pub var end: int

    pub fn init(self, s: int, e: int) {
        self.start = s
        self.end = e
    }
}

class Token {
    pub var kind: int
    pub var span: Span
    pub var text: str

    pub fn init(self, k: int, sp: Span, t: str) {
        self.kind = k
        self.span = sp
        self.text = t
    }
}
let src = "identifier_name " * 20000
var toks: list[Token] = []
var i = 0
while i < 100000 {
    let s = i * 16 % 300000
    toks.push(Token(1, Span(s, s + 15), src[s:s + 15]))
    i = i + 1
}
print(toks.len())
