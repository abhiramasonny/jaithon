class Token {
    pub var kind: int
    pub var start: int
    pub var end: int

    pub fn init(self, k: int, s: int, e: int) {
        self.kind = k
        self.start = s
        self.end = e
    }
}
let src = "identifier_name " * 20000
var toks: list[Token] = []
var i = 0
while i < 100000 {
    let s = i * 16 % 300000
    toks.push(Token(1, s, s + 15))
    i = i + 1
}
print(toks.len())
