class N {
    pub var kind: int
    pub var left: int
    pub var right: int
    pub var span: int
    pub var flags: bool

    pub fn init(self, kind: int, a: int, b: int) {
        self.kind = kind
        self.left = a
        self.right = b
        self.span = 0
        self.flags = false
    }
}
var total = 0
var i = 0
while i < 200000 {
    let n = N(3, i, i + 1)
    total = total + n.left + n.right
    i = i + 1
}
print(total)
