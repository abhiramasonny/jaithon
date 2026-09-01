class N { pub var kind: int
    pub var fields: dict[str, any]
    pub fn init(self, kind: int, a: any, b: any) { self.kind = kind
        self.fields = {}
        self.fields["left"] = a
        self.fields["right"] = b
        self.fields["span"] = 0
        self.fields["flags"] = false } }
var total = 0
var i = 0
while i < 200000 {
    let n = N(3, i, i + 1)
    let l: int = n.fields["left"]
    let r: int = n.fields["right"]
    total = total + l + r
    i = i + 1
}
print(total)
