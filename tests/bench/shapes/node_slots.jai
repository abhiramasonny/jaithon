class N {
    pub var kind: int
    pub var slots: list[any]

    pub fn init(self, kind: int, a: any, b: any) {
        self.kind = kind
        self.slots = [a, b, 0, false]
    }
}
let LEFT = 0
let RIGHT = 1
var total = 0
var i = 0
while i < 200000 {
    let n = N(3, i, i + 1)
    let l: int = n.slots[LEFT]
    let r: int = n.slots[RIGHT]
    total = total + l + r
    i = i + 1
}
print(total)
