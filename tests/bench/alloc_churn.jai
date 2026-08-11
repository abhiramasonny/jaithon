#: Short-lived objects, which is collector pressure rather than dispatch.
#:
#: Every instance here dies immediately, so the whole cost is allocation and
#: the collections it provokes. `string_build` measures the same thing through
#: strings; this isolates it from formatting.
class Point {
    pub var x: int
    pub var y: int

    fn init(self, x: int, y: int) {
        self.x = x
        self.y = y
    }

    pub fn dot(self, other: Point) -> int {
        return self.x * other.x + self.y * other.y
    }
}

var total = 0
var i = 0
while i < 4_000_000 {
    let a = Point(i % 100, i % 37)
    let b = Point(i % 53, i % 11)
    total = (total + a.dot(b)) % 1000000007
    i += 1
}
print(total)
