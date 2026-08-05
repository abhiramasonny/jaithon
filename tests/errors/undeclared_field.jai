# expect: E0702
# Fields must be declared in the class body; there is no accidental
# attribute creation.

class Point {
    pub var x: float
    fn init(self) { self.x = 0.0 }
}

let p = Point()
p.y = 1.0
