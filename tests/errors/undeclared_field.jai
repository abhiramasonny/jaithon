# expect: E0702
class Point {
    pub var x: float
    fn init(self) { self.x = 0.0 }
}

let p = Point()
p.y = 1.0
