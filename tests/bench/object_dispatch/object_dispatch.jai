#: Method calls through an inline cache, plus field access.
from std.os import env_or

let LEVEL = env_or("BENCH_LEVEL", "hard")
let SCALE = if LEVEL == "easy" { 16 } elif LEVEL == "medium" { 4 } else { 1 }
let ITERS = 10_000_000 // SCALE

class Vec2 {
    pub var x: float
    pub var y: float

    fn init(self, x: float, y: float) {
        self.x = x
        self.y = y
    }

    pub fn add(self, other: Vec2) -> Vec2 {
        return Vec2(self.x + other.x, self.y + other.y)
    }

    pub fn dot(self, other: Vec2) -> float {
        return self.x * other.x + self.y * other.y
    }
}

fn main() -> int {
    var acc = Vec2(0.0, 0.0)
    let step = Vec2(1.0, 2.0)
    var total = 0.0
    for _i in 0..ITERS {
        acc = acc.add(step)
        total += acc.dot(step)
    }
    print(f"{total:.1f}")
    return 0
}
