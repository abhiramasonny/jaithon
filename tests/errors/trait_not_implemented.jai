# expect: E0705

trait Shape {
    fn area(self) -> float
}

class Blob: Shape {
    var size: float
    fn init(self) { self.size = 1.0 }
}

print(Blob())
