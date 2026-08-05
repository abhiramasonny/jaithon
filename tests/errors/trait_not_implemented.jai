# expect: E0705
# A class listing a trait must implement every required method.

trait Shape {
    fn area(self) -> float
}

class Blob: Shape {
    var size: float
    fn init(self) { self.size = 1.0 }
}

print(Blob())
