#: Classes, visibility, inheritance, traits, enums, properties, and operator
#: overloading (spec §7).
class Point {
    pub var x: float
    pub var y: float

    fn init(self, x: float, y: float) {
        self.x = x
        self.y = y
    }

    pub fn length(self) -> float {
        return (self.x ** 2 + self.y ** 2) ** 0.5
    }

    pub static fn origin() -> Point {
        return Point(0.0, 0.0)
    }

    fn __add__(self, other: Point) -> Point {
        return Point(self.x + other.x, self.y + other.y)
    }

    fn __eq__(self, other: Point) -> bool {
        return self.x == other.x and self.y == other.y
    }

    fn __str__(self) -> str {
        return f"Point({self.x}, {self.y})"
    }
}

let p = Point(3.0, 4.0)
print(p.length())
print(p)
print(Point.origin())
print(p + Point(1.0, 1.0))
print(p == Point(3.0, 4.0))
print(p == Point(0.0, 0.0))

# --- visibility is enforced -------------------------------------------------
class Account {
    pub var owner: str
    var balance: float

    fn init(self, owner: str, opening: float) {
        self.owner = owner
        self.balance = opening
    }

    pub fn deposit(self, amount: float) -> void {
        if amount <= 0.0 { throw ValueError(f"amount must be positive, got {amount}") }
        self.balance += amount
    }

    pub fn balance_of(self) -> float {
        return self.balance
    }
}

let account = Account("Ada", 100.0)
account.deposit(50.0)
print(account.owner)
print(account.balance_of())

try {
    account.deposit(-1.0)
} catch e: ValueError {
    print(f"rejected: {e.message}")
}

# --- inheritance ------------------------------------------------------------
class Animal {
    pub var name: str

    fn init(self, name: str) {
        self.name = name
    }

    pub fn speak(self) -> str {
        return f"{self.name} makes a sound"
    }

    pub fn describe(self) -> str {
        return f"<{self.speak()}>"
    }
}

class Dog extends Animal {
    pub var breed: str

    fn init(self, name: str, breed: str) {
        super(name)
        self.breed = breed
    }

    pub fn speak(self) -> str {
        return f"{self.name} barks"
    }

    pub fn parent_speak(self) -> str {
        return super.speak()
    }
}

let d = Dog("Rex", "Husky")
print(d.speak())
print(d.describe())
print(d.parent_speak())
print(d.breed)
print(isinstance(d, Animal))
print(isinstance(d, Dog))

# --- traits -----------------------------------------------------------------
trait Shape {
    fn area(self) -> float

    fn name(self) -> str { return "shape" }
}

class Circle: Shape {
    var radius: float

    fn init(self, radius: float) { self.radius = radius }

    pub fn area(self) -> float { return 3.141592653589793 * self.radius ** 2 }

    pub fn name(self) -> str { return "circle" }
}

class Square: Shape {
    var side: float

    fn init(self, side: float) { self.side = side }

    pub fn area(self) -> float { return self.side ** 2 }
}

fn total_area(shapes: list[Shape]) -> float {
    var sum = 0.0
    for s in shapes { sum += s.area() }
    return sum
}

let shapes: list[Shape] = [Circle(1.0), Square(2.0)]
print(f"{total_area(shapes):.4f}")
print(shapes[0].name())
print(shapes[1].name())

# --- properties -------------------------------------------------------------
class Rect {
    var w: float
    var h: float

    fn init(self, w: float, h: float) {
        self.w = w
        self.h = h
    }

    pub get area() -> float { return self.w * self.h }

    pub set width(v: float) {
        if v < 0.0 { throw ValueError("width must be non-negative") }
        self.w = v
    }
}

let r = Rect(2.0, 3.0)
print(r.area)
r.width = 5.0
print(r.area)

# --- enums ------------------------------------------------------------------
enum Color { Red, Green, Blue }

fn hex_of(c: Color) -> str {
    return match c {
        Color.Red => "#ff0000",
        Color.Green => "#00ff00",
        Color.Blue => "#0000ff",
    }
}
print(hex_of(Color.Green))

enum Expr { Num(value: float), Add(left: Expr, right: Expr), Mul(left: Expr, right: Expr) }

fn eval(e: Expr) -> float {
    return match e {
        Expr.Num(v) => v,
        Expr.Add(l, r) => eval(l) + eval(r),
        Expr.Mul(l, r) => eval(l) * eval(r),
    }
}

let tree = Expr.Add(Expr.Num(2.0), Expr.Mul(Expr.Num(3.0), Expr.Num(4.0)))
print(eval(tree))

# --- generics ---------------------------------------------------------------
class Box[T] {
    var value: T

    fn init(self, value: T) { self.value = value }

    pub fn get(self) -> T { return self.value }

    pub fn map[U](self, f: fn(T) -> U) -> Box[U] { return Box[U](f(self.value)) }
}

let boxed = Box[int](21)
print(boxed.get())
print(boxed.map(|v| f"n={v}").get())
