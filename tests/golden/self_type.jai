#: The `Self` type (spec §7.2).
#:
#: Inside a trait `Self` is whichever type implements it; inside a class or an
#: enum it is that declaration. It is legal in return position, in parameter
#: position and in a field type, and a trait requirement mentioning `Self` is
#: checked against the implementer with `Self` substituted away — which is what
#: lets a default body promise to hand back the implementing type.
from std.time import Duration

# --- a trait that returns the implementing type ------------------------------
trait Scalable {
    #: Required, so every implementer must promise its own type back.
    fn scale(self, factor: int) -> Self

    #: A default body. `self` is only known to be a `Scalable` here, yet the
    #: declared result is `Self`, and each implementer inherits it at its own
    #: type rather than at `any`.
    fn twice(self) -> Self { return self.scale(2) }

    #: `Self` in parameter position: two implementers are combined only with
    #: their own kind.
    fn bigger(self, other: Self) -> Self {
        if self.size() >= other.size() { return self }
        return other
    }

    fn size(self) -> int
}

class Cents: Scalable {
    pub let amount: int

    pub fn init(self, amount: int) { self.amount = amount }

    pub fn scale(self, factor: int) -> Self { return Cents(self.amount * factor) }

    pub fn size(self) -> int { return self.amount }

    pub fn show(self) -> str { return f"{self.amount}c" }
}

class Grams: Scalable {
    pub let mass: int

    pub fn init(self, mass: int) { self.mass = mass }

    pub fn scale(self, factor: int) -> Self { return Grams(self.mass * factor) }

    pub fn size(self) -> int { return self.mass }

    pub fn show(self) -> str { return f"{self.mass}g" }
}

# The annotations are the point: each is checked, and neither is `any`.
let doubled: Cents = Cents(7).twice()
print(doubled.show())

let heavier: Grams = Grams(3).bigger(Grams(11))
print(heavier.show())
print(Grams(3).scale(5).show())

# --- `Self` in a class, without a trait --------------------------------------
class Counter {
    pub var count: int

    #: A field typed `Self?`: inside a class, `Self` is that class.
    pub var previous: Self? = null

    pub fn init(self, count: int) { self.count = count }

    pub fn step(self) -> Self {
        let next = Counter(self.count + 1)
        next.previous = self
        return next
    }

    #: A static method may name `Self` too.
    pub static fn zero() -> Self { return Counter(0) }
}

let second: Counter = Counter.zero().step().step()
print(second.count)
let first = second.previous
if first != null { print(first.count) }

# --- `Self` in an enum body ---------------------------------------------------
enum Light {
    Red,
    Green,

    pub fn flip(self) -> Self {
        return match self {
            Light.Red => Light.Green,
            Light.Green => Light.Red,
        }
    }
}

let green: Light = Light.Red.flip()
print(green == Light.Green)

# --- a trait extending a trait leaves `Self` abstract one level up -----------
trait Doubling: Scalable {
    fn quadruple(self) -> Self { return self.scale(4) }
}

class Ticks: Doubling {
    pub let n: int

    pub fn init(self, n: int) { self.n = n }

    pub fn scale(self, factor: int) -> Self { return Ticks(self.n * factor) }

    pub fn size(self) -> int { return self.n }
}

let quad: Ticks = Ticks(2).quadruple()
print(quad.n)

# --- std.core's Comparable is now written in terms of `Self` -----------------
#
# `max` used to be declared `-> any` because a trait had no way to say "the
# implementing type"; it says `-> Self` now, so the annotation below is a real
# check rather than a dynamic one.
let later: Duration = Duration.from_seconds(5.0).max(Duration.from_seconds(3.0))
let sooner: Duration = Duration.from_seconds(5.0).min(Duration.from_seconds(3.0))
print(later.to_str())
print(sooner.to_str())
print(Duration.from_seconds(1.0).lt(Duration.from_seconds(2.0)))
