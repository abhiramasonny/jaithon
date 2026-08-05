# expect: E0705
# A requirement written `-> Self` substitutes to the implementing class, so an
# implementation returning anything else is not compatible (spec §7.2).

trait Scalable {
    fn scale(self, factor: int) -> Self
}

class Cents: Scalable {
    pub let amount: int
    pub fn init(self, amount: int) { self.amount = amount }
    pub fn scale(self, factor: int) -> int { return self.amount * factor }
}

print(Cents(1).scale(2))
