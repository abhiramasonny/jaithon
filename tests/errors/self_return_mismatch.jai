# expect: E0705
trait Scalable {
    fn scale(self, factor: int) -> Self
}

class Cents: Scalable {
    pub let amount: int
    pub fn init(self, amount: int) { self.amount = amount }
    pub fn scale(self, factor: int) -> int { return self.amount * factor }
}

print(Cents(1).scale(2))
