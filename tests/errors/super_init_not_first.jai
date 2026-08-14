# expect: E0708
class Base {
    pub var n: int
    fn init(self, n: int) { self.n = n }
}

class Derived extends Base {
    pub var m: int
    fn init(self) {
        self.m = 1
        super(2)
    }
}

print(Derived())
