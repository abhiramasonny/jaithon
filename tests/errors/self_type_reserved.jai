# expect: E0114

class Self {
    pub let n: int
    pub fn init(self, n: int) { self.n = n }
}

print(Self(1).n)
