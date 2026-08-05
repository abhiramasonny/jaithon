# expect: E0114
# `Self` is in scope inside every class, trait and enum body (spec §7.2), so no
# declaration may introduce a type of that name.

class Self {
    pub let n: int
    pub fn init(self, n: int) { self.n = n }
}

print(Self(1).n)
