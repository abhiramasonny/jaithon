# expect: E0701

class Vault {
    var code: int
    fn init(self) { self.code = 0 }
    fn unlock(self) -> int { return self.code }
}

let v = Vault()
print(v.unlock())
