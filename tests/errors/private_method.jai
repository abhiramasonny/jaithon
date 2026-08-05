# expect: E0701
# Calling a private method from outside the declaring class.

class Vault {
    var code: int
    fn init(self) { self.code = 0 }
    fn unlock(self) -> int { return self.code }
}

let v = Vault()
print(v.unlock())
