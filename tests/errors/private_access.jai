# expect: E0701
# Reaching a private field from outside the declaring class.

class Account {
    var balance: float
    fn init(self) { self.balance = 0.0 }
}

let a = Account()
print(a.balance)
