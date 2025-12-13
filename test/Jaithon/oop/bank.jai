class Transaction
    var amount
    var type
    var balance

    func init(self, amount, type, balance)
        self.amount = amount
        self.type = type
        self.balance = balance
    end

    func show(self)
        print self.type + ": $" + str(self.amount) + " | Balance: $" + str(self.balance)
    end
end

class BankAccount
    var owner
    var balance
    var transactions

    func init(self, owner, initial)
        self.owner = owner
        self.balance = initial
        self.transactions = []
        var t = new Transaction(initial, "OPEN", initial)
        _push(self.transactions, t)
    end

    func deposit(self, amount)
        if amount <= 0
            print "Invalid deposit amount"
            return 0
        end
        self.balance = self.balance + amount
        var t = new Transaction(amount, "DEPOSIT", self.balance)
        _push(self.transactions, t)
        print "Deposited $" + str(amount)
        return self.balance
    end

    func withdraw(self, amount)
        if amount <= 0
            print "Invalid withdrawal amount"
            return 0
        end
        if amount > self.balance
            print "Insufficient funds"
            return 0
        end
        self.balance = self.balance - amount
        var t = new Transaction(amount, "WITHDRAW", self.balance)
        _push(self.transactions, t)
        print "Withdrew $" + str(amount)
        return self.balance
    end

    func getBalance(self)
        return self.balance
    end

    func showHistory(self)
        print "Transaction history for " + self.owner + ":"
        var i = 0
        while i < _alen(self.transactions)
            var t = _get(self.transactions, i)
            t.show()
            i = i + 1
        end
    end

    func summary(self)
        print "=== Account Summary ==="
        print "Owner: " + self.owner
        print "Balance: $" + str(self.balance)
        print "Total transactions: " + str(_alen(self.transactions))
    end
end

print "Creating account for Alice with $1000"
var account = new BankAccount("Alice", 1000)

account.deposit(500)
account.deposit(250)
account.withdraw(300)
account.withdraw(100)

print ""
account.showHistory()

print ""
account.summary()
