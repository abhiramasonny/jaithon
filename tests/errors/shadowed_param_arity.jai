# expect: E0602

class Ledger {
    pub fn weigh(self, value: int) -> int { return value }

    pub fn fold(self, weigh: fn(int, int) -> int) -> int {
        return weigh(1)
    }
}
