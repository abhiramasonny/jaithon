# expect: E0602
# A parameter shadows the same-named method, so the call is checked against the
# parameter's `fn` type. The method's one-argument shape does not license it.

class Ledger {
    pub fn weigh(self, value: int) -> int { return value }

    pub fn fold(self, weigh: fn(int, int) -> int) -> int {
        return weigh(1)
    }
}
