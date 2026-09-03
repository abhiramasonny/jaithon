#: The four combinatoric itertools: `product`, `permutations`, `combinations`
#: and `combinations_with_replacement`.
#:
#: All four buffer their input, because all four replay it: an arrangement is
#: an index vector over a pool that has to be there in full before the first
#: result exists. What they do not buffer is their *output* — the index vector
#: advances one step per `next`, so `product` over four ten-element pools
#: costs ten thousand results only if ten thousand are asked for.
#:
#: The emission order is CPython's, which is the order the index vector counts
#: in: rightmost position varying fastest for `product` and
#: `combinations_with_replacement`, and lexicographic in the *positions* of the
#: input — not in the values — for `permutations` and `combinations`. An input
#: that repeats a value therefore produces results that repeat, exactly as in
#: CPython.
#:
#: ```
#: from std.itertools import product, permutations, combinations
#: product([1, 2], "ab").collect()      # [(1, "a"), (1, "b"), (2, "a"), (2, "b")]
#: permutations([1, 2, 3], 2).collect() # [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]
#: combinations("abc", 2).collect()     # [("a", "b"), ("a", "c"), ("b", "c")]
#: ```
from std.iter import Iter, iter

class _Product extends Iter[any] {
    let pools: list[any]
    let indices: list[int] = []
    var first: bool = true
    var done: bool = false

    #: Count an odometer over `pools`, one digit per pool.
    pub fn init(self, pools: list[any]) {
        self.pools = pools
        for pool in pools {
            if pool.len() == 0 { self.done = true }
            self.indices.push(0)
        }
    }

    pub fn next(self) -> any {
        if self.done { return null }
        if self.first {
            self.first = false
            return self._emit()
        }
        var i = self.pools.len() - 1
        while i >= 0 {
            self.indices[i] += 1
            if self.indices[i] < self.pools[i].len() { return self._emit() }
            self.indices[i] = 0
            i -= 1
        }
        self.done = true
        return null
    }

    fn _emit(self) -> any {
        let row: list[any] = []
        for i in 0..self.pools.len() { row.push(self.pools[i][self.indices[i]]) }
        return tuple(row)
    }
}

class _WithReplacement extends Iter[any] {
    let pool: list[any]
    let n: int
    let r: int
    let indices: list[int] = []
    var first: bool = true
    var done: bool = false

    #: Advance a non-decreasing index vector of length `r` over `pool`.
    pub fn init(self, pool: list[any], r: int) {
        self.pool = pool
        self.n = pool.len()
        self.r = r
        self.done = self.n == 0 and r > 0
        for i in 0..r { self.indices.push(0) }
    }

    pub fn next(self) -> any {
        if self.done { return null }
        if self.first {
            self.first = false
            return self._emit()
        }
        var i = self.r - 1
        while i >= 0 and self.indices[i] == self.n - 1 { i -= 1 }
        if i < 0 {
            self.done = true
            return null
        }
        let raised = self.indices[i] + 1
        var j = i
        while j < self.r {
            self.indices[j] = raised
            j += 1
        }
        return self._emit()
    }

    fn _emit(self) -> any {
        let row: list[any] = []
        for i in 0..self.r { row.push(self.pool[self.indices[i]]) }
        return tuple(row)
    }
}

#: The cartesian product of several iterables.
#:
#: A class rather than a function because CPython's `repeat` is keyword-only
#: behind `*iterables`, and Jaithon refuses a parameter after a rest parameter
#: (E0112); `product.repeated` carries it instead. With no iterables — or with
#: `repeat` zero — the product is the empty tuple, once, which is the identity
#: the recursive definition wants; with any empty iterable it is nothing.
#:
#: ```
#: product("ab", [0, 1]).collect()      # [("a", 0), ("a", 1), ("b", 0), ("b", 1)]
#: product.repeated(2, "ab").collect()  # [("a", "a"), ("a", "b"), ("b", "a"), ("b", "b")]
#: product().collect()                  # [()]
#: product("ab", []).collect()          # []
#: ```
pub class product extends Iter[any] {
    let inner: Iter[any]

    #: Pair every element of every iterable with every other, rightmost first.
    pub fn init(self, ...iterables: any) {
        self.inner = _spin(iterables, 1)
    }

    #: The product of `iterables` with itself, `repeat` times over.
    pub static fn repeated(repeat: int, ...iterables: any) -> Iter[any] {
        return _spin(iterables, repeat)
    }

    #: Advance the odometer one place and read the pools it points at.
    pub fn next(self) -> any {
        return self.inner.next()
    }
}

#: The ordered arrangements of `r` of an iterable's elements.
#:
#: `r` defaults to null, meaning all of them. An `r` above the pool's size
#: yields nothing, and an `r` of zero yields the empty tuple once — both
#: CPython's answers.
#:
#: ```
#: permutations("abc").collect().len()     # 6
#: permutations("abc", 2).collect().len()  # 6
#: permutations("ab", 3).collect()         # []
#: permutations("ab", 0).collect()         # [()]
#: ```
pub fn permutations(iterable: any, r: int? = null) -> Iter[any] {
    return iter(iterable).permutations(r).map(|values| tuple(values))
}

#: The unordered selections of `r` of an iterable's elements, in index order.
#:
#: An `r` above the pool's size yields nothing; an `r` of zero yields the empty
#: tuple once.
#:
#: ```
#: combinations("abcd", 2).collect().len()   # 6
#: combinations("ab", 3).collect()           # []
#: combinations("ab", 0).collect()           # [()]
#: ```
pub fn combinations(iterable: any, r: int) -> Iter[any] {
    return iter(iterable).combinations(r).map(|values| tuple(values))
}

#: The selections of `r` elements that may repeat, in index order.
#:
#: Unlike `combinations`, an `r` above the pool's size is perfectly ordinary —
#: only an empty pool with a positive `r` yields nothing.
#:
#: ```
#: combinations_with_replacement("abc", 2).collect().len()   # 6
#: combinations_with_replacement("ab", 3).collect().len()    # 4
#: combinations_with_replacement("", 2).collect()            # []
#: combinations_with_replacement("", 0).collect()            # [()]
#: ```
pub fn combinations_with_replacement(iterable: any, r: int) -> Iter[any] {
    if r < 0 { throw ValueError("r must be non-negative") }
    return _WithReplacement(iter(iterable).collect(), r)
}

fn _spin(iterables: list[any], repeat: int) -> Iter[any] {
    if repeat < 0 { throw ValueError("repeat argument cannot be negative") }
    let base: list[any] = []
    for source in iterables { base.push(iter(source).collect()) }
    let pools: list[any] = []
    for round in 0..repeat {
        for pool in base { pools.push(pool) }
    }
    return _Product(pools)
}
