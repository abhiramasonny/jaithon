#: Lazy iterators that compose without building intermediate lists.
#:
#: ```
#: iter([1, 2, 3, 4, 5]).map(|x| x * x).filter(|x| x % 2 == 1).take(2).collect()
#: # [1, 9]
#: ```
#:
#: Exhaustion is signalled by returning `null` from `next`, so a lazy sequence
#: cannot contain `null`. Bound an infinite iterator (`count`, `repeat`,
#: `cycle`) with `take` or `take_while` before consuming it.
from std.core import Iterable, Iterator

#: The lazy-iterator protocol: a `next` method plus every adapter built on it.
#:
#: Subclass `Iter` and override `next`; every other method here is written
#: against `next` alone, so a subclass inherits the whole surface. An `Iter`
#: is consumed as it is read and cannot be restarted.
#:
#: ```
#: let doubled = iter([1, 2, 3]).map(|x| x * 2)
#: doubled.next()      # 2
#: doubled.collect()   # [4, 6]
#: doubled.next()      # null
#: ```
pub class Iter[T]: Iterable[T], Iterator[T] {
    #: Produce the next element, or null once the sequence is exhausted.
    #:
    #: Every adapter and consumer is written in terms of this alone; once
    #: exhausted, keep returning null rather than restarting.
    pub fn next(self) -> T? {
        throw RuntimeError(f"{type_of(self)} must override next")
    }

    #: Return self, because an `Iter` is already its own iterator.
    pub fn iter(self) -> Iterator[T] {
        return self
    }

    # The VM drives `for x in it` through these two, ending on StopIteration.
    fn __iter__(self) -> Iter[T] {
        return self
    }

    fn __next__(self) -> T {
        let value = self.next()
        if value is null { throw StopIteration("iterator is exhausted") }
        return value
    }

    #: Apply `f` to every element as it is pulled.
    #:
    #: ```
    #: iter([1, 2, 3]).map(|x| x * 10).collect()   # [10, 20, 30]
    #: ```
    pub fn map[U](self, f: fn(T) -> U) -> Map[T, U] {
        return Map(self, f)
    }

    #: Keep only the elements for which `pred` is true.
    #:
    #: ```
    #: iter([1, 2, 3, 4]).filter(|x| x % 2 == 0).collect()   # [2, 4]
    #: ```
    pub fn filter(self, pred: fn(T) -> bool) -> Filter[T] {
        return Filter(self, pred)
    }

    #: Apply `f` and keep the results that are not null.
    #:
    #: One pass instead of a `map` followed by a `filter`, which matters when
    #: `f` is a parse or a lookup that is allowed to fail.
    #:
    #: ```
    #: iter([1, 2, 3, 4]).filter_map(|x| if x % 2 == 0 { x * x } else { null }).collect()
    #: # [4, 16]
    #: ```
    pub fn filter_map[U](self, f: fn(T) -> U?) -> FilterMap[T, U] {
        return FilterMap(self, f)
    }

    #: Yield at most the first `n` elements.
    #:
    #: ```
    #: count(0).take(3).collect()   # [0, 1, 2]
    #: ```
    pub fn take(self, n: int) -> Take[T] {
        return Take(self, n)
    }

    #: Yield elements until `pred` is false, then stop.
    #:
    #: The element that fails `pred` is consumed from the source and discarded.
    #:
    #: ```
    #: iter([1, 2, 9, 3]).take_while(|x| x < 5).collect()   # [1, 2]
    #: ```
    pub fn take_while(self, pred: fn(T) -> bool) -> TakeWhile[T] {
        return TakeWhile(self, pred)
    }

    #: Discard the first `n` elements and yield the rest.
    #:
    #: ```
    #: iter([1, 2, 3, 4]).drop(2).collect()   # [3, 4]
    #: ```
    pub fn drop(self, n: int) -> Drop[T] {
        return Drop(self, n)
    }

    #: Discard the leading elements for which `pred` is true.
    #:
    #: ```
    #: iter([1, 2, 9, 1]).drop_while(|x| x < 5).collect()   # [9, 1]
    #: ```
    pub fn drop_while(self, pred: fn(T) -> bool) -> DropWhile[T] {
        return DropWhile(self, pred)
    }

    #: Yield the first element and then every `step`-th element after it.
    #:
    #: ```
    #: count(0).step_by(3).take(4).collect()   # [0, 3, 6, 9]
    #: ```
    pub fn step_by(self, step: int) -> StepBy[T] {
        return StepBy(self, step)
    }

    #: Yield this iterator's elements, then `other`'s.
    #:
    #: ```
    #: iter([1, 2]).chain([3, 4]).collect()   # [1, 2, 3, 4]
    #: ```
    pub fn chain(self, other: any) -> Chain[T] {
        return Chain([self, iter(other)])
    }

    #: Pair elements with `other`'s, stopping at the shorter of the two.
    #:
    #: ```
    #: iter([1, 2, 3]).zip(["a", "b"]).collect()   # [(1, "a"), (2, "b")]
    #: ```
    pub fn zip(self, other: any) -> Zip[T, any] {
        return Zip(self, iter(other))
    }

    #: Split an iterator of pairs into two independently pullable iterators.
    #:
    #: ```
    #: let sides = iter([(1, "a"), (2, "b")]).unzip()
    #: sides.left().collect()    # [1, 2]
    #: sides.right().collect()   # ["a", "b"]
    #: ```
    pub fn unzip(self) -> Unzip[any, any] {
        return Unzip(self)
    }

    #: Pair every element with its position, counting from `start`.
    #:
    #: ```
    #: iter(["a", "b"]).enumerate().collect()   # [(0, "a"), (1, "b")]
    #: ```
    pub fn enumerate(self, start: int = 0) -> Enumerate[T] {
        return Enumerate(self, start)
    }

    #: Concatenate the iterables produced by this iterator.
    #:
    #: ```
    #: iter([[1, 2], [], [3]]).flatten().collect()   # [1, 2, 3]
    #: ```
    pub fn flatten(self) -> Flatten[any] {
        return Flatten(self)
    }

    #: Map each element to an iterable and concatenate the results.
    #:
    #: ```
    #: iter([1, 2, 3]).flat_map(|x| [x, -x]).collect()   # [1, -1, 2, -2, 3, -3]
    #: ```
    pub fn flat_map[U](self, f: fn(T) -> any) -> FlatMap[T, U] {
        return FlatMap(self, f)
    }

    #: Wrap this iterator in one whose next element can be inspected first.
    #:
    #: ```
    #: let it = iter([1, 2]).peekable()
    #: it.peek()   # 1
    #: it.next()   # 1
    #: ```
    pub fn peekable(self) -> Peekable[T] {
        return Peekable(self)
    }

    #: Yield every overlapping run of `size` consecutive elements.
    #:
    #: Each window is a fresh list; a source shorter than `size` yields nothing.
    #:
    #: ```
    #: iter([1, 2, 3, 4]).windows(3).collect()   # [[1, 2, 3], [2, 3, 4]]
    #: ```
    pub fn windows(self, size: int) -> Windows[T] {
        return Windows(self, size)
    }

    #: Yield consecutive non-overlapping blocks of up to `size` elements.
    #:
    #: ```
    #: iter([1, 2, 3, 4, 5]).chunks(2).collect()   # [[1, 2], [3, 4], [5]]
    #: ```
    pub fn chunks(self, size: int) -> Chunks[T] {
        return Chunks(self, size)
    }

    #: Group *consecutive* elements that share a key.
    #:
    #: Like the Unix `uniq -c` grouping and unlike SQL's `GROUP BY`: sort first
    #: if the equal keys are not already adjacent.
    #:
    #: ```
    #: iter([1, 1, 2, 1]).group_by().collect()   # [(1, [1, 1]), (2, [2]), (1, [1])]
    #: ```
    pub fn group_by(self, key: fn(T) -> any = |x| x) -> GroupBy[T, any] {
        return GroupBy(self, key)
    }

    #: Drop elements equal to the one before them.
    #:
    #: ```
    #: iter([1, 1, 2, 2, 1]).dedup().collect()   # [1, 2, 1]
    #: iter(["a", "A", "b"]).dedup(|s| s.lower()).collect()   # ["a", "b"]
    #: ```
    pub fn dedup(self, key: fn(T) -> any = |x| x) -> Dedup[T] {
        return Dedup(self, key)
    }

    #: Yield the running accumulation of `f` over the elements.
    #:
    #: Unlike `fold`, which returns only the final value, `scan` yields every
    #: intermediate state. The initial state is not yielded.
    #:
    #: ```
    #: iter([1, 2, 3, 4]).scan(0, |total, x| total + x).collect()   # [1, 3, 6, 10]
    #: ```
    pub fn scan[S](self, initial: S, f: fn(S, T) -> S) -> Scan[T, S] {
        return Scan(self, initial, f)
    }

    #: Call `f` on every element as it passes through, unchanged.
    pub fn inspect(self, f: fn(T) -> void) -> Inspect[T] {
        return Inspect(self, f)
    }

    #: Alternate between this iterator and `other`, draining whichever remains.
    #:
    #: ```
    #: iter([1, 3, 5]).interleave([2, 4]).collect()   # [1, 2, 3, 4, 5]
    #: ```
    pub fn interleave(self, other: any) -> Interleave[T] {
        return Interleave(self, iter(other))
    }

    #: Pair every element of this iterator with every element of `other`.
    #:
    #: `other` is collected once, because it is replayed for each element here.
    #:
    #: ```
    #: iter([1, 2]).cross(["a", "b"]).collect()
    #: # [(1, "a"), (1, "b"), (2, "a"), (2, "b")]
    #: ```
    pub fn cross(self, other: any) -> Product[T, any] {
        return Product(self, iter(other))
    }

    #: Yield every ordered arrangement of `k` of the elements.
    #:
    #: `k` defaults to the whole source. The source is collected first, but the
    #: n!/(n-k)! arrangements themselves are generated one at a time, in the
    #: order the source gives.
    #:
    #: ```
    #: iter([1, 2, 3]).permutations(2).collect()
    #: # [[1, 2], [1, 3], [2, 1], [2, 3], [3, 1], [3, 2]]
    #: ```
    pub fn permutations(self, k: int? = null) -> Permutations[T] {
        return Permutations(self, k)
    }

    #: Yield every unordered selection of `k` of the elements.
    #:
    #: The source is collected first; the selections are generated lazily in
    #: index order.
    #:
    #: ```
    #: iter([1, 2, 3, 4]).combinations(3).collect()
    #: # [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
    #: ```
    pub fn combinations(self, k: int) -> Combinations[T] {
        return Combinations(self, k)
    }

    #: Repeat this iterator's elements forever, remembering them as they pass.
    #:
    #: An empty source stays empty rather than looping.
    #:
    #: ```
    #: iter([1, 2]).cycle().take(5).collect()   # [1, 2, 1, 2, 1]
    #: ```
    pub fn cycle(self) -> Iter[T] {
        return _Cycle(self)
    }

    #: Collect the remaining elements into a list.
    #:
    #: ```
    #: count(0).take(3).collect()   # [0, 1, 2]
    #: ```
    pub fn collect(self) -> list[T] {
        let out: list[T] = []
        var value = self.next()
        while value is not null {
            out.push(value)
            value = self.next()
        }
        return out
    }

    #: Collect an iterator of (key, value) pairs into a dict.
    #:
    #: A later pair overwrites an earlier one with the same key.
    #:
    #: ```
    #: iter([("a", 1)]).collect_dict()   # {"a": 1}
    #: ```
    pub fn collect_dict(self) -> dict[any, any] {
        let out: dict[any, any] = {}
        var pair = self.next()
        while pair is not null {
            let (key, value) = pair
            out[key] = value
            pair = self.next()
        }
        return out
    }

    #: Collect the remaining elements into a set, discarding duplicates.
    #:
    #: ```
    #: iter([1, 2, 2, 3]).collect_set().len()   # 3
    #: ```
    pub fn collect_set(self) -> set[T] {
        let out = set()
        var value = self.next()
        while value is not null {
            out.add(value)
            value = self.next()
        }
        return out
    }

    #: Combine the elements left to right, starting from `initial`.
    #:
    #: ```
    #: iter([1, 2, 3]).fold(0, |total, x| total + x)   # 6
    #: iter(["a", "b"]).fold("", |s, c| s + c)         # "ab"
    #: ```
    pub fn fold[A](self, initial: A, f: fn(A, T) -> A) -> A {
        var total = initial
        var value = self.next()
        while value is not null {
            total = f(total, value)
            value = self.next()
        }
        return total
    }

    #: Combine the elements with `f`, using the first as the seed.
    #:
    #: Returns null for an empty iterator, which is the honest answer: there is
    #: no identity element to fall back on.
    #:
    #: ```
    #: iter([3, 1, 2]).reduce(|a, b| if a > b { a } else { b })   # 3
    #: ```
    pub fn reduce(self, f: fn(T, T) -> T) -> T? {
        let first = self.next()
        if first is null { return null }
        var total = first
        var value = self.next()
        while value is not null {
            total = f(total, value)
            value = self.next()
        }
        return total
    }

    #: Call `f` on every element for its side effect.
    pub fn for_each(self, f: fn(T) -> void) -> void {
        var value = self.next()
        while value is not null {
            f(value)
            value = self.next()
        }
    }

    #: Consume the iterator and return how many elements it produced.
    #:
    #: ```
    #: iter([1, 2, 3]).filter(|x| x > 1).count_items()   # 2
    #: ```
    pub fn count_items(self) -> int {
        var total = 0
        var value = self.next()
        while value is not null {
            total += 1
            value = self.next()
        }
        return total
    }

    #: Add the elements together, or return 0 when there are none.
    #:
    #: The first element seeds the sum, so a sum of floats stays a float and no
    #: int/float mixing is introduced.
    #:
    #: ```
    #: iter([1, 2, 3]).sum()   # 6
    #: empty().sum()           # 0
    #: ```
    pub fn sum(self) -> any {
        let first = self.next()
        if first is null { return 0 }
        var total = first
        var value = self.next()
        while value is not null {
            total = total + value
            value = self.next()
        }
        return total
    }

    #: Multiply the elements together, or return 1 when there are none.
    #:
    #: ```
    #: iter([1, 2, 3, 4]).product()   # 24
    #: ```
    pub fn product(self) -> any {
        let first = self.next()
        if first is null { return 1 }
        var total = first
        var value = self.next()
        while value is not null {
            total = total * value
            value = self.next()
        }
        return total
    }

    #: The smallest element by `<`, or null when there are none.
    #:
    #: ```
    #: iter([3, 1, 2]).min()   # 1
    #: ```
    pub fn min(self) -> T? {
        let first = self.next()
        if first is null { return null }
        var best = first
        var value = self.next()
        while value is not null {
            if value < best { best = value }
            value = self.next()
        }
        return best
    }

    #: The largest element by `<`, or null when there are none.
    pub fn max(self) -> T? {
        let first = self.next()
        if first is null { return null }
        var best = first
        var value = self.next()
        while value is not null {
            if best < value { best = value }
            value = self.next()
        }
        return best
    }

    #: The element with the smallest `key`, or null when there are none.
    #:
    #: `key` is evaluated once per element. Ties keep the earliest element.
    #:
    #: ```
    #: iter(["aaa", "b", "cc"]).min_by(|s| s.len())   # "b"
    #: ```
    pub fn min_by(self, key: fn(T) -> any) -> T? {
        let first = self.next()
        if first is null { return null }
        var best = first
        var best_key = key(first)
        var value = self.next()
        while value is not null {
            let candidate = key(value)
            if candidate < best_key {
                best = value
                best_key = candidate
            }
            value = self.next()
        }
        return best
    }

    #: The element with the largest `key`, or null when there are none.
    #:
    #: Ties keep the earliest element.
    pub fn max_by(self, key: fn(T) -> any) -> T? {
        let first = self.next()
        if first is null { return null }
        var best = first
        var best_key = key(first)
        var value = self.next()
        while value is not null {
            let candidate = key(value)
            if best_key < candidate {
                best = value
                best_key = candidate
            }
            value = self.next()
        }
        return best
    }

    #: True when `pred` holds for every element; stops at the first failure.
    #:
    #: True for an empty iterator.
    #:
    #: ```
    #: iter([2, 4]).all(|x| x % 2 == 0)   # true
    #: ```
    pub fn all(self, pred: fn(T) -> bool) -> bool {
        var value = self.next()
        while value is not null {
            if not pred(value) { return false }
            value = self.next()
        }
        return true
    }

    #: True when `pred` holds for some element; stops at the first success.
    #:
    #: False for an empty iterator. Safe on an infinite iterator only when a
    #: match exists.
    #:
    #: ```
    #: count(0).any(|x| x > 10)   # true
    #: ```
    pub fn any(self, pred: fn(T) -> bool) -> bool {
        var value = self.next()
        while value is not null {
            if pred(value) { return true }
            value = self.next()
        }
        return false
    }

    #: The first element satisfying `pred`, or null.
    #:
    #: ```
    #: iter([1, 4, 9]).find(|x| x > 3)   # 4
    #: ```
    pub fn find(self, pred: fn(T) -> bool) -> T? {
        var value = self.next()
        while value is not null {
            if pred(value) { return value }
            value = self.next()
        }
        return null
    }

    #: The first non-null result of `f`, or null.
    #:
    #: ```
    #: iter([1, 2, 3]).find_map(|x| if x > 1 { x * 100 } else { null })   # 200
    #: ```
    pub fn find_map[U](self, f: fn(T) -> U?) -> U? {
        var value = self.next()
        while value is not null {
            let mapped = f(value)
            if mapped is not null { return mapped }
            value = self.next()
        }
        return null
    }

    #: The index of the first element satisfying `pred`, or null.
    #:
    #: The index counts from this iterator's current position, not from the
    #: start of any underlying container.
    #:
    #: ```
    #: iter([1, 4, 9]).position(|x| x > 3)   # 1
    #: ```
    pub fn position(self, pred: fn(T) -> bool) -> int? {
        var index = 0
        var value = self.next()
        while value is not null {
            if pred(value) { return index }
            index += 1
            value = self.next()
        }
        return null
    }

    #: The final element, or null when there are none.
    pub fn last(self) -> T? {
        var found = self.next()
        if found is null { return null }
        var value = self.next()
        while value is not null {
            found = value
            value = self.next()
        }
        return found
    }

    #: The element `n` positions ahead, discarding the ones before it.
    #:
    #: `nth(0)` is the next element. Returns null when the iterator runs out.
    #:
    #: ```
    #: count(0).nth(4)   # 4
    #: ```
    pub fn nth(self, n: int) -> T? {
        if n < 0 { throw ValueError(f"n must be non-negative, got {n}") }
        var remaining = n
        var value = self.next()
        while value is not null {
            if remaining == 0 { return value }
            remaining -= 1
            value = self.next()
        }
        return null
    }

    #: Split the elements into those satisfying `pred` and those not.
    #:
    #: ```
    #: iter([1, 2, 3, 4]).partition(|x| x % 2 == 0)   # ([2, 4], [1, 3])
    #: ```
    pub fn partition(self, pred: fn(T) -> bool) -> tuple[list[T], list[T]] {
        let matching: list[T] = []
        let rest: list[T] = []
        var value = self.next()
        while value is not null {
            if pred(value) { matching.push(value) } else { rest.push(value) }
            value = self.next()
        }
        return (matching, rest)
    }
}

#: An iterator that applies a function to each element of its source.
pub class Map[T, U] extends Iter[U] {
    let source: Iter[T]
    let f: fn(T) -> U

    #: Adapt `source` with `f`.
    pub fn init(self, source: Iter[T], f: fn(T) -> U) {
        self.source = source
        self.f = f
    }

    #: Pull one element and return `f` of it.
    pub fn next(self) -> U? {
        let value = self.source.next()
        if value is null { return null }
        return self.f(value)
    }
}

#: An iterator that yields only the elements satisfying a predicate.
pub class Filter[T] extends Iter[T] {
    let source: Iter[T]
    let pred: fn(T) -> bool

    #: Adapt `source` with `pred`.
    pub fn init(self, source: Iter[T], pred: fn(T) -> bool) {
        self.source = source
        self.pred = pred
    }

    #: Pull until an element satisfies `pred`.
    pub fn next(self) -> T? {
        var value = self.source.next()
        while value is not null {
            if self.pred(value) { return value }
            value = self.source.next()
        }
        return null
    }
}

#: An iterator that maps and drops the null results in one pass.
pub class FilterMap[T, U] extends Iter[U] {
    let source: Iter[T]
    let f: fn(T) -> U?

    #: Adapt `source` with the partial function `f`.
    pub fn init(self, source: Iter[T], f: fn(T) -> U?) {
        self.source = source
        self.f = f
    }

    #: Pull until `f` returns a non-null result.
    pub fn next(self) -> U? {
        var value = self.source.next()
        while value is not null {
            let mapped = self.f(value)
            if mapped is not null { return mapped }
            value = self.source.next()
        }
        return null
    }
}

#: An iterator that stops after a fixed number of elements.
pub class Take[T] extends Iter[T] {
    let source: Iter[T]
    var remaining: int

    #: Yield at most `n` elements of `source`.
    pub fn init(self, source: Iter[T], n: int) {
        if n < 0 { throw ValueError(f"take count must be non-negative, got {n}") }
        self.source = source
        self.remaining = n
    }

    #: Pull one element unless the budget is spent.
    pub fn next(self) -> T? {
        if self.remaining <= 0 { return null }
        let value = self.source.next()
        if value is null {
            self.remaining = 0
            return null
        }
        self.remaining -= 1
        return value
    }
}

#: An iterator that stops at the first element failing a predicate.
pub class TakeWhile[T] extends Iter[T] {
    let source: Iter[T]
    let pred: fn(T) -> bool
    var done: bool = false

    #: Yield elements of `source` while `pred` holds.
    pub fn init(self, source: Iter[T], pred: fn(T) -> bool) {
        self.source = source
        self.pred = pred
    }

    #: Pull one element, latching exhaustion once `pred` fails.
    pub fn next(self) -> T? {
        if self.done { return null }
        let value = self.source.next()
        if value is null or not self.pred(value) {
            self.done = true
            return null
        }
        return value
    }
}

#: An iterator that discards a fixed number of leading elements.
pub class Drop[T] extends Iter[T] {
    let source: Iter[T]
    var remaining: int

    #: Skip the first `n` elements of `source`.
    pub fn init(self, source: Iter[T], n: int) {
        if n < 0 { throw ValueError(f"drop count must be non-negative, got {n}") }
        self.source = source
        self.remaining = n
    }

    #: Pull, skipping the outstanding prefix on the first call.
    pub fn next(self) -> T? {
        while self.remaining > 0 {
            let skipped = self.source.next()
            if skipped is null {
                self.remaining = 0
                return null
            }
            self.remaining -= 1
        }
        return self.source.next()
    }
}

#: An iterator that discards leading elements satisfying a predicate.
pub class DropWhile[T] extends Iter[T] {
    let source: Iter[T]
    let pred: fn(T) -> bool
    var dropping: bool = true

    #: Skip the leading elements of `source` for which `pred` holds.
    pub fn init(self, source: Iter[T], pred: fn(T) -> bool) {
        self.source = source
        self.pred = pred
    }

    #: Pull, consuming the matching prefix on the first call.
    pub fn next(self) -> T? {
        if not self.dropping { return self.source.next() }
        var value = self.source.next()
        while value is not null {
            if not self.pred(value) {
                self.dropping = false
                return value
            }
            value = self.source.next()
        }
        self.dropping = false
        return null
    }
}

#: An iterator that yields every `step`-th element of its source.
pub class StepBy[T] extends Iter[T] {
    let source: Iter[T]
    let step: int
    var started: bool = false

    #: Yield the first element of `source` and then every `step`-th one.
    pub fn init(self, source: Iter[T], step: int) {
        if step < 1 { throw ValueError(f"step must be at least 1, got {step}") }
        self.source = source
        self.step = step
    }

    #: Pull one element, discarding the `step - 1` before it.
    pub fn next(self) -> T? {
        if not self.started {
            self.started = true
            return self.source.next()
        }
        var skipped = 1
        while skipped < self.step {
            if self.source.next() is null { return null }
            skipped += 1
        }
        return self.source.next()
    }
}

#: An iterator that yields several sources end to end.
pub class Chain[T] extends Iter[T] {
    let sources: list[Iter[T]]
    var index: int = 0

    #: Concatenate `sources` in order.
    pub fn init(self, sources: list[Iter[T]]) {
        self.sources = sources
    }

    #: Pull from the current source, advancing when it runs out.
    pub fn next(self) -> T? {
        while self.index < self.sources.len() {
            let value = self.sources[self.index].next()
            if value is not null { return value }
            self.index += 1
        }
        return null
    }
}

#: An iterator that pairs two sources element by element.
pub class Zip[A, B] extends Iter[tuple[A, B]] {
    let first: Iter[A]
    let second: Iter[B]

    #: Pair `first` with `second`, ending with the shorter one.
    pub fn init(self, first: Iter[A], second: Iter[B]) {
        self.first = first
        self.second = second
    }

    #: Pull one element from each source and pair them.
    pub fn next(self) -> tuple[A, B]? {
        let a = self.first.next()
        if a is null { return null }
        let b = self.second.next()
        if b is null { return null }
        return (a, b)
    }
}

#: Two iterators over the halves of one iterator of pairs.
#:
#: The sides are pulled independently, so whichever runs ahead buffers the
#: components the other has not taken yet. Consuming one side to the end
#: therefore buffers the whole of the other; consuming them in step buffers
#: nothing.
pub class Unzip[A, B] {
    let source: Iter[tuple[A, B]]
    var lefts: list[A] = []
    var rights: list[B] = []
    var left_head: int = 0
    var right_head: int = 0

    #: Split `source`, an iterator of pairs.
    pub fn init(self, source: Iter[tuple[A, B]]) {
        self.source = source
    }

    #: The iterator over the first component of each pair.
    pub fn left(self) -> Iter[A] {
        return _FromFn(fn() -> any { return self.next_left() })
    }

    #: The iterator over the second component of each pair.
    pub fn right(self) -> Iter[B] {
        return _FromFn(fn() -> any { return self.next_right() })
    }

    #: Pull one first component, buffering the second for the other side.
    #:
    #: `left()` is the usual way to reach this. It is public because the
    #: closure `left()` builds is not inside the class as far as the visibility
    #: check is concerned, so a private method is unreachable from it.
    pub fn next_left(self) -> A? {
        if self.left_head < self.lefts.len() {
            let value = self.lefts[self.left_head]
            self.left_head += 1
            if self.left_head == self.lefts.len() {
                self.lefts = []
                self.left_head = 0
            }
            return value
        }
        let pair = self.source.next()
        if pair is null { return null }
        let (a, b) = pair
        self.rights.push(b)
        return a
    }

    #: Pull one second component, buffering the first for the other side.
    pub fn next_right(self) -> B? {
        if self.right_head < self.rights.len() {
            let value = self.rights[self.right_head]
            self.right_head += 1
            if self.right_head == self.rights.len() {
                self.rights = []
                self.right_head = 0
            }
            return value
        }
        let pair = self.source.next()
        if pair is null { return null }
        let (a, b) = pair
        self.lefts.push(a)
        return b
    }
}

#: An iterator that pairs each element with its position.
pub class Enumerate[T] extends Iter[tuple[int, T]] {
    let source: Iter[T]
    var index: int

    #: Number the elements of `source`, counting from `start`.
    pub fn init(self, source: Iter[T], start: int = 0) {
        self.source = source
        self.index = start
    }

    #: Pull one element and tag it with the running index.
    pub fn next(self) -> tuple[int, T]? {
        let value = self.source.next()
        if value is null { return null }
        let pair = (self.index, value)
        self.index += 1
        return pair
    }
}

#: An iterator that concatenates the iterables produced by its source.
pub class Flatten[T] extends Iter[T] {
    let source: Iter[any]
    var current: any = null

    #: Flatten one level of `source`.
    pub fn init(self, source: Iter[any]) {
        self.source = source
    }

    #: Pull from the inner iterator, opening the next one when it runs out.
    pub fn next(self) -> T? {
        loop {
            if self.current is not null {
                let value = self.current.next()
                if value is not null { return value }
                self.current = null
            }
            let inner = self.source.next()
            if inner is null { return null }
            self.current = iter(inner)
        }
    }
}

#: An iterator that maps each element to an iterable and concatenates them.
pub class FlatMap[T, U] extends Iter[U] {
    let inner: Flatten[U]

    #: Map `source` through `f` and flatten the results.
    pub fn init(self, source: Iter[T], f: fn(T) -> any) {
        self.inner = Flatten(Map(source, f))
    }

    #: Pull one element from the flattened stream.
    pub fn next(self) -> U? {
        return self.inner.next()
    }
}

#: An iterator whose next element can be examined without consuming it.
pub class Peekable[T] extends Iter[T] {
    let source: Iter[T]
    var buffered: any = null

    #: Give `source` a one-element lookahead buffer.
    pub fn init(self, source: Iter[T]) {
        self.source = source
    }

    #: The next element without consuming it, or null at the end.
    pub fn peek(self) -> T? {
        if self.buffered is null { self.buffered = self.source.next() }
        return self.buffered
    }

    #: Take the buffered element if there is one, otherwise pull a fresh one.
    pub fn next(self) -> T? {
        if self.buffered is not null {
            let value = self.buffered
            self.buffered = null
            return value
        }
        return self.source.next()
    }
}

#: An iterator over the overlapping windows of a fixed width.
pub class Windows[T] extends Iter[list[T]] {
    let source: Iter[T]
    let size: int
    let buffer: list[T] = []
    var start: int = 0
    var primed: bool = false
    var done: bool = false

    #: Slide a window of `size` elements over `source`.
    pub fn init(self, source: Iter[T], size: int) {
        if size < 1 { throw ValueError(f"window size must be at least 1, got {size}") }
        self.source = source
        self.size = size
    }

    #: Advance the window by one element and copy it out.
    pub fn next(self) -> list[T]? {
        if self.done { return null }
        if not self.primed {
            while self.buffer.len() < self.size {
                let value = self.source.next()
                if value is null {
                    self.done = true
                    return null
                }
                self.buffer.push(value)
            }
            self.primed = true
            return self._snapshot()
        }
        let value = self.source.next()
        if value is null {
            self.done = true
            return null
        }
        # The buffer is a ring: overwrite the oldest slot instead of shifting.
        self.buffer[self.start] = value
        self.start = (self.start + 1) % self.size
        return self._snapshot()
    }

    fn _snapshot(self) -> list[T] {
        let out: list[T] = []
        for offset in 0..self.size {
            out.push(self.buffer[(self.start + offset) % self.size])
        }
        return out
    }
}

#: An iterator over consecutive non-overlapping blocks.
pub class Chunks[T] extends Iter[list[T]] {
    let source: Iter[T]
    let size: int

    #: Cut `source` into blocks of `size` elements.
    pub fn init(self, source: Iter[T], size: int) {
        if size < 1 { throw ValueError(f"chunk size must be at least 1, got {size}") }
        self.source = source
        self.size = size
    }

    #: Pull up to `size` elements; the final block may be shorter.
    pub fn next(self) -> list[T]? {
        let chunk: list[T] = []
        while chunk.len() < self.size {
            let value = self.source.next()
            if value is null { break }
            chunk.push(value)
        }
        if chunk.len() == 0 { return null }
        return chunk
    }
}

#: An iterator over the runs of consecutive elements sharing a key.
pub class GroupBy[T, K] extends Iter[tuple[K, list[T]]] {
    let source: Iter[T]
    let key: fn(T) -> K
    var pending: any = null
    var done: bool = false

    #: Group the consecutive elements of `source` by `key`.
    pub fn init(self, source: Iter[T], key: fn(T) -> K) {
        self.source = source
        self.key = key
    }

    #: Read one whole run, holding back the element that ended it.
    pub fn next(self) -> tuple[K, list[T]]? {
        if self.done { return null }
        var first = self.pending
        self.pending = null
        if first is null { first = self.source.next() }
        if first is null {
            self.done = true
            return null
        }
        let group_key = self.key(first)
        let group = [first]
        var value = self.source.next()
        while value is not null {
            if self.key(value) != group_key {
                self.pending = value
                return (group_key, group)
            }
            group.push(value)
            value = self.source.next()
        }
        self.done = true
        return (group_key, group)
    }
}

#: An iterator that drops elements equal to their predecessor.
pub class Dedup[T] extends Iter[T] {
    let source: Iter[T]
    let key: fn(T) -> any
    var last_key: any = null
    var has_last: bool = false

    #: Collapse the runs of `source` whose elements share a `key`.
    pub fn init(self, source: Iter[T], key: fn(T) -> any) {
        self.source = source
        self.key = key
    }

    #: Pull until an element differs from the one last yielded.
    pub fn next(self) -> T? {
        var value = self.source.next()
        while value is not null {
            let candidate = self.key(value)
            if not self.has_last or candidate != self.last_key {
                self.last_key = candidate
                self.has_last = true
                return value
            }
            value = self.source.next()
        }
        return null
    }
}

#: An iterator over the running states of a fold.
pub class Scan[T, S] extends Iter[S] {
    let source: Iter[T]
    let f: fn(S, T) -> S
    var state: any

    #: Accumulate `source` into `initial` with `f`, yielding each new state.
    pub fn init(self, source: Iter[T], initial: S, f: fn(S, T) -> S) {
        self.source = source
        self.state = initial
        self.f = f
    }

    #: Fold one more element in and yield the new state.
    pub fn next(self) -> S? {
        let value = self.source.next()
        if value is null { return null }
        self.state = self.f(self.state, value)
        return self.state
    }
}

#: An iterator that calls a function on each element it passes through.
pub class Inspect[T] extends Iter[T] {
    let source: Iter[T]
    let f: fn(T) -> void

    #: Run `f` on every element of `source` without changing the stream.
    pub fn init(self, source: Iter[T], f: fn(T) -> void) {
        self.source = source
        self.f = f
    }

    #: Pull one element, showing it to `f` on the way past.
    pub fn next(self) -> T? {
        let value = self.source.next()
        if value is not null { self.f(value) }
        return value
    }
}

#: An iterator that alternates between two sources.
pub class Interleave[T] extends Iter[T] {
    let first: Iter[T]
    let second: Iter[T]
    var use_first: bool = true
    var first_done: bool = false
    var second_done: bool = false

    #: Alternate `first` and `second`, starting with `first`.
    pub fn init(self, first: Iter[T], second: Iter[T]) {
        self.first = first
        self.second = second
    }

    #: Pull from whichever source has the turn, skipping an exhausted one.
    pub fn next(self) -> T? {
        var attempts = 0
        while attempts < 2 {
            attempts += 1
            let take_first = self.use_first
            self.use_first = not self.use_first
            let exhausted = take_first ? self.first_done : self.second_done
            if not exhausted {
                let source = take_first ? self.first : self.second
                let value = source.next()
                if value is not null { return value }
                if take_first { self.first_done = true } else { self.second_done = true }
            }
        }
        return null
    }
}

#: An iterator over the cartesian product of two sources.
#:
#: The second source is collected once, because it is replayed for every
#: element of the first.
pub class Product[A, B] extends Iter[tuple[A, B]] {
    let first: Iter[A]
    let second: list[B]
    var current: any = null
    var has_current: bool = false
    var index: int = 0

    #: Pair every element of `first` with every element of `second`.
    pub fn init(self, first: Iter[A], second: Iter[B]) {
        self.first = first
        self.second = second.collect()
    }

    #: Emit the next pair, advancing the outer element when the inner wraps.
    pub fn next(self) -> tuple[A, B]? {
        if self.second.len() == 0 { return null }
        if not self.has_current {
            let value = self.first.next()
            if value is null { return null }
            self.current = value
            self.has_current = true
            self.index = 0
        }
        let pair = (self.current, self.second[self.index])
        self.index += 1
        if self.index >= self.second.len() { self.has_current = false }
        return pair
    }
}

#: An iterator over the ordered arrangements of `k` of the source's elements.
#:
#: The source is collected first; the arrangements are produced one at a time
#: by rotating an index vector, so the memory cost is the pool plus `k`, not
#: the n!/(n-k)! results.
pub class Permutations[T] extends Iter[list[T]] {
    let pool: list[T]
    let n: int
    let k: int
    let indices: list[int] = []
    let cycles: list[int] = []
    var first: bool = true
    var done: bool = false

    #: Arrange `k` of `source`'s elements; `k` defaults to all of them.
    pub fn init(self, source: Iter[T], k: int? = null) {
        if k is not null and k < 0 { throw ValueError(f"k must be non-negative, got {k}") }
        self.pool = source.collect()
        self.n = self.pool.len()
        self.k = k ?? self.n
        self.done = self.k > self.n
        for i in 0..self.n { self.indices.push(i) }
        for i in 0..self.k { self.cycles.push(self.n - i) }
    }

    #: Advance the index vector to the next arrangement.
    pub fn next(self) -> list[T]? {
        if self.done { return null }
        if self.first {
            self.first = false
            return self._emit()
        }
        var i = self.k - 1
        while i >= 0 {
            self.cycles[i] -= 1
            if self.cycles[i] == 0 {
                let moved = self.indices[i]
                var j = i
                while j < self.n - 1 {
                    self.indices[j] = self.indices[j + 1]
                    j += 1
                }
                self.indices[self.n - 1] = moved
                self.cycles[i] = self.n - i
                i -= 1
            } else {
                let back = self.cycles[i]
                let swapped = self.indices[i]
                self.indices[i] = self.indices[self.n - back]
                self.indices[self.n - back] = swapped
                return self._emit()
            }
        }
        self.done = true
        return null
    }

    fn _emit(self) -> list[T] {
        let out: list[T] = []
        for i in 0..self.k { out.push(self.pool[self.indices[i]]) }
        return out
    }
}

#: An iterator over the unordered selections of `k` of the source's elements.
#:
#: The source is collected first; the selections are produced one at a time by
#: incrementing an index vector.
pub class Combinations[T] extends Iter[list[T]] {
    let pool: list[T]
    let n: int
    let k: int
    let indices: list[int] = []
    var first: bool = true
    var done: bool = false

    #: Select `k` of `source`'s elements at a time, in index order.
    pub fn init(self, source: Iter[T], k: int) {
        if k < 0 { throw ValueError(f"k must be non-negative, got {k}") }
        self.pool = source.collect()
        self.n = self.pool.len()
        self.k = k
        self.done = k > self.n
        for i in 0..k { self.indices.push(i) }
    }

    #: Advance the index vector to the next selection.
    pub fn next(self) -> list[T]? {
        if self.done { return null }
        if self.first {
            self.first = false
            return self._emit()
        }
        var i = self.k - 1
        while i >= 0 and self.indices[i] == i + self.n - self.k { i -= 1 }
        if i < 0 {
            self.done = true
            return null
        }
        self.indices[i] += 1
        var j = i + 1
        while j < self.k {
            self.indices[j] = self.indices[j - 1] + 1
            j += 1
        }
        return self._emit()
    }

    fn _emit(self) -> list[T] {
        let out: list[T] = []
        for i in 0..self.k { out.push(self.pool[self.indices[i]]) }
        return out
    }
}

class _SeqIter extends Iter[any] {
    let source: any
    let limit: int
    var index: int = 0

    #: Walk `source` — a list, tuple or string — by index.
    pub fn init(self, source: any) {
        self.source = source
        self.limit = source.len()
    }

    pub fn next(self) -> any {
        if self.index >= self.limit { return null }
        let value = self.source[self.index]
        self.index += 1
        return value
    }
}

class _Pull extends Iter[any] {
    let source: Iterator[any]

    #: Pull elements through the `next` of any `std.core` iterator.
    pub fn init(self, source: Iterator[any]) {
        self.source = source
    }

    pub fn next(self) -> any {
        return self.source.next()
    }
}

class _Count extends Iter[int] {
    var value: int
    let step: int

    #: Count from `start` in steps of `step`.
    pub fn init(self, start: int, step: int) {
        self.value = start
        self.step = step
    }

    pub fn next(self) -> int? {
        let current = self.value
        self.value += self.step
        return current
    }
}

class _Repeat extends Iter[any] {
    let value: any
    var remaining: int

    #: Yield `value` `remaining` times; a negative count means forever.
    pub fn init(self, value: any, remaining: int) {
        self.value = value
        self.remaining = remaining
    }

    pub fn next(self) -> any {
        if self.remaining == 0 { return null }
        if self.remaining > 0 { self.remaining -= 1 }
        return self.value
    }
}

class _Once extends Iter[any] {
    var value: any

    #: Yield `value` and then nothing.
    pub fn init(self, value: any) {
        self.value = value
    }

    pub fn next(self) -> any {
        let taken = self.value
        self.value = null
        return taken
    }
}

class _Empty extends Iter[any] {
    pub fn next(self) -> any {
        return null
    }
}

class _FromFn extends Iter[any] {
    let f: fn() -> any

    #: Call `f` once for every element.
    pub fn init(self, f: fn() -> any) {
        self.f = f
    }

    pub fn next(self) -> any {
        return self.f()
    }
}

class _Successors extends Iter[any] {
    let f: fn(any) -> any
    var current: any

    #: Yield `seed` and then the iterates of `f`.
    pub fn init(self, seed: any, f: fn(any) -> any) {
        self.current = seed
        self.f = f
    }

    pub fn next(self) -> any {
        let value = self.current
        if value is null { return null }
        self.current = self.f(value)
        return value
    }
}

class _RangeStep extends Iter[int] {
    var value: int
    let stop: int
    let step: int

    #: Step from `start` towards `stop` by `step`.
    pub fn init(self, start: int, stop: int, step: int) {
        self.value = start
        self.stop = stop
        self.step = step
    }

    pub fn next(self) -> int? {
        if self.step > 0 and self.value >= self.stop { return null }
        if self.step < 0 and self.value <= self.stop { return null }
        let current = self.value
        self.value += self.step
        return current
    }
}

class _Cycle extends Iter[any] {
    let source: Iter[any]
    let buffer: list[any] = []
    var filling: bool = true
    var index: int = 0

    #: Replay `source` forever, buffering it on the first pass.
    pub fn init(self, source: Iter[any]) {
        self.source = source
    }

    pub fn next(self) -> any {
        if self.filling {
            let value = self.source.next()
            if value is not null {
                self.buffer.push(value)
                return value
            }
            self.filling = false
        }
        if self.buffer.len() == 0 { return null }
        let value = self.buffer[self.index]
        self.index = (self.index + 1) % self.buffer.len()
        return value
    }
}

#: Wrap any iterable in a lazy `Iter`.
#:
#: A dict, set, or range is materialised into a list first, because the VM's
#: built-in iterator objects cannot be stepped from Jaithon; use `range_step`
#: for a lazy arithmetic sequence instead.
#:
#: ```
#: iter([1, 2, 3]).collect()       # [1, 2, 3]
#: iter("abc").take(2).collect()   # ["a", "b"]
#: ```
pub fn iter(source: any) -> Iter[any] {
    if isinstance(source, Iter) { return source }
    if isinstance(source, list) or isinstance(source, tuple) or isinstance(source, str) {
        return _SeqIter(source)
    }
    if isinstance(source, Iterator) { return _Pull(source) }
    if isinstance(source, Iterable) { return _Pull(source.iter()) }
    return _SeqIter([value for value in source])
}

#: Count from `start` forever in steps of `step`.
#:
#: ```
#: count(1).take(4).collect()        # [1, 2, 3, 4]
#: count(10, -2).take(3).collect()   # [10, 8, 6]
#: ```
pub fn count(start: int = 0, step: int = 1) -> Iter[int] {
    if step == 0 { throw ValueError("step must be non-zero, got 0") }
    return _Count(start, step)
}

#: Yield `value` `times` times, or forever when `times` is null.
#:
#: ```
#: repeat("x", 3).collect()        # ["x", "x", "x"]
#: repeat(0).take(2).collect()     # [0, 0]
#: ```
pub fn repeat(value: any, times: int? = null) -> Iter[any] {
    if times is not null and times < 0 {
        throw ValueError(f"times must be non-negative, got {times}")
    }
    return _Repeat(value, times ?? -1)
}

#: Repeat an iterable's elements forever, remembering them as they pass.
#:
#: The source is pulled once and replayed from the buffer thereafter, so it need
#: not be re-iterable. An empty source stays empty.
#:
#: ```
#: cycle([1, 2]).take(5).collect()   # [1, 2, 1, 2, 1]
#: ```
pub fn cycle(source: any) -> Iter[any] {
    return _Cycle(iter(source))
}

#: Yield `value` exactly once.
#:
#: ```
#: once(7).collect()   # [7]
#: ```
pub fn once(value: any) -> Iter[any] {
    return _Once(value)
}

#: Yield nothing.
#:
#: Useful as the identity for `chain` and as a base case in recursion.
#:
#: ```
#: empty().collect()   # []
#: ```
pub fn empty() -> Iter[any] {
    return _Empty()
}

#: Build an iterator that calls `f` for each element, ending when it returns null.
#:
#: ```
#: var n = 0
#: let counted = from_fn(fn() -> any {
#:     n += 1
#:     return if n <= 3 { n } else { null }
#: })
#: counted.collect()   # [1, 2, 3]
#: ```
pub fn from_fn(f: fn() -> any) -> Iter[any] {
    return _FromFn(f)
}

#: Yield `seed`, then `f` of it, and so on until `f` returns null.
#:
#: ```
#: successors(1, |x| if x < 100 { x * 2 } else { null }).collect()
#: # [1, 2, 4, 8, 16, 32, 64, 128]
#: ```
pub fn successors(seed: any, f: fn(any) -> any) -> Iter[any] {
    return _Successors(seed, f)
}

#: Count from `start` towards `stop` in steps of `step`, excluding `stop`.
#:
#: The lazy equivalent of a `..` range with a stride, and unlike `iter` over a
#: range it never materialises its elements.
#:
#: ```
#: range_step(0, 10, 3).collect()   # [0, 3, 6, 9]
#: range_step(5, 0, -2).collect()   # [5, 3, 1]
#: ```
pub fn range_step(start: int, stop: int, step: int) -> Iter[int] {
    if step == 0 { throw ValueError("step must be non-zero, got 0") }
    return _RangeStep(start, stop, step)
}
