#: Define the traits, the comparison protocol, and the small value types that
#: the rest of the standard library is built on.
#:
#: Imports nothing, so every other module can depend on it without risking an
#: import cycle. `std.prelude` re-exports the traits, `Ordering` and `cmp`.

#: Produce fresh iterators over a sequence of values.
#:
#: `for x in value` requires the iterated type to implement this trait.
pub trait Iterable[T] {
    #: Return a fresh iterator over the elements of `self`.
    #:
    #: Two calls must not interfere with each other: return a new iterator
    #: every time.
    fn iter(self) -> Iterator[T]
}

#: Walk a sequence of values one element at a time.
#:
#: Implement `next` and the default methods below come with it. Every adapter
#: among them — `map`, `filter`, `take` and the rest — is lazy: no work
#: happens until something calls `next` on the end of the chain.
pub trait Iterator[T]: Iterable[T] {
    #: Return `self`: an iterator is its own iterator.
    #:
    #: Lets `for x in xs.iter().map(f)` type-check — an adapter answers
    #: `Iterator[T]`, and without this a `for` over one would be a type error
    #: even though the VM (`jaiGetIter`) already accepts it. A class iterated
    #: more than once must override this, since a fresh iterator is what
    #: `Iterable` promises.
    fn iter(self) -> Iterator[T] { return self }

    #: Advance the iterator and return the next element, or null at the end.
    #:
    #: Once `next` has returned null it must keep returning null. Since null
    #: ends iteration, an iterator whose elements can themselves be null must
    #: wrap them — in a `Pair`, a tuple, or a one-element list.
    fn next(self) -> T?

    #: Apply `transform` to every element, lazily.
    #:
    #: A `transform` that returns null ends the iteration early — see `next`.
    #:
    #: ```
    #: from std.iter import iter
    #: iter([1, 2, 3]).map(|x| x * 2).collect()    # [2, 4, 6]
    #: ```
    fn map(self, transform: fn(T) -> any) -> Iterator[any] {
        return MapIter[T, any](self, transform)
    }

    #: Yield only the elements for which `predicate` returns true.
    fn filter(self, predicate: fn(T) -> bool) -> Iterator[T] {
        return FilterIter[T](self, predicate)
    }

    #: Yield at most the first `count` elements, then stop.
    fn take(self, count: int) -> Iterator[T] {
        if count < 0 { throw ValueError(f"take count must be non-negative, got {count}") }
        return TakeIter[T](self, count)
    }

    #: Skip the first `count` elements and yield every element after them.
    fn drop(self, count: int) -> Iterator[T] {
        if count < 0 { throw ValueError(f"drop count must be non-negative, got {count}") }
        return DropIter[T](self, count)
    }

    #: Pair every element with its zero-based position.
    #:
    #: ```
    #: from std.iter import iter
    #: iter(["a", "b"]).enumerate().collect()    # [(0, "a"), (1, "b")]
    #: ```
    fn enumerate(self) -> Iterator[tuple[int, T]] { return EnumerateIter[T](self) }

    #: Yield pairs drawn from `self` and `other`, stopping at the shorter one.
    #:
    #: ```
    #: from std.iter import iter
    #: iter([1, 2, 3]).zip(iter(["a", "b"])).collect()    # [(1, "a"), (2, "b")]
    #: ```
    fn zip(self, other: Iterator[any]) -> Iterator[tuple[T, any]] {
        return ZipIter[T, any](self, other)
    }

    #: Yield every element of `self`, then every element of `other`.
    fn chain(self, other: Iterator[T]) -> Iterator[T] { return ChainIter[T](self, other) }

    #: Yield the first element and then every `step`-th element after it.
    #:
    #: ```
    #: from std.iter import iter
    #: iter(0..10).step_by(3).collect()    # [0, 3, 6, 9]
    #: ```
    fn step_by(self, step: int) -> Iterator[T] {
        if step <= 0 { throw ValueError(f"step must be positive, got {step}") }
        return StepByIter[T](self, step)
    }

    #: Concatenate the elements, each of which must itself be `Iterable`.
    #:
    #: Flattening is one level deep; call it twice for a list of lists of lists.
    #:
    #: ```
    #: from std.iter import iter
    #: iter([[1, 2], [3]]).flatten().collect()    # [1, 2, 3]
    #: ```
    fn flatten(self) -> Iterator[any] { return FlattenIter(self) }

    #: Run `action` on every element as it passes through, yielding it unchanged.
    #:
    #: The debugging tap for a long chain. It does not force the iterator, so
    #: `action` runs only for the elements actually consumed.
    fn inspect(self, action: fn(T) -> void) -> Iterator[T] {
        let tap = fn(value: T) -> T {
            action(value)
            return value
        }
        return MapIter[T, T](self, tap)
    }

    #: Wrap `self` in an iterator that can look one element ahead.
    #:
    #: ```
    #: from std.iter import iter
    #: let numbers = Peekable(iter([1, 2]))
    #: numbers.peek()    # 1
    #: numbers.next()    # 1
    #: ```
    fn peekable(self) -> Peekable[T] { return Peekable[T](self) }

    #: Consume the iterator and return its elements as a list.
    fn collect(self) -> list[T] {
        var items: list[T] = []
        var value = self.next()
        while value is not null {
            items.push(value)
            value = self.next()
        }
        return items
    }

    #: Combine the elements left to right, starting from `initial`.
    #:
    #: ```
    #: from std.iter import iter
    #: iter([1, 2, 3]).fold(0, |total, x| total + x)    # 6
    #: ```
    fn fold(self, initial: any, combine: fn(any, T) -> any) -> any {
        var accumulator = initial
        var value = self.next()
        while value is not null {
            accumulator = combine(accumulator, value)
            value = self.next()
        }
        return accumulator
    }

    #: Consume the iterator and return how many elements it produced.
    fn count(self) -> int {
        var total = 0
        while self.next() is not null { total += 1 }
        return total
    }

    #: Return the first element satisfying `predicate`, or null.
    #:
    #: The iterator is left positioned just after the element returned, so the
    #: remaining elements can still be consumed.
    fn find(self, predicate: fn(T) -> bool) -> T? {
        var value = self.next()
        while value is not null {
            if predicate(value) { return value }
            value = self.next()
        }
        return null
    }

    #: Return the zero-based index of the first element satisfying `predicate`.
    #:
    #: Returns null when no element satisfies it.
    fn position(self, predicate: fn(T) -> bool) -> int? {
        var index = 0
        var value = self.next()
        while value is not null {
            if predicate(value) { return index }
            index += 1
            value = self.next()
        }
        return null
    }

    #: Report whether any element satisfies `predicate`, stopping at the first.
    fn any(self, predicate: fn(T) -> bool) -> bool { return self.find(predicate) is not null }

    #: Report whether every element satisfies `predicate`.
    #:
    #: True for an empty iterator, and stops at the first counter-example.
    fn all(self, predicate: fn(T) -> bool) -> bool {
        var value = self.next()
        while value is not null {
            if not predicate(value) { return false }
            value = self.next()
        }
        return true
    }

    #: Return the smallest element under the three-way `compare`, or null.
    #:
    #: Ties keep the earliest element.
    #:
    #: `std.iter`'s `min_by` takes a key instead of a comparison; this is the
    #: trait's own default, reached via a `Peekable` that doesn't override it.
    #:
    #: ```
    #: from std.iter import iter
    #: Peekable(iter(["aaa", "b"])).min_by(|a, b| cmp(a.len(), b.len()))    # "b"
    #: ```
    fn min_by(self, compare: fn(T, T) -> int) -> T? {
        var best = self.next()
        if best is null { return null }
        var value = self.next()
        while value is not null {
            if compare(value, best) < 0 { best = value }
            value = self.next()
        }
        return best
    }

    #: Return the largest element under the three-way `compare`, or null.
    #:
    #: Ties keep the earliest element.
    fn max_by(self, compare: fn(T, T) -> int) -> T? {
        var best = self.next()
        if best is null { return null }
        var value = self.next()
        while value is not null {
            if compare(value, best) > 0 { best = value }
            value = self.next()
        }
        return best
    }

    #: Consume the iterator and return its final element, or null when empty.
    fn last(self) -> T? {
        var result: T? = null
        var value = self.next()
        while value is not null {
            result = value
            value = self.next()
        }
        return result
    }

    #: Return the element at zero-based `index`, consuming everything before it.
    fn nth(self, index: int) -> T? {
        if index < 0 { throw ValueError(f"nth index must be non-negative, got {index}") }
        var remaining = index
        var value = self.next()
        while value is not null {
            if remaining == 0 { return value }
            remaining -= 1
            value = self.next()
        }
        return null
    }
}

#: Impose a total order on a type.
#:
#: Implement `compare`; the six comparisons derived from it come with it.
#:
#: The order is over one type, so every method takes and returns `Self`
#: (spec §7.2): an implementer is ordered against its own kind, and `max`
#: hands back one of its two arguments with that type intact.
pub trait Comparable {
    #: Return a negative int when `self` sorts before `other`, zero when the two
    #: are equivalent, and a positive int when `self` sorts after it.
    #:
    #: The relation must be a total order and must agree with `__eq__`.
    fn compare(self, other: Self) -> int

    #: Report whether `self` sorts strictly before `other`.
    fn lt(self, other: Self) -> bool { return self.compare(other) < 0 }

    #: Report whether `self` sorts before `other` or is equivalent to it.
    fn le(self, other: Self) -> bool { return self.compare(other) <= 0 }

    #: Report whether `self` sorts strictly after `other`.
    fn gt(self, other: Self) -> bool { return self.compare(other) > 0 }

    #: Report whether `self` sorts after `other` or is equivalent to it.
    fn ge(self, other: Self) -> bool { return self.compare(other) >= 0 }

    #: Return whichever of `self` and `other` sorts later, `self` on a tie.
    #:
    #: ```
    #: Pair(1, 2).max(Pair(1, 3)).second    # 3
    #: ```
    fn max(self, other: Self) -> Self {
        if self.compare(other) >= 0 { return self }
        return other
    }

    #: Return whichever of `self` and `other` sorts earlier, `self` on a tie.
    fn min(self, other: Self) -> Self {
        if self.compare(other) <= 0 { return self }
        return other
    }
}

#: Supply a hash code, so a type can be a dict key or a set member.
pub trait Hashable {
    #: Return a hash code for `self`.
    #:
    #: Values that compare equal must return the same hash, and the hash of a
    #: value must not change while it is a key in a dict or a member of a set.
    fn hash(self) -> int
}

#: Render a value for a human reader.
pub trait Printable {
    #: Return the human-readable rendering of `self`.
    #:
    #: A type that wants `print` and f-strings to use this rendering also
    #: defines `__str__` as a one-line delegate to it.
    fn to_str(self) -> str
}

#: Duplicate a value.
pub trait Cloneable {
    #: Return a copy of `self` that can be mutated without affecting `self`.
    #:
    #: The copy is shallow unless the implementation says otherwise. `Self`
    #: (spec §7.2) makes the promise exact: a clone of a `Matrix` is a
    #: `Matrix`, with no annotation needed.
    fn clone(self) -> Self
}

#: Supply the value a type falls back to when none is given.
#:
#: Generic containers use it to fill a slot they must create but cannot invent
#: a value for, such as the cells of a freshly sized matrix.
pub trait Default[T] {
    #: Return the value the type uses when no other value is supplied.
    static fn default() -> T
}

#: Hold a countable group of values that can be tested for membership.
pub trait Container[T] {
    #: Return the number of elements held.
    fn len(self) -> int

    #: Report whether `value` is held.
    fn contains(self, value: T) -> bool

    #: Report whether the container holds no elements.
    fn is_empty(self) -> bool { return self.len() == 0 }
}

#: The result of a three-way comparison.
pub enum Ordering {
    #: The left operand sorts before the right one, the case `cmp` and
    #: `Comparable.compare` report as a negative int.
    Less,
    #: Neither operand sorts before the other — equivalence under the order,
    #: not identity; `Comparable` requires whatever compares `Equal` to also
    #: be `==`.
    Equal,
    #: The left operand sorts after the right one, the case `cmp` and
    #: `Comparable.compare` report as a positive int.
    Greater,

    #: Return -1 for `Less`, 0 for `Equal` and 1 for `Greater`.
    pub fn to_int(self) -> int {
        return match self {
            Ordering.Less => -1,
            Ordering.Equal => 0,
            Ordering.Greater => 1,
        }
    }

    #: Return the ordering described by the sign of `value`.
    #:
    #: ```
    #: Ordering.from_int(cmp(3, 7))    # Ordering.Less
    #: ```
    pub static fn from_int(value: int) -> Ordering {
        if value < 0 { return Ordering.Less }
        if value > 0 { return Ordering.Greater }
        return Ordering.Equal
    }
}

#: An immutable two-field record, ordered by `first` and then `second`.
#:
#: Use it where a tuple would lose its meaning, or where the result must be
#: hashable.
#:
#: ```
#: let entry = Pair("width", 3)
#: entry.first     # "width"
#: entry.second    # 3
#: ```
pub class Pair[A, B]: Comparable, Hashable, Printable {
    #: The first component.
    pub let first: A

    #: The second component.
    pub let second: B

    #: Create the pair `(first, second)`.
    pub fn init(self, first: A, second: B) {
        self.first = first
        self.second = second
    }

    #: Return the pair with its two components exchanged.
    #:
    #: ```
    #: Pair(1, 2).swap().first    # 2
    #: ```
    pub fn swap(self) -> Pair[B, A] { return Pair[B, A](self.second, self.first) }

    #: Return the components as a plain tuple, ready for destructuring.
    pub fn to_tuple(self) -> tuple[A, B] { return (self.first, self.second) }

    #: Order pairs by `first`, breaking ties with `second`.
    pub fn compare(self, other: Pair[A, B]) -> int {
        let ordering = cmp(self.first, other.first)
        if ordering != 0 { return ordering }
        return cmp(self.second, other.second)
    }

    #: Return a hash combining both components.
    pub fn hash(self) -> int {
        let first_hash = hash(self.first)
        let second_hash = hash(self.second)
        # Plain xor would collide Pair(a, b) with Pair(b, a); the fold is a
        # bijection (spends no entropy) built from `>>` alone since `<<` here
        # would raise OverflowError once a hash occupies its top bits.
        return first_hash ^ second_hash ^ second_hash >> 13
    }

    #: Return the pair rendered as `Pair(first, second)`.
    #:
    #: ```
    #: Pair(1, 2).to_str()    # "Pair(1, 2)"
    #: ```
    pub fn to_str(self) -> str { return f"Pair({self.first}, {self.second})" }

    fn __eq__(self, other: any) -> bool {
        if not isinstance(other, Pair) { return false }
        return self.first == other.first and self.second == other.second
    }

    fn __hash__(self) -> int { return self.hash() }

    fn __str__(self) -> str { return self.to_str() }
}

#: An iterator that can look one element ahead without consuming it.
#:
#: Parsers and merge loops need exactly this, so the lookahead is one deep.
pub class Peekable[T]: Iterator[T], Iterable[T] {
    let source: Iterator[T]
    var buffered: T?
    var exhausted: bool

    #: Wrap `source` so that its next element can be inspected in advance.
    pub fn init(self, source: Iterator[T]) {
        self.source = source
        self.buffered = null
        self.exhausted = false
    }

    #: Return the next element without consuming it, or null at the end.
    #:
    #: ```
    #: from std.iter import iter
    #: let letters = Peekable(iter(["a", "b"]))
    #: letters.peek()    # "a"
    #: letters.peek()    # "a"
    #: ```
    pub fn peek(self) -> T? {
        let buffered = self.buffered
        if buffered is not null { return buffered }
        if self.exhausted { return null }
        let value = self.source.next()
        if value is null {
            self.exhausted = true
            return null
        }
        self.buffered = value
        return value
    }

    #: Return the buffered element if `peek` produced one, else the next.
    pub fn next(self) -> T? {
        let buffered = self.buffered
        if buffered is not null {
            self.buffered = null
            return buffered
        }
        if self.exhausted { return null }
        let value = self.source.next()
        if value is null { self.exhausted = true }
        return value
    }

    #: Return `self`: a peekable iterator is already an iterator.
    pub fn iter(self) -> Iterator[T] { return self }
}

#: Compare any two values with a total order, returning -1, 0 or 1.
#:
#: Knows null (sorts before everything), bool, int, float, str, and lists and
#: tuples (lexicographically, shorter first when one is a prefix of the
#: other). Any type implementing `Comparable` supplies its own answer. NaN
#: sorts after every other number, so a list of floats still has a total
#: order. Anything else raises `TypeError`.
#:
#: ```
#: cmp(3, 7)                  # -1
#: cmp("b", "a")              # 1
#: cmp([1, 2], [1, 2, 3])     # -1
#: ```
pub fn cmp(a: any, b: any) -> int {
    if a is null {
        if b is null { return 0 }
        return -1
    }
    if b is null { return 1 }
    if isinstance(a, bool) or isinstance(b, bool) {
        if not (isinstance(a, bool) and isinstance(b, bool)) {
            throw TypeError(f"cannot compare {type_of(a)} with {type_of(b)}")
        }
        return _sign(_bool_to_int(a) - _bool_to_int(b))
    }
    if _is_number(a) and _is_number(b) { return _cmp_number(a, b) }
    if isinstance(a, str) and isinstance(b, str) { return _sign(__prim__.str_cmp(a, b)) }
    if isinstance(a, Comparable) { return _sign(a.compare(b)) }
    if _is_sequence(a) and _is_sequence(b) { return _cmp_sequence(a, b) }
    throw TypeError(f"cannot compare {type_of(a)} with {type_of(b)}")
}

#: Return `value` unchanged.
#:
#: Pass it where an API wants a key function and the value is already the key.
pub fn identity(value: any) -> any { return value }

#: Return a function of no arguments that always yields `value`.
#:
#: It is the standard factory for a fixed default, as in a default-dict.
#:
#: ```
#: let zero = always(0)
#: zero()    # 0
#: ```
pub fn always(value: any) -> fn() -> any { return || value }

#: Do nothing.
#:
#: The default for an optional callback, so that a call site can invoke the hook
#: unconditionally instead of testing it for null on every iteration.
pub fn noop() -> void { }

# Private lazy adapters returned by `Iterator`'s default methods; their
# methods implement ones already documented on the trait, so no docs here.
class MapIter[T, U]: Iterator[U], Iterable[U] {
    let source: Iterator[T]
    let transform: fn(T) -> U

    pub fn init(self, source: Iterator[T], transform: fn(T) -> U) {
        self.source = source
        self.transform = transform
    }

    pub fn next(self) -> U? {
        let value = self.source.next()
        if value is null { return null }
        return self.transform(value)
    }

    pub fn iter(self) -> Iterator[U] { return self }
}

class FilterIter[T]: Iterator[T], Iterable[T] {
    let source: Iterator[T]
    let predicate: fn(T) -> bool

    pub fn init(self, source: Iterator[T], predicate: fn(T) -> bool) {
        self.source = source
        self.predicate = predicate
    }

    pub fn next(self) -> T? {
        var value = self.source.next()
        while value is not null {
            if self.predicate(value) { return value }
            value = self.source.next()
        }
        return null
    }

    pub fn iter(self) -> Iterator[T] { return self }
}

class TakeIter[T]: Iterator[T], Iterable[T] {
    let source: Iterator[T]
    var remaining: int

    pub fn init(self, source: Iterator[T], count: int) {
        self.source = source
        self.remaining = count
    }

    pub fn next(self) -> T? {
        if self.remaining == 0 { return null }
        let value = self.source.next()
        if value is null {
            self.remaining = 0
            return null
        }
        self.remaining -= 1
        return value
    }

    pub fn iter(self) -> Iterator[T] { return self }
}

class DropIter[T]: Iterator[T], Iterable[T] {
    let source: Iterator[T]
    var remaining: int

    pub fn init(self, source: Iterator[T], count: int) {
        self.source = source
        self.remaining = count
    }

    pub fn next(self) -> T? {
        # The skip is done on the first call, not in init, so that constructing
        # the adapter stays free and the source is not touched until asked.
        while self.remaining > 0 {
            self.remaining -= 1
            if self.source.next() is null {
                self.remaining = 0
                return null
            }
        }
        return self.source.next()
    }

    pub fn iter(self) -> Iterator[T] { return self }
}

class EnumerateIter[T]: Iterator[tuple[int, T]], Iterable[tuple[int, T]] {
    let source: Iterator[T]
    var index: int

    pub fn init(self, source: Iterator[T]) {
        self.source = source
        self.index = 0
    }

    pub fn next(self) -> tuple[int, T]? {
        let value = self.source.next()
        if value is null { return null }
        let numbered = (self.index, value)
        self.index += 1
        return numbered
    }

    pub fn iter(self) -> Iterator[tuple[int, T]] { return self }
}

class ZipIter[A, B]: Iterator[tuple[A, B]], Iterable[tuple[A, B]] {
    let left: Iterator[A]
    let right: Iterator[B]

    pub fn init(self, left: Iterator[A], right: Iterator[B]) {
        self.left = left
        self.right = right
    }

    pub fn next(self) -> tuple[A, B]? {
        let left_value = self.left.next()
        if left_value is null { return null }
        let right_value = self.right.next()
        if right_value is null { return null }
        return (left_value, right_value)
    }

    pub fn iter(self) -> Iterator[tuple[A, B]] { return self }
}

class ChainIter[T]: Iterator[T], Iterable[T] {
    let first: Iterator[T]
    let second: Iterator[T]
    var on_first: bool

    pub fn init(self, first: Iterator[T], second: Iterator[T]) {
        self.first = first
        self.second = second
        self.on_first = true
    }

    pub fn next(self) -> T? {
        if self.on_first {
            let value = self.first.next()
            if value is not null { return value }
            self.on_first = false
        }
        return self.second.next()
    }

    pub fn iter(self) -> Iterator[T] { return self }
}

class StepByIter[T]: Iterator[T], Iterable[T] {
    let source: Iterator[T]
    let step: int
    var started: bool

    pub fn init(self, source: Iterator[T], step: int) {
        self.source = source
        self.step = step
        self.started = false
    }

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

    pub fn iter(self) -> Iterator[T] { return self }
}

class FlattenIter: Iterator[any], Iterable[any] {
    let source: Iterator[any]
    var current: Iterator[any]?

    pub fn init(self, source: Iterator[any]) {
        self.source = source
        self.current = null
    }

    pub fn next(self) -> any {
        loop {
            var active = self.current
            if active is null {
                let nested = self.source.next()
                if nested is null { return null }
                active = nested.iter()
                self.current = active
            }
            let value = active.next()
            if value is not null { return value }
            self.current = null
        }
    }

    pub fn iter(self) -> Iterator[any] { return self }
}

fn _sign(value: int) -> int {
    if value < 0 { return -1 }
    if value > 0 { return 1 }
    return 0
}

fn _bool_to_int(value: bool) -> int {
    return if value { 1 } else { 0 }
}

fn _is_number(value: any) -> bool {
    return isinstance(value, int) or isinstance(value, float)
}

fn _is_sequence(value: any) -> bool {
    return isinstance(value, list) or isinstance(value, tuple)
}

fn _cmp_number(a: any, b: any) -> int {
    if __prim__.lt(a, b) { return -1 }
    if __prim__.lt(b, a) { return 1 }
    if __prim__.eq(a, b) { return 0 }
    # Neither ordered nor equal: at least one side is NaN. Sorting a list with a
    # NaN in it must still terminate with a consistent order, so NaN goes last.
    let a_is_nan = not __prim__.eq(a, a)
    let b_is_nan = not __prim__.eq(b, b)
    if a_is_nan and b_is_nan { return 0 }
    return if a_is_nan { 1 } else { -1 }
}

fn _cmp_sequence(a: any, b: any) -> int {
    let shared = min(a.len(), b.len())
    for index in 0..shared {
        let ordering = cmp(a[index], b[index])
        if ordering != 0 { return ordering }
    }
    return _sign(a.len() - b.len())
}
