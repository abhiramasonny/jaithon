#: The itertools that terminate with their shortest input: `accumulate`,
#: `chain`, `compress`, `dropwhile`, `filterfalse`, `groupby`, `islice`,
#: `pairwise`, `batched`, `starmap`, `takewhile`, `tee` and `zip_longest`.
#:
#: Each one is lazy in the same sense the rest of `std.iter` is: it holds its
#: sources, computes nothing until `next` is called, and pulls only as far as
#: the consumer asks. `tee` and `groupby` are the two that buffer, and each
#: says how much below.
#:
#: `chain` and `zip_longest` are classes rather than functions, because
#: CPython reaches part of their surface through a name the Jaithon signature
#: cannot carry: `chain.from_iterable` is an attribute of the callable, and
#: `zip_longest`'s `fillvalue` is keyword-only *after* `*iterables`, which
#: Jaithon rejects as E0112 — a parameter may not follow a rest parameter.
#: Both keep the CPython spelling for the common call and move the rest onto a
#: static method.
#:
#: ```
#: from std.itertools import chain, islice, pairwise
#: chain([1, 2], [3]).collect()                 # [1, 2, 3]
#: chain.from_iterable([[1, 2], [3]]).collect() # [1, 2, 3]
#: islice("abcdefg", 2, 5).collect()            # ["c", "d", "e"]
#: pairwise([1, 2, 3]).collect()                # [(1, 2), (2, 3)]
#: ```
from std.iter import Iter, iter

class _Accumulate extends Iter[any] {
    let source: Iter[any]
    let f: any
    let initial: any
    var total: any = null
    var started: bool = false
    var done: bool = false

    #: Run `f` over `source`, yielding every running total.
    pub fn init(self, source: Iter[any], f: any, initial: any) {
        self.source = source
        self.f = f
        self.initial = initial
    }

    pub fn next(self) -> any {
        if self.done { return null }
        if not self.started {
            self.started = true
            if self.initial is not null {
                self.total = self.initial
                return self.total
            }
            let first = self.source.next()
            if first is null {
                self.done = true
                return null
            }
            self.total = first
            return self.total
        }
        let value = self.source.next()
        if value is null {
            self.done = true
            return null
        }
        self.total = self.f is null ? self.total + value : self.f(self.total, value)
        return self.total
    }
}

class _ISlice extends Iter[any] {
    let source: Iter[any]
    let stop: any
    let step: int
    var wanted: int
    var seen: int = 0
    var done: bool = false

    #: Yield the source's elements at `start`, `start + step`, ... below `stop`.
    pub fn init(self, source: Iter[any], start: int, stop: any, step: int) {
        self.source = source
        self.stop = stop
        self.step = step
        self.wanted = start
    }

    pub fn next(self) -> any {
        if self.done { return null }
        if self.stop is not null and self.wanted >= self.stop {
            self.done = true
            return null
        }
        while self.seen <= self.wanted {
            let value = self.source.next()
            if value is null {
                self.done = true
                return null
            }
            self.seen += 1
            if self.seen - 1 == self.wanted {
                self.wanted += self.step
                return value
            }
        }
        self.done = true
        return null
    }
}

class _Batched extends Iter[any] {
    let source: Iter[list[any]]
    let size: int
    let strict: bool

    #: Cut `source` into tuples of `size`, refusing a short last one when asked.
    pub fn init(self, source: Iter[list[any]], size: int, strict: bool) {
        self.source = source
        self.size = size
        self.strict = strict
    }

    pub fn next(self) -> any {
        let batch = self.source.next()
        if batch is null { return null }
        if self.strict and batch.len() != self.size {
            throw ValueError("batched(): incomplete batch")
        }
        return tuple(batch)
    }
}

class _TeeBuffer {
    let source: Iter[any]
    let values: list[any] = []
    var done: bool = false

    #: Hold everything `source` has produced that some branch has not read.
    pub fn init(self, source: Iter[any]) {
        self.source = source
    }

    #: Return element `index`, pulling the source far enough to reach it.
    pub fn at(self, index: int) -> any {
        while not self.done and self.values.len() <= index {
            let value = self.source.next()
            if value is null {
                self.done = true
            } else {
                self.values.push(value)
            }
        }
        if index < self.values.len() { return self.values[index] }
        return null
    }
}

class _TeeBranch extends Iter[any] {
    let buffer: _TeeBuffer
    var index: int = 0

    #: Read one shared buffer at a position of this branch's own.
    pub fn init(self, buffer: _TeeBuffer) {
        self.buffer = buffer
    }

    pub fn next(self) -> any {
        let value = self.buffer.at(self.index)
        if value is null { return null }
        self.index += 1
        return value
    }
}

class _ZipLongest extends Iter[any] {
    let sources: list[any] = []
    let spent: list[bool] = []
    let fillvalue: any
    var done: bool = false

    #: Pair `iterables` off, padding the short ones with `fillvalue`.
    pub fn init(self, iterables: list[any], fillvalue: any) {
        self.fillvalue = fillvalue
        for source in iterables {
            self.sources.push(iter(source))
            self.spent.push(false)
        }
    }

    pub fn next(self) -> any {
        if self.done or self.sources.len() == 0 { return null }
        let row: list[any] = []
        var alive = 0
        for i in 0..self.sources.len() {
            if self.spent[i] {
                row.push(self.fillvalue)
            } else {
                let value = self.sources[i].next()
                if value is null {
                    self.spent[i] = true
                    row.push(self.fillvalue)
                } else {
                    row.push(value)
                    alive += 1
                }
            }
        }
        if alive == 0 {
            self.done = true
            return null
        }
        return tuple(row)
    }
}

class _Grouper extends Iter[any] {
    let owner: groupby
    let target: any
    let id: int

    #: Read the run of `owner` whose key is `target`, while it is still current.
    pub fn init(self, owner: groupby, target: any, id: int) {
        self.owner = owner
        self.target = target
        self.id = id
    }

    pub fn next(self) -> any {
        return self.owner.advance(self.target, self.id)
    }
}

#: Several iterables end to end, as one iterator.
#:
#: A class rather than a function so that `from_iterable` keeps its CPython
#: spelling — in CPython `chain` is a class too, and `chain.from_iterable` an
#: alternate constructor on it. Both forms are lazy in both dimensions: the
#: outer sequence of iterables is pulled one at a time, so
#: `chain.from_iterable` accepts an infinite outer iterable.
#:
#: ```
#: chain([1, 2], (3,), "45").collect()   # [1, 2, 3, "4", "5"]
#: chain().collect()                     # []
#: ```
pub class chain extends Iter[any] {
    let inner: Iter[any]

    #: Concatenate `iterables` in the order given.
    pub fn init(self, ...iterables: any) {
        self.inner = iter(iterables).flatten()
    }

    #: Concatenate the iterables that `iterables` itself yields.
    #:
    #: ```
    #: chain.from_iterable(["ab", "cd"]).collect()   # ["a", "b", "c", "d"]
    #: ```
    pub static fn from_iterable(iterables: any) -> Iter[any] {
        return iter(iterables).flatten()
    }

    #: Pull from the current source, opening the next when it runs out.
    pub fn next(self) -> any {
        return self.inner.next()
    }
}

#: Several iterables paired off, padded to the length of the longest.
#:
#: A class because CPython's `fillvalue` is keyword-only behind `*iterables`,
#: and Jaithon refuses a parameter after a rest parameter (E0112). The default
#: fill is null, exactly as CPython's is None; `with_fill` supplies another.
#: With no iterables at all it yields nothing, as CPython does.
#:
#: ```
#: zip_longest([1, 2], "a").collect()               # [(1, "a"), (2, null)]
#: zip_longest.with_fill(0, [1, 2], "a").collect()  # [(1, "a"), (2, 0)]
#: ```
pub class zip_longest extends Iter[any] {
    let inner: _ZipLongest

    #: Pair `iterables` off, padding the short ones with null.
    pub fn init(self, ...iterables: any) {
        self.inner = _ZipLongest(iterables, null)
    }

    #: Pair `iterables` off, padding the short ones with `fillvalue`.
    pub static fn with_fill(fillvalue: any, ...iterables: any) -> Iter[any] {
        return _ZipLongest(iterables, fillvalue)
    }

    #: Take one element from every live source and tuple them up.
    pub fn next(self) -> any {
        return self.inner.next()
    }
}

#: The runs of consecutive elements that share a key.
#:
#: Each `next` answers a `(key, group)` pair whose `group` is itself an
#: iterator, and — as in CPython — the group is a view on the one underlying
#: source rather than a copy of the run. Advancing the `groupby` invalidates
#: the group it handed out last, so a group that was not read is silently
#: skipped and a group read after the next `next` is empty. Materialise as you
#: go if you need the runs to outlive the walk:
#:
#: ```
#: let runs: list[any] = []
#: for pair in groupby([1, 1, 2, 1]) { runs.push((pair[0], pair[1].collect())) }
#: runs    # [(1, [1, 1]), (2, [2]), (1, [1])]
#: ```
#:
#: `key` defaults to null, which groups by the elements themselves.
pub class groupby extends Iter[any] {
    let source: Iter[any]
    let key: any
    var current_value: any = null
    var current_key: any = null
    var has_current: bool = false
    var target_key: any = null
    var has_target: bool = false
    var id: int = 0
    var done: bool = false

    #: Group the consecutive elements of `iterable` by `key`.
    pub fn init(self, iterable: any, key: any = null) {
        self.source = iter(iterable)
        self.key = key
    }

    #: Skip whatever is left of the last group and open the next one.
    pub fn next(self) -> any {
        self.id += 1
        if self.done { return null }
        while not self.has_current or self.has_target and self.current_key == self.target_key {
            let value = self.source.next()
            if value is null {
                self.has_current = false
                self.done = true
                return null
            }
            self.current_value = value
            self.current_key = self.key_of(value)
            self.has_current = true
        }
        self.target_key = self.current_key
        self.has_target = true
        return (self.current_key, _Grouper(self, self.target_key, self.id))
    }

    #: Apply the key function, or return the element when there is none.
    #:
    #: Public because the group iterator is a separate class and would not
    #: otherwise reach it.
    pub fn key_of(self, value: any) -> any {
        if self.key is null { return value }
        return self.key(value)
    }

    #: Yield one more element of the group identified by `target` and `id`.
    #:
    #: Answers null once that group is over — because its key changed, its
    #: source ran out, or the `groupby` has moved on to a later group. Public
    #: for the same reason `key_of` is.
    pub fn advance(self, target: any, id: int) -> any {
        if self.id != id or not self.has_current { return null }
        if self.current_key != target { return null }
        let value = self.current_value
        let following = self.source.next()
        if following is null {
            self.has_current = false
            self.done = true
        } else {
            self.current_value = following
            self.current_key = self.key_of(following)
        }
        return value
    }
}

#: The running totals of `iterable` under `func`.
#:
#: `func` defaults to null, which adds. `initial`, when it is not null, is
#: yielded first and every element folds into it — so the result is one longer
#: than the input, and an empty input still yields `initial`. That null is the
#: absent case is CPython's own convention, since its default is `None` too.
#:
#: ```
#: accumulate([1, 2, 3, 4]).collect()                     # [1, 3, 6, 10]
#: accumulate([1, 2, 3], |a, b| a * b).collect()          # [1, 2, 6]
#: accumulate([1, 2, 3], null, 100).collect()             # [100, 101, 103, 106]
#: accumulate([]).collect()                               # []
#: ```
pub fn accumulate(iterable: any, func: any = null, initial: any = null) -> Iter[any] {
    return _Accumulate(iter(iterable), func, initial)
}

#: The elements of `data` whose matching selector is truthy.
#:
#: Stops with whichever of the two runs out first, and truthiness is
#: `bool`'s — `0`, `""`, `[]` and null all deselect.
#:
#: ```
#: compress("abcdef", [1, 0, 1, 0, 1, 1]).collect()   # ["a", "c", "e", "f"]
#: ```
pub fn compress(data: any, selectors: any) -> Iter[any] {
    return iter(data).zip(iter(selectors)).filter(|pair| bool(pair[1])).map(|pair| pair[0])
}

#: Drop elements while `predicate` holds, then yield every one that follows.
#:
#: The element that first fails `predicate` is yielded, and `predicate` is
#: never called again after it.
#:
#: ```
#: dropwhile(|n| n < 5, [1, 4, 6, 4, 1]).collect()   # [6, 4, 1]
#: ```
pub fn dropwhile(predicate: any, iterable: any) -> Iter[any] {
    return iter(iterable).drop_while(|value| bool(predicate(value)))
}

#: Yield elements while `predicate` holds, then stop.
#:
#: The element that first fails `predicate` is consumed from the source and
#: discarded, which is what CPython does too.
#:
#: ```
#: takewhile(|n| n < 5, [1, 4, 6, 4, 1]).collect()   # [1, 4]
#: ```
pub fn takewhile(predicate: any, iterable: any) -> Iter[any] {
    return iter(iterable).take_while(|value| bool(predicate(value)))
}

#: Yield the elements for which `predicate` is false.
#:
#: A null `predicate` keeps the falsy elements themselves, as CPython's
#: `filterfalse(None, ...)` does.
#:
#: ```
#: filterfalse(|n| n % 2, [0, 1, 2, 3, 4]).collect()   # [0, 2, 4]
#: filterfalse(null, [1, 0, 2, 0]).collect()           # [0, 0]
#: ```
pub fn filterfalse(predicate: any, iterable: any) -> Iter[any] {
    if predicate is null { return iter(iterable).filter(|value| not bool(value)) }
    return iter(iterable).filter(|value| not bool(predicate(value)))
}

#: A slice of an iterable, by position, without materialising what it skips.
#:
#: The argument list is CPython's, and it is variadic for the same reason
#: CPython overloads on arity: `islice(it, stop)`, `islice(it, start, stop)`
#: and `islice(it, start, stop, step)` all mean different things. Any of the
#: three may be null, meaning 0, unbounded and 1 respectively. Negative values
#: are rejected, since a lazy source has no end to count back from.
#:
#: Only the elements up to the last one yielded are pulled, so what is left of
#: a shared source after an `islice` over it is what CPython leaves.
#:
#: ```
#: islice("abcdefg", 2).collect()          # ["a", "b"]
#: islice("abcdefg", 2, 5).collect()       # ["c", "d", "e"]
#: islice("abcdefg", 0, null, 2).collect() # ["a", "c", "e", "g"]
#: ```
pub fn islice(iterable: any, ...args: any) -> Iter[any] {
    var start: any = 0
    var stop: any = null
    var step: any = 1
    if args.len() == 1 {
        stop = args[0]
    } elif args.len() == 2 or args.len() == 3 {
        start = args[0]
        stop = args[1]
        if args.len() == 3 { step = args[2] }
    } else {
        throw TypeError(f"islice expected 2 to 4 arguments, got {args.len() + 1}")
    }
    let first = _bound(start, 0, "Indices")
    let last = stop is null ? null : _bound(stop, 0, "Stop argument")
    let stride = _bound(step, 1, "Step")
    if stride < 1 {
        throw ValueError("Step for islice() must be a positive integer or None.")
    }
    return _ISlice(iter(iterable), first, last, stride)
}

#: Every element of `iterable` paired with the one after it.
#:
#: Fewer than two elements yields nothing.
#:
#: ```
#: pairwise("abcd").collect()   # [("a", "b"), ("b", "c"), ("c", "d")]
#: pairwise([1]).collect()      # []
#: ```
pub fn pairwise(iterable: any) -> Iter[any] {
    return iter(iterable).windows(2).map(|pair| tuple(pair))
}

#: `iterable` cut into tuples of `n`, the last one possibly shorter.
#:
#: `strict` raises `ValueError` instead of yielding that short last batch,
#: matching CPython 3.13. It is an ordinary parameter here rather than a
#: keyword-only one, which Jaithon has no way to declare.
#:
#: ```
#: batched("abcdefg", 3).collect()   # [("a", "b", "c"), ("d", "e", "f"), ("g",)]
#: batched([], 3).collect()          # []
#: ```
pub fn batched(iterable: any, n: int, strict: bool = false) -> Iter[any] {
    if n < 1 { throw ValueError("n must be at least one") }
    return _Batched(iter(iterable).chunks(n), n, strict)
}

#: Call `function` with each element of `iterable` spread as its arguments.
#:
#: Every element has to be a list, tuple, string or range — something `list`
#: accepts — because a spread call needs a list.
#:
#: ```
#: starmap(|a, b| a ** b, [(2, 5), (3, 2)]).collect()   # [32, 9]
#: ```
pub fn starmap(function: any, iterable: any) -> Iter[any] {
    return iter(iterable).map(|args| function(...list(args)))
}

#: `n` independent iterators over one iterable.
#:
#: The branches share one buffer of what the source has produced, so a branch
#: that runs ahead makes the others cheap and a branch left behind is what
#: decides how much is held. Consuming one branch to the end therefore buffers
#: the whole source. Once the source is handed to `tee` it must not be used
#: directly, and a branch that has been read to exhaustion stays exhausted
#: while the others carry on.
#:
#: ```
#: let (a, b) = tee([1, 2, 3])
#: a.collect()   # [1, 2, 3]
#: b.collect()   # [1, 2, 3]
#: tee([1], 0)   # ()
#: ```
pub fn tee(iterable: any, n: int = 2) -> tuple {
    if n < 0 { throw ValueError("n must be >= 0") }
    let buffer = _TeeBuffer(iter(iterable))
    let branches: list[any] = []
    for i in 0..n { branches.push(_TeeBranch(buffer)) }
    return tuple(branches)
}

fn _bound(value: any, fallback: int, role: str) -> int {
    if value is null { return fallback }
    if type_of(value) != "int" {
        throw ValueError(f"{role} for islice() must be None or an integer.")
    }
    if value < 0 {
        throw ValueError(f"{role} for islice() must be None or an integer: 0 <= x.")
    }
    return value
}
