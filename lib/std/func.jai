#: Function combinators, and `Result`/`Option` for recoverable failure.
#:
#: The combinators build functions out of other functions: `compose` and `pipe`
#: for wiring, `curry` and `partial` for fixing arguments ahead of time,
#: `memoize`, `once` and `Lazy` for controlling when work actually happens.
#:
#: `Result` and `Option` are the ergonomic alternative to exceptions when a
#: failure is expected rather than exceptional. `try_call` turns a raising
#: function into a `Result`, and `map`/`and_then` thread the rest of the
#: computation through it without a check at every step:
#:
#: ```
#: try_call(|s| int(s), ["12"]).map(|n| n * 2).unwrap_or(0)   # 24
#: try_call(|s| int(s), ["x"]).map(|n| n * 2).unwrap_or(0)    # 0
#: ```
#:
#: Reach for an exception when the caller cannot reasonably continue, and for a
#: `Result` when it can.

#: A value that is either present or absent.
#:
#: A bare `T?` is the right choice for a single nullable value. `Option` earns
#: its keep when absence has to survive a chain of transformations, because
#: `map` and `and_then` short-circuit instead of every step re-testing for null.
#:
#: ```
#: Option.Some(2).map(|x| x * 3).unwrap_or(0)   # 6
#: Option.None.map(|x| x * 3).unwrap_or(0)      # 0
#: ```
pub enum Option[T] {
    #: A value is present. `Some(null)` still counts as present; it is
    #: `Option.of` that folds a null into `None`.
    Some(value: T),
    #: No value. `map` and `and_then` skip their function, `unwrap_or` gives
    #: the default, and `unwrap` raises `RuntimeError`.
    None,

    #: Wrap a nullable value, turning null into `None`.
    #:
    #: ```
    #: Option.of(3).is_some()      # true
    #: Option.of(null).is_some()   # false
    #: ```
    pub static fn of(value: T?) -> Option[T] {
        if value is null { return Option.None }
        return Option.Some(value)
    }

    #: True when a value is present.
    pub fn is_some(self) -> bool {
        return match self {
            Option.Some(_) => true,
            Option.None => false,
        }
    }

    #: True when no value is present.
    pub fn is_none(self) -> bool {
        return not self.is_some()
    }

    #: Apply `f` to the contained value, if there is one.
    #:
    #: ```
    #: Option.Some("ab").map(|s| s.len()).unwrap()   # 2
    #: ```
    pub fn map[U](self, f: fn(T) -> U) -> Option[U] {
        return match self {
            Option.Some(value) => Option.Some(f(value)),
            Option.None => Option.None,
        }
    }

    #: Apply `f`, which returns an `Option` of its own, and flatten the result.
    #:
    #: Chain the steps that can themselves come up empty.
    #:
    #: ```
    #: Option.Some(4).and_then(|x| if x > 2 { Option.Some(x) } else { Option.None }).unwrap()
    #: # 4
    #: ```
    pub fn and_then[U](self, f: fn(T) -> Option[U]) -> Option[U] {
        return match self {
            Option.Some(value) => f(value),
            Option.None => Option.None,
        }
    }

    #: Return the contained value, raising `RuntimeError` when there is none.
    #:
    #: Use `unwrap_or` wherever absence is an ordinary outcome.
    pub fn unwrap(self) -> T {
        match self {
            Option.Some(value) => { return value }
            Option.None => { throw RuntimeError("unwrap on Option.None") }
        }
    }

    #: Return the contained value, or `default` when there is none.
    pub fn unwrap_or(self, default: T) -> T {
        return match self {
            Option.Some(value) => value,
            Option.None => default,
        }
    }

    #: Convert to a `Result`, using `error` to describe the missing case.
    #:
    #: ```
    #: Option.Some(1).ok_or("missing").unwrap()   # 1
    #: Option.None.ok_or("missing").is_err()      # true
    #: ```
    pub fn ok_or[E](self, error: E) -> Result[T, E] {
        return match self {
            Option.Some(value) => Result.Ok(value),
            Option.None => Result.Err(error),
        }
    }

    #: Convert back to a plain nullable, for an API that expects `T?`.
    pub fn or_null(self) -> T? {
        return match self {
            Option.Some(value) => value,
            Option.None => null,
        }
    }
}

#: The outcome of an operation that can fail: a value or an error.
#:
#: ```
#: Result.Ok(2).map(|x| x + 1).unwrap()        # 3
#: Result.Err("nope").map(|x| x + 1).is_err()  # true
#: ```
pub enum Result[T, E] {
    #: The operation succeeded and produced `value`. `map` and `and_then` act
    #: on this branch; `map_err` passes it through untouched.
    Ok(value: T),
    #: The operation failed, and `error` says how. `E` is unconstrained, so an
    #: error can be an `Error`, a string, or an enum of failure cases.
    Err(error: E),

    #: True when the operation succeeded.
    pub fn is_ok(self) -> bool {
        return match self {
            Result.Ok(_) => true,
            Result.Err(_) => false,
        }
    }

    #: True when the operation failed.
    pub fn is_err(self) -> bool {
        return not self.is_ok()
    }

    #: Apply `f` to the success value, leaving an error untouched.
    pub fn map[U](self, f: fn(T) -> U) -> Result[U, E] {
        return match self {
            Result.Ok(value) => Result.Ok(f(value)),
            Result.Err(error) => Result.Err(error),
        }
    }

    #: Apply `f` to the error, leaving a success untouched.
    #:
    #: For translating a low-level failure into the vocabulary of the caller.
    pub fn map_err[F](self, f: fn(E) -> F) -> Result[T, F] {
        return match self {
            Result.Ok(value) => Result.Ok(value),
            Result.Err(error) => Result.Err(f(error)),
        }
    }

    #: Apply `f`, which returns a `Result` of its own, and flatten the result.
    #:
    #: This is the sequencing operator: the first error short-circuits the rest.
    #:
    #: ```
    #: Result.Ok(9).and_then(|x| Result.Ok(x + 1)).unwrap()   # 10
    #: ```
    pub fn and_then[U](self, f: fn(T) -> Result[U, E]) -> Result[U, E] {
        return match self {
            Result.Ok(value) => f(value),
            Result.Err(error) => Result.Err(error),
        }
    }

    #: Return the success value, raising `RuntimeError` on an error.
    #:
    #: The message carries the error, so an unwrapped failure is still readable
    #: in a traceback.
    pub fn unwrap(self) -> T {
        match self {
            Result.Ok(value) => { return value }
            Result.Err(error) => { throw RuntimeError(f"unwrap on an error result: {error}") }
        }
    }

    #: Return the success value, or `default` on an error.
    pub fn unwrap_or(self, default: T) -> T {
        return match self {
            Result.Ok(value) => value,
            Result.Err(_) => default,
        }
    }

    #: Discard the error and keep the success as an `Option`.
    pub fn ok(self) -> Option[T] {
        return match self {
            Result.Ok(value) => Option.Some(value),
            Result.Err(_) => Option.None,
        }
    }
}

#: A value computed at most once, the first time it is asked for.
#:
#: ```
#: var calls = 0
#: let answer = Lazy(fn() -> int {
#:     calls += 1
#:     return 42
#: })
#: answer.is_forced()   # false
#: answer.get()         # 42
#: answer.get()         # 42
#: calls                # 1
#: ```
pub class Lazy[T] {
    let thunk: fn() -> T
    var value: any = null
    var forced: bool = false

    #: Hold `thunk` without running it.
    pub fn init(self, thunk: fn() -> T) {
        self.thunk = thunk
    }

    #: Run the thunk if it has not run yet, and return its value.
    #:
    #: A thunk that raises is not remembered: the next `get` runs it again,
    #: which is what you want for a retryable initialisation and what you must
    #: guard against for an expensive one.
    pub fn get(self) -> T {
        if not self.forced {
            self.value = self.thunk()
            self.forced = true
        }
        return self.value
    }

    #: True once the thunk has run.
    pub fn is_forced(self) -> bool {
        return self.forced
    }

    fn __str__(self) -> str {
        if not self.forced { return "Lazy(<unforced>)" }
        return f"Lazy({self.value})"
    }
}

# A least-recently-used cache with a fixed capacity, backing `memoize`.
#
# Entries live in parallel slot lists, and `newer`/`older` thread a doubly
# linked recency order through the slot indices, so lookup, insertion and
# eviction are all O(1) without ever removing a dict entry. `index_of` can
# still name a key whose slot has since been recycled, which is why `find`
# re-checks the slot's own key; the map is rebuilt once it holds twice the live
# entries, costing O(capacity) once per capacity evictions.
#
# `find` and `value_at` are separate so that a cached null is still a hit.
class _LruCache {
    let capacity: int
    let keys: list[any] = []
    let values: list[any] = []
    let newer: list[int] = []
    let older: list[int] = []
    var index_of: dict[any, int] = {}
    var head: int = -1
    var tail: int = -1

    #: Hold at most `capacity` entries; a capacity of 0 means unbounded.
    pub fn init(self, capacity: int) {
        self.capacity = capacity
    }

    #: The slot holding `key`, or -1. A hit becomes the most recently used.
    pub fn find(self, key: any) -> int {
        let slot = self.index_of.get(key, -1)
        if slot < 0 { return -1 }
        if self.keys[slot] != key { return -1 }
        self._touch(slot)
        return slot
    }

    #: The value stored in `slot`, which must have come from `find`.
    pub fn value_at(self, slot: int) -> any {
        return self.values[slot]
    }

    #: Store `value` under `key`, evicting the least recently used entry if full.
    pub fn put(self, key: any, value: any) -> void {
        let existing = self.find(key)
        if existing >= 0 {
            self.values[existing] = value
            return
        }
        if self.capacity > 0 and self.keys.len() >= self.capacity {
            let victim = self.tail
            self.keys[victim] = key
            self.values[victim] = value
            self.index_of[key] = victim
            self._touch(victim)
            if self.index_of.len() > 2 * self.capacity { self._compact() }
            return
        }
        let slot = self.keys.len()
        self.keys.push(key)
        self.values.push(value)
        self.newer.push(-1)
        self.older.push(-1)
        self.index_of[key] = slot
        self._link_front(slot)
    }

    fn _touch(self, slot: int) -> void {
        if self.head == slot { return }
        self._unlink(slot)
        self._link_front(slot)
    }

    fn _unlink(self, slot: int) -> void {
        let more_recent = self.newer[slot]
        let less_recent = self.older[slot]
        if more_recent >= 0 {
            self.older[more_recent] = less_recent
        } else {
            self.head = less_recent
        }
        if less_recent >= 0 {
            self.newer[less_recent] = more_recent
        } else {
            self.tail = more_recent
        }
    }

    fn _link_front(self, slot: int) -> void {
        self.newer[slot] = -1
        self.older[slot] = self.head
        if self.head >= 0 { self.newer[self.head] = slot }
        self.head = slot
        if self.tail < 0 { self.tail = slot }
    }

    fn _compact(self) -> void {
        let fresh: dict[any, int] = {}
        for slot in 0..self.keys.len() { fresh[self.keys[slot]] = slot }
        self.index_of = fresh
    }
}

#: Return `value` unchanged.
#:
#: The identity element of `compose` and `pipe`, and the neutral choice wherever
#: an API asks for a transformation you do not want to apply.
#:
#: ```
#: identity(3)   # 3
#: ```
pub fn identity[T](value: T) -> T {
    return value
}

#: Build a function that ignores its arguments and always returns `value`.
#:
#: ```
#: let zero = constant(0)
#: zero()       # 0
#: zero(1, 2)   # 0
#: ```
pub fn constant[T](value: T) -> any {
    return fn(..._args: any) -> T { return value }
}

#: Compose functions right to left: `compose(f, g)(x)` is `f(g(x))`.
#:
#: With no arguments this is `identity`.
#:
#: ```
#: compose(|x| x + 1, |x| x * 2)(5)   # 11
#: ```
pub fn compose(...fs: fn(any) -> any) -> fn(any) -> any {
    return fn(value: any) -> any {
        var result = value
        var i = fs.len() - 1
        while i >= 0 {
            result = fs[i](result)
            i -= 1
        }
        return result
    }
}

#: Compose functions left to right: `pipe(f, g)(x)` is `g(f(x))`.
#:
#: The same wiring as `compose` in reading order, which is usually the clearer
#: of the two for a data-processing chain.
#:
#: ```
#: pipe(|x| x + 1, |x| x * 2)(5)   # 12
#: ```
pub fn pipe(...fs: fn(any) -> any) -> fn(any) -> any {
    return fn(value: any) -> any {
        var result = value
        for f in fs {
            result = f(result)
        }
        return result
    }
}

#: Turn an `arity`-argument function into one taking its arguments a few at a time.
#:
#: The curried function collects arguments and returns another curried function
#: until `arity` of them have arrived, then calls `f` with all of them at once.
#:
#: ```
#: let add3 = fn(a: int, b: int, c: int) -> int { return a + b + c }
#: curry(add3, 3)(1)(2)(3)   # 6
#: curry(add3, 3)(1, 2)(3)   # 6
#: curry(add3, 3)(1, 2, 3)   # 6
#: ```
pub fn curry(f: any, arity: int) -> any {
    if arity < 0 { throw ValueError(f"arity must be non-negative, got {arity}") }
    return _curry_step(f, arity, [])
}

#: Bind the leading arguments of `f`, leaving the rest for the caller.
#:
#: ```
#: let add = fn(a: int, b: int) -> int { return a + b }
#: partial(add, 10)(5)   # 15
#: ```
pub fn partial(f: any, ...bound: any) -> any {
    return fn(...rest: any) -> any { return f(...bound + rest) }
}

#: Swap the two arguments of a binary function.
#:
#: ```
#: flip(|a, b| a - b)(2, 10)   # 8
#: ```
pub fn flip(f: fn(any, any) -> any) -> fn(any, any) -> any {
    return fn(a: any, b: any) -> any { return f(b, a) }
}

#: Cache `f`'s results so that repeated calls with the same arguments are free.
#:
#: `key` maps the argument list to a cache key; it defaults to `repr` of that
#: list, which is correct for any argument with a faithful `__repr__` and is
#: worth replacing when a cheaper or more precise key exists. `max_size` bounds
#: the cache with least-recently-used eviction, and 0 leaves it unbounded.
#:
#: Only memoise a pure function: a cached call does not run `f` again, so any
#: side effect happens once.
#:
#: ```
#: var calls = 0
#: let square = memoize(fn(n: int) -> int {
#:     calls += 1
#:     return n * n
#: })
#: square(4)   # 16
#: square(4)   # 16
#: calls       # 1
#: ```
pub fn memoize(f: any, key: fn(list[any]) -> any = |args| repr(args), max_size: int = 0) -> any {
    if max_size < 0 {
        throw ValueError(f"max_size must be non-negative, got {max_size}")
    }
    let cache = _LruCache(max_size)
    return fn(...args: any) -> any {
        let cache_key = key(args)
        let slot = cache.find(cache_key)
        if slot >= 0 {
            return cache.value_at(slot)
        }
        let value = f(...args)
        cache.put(cache_key, value)
        return value
    }
}

#: Wrap `f` so that it runs at most once; later calls return the first result.
#:
#: For an initialisation that several call sites need and none of them owns.
#:
#: ```
#: var calls = 0
#: let setup = once(fn() -> int {
#:     calls += 1
#:     return 7
#: })
#: setup()   # 7
#: setup()   # 7
#: calls     # 1
#: ```
pub fn once(f: any) -> any {
    var called = false
    var result: any = null
    return fn(...args: any) -> any {
        if not called {
            result = f(...args)
            called = true
        }
        return result
    }
}

#: Wrap `f` so that only every `n`-th call reaches it.
#:
#: The wrapper returns `f`'s result on the calls that get through and null on
#: the ones it swallows, which makes it a cheap sampler for progress reporting
#: inside a hot loop.
#:
#: ```
#: var calls = 0
#: let sample = debounce_count(fn(x: int) -> int {
#:     calls += 1
#:     return x
#: }, 3)
#: sample(1)   # null
#: sample(2)   # null
#: sample(3)   # 3
#: calls       # 1
#: ```
pub fn debounce_count(f: any, n: int) -> any {
    if n < 1 { throw ValueError(f"n must be at least 1, got {n}") }
    var pending = 0
    return fn(...args: any) -> any {
        pending += 1
        if pending < n {
            return null
        }
        pending = 0
        return f(...args)
    }
}

#: Run `f` on `value` for its side effect and return `value` unchanged.
#:
#: For dropping a trace or an assertion into the middle of an expression
#: without having to break it into statements.
#:
#: ```
#: var seen = 0
#: tap(41, fn(x: int) -> void { seen = x }) + 1   # 42
#: seen                                           # 41
#: ```
pub fn tap[T](value: T, f: fn(T) -> void) -> T {
    f(value)
    return value
}

#: Apply `f` to `value` `n` times.
#:
#: ```
#: apply_n_times(|x| x * 2, 5, 1)   # 32
#: ```
pub fn apply_n_times[T](f: fn(T) -> T, n: int, value: T) -> T {
    if n < 0 { throw ValueError(f"n must be non-negative, got {n}") }
    var result = value
    for _step in 0..n { result = f(result) }
    return result
}

#: Iterate `f` from `start` until the value stops changing.
#:
#: `equals` decides when two successive values count as the same, which is
#: where a floating-point tolerance belongs. Raises `RuntimeError` after
#: `max_iterations` steps without convergence, because quietly returning a
#: value that is not a fixed point is the worse failure.
#:
#: ```
#: fixed_point(|x| if x > 1 { x // 2 } else { x }, 100)   # 1
#: let root = fixed_point(|x| (x + 2.0 / x) / 2.0, 1.0, |a, b| abs(a - b) < 1e-12)
#: abs(root - 1.4142135623730951) < 1e-9                  # true
#: ```
pub fn fixed_point[T](
    f: fn(T) -> T,
    start: T,
    equals: fn(T, T) -> bool = |a, b| a == b,
    max_iterations: int = 1000
) -> T {
    if max_iterations < 1 {
        throw ValueError(f"max_iterations must be at least 1, got {max_iterations}")
    }
    var current = start
    for _step in 0..max_iterations {
        let following = f(current)
        if equals(current, following) { return current }
        current = following
    }
    throw RuntimeError(f"no fixed point after {max_iterations} iterations, last value {current}")
}

#: Build a recursive function without giving it a name.
#:
#: `builder` receives the function being defined and returns its body, so an
#: anonymous function can call itself. This is the practical form of the Y
#: combinator, and it is what lets a recursive helper stay local to the
#: expression that needs it.
#:
#: ```
#: let factorial = recursive(|recurse| |n| if n <= 1 { 1 } else { n * recurse(n - 1) })
#: factorial(5)   # 120
#: ```
pub fn recursive(builder: fn(any) -> any) -> any {
    var target: any = null
    # The inner wrapper reads `target` at call time, by which point `builder`
    # has returned and the binding holds the finished function.
    target = builder(fn(...args: any) -> any { return target(...args) })
    return target
}

#: Call `f` with `args` and capture success or failure as a `Result`.
#:
#: The bridge from raising code into `Result` code: use it at the boundary
#: where a failure stops being exceptional and becomes an ordinary outcome the
#: caller is going to handle.
#:
#: ```
#: try_call(|x| 10 // x, [2]).unwrap()        # 5
#: try_call(|x| 10 // x, [0]).is_err()        # true
#: try_call(|x| 10 // x, [0]).unwrap_or(-1)   # -1
#: ```
pub fn try_call(f: any, args: list[any] = []) -> Result[any, Error] {
    try {
        return Result.Ok(f(...args))
    } catch error: Error {
        return Result.Err(error)
    }
}

fn _curry_step(f: any, arity: int, collected: list[any]) -> any {
    return fn(...args: any) -> any {
        let all = collected + args
        if all.len() >= arity {
            return f(...all)
        }
        return _curry_step(f, arity, all)
    }
}
