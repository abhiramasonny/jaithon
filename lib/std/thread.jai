#: OS threads: mutexes, atomics, bounded channels, a thread pool, and parallel
#: map, for and reduce over a list.
#:
#: **A worker callback (`spawn`, `ThreadPool.submit`, `parallel_map`,
#: `parallel_for`, `parallel_reduce`) must be a pure function of its inputs:**
#: read arguments and captured bindings, allocate, return a value -- that is
#: all.
#:
#: - Do not assign to a captured `var` -- it's captured by reference (spec
#:   §6), so two workers incrementing one counter race and undercount.
#: - Do not mutate a list, dict, set or object another thread can reach,
#:   including the list passed to `parallel_map`; reading is fine.
#: - Do not read a field another thread writes unless *every* access to that
#:   object happens under the same `Mutex`.
#: - `Mutex`, `Atomic` and `Channel` are the exceptions: safe to share from
#:   any thread. A worker publishes only through one of those three, or its
#:   return value.
#:
#: No condition variables exist, so `Channel.send`/`recv` and `Future.get`
#: block by spinning then sleeping with back-off capped at 2ms. A thread can
#: trail the event that woke it by up to that cap -- do not build a hard
#: deadline on it.
#:
#: A native mutex/atomic cell is owned by the `Mutex`/`Atomic` object that
#: created it and released when collected, so neither type has a `free`.
#:
#: ```
#: parallel_map([1, 2, 3, 4], |x| x * x)             # [1, 4, 9, 16]
#: parallel_reduce([1, 2, 3, 4], 0, |a, b| a + b)    # 10
#: ```
from std.core import Iterable, Iterator

# Chosen so a hand-off between threads almost always completes before the
# first sleep, keeping a saturated pool's throughput unaffected by back-off.
const _SPIN_LIMIT = 40
const _MIN_BACKOFF: float = 0.00002
const _MAX_BACKOFF: float = 0.002

#: Lists shorter than this are processed on the calling thread.
#:
#: Spawning a thread costs far more than mapping a few elements, so parallel
#: functions fall back to sequential below this length -- which also keeps
#: them deterministic in a doc example.
pub const MIN_PARALLEL_ITEMS = 64

#: Return the number of hardware threads the machine reports, at least 1.
pub fn cpu_count() -> int {
    let count = __prim__.cpu_count()
    if count < 1 { return 1 }
    return count
}

#: A mutual-exclusion lock guarding data shared between threads.
#:
#: **Not** reentrant: locking it twice from one thread deadlocks that thread.
#: Prefer `locked`, which cannot leak a lock down an exceptional path, over
#: hand-written `lock`/`unlock` pairs.
pub class Mutex {
    let handle: int

    #: Create an unlocked mutex.
    pub fn init(self) {
        self.handle = __prim__.mutex_new()
    }

    #: Acquire the lock, blocking until it is free.
    pub fn lock(self) -> void { __prim__.mutex_lock(self.handle) }

    #: Release the lock. Unlocking a lock this thread does not hold is an error.
    pub fn unlock(self) -> void { __prim__.mutex_unlock(self.handle) }

    #: Acquire the lock if it is free, and report whether it was acquired.
    pub fn try_lock(self) -> bool { return __prim__.mutex_try_lock(self.handle) }

    #: Run `body` with the lock held and return its result.
    #:
    #: The lock is released by a `defer`, so it is released even when `body`
    #: raises, and the exception continues to propagate.
    #:
    #: ```
    #: let lock = Mutex()
    #: lock.locked(|| 1 + 1)    # 2
    #: ```
    pub fn locked(self, body: fn() -> any) -> any {
        self.lock()
        defer { self.unlock() }
        return body()
    }
}

#: An integer that every thread can read and update without a lock.
#:
#: Use it for counters and flags. Anything larger needs a `Mutex` instead,
#: because a sequence of atomic operations is not itself atomic.
pub class Atomic {
    let cell: int

    #: Create an atomic integer holding `initial`.
    pub fn init(self, initial: int = 0) {
        self.cell = __prim__.atomic_new(initial)
    }

    #: Return the current value.
    pub fn get(self) -> int { return __prim__.atomic_load(self.cell) }

    #: Store `value`.
    pub fn set(self, value: int) -> void { __prim__.atomic_store(self.cell, value) }

    #: Add `delta` and return the value *after* the addition.
    #:
    #: ```
    #: let counter = Atomic(0)
    #: counter.add(5)     # 5
    #: counter.add(-2)    # 3
    #: ```
    pub fn add(self, delta: int) -> int {
        return __prim__.atomic_add(self.cell, delta) + delta
    }

    #: Store `value` and return the value it replaced.
    pub fn exchange(self, value: int) -> int {
        return __prim__.atomic_exchange(self.cell, value)
    }

    #: Store `desired` only if the current value is `expected`.
    #:
    #: Returns whether the store happened -- the building block for "exactly
    #: one thread wins".
    #:
    #: ```
    #: let flag = Atomic(0)
    #: flag.compare_and_set(0, 1)    # true
    #: flag.compare_and_set(0, 2)    # false
    #: ```
    pub fn compare_and_set(self, expected: int, desired: int) -> bool {
        return __prim__.atomic_cas(self.cell, expected, desired)
    }

    fn __str__(self) -> str { return f"Atomic({self.get()})" }
}

#: A bounded first-in-first-out queue that hands values between threads.
#:
#: `send` blocks while full and `recv` blocks while empty, so capacity is also
#: back-pressure. Closing means "no more values": queued values still arrive,
#: and `recv` returns null once the last one is taken.
#:
#: Null ends iteration, so a channel carrying null values must wrap them -- in
#: a one-element list, or a `Pair`.
#:
#: ```
#: let channel = Channel(2)
#: channel.send("a")
#: channel.len()         # 1
#: channel.recv()        # "a"
#: channel.try_recv()    # null
#: ```
pub class Channel: Iterable[any] {
    #: How many values the channel holds before `send` blocks.
    pub let capacity: int

    let lock: Mutex
    var slots: list[any]
    var head: int
    var count: int
    var closed: bool

    #: Create an empty channel holding at most `capacity` values.
    pub fn init(self, capacity: int = 1) {
        if capacity < 1 { throw ValueError(f"capacity must be at least 1, got {capacity}") }
        self.capacity = capacity
        self.lock = Mutex()
        self.slots = [null for _ in 0..capacity]
        self.head = 0
        self.count = 0
        self.closed = false
    }

    #: Append `value`, blocking while the channel is full.
    #:
    #: Raises `RuntimeError` when the channel is closed, which is a programming
    #: error: whoever closes a channel must already know no one will send again.
    pub fn send(self, value: any) -> void {
        var spins = 0
        loop {
            self.lock.lock()
            if self.closed {
                self.lock.unlock()
                throw RuntimeError("send on a closed Channel")
            }
            if self.count < self.capacity {
                self._push(value)
                self.lock.unlock()
                return
            }
            self.lock.unlock()
            spins = _back_off(spins)
        }
    }

    #: Append `value` if there is room, and report whether it was appended.
    #:
    #: Never blocks. Raises `RuntimeError` when the channel is closed.
    pub fn try_send(self, value: any) -> bool {
        self.lock.lock()
        defer { self.lock.unlock() }
        if self.closed { throw RuntimeError("try_send on a closed Channel") }
        if self.count == self.capacity { return false }
        self._push(value)
        return true
    }

    #: Remove and return the oldest value, blocking while the channel is empty.
    #:
    #: Returns null once the channel is closed and drained, which is the signal
    #: to stop:
    #:
    #:     var job = channel.recv()
    #:     while job is not null {
    #:         handle(job)
    #:         job = channel.recv()
    #:     }
    #:
    #: `for job in channel` is the same loop written with the iterator.
    pub fn recv(self) -> any {
        var spins = 0
        loop {
            self.lock.lock()
            if self.count > 0 {
                let value = self._pop()
                self.lock.unlock()
                return value
            }
            let drained = self.closed
            self.lock.unlock()
            if drained { return null }
            spins = _back_off(spins)
        }
    }

    #: Remove and return the oldest value, or null when none is available now.
    #:
    #: Never blocks. A null result means "nothing right now" and says nothing
    #: about whether the channel is closed; ask `is_closed` for that.
    pub fn try_recv(self) -> any {
        self.lock.lock()
        defer { self.lock.unlock() }
        if self.count == 0 { return null }
        return self._pop()
    }

    #: Refuse further sends and let every blocked receiver drain and finish.
    #:
    #: Closing twice is harmless.
    pub fn close(self) -> void {
        self.lock.lock()
        defer { self.lock.unlock() }
        self.closed = true
    }

    #: Report whether the channel has been closed.
    pub fn is_closed(self) -> bool {
        self.lock.lock()
        defer { self.lock.unlock() }
        return self.closed
    }

    #: Number of values waiting to be received.
    #:
    #: A snapshot: by the time the caller reads it another thread may have
    #: changed it. Use it for metrics, never for a decision.
    pub fn len(self) -> int {
        self.lock.lock()
        defer { self.lock.unlock() }
        return self.count
    }

    #: Iterate received values until the channel is closed and drained.
    #:
    #: Several threads may iterate one channel at once; each value goes to
    #: exactly one of them.
    pub fn iter(self) -> Iterator[any] { return _ChannelIter(self) }

    fn _push(self, value: any) -> void {
        self.slots[(self.head + self.count) % self.capacity] = value
        self.count += 1
    }

    fn _pop(self) -> any {
        let value = self.slots[self.head]
        self.slots[self.head] = null
        self.head = (self.head + 1) % self.capacity
        self.count -= 1
        return value
    }

    fn __len__(self) -> int { return self.len() }
}

#: A running OS thread whose result can be collected with `join`.
pub class Handle {
    let id: int
    var outcome: _Outcome?
    var joined: bool

    #: Adopt an already-started native thread `id`.
    #:
    #: `spawn` is the only correct caller: a `Handle` built for an id this
    #: module did not start will fail to join.
    pub fn init(self, id: int) {
        self.id = id
        self.outcome = null
        self.joined = false
    }

    #: Wait for the thread to finish and return the value its body produced.
    #:
    #: An exception that escaped the body is re-raised here, on the joining
    #: thread. Joining twice returns the same result again rather than blocking.
    pub fn join(self) -> any {
        if not self.joined {
            self.outcome = __prim__.thread_join(self.id)
            self.joined = true
        }
        let outcome = self.outcome ?? (throw RuntimeError(f"thread {self.id} produced no outcome"))
        return outcome.unwrap()
    }

    #: Report whether `join` has already collected this thread.
    pub fn is_joined(self) -> bool { return self.joined }
}

#: Run `body` on a new OS thread and return a handle to it.
#:
#: `body` must obey the purity contract at the top of this module. Every thread
#: must be joined exactly once; a thread nobody joins keeps its stack alive
#: until the process exits.
#:
#:     let worker = spawn(|| expensive(input))
#:     let other = expensive(second_input)
#:     let result = combine(worker.join(), other)
pub fn spawn(body: fn() -> any) -> Handle {
    return Handle(__prim__.thread_spawn(_guarded(body)))
}

#: The eventual result of work submitted to a `ThreadPool`.
pub class Future {
    let slot: Channel
    var outcome: _Outcome?

    #: Wrap the one-value channel the pool will deliver the outcome on.
    #:
    #: `ThreadPool.submit` is the only correct caller.
    pub fn init(self, slot: Channel) {
        self.slot = slot
        self.outcome = null
    }

    #: Report whether the result has arrived, without blocking.
    pub fn is_ready(self) -> bool {
        if self.outcome is not null { return true }
        let received = self.slot.try_recv()
        if received is null { return false }
        self.outcome = received
        return true
    }

    #: Wait for the work to finish and return its value.
    #:
    #: An exception raised by the submitted function is re-raised here. Calling
    #: `get` twice returns the same result. Raises `RuntimeError` if the pool
    #: shut down before the work ran.
    pub fn get(self) -> any {
        var outcome = self.outcome
        if outcome is null {
            outcome = self.slot.recv()
            if outcome is null { throw RuntimeError("the pool shut down before this job ran") }
            self.outcome = outcome
        }
        return outcome.unwrap()
    }
}

#: A fixed set of worker threads pulling jobs off one bounded queue.
#:
#: Create the pool once and keep it: the point is to pay for thread creation
#: once instead of once per job. `submit` blocks when the queue is full,
#: throttling the producer instead of letting the backlog grow.
#:
#:     let pool = ThreadPool()
#:     defer { pool.join() }
#:     let futures = paths.map(|path| pool.submit(|| checksum(path)))
#:     let sums = futures.map(|future| future.get())
pub class ThreadPool {
    let jobs: Channel
    let threads: list[Handle]

    #: Start `workers` threads, or one per hardware thread when `workers` is 0.
    #:
    #: `queue_capacity` bounds how many submitted-but-unstarted jobs are held;
    #: 0 asks for four per worker, which keeps every worker fed across a burst
    #: without letting a runaway producer queue millions of closures.
    pub fn init(self, workers: int = 0, queue_capacity: int = 0) {
        if workers < 0 { throw ValueError(f"workers must be non-negative, got {workers}") }
        if queue_capacity < 0 {
            throw ValueError(f"queue_capacity must be non-negative, got {queue_capacity}")
        }
        let size = if workers > 0 { workers } else { cpu_count() }
        let capacity = if queue_capacity > 0 { queue_capacity } else { size * 4 }
        self.jobs = Channel(capacity)
        self.threads = []
        for _ in 0..size { self.threads.push(spawn(_worker(self.jobs))) }
    }

    #: Number of worker threads.
    pub fn size(self) -> int { return self.threads.len() }

    #: Number of jobs submitted but not yet picked up by a worker.
    pub fn pending(self) -> int { return self.jobs.len() }

    #: Queue `work` and return a `Future` for its result.
    #:
    #: Blocks while the queue is full. Raises `RuntimeError` after `close`.
    pub fn submit(self, work: fn() -> any) -> Future {
        let slot = Channel(1)
        self.jobs.send(_Job(work, slot))
        return Future(slot)
    }

    #: Stop accepting work. Jobs already queued still run.
    pub fn close(self) -> void { self.jobs.close() }

    #: Close the pool and wait for every worker to finish and exit.
    #:
    #: Calling it twice is harmless, so it is safe in a `defer`.
    pub fn join(self) -> void {
        self.close()
        for thread in self.threads {
            if not thread.is_joined() { thread.join() }
        }
    }
}

#: Apply `transform` to every element of `items` in parallel, in order.
#:
#: Produces the same list as `items.map(transform)`; only the order elements
#: are *visited* in is unspecified. Each worker builds its own slice, so
#: nothing is shared or contended. `transform` must obey the purity contract
#: at the top of this module.
#:
#: `workers` is the number of threads, or 0 for one per hardware thread.
#:
#: ```
#: parallel_map([1, 2, 3, 4], |x| x * x)    # [1, 4, 9, 16]
#: ```
pub fn parallel_map(items: list[any], transform: fn(any) -> any, workers: int = 0) -> list[any] {
    let parts = _split(items.len(), _worker_count(workers, items.len()))
    if parts.len() <= 1 { return items.map(transform) }
    let handles: list[Handle] = []
    for (start, stop) in parts { handles.push(spawn(_map_chunk(items, transform, start, stop))) }
    let result: list[any] = []
    for handle in handles {
        for value in handle.join() { result.push(value) }
    }
    return result
}

#: Run `body` once per element of `items`, in parallel, for its side effects.
#:
#: The one function here whose callback need *not* be pure. Effects must still
#: be safe to interleave: write to state private to the body, an `Atomic`, a
#: `Channel`, or state guarded by a `Mutex`. Visit order is unspecified.
#:
#:     let failures = Atomic(0)
#:     parallel_for(urls, fn(url) -> void {
#:         if not fetch(url) { failures.add(1) }
#:     })
pub fn parallel_for(items: list[any], body: fn(any) -> void, workers: int = 0) -> void {
    let parts = _split(items.len(), _worker_count(workers, items.len()))
    if parts.len() <= 1 {
        for item in items { body(item) }
        return
    }
    let handles: list[Handle] = []
    for (start, stop) in parts { handles.push(spawn(_for_chunk(items, body, start, stop))) }
    for handle in handles { handle.join() }
}

#: Combine every element of `items` in parallel, starting from `identity`.
#:
#: Each worker folds one contiguous chunk starting from `identity`, and the
#: chunk results are then combined left to right. That is only equal to the
#: sequential fold when **`combine` is associative and `identity` is its
#: identity element** — `(0, +)`, `(1, *)`, `(-inf, max)`, `("", concat)`.
#: Subtraction and float addition are not associative; the first is wrong here,
#: the second only differs in the last few bits.
#:
#: ```
#: parallel_reduce([1, 2, 3, 4], 0, |a, b| a + b)    # 10
#: parallel_reduce([1, 2, 3, 4], 1, |a, b| a * b)    # 24
#: ```
pub fn parallel_reduce(
    items: list[any],
    identity: any,
    combine: fn(any, any) -> any,
    workers: int = 0
) -> any {
    let parts = _split(items.len(), _worker_count(workers, items.len()))
    if parts.len() <= 1 {
        var total = identity
        for item in items { total = combine(total, item) }
        return total
    }
    let handles: list[Handle] = []
    for (start, stop) in parts {
        handles.push(spawn(_reduce_chunk(items, identity, combine, start, stop)))
    }
    var total = identity
    for handle in handles { total = combine(total, handle.join()) }
    return total
}

class _Outcome {
    let value: any
    let error: Error?

    pub fn init(self, value: any, error: Error?) {
        self.value = value
        self.error = error
    }

    pub fn unwrap(self) -> any {
        let error = self.error
        if error is not null { throw error }
        return self.value
    }
}

class _Job {
    let work: fn() -> any
    let slot: Channel

    pub fn init(self, work: fn() -> any, slot: Channel) {
        self.work = work
        self.slot = slot
    }

    pub fn run(self) -> void {
        try {
            self.slot.send(_Outcome(self.work(), null))
        } catch error: Error {
            self.slot.send(_Outcome(null, error))
        }
    }
}

class _ChannelIter: Iterator[any], Iterable[any] {
    let channel: Channel

    pub fn init(self, channel: Channel) { self.channel = channel }

    pub fn next(self) -> any { return self.channel.recv() }

    pub fn iter(self) -> Iterator[any] { return self }
}

fn _guarded(body: fn() -> any) -> fn() -> _Outcome {
    return fn() -> _Outcome {
        try {
            return _Outcome(body(), null)
        } catch error: Error {
            return _Outcome(null, error)
        }
    }
}

fn _worker(jobs: Channel) -> fn() -> any {
    return fn() -> any {
        for job in jobs {
            job.run()
        }
        return null
    }
}

fn _map_chunk(items: list[any], transform: fn(any) -> any, start: int, stop: int) -> fn() -> any {
    return fn() -> any {
        let part: list[any] = []
        for index in start..stop {
            part.push(transform(items[index]))
        }
        return part
    }
}

fn _for_chunk(items: list[any], body: fn(any) -> void, start: int, stop: int) -> fn() -> any {
    return fn() -> any {
        for index in start..stop {
            body(items[index])
        }
        return null
    }
}

fn _reduce_chunk(
    items: list[any],
    identity: any,
    combine: fn(any, any) -> any,
    start: int,
    stop: int
) -> fn() -> any {
    return fn() -> any {
        var total = identity
        for index in start..stop {
            total = combine(total, items[index])
        }
        return total
    }
}

fn _worker_count(requested: int, total: int) -> int {
    if requested < 0 { throw ValueError(f"workers must be non-negative, got {requested}") }
    if total < MIN_PARALLEL_ITEMS { return 1 }
    let wanted = if requested > 0 { requested } else { cpu_count() }
    return min(wanted, total)
}

fn _split(count: int, parts: int) -> list[tuple[int, int]] {
    let bounds: list[tuple[int, int]] = []
    if count == 0 or parts < 1 { return bounds }
    let base = count // parts
    let extra = count % parts
    var start = 0
    for index in 0..parts {
        let size = base + (if index < extra { 1 } else { 0 })
        if size > 0 {
            bounds.push((start, start + size))
            start += size
        }
    }
    return bounds
}

fn _back_off(spins: int) -> int {
    if spins < _SPIN_LIMIT { return spins + 1 }
    let doublings = min(spins - _SPIN_LIMIT, 7)
    __prim__.sleep(min(_MIN_BACKOFF * (1 << doublings), _MAX_BACKOFF))
    return spins + 1
}
