#: Sorting algorithms: comparison sorts, integer-key sorts, topological order.
#:
#: Every function here returns a **new** list and never reorders its argument.
#: A comparator is `fn(a, b) -> int`: negative when `a` must precede `b`, zero
#: when the two are interchangeable, positive otherwise. Where a comparator is
#: optional it defaults to `natural_order`, which uses `<`.
#:
#: Which one to reach for:
#:
#:   sort_by / tim_sort   The default. Stable, O(n log n) worst case, and O(n)
#:                        when the input already consists of sorted runs.
#:   merge_sort           Stable, O(n log n) worst case, O(n) extra memory.
#:                        Predictable; the right choice for adversarial input.
#:   quick_sort           Unstable, O(1) extra memory, the fastest inner loop on
#:                        shuffled data. Depth-limited, so still O(n log n).
#:   heap_sort            Unstable, O(n log n) worst case, O(1) extra memory,
#:                        no recursion. Slower in practice: it thrashes cache.
#:   insertion_sort       Stable, O(n^2), but unbeatable below ~16 elements and
#:                        O(n) when the input is nearly sorted.
#:   counting_sort        Stable, O(n + k) for integer keys spanning a range k.
#:   radix_sort           Stable, O(8n) for arbitrary 64-bit integer keys.
#:   bucket_sort          Stable, O(n) expected for floats spread evenly.
#:   topological_sort     O((V + E) log V) when order is set by dependencies.
#:
#:   sort_by_key          Any of the above, with the key computed once.
#:   partial_sort         O(n log k) for the k smallest, instead of O(n log n).
#:   nth_element          O(n) expected for one order statistic.
const INSERTION_CUTOFF = 16
const MIN_MERGE = 64
const MAX_COUNTING_RANGE = 1_048_576
const RADIX_PASSES = 8
const RADIX_BUCKETS = 256

#: A three-way comparison: negative when `a` precedes `b`, zero when the two
#: are interchangeable, positive otherwise.
pub type Comparator[T] = fn(T, T) -> int

#: Compare two values with `<`, returning -1, 0, or 1.
#:
#: This is the default comparator. Build others from it: `|a, b|
#: natural_order(b, a)` sorts descending, `|a, b| natural_order(a.len(),
#: b.len())` sorts by length.
#:
#: ```
#: natural_order(1, 2)     # -1
#: natural_order("b", "a") # 1
#: ```
pub fn natural_order(a: any, b: any) -> int {
    if a < b { return -1 }
    if b < a { return 1 }
    return 0
}

#: Sort `xs` with an explicit comparator.
#:
#: Delegates to `tim_sort`. Prefer this over naming an algorithm unless you
#: have measured a reason not to.
#:
#: ```
#: sort_by(["bb", "a", "ccc"], |a, b| natural_order(a.len(), b.len()))
#: # ["a", "bb", "ccc"]
#: ```
pub fn sort_by[T](xs: list[T], cmp: Comparator[T]) -> list[T] {
    return tim_sort(xs, cmp)
}

#: Sort `xs` by a key computed exactly once per element.
#:
#: The naive `sort_by(xs, |a, b| natural_order(key(a), key(b)))` evaluates
#: `key` about 2n log n times; this pairs each key with its position, sorts the
#: pairs, then projects back (the Schwartzian transform), running `key`
#: exactly n times — worth it once `key` costs more than a field read.
#:
#: Only keys are compared, so `T` itself need not be ordered, and ties keep
#: their original order.
#:
#: ```
#: sort_by_key(["bob", "Alice"], |s| s.upper())   # ["Alice", "bob"]
#: ```
pub fn sort_by_key[T, K](xs: list[T], key: fn(T) -> K) -> list[T] {
    let decorated: list[tuple[K, int]] = []
    for (position, item) in xs.enumerate() {
        decorated.push((key(item), position))
    }
    let ordered = tim_sort(decorated, |a, b| natural_order(a[0], b[0]))
    let out: list[T] = []
    for pair in ordered {
        out.push(xs[pair[1]])
    }
    return out
}

#: Sort `xs` with a stable top-down merge sort.
#:
#: Ranges of 16 or fewer elements finish with insertion sort; a run whose
#: halves are already in order skips the merge entirely.
#:
#: ```
#: merge_sort([3, 1, 2])   # [1, 2, 3]
#: ```
pub fn merge_sort[T](xs: list[T], cmp: Comparator[T] = natural_order) -> list[T] {
    let out = [x for x in xs]
    if out.len() < 2 { return out }
    let buffer = [x for x in xs]
    _merge_sort_range(out, buffer, 0, out.len(), cmp)
    return out
}

#: Sort `xs` with a median-of-three quicksort.
#:
#: The pivot is the median of the first, middle, and last element, which makes
#: already-sorted input the best case rather than the worst. Recursion
#: descends into the smaller partition only (O(log n) stack), and a depth
#: budget of 2*log2(n) + 8 hands the range to `heap_sort` if partitioning goes
#: badly — so the worst case is O(n log n), not O(n^2).
#:
#: ```
#: quick_sort([3, 1, 2])   # [1, 2, 3]
#: ```
pub fn quick_sort[T](xs: list[T], cmp: Comparator[T] = natural_order) -> list[T] {
    let a = [x for x in xs]
    if a.len() < 2 { return a }
    _quick_range(a, 0, a.len(), cmp, 2 * _floor_log2(a.len()) + 8)
    return a
}

#: Sort `xs` with a binary max-heap.
#:
#: No recursion and O(1) extra memory — why it's the fallback for `quick_sort`
#: and `nth_element`.
#:
#: ```
#: heap_sort([3, 1, 2])    # [1, 2, 3]
#: ```
pub fn heap_sort[T](xs: list[T], cmp: Comparator[T] = natural_order) -> list[T] {
    let a = [x for x in xs]
    _heap_sort_range(a, 0, a.len(), cmp)
    return a
}

#: Sort `xs` by repeatedly inserting into a sorted prefix.
#:
#: The fastest option below roughly 16 elements — it touches only adjacent
#: memory. `merge_sort`, `quick_sort`, and `tim_sort` all finish on it.
#:
#: ```
#: insertion_sort([3, 1, 2])   # [1, 2, 3]
#: ```
pub fn insertion_sort[T](xs: list[T], cmp: Comparator[T] = natural_order) -> list[T] {
    let a = [x for x in xs]
    _insertion_range(a, 0, a.len(), cmp)
    return a
}

#: Sort `xs` with TimSort: natural run detection plus balanced merging.
#:
#: The list is scanned for maximal ascending or *strictly* descending runs —
#: strictness is what keeps reversing them in place stable. Short runs are
#: grown by binary insertion, and runs are merged under the invariants
#: `len[i-2] > len[i-1] + len[i]` and `len[i-1] > len[i]`, keeping every merge
#: between runs of similar size. Galloping mode is not implemented, so two
#: runs that interleave in long blocks cost O(n) rather than O(log n) per block.
#:
#: ```
#: tim_sort([1, 2, 3, 0])   # [0, 1, 2, 3]
#: ```
pub fn tim_sort[T](xs: list[T], cmp: Comparator[T] = natural_order) -> list[T] {
    let a = [x for x in xs]
    let n = a.len()
    if n < MIN_MERGE {
        _insertion_range(a, 0, n, cmp)
        return a
    }
    let buffer = [x for x in xs]
    let min_run = _min_run_length(n)
    let starts: list[int] = []
    let lengths: list[int] = []
    var position = 0
    while position < n {
        var length = _count_run(a, position, n, cmp)
        if length < min_run {
            let bounded = min(min_run, n - position)
            _binary_insertion(a, position, position + bounded, position + length, cmp)
            length = bounded
        }
        starts.push(position)
        lengths.push(length)
        _merge_collapse(a, buffer, starts, lengths, cmp)
        position += length
    }
    while lengths.len() > 1 {
        _merge_runs_at(a, buffer, starts, lengths, lengths.len() - 2, cmp)
    }
    return a
}

#: Sort `items` by an integer key using a count of each distinct key.
#:
#: `k` is the span between the smallest and largest key; not a comparison
#: sort, so it beats the O(n log n) bound when k is comparable to n. Raises
#: `ValueError` past `MAX_COUNTING_RANGE`, where `radix_sort` is the answer instead.
#:
#: ```
#: counting_sort([3, 1, 2, 1])                            # [1, 1, 2, 3]
#: counting_sort(["ccc", "a", "bb"], |s| s.len())         # ["a", "bb", "ccc"]
#: ```
pub fn counting_sort[T](items: list[T], key: fn(T) -> int = _int_key) -> list[T] {
    let n = items.len()
    if n == 0 { return [] }
    let keys: list[int] = []
    for item in items {
        keys.push(key(item))
    }
    var lowest = keys[0]
    var highest = keys[0]
    for value in keys {
        if value < lowest { lowest = value }
        if value > highest { highest = value }
    }
    let span = highest - lowest + 1
    if span > MAX_COUNTING_RANGE {
        throw ValueError(
            f"counting_sort key span must be at most {MAX_COUNTING_RANGE}, got {span}; " +
                "use radix_sort for wide integer keys"
        )
    }
    let counts = [0 for _ in 0..span]
    for value in keys {
        counts[value - lowest] += 1
    }
    var running = 0
    for offset in 0..span {
        let occurrences = counts[offset]
        counts[offset] = running
        running += occurrences
    }
    let out = [items[0] for _ in 0..n]
    for i in 0..n {
        let bucket = keys[i] - lowest
        out[counts[bucket]] = items[i]
        counts[bucket] += 1
    }
    return out
}

#: Sort `items` by an integer key, one byte at a time, least significant first.
#:
#: Each pass is a stable counting sort on one byte; a pass whose byte is
#: identical across the whole input is skipped, so small magnitudes cost two
#: or three passes rather than eight. The most significant byte has its sign
#: bit flipped, which orders negatives correctly.
#:
#: Prefer it over `counting_sort` when keys are sparse, and over comparison
#: sorts when n is large (roughly n > 1000) and the key really is an integer.
#:
#: ```
#: radix_sort([170, -45, 75, 90])   # [-45, 75, 90, 170]
#: ```
pub fn radix_sort[T](items: list[T], key: fn(T) -> int = _int_key) -> list[T] {
    let n = items.len()
    var values = [item for item in items]
    if n < 2 { return values }
    var keys: list[int] = []
    for item in items {
        keys.push(key(item))
    }
    var value_buffer = [item for item in items]
    var key_buffer = [k for k in keys]
    for pass in 0..RADIX_PASSES {
        let counts = [0 for _ in 0..RADIX_BUCKETS]
        for value in keys {
            counts[_key_byte(value, pass)] += 1
        }
        if counts[_key_byte(keys[0], pass)] == n { continue }
        var running = 0
        for bucket in 0..RADIX_BUCKETS {
            let occurrences = counts[bucket]
            counts[bucket] = running
            running += occurrences
        }
        for i in 0..n {
            let bucket = _key_byte(keys[i], pass)
            let target = counts[bucket]
            counts[bucket] = target + 1
            value_buffer[target] = values[i]
            key_buffer[target] = keys[i]
        }
        let swapped_values = values
        values = value_buffer
        value_buffer = swapped_values
        let swapped_keys = keys
        keys = key_buffer
        key_buffer = swapped_keys
    }
    return values
}

#: Sort floats by scattering them into evenly spaced buckets.
#:
#: O(n) expected when values are spread uniformly; degrades to O(n^2) when
#: skewed enough to pile everything into one bucket — use `tim_sort` for
#: unknown distributions.
#:
#: Raises `ValueError` on NaN or an infinity, neither of which can be placed
#: in a bucket.
#:
#: ```
#: bucket_sort([0.42, 0.13, 0.71])   # [0.13, 0.42, 0.71]
#: ```
pub fn bucket_sort(xs: list[float], bucket_count: int = 0) -> list[float] {
    if bucket_count < 0 {
        throw ValueError(f"bucket_count must be non-negative, got {bucket_count}")
    }
    let n = xs.len()
    let copy = [x for x in xs]
    if n < 2 { return copy }
    var lowest = xs[0]
    var highest = xs[0]
    for x in xs {
        # `x - x` is NaN for both NaN and the infinities, and 0.0 otherwise.
        if x - x != 0.0 {
            throw ValueError(f"bucket_sort requires finite values, got {x}")
        }
        if x < lowest { lowest = x }
        if x > highest { highest = x }
    }
    if highest == lowest { return copy }
    let count = bucket_count > 0 ? bucket_count : n
    let buckets: list[list[float]] = [[] for _ in 0..count]
    let scale = count / (highest - lowest)
    for x in xs {
        var index = int((x - lowest) * scale)
        if index >= count { index = count - 1 }
        buckets[index].push(x)
    }
    let out: list[float] = []
    for bucket in buckets {
        _insertion_range(bucket, 0, bucket.len(), natural_order)
        for value in bucket {
            out.push(value)
        }
    }
    return out
}

#: Order `items` so that every item follows the items it depends on.
#:
#: `key(item)` names an item; `depends_on(item)` lists the keys that must come
#: first. Kahn's algorithm over a min-heap of ready items.
#:
#: Among all valid orderings this returns the one closest to the input: the
#: ready item with the smallest input position wins at every step, so an
#: already-ordered input comes back unchanged — useful for a build order or
#: module initialisation order.
#:
#: Raises `ValueError` on a duplicate key, an unknown dependency, or a cycle.
#:
#: ```
#: let deps = {"deploy": ["test"], "test": ["build"], "build": []}
#: topological_sort(["deploy", "build", "test"], |name| name, |name| deps[name])
#: # ["build", "test", "deploy"]
#: ```
pub fn topological_sort[T, K](
    items: list[T],
    key: fn(T) -> K,
    depends_on: fn(T) -> list[K]
) -> list[T] {
    let n = items.len()
    let position_of: dict[K, int] = {}
    for (position, item) in items.enumerate() {
        let name = key(item)
        if name in position_of {
            let first = position_of[name]
            throw ValueError(f"duplicate key {name} at positions {first} and {position}")
        }
        position_of[name] = position
    }
    let dependents: list[list[int]] = [[] for _ in 0..n]
    let pending = [0 for _ in 0..n]
    for (position, item) in items.enumerate() {
        for dependency in depends_on(item) {
            if dependency not in position_of {
                throw ValueError(f"unknown dependency {dependency} required by {key(item)}")
            }
            dependents[position_of[dependency]].push(position)
            pending[position] += 1
        }
    }
    let ready: list[int] = []
    for position in 0..n {
        if pending[position] == 0 { _heap_push_min(ready, position) }
    }
    let out: list[T] = []
    while ready.len() > 0 {
        let position = _heap_pop_min(ready)
        out.push(items[position])
        for dependent in dependents[position] {
            pending[dependent] -= 1
            if pending[dependent] == 0 { _heap_push_min(ready, dependent) }
        }
    }
    if out.len() < n {
        let stuck: list[K] = []
        for position in 0..n {
            if pending[position] > 0 { stuck.push(key(items[position])) }
        }
        throw ValueError(f"dependency cycle among {stuck}")
    }
    return out
}

#: Return the `count` smallest elements of `xs`, in ascending order.
#:
#: Keeps a max-heap of the best `count` elements seen so far; each later
#: element either loses to the root in one comparison or replaces it in
#: O(log count) — much cheaper than sorting everything and slicing. Not stable.
#:
#: ```
#: partial_sort([5, 1, 4, 2, 3], 3)   # [1, 2, 3]
#: ```
pub fn partial_sort[T](xs: list[T], count: int, cmp: Comparator[T] = natural_order) -> list[T] {
    if count < 0 { throw ValueError(f"count must be non-negative, got {count}") }
    let n = xs.len()
    let k = min(count, n)
    if k == 0 { return [] }
    let heap: list[T] = []
    for offset in 0..k {
        heap.push(xs[offset])
    }
    var start = k // 2 - 1
    while start >= 0 {
        _sift_down(heap, 0, start, k, cmp)
        start -= 1
    }
    var position = k
    while position < n {
        if cmp(xs[position], heap[0]) < 0 {
            heap[0] = xs[position]
            _sift_down(heap, 0, 0, k, cmp)
        }
        position += 1
    }
    var last = k - 1
    while last > 0 {
        _swap(heap, 0, last)
        _sift_down(heap, 0, 0, last, cmp)
        last -= 1
    }
    return heap
}

#: Return the element that would sit at index `n` if `xs` were sorted.
#:
#: Quickselect: partition as in `quick_sort`, then recurse into the one side
#: that can contain index `n` — O(n) expected, since the work halves each
#: round. A depth budget hands the range to `heap_sort` on bad partitioning,
#: so this is introselect: worst case O(n log n), not O(n^2).
#:
#: ```
#: nth_element([5, 1, 4, 2, 3], 2)   # 3
#: ```
pub fn nth_element[T](xs: list[T], n: int, cmp: Comparator[T] = natural_order) -> T {
    let count = xs.len()
    if n < 0 or n >= count {
        throw IndexError(f"nth_element index must be in 0..{count}, got {n}")
    }
    let a = [x for x in xs]
    var lo = 0
    var hi = count
    var budget = 2 * _floor_log2(count) + 8
    loop {
        if hi - lo <= INSERTION_CUTOFF {
            _insertion_range(a, lo, hi, cmp)
            break
        }
        if budget == 0 {
            _heap_sort_range(a, lo, hi, cmp)
            break
        }
        budget -= 1
        let pivot = _partition(a, lo, hi, cmp)
        if pivot == n { return a[pivot] }
        if n < pivot { hi = pivot } else { lo = pivot + 1 }
    }
    return a[n]
}

#: Report whether `xs` is in non-descending order.
#:
#: Worth calling before a sort only when the answer is usually yes; `tim_sort`
#: already detects sorted input in one pass.
#:
#: ```
#: is_sorted([1, 2, 2, 3])   # true
#: is_sorted([2, 1])         # false
#: ```
pub fn is_sorted[T](xs: list[T], cmp: Comparator[T] = natural_order) -> bool {
    var position = 1
    while position < xs.len() {
        if cmp(xs[position], xs[position - 1]) < 0 { return false }
        position += 1
    }
    return true
}

#: Merge two sorted lists into one sorted list.
#:
#: Ties resolve in favour of `a`, so merging a list with a later batch keeps
#: the earlier batch first.
#:
#: ```
#: merge([1, 4], [2, 3])   # [1, 2, 3, 4]
#: ```
pub fn merge[T](a: list[T], b: list[T], cmp: Comparator[T] = natural_order) -> list[T] {
    let out: list[T] = []
    var i = 0
    var j = 0
    while i < a.len() and j < b.len() {
        if cmp(b[j], a[i]) < 0 {
            out.push(b[j])
            j += 1
        } else {
            out.push(a[i])
            i += 1
        }
    }
    while i < a.len() {
        out.push(a[i])
        i += 1
    }
    while j < b.len() {
        out.push(b[j])
        j += 1
    }
    return out
}

fn _int_key(value: int) -> int {
    return value
}

fn _swap[T](a: list[T], i: int, j: int) -> void {
    let held = a[i]
    a[i] = a[j]
    a[j] = held
}

fn _floor_log2(n: int) -> int {
    var remaining = n
    var result = 0
    while remaining > 1 {
        remaining = remaining >> 1
        result += 1
    }
    return result
}

fn _key_byte(key: int, pass: int) -> int {
    if pass == RADIX_PASSES - 1 { return key >> 56 & 0xFF ^ 0x80 }
    return key >> 8 * pass & 0xFF
}

fn _insertion_range[T](a: list[T], lo: int, hi: int, cmp: Comparator[T]) -> void {
    var position = lo + 1
    while position < hi {
        let value = a[position]
        var hole = position - 1
        while hole >= lo and cmp(value, a[hole]) < 0 {
            a[hole + 1] = a[hole]
            hole -= 1
        }
        a[hole + 1] = value
        position += 1
    }
}

#: Merge the sorted ranges `a[lo:mid)` and `a[mid:hi)` in place, stably.
#:
#: Only the left half is copied out, so the scratch buffer never needs more
#: than half the array. Taking from the left on ties is what preserves order.
fn _merge_adjacent[T](
    a: list[T],
    buffer: list[T],
    lo: int,
    mid: int,
    hi: int,
    cmp: Comparator[T]
) -> void {
    if mid == lo or mid == hi { return }
    if cmp(a[mid], a[mid - 1]) >= 0 { return }
    let left_length = mid - lo
    for offset in 0..left_length {
        buffer[offset] = a[lo + offset]
    }
    var i = 0
    var j = mid
    var write = lo
    while i < left_length and j < hi {
        if cmp(a[j], buffer[i]) < 0 {
            a[write] = a[j]
            j += 1
        } else {
            a[write] = buffer[i]
            i += 1
        }
        write += 1
    }
    while i < left_length {
        a[write] = buffer[i]
        i += 1
        write += 1
    }
}

fn _merge_sort_range[T](a: list[T], buffer: list[T], lo: int, hi: int, cmp: Comparator[T]) -> void {
    if hi - lo <= INSERTION_CUTOFF {
        _insertion_range(a, lo, hi, cmp)
        return
    }
    let mid = lo + (hi - lo) // 2
    _merge_sort_range(a, buffer, lo, mid, cmp)
    _merge_sort_range(a, buffer, mid, hi, cmp)
    _merge_adjacent(a, buffer, lo, mid, hi, cmp)
}

fn _sift_down[T](a: list[T], lo: int, start: int, limit: int, cmp: Comparator[T]) -> void {
    var root = start
    loop {
        let child = lo + 2 * (root - lo) + 1
        if child >= limit { break }
        var larger = child
        if child + 1 < limit and cmp(a[child], a[child + 1]) < 0 { larger = child + 1 }
        if cmp(a[root], a[larger]) >= 0 { break }
        _swap(a, root, larger)
        root = larger
    }
}

fn _heap_sort_range[T](a: list[T], lo: int, hi: int, cmp: Comparator[T]) -> void {
    var start = lo + (hi - lo) // 2 - 1
    while start >= lo {
        _sift_down(a, lo, start, hi, cmp)
        start -= 1
    }
    var last = hi - 1
    while last > lo {
        _swap(a, lo, last)
        _sift_down(a, lo, lo, last, cmp)
        last -= 1
    }
}

#: Partition `a[lo:hi)` around the median of its first, middle, and last
#: element, and return the final index of that pivot.
#:
#: Requires at least three elements. After ordering the three samples, the
#: median is parked at `hi - 2`, which leaves `a[lo] <= pivot` and
#: `a[hi - 1] >= pivot` as sentinels: neither scan can run off the range, so
#: the inner loops carry no bounds check.
fn _partition[T](a: list[T], lo: int, hi: int, cmp: Comparator[T]) -> int {
    let last = hi - 1
    let mid = lo + (hi - lo) // 2
    if cmp(a[mid], a[lo]) < 0 { _swap(a, lo, mid) }
    if cmp(a[last], a[lo]) < 0 { _swap(a, lo, last) }
    if cmp(a[last], a[mid]) < 0 { _swap(a, mid, last) }
    _swap(a, mid, last - 1)
    let pivot = a[last - 1]
    var i = lo
    var j = last - 1
    loop {
        i += 1
        while cmp(a[i], pivot) < 0 { i += 1 }
        j -= 1
        while cmp(a[j], pivot) > 0 { j -= 1 }
        if i >= j { break }
        _swap(a, i, j)
    }
    _swap(a, i, last - 1)
    return i
}

fn _quick_range[T](
    a: list[T],
    from_index: int,
    to_index: int,
    cmp: Comparator[T],
    depth_limit: int
) -> void {
    var lo = from_index
    var hi = to_index
    var depth = depth_limit
    while hi - lo > INSERTION_CUTOFF {
        if depth == 0 {
            _heap_sort_range(a, lo, hi, cmp)
            return
        }
        depth -= 1
        let pivot = _partition(a, lo, hi, cmp)
        if pivot - lo < hi - pivot - 1 {
            _quick_range(a, lo, pivot, cmp, depth)
            lo = pivot + 1
        } else {
            _quick_range(a, pivot + 1, hi, cmp, depth)
            hi = pivot
        }
    }
    _insertion_range(a, lo, hi, cmp)
}

#: Choose TimSort's minimum run length: n / 2^k rounded up, landing between
#: `MIN_MERGE / 2` and `MIN_MERGE`, so that the number of runs is close to a
#: power of two and the merge tree stays balanced. Dropped low bits are folded
#: back in as a carry, because rounding down would leave a short final run and
#: an unbalanced last merge.
fn _min_run_length(n: int) -> int {
    var remaining = n
    var carry = 0
    while remaining >= MIN_MERGE {
        carry = carry | remaining & 1
        remaining = remaining >> 1
    }
    return remaining + carry
}

#: Measure the run starting at `lo`, reversing it if it descends.
fn _count_run[T](a: list[T], lo: int, hi: int, cmp: Comparator[T]) -> int {
    if lo + 1 >= hi { return hi - lo }
    var run_end = lo + 1
    if cmp(a[run_end], a[lo]) < 0 {
        while run_end + 1 < hi and cmp(a[run_end + 1], a[run_end]) < 0 { run_end += 1 }
        _reverse_range(a, lo, run_end + 1)
    } else {
        while run_end + 1 < hi and cmp(a[run_end + 1], a[run_end]) >= 0 { run_end += 1 }
    }
    return run_end - lo + 1
}

fn _reverse_range[T](a: list[T], lo: int, hi: int) -> void {
    var left = lo
    var right = hi - 1
    while left < right {
        _swap(a, left, right)
        left += 1
        right -= 1
    }
}

#: Sort `a[lo:hi)` given that `a[lo:start)` is already sorted, placing each
#: remaining element with a binary search. Halves the comparisons of a plain
#: insertion sort; the moves are unchanged.
fn _binary_insertion[T](a: list[T], lo: int, hi: int, start: int, cmp: Comparator[T]) -> void {
    var position = start
    if position == lo { position += 1 }
    while position < hi {
        let value = a[position]
        var left = lo
        var right = position
        while left < right {
            let mid = left + (right - left) // 2
            if cmp(value, a[mid]) < 0 { right = mid } else { left = mid + 1 }
        }
        var hole = position
        while hole > left {
            a[hole] = a[hole - 1]
            hole -= 1
        }
        a[left] = value
        position += 1
    }
}

#: Merge runs `index` and `index + 1` on the pending-run stack.
fn _merge_runs_at[T](
    a: list[T],
    buffer: list[T],
    starts: list[int],
    lengths: list[int],
    index: int,
    cmp: Comparator[T]
) -> void {
    let lo = starts[index]
    let mid = starts[index + 1]
    _merge_adjacent(a, buffer, lo, mid, mid + lengths[index + 1], cmp)
    lengths[index] = lengths[index] + lengths[index + 1]
    let top = lengths.len() - 1
    if index + 1 < top {
        starts[index + 1] = starts[top]
        lengths[index + 1] = lengths[top]
    }
    starts.pop()
    lengths.pop()
}

#: Merge pending runs until the stack satisfies TimSort's two invariants.
#:
#: `lengths[i-2] > lengths[i-1] + lengths[i]` and `lengths[i-1] > lengths[i]`
#: keep the run lengths growing at least as fast as Fibonacci, bounding the
#: stack. The `index > 1` test corrects the originally published algorithm,
#: which could leave the deeper invariant violated.
fn _merge_collapse[T](
    a: list[T],
    buffer: list[T],
    starts: list[int],
    lengths: list[int],
    cmp: Comparator[T]
) -> void {
    while lengths.len() > 1 {
        var index = lengths.len() - 2
        var must_merge = lengths[index] <= lengths[index + 1]
        if index > 0 and lengths[index - 1] <= lengths[index] + lengths[index + 1] {
            must_merge = true
        }
        if index > 1 and lengths[index - 2] <= lengths[index - 1] + lengths[index] {
            must_merge = true
        }
        if not must_merge { return }
        if index > 0 and lengths[index - 1] < lengths[index + 1] { index -= 1 }
        _merge_runs_at(a, buffer, starts, lengths, index, cmp)
    }
}

fn _heap_push_min(heap: list[int], value: int) -> void {
    heap.push(value)
    var child = heap.len() - 1
    while child > 0 {
        let parent = (child - 1) // 2
        if heap[parent] <= heap[child] { break }
        _swap(heap, parent, child)
        child = parent
    }
}

fn _heap_pop_min(heap: list[int]) -> int {
    let smallest = heap[0]
    let moved = heap.pop()
    if heap.len() == 0 { return smallest }
    heap[0] = moved
    var parent = 0
    loop {
        let left = 2 * parent + 1
        if left >= heap.len() { break }
        var child = left
        if left + 1 < heap.len() and heap[left + 1] < heap[left] { child = left + 1 }
        if heap[parent] <= heap[child] { break }
        _swap(heap, parent, child)
        parent = child
    }
    return smallest
}
