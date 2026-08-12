#: Algorithms over lists that the built-in list methods do not cover.
#:
#: Functions needing an ordering or an equality take one as an optional last
#: argument, falling back to `<` and `==`.
const _INT_MAX = 0x7FFFFFFFFFFFFFFF
const _UNIFORM_STEPS = 9007199254740992
const _UNIFORM_SCALE = 9007199254740992.0

#: A one-argument test used to select elements.
pub type Predicate[T] = fn(T) -> bool

#: A projection from an element to the value it should be grouped or ranked by.
pub type KeyFn[T] = fn(T) -> any

#: A strict ordering: `less(a, b)` is true when `a` sorts before `b`.
pub type Less[T] = fn(T, T) -> bool

#: An equivalence: `same(a, b)` is true when the two elements count as equal.
pub type Same[T] = fn(T, T) -> bool

#: A source of randomness: `rng(bound)` returns an integer in `[0, bound)`.
#:
#: Passing one makes `shuffled`, `sample` and `weighted_choice` deterministic.
pub type Rng = fn(int) -> int

#: Split `xs` into the elements satisfying `pred` and the elements that do not.
#:
#: ```
#: partition([1, 2, 3, 4], |n| n % 2 == 0)    # ([2, 4], [1, 3])
#: ```
pub fn partition[T](xs: list[T], pred: Predicate[T]) -> tuple[list[T], list[T]] {
    let matching: list[T] = []
    let rest: list[T] = []
    for x in xs {
        if pred(x) { matching.push(x) } else { rest.push(x) }
    }
    return (matching, rest)
}

#: Group the elements of `xs` into a dict keyed by `key(element)`.
#:
#: Each group keeps the relative order its elements had in `xs`.
#:
#: ```
#: group_by([1, 2, 3, 4], |n| n % 2) == {1: [1, 3], 0: [2, 4]}    # true
#: ```
pub fn group_by[T](xs: list[T], key: KeyFn[T]) -> dict[any, list[T]] {
    let groups: dict[any, list[T]] = {}
    for x in xs {
        let group_key = key(x)
        if group_key not in groups { groups[group_key] = [] }
        groups[group_key].push(x)
    }
    return groups
}

#: Cut `xs` into runs of adjacent elements for which `same(previous, current)`.
#:
#: ```
#: chunk_by([1, 2, 4, 5, 7], |a, b| b - a == 1)    # [[1, 2], [4, 5], [7]]
#: ```
pub fn chunk_by[T](xs: list[T], same: Same[T]) -> list[list[T]] {
    if xs.len() == 0 { return [] }
    let chunks: list[list[T]] = []
    var current: list[T] = [xs[0]]
    for index in 1..xs.len() {
        if same(current[current.len() - 1], xs[index]) {
            current.push(xs[index])
        } else {
            chunks.push(current)
            current = [xs[index]]
        }
    }
    chunks.push(current)
    return chunks
}

#: Split `xs` into its first `index` elements and the rest.
pub fn split_at[T](xs: list[T], index: int) -> tuple[list[T], list[T]] {
    if index < 0 or index > xs.len() {
        throw IndexError(f"split index must be in 0..={xs.len()}, got {index}")
    }
    return (xs[0:index], xs[index:xs.len()])
}

#: Split `xs` into segments at every element satisfying `pred`, dropping it.
#:
#: A separator at either end produces an empty segment, so the result always
#: holds one more list than there were separators.
#:
#: ```
#: split_when([1, 0, 2, 3, 0], |n| n == 0)    # [[1], [2, 3], []]
#: ```
pub fn split_when[T](xs: list[T], pred: Predicate[T]) -> list[list[T]] {
    let parts: list[list[T]] = []
    var current: list[T] = []
    for x in xs {
        if pred(x) {
            parts.push(current)
            current = []
        } else {
            current.push(x)
        }
    }
    parts.push(current)
    return parts
}

#: Rotate `xs` left by `n` places; a negative `n` rotates right.
#:
#: ```
#: rotate([1, 2, 3, 4, 5], 2)    # [3, 4, 5, 1, 2]
#: ```
pub fn rotate[T](xs: list[T], n: int) -> list[T] {
    let count = xs.len()
    if count == 0 { return [] }
    # Two `%` so the offset is non-negative whatever sign `%` gives.
    let shift = (n % count + count) % count
    return xs[shift:count] + xs[0:shift]
}

#: Take one element from each list in turn until every list is exhausted.
#:
#: ```
#: interleave([[1, 2, 3], [10, 20]])    # [1, 10, 2, 20, 3]
#: ```
pub fn interleave[T](lists: list[list[T]]) -> list[T] {
    var longest = 0
    for part in lists { longest = max(longest, part.len()) }
    let result: list[T] = []
    for index in 0..longest {
        for part in lists {
            if index < part.len() { result.push(part[index]) }
        }
    }
    return result
}

#: Exchange the rows and columns of a rectangular list of lists.
pub fn transpose[T](rows: list[list[T]]) -> list[list[T]] {
    if rows.len() == 0 { return [] }
    let width = rows[0].len()
    for row in rows {
        if row.len() != width {
            throw ValueError(f"ragged rows: expected width {width}, got {row.len()}")
        }
    }
    return [[row[column] for row in rows] for column in 0..width]
}

#: Every way of picking one element from each of `lists`, last index varying fastest.
#:
#: An empty outer list yields `[[]]`; any empty member yields `[]`.
#:
#: ```
#: cartesian_product([[1, 2], [3]])    # [[1, 3], [2, 3]]
#: ```
pub fn cartesian_product[T](lists: list[list[T]]) -> list[list[T]] {
    var result: list[list[T]] = [[]]
    for part in lists {
        let next: list[list[T]] = []
        for prefix in result {
            for item in part { next.push(prefix + [item]) }
        }
        result = next
    }
    return result
}

#: Every ordered arrangement of `k` elements of `xs`, in index order.
#:
#: `k` defaults to the length of `xs`. Elements are distinguished by position,
#: so a list with repeats yields repeated arrangements.
#:
#: ```
#: permutations([1, 2, 3], 2)    # [[1, 2], [1, 3], [2, 1], [2, 3], [3, 1], [3, 2]]
#: ```
pub fn permutations[T](xs: list[T], k: int? = null) -> list[list[T]] {
    let size = k ?? xs.len()
    if size < 0 { throw ValueError(f"k must be non-negative, got {size}") }
    if size > xs.len() { return [] }
    let used = [false for _slot in 0..xs.len()]
    let result: list[list[T]] = []
    _permute(xs, size, used, [], result)
    return result
}

#: Every unordered selection of `k` elements of `xs`, in index order.
#:
#: ```
#: combinations([1, 2, 3], 2)    # [[1, 2], [1, 3], [2, 3]]
#: ```
pub fn combinations[T](xs: list[T], k: int) -> list[list[T]] {
    if k < 0 { throw ValueError(f"k must be non-negative, got {k}") }
    if k > xs.len() { return [] }
    let result: list[list[T]] = []
    _choose(xs, k, 0, [], result)
    return result
}

#: Every subset of `xs`, ordered by the bit pattern of the elements taken.
#:
#: ```
#: power_set([1, 2])    # [[], [1], [2], [1, 2]]
#: ```
pub fn power_set[T](xs: list[T]) -> list[list[T]] {
    if xs.len() > 62 {
        throw ValueError(f"power set of {xs.len()} elements does not fit in memory")
    }
    let result: list[list[T]] = []
    for mask in 0..1 << xs.len() {
        result.push([xs[index] for index in 0..xs.len() if mask >> index & 1 == 1])
    }
    return result
}

#: Every contiguous window of `size` elements, advancing by `step` each time.
#:
#: Partial windows at the end are dropped.
#:
#: ```
#: sliding_window([1, 2, 3, 4], 2)    # [[1, 2], [2, 3], [3, 4]]
#: ```
pub fn sliding_window[T](xs: list[T], size: int, step: int = 1) -> list[list[T]] {
    if size <= 0 { throw ValueError(f"window size must be positive, got {size}") }
    if step <= 0 { throw ValueError(f"step must be positive, got {step}") }
    let windows: list[list[T]] = []
    var start = 0
    while start + size <= xs.len() {
        windows.push(xs[start:start + size])
        start += step
    }
    return windows
}

#: Collapse each run of equal adjacent elements into a `(value, count)` pair.
#:
#: ```
#: run_length_encode(["a", "a", "b"])    # [("a", 2), ("b", 1)]
#: ```
pub fn run_length_encode[T](xs: list[T]) -> list[tuple[T, int]] {
    let runs: list[tuple[T, int]] = []
    var index = 0
    while index < xs.len() {
        var count = 1
        while index + count < xs.len() and xs[index + count] == xs[index] {
            count += 1
        }
        runs.push((xs[index], count))
        index += count
    }
    return runs
}

#: Expand `(value, count)` pairs back into the list they encode.
pub fn run_length_decode[T](runs: list[tuple[T, int]]) -> list[T] {
    let out: list[T] = []
    for (value, count) in runs {
        if count < 0 { throw ValueError(f"run length must be non-negative, got {count}") }
        for _repeat in 0..count { out.push(value) }
    }
    return out
}

#: Drop every element equal to the one before it, keeping non-adjacent repeats.
#:
#: ```
#: dedupe_consecutive([1, 1, 2, 2, 1])    # [1, 2, 1]
#: ```
pub fn dedupe_consecutive[T](xs: list[T], same: Same[T]? = null) -> list[T] {
    let matches: Same[T] = same ?? _equals
    let out: list[T] = []
    for x in xs {
        if out.len() == 0 or not matches(out[out.len() - 1], x) { out.push(x) }
    }
    return out
}

#: Count how many times each element occurs in `xs`.
#:
#: ```
#: frequency(["a", "b", "a"]) == {"a": 2, "b": 1}    # true
#: ```
pub fn frequency[T](xs: list[T]) -> dict[T, int] {
    let counts: dict[T, int] = {}
    for x in xs { counts[x] = counts.get(x, 0) + 1 }
    return counts
}

#: The `k` most frequent elements as `(value, count)` pairs, highest count first.
#:
#: Ties keep the order in which the values were first counted; `k` defaults to
#: every distinct value.
#:
#: ```
#: most_common(["a", "b", "a"], 1)    # [("a", 2)]
#: ```
pub fn most_common[T](xs: list[T], k: int? = null) -> list[tuple[T, int]] {
    let pairs = [(value, count) for (value, count) in frequency(xs).items()]
    let ordered = _sort_by(pairs, |a, b| a[1] > b[1])
    let limit = k ?? ordered.len()
    if limit < 0 { throw ValueError(f"k must be non-negative, got {limit}") }
    if limit >= ordered.len() { return ordered }
    return ordered[0:limit]
}

#: The index of the smallest element of `xs`, or of the smallest `key(element)`.
#:
#: The first index wins a tie; raises `ValueError` on an empty list.
#:
#: ```
#: argmin([3, 1, 2])                            # 1
#: argmin(["aaa", "b"], |word| word.len())      # 1
#: ```
pub fn argmin[T](xs: list[T], key: KeyFn[T]? = null) -> int {
    return _arg_extreme(xs, key ?? _identity, _less, "argmin")
}

#: The index of the largest element of `xs`, or of the largest `key(element)`.
#:
#: The first index wins a tie; raises `ValueError` on an empty list.
pub fn argmax[T](xs: list[T], key: KeyFn[T]? = null) -> int {
    return _arg_extreme(xs, key ?? _identity, _greater, "argmax")
}

#: The running totals of `xs`: element `i` is the sum of `xs[0..=i]`.
pub fn cumulative_sum[T](xs: list[T]) -> list[T] {
    if xs.len() == 0 { return [] }
    let out: list[T] = [xs[0]]
    for index in 1..xs.len() { out.push(out[index - 1] + xs[index]) }
    return out
}

#: The gaps between adjacent elements; the inverse of `cumulative_sum`.
pub fn differences[T](xs: list[T]) -> list[T] {
    let out: list[T] = []
    for index in 1..xs.len() { out.push(xs[index] - xs[index - 1]) }
    return out
}

#: Combine `a` and `b` element by element, stopping at the shorter list.
#:
#: ```
#: zip_with([1, 2], [10, 20], |a, b| a * b)    # [10, 40]
#: ```
pub fn zip_with[A, B, C](a: list[A], b: list[B], combine: fn(A, B) -> C) -> list[C] {
    let out: list[C] = []
    for index in 0..min(a.len(), b.len()) { out.push(combine(a[index], b[index])) }
    return out
}

#: Split a list of pairs into a pair of lists; the inverse of `zip`.
pub fn unzip[A, B](pairs: list[tuple[A, B]]) -> tuple[list[A], list[B]] {
    return ([pair[0] for pair in pairs], [pair[1] for pair in pairs])
}

#: Every intermediate result of folding `combine` over `xs`, `initial` first.
#:
#: The result holds one more element than `xs`; its last element is the fold.
#:
#: ```
#: scan([1, 2, 3], 0, |total, n| total + n)    # [0, 1, 3, 6]
#: ```
pub fn scan[T, A](xs: list[T], initial: A, combine: fn(A, T) -> A) -> list[A] {
    let out: list[A] = [initial]
    var accumulator = initial
    for x in xs {
        accumulator = combine(accumulator, x)
        out.push(accumulator)
    }
    return out
}

#: The longest leading run of elements satisfying `pred`.
pub fn take_while[T](xs: list[T], pred: Predicate[T]) -> list[T] {
    let out: list[T] = []
    for x in xs {
        if not pred(x) { break }
        out.push(x)
    }
    return out
}

#: Everything after the longest leading run of elements satisfying `pred`.
pub fn drop_while[T](xs: list[T], pred: Predicate[T]) -> list[T] {
    var start = 0
    while start < xs.len() and pred(xs[start]) { start += 1 }
    return xs[start:xs.len()]
}

#: A copy of the sorted list `xs` with `value` added, still sorted.
#:
#: `value` lands after anything it compares equal to, so repeated insertion is
#: stable. `xs` must already be sorted by `less`.
#:
#: ```
#: insert_sorted([1, 3, 5], 4)    # [1, 3, 4, 5]
#: ```
pub fn insert_sorted[T](xs: list[T], value: T, less: Less[T]? = null) -> list[T] {
    let position = _upper_bound(xs, value, less ?? _less)
    return xs[0:position] + [value] + xs[position:xs.len()]
}

#: Insert `value` into the sorted list `xs` in place and return its index.
#:
#: The position is found in O(log n) comparisons; the shift itself is O(n).
#:
#: ```
#: let xs = [1, 3, 5]
#: binary_insert(xs, 4)    # 2
#: xs                      # [1, 3, 4, 5]
#: ```
pub fn binary_insert[T](xs: list[T], value: T, less: Less[T]? = null) -> int {
    let position = _upper_bound(xs, value, less ?? _less)
    xs.insert(position, value)
    return position
}

#: Report whether `xs` is in non-decreasing order under `less`.
pub fn is_sorted[T](xs: list[T], less: Less[T]? = null) -> bool {
    let ordering: Less[T] = less ?? _less
    for index in 1..xs.len() {
        if ordering(xs[index], xs[index - 1]) { return false }
    }
    return true
}

#: A uniformly shuffled copy of `xs`, by the Fisher-Yates algorithm.
#:
#: ```
#: shuffled([1, 2, 3], |_bound| 0)    # [2, 3, 1]
#: ```
pub fn shuffled[T](xs: list[T], rng: Rng? = null) -> list[T] {
    let sampler: Rng = rng ?? _default_below
    let out = xs[0:xs.len()]
    var index = out.len() - 1
    while index > 0 {
        let target = _below(index + 1, sampler)
        let held = out[index]
        out[index] = out[target]
        out[target] = held
        index -= 1
    }
    return out
}

#: Draw `k` distinct elements of `xs` uniformly at random, without replacement.
#:
#: Runs in O(n) and never reuses a position, so it stays correct for `k` close
#: to the length of `xs`.
#:
#: ```
#: sample([1, 2, 3, 4], 2, |_bound| 0)    # [1, 2]
#: ```
pub fn sample[T](xs: list[T], k: int, rng: Rng? = null) -> list[T] {
    if k < 0 or k > xs.len() {
        throw ValueError(f"cannot sample {k} elements from a list of {xs.len()}")
    }
    let sampler: Rng = rng ?? _default_below
    let pool = xs[0:xs.len()]
    let out: list[T] = []
    for index in 0..k {
        let pick = index + _below(pool.len() - index, sampler)
        let held = pool[index]
        pool[index] = pool[pick]
        pool[pick] = held
        out.push(pool[index])
    }
    return out
}

#: Choose one element of `xs` with probability proportional to its weight.
#:
#: Weights must be non-negative and must not all be zero.
#:
#: ```
#: weighted_choice(["a", "b"], [1.0, 3.0], |_bound| 0)    # "a"
#: ```
pub fn weighted_choice[T](xs: list[T], weights: list[float], rng: Rng? = null) -> T {
    if xs.len() == 0 { throw ValueError("weighted_choice from an empty list") }
    if weights.len() != xs.len() {
        throw ValueError(f"expected {xs.len()} weights, got {weights.len()}")
    }
    var total = 0.0
    for weight in weights {
        if weight < 0.0 { throw ValueError(f"weights must be non-negative, got {weight}") }
        total += weight
    }
    if total <= 0.0 { throw ValueError("weights must not all be zero") }
    let sampler: Rng = rng ?? _default_below
    var target = _below(_UNIFORM_STEPS, sampler) / _UNIFORM_SCALE * total
    for index in 0..xs.len() {
        target -= weights[index]
        if target < 0.0 { return xs[index] }
    }
    # Reachable only when rounding leaves `target` at exactly the total.
    return xs[xs.len() - 1]
}

#: Flatten nested lists of any depth into a single list of leaves.
#:
#: ```
#: flatten_deep([1, [2, [3, 4]]])    # [1, 2, 3, 4]
#: ```
pub fn flatten_deep(xs: list[any]) -> list[any] {
    let out: list[any] = []
    _flatten_into(xs, out)
    return out
}

#: Compare `a` and `b` lexicographically, returning -1, 0 or 1.
#:
#: A prefix sorts before the list that extends it.
#:
#: ```
#: compare([1, 2], [1, 3])    # -1
#: compare([1, 2], [1])       # 1
#: ```
pub fn compare[T](a: list[T], b: list[T], less: Less[T]? = null) -> int {
    let ordering: Less[T] = less ?? _less
    for index in 0..min(a.len(), b.len()) {
        if ordering(a[index], b[index]) { return -1 }
        if ordering(b[index], a[index]) { return 1 }
    }
    if a.len() < b.len() { return -1 }
    if a.len() > b.len() { return 1 }
    return 0
}

#: Report whether `a` and `b` match element by element under `same`.
#:
#: ```
#: equal_by(["A"], ["a"], |x, y| x.lower() == y.lower())    # true
#: ```
pub fn equal_by[T](a: list[T], b: list[T], same: Same[T]) -> bool {
    if a.len() != b.len() { return false }
    for index in 0..a.len() {
        if not same(a[index], b[index]) { return false }
    }
    return true
}

#: Report whether `a` and `b` hold the same elements the same number of times.
#:
#: Runs in O(n) using a count table, so the elements must be hashable.
#:
#: ```
#: is_permutation([1, 2, 2], [2, 1, 2])    # true
#: ```
pub fn is_permutation[T](a: list[T], b: list[T]) -> bool {
    if a.len() != b.len() { return false }
    let counts = frequency(a)
    for value in b {
        let remaining = counts.get(value, 0)
        if remaining == 0 { return false }
        counts[value] = remaining - 1
    }
    return true
}

fn _permute[T](
    xs: list[T],
    size: int,
    used: list[bool],
    current: list[T],
    result: list[list[T]]
) -> void {
    if current.len() == size {
        result.push(current[0:size])
        return
    }
    for index in 0..xs.len() {
        if used[index] { continue }
        used[index] = true
        current.push(xs[index])
        _permute(xs, size, used, current, result)
        current.pop()
        used[index] = false
    }
}

fn _choose[T](xs: list[T], k: int, start: int, current: list[T], result: list[list[T]]) -> void {
    if current.len() == k {
        result.push(current[0:k])
        return
    }
    for index in start..xs.len() {
        # Stop as soon as too few elements remain to finish this combination.
        if xs.len() - index < k - current.len() { break }
        current.push(xs[index])
        _choose(xs, k, index + 1, current, result)
        current.pop()
    }
}

fn _flatten_into(xs: list[any], out: list[any]) -> void {
    for item in xs {
        if isinstance(item, list) { _flatten_into(item, out) } else { out.push(item) }
    }
}

fn _arg_extreme[T](xs: list[T], project: KeyFn[T], better: Less[any], name: str) -> int {
    if xs.len() == 0 { throw ValueError(f"{name} of an empty list") }
    var best = 0
    var best_key = project(xs[0])
    for index in 1..xs.len() {
        let candidate = project(xs[index])
        if better(candidate, best_key) {
            best = index
            best_key = candidate
        }
    }
    return best
}

#: The index one past the last element of sorted `xs` that is not after `value`.
fn _upper_bound[T](xs: list[T], value: T, less: Less[T]) -> int {
    var low = 0
    var high = xs.len()
    while low < high {
        let middle = (low + high) // 2
        if less(value, xs[middle]) { high = middle } else { low = middle + 1 }
    }
    return low
}

#: A stable merge sort, used where an ordering other than `<` is needed.
fn _sort_by[T](xs: list[T], less: Less[T]) -> list[T] {
    if xs.len() <= 1 { return xs[0:xs.len()] }
    let middle = xs.len() // 2
    return _merge(_sort_by(xs[0:middle], less), _sort_by(xs[middle:xs.len()], less), less)
}

fn _merge[T](left: list[T], right: list[T], less: Less[T]) -> list[T] {
    let out: list[T] = []
    var left_index = 0
    var right_index = 0
    while left_index < left.len() and right_index < right.len() {
        # Strict `<` keeps equal elements in original order, which is what makes this stable.
        if less(right[right_index], left[left_index]) {
            out.push(right[right_index])
            right_index += 1
        } else {
            out.push(left[left_index])
            left_index += 1
        }
    }
    return out + left[left_index:left.len()] + right[right_index:right.len()]
}

fn _below(bound: int, sampler: Rng) -> int {
    let value = sampler(bound)
    if value < 0 or value >= bound {
        throw ValueError(f"rng returned {value}, expected a value in [0, {bound})")
    }
    return value
}

fn _default_below(bound: int) -> int {
    if bound <= 0 { throw ValueError(f"bound must be positive, got {bound}") }
    # Rejects the short final block of the 63-bit range so every residue is equally reachable.
    let ceiling = _INT_MAX - (_INT_MAX % bound + 1) % bound
    loop {
        let draw = __prim__.random_u64() & _INT_MAX
        if draw <= ceiling { return draw % bound }
    }
}

fn _less(a: any, b: any) -> bool {
    return a < b
}

fn _greater(a: any, b: any) -> bool {
    return a > b
}

fn _equals(a: any, b: any) -> bool {
    return a == b
}

fn _identity(value: any) -> any {
    return value
}
