#: Merge sort written in the language, not the runtime's own sort.
#:
#: `list.sort()` would time two C implementations against each other and say
#: nothing about the interpreter. This is list indexing, comparison and
#: recursion in the loop instead.
from std.os import env_or

let LEVEL = env_or("BENCH_LEVEL", "hard")
let SCALE = if LEVEL == "easy" { 16 } elif LEVEL == "medium" { 4 } else { 1 }
let ITEMS = 1_000_000 // SCALE

fn merge(left: list[int], right: list[int]) -> list[int] {
    var out: list[int] = []
    var i = 0
    var j = 0
    while i < left.len() and j < right.len() {
        if left[i] <= right[j] {
            out.push(left[i])
            i += 1
        } else {
            out.push(right[j])
            j += 1
        }
    }
    while i < left.len() {
        out.push(left[i])
        i += 1
    }
    while j < right.len() {
        out.push(right[j])
        j += 1
    }
    return out
}

fn sort(values: list[int]) -> list[int] {
    if values.len() <= 1 { return values }
    let mid = values.len() // 2
    return merge(sort(values[0:mid]), sort(values[mid:values.len()]))
}

fn main() -> int {
    var data: list[int] = []
    var seed = 12345
    for _i in 0..ITEMS {
        seed = (seed * 1103515245 + 12345) % 2147483648
        data.push(seed % 100000)
    }
    let sorted = sort(data)
    var checksum = 0
    var at = 0
    while at < sorted.len() {
        checksum = (checksum + sorted[at] * (at % 7 + 1)) % 1000000007
        at += 1
    }
    print(sorted.len())
    print(checksum)
    return 0
}
