#: gc-stress-every: 50
#: Same reason as jit_field_obj: a hot loop revisits one program point, so
#: collecting on every allocation buys nothing after the first pass. Bare was
#: 15,866ms; at N=50 it is 151ms and still collects 8,204 times.
#: `s[a:b]` on a string used to decline the enclosing loop -- `OP_GET_SLICE`
#: accepted only SLOT_LIST and said "slicing a container this tier does not
#: model" for everything else, which is what kept `word_freq`'s scan loop (64%
#: of that benchmark) and `json_parse`'s token cutter interpreted.
#:
#: It is the same runtime call either way; what the arm needs is the container's
#: object type pinned before the descriptor, so the branch `jaiSliceGet` takes
#: is settled. So the things to check are that the clamped forms agree with the
#: interpreter byte for byte, and that a slot which held a string when the loop
#: was compiled and holds a list on the next entry DEOPTIMISES rather than
#: slicing a string's header as a list's.
fn cut(s: str, n: int) -> int {
    var t = 0
    var i = 0
    while i < n {
        t = t + s[1:4].len() + s[:3].len() + s[2:].len() + s[:].len()
        t = t + s[3:99].len() + s[-2:].len()
        i = i + 1
    }
    return t // n
}

fn first(xs: any, n: int) -> int {
    var t = 0
    var i = 0
    while i < n {
        t = t + xs[0:2].len()
        i = i + 1
    }
    return t // n
}

print(cut("abcdefgh", 200000))

print(first("abcdefgh", 200000))
print(first([1, 2, 3, 4], 200000))
print(first("abcdefgh", 200000))
