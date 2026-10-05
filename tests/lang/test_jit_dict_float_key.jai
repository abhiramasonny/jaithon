#: A dict read inside float arithmetic. OP_GET_INDEX may leave float entries
#: in the FP bank, which suits a list read; the dict arm instead stores its key
#: from the X register and makes a call that clobbers the bank. A float key
#: read from a local's FP home went in as stale bits, and `d[x]` raised
#: KeyError for a key that was there.
from std.test import assert_eq

fn weighted(d: dict[float, float], keys: list[float]) -> float {
    var s = 0.0
    for i in 0..len(keys) {
        let x = keys[i]
        s = s + x * d[x]
    }
    return s
}

fn scaled_by_name(d: dict[str, float], xs: list[float]) -> float {
    var s = 0.0
    for i in 0..len(xs) {
        let x = xs[i]
        s = s + x * d["w"] + (x + 1.0) * d["v"]
    }
    return s
}

fn test_a_float_key_from_a_local_finds_its_entry() -> void {
    var d: dict[float, float] = {}
    var keys: list[float] = []
    for i in 0..50 {
        let k = float(i) * 0.5
        d[k] = k + 1.0
        keys.push(k)
    }
    for _round in 0..200 { assert_eq(weighted(d, keys), 10718.75) }
}

fn test_live_float_operands_survive_a_dict_read() -> void {
    let d: dict[str, float] = {"w": 2.0, "v": 0.5}
    var xs: list[float] = []
    for i in 0..40 { xs.push(float(i) * 0.25) }
    for _round in 0..200 { assert_eq(scaled_by_name(d, xs), 507.5) }
}
