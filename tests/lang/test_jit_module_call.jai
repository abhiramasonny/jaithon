#: A module-qualified call from inside a compiled body.
#:
#: `math.floor(x)` puts the MODULE under the arguments where a plain global
#: call puts the function, and the tier had no arm for it -- so every `std.*`
#: call declined the loop around it. The entry it drops afterwards holds a
#: register, which is the part that had to change in the call path, so what
#: these check is that the surrounding values survive it.
import std.math
from std.test import assert_eq, assert_true

fn mixed(values: list[float], rounds: int) -> float {
    var total = 0.0
    for _round in 0..rounds {
        total = 0.0
        for index in 0..values.len() {
            #: Values either side of the call, so a register accounted for
            #: wrongly when the module entry is dropped would show up here.
            let before = values[index] * 2.0
            let floored = math.floor(values[index])
            let after = values[index] + 1.0
            total += before + floored + after
        }
    }
    return total
}

fn test_a_module_call_agrees_hot_and_cold() -> void {
    var values: list[float] = []
    for index in 0..64 { values.push(float(index % 17) * 0.75 - 6.0) }
    let cold = mixed(values, 1)
    let hot = mixed(values, 3000)
    assert_eq(hot, cold, "a compiled module call disagreed with a cold one")
}

fn test_the_arguments_either_side_survive() -> void {
    var values: list[float] = []
    for index in 0..32 { values.push(float(index) - 8.0) }
    var expected = 0.0
    for index in 0..32 {
        let v = values[index]
        expected += v * 2.0 + math.floor(v) + v + 1.0
    }
    let hot = mixed(values, 2000)
    assert_eq(hot, expected, "a value either side of the call was clobbered")
}

fn test_several_module_calls_in_one_expression() -> void {
    var total = 0.0
    for _round in 0..3000 {
        total = 0.0
        for index in 0..16 {
            let v = float(index) - 7.5
            total += math.floor(v) + math.ceil(v) + math.fabs(v)
        }
    }
    var expected = 0.0
    for index in 0..16 {
        let v = float(index) - 7.5
        expected += math.floor(v) + math.ceil(v) + math.fabs(v)
    }
    assert_eq(total, expected, "stacked module calls came back wrong")
}
