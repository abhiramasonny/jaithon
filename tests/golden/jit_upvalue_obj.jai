#: gc-stress-every: 50
#: A captured upvalue of raw object type must not decline the enclosing
#: closure.
#:
#: `OP_GET_UPVALUE` classified a capture into `SLOT_INT`/`SLOT_FLOAT` only
#: and declined for everything else -- a closure over a `str`, `list`,
#: `dict`, `bool`, or instance never compiled, even though every other read
#: site (`OP_GET_FIELD`, `OP_GET_GLOBAL`) already handles the full range.
#: Widened to the same ladder, with the same guards: `SLOT_LIST` confirms
#: `Obj.type == OBJ_LIST`, `SLOT_INST` confirms `Obj.type == OBJ_INSTANCE`
#: and the class shape, before either is trusted.
class Counter {
    pub var n: int

    fn init(self, n: int) -> void { self.n = n }
}

fn make_str_len_summer(s: any) -> fn() -> int {
    return fn() -> int { return s.len() }
}

fn make_list_len_summer(xs: any) -> fn() -> int {
    return fn() -> int { return xs.len() }
}

fn make_dict_len_summer(d: any) -> fn() -> int {
    return fn() -> int { return d.len() }
}

fn make_counter_reader(c: any) -> fn() -> int {
    return fn() -> int { return c.n }
}

fn make_bool_reader(b: any) -> fn() -> int {
    return fn() -> int { return if b { 1 } else { 0 } }
}

let str_sum = make_str_len_summer("hello world")
let list_sum = make_list_len_summer([1, 2, 3, 4])
let dict_sum = make_dict_len_summer({"a": 1, "b": 2})
let counter_read = make_counter_reader(Counter(42))
let bool_read = make_bool_reader(true)

var total = 0
var i = 0
while i < 200000 {
    total = total + str_sum() + list_sum() + dict_sum() + counter_read() + bool_read()
    i = i + 1
}
print(total)
