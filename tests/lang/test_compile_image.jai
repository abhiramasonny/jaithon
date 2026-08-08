#: `__prim__.compile_image` hands over the C front end's own `.jaic` image for a
#: source string.
#:
#: `__prim__.compile` answers an `ObjClosure`, so the bytes themselves were
#: unreachable from Jaithon and every differential check had to be driven from
#: C. `compile.mod.verify` was written against bytes that did not exist and has
#: been dead since; these are the tests for the primitive that revives it.
from std.test import assert_eq, assert_throws, assert_true
from jaithon.compile.jaic import MAGIC, read_module, source_hash

fn _bytes_to_list(data: any) -> list[int] {
    var out: list[int] = []
    for index in range(data.len()) { out.push(data[index]) }
    return out
}

fn test_an_image_starts_with_the_container_magic() -> void {
    let image = _bytes_to_list(__prim__.compile_image("let answer = 42\n", "<test>"))
    assert_true(image.len() > 32)
    assert_eq(image[:4], MAGIC)
}

fn test_an_image_reads_back_as_a_module() -> void {
    let text = "let answer = 42\n"
    let image = _bytes_to_list(__prim__.compile_image(text, "<test>"))
    let mod = read_module(image, source_hash(text))
    assert_true(mod is not null)
    assert_eq(mod.path, "<test>")
}

fn test_the_optimisation_level_reaches_the_compiler() -> void {
    # -O0 and -O2 do not produce the same code for a loop the optimiser can
    # touch, which is the whole reason the level is a parameter: a comparison
    # run at -O0 that silently compiled the reference at -O2 blames the front
    # ends for the flag.
    let text = "fn spin() -> int {\n" +
        "    var total = 0\n" +
        "    for i in 0..10 { total += i }\n" +
        "    return total\n" +
        "}\n"
    let flat = _bytes_to_list(__prim__.compile_image(text, "<test>", 0))
    let optimised = _bytes_to_list(__prim__.compile_image(text, "<test>", 2))
    assert_true(flat != optimised)
}

fn test_a_rejected_source_raises_rather_than_answering_bytes() -> void {
    let error = assert_throws(ParseError, || __prim__.compile_image("let x = \n", "<test>"))
    assert_true(error.message.len() > 0)
}
