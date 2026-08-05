#: A file for `:load` to pull into a REPL session.
#:
#: Its top level runs on load and its bindings stay in the session, which is
#: what `tests/repl/meta_load.repl` asserts.
let loaded_value = 21

fn loaded_double(n: int) -> int {
    return n * 2
}

print("loaded module ran")
