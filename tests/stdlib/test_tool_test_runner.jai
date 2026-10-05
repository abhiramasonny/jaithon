#: `jaithon test` running files in parallel workers.
#:
#: Each case writes a small suite to a fresh directory and runs the real binary
#: over it, because what is being checked is the parent's bookkeeping: which
#: worker a result came from, and what it does when a worker dies.
from std.io import Path
from std.test import assert_eq, assert_true

fn _suite(name: str, files: dict[str, str]) -> str {
    let dir = __prim__.os_env("TMPDIR") ?? "/tmp"
    let root = f"{dir}/jaithon-runner-{name}-{__prim__.time_wall()}"
    __prim__.io_mkdir(root)
    for (file, source) in files.items() {
        let handle = __prim__.io_open(root + "/" + file, "wb")
        __prim__.io_write(handle, source)
        __prim__.io_close(handle)
    }
    return root
}

#: Recursive: the runner leaves its timing record in `__jaicache__` there.
fn _drop(root: str) -> void { Path(root).remove(true) }

fn _run_test(arguments: list[str]) -> dict[str, any] {
    let command = [__prim__.os_argv()[0], "test"]
    command.extend(arguments)
    return __prim__.os_spawn(command, {})
}

#: Drop the timings, the one part of the output that differs run to run. Each
#: is the last dimmed span on its line.
fn _untimed(text: str) -> str {
    let dim = "\u{1B}[2m"
    let kept: list[str] = []
    for line in text.split("\n") {
        let cut = line.rfind(dim)
        kept.push(if cut >= 0 { line[0:cut] } else { line })
    }
    return "\n".join(kept)
}

let PASSING = "from std.test import assert_eq\nfn test_adds() -> void { assert_eq(1 + 1, 2) }\n"

#: A worker that died before reporting used to vanish from the output: its
#: file printed nothing, the summary counted only the files that reported, and
#: the run exited 0.
fn test_a_worker_that_exits_early_is_reported_not_dropped() -> void {
    let files = {
        "test_fine.jai": PASSING,
        "test_dies.jai": "fn test_leaves() -> void { __prim__.os_exit(3) }\n",
    }
    let root = _suite("dies", files)
    let result = _run_test(["--jobs=2", root])
    _drop(root)
    assert_eq(result["exit_code"], 1)
    let out: str = result["stdout"]
    assert_true(out.contains("test_dies"), out)
    assert_true(out.contains("exited with status 3"), out)
    assert_true(out.contains("test_fine"), out)
}

fn test_parallel_output_matches_the_sequential_runner() -> void {
    let files = {
        "test_one.jai": PASSING,
        "test_two.jai": PASSING + "fn test_concat() -> void { assert_eq(\"a\" + \"b\", \"ab\") }\n",
        "test_three.jai": "fn helper() -> void { }\n",
    }
    let root = _suite("same", files)
    let sequential = _run_test(["--jobs=1", "--verbose", root])
    let parallel = _run_test(["--jobs=3", "--verbose", root])
    _drop(root)
    assert_eq(parallel["exit_code"], 0)
    assert_eq(_untimed(parallel["stdout"]), _untimed(sequential["stdout"]))
    assert_eq(parallel["stderr"], sequential["stderr"])
}

#: A worker's standard input is not the runner's: a test that reads it gets
#: end of input rather than waiting on a pipe nobody writes to.
fn test_a_test_reading_stdin_sees_end_of_input() -> void {
    let files = {
        "test_reads.jai": "from std.test import assert_eq\nfrom std.io import read_all_stdin\nfn test_reads() -> void { assert_eq(read_all_stdin(), \"\") }\n",
        "test_other.jai": PASSING,
    }
    let root = _suite("stdin", files)
    let result = _run_test(["--jobs=2", root])
    _drop(root)
    assert_eq(result["exit_code"], 0, result["stdout"])
}

#: The second run starts files in the order the first one's timings suggest;
#: what it prints must not depend on that order.
fn test_a_run_ordered_by_recorded_times_prints_the_same() -> void {
    let files = {
        "test_one.jai": PASSING,
        "test_two.jai": PASSING,
        "test_three.jai": PASSING,
    }
    let root = _suite("again", files)
    let first = _run_test(["--jobs=3", "--verbose", root])
    let recorded = Path(root).join("__jaicache__").join("test_times").exists()
    let second = _run_test(["--jobs=3", "--verbose", root])
    _drop(root)
    #: JAITHON_TEST_FORK=0 reaches the inner run too, and that runner keeps no record.
    if (__prim__.os_env("JAITHON_TEST_FORK") ?? "1") != "0" { assert_true(recorded) }
    assert_eq(_untimed(second["stdout"]), _untimed(first["stdout"]))
}
