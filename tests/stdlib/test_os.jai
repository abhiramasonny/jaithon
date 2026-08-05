#: `std.os`'s process surface, over the `os_spawn` primitive (spec §10).
#:
#: The primitive was specified and never implemented, so `run`, `run_checked`,
#: `shell`, `spawn` and every `Process` method raised `NameError` — and so did
#: `hostname`, `username` and `pid`, which all derive their answer from a child
#: process. These pin the contract the module documents: PATH lookup, the exit
#: status, the two output streams captured separately, `cwd` and `environment`,
#: standard input, and a live child that can be read, waited for and signalled.
from std.os import SIGTERM, cpu_count, hostname, pid, run, run_checked
from std.os import shell, spawn, username, which
from std.test import assert_eq, assert_false, assert_gt, assert_throws, assert_true

fn test_run_captures_stdout_and_status() -> void {
    let result = run(["echo", "one two"])
    assert_eq(result.stdout, "one two\n")
    assert_eq(result.stderr, "")
    assert_eq(result.exit_code, 0)
    assert_true(result.ok())
}

fn test_run_reports_a_failing_command() -> void {
    let result = run(["sh", "-c", "exit 3"])
    assert_eq(result.exit_code, 3)
    assert_false(result.ok())
}

fn test_the_two_streams_stay_apart() -> void {
    let result = run(["sh", "-c", "echo out; echo err 1>&2"])
    assert_eq(result.stdout, "out\n")
    assert_eq(result.stderr, "err\n")
}

fn test_stdin_is_written_and_then_closed() -> void {
    # `cat` only ends when its input does, so this also proves the write end
    # is closed rather than left open.
    assert_eq(run(["cat"], stdin_text: "piped\n").stdout, "piped\n")
}

#: Neither stream fits in a pipe buffer, and the input does not either. A
#: spawn that wrote all of stdin before reading anything, or read stdout to
#: the end before touching stderr, deadlocks here rather than failing.
fn test_large_streams_do_not_deadlock() -> void {
    let big = "x".repeat(300000)
    let echoed = run(["cat"], stdin_text: big)
    assert_eq(echoed.stdout.len(), 300000)
    assert_eq(echoed.exit_code, 0)

    let both = run(["sh", "-c", "yes ab | head -c 200000; yes cd | head -c 200000 1>&2"])
    assert_eq(both.stdout.len(), 200000)
    assert_eq(both.stderr.len(), 200000)
}

fn test_cwd_and_environment_options() -> void {
    assert_true(run(["pwd"], cwd: "/").stdout.strip().starts_with("/"))
    let result = run(["sh", "-c", "echo $JAI_TEST_VAR"], environment: {"JAI_TEST_VAR": "hello"})
    assert_eq(result.stdout, "hello\n")
}

fn test_a_missing_program_is_a_file_not_found_error() -> void {
    let _e = assert_throws(FileNotFoundError, || run(["no-such-program-xyzzy"]))
    assert_eq(which("no-such-program-xyzzy"), null)
    assert_true(which("sh") is not null)
}

fn test_run_checked_raises_on_a_nonzero_status() -> void {
    assert_eq(run_checked(["echo", "fine"]).stdout, "fine\n")
    let _e = assert_throws(OSError, || run_checked(["sh", "-c", "echo bad 1>&2; exit 1"]))
}

fn test_shell_runs_a_command_line() -> void {
    assert_eq(shell("echo one; echo two").stdout, "one\ntwo\n")
}

fn test_an_empty_command_is_rejected() -> void {
    let _e = assert_throws(ValueError, || run([]))
}

#: A live child: write to it, read what it answers, wait for its status.
fn test_spawn_talks_to_a_running_child() -> void {
    let child = spawn(["sh", "-c", "read line; echo got $line; exit 7"])
    assert_gt(child.pid, 0)
    child.write_line("ping")
    child.close_stdin()
    assert_eq(child.read_line(), "got ping")
    assert_eq(child.wait(), 7)
    # Waiting twice answers from the remembered status rather than waiting for
    # a process that has already been reaped.
    assert_eq(child.wait(), 7)
    assert_eq(child.poll(), 7)
    assert_false(child.is_running())
}

fn test_a_child_can_be_signalled() -> void {
    let child = spawn(["sleep", "30"])
    assert_true(child.is_running())
    child.signal(SIGTERM)
    assert_eq(child.wait(), 128 + SIGTERM)
}

fn test_the_derived_facts_need_a_child_process() -> void {
    assert_gt(pid(), 0)
    assert_gt(hostname().len(), 0)
    assert_gt(username().len(), 0)
    assert_gt(cpu_count(), 0)
}
