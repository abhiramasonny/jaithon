#: The test runner behind `jaithon test`.
#:
#: Discovery is by convention only — a file and its top-level, no-argument
#: functions named `test_*` — so there is no registration list to forget.
#: `--doc` instead runs every fenced block in a `#:` comment as a test; a
#: line with a trailing `# value` comment asserts that value, which is what
#: keeps stale examples from passing silently.
#:
#: ```
#: expected_of("f(1)   # 2")          # "2"
#: expected_of("f(1)")                # null
#: is_test_name("test_adds")          # true
#: is_test_name("tester")             # false
#: ```
from jaithon.ast import NodeKind
from jaithon.ast_unparse import to_source
from jaithon.compile.diag import DiagnosticBag
from jaithon.compile.lexer import Lexer
from jaithon.compile.parser import parse
from jaithon.compile.token import TokenKind
from std.fmt import bold, green, red
from std.io import Path, print_err, read_file
from std.os import argv, cpu_count, env, spawn, temp_dir
from std.test import Outcome, Report, Test, TestResult
from std.time import Duration
from .args import Arguments
from .doc import Example, ModuleDoc, collect_module, module_name_of
from .fmt import expand_paths, parse_module, scan_comments, tail_arguments

#: The prefix a function's name must have to be run as a test.
pub const TEST_PREFIX: str = "test_"

#: Report whether `name` names a test.
#:
#: ```
#: is_test_name("test_adds")    # true
#: is_test_name("testing")      # false
#: ```
pub fn is_test_name(name: str) -> bool {
    return name.starts_with(TEST_PREFIX) and name.len() > TEST_PREFIX.len()
}

#: Report whether `path` is a file the runner should look inside when walking
#: a directory.
#:
#: The obvious alternative — anything under `tests` — is wrong: that
#: directory also holds benchmarks and diagnostic fixtures that don't compile
#: *on purpose*, and walking into them would report the suite red for files
#: doing their job.
#:
#: ```
#: is_test_file(Path("tests/lang/test_closures.jai"))    # true
#: is_test_file(Path("src/test_helpers.jai"))            # true
#: is_test_file(Path("tests/errors/unclosed_brace.jai")) # false
#: is_test_file(Path("lib/std/str.jai"))                 # false
#: ```
pub fn is_test_file(path: Path) -> bool {
    return is_test_name(path.stem())
}

#: Return the names of the test functions declared at the top level of `source`.
#:
#: A `test_` function that takes arguments is not one the runner can call, so it
#: is left out rather than called wrongly.
#:
#: ```
#: test_functions("fn test_a() -> void { }\nfn helper() -> void { }\n", "t.jai")
#: #     ["test_a"]
#: ```
pub fn test_functions(source: str, path: str) -> list[str] {
    let program = parse_module(source, path)
    let names: list[str] = []
    for statement in program.child_list("stmts") {
        if statement.kind != NodeKind.FnDecl { continue }
        let name = statement.opt_text("name")
        if name is null or not is_test_name(name) { continue }
        if statement.records("params").len() > 0 { continue }
        names.push(name)
    }
    return names
}

#: Return the expected-value comment of an example line, or null.
#:
#: The value is the text after the last `#` that is not inside a string, which
#: is how the examples throughout `lib/` are written.
#:
#: ```
#: expected_of("counts.get(\"a\")    # 3")    # "3"
#: expected_of("let x = 1")                   # null
#: ```
pub fn expected_of(line: str) -> str? {
    var found: str? = null
    for comment in scan_comments(line) {
        if not comment.own_line { found = comment.body() }
    }
    return found
}

#: Return the code of an example line, without its expected-value comment.
#:
#: ```
#: code_of("f(1)   # 2")    # "f(1)"
#: ```
pub fn code_of(line: str) -> str {
    var cut = -1
    for comment in scan_comments(line) {
        if not comment.own_line { cut = comment.column - 1 }
    }
    if cut < 0 { return line.strip() }
    return line[0:cut].strip()
}

#: Run every test in one file and return the results.
#:
#: `filter`, when given, keeps only tests whose `module:name` contains it.
#:
#: Each file runs in a namespace of its own: run in the runner's namespace, a
#: test file's `import std.str as str` would rebind the runner's own `str`
#: and break the next report — and files could no longer be run in any order.
pub fn run_file(path: str, filter: str?, verbose: bool) -> list[TestResult] {
    let source = read_file(path)
    let suite = module_name_of(path)
    var names: list[str] = []
    try {
        names = test_functions(source, path)
    } catch error: ParseError {
        return [_broken(suite, "parse", error.message)]
    }

    let wanted = [name for name in names if _matches(suite, name, filter)]
    if wanted.len() == 0 { return [] }

    let namespace = f"<test {path}>"
    try {
        __prim__.exec(source, namespace)
    } catch error: Error {
        return [_broken(suite, "load", f"{type_of(error)}: {error.message}")]
    }

    let results: list[TestResult] = []
    for name in wanted {
        let call = name + "()"
        let result = Test(name, fn() -> any { return __prim__.exec(call, namespace) }).run(suite)
        if verbose { print("  " + result.to_str()) }
        results.push(result)
    }
    return results
}

#: Build the runnable program for one doc example.
#:
#: The lines run inside an immediately-called anonymous function, so one
#: example's `let` bindings can't collide with the next one's.
#:
#: The module surface is imported by listing names, not `from m import *` —
#: the wildcard has no bytecode form (`E0902`), so examples built on it never
#: compiled and `--doc` found nothing. Only names the example actually uses
#: are imported, to dodge an unused-import warning (`W0100`) and a duplicate
#: declaration (`E0302`) for names already in scope.
pub fn example_program(mod: ModuleDoc, imports: list[str], example: Example) -> str {
    let body: list[str] = []
    let own: list[str] = []
    var asserted = false
    for line in example.lines {
        let code = code_of(line)
        if code.len() == 0 {
            body.push("")
            continue
        }
        if _is_import(code) {
            own.push(code)
            continue
        }
        let expected = expected_of(line)
        if expected is not null and _is_expression(code) and _is_expression(expected) {
            body.push(f"    assert_eq({code}, {expected})")
            asserted = true
        } else {
            body.push("    " + line)
        }
    }

    let used = _names_used(body)
    let bound: set[str] = set()
    let head: list[str] = []
    if asserted {
        head.push("from std.test import assert_eq")
        bound.add("assert_eq")
    }
    for (statement, names) in _import_bindings(mod, own, used) {
        head.push(statement)
        for name in names { bound.add(name) }
    }
    for (statement, names) in _import_bindings(mod, imports, used) {
        let fresh = [name for name in names if name not in bound]
        if fresh.len() != names.len() { continue }
        head.push(statement)
        for name in names { bound.add(name) }
    }
    if mod.name.contains(".") {
        let surface: list[str] = []
        for entry in mod.entries {
            if entry.name not in used or entry.name in bound { continue }
            bound.add(entry.name)
            surface.push(entry.name)
        }
        if surface.len() > 0 { head.push(f"from {mod.name} import " + ", ".join(surface)) }
    }

    head.push("fn() -> any {")
    head.extend(body)
    head.push("}()")
    return "\n".join(head) + "\n"
}

#: Run the doc examples of one file and return the results.
#:
#: An example that does not compile is a failing test, not a silent skip: the
#: point of running them is to find the ones that have gone stale.
pub fn run_doc_examples(path: str, filter: str?, verbose: bool) -> list[TestResult] {
    let suite = module_name_of(path)
    try {
        return _run_examples(collect_module(path), _imports_of(path), suite, filter, verbose)
    } catch error: ParseError {
        return [_broken(suite, "parse", error.message)]
    }
}

#: Run `jaithon test`.
#:
#: ```text
#: jaithon test [--verbose] [--doc] [--filter=substring] [paths...]
#: ```
#:
#: With no path, `tests` is searched (or `lib` when `--doc` is given). Exits
#: 1 when anything failed — what the CI gate reads.

#: Serialise one result onto a single line for a parent process to parse.
#:
#: Tabs separate the fields and newlines inside a message or traceback are
#: escaped, because a failure's text is arbitrary and both characters occur in
#: real tracebacks.
fn _escape_field(text: str) -> str {
    return text.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")
}

fn _unescape_field(text: str) -> str {
    var out = ""
    var index = 0
    while index < text.len() {
        let ch = text[index]
        if ch == "\\" and index + 1 < text.len() {
            let next = text[index + 1]
            if next == "n" { out += "\n" } elif next == "t" { out += "\t" } else { out += next }
            index += 2
            continue
        }
        out += ch
        index += 1
    }
    return out
}

fn _has_gc_stress(arguments: list[str]) -> bool {
    for argument in arguments {
        if argument.starts_with("--gc-stress") { return true }
    }
    return false
}

fn _outcome_code(outcome: Outcome) -> str {
    if outcome == Outcome.Passed { return "P" }
    if outcome == Outcome.Failed { return "F" }
    if outcome == Outcome.Errored { return "E" }
    return "S"
}

fn _outcome_of(code: str) -> Outcome {
    if code == "P" { return Outcome.Passed }
    if code == "F" { return Outcome.Failed }
    if code == "E" { return Outcome.Errored }
    return Outcome.Skipped
}

fn _emit_shard(results: list[TestResult]) -> void {
    for result in results {
        let fields = [
            _outcome_code(result.outcome),
            str(result.duration.to_nanos()),
            _escape_field(result.suite),
            _escape_field(result.name),
            _escape_field(result.message),
            _escape_field(result.trace),
        ]
        print("@@R " + "\t".join(fields))
    }
}

fn _parse_shard(text: str) -> list[TestResult] {
    let results: list[TestResult] = []
    for line in text.split("\n") {
        if not line.starts_with("@@R ") { continue }
        let fields = line[4:].split("\t")
        if fields.len() < 6 { continue }
        results.push(
            TestResult(
                _unescape_field(fields[3]),
                _unescape_field(fields[2]),
                _outcome_of(fields[0]),
                Duration(int(fields[1])),
                _unescape_field(fields[4]),
                _unescape_field(fields[5])
            )
        )
    }
    return results
}

#: Run each path in its own process, `limit` at a time.
#:
#: MEASURED: the jaicv suite is 100.0s in one process and 37.2s across eight,
#: and 37.2s is what `test_against_opencv.jai` costs on its own -- so the
#: parallelism is near-perfect and that one file is the whole remaining
#: ceiling. Per-process startup does not eat it: three files cost 40.1s run
#: separately against 41.6s in one process, because loading a module twice is
#: cheaper than the suite is long.
#:
#: Processes rather than threads: the VM has one heap and one collector, and
#: nothing about it is thread-safe.
fn _run_sharded(exe: str, paths: list[str], options: _Options, limit: int) -> list[TestResult] {
    var running: list[any] = []
    var captured: dict[str, str] = {}
    #: Warnings a test file raises while compiling go to the CHILD's stderr.
    #: Dropping them would make a parallel run quieter than a sequential one,
    #: which is a real difference in output and was caught by diffing the two.
    var noise: dict[str, str] = {}
    var failures: dict[str, int] = {}
    var next_path = 0

    fn launch(path: str) -> void {
        let process = spawn(_worker_command(exe, path))
        #: Nothing is ever written to it; a test reading it gets end of input.
        process.close_stdin()
        running.push((path, process))
    }

    while next_path < paths.len() or running.len() > 0 {
        while running.len() < limit and next_path < paths.len() {
            launch(paths[next_path])
            next_path += 1
        }
        #: `read_stdout` blocks until the child closes it, which is what makes
        #: this a wait as well as a read. Draining the oldest first keeps the
        #: reported order the same as the sequential runner's.
        let (path, process) = running[0]
        captured[path] = process.read_stdout()
        noise[path] = process.read_stderr()
        let status = process.wait()
        if status != 0 { failures[path] = status }
        running = running[1:]
    }
    return _report_shards(paths, captured, noise, failures, options)
}

#: The command that runs `path` alone in a worker process.
#:
#: Everything the process was invoked with EXCEPT the paths, so a worker
#: inherits flags this tool never parses -- `--gc-stress` is the one that
#: matters, since it belongs to the CLI rather than to `test`, and a
#: hand-built command silently dropped it. Rebuilding the command from
#: argv() means a flag added later needs no change here.
fn _worker_command(exe: str, path: str) -> list[str] {
    let own = argv()
    var command = [exe, "test", "--_shard", "--jobs=1"]
    var index = 1
    while index < own.len() {
        let argument = own[index]
        #: `test` itself, the paths, and any --jobs the caller gave: all
        #: replaced here.
        let is_path = not argument.starts_with("-")
        let is_jobs = argument.starts_with("--jobs") or argument.starts_with("-j")
        if not is_path and not is_jobs { command.push(argument) }
        index += 1
    }
    command.push(path)
    return command
}

#: Print each file's results in path order, as the sequential runner would.
#:
#: `failures` maps a path to the status its worker died with; a file that
#: died before reporting anything is an error rather than a silent omission.
fn _report_shards(
    paths: list[str],
    captured: dict[str, str],
    noise: dict[str, str],
    failures: dict[str, int],
    options: _Options
) -> list[TestResult] {
    let results: list[TestResult] = []
    for path in paths {
        let text = captured.get(path) ?? ""
        var found = _parse_shard(text)
        let status = failures.get(path)
        if found.len() == 0 and status is not null {
            let message = f"the test process exited with status {status} before reporting"
            found = [_broken(module_name_of(path), "crash", message)]
        }
        let complaints = noise.get(path) ?? ""
        if complaints.len() > 0 { print_err(complaints.rstrip()) }
        #: Same shape as the sequential runner: per-test lines when verbose,
        #: the one-line summary when not. Printing the summary either way made
        #: a parallel run unparseable to anything reading for `  pass  `. The
        #: sequential runner names every file when verbose, tests or not.
        if options.verbose { print(bold(module_name_of(path))) }
        if found.len() == 0 { continue }
        if options.verbose {
            for result in found { print("  " + result.to_str()) }
        } else {
            print(_file_line(path, found))
        }
        results.extend(found)
    }
    return results
}

#: JAITHON_TEST_FORK=0 goes back to starting a fresh `jaithon test --_shard`
#: process per file; read once.
let _FORK_WORKERS: bool = (env("JAITHON_TEST_FORK") ?? "1") != "0"

#: Run each path in a forked copy of this process, `limit` at a time.
#:
#: A spawned worker spends ~42M cycles loading the front end and the runner
#: and ~130M more having the checker parse the signatures of `std.test`'s
#: import closure -- for a small test file that is three quarters of its cost
#: (tests/lang: 21.7G cycles across 107 workers against 6.3G in one process).
#: A fork starts from this runner with all of that already done, and is still
#: a process of its own, so a file gets fresh module state, a crash or a call
#: to exit takes out only that file, and the result is attributed to it.
#:
#: A forked child cannot use the GPU (see _NEEDS_EXEC), so a file that loads
#: a native framework is run again in a worker started from scratch.
#:
#: Children are reaped in the order they finish, not the order they started:
#: draining the oldest first left every free core idle behind one slow file
#: (the unit list: 33.8s wall for 82s of CPU on 12 cores). Output goes to two
#: files per child rather than pipes, so no child can block on a full pipe,
#: and is printed in path order at the end.
fn _run_forked(exe: str, paths: list[str], options: _Options, limit: int) -> list[TestResult] {
    _warm_for_children()
    let scratch = _scratch_dir()
    let history = _History.load(paths)
    let order = history.slowest_first(paths)
    var captured: dict[str, str] = {}
    var noise: dict[str, str] = {}
    var failures: dict[str, int] = {}
    var running: dict[int, int] = {}
    var started: dict[int, int] = {}
    var spawned: set[int] = set()
    var again: list[int] = []
    var next_order = 0
    while next_order < order.len() or again.len() > 0 or running.len() > 0 {
        while running.len() < limit and (again.len() > 0 or next_order < order.len()) {
            let from_queue = again.len() == 0
            let index = if from_queue { order[next_order] } else { again.pop() }
            if from_queue { next_order += 1 }
            let base = f"{scratch}/{index}"
            started[index] = __prim__.time_mono()
            if not from_queue or history.needs_exec(paths[index]) {
                let command = _worker_command(exe, paths[index])
                let child = __prim__.os_spawn_to(command, base + ".out", base + ".err")
                running[child] = index
                spawned.add(index)
                continue
            }
            let child = __prim__.os_fork_to(base + ".out", base + ".err")
            if child == 0 { _run_child(paths[index], options) }
            running[child] = index
        }
        let done = __prim__.os_wait_any()
        let index = running.get(done[0])
        if index is null { continue }
        running.remove(done[0])
        if done[1] == _NEEDS_EXEC and index not in spawned {
            again.push(index)
            continue
        }
        let path = paths[index]
        let base = f"{scratch}/{index}"
        history.record(path, __prim__.time_mono() - (started.get(index) ?? 0), index in spawned)
        captured[path] = _take_file(base + ".out")
        noise[path] = _take_file(base + ".err")
        if done[1] != 0 { failures[path] = done[1] }
    }
    _remove_quietly(Path(scratch), true)
    history.save()
    return _report_shards(paths, captured, noise, failures, options)
}

#: How long each test file took last time, and whether it had to run in a
#: process of its own, kept beside the files in `__jaicache__/test_times`.
#:
#: Only a hint: it orders the work, never what is reported. Starting the
#: longest file first keeps it off the end of the run, where it would leave
#: every other core idle; and a file already known to load a native framework
#: skips the forked attempt that would only exit with _NEEDS_EXEC.
class _History {
    var times: dict[str, int]
    var exec_only: set[str]
    var dirs: dict[str, bool]

    pub fn init(self) {
        self.times = {}
        self.exec_only = set()
        self.dirs = {}
    }

    pub static fn load(paths: list[str]) -> _History {
        let history = _History()
        for path in paths {
            let dir = Path(path).parent().text
            if dir in history.dirs { continue }
            history.dirs[dir] = false
            try {
                let file = Path(_History.file_of(dir))
                if not file.exists() { continue }
                for line in file.read_text().split("\n") {
                    let fields = line.split("\t")
                    if fields.len() != 3 { continue }
                    let full = dir + "/" + fields[2]
                    history.times[full] = int(fields[0])
                    if fields[1] == "x" { history.exec_only.add(full) }
                }
            } catch _error: Error { }
        }
        return history
    }

    static fn file_of(dir: str) -> str { return Path(dir).join("__jaicache__").join("test_times").text }

    #: Path indexes, longest recorded first; files with no record keep their
    #: place ahead of every recorded one, so a first run is in path order.
    pub fn slowest_first(self, paths: list[str]) -> list[int] {
        let unknown: list[int] = []
        let by_time: dict[int, list[int]] = {}
        for index in 0..paths.len() {
            let took = self.times.get(_History.key(paths[index]))
            if took is null {
                unknown.push(index)
                continue
            }
            let same = by_time.get(took)
            if same is null { by_time[took] = [index] } else { same.push(index) }
        }
        let order = unknown
        let times = by_time.keys().sorted()
        var at = times.len() - 1
        while at >= 0 {
            order.extend(by_time[times[at]])
            at -= 1
        }
        return order
    }

    pub fn needs_exec(self, path: str) -> bool { return _History.key(path) in self.exec_only }

    #: Built the same way from a path as from a record, so the two always
    #: match however the path was spelled (`a//b`, `./a`).
    static fn key(path: str) -> str {
        let file = Path(path)
        return file.parent().text + "/" + file.name()
    }

    pub fn record(self, path: str, nanos: int, exec_only: bool) -> void {
        let key = _History.key(path)
        self.times[key] = nanos
        if exec_only { self.exec_only.add(key) } else { self.exec_only.discard(key) }
        self.dirs[Path(path).parent().text] = true
    }

    #: Rewrite each directory's record that this run touched. Written to a
    #: temporary name and renamed over, so a concurrent run reads either the
    #: old record or the new one.
    pub fn save(self) -> void {
        for (dir, touched) in self.dirs.items() {
            if not touched { continue }
            let lines: list[str] = []
            for (key, nanos) in self.times.items() {
                if not key.starts_with(dir + "/") { continue }
                let name = key[dir.len() + 1:]
                if name.contains("/") { continue }
                let flag = if key in self.exec_only { "x" } else { "-" }
                lines.push(f"{nanos}\t{flag}\t{name}")
            }
            try {
                let cache = Path(dir).join("__jaicache__")
                cache.mkdir(false, true)
                let target = _History.file_of(dir)
                let staging = target + "." + str(__prim__.time_wall())
                Path(staging).write_text("\n".join(lines.sorted()) + "\n")
                __prim__.io_rename(staging, target)
            } catch _error: Error { }
        }
    }
}

#: The status a forked child exits with when it was about to load a native
#: framework. Frameworks that talk to a system service over XPC refuse a
#: process that was forked and not exec'd -- Metal's shader compiler reports
#: "the process is unavailable" and every GPU test fails -- so the loader
#: stops the child before it tries (JAI_EXIT_NEEDS_EXEC in native.h).
const _NEEDS_EXEC: int = 75

#: The child's whole life: run one file, print its `@@R` lines, exit. It must
#: never return into the parent's loop, whatever goes wrong.
fn _run_child(path: str, options: _Options) -> void {
    var code = 0
    try {
        let found = if options.doc {
            run_doc_examples(path, options.filter, false)
        } else {
            run_file(path, options.filter, false)
        }
        _emit_shard(found)
    } catch error: Error {
        print_err(f"jaithon test: {path}: {type_of(error)}: {error.message}")
        code = 70
    }
    __prim__.os_exit(code)
}

#: Have the checker parse `std.test`'s signatures once, here, so every child
#: inherits them instead of parsing them again. Only signatures: running a
#: test module's imports here would share their state between files.
fn _warm_for_children() -> void {
    try {
        __prim__.exec("from std.test import assert_eq\nlet _ = assert_eq\n", "<test warm-up>")
    } catch _error: Error { }
}

#: A fresh directory under the temp dir for the children's output files.
fn _scratch_dir() -> str {
    let base = temp_dir() + "/jaithon-test-" + str(__prim__.time_wall())
    var attempt = 0
    loop {
        let candidate = if attempt == 0 { base } else { f"{base}-{attempt}" }
        if not Path(candidate).exists() {
            try {
                Path(candidate).mkdir()
                return candidate
            } catch _error: Error { }
        }
        attempt += 1
    }
}

fn _take_file(path: str) -> str {
    let file = Path(path)
    if not file.exists() { return "" }
    let text = file.read_text()
    _remove_quietly(file, false)
    return text
}

fn _remove_quietly(path: Path, recursive: bool) -> void {
    try {
        path.remove(recursive)
    } catch _error: Error { }
}

pub fn main(args: list[str]) -> int {
    let options = _Options.parse(tail_arguments(args, "test"))
    if options.usage_error is not null {
        print_err(f"jaithon test: {options.usage_error}")
        print_err("usage: jaithon test [--verbose] [--doc] [--filter=substring] [paths...]")
        return 2
    }

    let paths = if options.doc {
        expand_paths(options.paths)
    } else {
        expand_paths(options.paths, is_test_file)
    }
    #: One file per core unless asked otherwise. A worker is always 1, which
    #: is what stops it spawning workers of its own.
    let wanted = if options.shard { 1 } else { options.jobs }
    #: `--gc-stress` collects on (almost) every allocation, and the parallel
    #: parent allocates for its own bookkeeping -- spawning, reading each
    #: child's output, parsing it back into results. Under stress that
    #: bookkeeping costs far more than the tests: two files that take 3.3s
    #: sequentially had not finished after ten minutes. Stress runs are about
    #: correctness rather than wall time, and the sequential path is the one
    #: worth stressing, so they stay sequential.
    let stressed = _has_gc_stress(argv())
    let limit = if stressed { 1 } elif wanted > 0 { wanted } else { min(cpu_count(), paths.len()) }

    if limit > 1 and paths.len() > 1 {
        #: `args` here is the tool's own arguments, not the process's, so the
        #: binary has to come from os.argv().
        let exe = argv()[0]
        let results = if _FORK_WORKERS {
            _run_forked(exe, paths, options, limit)
        } else {
            _run_sharded(exe, paths, options, limit)
        }
        if results.len() == 0 {
            print_err("jaithon test: no tests found")
            return 1
        }
        let report = Report(results)
        report.print_summary()
        return report.exit_code()
    }

    if options.shard {
        #: A worker prints `@@R` lines and nothing else; the parent formats
        #: the file lines and the summary so the output is byte-identical to
        #: the sequential runner's.
        var shard_results: list[TestResult] = []
        for path in paths {
            let found = if options.doc {
                run_doc_examples(path, options.filter, false)
            } else {
                run_file(path, options.filter, false)
            }
            shard_results.extend(found)
        }
        _emit_shard(shard_results)
        return 0
    }

    let results: list[TestResult] = []
    for path in paths {
        if options.verbose { print(bold(module_name_of(path))) }
        let found = if options.doc {
            run_doc_examples(path, options.filter, options.verbose)
        } else {
            run_file(path, options.filter, options.verbose)
        }
        if found.len() == 0 { continue }
        if not options.verbose { print(_file_line(path, found)) }
        results.extend(found)
    }

    if results.len() == 0 {
        print_err("jaithon test: no tests found")
        return 1
    }
    let report = Report(results)
    report.print_summary()
    return report.exit_code()
}

# --- Running -----------------------------------------------------------------
fn _is_expression(text: str) -> bool {
    let program = parse(text, 0, DiagnosticBag())
    if program is null { return false }
    let stmts = program.child_list("stmts")
    if stmts.len() != 1 { return false }
    return stmts[0].kind == NodeKind.ExprStmt
}

fn _matches(suite: str, name: str, filter: str?) -> bool {
    if filter is null { return true }
    return f"{suite}:{name}".contains(filter)
}

fn _broken(suite: str, stage: str, message: str) -> TestResult {
    return TestResult(stage, suite, Outcome.Errored, Duration.zero(), message, "")
}

fn _run_examples(
    mod: ModuleDoc,
    imports: list[str],
    suite: str,
    filter: str?,
    verbose: bool
) -> list[TestResult] {
    let results: list[TestResult] = []
    for (owner, example) in _examples_of(mod) {
        if not example.runnable { continue }
        let name = f"{owner} (line {example.line})"
        if not _matches(suite, name, filter) { continue }
        let program = example_program(mod, imports, example)
        let namespace = f"<doc {mod.name}:{example.line}>"
        let result = Test(name, fn() -> any { return __prim__.exec(program, namespace) }).run(suite)
        if verbose { print("  " + result.to_str()) }
        results.push(result)
    }
    return results
}

fn _examples_of(mod: ModuleDoc) -> list[tuple[str, Example]] {
    let found: list[tuple[str, Example]] = []
    let doc = mod.doc
    if doc is not null {
        for example in doc.examples { found.push((mod.name, example)) }
    }
    for entry in mod.entries {
        let entry_doc = entry.doc
        if entry_doc is not null {
            for example in entry_doc.examples { found.push((entry.name, example)) }
        }
        for member in entry.members {
            let member_doc = member.doc
            if member_doc is null { continue }
            for example in member_doc.examples {
                found.push((entry.name + "." + member.name, example))
            }
        }
    }
    return found
}

fn _names_used(lines: list[str]) -> set[str] {
    let used: set[str] = set()
    var previous = TokenKind.Eof
    let lexer = Lexer("\n".join(lines) + "\n", 0, DiagnosticBag())
    for token in lexer.tokenize() {
        let member = previous == TokenKind.Dot or previous == TokenKind.QuestionDot
        if token.kind == TokenKind.Ident and not member { used.add(token.word) }
        previous = token.kind
    }
    return used
}

fn _is_import(code: str) -> bool {
    return code.starts_with("import ") or code.starts_with("from ")
}

fn _import_bindings(
    mod: ModuleDoc,
    imports: list[str],
    used: set[str]
) -> list[tuple[str, list[str]]] {
    if imports.len() == 0 { return [] }
    let bindings: list[tuple[str, list[str]]] = []
    let program = parse_module("\n".join(imports) + "\n", "<imports>")
    for statement in program.child_list("stmts") {
        let path = _absolute_path(statement.text("path"), mod)
        if statement.kind == NodeKind.Import {
            let parts = path.split(".")
            let alias = statement.opt_text("alias")
            let bound = alias ?? parts[parts.len() - 1]
            if bound not in used { continue }
            let tail = if alias is null { "" } else { " as " + alias }
            bindings.push((f"import {path}" + tail, [bound]))
            continue
        }
        if statement.kind != NodeKind.FromImport { continue }
        let names: list[str] = []
        let spellings: list[str] = []
        for item in statement.records("items") {
            let bound = item["alias"] ?? item["name"]
            if bound not in used { continue }
            names.push(bound)
            spellings.push(if item["alias"] is null {
                item["name"]
            } else {
                item["name"] + " as " + item["alias"]
            })
        }
        if names.len() == 0 { continue }
        bindings.push((f"from {path} import " + ", ".join(spellings), names))
    }
    return bindings
}

fn _absolute_path(path: str, mod: ModuleDoc) -> str {
    if not path.starts_with(".") { return path }
    var depth = 0
    while depth < path.len() and path[depth] == "." { depth += 1 }

    let parts = mod.name.split(".")
    if not mod.path.ends_with("mod.jai") and parts.len() > 0 { parts.pop() }
    for _step in 1..depth {
        if parts.len() > 0 { parts.pop() }
    }
    let tail = path[depth:path.len()]
    if tail.len() > 0 { parts.push(tail) }
    return ".".join(parts)
}

fn _imports_of(path: str) -> list[str] {
    let program = parse_module(read_file(path), path)
    let out: list[str] = []
    for statement in program.child_list("stmts") {
        if statement.kind == NodeKind.Import or statement.kind == NodeKind.FromImport {
            out.push(to_source(statement))
        }
    }
    return out
}

# --- Reporting ---------------------------------------------------------------
fn _file_line(path: str, results: list[TestResult]) -> str {
    var passed = 0
    var failed = 0
    for result in results {
        if result.outcome.is_passed() { passed += 1 }
        if result.outcome.is_failure() { failed += 1 }
    }
    let counts = if failed == 0 {
        green(f"{passed} ok")
    } else {
        red(f"{passed} ok, {failed} failed")
    }
    return module_name_of(path) + "  " + counts
}

class _Options {
    pub var verbose: bool
    pub var doc: bool
    pub var filter: str?
    pub var paths: list[str]
    pub var usage_error: str?
    #: How many files to run at once. 0 means "pick one per core"; 1 keeps the
    #: old single-process behaviour, which is what a worker runs under.
    pub var jobs: int
    #: Set on a worker. It runs exactly one file and prints its results as
    #: `@@R` lines for the parent to parse, with no file line and no summary.
    pub var shard: bool

    pub fn init(self) {
        self.verbose = false
        self.doc = false
        self.filter = null
        self.paths = []
        self.usage_error = null
        self.jobs = 0
        self.shard = false
    }

    pub static fn parse(args: list[str]) -> _Options {
        let options = _Options()
        let reader = Arguments(args)
        loop {
            let argument = reader.next()
            if argument is null { break }
            if not argument.is_flag() {
                options.paths.push(argument.text)
                continue
            }
            if argument.name == "--filter" or argument.name == "-k" {
                let value = reader.value_of(argument)
                if value is null { break }
                options.filter = value
            } elif argument.name == "--verbose" or argument.name == "-v" {
                if reader.is_bare(argument) { options.verbose = true }
            } elif argument.name == "--doc" {
                if reader.is_bare(argument) { options.doc = true }
            } elif argument.name == "--jobs" or argument.name == "-j" {
                let value = reader.value_of(argument)
                if value is null { break }
                options.jobs = int(value)
            } elif argument.name == "--_shard" {
                if reader.is_bare(argument) { options.shard = true }
            } else {
                reader.unknown(argument)
            }
        }
        options.usage_error = reader.error
        if options.paths.len() == 0 {
            options.paths = if options.doc { ["lib"] } else { ["tests"] }
        }
        return options
    }
}
