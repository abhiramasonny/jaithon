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
from jaithon.compile.token import Token, TokenKind
from std.fmt import bold, green, red
from std.io import Path, print_err, read_bytes, read_file
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

#: JAITHON_TEST_SCAN=0 finds test names by parsing the file, as it used to;
#: read once.
let _SCAN_NAMES: bool = (env("JAITHON_TEST_SCAN") ?? "1") != "0"

#: The names `test_functions` finds, read off the tokens alone: a `fn` outside
#: every brace, named like a test, whose parameter list is empty. Null when the
#: source does not even lex, so the caller can fall back to the parse.
#:
#: ```
#: _scan_test_names("fn test_a() -> void { fn test_b() { } }\nfn test_c(x: int) { }\n")
#: #     ["test_a"]
#: ```
pub fn _scan_test_names(source: str) -> list[str]? {
    let lexer = Lexer(source, 0, DiagnosticBag())
    let tokens = lexer.tokenize()
    if lexer.had_error { return null }
    let names: list[str] = []
    let count = tokens.len()
    var depth = 0
    var index = 0
    while index < count {
        let kind = tokens[index].kind
        if kind == TokenKind.LBrace {
            depth += 1
        } elif kind == TokenKind.RBrace {
            depth -= 1
        } elif kind == TokenKind.KwFn and depth == 0 and index + 1 < count {
            let name_token = tokens[index + 1]
            if name_token.kind == TokenKind.Ident and is_test_name(name_token.word) {
                if _no_parameters(tokens, index + 2) { names.push(name_token.word) }
            }
        }
        index += 1
    }
    return names
}

#: Whether the parameter list starting at `tokens[at]` (after any generic
#: brackets) is empty.
fn _no_parameters(tokens: list[Token], start: int) -> bool {
    let count = tokens.len()
    var at = start
    if at < count and tokens[at].kind == TokenKind.LBracket {
        var nesting = 0
        while at < count {
            let kind = tokens[at].kind
            if kind == TokenKind.LBracket { nesting += 1 }
            if kind == TokenKind.RBracket {
                nesting -= 1
                if nesting == 0 { break }
            }
            at += 1
        }
        at += 1
    }
    if at >= count or tokens[at].kind != TokenKind.LParen { return false }
    at += 1
    while at < count and tokens[at].kind == TokenKind.Newline { at += 1 }
    return at < count and tokens[at].kind == TokenKind.RParen
}

#: A `parse` result for `path` when it does not parse, else nothing: the
#: answer the runner gave before it found names without parsing.
fn _unless_unparseable(source: str, path: str, suite: str) -> list[TestResult] {
    if not _SCAN_NAMES { return [] }
    try {
        let _names = test_functions(source, path)
    } catch error: ParseError {
        return [_broken(suite, "parse", error.message)]
    }
    return []
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
    #: The names come from the tokens when they can: the parse that used to
    #: find them was the second full compile of every file. Anything the scan
    #: cannot settle -- no test wanted, or a file that will not load -- goes
    #: back to the parse, so what is reported is the same either way.
    var names = if _SCAN_NAMES { _scan_test_names(source) } else { null }
    if names is null {
        try {
            names = test_functions(source, path)
        } catch error: ParseError {
            return [_broken(suite, "parse", error.message)]
        }
    }

    let wanted = [name for name in names ?? [] if _matches(suite, name, filter)]
    if wanted.len() == 0 { return _unless_unparseable(source, path, suite) }

    let namespace = f"<test {path}>"
    try {
        __prim__.exec(source, namespace)
    } catch error: Error {
        let unparseable = _unless_unparseable(source, path, suite)
        if unparseable.len() > 0 { return unparseable }
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
        let process = spawn(_worker_command(exe, path, options))
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
#: The tool's own flags come from `options`: argv() here is only the binary,
#: so a worker rebuilt from it ran without `--doc` or `--filter` -- a parallel
#: `test --doc` found no tests at all, and `--filter` filtered nothing. What
#: argv() does carry past the binary is still passed on, so a CLI flag that
#: ever shows up there reaches the worker too.
fn _worker_command(exe: str, path: str, options: _Options) -> list[str] {
    let own = argv()
    var command = [exe]
    #: A forked worker inherits this process's collector; one started from
    #: scratch only knows the cadence if it is told, and without it a stress
    #: run ran every such file (all of the GPU ones) unstressed.
    let every = _stress_every()
    if every > 0 { command.push(f"--gc-stress={every}") }
    command.extend(["test", "--_shard", "--jobs=1"])
    if options.doc { command.push("--doc") }
    let filter = options.filter
    if filter is not null { command.push("--filter=" + filter) }
    var index = 1
    while index < own.len() {
        let argument = own[index]
        #: `test` itself, the paths, and the flags rebuilt above.
        let is_path = not argument.starts_with("-")
        let is_jobs = argument.starts_with("--jobs") or argument.starts_with("-j")
        let is_own = argument == "--doc" or argument.starts_with("--filter") or argument == "-k"
        if not is_path and not is_jobs and not is_own { command.push(argument) }
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
        if complaints.len() > 0 {
            #: Standard output is buffered and standard error is not: without
            #: the flush, a reader of both through one pipe (run_tests.sh uses
            #: 2>&1) got a warning spliced into the middle of a result line,
            #: wherever the buffer happened to fill, and misnamed that test.
            __prim__.io_flush(1)
            print_err(complaints.rstrip())
        }
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
    let scratch = _scratch_dir()
    if scratch is null {
        #: Nowhere to put the children's output -- TMPDIR missing, read-only
        #: or full. Workers started from scratch report over pipes and need no
        #: files, so run every file that way instead.
        return _run_sharded(exe, paths, options, limit)
    }
    _warm_for_children()
    let history = _History.load(paths, options)
    let order = history.slowest_first(paths)
    var captured: dict[str, str] = {}
    var noise: dict[str, str] = {}
    var failures: dict[str, int] = {}
    var running: dict[int, int] = {}
    var started: dict[int, int] = {}
    var spawned: set[int] = set()
    #: Files whose forked child exited with _NEEDS_EXEC in this run.
    var refused: set[int] = set()
    var again: list[int] = []
    #: The output files go whatever happens here; a runner that died part way
    #: used to leave its directory under TMPDIR behind.
    try {
        var next_order = 0
        while next_order < order.len() or again.len() > 0 or running.len() > 0 {
            while running.len() < limit and (again.len() > 0 or next_order < order.len()) {
                let from_queue = again.len() == 0
                let index = if from_queue { order[next_order] } else { again.pop() }
                if from_queue { next_order += 1 }
                let base = f"{scratch}/{index}"
                started[index] = __prim__.time_mono()
                if not from_queue or history.needs_exec(paths[index]) {
                    let command = _worker_command(exe, paths[index], options)
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
                refused.add(index)
                again.push(index)
                continue
            }
            let path = paths[index]
            let base = f"{scratch}/{index}"
            history.record_time(path, __prim__.time_mono() - (started.get(index) ?? 0))
            if index in refused {
                history.needed_exec(path)
            } elif index in spawned {
                history.spawned_on_record(path)
            } else {
                history.forked(path)
            }
            captured[path] = _take_file(base + ".out")
            noise[path] = _take_file(base + ".err")
            if done[1] != 0 { failures[path] = done[1] }
        }
    } finally {
        _remove_quietly(Path(scratch), true)
    }
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
    #: Files that loaded a native framework, each with the number of runs
    #: since a forked child last proved it: they skip the fork until that
    #: reaches _EXEC_RECHECK, so a file that stops needing a framework goes
    #: back to forking instead of being spawned forever.
    var exec_only: dict[str, int]
    var dirs: dict[str, bool]
    #: `test_times`, or `doc_times` for `--doc`: a doc run times other work.
    var name: str
    #: A filtered run times only part of each file, and a forked child that
    #: finished may just have skipped the test that loads a framework. Such a
    #: run only adds what it learns about frameworks; it used to overwrite the
    #: record with near-zero times and lose the slowest-first order.
    var partial: bool

    pub fn init(self) {
        self.times = {}
        self.exec_only = {}
        self.dirs = {}
        self.name = "test_times"
        self.partial = false
    }

    pub static fn load(paths: list[str], options: _Options) -> _History {
        let history = _History()
        if options.doc { history.name = "doc_times" }
        history.partial = options.filter is not null
        for path in paths {
            let dir = Path(path).parent().text
            if dir in history.dirs { continue }
            history.dirs[dir] = false
            try {
                let file = Path(history.file_of(dir))
                if not file.exists() { continue }
                for line in file.read_text().split("\n") {
                    let fields = line.split("\t")
                    if fields.len() != 3 { continue }
                    let full = dir + "/" + fields[2]
                    history.times[full] = int(fields[0])
                    let flag = fields[1]
                    if flag.starts_with("x") {
                        history.exec_only[full] = if flag.len() > 1 { int(flag[1:]) } else { 0 }
                    }
                }
            } catch _error: Error {
            }
        }
        return history
    }

    fn file_of(self, dir: str) -> str {
        return Path(dir).join("__jaicache__").join(self.name).text
    }

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

    pub fn needs_exec(self, path: str) -> bool {
        let runs = self.exec_only.get(_History.key(path))
        return runs is not null and runs < _EXEC_RECHECK
    }

    #: Built the same way from a path as from a record, so the two always
    #: match however the path was spelled (`a//b`, `./a`).
    static fn key(path: str) -> str {
        let file = Path(path)
        return file.parent().text + "/" + file.name()
    }

    pub fn record_time(self, path: str, nanos: int) -> void {
        if self.partial { return }
        self.times[_History.key(path)] = nanos
        self.touch(path)
    }

    #: Its forked child exited with _NEEDS_EXEC: proof it needs a fresh process.
    pub fn needed_exec(self, path: str) -> void {
        self.exec_only[_History.key(path)] = 0
        self.touch(path)
    }

    #: Spawned without trying a fork, on the strength of the record alone.
    pub fn spawned_on_record(self, path: str) -> void {
        let key = _History.key(path)
        self.exec_only[key] = (self.exec_only.get(key) ?? 0) + 1
        self.touch(path)
    }

    #: Its forked child ran to the end without loading a framework.
    pub fn forked(self, path: str) -> void {
        let key = _History.key(path)
        if self.partial or key not in self.exec_only { return }
        self.exec_only.remove(key)
        self.touch(path)
    }

    fn touch(self, path: str) -> void { self.dirs[Path(path).parent().text] = true }

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
                let runs = self.exec_only.get(key)
                let flag = if runs is null { "-" } else { f"x{runs}" }
                lines.push(f"{nanos}\t{flag}\t{name}")
            }
            try {
                let cache = Path(dir).join("__jaicache__")
                cache.mkdir(false, true)
                let target = self.file_of(dir)
                let staging = target + "." + str(__prim__.time_wall())
                Path(staging).write_text("\n".join(lines.sorted()) + "\n")
                __prim__.io_rename(staging, target)
            } catch _error: Error {
            }
        }
    }
}

#: The status a forked child exits with when it was about to load a native
#: framework. Frameworks that talk to a system service over XPC refuse a
#: process that was forked and not exec'd -- Metal's shader compiler reports
#: "the process is unavailable" and every GPU test fails -- so the loader
#: stops the child before it tries (JAI_EXIT_NEEDS_EXEC in native.h).
const _NEEDS_EXEC: int = 75

#: How many runs a file recorded as needing a fresh process is spawned
#: straight away before a fork is tried again. A wasted fork costs a fraction
#: of a second; a stale record costs a spawn's start-up on every run.
const _EXEC_RECHECK: int = 16

#: The collector's stress cadence in this process, 0 when not stressed.
fn _stress_every() -> int {
    let stats: dict[str, any] = __prim__.gc_stats()
    if stats.get("stress") != true { return 0 }
    let every = stats.get("stress_every")
    return if every is int { every } else { 1 }
}

#: The child's whole life: run one file, print its `@@R` lines, exit. It must
#: never return into the parent's loop, whatever goes wrong.
fn _run_child(path: str, options: _Options) -> void {
    var code = 70
    #: The exit is in a `finally` so that nothing, caught or not, can carry
    #: this child up into the parent's loop -- where it would run the parent's
    #: cleanup and delete the directory every other child is writing to.
    try {
        let found = if options.doc {
            run_doc_examples(path, options.filter, false)
        } else {
            run_file(path, options.filter, false)
        }
        _emit_shard(found)
        code = 0
    } catch error: Error {
        print_err(f"jaithon test: {path}: {type_of(error)}: {error.message}")
    } finally {
        __prim__.os_exit(code)
    }
}

#: Have the checker parse `std.test`'s signatures once, here, so every child
#: inherits them instead of parsing them again. Only signatures: running a
#: test module's imports here would share their state between files.
fn _warm_for_children() -> void {
    try {
        __prim__.exec("from std.test import assert_eq\nlet _ = assert_eq\n", "<test warm-up>")
    } catch _error: Error {
    }
}

const _SCRATCH_ATTEMPTS: int = 16

#: A fresh directory under the temp dir for the children's output files, or
#: null when none can be made. Bounded: a missing or read-only temp dir fails
#: every attempt, and retrying without end hung the runner with no output.
fn _scratch_dir() -> str? {
    let base = temp_dir() + "/jaithon-test-" + str(__prim__.time_wall())
    for attempt in 0.._SCRATCH_ATTEMPTS {
        let candidate = if attempt == 0 { base } else { f"{base}-{attempt}" }
        if Path(candidate).exists() { continue }
        try {
            Path(candidate).mkdir()
            return candidate
        } catch _error: Error {
        }
    }
    return null
}

fn _take_file(path: str) -> str {
    let file = Path(path)
    if not file.exists() { return "" }
    let text = _text_of(file)
    _remove_quietly(file, false)
    return text
}

#: A child's output as text. A test may print bytes that are not UTF-8; then
#: every byte past ASCII shows as U+FFFD, rather than a decoding error taking
#: the whole runner down with every other file's results.
fn _text_of(file: Path) -> str {
    try {
        return file.read_text()
    } catch _error: Error {
    }
    let shown: list[str] = []
    for byte in read_bytes(file.text).to_list() {
        shown.push(if byte < 128 { chr(byte) } else { "\u{FFFD}" })
    }
    return "".join(shown)
}

fn _remove_quietly(path: Path, recursive: bool) -> void {
    try {
        path.remove(recursive)
    } catch _error: Error {
    }
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
    #: `--gc-stress` is the CLI's, and it never reached this tool's argv(),
    #: which holds only the binary: the check here read argv() and was always
    #: false, so a stress run went parallel and its spawned workers -- started
    #: without the flag -- ran every test unstressed (tests/stdlib at
    #: `--gc-stress=20`: 2.4s; actually stressed, 75s). The collector's own
    #: state says whether this process is stressed. A forked worker inherits
    #: it, and runs with a heap of its own file's size, so a stress run stays
    #: parallel there; a spawned one would not, so without forking it is
    #: sequential, as it was always meant to be.
    let stats: dict[str, any] = __prim__.gc_stats()
    let stressed = stats.get("stress") == true
    let limit = if stressed and not _FORK_WORKERS {
        1
    } elif wanted > 0 {
        wanted
    } else {
        min(cpu_count(), paths.len())
    }

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
