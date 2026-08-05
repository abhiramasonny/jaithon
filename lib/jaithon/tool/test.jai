#: The test runner behind `jaithon test`.
#:
#: Discovery is by convention and nothing else: walking a directory picks up
#: the files whose name begins with `test_`, and a test is a top-level function
#: taking no arguments whose name also begins with `test_`. There is no
#: registration list to forget to add to. A path named on the command line is
#: run whatever it is called — naming a file means wanting that file.
#:
#: Each file is parsed before it is run, so the runner knows which functions it
#: is about to call and can report `path: no tests` instead of executing a file
#: for nothing. The functions themselves are run through `std.test`, which turns
#: an `AssertionError` into a failure, any other exception into an error, and
#: either way keeps going — one broken test never ends the run.
#:
#: `--doc` runs the examples in `#:` comments. Every fenced block in a doc
#: comment becomes a test: the lines run in order, and a line with a trailing
#: `# value` comment becomes an assertion that the line evaluates to that value.
#: That is what keeps `spec/STYLE.md` §6's promise that the examples are
#: correct — they are executed, and a stale one fails the build.
#:
#: ```
#: expected_of("f(1)   # 2")          # "2"
#: expected_of("f(1)")                # null
#: is_test_name("test_adds")          # true
#: is_test_name("tester")             # false
#: ```
from jaithon.ast import NodeKind, to_source
from jaithon.compile.lexer import DiagnosticBag, Lexer, TokenKind
from jaithon.compile.parser import parse
from std.fmt import bold, green, red
from std.io import Path, print_err, read_file
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
#: The rule is the file name and nothing else. "Anything under `tests`" is the
#: obvious alternative and it is wrong: `tests` also holds the benchmarks and
#: the diagnostic fixtures, and the fixtures do not compile *on purpose* — a
#: runner that walks into them reports the suite red for files that are doing
#: their job. A path named on the command line is run whatever it is called.
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
#: The file is executed once, in a namespace of its own, and then each test
#: function is called by name. `filter`, when given, keeps only the tests whose
#: `module:name` contains it.
#:
#: The namespace matters: a test file's own imports must not reach the runner.
#: Run in this module's namespace, `import std.str as str` inside a test file
#: rebinds the runner's `str` and the next report the runner formats fails on
#: it. It also keeps two test files from seeing each other's bindings, which is
#: what makes the order files are run in stop mattering.
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
#: The example runs with the module's own imports in scope, plus everything the
#: module exports — when the module is importable under its own name, which is
#: to say it lives under `lib/` — so an example may name what its module names
#: whether that module is `std.list` or `jaithon.ast`. The lines
#: themselves go inside an anonymous function that is called immediately: that
#: keeps one example's `let` bindings from colliding with the next one's.
#:
#: The module's own surface is imported by listing its public names, not as
#: `from m import *`: the wildcard is in the grammar but has no bytecode form
#: (`E0902` from `emitFromImport`), so every example built on one failed to
#: compile and `--doc` found no tests at all.
#:
#: Only the names the example actually mentions are imported, which is what
#: keeps the program quiet: an import nothing uses is `W0100`, and importing a
#: whole module's surface for a two-line example would bury the example's own
#: output under a hundred warnings. It also settles the collision — a name the
#: module already imported is never imported twice, and a name declared twice in
#: one scope is `E0302`.
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
        # An example may open with an import of its own, and it belongs at the
        # top of the program rather than inside the wrapper: what a reader has
        # to type to run the example is exactly what the example shows.
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

    # The body, not the example: a name may appear only in an expected-value
    # comment — `kind_from_name("AST_CALL")   # NodeKind.Call` — and the comment
    # is where the tokenizer stops looking. By the time the body is built that
    # comment has become the right-hand side of an assertion, and the name is
    # code like any other.
    let used = _names_used(body)
    let bound: set[str] = set()
    let head: list[str] = []
    if asserted {
        head.push("from std.test import assert_eq")
        bound.add("assert_eq")
    }
    # The example's own imports first: what it asked for by name wins over what
    # the module happens to import under the same name.
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
    # Only when the module is importable under its own name. `module_name_of`
    # returns the dotted path for a file under `lib/` — `std.list`,
    # `jaithon.ast` — and a bare stem for anything else; a test or example file
    # names no module, and importing from its stem would be `E0800`.
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
#: With no path, `tests` is searched. `--doc` runs the examples in `#:` comments
#: instead of the `test_` functions; give both `--doc` and paths under `lib` to
#: check the library's documentation. The exit status is 1 when anything failed,
#: which is what the CI gate reads.
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

# Whether `text` is one expression, and so can stand on either side of an
# assertion. Doc comments end lines with prose as often as with a value, and
# `# true: a file with an error has no tree` must not become an assertion.
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
        # A namespace of its own, for the reason `run_file` gives: an example
        # of `std.json` opens with `from std.json import parse`, and run in
        # this module's namespace that rebinds the runner's own `parse` — the
        # next example to reach `_is_expression` then dies on it, and the run
        # ends wherever the alphabet put the module that did it.
        let namespace = f"<doc {mod.name}:{example.line}>"
        let result = Test(name, fn() -> any { return __prim__.exec(program, namespace) }).run(suite)
        if verbose { print("  " + result.to_str()) }
        results.push(result)
    }
    return results
}

# Every example in a module, paired with the name of what it documents.
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

# Every identifier an example mentions. The example is tokenized rather than
# searched for substrings, so a name inside a string literal or a comment is
# not mistaken for a use of it — that is the difference between importing what
# the example needs and importing whatever happens to be spelled in it.
#
# A lexical error costs the tokens after it and nothing else: an example that
# does not lex will not compile either, and the failure the runner wants to
# report is that one, not a missing import.
fn _names_used(lines: list[str]) -> set[str] {
    let used: set[str] = set()
    var previous = TokenKind.Eof
    let lexer = Lexer("\n".join(lines) + "\n", 0, DiagnosticBag())
    for token in lexer.tokenize() {
        # `summary.mean()` is not a use of the free function `mean`: a name
        # after a dot is a member of whatever precedes it, and importing it
        # would leave an import nothing uses.
        let member = previous == TokenKind.Dot or previous == TokenKind.QuestionDot
        if token.kind == TokenKind.Ident and not member { used.add(token.text) }
        previous = token.kind
    }
    return used
}

# Whether `code` is an import statement, which the wrapper has to hoist.
fn _is_import(code: str) -> bool {
    return code.starts_with("import ") or code.starts_with("from ")
}

# The module's import statements cut down to the names `used` mentions, each
# paired with the names it goes on to bind.
#
# The cut is per name and not per statement because the statements are written
# for the module, not for the example: `from std.iter import iter, cycle` is
# one statement, and an example that says `iter` and not `cycle` still has to
# not import `cycle`. `import std.math` binds `math` — the last component of
# the path — unless it was given an alias, and there is nothing to cut down.
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

# A relative import path, spelled absolutely. The example does not run in the
# module's file, so `.graph` has nothing to be relative to and the loader looks
# for it beside the working directory instead. A `mod.jai` file *is* its
# package, so one leading dot already names a child of it; anything else is a
# module inside a package, and one dot means a sibling.
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

# The module's own import statements, as source, to put back around an example.
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

    pub fn init(self) {
        self.verbose = false
        self.doc = false
        self.filter = null
        self.paths = []
        self.usage_error = null
    }

    pub static fn parse(args: list[str]) -> _Options {
        let options = _Options()
        let reader = Arguments(args)
        while true {
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
            } else {
                reader.unknown(argument)
            }
        }
        options.usage_error = reader.error
        # Two jobs, two defaults. `test_` functions live under `tests`, but the
        # examples `--doc` runs are the ones `jaithon doc` publishes, so the
        # default there has to be the default of `jaithon.tool.doc` — pointed at
        # `tests` it found nothing and reported "no tests found" for a tree
        # holding seven hundred examples.
        if options.paths.len() == 0 {
            options.paths = if options.doc { ["lib"] } else { ["tests"] }
        }
        return options
    }
}
