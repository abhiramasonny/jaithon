#: The `jaithon fmt` command line: which files to visit and what to do with them.
from jaithon.tool.args import Arguments
from std.io import Path, print_err, read_file, write_file
from .diff import check_report, unified_diff
from .format import _format
from .migrate import migrate2

#: Expand command-line paths into the `.jai` files they name.
#:
#: A file argument is taken as given (not filtered by `keep`); a directory
#: argument is walked and filtered. Raises `FileNotFoundError` for a missing
#: argument — silently processing nothing is how a misspelled path passes CI.
pub fn expand_paths(inputs: list[str], keep: fn(Path) -> bool = _every_file) -> list[str] {
    let found: list[str] = []
    let seen: set[str] = set()
    for input in inputs {
        let path = Path(input)
        if path.is_dir() {
            for entry in path.walk() {
                if entry.is_dir() or entry.extension() != "jai" { continue }
                if not keep(entry) { continue }
                if entry.text not in seen {
                    seen.add(entry.text)
                    found.push(entry.text)
                }
            }
        } elif path.exists() {
            if path.text not in seen {
                seen.add(path.text)
                found.push(path.text)
            }
        } else {
            throw FileNotFoundError(f"no such file or directory: {input}")
        }
    }
    return found.sorted()
}

#: Drop the leading subcommand or script path from `args`.
#:
#: Shared by the four `jaithon.tool` entry points: the CLI hands a subcommand
#: its own name (or a script its path) in `args[0]`.
#:
#: ```
#: tail_arguments(["fmt", "--check"], "fmt")          # ["--check"]
#: tail_arguments(["--check"], "fmt")                 # ["--check"]
#: tail_arguments(["lib/jaithon/tool/fmt.jai"], "fmt")    # []
#: tail_arguments(["a.jai"], "fmt")                   # ["a.jai"]
#: ```
pub fn tail_arguments(args: list[str], command: str) -> list[str] {
    if args.len() == 0 { return args }
    if args[0] == command { return args[1:args.len()] }
    if args[0].ends_with("/" + command + ".jai") { return args[1:args.len()] }
    return args
}

#: Run `jaithon fmt`.
#:
#: ```text
#: jaithon fmt [--check] [--diff] [--migrate2] [paths...]
#: ```
#:
#: `--check` reports each unformatted file and exits 1 (the CI gate); `--diff`
#: prints a unified diff instead of writing; `--migrate2` runs the Jaithon 2
#: rewrite first. Exit status: 0 clean, 1 for `--check` findings, 2 on parse
#: failure.
pub fn main(args: list[str]) -> int {
    let options = _Options.parse(tail_arguments(args, "fmt"))
    if options.usage_error is not null {
        print_err(f"jaithon fmt: {options.usage_error}")
        print_err("usage: jaithon fmt [--check] [--diff] [--migrate2] [paths...]")
        return 2
    }

    var failed = 0
    var unformatted = 0
    for path in expand_paths(options.paths) {
        let source = read_file(path)
        let input = if options.migrate { migrate2(source) } else { source }
        var output = input
        try {
            let (formatted, lost) = _format(input, path)
            if lost.len() > 0 {
                print_err(
                    f"{path}:{lost[0].line}: the formatter cannot place " +
                        f"{lost.len()} comment(s); the file is left unchanged"
                )
            } else {
                output = formatted
            }
        } catch error: ParseError {
            if options.migrate {
                print_err(f"{path}: migrated but not formatted: {error.message}")
                if not options.check and not options.diff { write_file(path, input) }
            } else {
                print_err(f"{path}: {error.message}")
            }
            failed += 1
            continue
        }
        if output == source { continue }
        unformatted += 1
        if options.check { print(check_report(source, output, path)) }
        if options.diff { print(unified_diff(source, output, path)) }
        if not options.check and not options.diff {
            write_file(path, output)
            print(path)
        }
    }

    if failed > 0 { return 2 }
    if options.check and unformatted > 0 { return 1 }
    return 0
}

fn _every_file(path: Path) -> bool { return not is_diagnostic_fixture(path) }

#: Report whether `path` is a diagnostic fixture: a file whose first line reads
#: `# expect: Exxxx`, declaring the error it exists to provoke.
#:
#: Invalid on purpose, so a directory walk skips it; naming one explicitly
#: still processes it.
#:
#: ```
#: is_diagnostic_fixture(Path("tests/errors/unclosed_brace.jai"))    # true
#: is_diagnostic_fixture(Path("lib/std/str.jai"))                    # false
#: ```
pub fn is_diagnostic_fixture(path: Path) -> bool {
    let first = _first_line_of(read_file(path.text))
    if not first.starts_with("#") { return false }
    let body = first[1:first.len()].strip()
    if not body.starts_with("expect:") { return false }
    let code = body[7:body.len()].strip()
    return code.len() >= 5 and (code.starts_with("E") or code.starts_with("W"))
}

fn _first_line_of(source: str) -> str {
    let stop = source.find("\n")
    return if stop < 0 { source } else { source[0:stop] }
}

class _Options {
    pub var check: bool
    pub var diff: bool
    pub var migrate: bool
    pub var paths: list[str]
    pub var usage_error: str?

    pub fn init(self) {
        self.check = false
        self.diff = false
        self.migrate = false
        self.paths = []
        self.usage_error = null
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
            if argument.name == "--check" {
                if reader.is_bare(argument) { options.check = true }
            } elif argument.name == "--diff" {
                if reader.is_bare(argument) { options.diff = true }
            } elif argument.name == "--migrate2" {
                if reader.is_bare(argument) { options.migrate = true }
            } else {
                reader.unknown(argument)
            }
        }
        options.usage_error = reader.error
        if options.paths.len() == 0 { options.paths = ["."] }
        return options
    }
}
