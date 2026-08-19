#: The benchmark harness behind `jaithon bench`.
#:
#: A file with top-level `bench_` functions is measured one function at a
#: time; a file with only `main` is run as a whole program instead — what
#: every `tests/bench` pair is, so it stays comparable to its Python peer.
#:
#: Discovery is `tests/bench` only: the rest of `tests` holds files (like
#: `tests/errors`) that don't parse *on purpose*, and walking into them would
#: report failures for files doing their job.
#:
#: Calibration lifts a sample above the clock's resolution before timing it.
#: Both mean and median are reported since a timing distribution is always
#: skewed — an interrupted run inflates the mean, so the median, not the
#: mean, is what a regression is measured against.
#:
#: With a baseline present, any benchmark slower than its recorded median by
#: more than the threshold is a regression, and exits non-zero.
#:
#: ```
#: is_bench_name("bench_sort")    # true
#: is_bench_name("benchmark")     # false
#: ```
from jaithon.ast import Node, NodeKind
from std.fmt import Align, Table, TableStyle, bold, dim, fixed, green, red
from std.io import Path, print_err, read_file
from std.os import argv as os_argv, run, which
from std.str import to_float
from std.json import parse as parse_json, stringify
from std.test import BenchmarkResult, benchmark
from std.time import Instant
from .args import Arguments
from .doc import module_name_of
from .fmt import expand_paths, parse_module, tail_arguments

#: The prefix a function's name must have to be measured.
pub const BENCH_PREFIX: str = "bench_"

#: The file `--baseline` defaults to.
pub const DEFAULT_BASELINE: str = "bench.baseline.json"

#: How much slower than its baseline a benchmark may be before it is reported
#: as a regression.
pub const DEFAULT_THRESHOLD: float = 0.10

#: Roughly how long, in seconds, each benchmark is given.
pub const DEFAULT_BUDGET: float = 1.0

#: The shortest sample the harness will accept, in seconds.
#:
#: A sample below this is mostly clock resolution, so the batch size is raised
#: until a sample clears it.
pub const MIN_SAMPLE_SECONDS: float = 0.001

const _MIN_SAMPLES = 10
const _MAX_SAMPLES = 200
const _MAX_BATCH = 10_000_000
const _BASELINE_VERSION = 1

const _MIN_PROGRAM_SAMPLES = 5
const _MAX_PROGRAM_SAMPLES = 50

#: Report whether `name` names a benchmark.
#:
#: ```
#: is_bench_name("bench_fib")    # true
#: is_bench_name("bench_")       # false
#: ```
pub fn is_bench_name(name: str) -> bool {
    return name.starts_with(BENCH_PREFIX) and name.len() > BENCH_PREFIX.len()
}

#: Return the names of the benchmark functions declared at the top of `source`.
#:
#: ```
#: bench_functions("fn bench_a() -> void { }\nfn other() -> void { }\n", "b.jai")
#: #     ["bench_a"]
#: ```
pub fn bench_functions(source: str, path: str) -> list[str] {
    return _bench_functions_in(parse_module(source, path))
}

#: Report whether `source` declares a program entry point.
#:
#: A file with no `bench_` function but a `main` is still a benchmark, run
#: as a whole program — what `tests/bench` pairs need to stay comparable
#: with their Python peer.
#:
#: ```
#: declares_main("fn main() -> int { return 0 }\n", "b.jai")    # true
#: declares_main("fn other() -> int { return 0 }\n", "b.jai")   # false
#: ```
pub fn declares_main(source: str, path: str) -> bool {
    return _declares_main_in(parse_module(source, path))
}

fn _bench_functions_in(program: Node) -> list[str] {
    let names: list[str] = []
    for statement in program.child_list("stmts") {
        if statement.kind != NodeKind.FnDecl { continue }
        let name = statement.opt_text("name")
        if name is null or not is_bench_name(name) { continue }
        if statement.records("params").len() > 0 { continue }
        names.push(name)
    }
    return names
}

fn _declares_main_in(program: Node) -> bool {
    for statement in program.child_list("stmts") {
        if statement.kind != NodeKind.FnDecl { continue }
        if statement.opt_text("name") == "main" { return true }
    }
    return false
}

#: Choose the batch size and sample count for `body` within `budget` seconds.
#:
#: The batch lifts a sample above clock resolution; the count is what the
#: budget affords, never fewer than ten (a stddev over three samples isn't a
#: measurement) and never more than two hundred.
pub fn calibrate(body: fn() -> any, budget: float) -> tuple[int, int] {
    body()
    let start = Instant.now()
    body()
    let single = max(start.elapsed().to_seconds(), 1e-9)

    var batch = int(MIN_SAMPLE_SECONDS / single)
    if batch < 1 { batch = 1 }
    if batch > _MAX_BATCH { batch = _MAX_BATCH }

    let per_sample = single * batch
    var samples = int(budget / per_sample)
    if samples < _MIN_SAMPLES { samples = _MIN_SAMPLES }
    if samples > _MAX_SAMPLES { samples = _MAX_SAMPLES }
    return (batch, samples)
}

#: Measure `body` and return its timings.
#:
#: Calibrates, warms up and then samples, so the caller supplies a time budget
#: rather than a sample count it would have to guess.
pub fn measure(name: str, body: fn() -> any, budget: float = DEFAULT_BUDGET) -> BenchmarkResult {
    let (batch, samples) = calibrate(body, budget)
    return benchmark(name, body, samples, batch: batch)
}

#: Measure `command` by running it to completion, repeatedly.
#:
#: Each sample is one process, so the budget only buys sample count (floor
#: five, so a median and spread exist). The first run is an unrecorded
#: warm-up — it pays for the page and bytecode cache, and charging that to
#: every sample would overstate them all by the same wrong amount.
#:
#: Raises `OSError` when the program fails, since a benchmark that computes
#: nothing is arbitrarily fast.
pub fn measure_program(
    name: str,
    command: list[str],
    budget: float = DEFAULT_BUDGET
) -> BenchmarkResult {
    let warm = _time_process(command)
    var samples = int(budget / max(warm, 1e-6))
    if samples < _MIN_PROGRAM_SAMPLES { samples = _MIN_PROGRAM_SAMPLES }
    if samples > _MAX_PROGRAM_SAMPLES { samples = _MAX_PROGRAM_SAMPLES }

    let timings: list[float] = []
    for _round in 0..samples { timings.push(_time_process(command)) }
    return BenchmarkResult(name, timings)
}

#: Return the `jaithon` executable to run a program benchmark with.
#:
#: `argv[0]` is preferred so a build tree benchmarks its own build, not
#: whichever `jaithon` happens to be on `PATH`; anything else falls back to
#: that.
pub fn jaithon_executable() -> str {
    let argv = os_argv()
    if argv.len() > 0 and Path(argv[0]).is_executable() { return argv[0] }
    return which("jaithon") ?? "jaithon"
}

#: Quote `argument` so that a shell reads it as one word.
#:
#: Single quotes take everything literally; the only character they cannot
#: carry is a single quote, which is closed, escaped and reopened.
#:
#: ```
#: shell_quote("tests/bench/nbody.jai")    # "'tests/bench/nbody.jai'"
#: shell_quote("it's")                     # "'it'\\''s'"
#: ```
pub fn shell_quote(argument: str) -> str {
    return "'" + argument.replace("'", "'\\''") + "'"
}

fn _time_process(command: list[str]) -> float {
    let start = Instant.now()
    let finished = run(command)
    let elapsed = start.elapsed().to_seconds()
    if finished.exit_code != 0 {
        let reason = (finished.stderr + finished.stdout).strip()
        throw OSError(
            f"{command[command.len() - 1]} exited with status {finished.exit_code}: {reason}"
        )
    }
    return elapsed
}

#: One benchmark measured against what it used to be.
pub class Change {
    #: The benchmark's name.
    pub let name: str

    #: The median sample now, in seconds.
    pub let now: float

    #: The median sample in the baseline, or null when it is new.
    pub let before: float?

    #: Record a comparison of `now` against `before`.
    pub fn init(self, name: str, now: float, before: float?) {
        self.name = name
        self.now = now
        self.before = before
    }

    #: How much slower this is than its baseline: 0.1 means ten per cent slower.
    #:
    #: Returns 0.0 for a benchmark with no baseline, which cannot have changed.
    #:
    #: ```
    #: Change("b", 2.0, 1.0).drift()     # 1.0
    #: Change("b", 1.0, null).drift()    # 0.0
    #: ```
    pub fn drift(self) -> float {
        let before = self.before
        if before is null or before <= 0.0 { return 0.0 }
        return (self.now - before) / before
    }

    #: Whether this is slower than its baseline by more than `threshold`.
    pub fn is_regression(self, threshold: float) -> bool {
        return self.before is not null and self.drift() > threshold
    }

    #: Whether this is faster than its baseline by more than `threshold`.
    pub fn is_improvement(self, threshold: float) -> bool {
        return self.before is not null and self.drift() < -threshold
    }

    #: The drift as a signed percentage, or `new` when there is no baseline.
    #:
    #: ```
    #: Change("b", 2.0, 1.0).summary()     # "+100.0%"
    #: Change("b", 1.0, null).summary()    # "new"
    #: ```
    pub fn summary(self) -> str {
        if self.before is null { return "new" }
        let percent = self.drift() * 100.0
        let sign = if percent >= 0.0 { "+" } else { "" }
        return sign + fixed(percent, 1) + "%"
    }
}

#: The recorded medians of a previous run.
pub class Baseline {
    #: Median seconds per call, keyed by benchmark name.
    pub let medians: dict[str, float]

    #: Build a baseline from recorded medians.
    pub fn init(self, medians: dict[str, float]) {
        self.medians = medians
    }

    #: Read a baseline from `path`, or return an empty one when it is absent.
    #:
    #: Raises `ValueError` when the file is there but is not a baseline: a
    #: corrupt file that silently measures against nothing is worse than one
    #: that says so.
    pub static fn load(path: str) -> Baseline {
        let file = Path(path)
        if not file.exists() { return Baseline({}) }
        let document = parse_json(file.read_text())
        if not isinstance(document, dict) or "benchmarks" not in document {
            throw ValueError(f"{path} is not a benchmark baseline")
        }
        let medians: dict[str, float] = {}
        for record in document["benchmarks"] {
            medians[record["name"]] = float(record["median"])
        }
        return Baseline(medians)
    }

    #: Write `results` to `path` as the new baseline.
    pub static fn save(path: str, results: list[BenchmarkResult]) -> void {
        let records = [{"name": result.name, "median": result.median(), "mean": result.mean(), "fastest": result.fastest(), "samples": result.iterations} for result in results]
        let document = {"version": _BASELINE_VERSION, "benchmarks": records}
        Path(path).write_text(stringify(document, indent: 2, sort_keys: true) + "\n")
    }

    #: Compare `result` against what was recorded for it.
    pub fn compare(self, result: BenchmarkResult) -> Change {
        return Change(result.name, result.median(), self.medians.get(result.name, null))
    }
}

#: Run every benchmark in one file and return the results.
#:
#: `bench_` functions win when present; a file with only `main` runs as a
#: whole program instead. A file that's neither is silently skipped —
#: walking a directory turns up plenty of files that were never meant to be
#: benchmarks.
pub fn run_file(path: str, filter: str?, budget: float) -> list[BenchmarkResult] {
    let source = read_file(path)
    let suite = module_name_of(path)
    var program: Node? = null
    try {
        program = parse_module(source, path)
    } catch error: ParseError {
        print_err(f"bench: {error.message}")
        return []
    }
    if program is null { return [] }

    let names = _bench_functions_in(program)
    if names.len() == 0 {
        if not _declares_main_in(program) { return [] }
        return _run_program(path, suite, filter, budget)
    }

    let wanted = [name for name in names if filter is null or name.contains(filter)]
    if wanted.len() == 0 { return [] }

    try {
        __prim__.exec(source)
    } catch error: Error {
        print_err(f"bench: {path}: {type_of(error)}: {error.message}")
        return []
    }

    let results: list[BenchmarkResult] = []
    for name in wanted {
        let call = name + "()"
        let body = fn() -> any { return __prim__.exec(call) }
        try {
            results.push(measure(f"{suite}.{name}", body, budget))
        } catch error: Error {
            print_err(f"bench: {suite}.{name} raised {type_of(error)}: {error.message}")
        }
    }
    return results
}

fn _run_program(path: str, suite: str, filter: str?, budget: float) -> list[BenchmarkResult] {
    if filter is not null and not suite.contains(filter) { return [] }
    let command = [jaithon_executable(), "run", path]
    try {
        return [measure_program(suite, command, budget)]
    } catch error: Error {
        print_err(f"bench: {suite}: {type_of(error)}: {error.message}")
        return []
    }
}

#: Render the results, and their drift from `baseline`, as a table.
pub fn render_report(results: list[BenchmarkResult], baseline: Baseline, threshold: float) -> str {
    let table = Table()
    table.add_column("benchmark")
    table.add_column("median", Align.Right)
    table.add_column("mean", Align.Right)
    table.add_column("fastest", Align.Right)
    table.add_column("slowest", Align.Right)
    table.add_column("stddev", Align.Right)
    table.add_column("ops/s", Align.Right)
    table.add_column("vs base", Align.Right)
    for result in results {
        let change = baseline.compare(result)
        table.add_row(
            [
                result.name,
                _seconds(result.median()),
                _seconds(result.mean()),
                _seconds(result.fastest()),
                _seconds(result.slowest()),
                _seconds(result.deviation()),
                _rate(result.per_second()),
                _painted(change, threshold),
            ]
        )
    }
    return table.render(TableStyle.Box)
}

#: Run `jaithon bench`.
#:
#: ```text
#: jaithon bench [--baseline file] [--save] [--threshold f] [--budget s]
#:               [--filter substring] [paths...]
#: ```
#:
#: With no path, `tests/bench` is measured. `--budget`, not `--time`: the
#: driver owns that flag for timing a whole run. Exits 1 when any benchmark
#: is slower than its baseline by more than the threshold — what makes a
#: regression fail a build instead of surfacing six months later.
pub fn main(args: list[str]) -> int {
    let options = _Options.parse(tail_arguments(args, "bench"))
    if options.usage_error is not null {
        print_err(f"jaithon bench: {options.usage_error}")
        print_err(
            "usage: jaithon bench [--baseline file] [--save] [--threshold f] " +
                "[--budget s] [--filter substring] [paths...]"
        )
        return 2
    }

    var paths: list[str] = []
    try {
        paths = expand_paths(options.paths)
    } catch error: FileNotFoundError {
        print_err(f"jaithon bench: {error.message}")
        return 1
    }
    var want_jaitensor = false
    for asked in options.paths {
        if asked.contains("jaitensor") { want_jaitensor = true }
    }

    let baseline = Baseline.load(options.baseline)
    let results: list[BenchmarkResult] = []
    for path in paths {
        if not want_jaitensor and path.contains("/jaitensor/") { continue }
        results.extend(run_file(path, options.filter, options.budget))
    }
    if results.len() == 0 {
        print_err("jaithon bench: no benchmarks found")
        return 1
    }

    print(render_report(results, baseline, options.threshold))

    if options.save {
        Baseline.save(options.baseline, results)
        print(f"baseline written to {options.baseline}")
        return 0
    }

    let regressions: list[Change] = []
    for result in results {
        let change = baseline.compare(result)
        if change.is_regression(options.threshold) { regressions.push(change) }
    }
    if regressions.len() == 0 {
        print(dim(f"no regression beyond {fixed(options.threshold * 100.0, 1)}%"))
        return 0
    }
    print("")
    let word = if regressions.len() == 1 { "regression" } else { "regressions" }
    print(bold(red(f"{regressions.len()} {word}")))
    for change in regressions {
        print(
            "  " +
                change.name +
                "  " +
                change.summary() +
                dim(f"  {_seconds(change.before ?? 0.0)} -> {_seconds(change.now)}")
        )
    }
    return 1
}

# --- Formatting --------------------------------------------------------------
fn _seconds(value: float) -> str {
    if value <= 0.0 { return "0" }
    if value < 1e-6 { return fixed(value * 1e9, 1) + "ns" }
    if value < 1e-3 { return fixed(value * 1e6, 2) + "µs" }
    if value < 1.0 { return fixed(value * 1e3, 2) + "ms" }
    return fixed(value, 3) + "s"
}

fn _rate(per_second: float) -> str {
    if per_second >= 1e9 { return fixed(per_second / 1e9, 2) + "G" }
    if per_second >= 1e6 { return fixed(per_second / 1e6, 2) + "M" }
    if per_second >= 1e3 { return fixed(per_second / 1e3, 2) + "k" }
    return fixed(per_second, 1)
}

fn _painted(change: Change, threshold: float) -> str {
    let text = change.summary()
    if change.is_regression(threshold) { return red(text) }
    if change.is_improvement(threshold) { return green(text) }
    return text
}

class _Options {
    pub var baseline: str
    pub var save: bool
    pub var threshold: float
    pub var budget: float
    pub var filter: str?
    pub var paths: list[str]
    pub var usage_error: str?

    pub fn init(self) {
        self.baseline = DEFAULT_BASELINE
        self.save = false
        self.threshold = DEFAULT_THRESHOLD
        self.budget = DEFAULT_BUDGET
        self.filter = null
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
            let name = argument.name
            if name == "--baseline" or
                name == "--threshold" or
                name == "--budget" or
                name == "--filter" {
                let value = reader.value_of(argument)
                if value is null { break }
                options._set(name, value)
                if options.usage_error is not null { return options }
            } elif name == "--save" {
                if reader.is_bare(argument) { options.save = true }
            } else {
                reader.unknown(argument)
            }
        }
        if options.usage_error is null { options.usage_error = reader.error }
        if options.paths.len() == 0 { options.paths = ["tests/bench"] }
        return options
    }

    pub fn _set(self, name: str, value: str) -> void {
        if name == "--baseline" {
            self.baseline = value
            return
        }
        if name == "--filter" {
            self.filter = value
            return
        }
        let number = to_float(value)
        if number is null {
            self.usage_error = f"{name} needs a number, got {repr(value)}"
            return
        }
        if name == "--threshold" { self.threshold = number } else { self.budget = number }
    }
}
