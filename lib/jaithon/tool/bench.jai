#: The benchmark harness behind `jaithon bench`.
#:
#: There are two kinds of benchmark and the file says which it is. A file
#: declaring top-level `bench_` functions taking no arguments is measured a
#: function at a time, in this process. A file declaring only `main` — which is
#: what everything in `tests/bench` is, because each of those has a Python peer
#: that has to compute the same answer — is measured as a whole program, by
#: running it the way a user would and timing the process. Either way the
#: result is reported with its mean, median, fastest and slowest sample, its
#: standard deviation and its throughput.
#:
#: Discovery is `tests/bench` and nothing else. `tests` as a whole is the wrong
#: root: it also holds `tests/errors`, whose files do not parse *on purpose*,
#: and a harness that walks into those reports seven parse failures for seven
#: files doing their job. A path named on the command line is measured whatever
#: it is called, because naming a file means wanting that file.
#:
#: Calibration is what makes the numbers mean anything. A body that takes forty
#: nanoseconds cannot be timed against a clock whose resolution is a
#: microsecond, so the harness first times one call, then chooses a batch size
#: that puts a whole sample comfortably above the clock's resolution, and
#: finally chooses how many samples fit in the time budget. Every reported
#: figure is per call, whatever the batch turned out to be.
#:
#: Both the mean and the median are reported because they disagree exactly when
#: the distribution is skewed, and a timing distribution always is: nothing
#: makes a run faster than it can be, and a great many things make one slower.
#: When the mean sits well above the median, something interrupted the run and
#: the median is the honest number — which is why it, and not the mean, is what
#: a regression is measured against.
#:
#: With a baseline file present, every benchmark is compared against its
#: recorded median and anything slower by more than the threshold is a
#: regression, and a regression is a non-zero exit status.
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

# A whole-program sample costs a process, so the counts are much smaller than
# the in-process ones. Five is `scripts/run_bench.sh`'s RUNS, and the least a
# median means anything over.
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
#: A file with no `bench_` function but with a `main` is still a benchmark: it
#: is measured as a whole program rather than a function at a time. That is
#: what the pairs in `tests/bench` are, each with a Python peer computing the
#: same answer, and measuring them any other way would stop the two being
#: comparable.
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
#: The batch is how many calls make up one sample and exists to lift a sample
#: above the clock's resolution; the count is how many samples the budget
#: affords, never fewer than ten — a standard deviation over three samples is
#: not a measurement — and never more than two hundred.
pub fn calibrate(body: fn() -> any, budget: float) -> tuple[int, int] {
    # One untimed call pays for whatever the first call has to build.
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
#: A whole program cannot be batched — every sample is one process — so the
#: only thing the budget buys is the number of samples, and the floor is five
#: so that a median and a spread exist at all. The first run is a warm-up and
#: is not recorded: it is the one that pays for reading the file, filling the
#: page cache and writing the bytecode cache, and charging that to the program
#: would overstate every benchmark by the same wrong amount.
#:
#: Raises `OSError` when the program fails, because a benchmark that computes
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
#: `argv[0]` is the running interpreter whenever the tool was reached through
#: the CLI, which is the case that matters: a build tree must benchmark its own
#: build and not whichever `jaithon` happens to be installed. Anything else
#: falls back to the one on `PATH`.
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

# One timed run of a child process. The clock covers the whole process, exactly
# as `scripts/run_bench.sh` times it, so the two agree on what they measured.
#
# `run` starts the program directly, so what is timed is one process and no
# shell: the argument list cannot be re-parsed, and standard error arrives
# separately instead of interleaved into the output a failure has to report.
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
#: `bench_` functions win when the file has any: it is executed once, in this
#: module's namespace, and then each is called by name. A file with none but
#: with a `main` is measured as a whole program instead. A file that is neither
#: contributes nothing and says nothing — walking a directory turns up plenty
#: of files that were never meant to be benchmarks.
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
#: With no path, `tests/bench` is measured. `--save` writes the run to the
#: baseline file instead of comparing against it. `--budget` is roughly how long
#: each benchmark is given, in seconds; it is not spelled `--time` because the
#: driver owns that flag for timing a whole run, and one word cannot mean two
#: things on one command line. The exit status is 1 when any benchmark is slower
#: than its baseline by more than the threshold, which is what makes a
#: performance regression fail a build rather than be noticed six months later.
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
        # The default root is only a default: being run from somewhere without
        # one is a thing to say plainly, not a traceback.
        print_err(f"jaithon bench: {error.message}")
        return 1
    }

    let baseline = Baseline.load(options.baseline)
    let results: list[BenchmarkResult] = []
    for path in paths {
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

# Seconds at whatever scale keeps three significant figures readable.
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
        while true {
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
