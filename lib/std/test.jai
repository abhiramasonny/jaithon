#: The testing framework the repository runs on: assertions, suites, a runner
#: and a benchmark harness.
#:
#: ```
#: describe("arithmetic", fn() {
#:     it("adds", fn() { assert_eq(2 + 2, 4) })
#:     it("divides by zero loudly", fn() {
#:         assert_throws(DivisionByZeroError, || 1 / 0)
#:     })
#: })
#: run_all(verbose: false).exit_code()    # 0
#: ```
#:
#: Every assertion failure names both values with `repr`, since a bare
#: "assertion failed" costs the reader the run they'd have to repeat.
import std.algo.stat as stat
import std.math as math

from std.core import cmp, noop
from std.fmt import bold, dim, fixed, green, red, yellow
from std.time import Duration, Instant

#: How many samples `benchmark` and `compare` take when not told otherwise.
pub const DEFAULT_ITERATIONS: int = 1000

# Truncated past this length -- an unclipped list makes a failure message unreadable.
const _QUOTE_LIMIT = 240

#: How a single test ended.
pub enum Outcome {
    #: The body ran to the end without raising.
    Passed,
    #: An assertion in the body was not met, so the body raised
    #: `AssertionError`. The failure is the test doing its job.
    Failed,
    #: The body raised something that was not an `AssertionError`, or a hook
    #: around it did. The fault could be in the test as much as the code it
    #: was checking, since no verdict was reached.
    Errored,
    #: The test was declared with `skip: true` and never run. It is counted
    #: separately from passes, so a disabled test stays visible instead of
    #: vanishing from the totals.
    Skipped,

    #: Report whether this outcome should turn the run red.
    pub fn is_failure(self) -> bool {
        return match self {
            Outcome.Failed => true,
            Outcome.Errored => true,
            Outcome.Passed => false,
            Outcome.Skipped => false,
        }
    }

    #: Report whether the test ran and passed.
    pub fn is_passed(self) -> bool {
        return match self {
            Outcome.Passed => true,
            _ => false,
        }
    }

    #: Report whether the test was declared but not run.
    pub fn is_skipped(self) -> bool {
        return match self {
            Outcome.Skipped => true,
            _ => false,
        }
    }

    #: Return the short label the runner prints in its left column.
    pub fn label(self) -> str {
        return match self {
            Outcome.Passed => "pass",
            Outcome.Failed => "FAIL",
            Outcome.Errored => "ERROR",
            Outcome.Skipped => "skip",
        }
    }
}

#: What one test did, and how long it took doing it.
pub class TestResult {
    #: The name given to `it` or `Suite.test`.
    pub let name: str

    #: The name of the suite the test belongs to, empty when it has none.
    pub let suite: str

    #: How the test ended.
    pub let outcome: Outcome

    #: How long the body ran, its setup and teardown hooks included.
    pub let duration: Duration

    #: The failure message, empty when the test passed.
    pub let message: str

    #: The interpreter traceback for the failure, empty when the test passed.
    pub let trace: str

    #: Record the outcome of one test.
    pub fn init(
        self,
        name: str,
        suite: str,
        outcome: Outcome,
        duration: Duration,
        message: str,
        trace: str
    ) {
        self.name = name
        self.suite = suite
        self.outcome = outcome
        self.duration = duration
        self.message = message
        self.trace = trace
    }

    #: Return the test's name qualified by its suite.
    #:
    #: ```
    #: let result = TestResult("adds", "math", Outcome.Passed, Duration.zero(), "", "")
    #: result.full_name()    # "math > adds"
    #: ```
    pub fn full_name(self) -> str {
        if self.suite.len() == 0 { return self.name }
        return self.suite + " > " + self.name
    }

    #: Return the one-line form the runner prints as the suite runs.
    pub fn to_str(self) -> str {
        let label = _colored(self.outcome, self.outcome.label())
        return f"{label}  {self.name}  " + dim(_duration_text(self.duration))
    }

    fn __str__(self) -> str { return self.to_str() }
}

#: One named test: a name, a body, and the ability to run it exactly once.
pub class Test {
    #: The name reported for this test.
    pub let name: str

    #: Whether the body is to be skipped rather than run.
    pub let skipped: bool

    let body: fn() -> any

    #: Create a test that runs `body`.
    pub fn init(self, name: str, body: fn() -> any, skipped: bool = false) {
        if name.len() == 0 { throw ValueError("a test needs a name") }
        self.name = name
        self.body = body
        self.skipped = skipped
    }

    #: Run the body between the two hooks and return what happened.
    #:
    #: Never raises: an `AssertionError` becomes `Outcome.Failed`, any other
    #: exception becomes `Outcome.Errored`; the teardown hook always runs.
    #:
    #: ```
    #: Test("passes", fn() { assert_true(true) }).run().outcome.is_passed()    # true
    #: Test("fails", fn() { assert_true(false) }).run().outcome.is_failure()   # true
    #: ```
    pub fn run(
        self,
        suite: str = "",
        before: fn() -> any = noop,
        after: fn() -> any = noop
    ) -> TestResult {
        if self.skipped {
            return TestResult(self.name, suite, Outcome.Skipped, Duration.zero(), "", "")
        }
        let start = Instant.now()
        var outcome = Outcome.Passed
        var message = ""
        var trace = ""
        try {
            before()
            self.body()
        } catch failure: AssertionError {
            outcome = Outcome.Failed
            message = failure.message
            trace = _current_trace()
        } catch error: Error {
            outcome = Outcome.Errored
            message = f"{type_of(error)}: {error.message}"
            trace = _current_trace()
        }
        try {
            after()
        } catch error: Error {
            if not outcome.is_failure() {
                outcome = Outcome.Errored
                message = f"teardown raised {type_of(error)}: {error.message}"
                trace = _current_trace()
            }
        }
        return TestResult(self.name, suite, outcome, start.elapsed(), message, trace)
    }
}

#: A named group of tests sharing setup and teardown hooks.
#:
#: ```
#: let suite = Suite("squares")
#: for value in [1, 2, 3] { suite.test(f"squares {value}", fn() { assert_true(true) }) }
#: suite.run().len()    # 3
#: ```
pub class Suite {
    #: The name reported for this suite.
    pub let name: str

    var tests: list[Test]
    var before_each: fn() -> any
    var after_each: fn() -> any
    var before_all: fn() -> any
    var after_all: fn() -> any

    #: Create an empty suite called `name`.
    pub fn init(self, name: str) {
        self.name = name
        self.tests = []
        self.before_each = noop
        self.after_each = noop
        self.before_all = noop
        self.after_all = noop
    }

    #: Add a test that runs `body`.
    pub fn test(self, name: str, body: fn() -> any, skip: bool = false) -> void {
        self.tests.push(Test(name, body, skip))
    }

    #: Run `hook` before every test in this suite.
    pub fn setup(self, hook: fn() -> any) -> void { self.before_each = hook }

    #: Run `hook` after every test in this suite, pass or fail.
    pub fn teardown(self, hook: fn() -> any) -> void { self.after_each = hook }

    #: Run `hook` once, before the suite's first test.
    #:
    #: Paid once, so put an expensive fixture here — but it's shared, and a
    #: test that mutates it affects every test after it.
    pub fn setup_all(self, hook: fn() -> any) -> void { self.before_all = hook }

    #: Run `hook` once, after the suite's last test.
    pub fn teardown_all(self, hook: fn() -> any) -> void { self.after_all = hook }

    #: The number of tests added so far.
    pub fn len(self) -> int { return self.tests.len() }

    #: Run every test in order and return their results.
    #:
    #: A `setup_all` failure fails every test in the suite instead of aborting
    #: the run, since those tests needed the fixture it never built.
    pub fn run(self) -> list[TestResult] {
        let results: list[TestResult] = []
        try {
            self.before_all()
        } catch error: Error {
            let reason = f"suite setup raised {type_of(error)}: {error.message}"
            let trace = _current_trace()
            for test in self.tests {
                results.push(
                    TestResult(
                        test.name,
                        self.name,
                        Outcome.Errored,
                        Duration.zero(),
                        reason,
                        trace
                    )
                )
            }
            return results
        }
        for test in self.tests {
            results.push(test.run(self.name, self.before_each, self.after_each))
        }
        try {
            self.after_all()
        } catch error: Error {
            results.push(
                TestResult(
                    "teardown_all",
                    self.name,
                    Outcome.Errored,
                    Duration.zero(),
                    f"{type_of(error)}: {error.message}",
                    _current_trace()
                )
            )
        }
        return results
    }

    fn __len__(self) -> int { return self.tests.len() }
}

#: The results of a whole run, and the summary printed from them.
pub class Report {
    #: Every result, in the order the tests ran.
    pub let results: list[TestResult]

    #: Collect `results` into a report.
    pub fn init(self, results: list[TestResult]) { self.results = results }

    #: How many tests passed.
    pub fn passed(self) -> int {
        var total = 0
        for result in self.results {
            if result.outcome.is_passed() { total += 1 }
        }
        return total
    }

    #: How many tests failed an assertion or raised.
    pub fn failed(self) -> int { return self.failures().len() }

    #: How many tests were declared but not run.
    pub fn skipped(self) -> int {
        var total = 0
        for result in self.results {
            if result.outcome.is_skipped() { total += 1 }
        }
        return total
    }

    #: How many results there are in all, skipped tests included.
    pub fn total(self) -> int { return self.results.len() }

    #: The results that failed, in the order they ran.
    pub fn failures(self) -> list[TestResult] {
        return [result for result in self.results if result.outcome.is_failure()]
    }

    #: The total time spent inside test bodies.
    pub fn duration(self) -> Duration {
        var nanos = 0
        for result in self.results { nanos += result.duration.to_nanos() }
        return Duration(nanos)
    }

    #: Return 0 when everything passed and 1 when anything did not.
    pub fn exit_code(self) -> int {
        if self.failed() > 0 { return 1 }
        return 0
    }

    #: Print the failures in full, then the counts.
    pub fn print_summary(self) -> void {
        let broken = self.failures()
        if broken.len() > 0 {
            print("")
            print(bold("FAILURES"))
            for result in broken {
                let heading = _colored(result.outcome, result.outcome.label())
                print("  " + heading + "  " + bold(result.full_name()))
                for line in result.message.split("\n") { print("        " + line) }
                for line in result.trace.split("\n") {
                    if line.len() > 0 { print(dim("        " + line)) }
                }
            }
        }
        print("")
        let skipped = self.skipped()
        let tail = skipped > 0 ? f", {skipped} skipped" : ""
        let counts = f"{self.passed()} passed, {self.failed()} failed{tail}, " +
            f"{self.total()} total"
        let paint = self.failed() > 0 ? red : green
        print(paint(counts) + dim(" in " + _duration_text(self.duration())))
    }
}

#: The timings of one benchmarked body.
#:
#: Mean and median disagree when the distribution is skewed, which timing
#: data almost always is; a mean well above the median means the run was
#: interrupted, and the median is the honest number.
pub class BenchmarkResult {
    #: The name the benchmark was given.
    pub let name: str

    #: How many samples were taken.
    pub let iterations: int

    #: Every sample, in seconds, in the order measured.
    pub let samples: list[float]

    #: Build a result from the seconds each sample took.
    pub fn init(self, name: str, samples: list[float]) {
        if samples.len() == 0 { throw ValueError(f"benchmark {repr(name)} took no samples") }
        self.name = name
        self.samples = samples
        self.iterations = samples.len()
    }

    #: The arithmetic mean of the samples, in seconds.
    pub fn mean(self) -> float { return math.mean(self.samples) }

    #: The median sample, in seconds.
    pub fn median(self) -> float { return stat.median(self.samples) }

    #: The fastest sample, in seconds — the closest thing to the true cost.
    pub fn fastest(self) -> float {
        let (lowest, _highest) = stat.min_max(self.samples)
        return lowest
    }

    #: The slowest sample, in seconds.
    pub fn slowest(self) -> float {
        let (_lowest, highest) = stat.min_max(self.samples)
        return highest
    }

    #: The sample standard deviation, in seconds; 0.0 for a single sample.
    pub fn deviation(self) -> float {
        if self.samples.len() < 2 { return 0.0 }
        return stat.stdev(self.samples)
    }

    #: How many times the body would run in a second, at the median.
    pub fn per_second(self) -> float {
        let middle = self.median()
        if middle <= 0.0 { return 0.0 }
        return 1.0 / middle
    }

    #: Return the one-line report: mean, median, range and spread.
    pub fn to_str(self) -> str {
        let parts = [
            f"mean {_duration_of(self.mean())}",
            f"median {_duration_of(self.median())}",
            f"min {_duration_of(self.fastest())}",
            f"max {_duration_of(self.slowest())}",
            f"sd {_duration_of(self.deviation())}",
        ]
        return f"{self.name}: " + ", ".join(parts) + f" ({self.iterations} runs)"
    }

    #: Print `to_str`.
    pub fn print(self) -> void { print(self.to_str()) }

    fn __str__(self) -> str { return self.to_str() }
}

#: Two benchmarks measured against each other.
pub class Comparison {
    #: The first implementation's timings.
    pub let first: BenchmarkResult

    #: The second implementation's timings.
    pub let second: BenchmarkResult

    #: Pair two results measured over the same interleaved rounds.
    pub fn init(self, first: BenchmarkResult, second: BenchmarkResult) {
        self.first = first
        self.second = second
    }

    #: How many times faster the winner is, at the median. Never below 1.0.
    pub fn speedup(self) -> float {
        let quicker = self.first.median()
        let slower = self.second.median()
        if quicker <= 0.0 or slower <= 0.0 { return 1.0 }
        if quicker < slower { return slower / quicker }
        return quicker / slower
    }

    #: The name of the faster implementation, at the median.
    pub fn winner(self) -> str {
        if self.first.median() <= self.second.median() { return self.first.name }
        return self.second.name
    }

    #: Return both lines of timings and the verdict.
    pub fn to_str(self) -> str {
        return self.first.to_str() +
            "\n" +
            self.second.to_str() +
            "\n" +
            f"{self.winner()} is {fixed(self.speedup(), 2)}x faster at the median"
    }

    #: Print `to_str`.
    pub fn print(self) -> void { print(self.to_str()) }

    fn __str__(self) -> str { return self.to_str() }
}

# --- Assertions --------------------------------------------------------------

#: Fail unless `value` is true.
#:
#: ```
#: assert_true(1 < 2)
#: ```
pub fn assert_true(value: bool, message: str = "") -> void {
    if value { return }
    throw AssertionError(_explain("expected true, got false", message))
}

#: Fail unless `value` is false.
#:
#: ```
#: assert_false("a" == "b")
#: ```
pub fn assert_false(value: bool, message: str = "") -> void {
    if not value { return }
    throw AssertionError(_explain("expected false, got true", message))
}

#: Fail unless `actual` equals `expected`.
#:
#: Types are named too when the two reprs come out identical, as `1` vs `1.0`
#: does once the type is the only difference left.
#:
#: ```
#: assert_eq(2 + 2, 4)
#: assert_eq("ab".upper(), "AB")
#: ```
pub fn assert_eq(actual: any, expected: any, message: str = "") -> void {
    if actual == expected { return }
    let shown_actual = _quote(actual)
    let shown_expected = _quote(expected)
    var detail = f"expected {shown_expected}, got {shown_actual}"
    if shown_actual == shown_expected {
        detail = detail + f" (types {type_of(expected)} and {type_of(actual)})"
    }
    throw AssertionError(_explain(detail, message))
}

#: Fail when `actual` equals `forbidden`.
#:
#: ```
#: assert_ne(1, 2)
#: ```
pub fn assert_ne(actual: any, forbidden: any, message: str = "") -> void {
    if actual != forbidden { return }
    throw AssertionError(_explain(f"expected any value other than {_quote(forbidden)}", message))
}

#: Fail unless `smaller` sorts strictly before `larger`.
#:
#: ```
#: assert_lt(1, 2)
#: assert_lt("apple", "banana")
#: ```
pub fn assert_lt(smaller: any, larger: any, message: str = "") -> void {
    if cmp(smaller, larger) < 0 { return }
    throw AssertionError(_explain(f"expected {_quote(smaller)} < {_quote(larger)}", message))
}

#: Fail unless `larger` sorts strictly after `smaller`.
#:
#: ```
#: assert_gt(2, 1)
#: ```
pub fn assert_gt(larger: any, smaller: any, message: str = "") -> void {
    if cmp(larger, smaller) > 0 { return }
    throw AssertionError(_explain(f"expected {_quote(larger)} > {_quote(smaller)}", message))
}

#: Fail unless `actual` is within `tolerance` of `expected`.
#:
#: The tolerance is absolute. Computed floats rarely match bit-for-bit, so
#: `assert_eq` fails on them for no real reason — this is the assertion for
#: those cases.
#:
#: ```
#: assert_near(0.1 + 0.2, 0.3)
#: assert_near(22.0 / 7.0, 3.14159, 0.01)
#: ```
pub fn assert_near(
    actual: float,
    expected: float,
    tolerance: float = 1e-9,
    message: str = ""
) -> void {
    if tolerance < 0.0 { throw ValueError(f"tolerance must be non-negative, got {tolerance}") }
    if actual != actual or expected != expected {
        throw AssertionError(
            _explain(f"expected {expected} within {tolerance}, got {actual}", message)
        )
    }
    let difference = actual - expected
    let distance = difference < 0.0 ? -difference : difference
    if distance <= tolerance { return }
    throw AssertionError(
        _explain(
            f"expected {expected} within {tolerance}, got {actual} (off by {distance})",
            message
        )
    )
}

#: Fail unless `body` raises an exception of class `expected`, and return it.
#:
#: A subclass counts: `assert_throws(LookupError, body)` accepts a `KeyError`.
#: The exception is returned so its message can be checked too — asserting
#: only the class lets a test pass when the code raised for the wrong reason.
#:
#: ```
#: let error = assert_throws(IndexError, || [1][5])
#: error.message.len() > 0    # true
#: ```
pub fn assert_throws(expected: any, body: fn() -> any, message: str = "") -> Error {
    try {
        body()
    } catch caught: Error {
        if isinstance(caught, expected) { return caught }
        throw AssertionError(
            _explain(f"expected {expected}, got {type_of(caught)}: {caught.message}", message)
        )
    }
    throw AssertionError(_explain(f"expected {expected} to be raised, but nothing was", message))
}

#: Fail unless `container` holds `value`.
#:
#: Works on whatever `in` works on: a list, a str, a set, or a dict's keys.
#:
#: ```
#: assert_contains([1, 2, 3], 2)
#: assert_contains("hello", "ell")
#: assert_contains({"a": 1}, "a")
#: ```
pub fn assert_contains(container: any, value: any, message: str = "") -> void {
    if value in container { return }
    throw AssertionError(
        _explain(f"expected {_quote(container)} to contain {_quote(value)}", message)
    )
}

#: Fail unless `items` is in order, naming the first pair that is not.
#:
#: Ordered by `std.core.cmp`. Equal neighbours are in order either way.
#:
#: ```
#: assert_sorted([1, 2, 2, 3])
#: assert_sorted([3, 2, 1], descending: true)
#: ```
pub fn assert_sorted(items: list[any], descending: bool = false, message: str = "") -> void {
    for index in 1..items.len() {
        let ordering = cmp(items[index - 1], items[index])
        let out_of_order = descending ? ordering < 0 : ordering > 0
        if out_of_order {
            let direction = descending ? "descending" : "ascending"
            throw AssertionError(
                _explain(
                    f"expected {direction} order, but element {index - 1} is " +
                        f"{_quote(items[index - 1])} and element {index} is {_quote(items[index])}",
                    message
                )
            )
        }
    }
}

#: Fail unless `container` holds exactly `expected` elements.
#:
#: ```
#: assert_len([1, 2, 3], 3)
#: assert_len("hello", 5)
#: ```
pub fn assert_len(container: any, expected: int, message: str = "") -> void {
    let actual = len(container)
    if actual == expected { return }
    throw AssertionError(
        _explain(
            f"expected {expected} elements, got {actual} in {_quote(container)}",
            message
        )
    )
}

#: Fail unconditionally with `message`.
#:
#: For the branch that should not have been reached: the match arm that proves
#: the value was wrong, the line after a call that should have raised.
pub fn fail(message: str) -> void { throw AssertionError(message) }

# --- Declaring and running tests ----------------------------------------------
var _suites: list[Suite] = []
var _current: Suite? = null

#: Declare a suite named `name`, whose body calls `it` for each of its tests.
#:
#: Registered for `run_all` but also returned, for a caller that wants to run
#: it alone. Describes may nest: an inner `it` attaches to the inner suite.
#:
#: ```
#: let suite = describe("empty", fn() { })
#: suite.len()    # 0
#: ```
pub fn describe(name: str, body: fn() -> any) -> Suite {
    let suite = Suite(name)
    let enclosing = _current
    _current = suite
    defer { _current = enclosing }
    body()
    _suites.push(suite)
    return suite
}

#: Declare a test inside the enclosing `describe`.
#:
#: Raises `RuntimeError` outside a describe block. `skip: true` declares a
#: test without running it, still counted rather than silently missing.
pub fn it(name: str, body: fn() -> any, skip: bool = false) -> void {
    let suite = _current
    if suite is null {
        throw RuntimeError(f"it({repr(name)}) must be called inside a describe block")
    }
    suite.test(name, body, skip)
}

#: Run every registered suite, print the results, and return the report.
#:
#: The registry is cleared on the way out, so a second call won't re-run the
#: same suites.
pub fn run_all(verbose: bool = true) -> Report {
    let results: list[TestResult] = []
    for suite in _suites {
        if verbose and suite.len() > 0 { print(bold(suite.name)) }
        for result in suite.run() {
            results.push(result)
            if verbose { print("  " + result.to_str()) }
        }
    }
    reset()
    let report = Report(results)
    report.print_summary()
    return report
}

#: Forget every registered suite.
#:
#: Public so the framework's own tests can reset the registry without leaking
#: their suites into the run that's testing them.
pub fn reset() -> void {
    _suites = []
    _current = null
}

#: Run `body` once per row of `cases`, reporting which row failed.
#:
#: A plain loop over the same cases would lose the one thing table-driven
#: tests need: knowing which row broke.
#:
#: ```
#: for_each_case(
#:     [(1, 1), (2, 4), (3, 9)],
#:     fn(row) { assert_eq(row[0] * row[0], row[1]) }
#: )
#: ```
pub fn for_each_case(cases: list[any], body: fn(any) -> any) -> void {
    for (index, row) in cases.enumerate() {
        try {
            body(row)
        } catch failure: AssertionError {
            throw AssertionError(f"case {index} {_quote(row)}: {failure.message}")
        }
    }
}

# --- Benchmarking --------------------------------------------------------------

#: Time `body` over `iterations` samples and report the distribution.
#:
#: A warmup phase runs unmeasured first, since first-call costs — init, cold
#: caches, branch prediction — would otherwise land on sample one. Defaults
#: to a tenth of `iterations`, capped at 100.
#:
#: `batch` runs the body `batch` times per sample and divides, so bodies
#: faster than the clock's resolution can still be measured.
#:
#: ```
#: let result = benchmark("addition", fn() { let _sum = 1 + 1 }, 50)
#: result.iterations         # 50
#: result.median() >= 0.0    # true
#: ```
pub fn benchmark(
    name: str,
    body: fn() -> any,
    iterations: int = DEFAULT_ITERATIONS,
    warmup: int? = null,
    batch: int = 1
) -> BenchmarkResult {
    _check_run(iterations, batch)
    _warm(body, warmup ?? _default_warmup(iterations))
    let samples: list[float] = []
    for _round in 0..iterations { samples.push(_time_batch(body, batch)) }
    return BenchmarkResult(name, samples)
}

#: Time two implementations against each other and report which is faster.
#:
#: Measured in alternating rounds rather than back-to-back, so machine drift
#: (a core clocking down, a collector pause) is split between them instead of
#: all charged to whichever ran during it.
#:
#: ```
#: let verdict = compare("a", fn() { let _x = 1 }, "b", fn() { let _y = 2 }, 20)
#: verdict.speedup() >= 1.0    # true
#: ```
pub fn compare(
    first_name: str,
    first: fn() -> any,
    second_name: str,
    second: fn() -> any,
    iterations: int = DEFAULT_ITERATIONS,
    warmup: int? = null,
    batch: int = 1
) -> Comparison {
    _check_run(iterations, batch)
    let rounds = warmup ?? _default_warmup(iterations)
    _warm(first, rounds)
    _warm(second, rounds)
    let first_samples: list[float] = []
    let second_samples: list[float] = []
    for _round in 0..iterations {
        first_samples.push(_time_batch(first, batch))
        second_samples.push(_time_batch(second, batch))
    }
    return Comparison(
        BenchmarkResult(first_name, first_samples),
        BenchmarkResult(second_name, second_samples)
    )
}

fn _check_run(iterations: int, batch: int) -> void {
    if iterations <= 0 { throw ValueError(f"iterations must be positive, got {iterations}") }
    if batch <= 0 { throw ValueError(f"batch must be positive, got {batch}") }
}

fn _default_warmup(iterations: int) -> int {
    let tenth = iterations // 10
    if tenth < 1 { return 1 }
    if tenth > 100 { return 100 }
    return tenth
}

fn _warm(body: fn() -> any, rounds: int) -> void {
    if rounds < 0 { throw ValueError(f"warmup must be non-negative, got {rounds}") }
    for _round in 0..rounds { body() }
}

fn _time_batch(body: fn() -> any, batch: int) -> float {
    let start = Instant.now()
    for _run in 0..batch { body() }
    return start.elapsed().to_seconds() / batch
}

fn _explain(detail: str, note: str) -> str {
    if note.len() == 0 { return detail }
    return note + ": " + detail
}

#: Return `repr(value)`, cut short when it is too long to read in a failure.
fn _quote(value: any) -> str {
    let text = repr(value)
    if text.len() <= _QUOTE_LIMIT { return text }
    return text[0:_QUOTE_LIMIT] + f"... ({text.len()} characters)"
}

#: Return the interpreter's traceback for the exception being handled.
#:
#: The primitive may return either text or a list of frames; both are folded
#: into one string here.
fn _current_trace() -> str {
    let raw = __prim__.traceback()
    if raw is null { return "" }
    if isinstance(raw, list) { return "\n".join([str(frame) for frame in raw]) }
    return str(raw)
}

fn _colored(outcome: Outcome, text: str) -> str {
    return match outcome {
        Outcome.Passed => green(text),
        Outcome.Failed => red(text),
        Outcome.Errored => red(text),
        Outcome.Skipped => yellow(text),
    }
}

fn _duration_text(duration: Duration) -> str { return _duration_of(duration.to_seconds()) }

#: Render `seconds` in the largest unit that leaves it above 1.
fn _duration_of(seconds: float) -> str {
    if seconds < 1e-6 { return fixed(seconds * 1e9, 1) + "ns" }
    if seconds < 1e-3 { return fixed(seconds * 1e6, 2) + "us" }
    if seconds < 1.0 { return fixed(seconds * 1e3, 3) + "ms" }
    return fixed(seconds, 3) + "s"
}
