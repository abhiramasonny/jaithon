#: A lazy data pipeline over a synthetic request log.
#:
#: The point is `std.iter`: every stage below is an adapter that pulls one
#: element at a time, so a pipeline ending in `find` stops the moment it has an
#: answer instead of building the intermediate lists first. The last section
#: proves it by counting how many records the source actually yielded.
#:
#: `std.collections` supplies the parts that genuinely need to remember
#: everything at once: `Counter` for tallies, `DefaultDict` for grouping, and
#: `nlargest` for a top-k that never sorts the whole log.
#:
#: The log is generated from a fixed seed, so the numbers are the same on every
#: run and on every machine — it is synthetic data, not a recording.
#:
#:     jaithon run examples/pipeline.jai [request-count]
import std.iter
from std.collections import Counter, DefaultDict, nlargest
from std.random import Random

let ENDPOINTS = ["/health", "/search", "/checkout", "/media/upload"]

# Uploads are slow on purpose: a pipeline that cannot tell a slow endpoint from
# a broken one is not measuring anything.
let BASE_MILLIS = {"/health": 4, "/search": 45, "/checkout": 120, "/media/upload": 900}

# At or above this, a request counts as slow in the report below.
const SLOW_MILLIS = 500

# How many consecutive requests the rolling average covers.
const WINDOW = 8

#: One line of the log.
class Request {
    pub var endpoint: str
    pub var millis: int
    pub var status: int

    fn init(self, endpoint: str, millis: int, status: int) {
        self.endpoint = endpoint
        self.millis = millis
        self.status = status
    }

    #: A 5xx is the server's fault; a 4xx is the caller's and is not an outage.
    pub fn is_error(self) -> bool { return self.status >= 500 }

    fn __str__(self) -> str {
        let time = self.millis.to_str().pad_left(5)
        return f"{self.endpoint.pad_right(14)} {time}ms  {self.status}"
    }
}

#: Counts how many times a lazy pipeline pulled from its source.
#:
#: A captured `let` is copied into the closure (spec §6), so the tally lives in
#: a field: mutating through `self` is what makes it visible to the caller.
class Meter {
    pub var count: int

    fn init(self) { self.count = 0 }

    pub fn tick(self) -> void { self.count += 1 }
}

#: Build `count` requests from `rng`.
#:
#: Latency is the endpoint's base time plus a heavy tail, so the slowest few
#: per cent land far above the median — which is what makes a rolling average
#: worth computing rather than one mean over everything.
pub fn synthesize(count: int, rng: Random) -> list[Request] {
    var log: list[Request] = []
    for _index in 0..count {
        let endpoint = rng.choice(ENDPOINTS)
        let base = BASE_MILLIS[endpoint]
        let spike = if rng.bool(0.06) { rng.int_range(3, 12) } else { 1 }
        let millis = base * spike + rng.int_range(0, base + 1)
        # Failure tracks latency instead of being sprinkled independently: the
        # slow request is the one that times out.
        let status = if rng.bool(0.02 + millis / 40000.0) { 503 } else { 200 }
        log.push(Request(endpoint, millis, status))
    }
    return log
}

#: Print how many requests each endpoint served and how many failed.
pub fn report_traffic(log: list[Request]) -> void {
    let by_endpoint = Counter[str].from_list(iter.iter(log).map(|r| r.endpoint).collect())
    let failures = Counter[str].from_list(
        iter.iter(log).filter(|r| r.is_error()).map(|r| r.endpoint).collect()
    )

    print("endpoint         requests  failed    rate")
    for (endpoint, total) in by_endpoint.most_common() {
        let failed = failures.get(endpoint)
        let rate = (100.0 * failed / total).round(1)
        print(
            f"{endpoint.pad_right(14)} {total.to_str().pad_left(9)} " +
                f"{failed.to_str().pad_left(7)} {rate.to_str().pad_left(7)}%"
        )
    }
    print("")
}

#: Print the slowest run of `WINDOW` consecutive requests.
#:
#: `windows` yields overlapping slices lazily, so this walks the log once and
#: holds `WINDOW` elements at a time rather than materialising every slice.
pub fn report_worst_window(log: list[Request]) -> void {
    let averages = iter.iter(log).map(|r| r.millis).windows(WINDOW).map(|w| float(w.sum()) / float(w.len())).collect()
    if averages.len() == 0 {
        print(f"fewer than {WINDOW} requests; no rolling average\n")
        return
    }

    var worst_at = 0
    for (index, average) in averages.enumerate() {
        if average > averages[worst_at] { worst_at = index }
    }
    let mean = averages[worst_at].round(1)
    print(f"worst {WINDOW}-request window starts at #{worst_at}, mean {mean}ms")
    for request in log[worst_at:worst_at + WINDOW] { print(f"    {request}") }
    print("")
}

#: Print the slowest requests overall, and the slowest one per endpoint.
#:
#: `nlargest` keeps a heap of `k`, so a top-3 costs O(n log k) rather than the
#: O(n log n) of sorting a log that is then thrown away.
pub fn report_slowest(log: list[Request]) -> void {
    # Rank the requests themselves rather than their latencies: pulling the top
    # three numbers and then searching back for a request with each would pick
    # an arbitrary one of any tie.
    print("slowest requests overall")
    for request in nlargest(3, log, |a, b| a.millis < b.millis) {
        print(f"    {request}")
    }

    let grouped = DefaultDict[str, list[int]](|| [])
    for request in log { grouped.get(request.endpoint).push(request.millis) }

    print("slowest per endpoint")
    for endpoint in ENDPOINTS {
        let times = grouped.get(endpoint)
        if times.len() == 0 { continue }
        let worst = (times.max() ?? 0).to_str().pad_left(5)
        print(f"    {endpoint.pad_right(14)} {worst}ms")
    }
    print("")
}

#: Show that the pipeline stops pulling once it has an answer.
pub fn report_laziness(log: list[Request]) -> void {
    let found = Meter()
    let first = iter.iter(log).inspect(|_r| found.tick()).filter(|r| r.millis >= SLOW_MILLIS).find(
        |r| r.is_error()
    )

    if first == null {
        print(f"no slow failure in {log.len()} requests (pulled {found.count})")
    } else {
        print(f"first slow failure: {first}")
        print(f"    pulled {found.count} of {log.len()} requests to find it")
    }

    # The same source consumed to the end, to make the contrast concrete.
    let all = Meter()
    let slow = iter.iter(log).inspect(|_r| all.tick()).filter(|r| r.millis >= SLOW_MILLIS).count_items()
    print(f"    {slow} slow requests in total, pulled {all.count}")
}

fn main(args: list[str]) -> int {
    let count = if args.len() > 1 { args[1].to_int() ?? 400 } else { 400 }
    if count <= 0 {
        print("request-count must be positive")
        return 1
    }

    let log = synthesize(count, Random.with_seed(20260805))
    print(f"{log.len()} requests over {ENDPOINTS.len()} endpoints\n")

    report_traffic(log)
    report_worst_window(log)
    report_slowest(log)
    report_laziness(log)
    return 0
}
