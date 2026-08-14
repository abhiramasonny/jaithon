#: Durations, monotonic timing, and a calendar over the Unix epoch.
#:
#: Use `Instant` for elapsed time, never `DateTime`: the wall clock can jump
#: backwards (NTP, user changes), making elapsed-time math on it meaningless.
#:
#: ```
#: DateTime.from_unix(0).to_iso()               # "1970-01-01T00:00:00Z"
#: DateTime.from_unix(951782400).to_iso()       # "2000-02-29T00:00:00Z"
#: Duration.from_millis(1500.0).to_str()          # "1.5s"
#: ```
from std.core import Comparable, Printable

const _NANOS_PER_MICRO = 1_000
const _NANOS_PER_MILLI = 1_000_000
const _NANOS_PER_SECOND = 1_000_000_000
const _SECONDS_PER_MINUTE = 60
const _SECONDS_PER_HOUR = 3_600
const _SECONDS_PER_DAY = 86_400
const _NANOS_PER_MINUTE = 60_000_000_000
const _NANOS_PER_HOUR = 3_600_000_000_000
const _NANOS_PER_DAY = 86_400_000_000_000
const _NANOS_MIN = -9223372036854775807 - 1

# int64 bounds aren't exactly representable as floats; this is the nearest safe magnitude.
const _NANOS_LIMIT_F: float = 9.2e18

let _MONTH_LENGTHS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

let _MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

let _DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

#: A signed span of time, held as a whole number of nanoseconds.
#:
#: Immutable (every operation returns a new one); a span past roughly +/-292
#: years raises `OverflowError` rather than wrapping.
#:
#: ```
#: Duration.from_minutes(1.5).to_str()                     # "1m 30s"
#: Duration.from_seconds(2.0).add(Duration.from_millis(500.0)).to_str()   # "2.5s"
#: ```
pub class Duration: Comparable, Printable {
    let nanos: int

    #: Create a duration of `nanos` nanoseconds, which may be negative.
    #:
    #: ```
    #: Duration(1_500_000_000).to_str()    # "1.5s"
    #: ```
    pub fn init(self, nanos: int) {
        if nanos == _NANOS_MIN {
            throw OverflowError(f"duration of {_NANOS_MIN}ns has no representable magnitude")
        }
        self.nanos = nanos
    }

    #: Return the zero duration.
    pub static fn zero() -> Duration { return Duration(0) }

    #: Create a duration of `nanos` nanoseconds.
    pub static fn from_nanos(nanos: int) -> Duration { return Duration(nanos) }

    #: Create a duration of `micros` microseconds, rounded to the nearest nanosecond.
    pub static fn from_micros(micros: float) -> Duration {
        return _from_unit(micros, _NANOS_PER_MICRO, "microseconds")
    }

    #: Create a duration of `millis` milliseconds, rounded to the nearest nanosecond.
    #:
    #: ```
    #: Duration.from_millis(250.0).to_str()    # "250ms"
    #: ```
    pub static fn from_millis(millis: float) -> Duration {
        return _from_unit(millis, _NANOS_PER_MILLI, "milliseconds")
    }

    #: Create a duration of `seconds` seconds, rounded to the nearest nanosecond.
    pub static fn from_seconds(seconds: float) -> Duration {
        return _from_unit(seconds, _NANOS_PER_SECOND, "seconds")
    }

    #: Create a duration of `minutes` minutes, rounded to the nearest nanosecond.
    pub static fn from_minutes(minutes: float) -> Duration {
        return _from_unit(minutes, _NANOS_PER_MINUTE, "minutes")
    }

    #: Create a duration of `hours` hours, rounded to the nearest nanosecond.
    pub static fn from_hours(hours: float) -> Duration {
        return _from_unit(hours, _NANOS_PER_HOUR, "hours")
    }

    #: Create a duration of `days` days, where a day is exactly 24 hours.
    pub static fn from_days(days: float) -> Duration {
        return _from_unit(days, _NANOS_PER_DAY, "days")
    }

    #: Return the span in whole nanoseconds, exactly.
    pub fn to_nanos(self) -> int { return self.nanos }

    #: Return the span in microseconds, with a fractional part.
    pub fn to_micros(self) -> float { return float(self.nanos) / float(_NANOS_PER_MICRO) }

    #: Return the span in milliseconds, with a fractional part.
    pub fn to_millis(self) -> float { return float(self.nanos) / float(_NANOS_PER_MILLI) }

    #: Return the span in seconds, with a fractional part.
    #:
    #: ```
    #: Duration.from_millis(1500.0).to_seconds()    # 1.5
    #: ```
    pub fn to_seconds(self) -> float { return float(self.nanos) / float(_NANOS_PER_SECOND) }

    #: Return the span in minutes, with a fractional part.
    pub fn to_minutes(self) -> float { return float(self.nanos) / float(_NANOS_PER_MINUTE) }

    #: Return the span in hours, with a fractional part.
    pub fn to_hours(self) -> float { return float(self.nanos) / float(_NANOS_PER_HOUR) }

    #: Return the span in days of exactly 24 hours, with a fractional part.
    pub fn to_days(self) -> float { return float(self.nanos) / float(_NANOS_PER_DAY) }

    #: Return the whole seconds in the span, truncated towards zero.
    #:
    #: Together with `subsec_nanos`, splits the span without losing precision,
    #: unlike `to_seconds` past about 104 days.
    #:
    #: ```
    #: Duration.from_millis(-1500.0).whole_seconds()    # -1
    #: Duration.from_millis(-1500.0).subsec_nanos()     # -500000000
    #: ```
    pub fn whole_seconds(self) -> int {
        if self.nanos < 0 { return -(-self.nanos // _NANOS_PER_SECOND) }
        return self.nanos // _NANOS_PER_SECOND
    }

    #: Return the nanoseconds left over after `whole_seconds`, with the same sign.
    pub fn subsec_nanos(self) -> int {
        return self.nanos - self.whole_seconds() * _NANOS_PER_SECOND
    }

    #: Return `self + other`.
    pub fn add(self, other: Duration) -> Duration { return Duration(self.nanos + other.nanos) }

    #: Return `self - other`.
    pub fn sub(self, other: Duration) -> Duration { return Duration(self.nanos - other.nanos) }

    #: Return the span multiplied by `factor`, rounded to the nearest nanosecond.
    #:
    #: ```
    #: Duration.from_seconds(1.0).scale(2.5).to_str()    # "2.5s"
    #: ```
    pub fn scale(self, factor: float) -> Duration {
        return _from_unit(self.nanos * factor, 1, "nanoseconds")
    }

    #: Return the span divided by `divisor`, rounded to the nearest nanosecond.
    pub fn divide(self, divisor: float) -> Duration {
        if divisor == 0.0 { throw DivisionByZeroError("cannot divide a duration by zero") }
        return _from_unit(self.nanos / divisor, 1, "nanoseconds")
    }

    #: Return how many times `other` fits into `self`, with a fractional part.
    pub fn ratio(self, other: Duration) -> float {
        if other.nanos == 0 {
            throw DivisionByZeroError("cannot divide a duration by a zero duration")
        }
        return float(self.nanos) / float(other.nanos)
    }

    #: Return the span with its sign flipped.
    pub fn negate(self) -> Duration { return Duration(-self.nanos) }

    #: Return the span with its sign removed.
    pub fn absolute(self) -> Duration {
        if self.nanos < 0 { return Duration(-self.nanos) }
        return self
    }

    #: Whether the span is exactly zero.
    pub fn is_zero(self) -> bool { return self.nanos == 0 }

    #: Whether the span points backwards.
    pub fn is_negative(self) -> bool { return self.nanos < 0 }

    #: Order two spans by length, negative spans first.
    pub fn compare(self, other: Duration) -> int {
        if self.nanos < other.nanos { return -1 }
        if self.nanos > other.nanos { return 1 }
        return 0
    }

    #: Render the span for a human reader.
    #:
    #: ```
    #: Duration.from_nanos(1500).to_str()          # "1.5µs"
    #: Duration.from_millis(250.0).to_str()          # "250ms"
    #: Duration.from_seconds(90.0).to_str()        # "1m 30s"
    #: Duration.from_hours(25.0).to_str()          # "1d 1h 0m 0s"
    #: Duration.from_seconds(-1.5).to_str()        # "-1.5s"
    #: Duration.zero().to_str()                    # "0s"
    #: ```
    pub fn to_str(self) -> str {
        if self.nanos == 0 { return "0s" }
        let sign = if self.nanos < 0 { "-" } else { "" }
        let magnitude = if self.nanos < 0 { -self.nanos } else { self.nanos }
        if magnitude < _NANOS_PER_MICRO { return f"{sign}{magnitude}ns" }
        if magnitude < _NANOS_PER_MILLI {
            return sign + _scaled(magnitude, _NANOS_PER_MICRO, 3, "µs")
        }
        if magnitude < _NANOS_PER_SECOND {
            return sign + _scaled(magnitude, _NANOS_PER_MILLI, 6, "ms")
        }
        var rest = magnitude
        let days = rest // _NANOS_PER_DAY
        rest -= days * _NANOS_PER_DAY
        let hours = rest // _NANOS_PER_HOUR
        rest -= hours * _NANOS_PER_HOUR
        let minutes = rest // _NANOS_PER_MINUTE
        rest -= minutes * _NANOS_PER_MINUTE
        var parts: list[str] = []
        if days > 0 { parts.push(f"{days}d") }
        if parts.len() > 0 or hours > 0 { parts.push(f"{hours}h") }
        if parts.len() > 0 or minutes > 0 { parts.push(f"{minutes}m") }
        parts.push(_scaled(rest, _NANOS_PER_SECOND, 9, "s"))
        return sign + " ".join(parts)
    }

    fn __add__(self, other: Duration) -> Duration { return self.add(other) }

    fn __sub__(self, other: Duration) -> Duration { return self.sub(other) }

    fn __mul__(self, factor: float) -> Duration { return self.scale(factor) }

    fn __neg__(self) -> Duration { return self.negate() }

    fn __eq__(self, other: any) -> bool {
        if not isinstance(other, Duration) { return false }
        return self.nanos == other.to_nanos()
    }

    fn __lt__(self, other: Duration) -> bool { return self.nanos < other.to_nanos() }

    fn __le__(self, other: Duration) -> bool { return self.nanos <= other.to_nanos() }

    fn __hash__(self) -> int { return hash(self.nanos) }

    fn __str__(self) -> str { return self.to_str() }

    fn __repr__(self) -> str { return f"Duration({self.nanos})" }
}

#: A reading of the monotonic clock, for measuring elapsed time.
#:
#: Its origin is arbitrary and differs between runs, so only compare it with
#: another `Instant` from the same process; it never moves backwards.
#:
#: ```
#: let start = Instant.now()
#: start.elapsed().is_negative()    # false
#: ```
pub class Instant: Comparable, Printable {
    let nanos: int

    #: Adopt a raw monotonic reading of `nanos` nanoseconds.
    #:
    #: For reconstructing a reading carried across a channel or checkpoint;
    #: use `Instant.now()` for a fresh one.
    pub fn init(self, nanos: int) { self.nanos = nanos }

    #: Read the monotonic clock.
    pub static fn now() -> Instant { return Instant(_mono_nanos()) }

    #: Return the raw monotonic reading in nanoseconds.
    pub fn to_nanos(self) -> int { return self.nanos }

    #: Return how long ago `self` was read.
    #:
    #: Never negative: the underlying clock never moves backwards.
    pub fn elapsed(self) -> Duration { return Duration(_mono_nanos() - self.nanos) }

    #: Return the span from `earlier` to `self`, negative when `earlier` is later.
    pub fn duration_since(self, earlier: Instant) -> Duration {
        return Duration(self.nanos - earlier.to_nanos())
    }

    #: Return the instant `duration` after `self`.
    pub fn plus(self, duration: Duration) -> Instant {
        return Instant(self.nanos + duration.to_nanos())
    }

    #: Return the instant `duration` before `self`.
    pub fn minus(self, duration: Duration) -> Instant {
        return Instant(self.nanos - duration.to_nanos())
    }

    #: Order two readings, the earlier one first.
    pub fn compare(self, other: Instant) -> int {
        if self.nanos < other.to_nanos() { return -1 }
        if self.nanos > other.to_nanos() { return 1 }
        return 0
    }

    #: Render the reading as `Instant(<nanos>ns)`.
    pub fn to_str(self) -> str { return f"Instant({self.nanos}ns)" }

    fn __sub__(self, other: Instant) -> Duration { return self.duration_since(other) }

    fn __eq__(self, other: any) -> bool {
        if not isinstance(other, Instant) { return false }
        return self.nanos == other.to_nanos()
    }

    fn __lt__(self, other: Instant) -> bool { return self.nanos < other.to_nanos() }

    fn __hash__(self) -> int { return hash(self.nanos) }

    fn __str__(self) -> str { return self.to_str() }
}

#: An accumulating timer that can be stopped, resumed, and split into laps.
#:
#: Unlike `Instant.elapsed` (one span), sums a total across several spans —
#: e.g. profiling a function called in a loop.
#:
#: ```
#: let watch = Stopwatch.start_new()
#: let first = watch.lap()
#: let second = watch.lap()
#: watch.laps().len()                          # 2
#: watch.elapsed().ge(first.add(second))       # true
#: ```
pub class Stopwatch: Printable {
    var started_at: Instant?
    var accumulated: Duration
    var splits: list[Duration]
    var lapped: Duration

    #: Create a stopwatch that is not running and reads zero.
    pub fn init(self) {
        self.started_at = null
        self.accumulated = Duration.zero()
        self.splits = []
        self.lapped = Duration.zero()
    }

    #: Create a stopwatch that is already running.
    pub static fn start_new() -> Stopwatch {
        let watch = Stopwatch()
        watch.start()
        return watch
    }

    #: Start counting, or keep counting if already running.
    pub fn start(self) -> void {
        if self.started_at is null { self.started_at = Instant.now() }
    }

    #: Stop counting and return the total elapsed so far.
    #:
    #: Stopping a stopped watch changes nothing and returns the same total.
    pub fn stop(self) -> Duration {
        let started = self.started_at
        if started is not null {
            self.accumulated = self.accumulated.add(started.elapsed())
            self.started_at = null
        }
        return self.accumulated
    }

    #: Return the total counted so far, whether running or stopped.
    pub fn elapsed(self) -> Duration {
        let started = self.started_at
        if started is null { return self.accumulated }
        return self.accumulated.add(started.elapsed())
    }

    #: Whether the stopwatch is currently counting.
    pub fn is_running(self) -> bool { return self.started_at is not null }

    #: Record a split and return the span since the previous lap.
    pub fn lap(self) -> Duration {
        let total = self.elapsed()
        let split = total.sub(self.lapped)
        self.lapped = total
        self.splits.push(split)
        return split
    }

    #: Return a copy of the splits recorded by `lap`, oldest first.
    pub fn laps(self) -> list[Duration] { return [split for split in self.splits] }

    #: Stop the stopwatch and clear the total and the laps.
    pub fn reset(self) -> void {
        self.started_at = null
        self.accumulated = Duration.zero()
        self.splits = []
        self.lapped = Duration.zero()
    }

    #: Return the total counted so far and restart from zero.
    pub fn restart(self) -> Duration {
        let total = self.elapsed()
        self.reset()
        self.start()
        return total
    }

    #: Render the current total the way `Duration.to_str` does.
    pub fn to_str(self) -> str { return self.elapsed().to_str() }

    fn __str__(self) -> str { return self.to_str() }
}

#: A civil date and time at a fixed offset from UTC.
#:
#: Two values naming the same instant at different offsets compare equal.
#: Calendar is proleptic Gregorian, rules projected back before 1582 (as ISO 8601 does).
#:
#: ```
#: let moment = DateTime.of(2024, 2, 29, 12, 30, 15)
#: moment.to_iso()             # "2024-02-29T12:30:15Z"
#: moment.day_of_year          # 60
#: moment.weekday              # 3
#: moment.to_unix()            # 1709209815
#: ```
pub class DateTime: Comparable, Printable {
    #: Whole seconds since 1970-01-01T00:00:00Z, negative before it.
    pub let unix_second: int

    #: Nanoseconds past `unix_second`, always in 0..999999999.
    pub let nanosecond: int

    #: Minutes east of UTC that the calendar fields are expressed in.
    pub let offset_minutes: int

    #: The proleptic Gregorian year.
    pub let year: int

    #: The month, 1 for January through 12 for December.
    pub let month: int

    #: The day of the month, from 1.
    pub let day: int

    #: The hour of the day, 0 through 23.
    pub let hour: int

    #: The minute of the hour, 0 through 59.
    pub let minute: int

    #: The second of the minute, 0 through 59.
    pub let second: int

    #: The day of the week, 0 for Monday through 6 for Sunday.
    pub let weekday: int

    #: The day of the year, 1 through 366.
    pub let day_of_year: int

    #: Create the moment `unix_second` seconds and `nanosecond` nanoseconds
    #: after the epoch, rendered at `offset_minutes` east of UTC.
    #:
    #: A `nanosecond` outside 0..999999999 is carried into the seconds —
    #: arithmetic can hand over an unnormalised pair.
    #:
    #: ```
    #: DateTime(0, 0, 0).to_iso()      # "1970-01-01T00:00:00Z"
    #: DateTime(0, 0, 60).to_iso()     # "1970-01-01T01:00:00+01:00"
    #: ```
    pub fn init(self, unix_second: int, nanosecond: int = 0, offset_minutes: int = 0) {
        if offset_minutes <= -1440 or offset_minutes >= 1440 {
            throw ValueError(f"offset must be within +/-1439 minutes, got {offset_minutes}")
        }
        let carry = _floor_div(nanosecond, _NANOS_PER_SECOND)
        self.unix_second = unix_second + carry
        self.nanosecond = nanosecond - carry * _NANOS_PER_SECOND
        self.offset_minutes = offset_minutes

        let local = self.unix_second + offset_minutes * _SECONDS_PER_MINUTE
        let days = _floor_div(local, _SECONDS_PER_DAY)
        let within = local - days * _SECONDS_PER_DAY
        let (year, month, day) = _civil_from_days(days)
        self.year = year
        self.month = month
        self.day = day
        self.hour = within // _SECONDS_PER_HOUR
        self.minute = (within - self.hour * _SECONDS_PER_HOUR) // _SECONDS_PER_MINUTE
        self.second = within - self.hour * _SECONDS_PER_HOUR - self.minute * _SECONDS_PER_MINUTE
        # 1970-01-01 was a Thursday, which is index 3 when Monday is 0.
        self.weekday = _floor_mod(days + 3, 7)
        self.day_of_year = days - _days_from_civil(year, 1, 1) + 1
    }

    #: Read the wall clock.
    #:
    #: Result is UTC — there's no time-zone database to find a local offset;
    #: pass one to `with_offset` for local rendering.
    pub static fn now() -> DateTime {
        let nanos = _wall_nanos()
        let seconds = _floor_div(nanos, _NANOS_PER_SECOND)
        return DateTime(seconds, nanos - seconds * _NANOS_PER_SECOND, 0)
    }

    #: Create the moment `seconds` after the epoch, rendered at `offset_minutes`.
    #:
    #: ```
    #: DateTime.from_unix(1704067199).to_iso()    # "2023-12-31T23:59:59Z"
    #: DateTime.from_unix(-2208988800).to_iso()   # "1900-01-01T00:00:00Z"
    #: ```
    pub static fn from_unix(seconds: int, offset_minutes: int = 0) -> DateTime {
        return DateTime(seconds, 0, offset_minutes)
    }

    #: Create a moment from calendar fields, which must name a real date.
    #:
    #: ```
    #: DateTime.of(2000, 2, 29).to_iso()             # "2000-02-29T00:00:00Z"
    #: DateTime.of(2024, 1, 1, 0, 0, 0, 0, 330).to_unix()   # 1704047400
    #: ```
    pub static fn of(
        year: int,
        month: int,
        day: int,
        hour: int = 0,
        minute: int = 0,
        second: int = 0,
        nanosecond: int = 0,
        offset_minutes: int = 0
    ) -> DateTime {
        if month < 1 or month > 12 {
            throw ValueError(f"month must be 1 to 12, got {month}")
        }
        let last = days_in_month(year, month)
        if day < 1 or day > last {
            throw ValueError(f"{year}-{_pad(month, 2)} has {last} days, got day {day}")
        }
        if hour < 0 or hour > 23 { throw ValueError(f"hour must be 0 to 23, got {hour}") }
        if minute < 0 or minute > 59 { throw ValueError(f"minute must be 0 to 59, got {minute}") }
        if second < 0 or second > 59 { throw ValueError(f"second must be 0 to 59, got {second}") }
        if nanosecond < 0 or nanosecond >= _NANOS_PER_SECOND {
            throw ValueError(f"nanosecond must be 0 to 999999999, got {nanosecond}")
        }
        let time_of_day = hour * _SECONDS_PER_HOUR + minute * _SECONDS_PER_MINUTE + second
        let local = _days_from_civil(year, month, day) * _SECONDS_PER_DAY + time_of_day
        return DateTime(local - offset_minutes * _SECONDS_PER_MINUTE, nanosecond, offset_minutes)
    }

    #: Parse an ISO 8601 timestamp.
    #:
    #: ```
    #: DateTime.from_iso("2024-02-29T12:00:00Z").to_unix()        # 1709208000
    #: DateTime.from_iso("2024-02-29T12:00:00+02:00").to_unix()   # 1709200800
    #: DateTime.from_iso("2024-02-29").to_iso()                   # "2024-02-29T00:00:00Z"
    #: DateTime.from_iso("1970-01-01T00:00:00.5Z").nanosecond     # 500000000
    #: ```
    pub static fn from_iso(text: str) -> DateTime {
        let source = text.strip()
        if source.len() < 10 or source[4] != "-" or source[7] != "-" {
            throw ParseError(f"expected an ISO 8601 timestamp, got {repr(text)}")
        }
        let year = _digits_at(source, 0, 4, text)
        let month = _digits_at(source, 5, 2, text)
        let day = _digits_at(source, 8, 2, text)
        var index = 10
        var hour = 0
        var minute = 0
        var second = 0
        var nanosecond = 0
        var offset = 0
        if index < source.len() and (source[index] == "T" or source[index] == " ") {
            index += 1
            hour = _digits_at(source, index, 2, text)
            index += 2
            if index >= source.len() or source[index] != ":" {
                throw ParseError(f"expected `:` after the hour in {repr(text)}")
            }
            index += 1
            minute = _digits_at(source, index, 2, text)
            index += 2
            if index < source.len() and source[index] == ":" {
                index += 1
                second = _digits_at(source, index, 2, text)
                index += 2
                if index < source.len() and source[index] == "." {
                    index += 1
                    let start = index
                    while index < source.len() and _is_digit(source[index]) { index += 1 }
                    if index == start {
                        throw ParseError(f"expected digits after the decimal point in {repr(text)}")
                    }
                    nanosecond = _fraction_nanos(source[start:index])
                }
            }
        }
        if index < source.len() {
            let mark = source[index]
            if mark == "Z" or mark == "z" {
                index += 1
            } elif mark == "+" or mark == "-" {
                index += 1
                let sign = if mark == "-" { -1 } else { 1 }
                let offset_hour = _digits_at(source, index, 2, text)
                index += 2
                var offset_minute = 0
                if index < source.len() and source[index] == ":" { index += 1 }
                if index < source.len() {
                    offset_minute = _digits_at(source, index, 2, text)
                    index += 2
                }
                offset = sign * (offset_hour * 60 + offset_minute)
            } else {
                throw ParseError(f"expected `Z` or a UTC offset in {repr(text)}")
            }
        }
        if index != source.len() {
            throw ParseError(f"trailing characters after the timestamp in {repr(text)}")
        }
        return DateTime.of(year, month, day, hour, minute, second, nanosecond, offset)
    }

    #: Return the whole seconds since the epoch, dropping the nanosecond field.
    #:
    #: ```
    #: DateTime.of(1970, 1, 1).to_unix()    # 0
    #: ```
    pub fn to_unix(self) -> int { return self.unix_second }

    #: Return the nanoseconds since the epoch.
    #:
    #: Raises `OverflowError` outside 1677-09-21 to 2262-04-11 (int64 range).
    pub fn to_unix_nanos(self) -> int {
        return self.unix_second * _NANOS_PER_SECOND + self.nanosecond
    }

    #: Render the moment as an ISO 8601 timestamp.
    #:
    #: ```
    #: DateTime.of(2024, 12, 31, 23, 59, 59).to_iso()   # "2024-12-31T23:59:59Z"
    #: DateTime.of(2024, 1, 1, 0, 0, 0, 500000000).to_iso()
    #: # "2024-01-01T00:00:00.5Z"
    #: ```
    pub fn to_iso(self) -> str {
        var text = self.format("%Y-%m-%dT%H:%M:%S")
        if self.nanosecond != 0 {
            text = text + "." + _trim_zeros(_pad(self.nanosecond, 9))
        }
        if self.offset_minutes == 0 { return text + "Z" }
        return text + _offset_text(self.offset_minutes, true)
    }

    #: Render the moment with strftime-style specifiers.
    #:
    #: | Spec | Meaning | Spec | Meaning |
    #: |---|---|---|---|
    #: | `%Y` | year, at least four digits | `%y` | year modulo 100, two digits |
    #: | `%m` | month, 01 to 12 | `%B` | month name |
    #: | `%b` | month name, three letters | `%d` | day, 01 to 31 |
    #: | `%e` | day, space padded | `%j` | day of year, 001 to 366 |
    #: | `%H` | hour, 00 to 23 | `%I` | hour, 01 to 12 |
    #: | `%p` | `AM` or `PM` | `%M` | minute, 00 to 59 |
    #: | `%S` | second, 00 to 59 | `%f` | microsecond, six digits |
    #: | `%N` | nanosecond, nine digits | `%A` | weekday name |
    #: | `%a` | weekday name, three letters | `%u` | weekday, 1 is Monday |
    #: | `%w` | weekday, 0 is Sunday | `%z` | offset as `+HHMM` |
    #: | `%Z` | `UTC` or the offset | `%s` | seconds since the epoch |
    #: | `%F` | same as `%Y-%m-%d` | `%T` | same as `%H:%M:%S` |
    #: | `%R` | same as `%H:%M` | `%D` | same as `%m/%d/%y` |
    #: | `%n` | newline | `%t` | tab |
    #: | `%%` | a literal `%` | | |
    #:
    #: An unknown specifier raises `ValueError` instead of passing through
    #: silently — a silent `%q` typo would otherwise reach production.
    #:
    #: ```
    #: let moment = DateTime.of(2024, 2, 29, 13, 5, 9)
    #: moment.format("%Y-%m-%d %H:%M:%S")    # "2024-02-29 13:05:09"
    #: moment.format("%A %e %B %Y")          # "Thursday 29 February 2024"
    #: moment.format("%I:%M %p")             # "01:05 PM"
    #: ```
    pub fn format(self, pattern: str) -> str {
        var parts: list[str] = []
        var index = 0
        while index < pattern.len() {
            let character = pattern[index]
            if character != "%" {
                parts.push(character)
                index += 1
                continue
            }
            if index + 1 >= pattern.len() {
                throw ValueError(f"format pattern ends with a lone `%`: {repr(pattern)}")
            }
            index += 2
            parts.push(self._specifier(pattern[index - 1], pattern))
        }
        return "".join(parts)
    }

    #: Return the moment `duration` later.
    #:
    #: ```
    #: DateTime.of(2024, 2, 28, 23, 0, 0).add(Duration.from_hours(2.0)).to_iso()
    #: # "2024-02-29T01:00:00Z"
    #: ```
    pub fn add(self, duration: Duration) -> DateTime {
        let nanos = duration.to_nanos()
        let seconds = _floor_div(nanos, _NANOS_PER_SECOND)
        return DateTime(
            self.unix_second + seconds,
            self.nanosecond + (nanos - seconds * _NANOS_PER_SECOND),
            self.offset_minutes
        )
    }

    #: Return the moment `duration` earlier.
    #:
    #: ```
    #: DateTime.of(2024, 3, 1).subtract(Duration.from_days(1.0)).to_iso()
    #: # "2024-02-29T00:00:00Z"
    #: ```
    pub fn subtract(self, duration: Duration) -> DateTime { return self.add(duration.negate()) }

    #: Return the span from `other` to `self`, negative when `other` is later.
    #:
    #: ```
    #: let end = DateTime.of(2024, 1, 1, 0, 1, 30)
    #: end.difference(DateTime.of(2024, 1, 1)).to_str()    # "1m 30s"
    #: ```
    pub fn difference(self, other: DateTime) -> Duration {
        return Duration(
            (self.unix_second - other.unix_second) * _NANOS_PER_SECOND + self.nanosecond -
                other.nanosecond
        )
    }

    #: Return the same instant rendered at `offset_minutes` east of UTC.
    #:
    #: ```
    #: let utc = DateTime.of(2024, 6, 1, 12, 0, 0)
    #: utc.with_offset(-300).to_iso()    # "2024-06-01T07:00:00-05:00"
    #: utc.with_offset(-300) == utc      # true
    #: ```
    pub fn with_offset(self, offset_minutes: int) -> DateTime {
        return DateTime(self.unix_second, self.nanosecond, offset_minutes)
    }

    #: Return the name of the weekday, `"Monday"` through `"Sunday"`.
    pub fn weekday_name(self) -> str { return _DAY_NAMES[self.weekday] }

    #: Return the name of the month, `"January"` through `"December"`.
    pub fn month_name(self) -> str { return _MONTH_NAMES[self.month - 1] }

    #: Whether the year of this moment has 366 days.
    pub fn is_leap_year(self) -> bool { return is_leap_year(self.year) }

    #: Order two moments on the timeline, ignoring the offset they render at.
    pub fn compare(self, other: DateTime) -> int {
        if self.unix_second != other.unix_second {
            return if self.unix_second < other.unix_second { -1 } else { 1 }
        }
        if self.nanosecond < other.nanosecond { return -1 }
        if self.nanosecond > other.nanosecond { return 1 }
        return 0
    }

    #: Render the moment as an ISO 8601 timestamp.
    pub fn to_str(self) -> str { return self.to_iso() }

    fn _specifier(self, code: str, pattern: str) -> str {
        return match code {
            "Y" => self.year < 0 ? "-" + _pad(-self.year, 4) : _pad(self.year, 4),
            "y" => _pad(_floor_mod(self.year, 100), 2),
            "m" => _pad(self.month, 2),
            "B" => self.month_name(),
            "b" => self.month_name()[0:3],
            "d" => _pad(self.day, 2),
            "e" => self.day < 10 ? " " + str(self.day) : str(self.day),
            "j" => _pad(self.day_of_year, 3),
            "H" => _pad(self.hour, 2),
            "I" => _pad(_twelve_hour(self.hour), 2),
            "p" => self.hour < 12 ? "AM" : "PM",
            "M" => _pad(self.minute, 2),
            "S" => _pad(self.second, 2),
            "f" => _pad(self.nanosecond // 1000, 6),
            "N" => _pad(self.nanosecond, 9),
            "A" => self.weekday_name(),
            "a" => self.weekday_name()[0:3],
            "u" => str(self.weekday + 1),
            "w" => str(_floor_mod(self.weekday + 1, 7)),
            "z" => _offset_text(self.offset_minutes, false),
            "Z" => self.offset_minutes == 0 ? "UTC" : _offset_text(self.offset_minutes, true),
            "s" => str(self.unix_second),
            "F" => self.format("%Y-%m-%d"),
            "T" => self.format("%H:%M:%S"),
            "R" => self.format("%H:%M"),
            "D" => self.format("%m/%d/%y"),
            "n" => "\n",
            "t" => "\t",
            "%" => "%",
            _ => throw ValueError(f"unknown format specifier `%{code}` in {repr(pattern)}"),
        }
    }

    fn __add__(self, duration: Duration) -> DateTime { return self.add(duration) }

    fn __sub__(self, duration: Duration) -> DateTime { return self.subtract(duration) }

    fn __eq__(self, other: any) -> bool {
        if not isinstance(other, DateTime) { return false }
        return self.unix_second == other.unix_second and self.nanosecond == other.nanosecond
    }

    fn __lt__(self, other: DateTime) -> bool { return self.compare(other) < 0 }

    fn __le__(self, other: DateTime) -> bool { return self.compare(other) <= 0 }

    fn __hash__(self) -> int { return hash(self.unix_second) ^ hash(self.nanosecond) }

    fn __str__(self) -> str { return self.to_iso() }
}

#: Whether `year` has 366 days in the proleptic Gregorian calendar.
#:
#: ```
#: is_leap_year(2024)    # true
#: is_leap_year(1900)    # false
#: is_leap_year(2000)    # true
#: ```
pub fn is_leap_year(year: int) -> bool {
    if _floor_mod(year, 400) == 0 { return true }
    if _floor_mod(year, 100) == 0 { return false }
    return _floor_mod(year, 4) == 0
}

#: Return the number of days in `month` of `year`.
#:
#: ```
#: days_in_month(2024, 2)    # 29
#: days_in_month(2023, 2)    # 28
#: days_in_month(2024, 12)   # 31
#: ```
pub fn days_in_month(year: int, month: int) -> int {
    if month < 1 or month > 12 { throw ValueError(f"month must be 1 to 12, got {month}") }
    if month == 2 and is_leap_year(year) { return 29 }
    return _MONTH_LENGTHS[month - 1]
}

#: Pause this thread for `duration`, or return at once when it is not positive.
#:
#: The sleep is a lower bound: the scheduler decides when the thread runs again.
pub fn sleep(duration: Duration) -> void {
    if duration.to_nanos() <= 0 { return }
    _sleep_seconds(duration.to_seconds())
}

#: Run `action` and return its result together with how long it took.
#:
#: Duration comes from the monotonic clock, unaffected by wall-clock changes
#: during the call. An exception from `action` propagates and produces no duration.
#:
#: ```
#: let (value, spent) = timed(|| 6 * 7)
#: value                     # 42
#: spent.is_negative()       # false
#: ```
pub fn timed(action: fn() -> any) -> tuple[any, Duration] {
    let start = Instant.now()
    let result = action()
    return (result, start.elapsed())
}

fn _mono_nanos() -> int { return __prim__.time_mono() }

fn _wall_nanos() -> int { return __prim__.time_wall() }

fn _sleep_seconds(seconds: float) -> void { __prim__.time_sleep(seconds) }

fn _from_unit(value: float, unit: int, name: str) -> Duration {
    let nanos = value * unit
    if nanos > _NANOS_LIMIT_F or nanos < -_NANOS_LIMIT_F {
        throw OverflowError(f"{value} {name} does not fit in a Duration")
    }
    return Duration(int(__prim__.f64_round(nanos)))
}

fn _scaled(value: int, unit: int, width: int, suffix: str) -> str {
    let whole = value // unit
    let fraction = value - whole * unit
    if fraction == 0 { return f"{whole}{suffix}" }
    let digits = _trim_zeros(_pad(fraction, width))
    return f"{whole}.{digits}{suffix}"
}

fn _pad(value: int, width: int) -> str {
    var text = str(value)
    while text.len() < width { text = "0" + text }
    return text
}

fn _trim_zeros(text: str) -> str {
    var end = text.len()
    while end > 1 and text[end - 1] == "0" { end -= 1 }
    return text[0:end]
}

fn _twelve_hour(hour: int) -> int {
    let wrapped = _floor_mod(hour, 12)
    return if wrapped == 0 { 12 } else { wrapped }
}

fn _offset_text(offset_minutes: int, colon: bool) -> str {
    let sign = if offset_minutes < 0 { "-" } else { "+" }
    let magnitude = if offset_minutes < 0 { -offset_minutes } else { offset_minutes }
    let hours = magnitude // 60
    let separator = if colon { ":" } else { "" }
    return sign + _pad(hours, 2) + separator + _pad(magnitude - hours * 60, 2)
}

fn _is_digit(character: str) -> bool {
    let code = ord(character)
    return code >= 48 and code <= 57
}

fn _digits_at(source: str, start: int, width: int, original: str) -> int {
    if start + width > source.len() {
        throw ParseError(f"expected {width} digits in {repr(original)}")
    }
    var value = 0
    for index in start..start + width {
        let character = source[index]
        if not _is_digit(character) {
            throw ParseError(f"expected a digit, got {repr(character)} in {repr(original)}")
        }
        value = value * 10 + (ord(character) - 48)
    }
    return value
}

fn _fraction_nanos(digits: str) -> int {
    var value = 0
    for index in 0..9 {
        let digit = if index < digits.len() { ord(digits[index]) - 48 } else { 0 }
        value = value * 10 + digit
    }
    return value
}

fn _days_from_civil(year: int, month: int, day: int) -> int {
    let shifted = if month <= 2 { year - 1 } else { year }
    let era = _floor_div(shifted, 400)
    let year_of_era = shifted - era * 400
    let shifted_month = if month > 2 { month - 3 } else { month + 9 }
    let day_of_year = (153 * shifted_month + 2) // 5 + day - 1
    let day_of_era = year_of_era * 365 + year_of_era // 4 - year_of_era // 100 + day_of_year
    return era * 146097 + day_of_era - 719468
}

fn _civil_from_days(days: int) -> tuple[int, int, int] {
    let shifted = days + 719468
    let era = _floor_div(shifted, 146097)
    let day_of_era = shifted - era * 146097
    let year_of_era = (day_of_era - day_of_era // 1460 + day_of_era // 36524 - day_of_era // 146096) //
        365
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era // 4 - year_of_era // 100)
    let shifted_month = (5 * day_of_year + 2) // 153
    let day = day_of_year - (153 * shifted_month + 2) // 5 + 1
    let month = if shifted_month < 10 { shifted_month + 3 } else { shifted_month - 9 }
    let year = year_of_era + era * 400 + (if month <= 2 { 1 } else { 0 })
    return (year, month, day)
}

fn _floor_div(value: int, divisor: int) -> int {
    let quotient = value // divisor
    if quotient * divisor > value { return quotient - 1 }
    return quotient
}

fn _floor_mod(value: int, divisor: int) -> int {
    return value - _floor_div(value, divisor) * divisor
}
