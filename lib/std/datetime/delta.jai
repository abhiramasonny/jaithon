#: `timedelta`, a signed span held as normalised days, seconds and microseconds.
#:
#: Normalisation is CPython's: `0 <= microseconds < 1000000`,
#: `0 <= seconds < 86400`, and `-999999999 <= days <= 999999999`, so a
#: negative span carries its sign entirely in `days`. That is why
#: `timedelta(seconds=-1)` reads as `-1 day, 23:59:59`.
from .common import MAX_DELTA_DAYS, MICROS_PER_SECOND, SECONDS_PER_DAY, pad

#: A signed span of time, to microsecond resolution.
#:
#: ```
#: timedelta(days: 1, seconds: 2).total_seconds()    # 86402.0
#: str(timedelta(seconds: -1))                       # "-1 day, 23:59:59"
#: repr(timedelta(weeks: 2))                         # "datetime.timedelta(days=14)"
#: ```
pub class timedelta {
    #: Whole days, the only field that carries the sign.
    pub let days: int

    #: Seconds past `days`, always in 0..86399.
    pub let seconds: int

    #: Microseconds past `seconds`, always in 0..999999.
    pub let microseconds: int

    #: The most negative span, `timedelta(days: -999999999)`.
    pub static var min: timedelta

    #: The largest span, `timedelta(days: 999999999, hours: 23, minutes: 59,
    #: seconds: 59, microseconds: 999999)`.
    pub static var max: timedelta

    #: The smallest difference between two unequal spans, one microsecond.
    pub static var resolution: timedelta

    #: Build a span from any mixture of the seven units, then normalise it.
    #:
    #: Every argument is a whole number and every argument may be negative;
    #: the sum is what is normalised, so `timedelta(days: 1, hours: -25)` is
    #: minus one hour.
    #:
    #: ```
    #: timedelta().days                        # 0
    #: timedelta(milliseconds: 1500).seconds   # 1
    #: timedelta(hours: -1).days               # -1
    #: ```
    pub fn init(
        self,
        days: int = 0,
        seconds: int = 0,
        microseconds: int = 0,
        milliseconds: int = 0,
        minutes: int = 0,
        hours: int = 0,
        weeks: int = 0
    ) {
        let raw_micros = microseconds + milliseconds * 1_000
        let carried_seconds = raw_micros // MICROS_PER_SECOND
        self.microseconds = raw_micros - carried_seconds * MICROS_PER_SECOND
        let raw_seconds = seconds + minutes * 60 + hours * 3_600 + carried_seconds
        let carried_days = raw_seconds // SECONDS_PER_DAY
        self.seconds = raw_seconds - carried_days * SECONDS_PER_DAY
        let total_days = days + weeks * 7 + carried_days
        if total_days < -MAX_DELTA_DAYS or total_days > MAX_DELTA_DAYS {
            throw OverflowError(f"days={total_days}; must have magnitude <= {MAX_DELTA_DAYS}")
        }
        self.days = total_days
    }

    #: Return the span as a float count of seconds.
    #:
    #: Rounding matches CPython's, which divides the exact microsecond total
    #: by a million rather than adding two floats: below the point where a
    #: microsecond total stops fitting a double exactly, that division is
    #: done in whole numbers.
    #:
    #: ```
    #: timedelta(days: 1, microseconds: 1).total_seconds()   # 86400.000001
    #: timedelta(microseconds: -1).total_seconds()           # -1e-06
    #: ```
    pub fn total_seconds(self) -> float {
        let whole = self.days * SECONDS_PER_DAY + self.seconds
        if whole >= -9_007_199_254 and whole <= 9_007_199_254 {
            return float(whole * MICROS_PER_SECOND + self.microseconds) / 1_000_000.0
        }
        return float(whole) + float(self.microseconds) / 1_000_000.0
    }

    #: Return the span with its sign flipped.
    pub fn negate(self) -> timedelta {
        return timedelta(-self.days, -self.seconds, -self.microseconds)
    }

    #: Return the span's magnitude.
    #:
    #: Named rather than spelled `abs(...)`: Jaithon's `abs` is a numeric
    #: builtin with no hook a class can answer.
    #:
    #: ```
    #: str(timedelta(seconds: -1).absolute())    # "0:00:01"
    #: ```
    pub fn absolute(self) -> timedelta {
        if self.days < 0 { return self.negate() }
        return self
    }

    #: Order two spans, returning a negative int, zero, or a positive int.
    pub fn compare_to(self, other: timedelta) -> int {
        if self.days != other.days { return if self.days < other.days { -1 } else { 1 } }
        if self.seconds != other.seconds {
            return if self.seconds < other.seconds { -1 } else { 1 }
        }
        if self.microseconds != other.microseconds {
            return if self.microseconds < other.microseconds { -1 } else { 1 }
        }
        return 0
    }

    fn __add__(self, other: timedelta) -> timedelta {
        return timedelta(
            self.days + other.days,
            self.seconds + other.seconds,
            self.microseconds + other.microseconds
        )
    }

    fn __sub__(self, other: timedelta) -> timedelta {
        return timedelta(
            self.days - other.days,
            self.seconds - other.seconds,
            self.microseconds - other.microseconds
        )
    }

    fn __mul__(self, factor: int) -> timedelta {
        return timedelta(self.days * factor, self.seconds * factor, self.microseconds * factor)
    }

    fn __neg__(self) -> timedelta { return self.negate() }

    fn __eq__(self, other: any) -> bool {
        if not isinstance(other, timedelta) { return false }
        return self.days == other.days and
            self.seconds == other.seconds and
            self.microseconds == other.microseconds
    }

    fn __lt__(self, other: timedelta) -> bool { return self.compare_to(other) < 0 }

    fn __le__(self, other: timedelta) -> bool { return self.compare_to(other) <= 0 }

    fn __hash__(self) -> int {
        return hash(self.days) ^ hash(self.seconds) ^ hash(self.microseconds)
    }

    fn __str__(self) -> str {
        let minutes = self.seconds // 60
        let hours = minutes // 60
        var text = f"{hours}:{pad(minutes - hours * 60, 2)}:{pad(self.seconds - minutes * 60, 2)}"
        if self.days != 0 {
            let plural = if self.days == 1 or self.days == -1 { "" } else { "s" }
            text = f"{self.days} day{plural}, " + text
        }
        if self.microseconds != 0 { text = text + "." + pad(self.microseconds, 6) }
        return text
    }

    fn __repr__(self) -> str {
        var parts: list[str] = []
        if self.days != 0 { parts.push(f"days={self.days}") }
        if self.seconds != 0 { parts.push(f"seconds={self.seconds}") }
        if self.microseconds != 0 { parts.push(f"microseconds={self.microseconds}") }
        if parts.len() == 0 { parts.push("0") }
        return "datetime.timedelta(" + ", ".join(parts) + ")"
    }
}

timedelta.min = timedelta(-MAX_DELTA_DAYS)
timedelta.max = timedelta(MAX_DELTA_DAYS, SECONDS_PER_DAY - 1, MICROS_PER_SECOND - 1)
timedelta.resolution = timedelta(0, 0, 1)
