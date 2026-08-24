#: Everything that only makes sense because a column holds timestamps.
#:
#: The storage is `column.jai`'s two-lane `Datetime`: milliseconds since the
#: Unix epoch, exact to 48 bits, recombined in `long` and never in `float`.
#: This module is what turns that integer into a calendar — fields, periods,
#: frequencies, bins — and back again.
#:
#: Three ideas carry the whole file.
#:
#: The first is that the calendar is arithmetic, not a lookup. A field
#: accessor is one kernel that divides the timestamp into a day number and a
#: millisecond of the day, runs Howard Hinnant's civil-from-days on the day
#: number, and reads the answer out. That kernel does its work in `int`: 48
#: bits of milliseconds is 3.2 million days, which fits an `int` with room to
#: spare, and a 32-bit division by a literal strength-reduces to a multiply
#: while the 64-bit one is a call. One `long` division per row buys every
#: field; nothing below it is 64-bit.
#:
#: The second is that a frequency is a period numbering. `bucket_of` maps each
#: timestamp to the index of the period it falls in, and that lane of codes is
#: what makes a resample a group-by: the same bins, the same aggregations, the
#: same result shape. Fixed frequencies — ms, s, min, H, D, W — divide by a
#: constant span; calendar ones — M, Q, A — divide a month index, because a
#: month is not a number of milliseconds and pretending otherwise is how a
#: monthly total quietly loses a day every February.
#:
#: The third is that a resample bucket and a time-offset window are the same
#: shape: a row range per output. Both are found by binary search over a
#: monotone lane and both are reduced by one kernel, so `resample("1D").mean()`
#: and `rolling("7D").mean()` cannot drift apart.
#:
#: Parsing and formatting are the exception. A string lives on the host and
#: there is no device path to one, so `to_datetime` and `dt_strftime` are host
#: loops — but they are host loops over the *distinct* strings, not over the
#: rows: a `Str` column is dictionary-encoded, so parsing one costs a parse per
#: level and a kernel over the codes.
#:
#: ```
#: let stamps = to_datetime(["2024-01-01", "2024-02-14", "2024-02-29"])
#: defer { stamps.free() }
#: dt_month(stamps).to_ints()          # [1, 2, 2]
#: dt_is_month_end(stamps).to_bools()  # [false, false, true]
#: parse_freq(5min).to_str()           # 5min
#: ```
import std.gpu as gpu
from std.gpu import Buffer, Kernel
from std.math import NAN, is_nan
from std.time import DateTime, days_in_month, is_leap_year
from .column import Agg, Column, DataFrame, Dtype, Index, Series, Value
from .column import MAX_ROWS, MISSING_POSITION, NA, WIDE_LIMIT, WIDE_SCALE
from .column import iota, positions_buffer, prefix_sum, wide_pack, wide_unpack

#: Milliseconds in the fixed spans. `Datetime` is datetime64[ms], so these are
#: whole numbers and the arithmetic over them is exact.
pub const MILLIS_PER_SECOND = 1000
pub const MILLIS_PER_MINUTE = 60000
pub const MILLIS_PER_HOUR = 3600000
pub const MILLIS_PER_DAY = 86400000
pub const MILLIS_PER_WEEK = 604800000

#: The code `bucket_of` gives a row whose timestamp is missing.
#:
#: Negative, so it can never collide with a bucket of a resample — those are
#: numbered from zero by construction.
pub const NO_BUCKET = -1

#: Threads in a window group, and the widest run one can stage.
#:
#: The same pair `reduce.jai` uses, for the same reason: 256 outputs whose
#: windows each span `w` rows touch `255 + w` rows between them, so a
#: 1024-float tile carries every window up to 769.
const WIN_GROUP = 256
const WIN_TILE = 1024

#: Where one thread per output stops paying and one threadgroup per output
#: starts.
#:
#: The thin kernel's parallelism is its output count; the group kernel's is
#: 256 times that but it pays eight barriers per fold. A monthly resample of
#: ten million rows is twelve outputs — twelve threads would leave the machine
#: idle — while a daily one is thousands and wants the tile. 512 rows an output
#: is the side of the crossover that keeps both cases off the wrong kernel.
const GROUP_PER_OUTPUT_ROWS = 512

#: The fixed grid of the exact min/max reduction, as in `column.jai`'s null
#: count: a grid that strides the column rather than one thread per row.
const MINMAX_GROUP = 256
const MINMAX_GROUPS = 256

const FIELD_YEAR = 0
const FIELD_MONTH = 1
const FIELD_DAY = 2
const FIELD_HOUR = 3
const FIELD_MINUTE = 4
const FIELD_SECOND = 5
const FIELD_MILLISECOND = 6
const FIELD_WEEKDAY = 7
const FIELD_DAYOFYEAR = 8
const FIELD_QUARTER = 9
const FIELD_WEEK = 10
const FIELD_DAYS_IN_MONTH = 11
const FIELD_MONTH_START = 12
const FIELD_MONTH_END = 13
const FIELD_QUARTER_START = 14
const FIELD_QUARTER_END = 15
const FIELD_YEAR_START = 16
const FIELD_YEAR_END = 17
const FIELD_NORMALIZE = 18
const FIELD_TIME = 19

const TS_SUM = 0
const TS_MEAN = 1
const TS_COUNT = 2
const TS_PROD = 3
const TS_STD = 4
const TS_VAR = 5
const TS_SEM = 6
const TS_ANY = 7
const TS_ALL = 8
const TS_SIZE = 9
const TS_ARGMIN = 10
const TS_ARGMAX = 11
const TS_ARGFIRST = 12
const TS_ARGLAST = 13

const MATCH_EXACT = 0
const MATCH_BACKWARD = 1
const MATCH_FORWARD = 2

#: Fixed span in milliseconds, or a month count.
const BUCKET_FIXED = 0
const BUCKET_MONTHS = 1

let SOURCE = """#include <metal_stdlib>
using namespace metal;

constexpr constant uint WIN_GROUP = 256;
constexpr constant uint WIN_TILE = 1024;
constexpr constant uint MINMAX_GROUP = 256;
constexpr constant uint MINMAX_GROUPS = 256;
constexpr constant long WIDE_SCALE = 16777216L;
constexpr constant long MS_PER_DAY = 86400000L;

/* Out of range for a timestamp by construction: the two lanes hold 48 bits and
 * this is 2^48, so a group that saw no present row reports a value the host
 * recognises rather than one it might believe. */
constexpr constant long EMPTY_LOW = 281474976710656L;
constexpr constant long EMPTY_HIGH = -281474976710656L;

constexpr constant uint FIELD_YEAR = 0u;
constexpr constant uint FIELD_MONTH = 1u;
constexpr constant uint FIELD_DAY = 2u;
constexpr constant uint FIELD_HOUR = 3u;
constexpr constant uint FIELD_MINUTE = 4u;
constexpr constant uint FIELD_SECOND = 5u;
constexpr constant uint FIELD_MILLISECOND = 6u;
constexpr constant uint FIELD_WEEKDAY = 7u;
constexpr constant uint FIELD_DAYOFYEAR = 8u;
constexpr constant uint FIELD_QUARTER = 9u;
constexpr constant uint FIELD_WEEK = 10u;
constexpr constant uint FIELD_DAYS_IN_MONTH = 11u;
constexpr constant uint FIELD_MONTH_START = 12u;
constexpr constant uint FIELD_MONTH_END = 13u;
constexpr constant uint FIELD_QUARTER_START = 14u;
constexpr constant uint FIELD_QUARTER_END = 15u;
constexpr constant uint FIELD_YEAR_START = 16u;
constexpr constant uint FIELD_YEAR_END = 17u;
constexpr constant uint FIELD_NORMALIZE = 18u;

constexpr constant uint TS_SUM = 0u;
constexpr constant uint TS_MEAN = 1u;
constexpr constant uint TS_COUNT = 2u;
constexpr constant uint TS_PROD = 3u;
constexpr constant uint TS_STD = 4u;
constexpr constant uint TS_VAR = 5u;
constexpr constant uint TS_SEM = 6u;
constexpr constant uint TS_ANY = 7u;
constexpr constant uint TS_ALL = 8u;
constexpr constant uint TS_SIZE = 9u;
constexpr constant uint TS_ARGMIN = 10u;
constexpr constant uint TS_ARGMAX = 11u;
constexpr constant uint TS_ARGFIRST = 12u;
constexpr constant uint TS_ARGLAST = 13u;

constexpr constant uint MATCH_EXACT = 0u;
constexpr constant uint MATCH_BACKWARD = 1u;
constexpr constant uint MATCH_FORWARD = 2u;

constexpr constant uint BUCKET_FIXED = 0u;

inline float missing_value() { return as_type<float>(0x7FC00000u); }

inline long wide_value(float low, float high) {
    return (long)high * WIDE_SCALE + (long)low;
}

inline void wide_store(device float* low, device float* high, uint at, long value) {
    long hi = value >> 24;
    low[at] = (float)(value - (hi << 24));
    high[at] = (float)hi;
}

/* A scalar arrives as the same two halves a lane holds, because a kernel
 * argument is a uint and a millisecond timestamp is not one. */
inline long wide_scalar(int low, int high) {
    return (long)high * WIDE_SCALE + (long)low;
}

/* Floor division, not the truncating kind. Every timestamp before 1970 is
 * negative, and a quotient that rounded toward zero would put each of them one
 * period late — a bug that never shows on modern data and is always wrong. */
inline long floor_div(long a, long b) {
    long q = a / b;
    return (a - q * b < 0) ? q - 1 : q;
}

inline int floor_div_i(int a, int b) {
    int q = a / b;
    return (a - q * b < 0) ? q - 1 : q;
}

inline int floor_mod_i(int a, int b) {
    return a - floor_div_i(a, b) * b;
}

/* Howard Hinnant's civil-from-days, in int rather than long.
 *
 * Each level takes one division and recovers its remainder by subtraction —
 * `doe` from `era`, `doy` from `yoe`, the day from the month — so the whole
 * calendar costs six divides and no modulo at all, which matters on a device
 * that has neither instruction and synthesises both. */
inline void civil_from_days(int z, thread int& y, thread int& m, thread int& d) {
    int shifted = z + 719468;
    int era = floor_div_i(shifted, 146097);
    int doe = shifted - era * 146097;
    int yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    int doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    int mp = (5 * doy + 2) / 153;
    d = doy - (153 * mp + 2) / 5 + 1;
    m = mp + ((mp < 10) ? 3 : -9);
    y = yoe + era * 400 + ((m <= 2) ? 1 : 0);
}

inline int days_from_civil(int y, int m, int d) {
    int yy = y - ((m <= 2) ? 1 : 0);
    int era = floor_div_i(yy, 400);
    int yoe = yy - era * 400;
    int doy = (153 * (m + ((m > 2) ? -3 : 9)) + 2) / 5 + d - 1;
    int doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + doe - 719468;
}

inline bool leap_year(int y) {
    return (floor_mod_i(y, 4) == 0) && ((floor_mod_i(y, 100) != 0) || (floor_mod_i(y, 400) == 0));
}

inline int month_length(int y, int m) {
    if (m == 2) return leap_year(y) ? 29 : 28;
    return ((m == 4) || (m == 6) || (m == 9) || (m == 11)) ? 30 : 31;
}

/* 1970-01-01 was a Thursday, which is 3 when Monday is 0. */
inline int weekday_of(int days) {
    return floor_mod_i(days + 3, 7);
}

/* A year has 53 ISO weeks when it starts on a Thursday, or when the year
 * before it was a leap year starting on a Wednesday. */
inline int iso_weeks_in(int y) {
    int a = floor_mod_i(y + floor_div_i(y, 4) - floor_div_i(y, 100) + floor_div_i(y, 400), 7);
    int p = y - 1;
    int b = floor_mod_i(p + floor_div_i(p, 4) - floor_div_i(p, 100) + floor_div_i(p, 400), 7);
    return ((a == 4) || (b == 3)) ? 53 : 52;
}

/* `doy` counts from 1 and `weekday` is 0 for Monday, so the numerator is at
 * least 4 and the division needs no floor. */
inline int iso_week_of(int y, int doy, int weekday) {
    int week = (doy - weekday + 9) / 7;
    if (week < 1) return iso_weeks_in(y - 1);
    if (week > iso_weeks_in(y)) return 1;
    return week;
}

/* Float32 reinterpreted so an unsigned compare is the float compare, as in
 * `reduce.jai`. Widened to long here so that one key type carries both a float
 * payload and an exact 48-bit timestamp, and one fold serves min, max, first
 * and last over either. */
inline long order_key(float value) {
    uint bits = as_type<uint>(value);
    uint flipped = ((bits & 0x80000000u) != 0u) ? ~bits : (bits | 0x80000000u);
    return (long)flipped;
}

kernel void ts_field(device const float* values [[buffer(0)]],
                     device const float* wide [[buffer(1)]],
                     device const float* valid [[buffer(2)]],
                     device float* out_values [[buffer(3)]],
                     device float* out_wide [[buffer(4)]],
                     device float* out_valid [[buffer(5)]],
                     constant uint& count [[buffer(6)]],
                     constant uint& has_valid [[buffer(7)]],
                     constant uint& wants_wide [[buffer(8)]],
                     constant uint& field [[buffer(9)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    bool present = (has_valid == 0u) || (valid[gid] != 0.0f);
    if (has_valid != 0u) out_valid[gid] = present ? 1.0f : 0.0f;
    if (!present) {
        out_values[gid] = 0.0f;
        if (wants_wide != 0u) out_wide[gid] = 0.0f;
        return;
    }
    long stamp = wide_value(values[gid], wide[gid]);
    /* The only 64-bit division in the file's hot path. Everything below it is
     * int arithmetic on a day number that cannot leave 32 bits. */
    long day_index = floor_div(stamp, MS_PER_DAY);
    int days = (int)day_index;
    int within = (int)(stamp - day_index * MS_PER_DAY);
    int y = 0;
    int m = 0;
    int d = 0;
    civil_from_days(days, y, m, d);
    long answer = 0;
    if (field == FIELD_YEAR) {
        answer = (long)y;
    } else if (field == FIELD_MONTH) {
        answer = (long)m;
    } else if (field == FIELD_DAY) {
        answer = (long)d;
    } else if (field <= FIELD_MILLISECOND) {
        int hour = within / 3600000;
        int rest = within - hour * 3600000;
        int minute = rest / 60000;
        rest = rest - minute * 60000;
        int second = rest / 1000;
        int milli = rest - second * 1000;
        if (field == FIELD_HOUR) answer = (long)hour;
        else if (field == FIELD_MINUTE) answer = (long)minute;
        else if (field == FIELD_SECOND) answer = (long)second;
        else answer = (long)milli;
    } else if (field == FIELD_WEEKDAY) {
        answer = (long)weekday_of(days);
    } else if (field == FIELD_DAYOFYEAR) {
        answer = (long)(days - days_from_civil(y, 1, 1) + 1);
    } else if (field == FIELD_QUARTER) {
        answer = (long)((m + 2) / 3);
    } else if (field == FIELD_WEEK) {
        int doy = days - days_from_civil(y, 1, 1) + 1;
        answer = (long)iso_week_of(y, doy, weekday_of(days));
    } else if (field == FIELD_DAYS_IN_MONTH) {
        answer = (long)month_length(y, m);
    } else if (field == FIELD_MONTH_START) {
        answer = (d == 1) ? 1 : 0;
    } else if (field == FIELD_MONTH_END) {
        answer = (d == month_length(y, m)) ? 1 : 0;
    } else if (field == FIELD_QUARTER_START) {
        bool opens = (d == 1) && ((m == 1) || (m == 4) || (m == 7) || (m == 10));
        answer = opens ? 1 : 0;
    } else if (field == FIELD_QUARTER_END) {
        bool closes = (d == month_length(y, m)) && ((m == 3) || (m == 6) || (m == 9) || (m == 12));
        answer = closes ? 1 : 0;
    } else if (field == FIELD_YEAR_START) {
        answer = ((d == 1) && (m == 1)) ? 1 : 0;
    } else if (field == FIELD_YEAR_END) {
        answer = ((d == 31) && (m == 12)) ? 1 : 0;
    } else if (field == FIELD_NORMALIZE) {
        answer = day_index * MS_PER_DAY;
    } else {
        answer = (long)within;
    }
    if (wants_wide != 0u) wide_store(out_values, out_wide, gid, answer);
    else out_values[gid] = (float)answer;
}

/* The period index of every timestamp: the lane that makes a resample a
 * group-by. A missing timestamp gets -1, which no resample bucket can be. */
kernel void ts_bucket(device const float* values [[buffer(0)]],
                      device const float* wide [[buffer(1)]],
                      device const float* valid [[buffer(2)]],
                      device float* codes [[buffer(3)]],
                      constant uint& count [[buffer(4)]],
                      constant uint& has_valid [[buffer(5)]],
                      constant uint& mode [[buffer(6)]],
                      constant int& span_low [[buffer(7)]],
                      constant int& span_high [[buffer(8)]],
                      constant int& origin_low [[buffer(9)]],
                      constant int& origin_high [[buffer(10)]],
                      constant int& months [[buffer(11)]],
                      constant int& shift [[buffer(12)]],
                      constant int& base [[buffer(13)]],
                      constant uint& closed_right [[buffer(14)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if ((has_valid != 0u) && (valid[gid] == 0.0f)) {
        codes[gid] = -1.0f;
        return;
    }
    long stamp = wide_value(values[gid], wide[gid]);
    /* A right-closed bin is (start, end], so a timestamp exactly on an edge
     * belongs to the bin below it. One millisecond off the front turns that
     * into the left-closed formula every other line here assumes. */
    if (closed_right != 0u) stamp -= 1;
    if (mode == BUCKET_FIXED) {
        long span = wide_scalar(span_low, span_high);
        long origin = wide_scalar(origin_low, origin_high);
        codes[gid] = (float)(floor_div(stamp - origin, span) - (long)base);
        return;
    }
    int days = (int)floor_div(stamp, MS_PER_DAY);
    int y = 0;
    int m = 0;
    int d = 0;
    civil_from_days(days, y, m, d);
    int month_index = y * 12 + (m - 1);
    codes[gid] = (float)(floor_div_i(month_index - shift, months) - base);
}

/* The instant a period is stamped with: its start, its right edge, or the
 * midnight of its last day — which is where pandas puts an `M` label. */
kernel void ts_period_stamp(device float* out_values [[buffer(0)]],
                            device float* out_wide [[buffer(1)]],
                            constant uint& count [[buffer(2)]],
                            constant uint& mode [[buffer(3)]],
                            constant int& span_low [[buffer(4)]],
                            constant int& span_high [[buffer(5)]],
                            constant int& origin_low [[buffer(6)]],
                            constant int& origin_high [[buffer(7)]],
                            constant int& months [[buffer(8)]],
                            constant int& shift [[buffer(9)]],
                            constant int& base [[buffer(10)]],
                            constant uint& edge [[buffer(11)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if (mode == BUCKET_FIXED) {
        long span = wide_scalar(span_low, span_high);
        long origin = wide_scalar(origin_low, origin_high);
        long start = origin + (long)((int)gid + base) * span;
        long stamp = start;
        if (edge == 1u) stamp = start + span;
        /* Edge 2 is the midnight that closes the period rather than the one
         * that opens the next: a weekly bin labelled on its Sunday, a monthly
         * one on the 31st. It is the label pandas prints and the one a reader
         * checks against a calendar. */
        if (edge == 2u) stamp = start + span - MS_PER_DAY;
        wide_store(out_values, out_wide, gid, stamp);
        return;
    }
    int period = (int)gid + base;
    int first_month = period * months + shift;
    int use = (edge == 0u) ? first_month : first_month + months;
    int y = floor_div_i(use, 12);
    int m = use - y * 12 + 1;
    int days = days_from_civil(y, m, 1);
    if (edge == 2u) days -= 1;
    wide_store(out_values, out_wide, gid, (long)days * MS_PER_DAY);
}

/* floor, ceiling and round onto a frequency's grid, in one kernel because the
 * three differ only in which of the two neighbouring edges they pick. */
kernel void ts_snap(device const float* values [[buffer(0)]],
                    device const float* wide [[buffer(1)]],
                    device const float* valid [[buffer(2)]],
                    device float* out_values [[buffer(3)]],
                    device float* out_wide [[buffer(4)]],
                    device float* out_valid [[buffer(5)]],
                    constant uint& count [[buffer(6)]],
                    constant uint& has_valid [[buffer(7)]],
                    constant uint& mode [[buffer(8)]],
                    constant int& span_low [[buffer(9)]],
                    constant int& span_high [[buffer(10)]],
                    constant int& origin_low [[buffer(11)]],
                    constant int& origin_high [[buffer(12)]],
                    constant int& months [[buffer(13)]],
                    constant int& shift [[buffer(14)]],
                    constant uint& how [[buffer(15)]],
                    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    bool present = (has_valid == 0u) || (valid[gid] != 0.0f);
    if (has_valid != 0u) out_valid[gid] = present ? 1.0f : 0.0f;
    if (!present) {
        out_values[gid] = 0.0f;
        out_wide[gid] = 0.0f;
        return;
    }
    long stamp = wide_value(values[gid], wide[gid]);
    long low = 0;
    long high = 0;
    long period = 0;
    if (mode == BUCKET_FIXED) {
        long span = wide_scalar(span_low, span_high);
        long origin = wide_scalar(origin_low, origin_high);
        period = floor_div(stamp - origin, span);
        low = origin + period * span;
        high = low + span;
    } else {
        int days = (int)floor_div(stamp, MS_PER_DAY);
        int y = 0;
        int m = 0;
        int d = 0;
        civil_from_days(days, y, m, d);
        int index = floor_div_i(y * 12 + (m - 1) - shift, months);
        period = (long)index;
        int first = index * months + shift;
        int fy = floor_div_i(first, 12);
        low = (long)days_from_civil(fy, first - fy * 12 + 1, 1) * MS_PER_DAY;
        int next = first + months;
        int ny = floor_div_i(next, 12);
        high = (long)days_from_civil(ny, next - ny * 12 + 1, 1) * MS_PER_DAY;
    }
    long answer = low;
    if (how == 1u) {
        answer = (stamp == low) ? low : high;
    } else if (how == 2u) {
        long below = stamp - low;
        long above = high - stamp;
        /* Ties to the even period, which is what a Timestamp.round does and
         * what keeps a long grid from drifting upward one edge at a time. */
        if (below > above) answer = high;
        else if (below == above) answer = ((period & 1L) == 0L) ? low : high;
    }
    wide_store(out_values, out_wide, gid, answer);
}

kernel void ts_shift(device const float* values [[buffer(0)]],
                     device const float* wide [[buffer(1)]],
                     device float* out_values [[buffer(2)]],
                     device float* out_wide [[buffer(3)]],
                     constant uint& count [[buffer(4)]],
                     constant int& delta_low [[buffer(5)]],
                     constant int& delta_high [[buffer(6)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    long moved = wide_value(values[gid], wide[gid]) + wide_scalar(delta_low, delta_high);
    wide_store(out_values, out_wide, gid, moved);
}

/* Four rows at a time, for a column whose length divides by four. This kernel
 * moves thirty-two bytes and does one add, so the quartered thread count is
 * very nearly a quartered cost. */
kernel void ts_shift4(device const float4* values [[buffer(0)]],
                      device const float4* wide [[buffer(1)]],
                      device float4* out_values [[buffer(2)]],
                      device float4* out_wide [[buffer(3)]],
                      constant uint& quads [[buffer(4)]],
                      constant int& delta_low [[buffer(5)]],
                      constant int& delta_high [[buffer(6)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid >= quads) return;
    float4 low = values[gid];
    float4 high = wide[gid];
    long delta = wide_scalar(delta_low, delta_high);
    float4 out_low = float4(0.0f);
    float4 out_high = float4(0.0f);
    for (uint k = 0u; k < 4u; k++) {
        long moved = (long)high[k] * WIDE_SCALE + (long)low[k] + delta;
        long top = moved >> 24;
        out_low[k] = (float)(moved - (top << 24));
        out_high[k] = (float)top;
    }
    out_values[gid] = out_low;
    out_wide[gid] = out_high;
}

/* Adding months is not adding milliseconds: the 31st of January plus one month
 * is the 28th of February, because there is no 31st. Clamping to the shorter
 * month is what a DateOffset does and what a reader means. */
kernel void ts_shift_months(device const float* values [[buffer(0)]],
                            device const float* wide [[buffer(1)]],
                            device float* out_values [[buffer(2)]],
                            device float* out_wide [[buffer(3)]],
                            constant uint& count [[buffer(4)]],
                            constant int& months [[buffer(5)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    long stamp = wide_value(values[gid], wide[gid]);
    long day_index = floor_div(stamp, MS_PER_DAY);
    long within = stamp - day_index * MS_PER_DAY;
    int y = 0;
    int m = 0;
    int d = 0;
    civil_from_days((int)day_index, y, m, d);
    int moved = y * 12 + (m - 1) + months;
    int ny = floor_div_i(moved, 12);
    int nm = moved - ny * 12 + 1;
    int nd = min(d, month_length(ny, nm));
    wide_store(out_values, out_wide, gid,
               (long)days_from_civil(ny, nm, nd) * MS_PER_DAY + within);
}

/* An integer epoch in some other unit read as milliseconds. Dividing rather
 * than multiplying for the finer units, because microseconds and nanoseconds
 * carry digits datetime64[ms] does not keep and truncating them is the only
 * honest thing left. */
kernel void ts_rescale(device const float* values [[buffer(0)]],
                       device const float* wide [[buffer(1)]],
                       device float* out_values [[buffer(2)]],
                       device float* out_wide [[buffer(3)]],
                       constant uint& count [[buffer(4)]],
                       constant int& factor [[buffer(5)]],
                       constant uint& divide [[buffer(6)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    long raw = wide_value(values[gid], wide[gid]);
    long scaled = (divide != 0u) ? floor_div(raw, (long)factor) : raw * (long)factor;
    wide_store(out_values, out_wide, gid, scaled);
}

/* A parsed dictionary spread back over the rows.
 *
 * This is why parsing a `Str` column costs a parse per level and not per row:
 * the host parses the levels, uploads two lanes and a validity flag as long as
 * the dictionary, and the rows are one gather. */
kernel void ts_map_levels(device const float* codes [[buffer(0)]],
                          device const float* row_valid [[buffer(1)]],
                          device const float* level_low [[buffer(2)]],
                          device const float* level_high [[buffer(3)]],
                          device const float* level_valid [[buffer(4)]],
                          device float* out_values [[buffer(5)]],
                          device float* out_wide [[buffer(6)]],
                          device float* out_valid [[buffer(7)]],
                          constant uint& count [[buffer(8)]],
                          constant uint& has_valid [[buffer(9)]],
                          constant uint& levels [[buffer(10)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    int code = (int)codes[gid];
    bool present = (has_valid == 0u) || (row_valid[gid] != 0.0f);
    if (code < 0 || code >= (int)levels) present = false;
    int safe = (code < 0 || code >= (int)levels) ? 0 : code;
    if (present && level_valid[safe] == 0.0f) present = false;
    out_valid[gid] = present ? 1.0f : 0.0f;
    out_values[gid] = present ? level_low[safe] : 0.0f;
    out_wide[gid] = present ? level_high[safe] : 0.0f;
}

/* Where each bucket's rows begin and end, by binary search over the monotone
 * code lane.
 *
 * The two searches nest — every row of bucket `b` precedes every row of `b+1`
 * — so the second starts where the first stopped rather than at zero. That is
 * also why an empty bucket falls out for free: its two bounds coincide. */
kernel void ts_code_bounds(device const float* codes [[buffer(0)]],
                           device float* starts [[buffer(1)]],
                           device float* ends [[buffer(2)]],
                           constant uint& buckets [[buffer(3)]],
                           constant uint& rows [[buffer(4)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= buckets) return;
    float wanted = (float)gid;
    uint low = 0u;
    uint high = rows;
    while (low < high) {
        uint mid = low + ((high - low) >> 1);
        if (codes[mid] < wanted) low = mid + 1u;
        else high = mid;
    }
    uint first = low;
    high = rows;
    float next = wanted + 1.0f;
    while (low < high) {
        uint mid = low + ((high - low) >> 1);
        if (codes[mid] < next) low = mid + 1u;
        else high = mid;
    }
    starts[gid] = (float)first;
    ends[gid] = (float)low;
}

/* The row range a time-offset window covers, one thread per row.
 *
 * The bounds are monotone, so a single host merge would find all of them in
 * O(n) against this kernel's O(n log n) — but a merge is one thread, and a
 * host loop over more than a few thousand rows costs seconds in this language.
 * The searches of neighbouring threads share their first several levels, which
 * is exactly the part of the tree that stays in cache. */
kernel void ts_window_bounds(device const float* values [[buffer(0)]],
                             device const float* wide [[buffer(1)]],
                             device float* starts [[buffer(2)]],
                             device float* ends [[buffer(3)]],
                             constant uint& count [[buffer(4)]],
                             constant int& span_low [[buffer(5)]],
                             constant int& span_high [[buffer(6)]],
                             constant uint& open_left [[buffer(7)]],
                             constant uint& open_right [[buffer(8)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    long here = wide_value(values[gid], wide[gid]);
    long span = wide_scalar(span_low, span_high);
    /* An open left edge excludes a row exactly one span back, so the search
     * target moves up by the one millisecond the store can resolve. */
    long target = here - span + ((open_left != 0u) ? 1 : 0);
    uint low = 0u;
    uint high = gid + 1u;
    while (low < high) {
        uint mid = low + ((high - low) >> 1);
        if (wide_value(values[mid], wide[mid]) < target) low = mid + 1u;
        else high = mid;
    }
    starts[gid] = (float)low;
    if (open_right == 0u) {
        ends[gid] = (float)(gid + 1u);
        return;
    }
    /* An open right edge drops every row sharing this timestamp, this one
     * included, which is what `closed=left` means on a duplicated index. */
    uint hunt_low = low;
    uint hunt_high = gid + 1u;
    while (hunt_low < hunt_high) {
        uint mid = hunt_low + ((hunt_high - hunt_low) >> 1);
        if (wide_value(values[mid], wide[mid]) < here) hunt_low = mid + 1u;
        else hunt_high = mid;
    }
    ends[gid] = (float)hunt_low;
}

/* For each wanted instant, the source row that answers for it: the exact
 * match, the last row at or before it, or the first at or after. Upsampling
 * with ffill or bfill is this lane plus one `take`. */
kernel void ts_match_positions(device const float* hay_values [[buffer(0)]],
                               device const float* hay_wide [[buffer(1)]],
                               device const float* want_values [[buffer(2)]],
                               device const float* want_wide [[buffer(3)]],
                               device float* out [[buffer(4)]],
                               constant uint& wanted [[buffer(5)]],
                               constant uint& rows [[buffer(6)]],
                               constant uint& mode [[buffer(7)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= wanted) return;
    long target = wide_value(want_values[gid], want_wide[gid]);
    uint low = 0u;
    uint high = rows;
    while (low < high) {
        uint mid = low + ((high - low) >> 1);
        if (wide_value(hay_values[mid], hay_wide[mid]) < target) low = mid + 1u;
        else high = mid;
    }
    if (mode == MATCH_FORWARD) {
        out[gid] = (low < rows) ? (float)low : -1.0f;
        return;
    }
    bool hit = (low < rows) && (wide_value(hay_values[low], hay_wide[low]) == target);
    if (mode == MATCH_EXACT) {
        out[gid] = hit ? (float)low : -1.0f;
        return;
    }
    if (hit) {
        /* The last row carrying the instant, not the first, so a duplicated
         * timestamp fills forward from its final observation. */
        uint scan = low;
        while (scan + 1u < rows && wide_value(hay_values[scan + 1u], hay_wide[scan + 1u]) == target) {
            scan += 1u;
        }
        out[gid] = (float)scan;
        return;
    }
    out[gid] = (low > 0u) ? (float)(low - 1u) : -1.0f;
}

/* One flag for the whole column: is the timestamp lane non-decreasing.
 *
 * Every thread that finds a descent writes the same 1.0, so the race has one
 * outcome and no atomic is needed. */
kernel void ts_monotone(device const float* values [[buffer(0)]],
                        device const float* wide [[buffer(1)]],
                        device float* flag [[buffer(2)]],
                        constant uint& count [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid == 0u || gid >= count) return;
    long here = wide_value(values[gid], wide[gid]);
    long before = wide_value(values[gid - 1u], wide[gid - 1u]);
    if (here < before) flag[0] = 1.0f;
}

/* The first and last present instant, folded in long.
 *
 * The float lane a numeric reduction would use is exact only to 2^24
 * milliseconds — under five hours — and a resample has to know the true ends
 * of its data to number its bins from zero. So this reads both lanes and folds
 * 64-bit, on the same fixed striding grid `column.jai` counts nulls with. */
kernel void ts_minmax(device const float* values [[buffer(0)]],
                      device const float* wide [[buffer(1)]],
                      device const float* valid [[buffer(2)]],
                      device float* partials [[buffer(3)]],
                      constant uint& count [[buffer(4)]],
                      constant uint& has_valid [[buffer(5)]],
                      uint gid [[thread_position_in_grid]],
                      uint lid [[thread_index_in_threadgroup]],
                      uint wid [[threadgroup_position_in_grid]]) {
    threadgroup long lowest[MINMAX_GROUP];
    threadgroup long highest[MINMAX_GROUP];
    long own_low = EMPTY_LOW;
    long own_high = EMPTY_HIGH;
    for (uint i = gid; i < count; i += MINMAX_GROUP * MINMAX_GROUPS) {
        if ((has_valid != 0u) && (valid[i] == 0.0f)) continue;
        long stamp = wide_value(values[i], wide[i]);
        own_low = min(own_low, stamp);
        own_high = max(own_high, stamp);
    }
    lowest[lid] = own_low;
    highest[lid] = own_high;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = MINMAX_GROUP / 2u; stride > 0u; stride >>= 1) {
        if (lid < stride) {
            lowest[lid] = min(lowest[lid], lowest[lid + stride]);
            highest[lid] = max(highest[lid], highest[lid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid != 0u) return;
    long low = lowest[0];
    long high = highest[0];
    long a = low >> 24;
    long b = high >> 24;
    partials[wid * 4u + 0u] = (float)(low - (a << 24));
    partials[wid * 4u + 1u] = (float)a;
    partials[wid * 4u + 2u] = (float)(high - (b << 24));
    partials[wid * 4u + 3u] = (float)b;
}

/* Rows whose time of day falls in a window, with the window allowed to wrap
 * past midnight — `between_time(22:00, 02:00)` is a night shift, not an empty
 * answer. */
kernel void ts_time_mask(device const float* values [[buffer(0)]],
                         device const float* wide [[buffer(1)]],
                         device const float* valid [[buffer(2)]],
                         device float* out [[buffer(3)]],
                         constant uint& count [[buffer(4)]],
                         constant uint& has_valid [[buffer(5)]],
                         constant int& from_ms [[buffer(6)]],
                         constant int& to_ms [[buffer(7)]],
                         constant uint& keep_from [[buffer(8)]],
                         constant uint& keep_to [[buffer(9)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if ((has_valid != 0u) && (valid[gid] == 0.0f)) {
        out[gid] = 0.0f;
        return;
    }
    long stamp = wide_value(values[gid], wide[gid]);
    int within = (int)(stamp - floor_div(stamp, MS_PER_DAY) * MS_PER_DAY);
    bool after = (within > from_ms) || ((keep_from != 0u) && (within == from_ms));
    bool before = (within < to_ms) || ((keep_to != 0u) && (within == to_ms));
    bool keep = (from_ms <= to_ms) ? (after && before) : (after || before);
    out[gid] = keep ? 1.0f : 0.0f;
}

/* Rows inside a timestamp range. A mask rather than a slice, so `truncate`
 * asks nothing of the order the rows arrived in. */
kernel void ts_range_mask(device const float* values [[buffer(0)]],
                          device const float* wide [[buffer(1)]],
                          device const float* valid [[buffer(2)]],
                          device float* out [[buffer(3)]],
                          constant uint& count [[buffer(4)]],
                          constant uint& has_valid [[buffer(5)]],
                          constant int& from_low [[buffer(6)]],
                          constant int& from_high [[buffer(7)]],
                          constant int& to_low [[buffer(8)]],
                          constant int& to_high [[buffer(9)]],
                          constant uint& has_from [[buffer(10)]],
                          constant uint& has_to [[buffer(11)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if ((has_valid != 0u) && (valid[gid] == 0.0f)) {
        out[gid] = 0.0f;
        return;
    }
    long stamp = wide_value(values[gid], wide[gid]);
    bool keep = true;
    if (has_from != 0u && stamp < wide_scalar(from_low, from_high)) keep = false;
    if (has_to != 0u && stamp > wide_scalar(to_low, to_high)) keep = false;
    out[gid] = keep ? 1.0f : 0.0f;
}

/* The statistic a run of values adds up to, once someone has added them up.
 *
 * Shared by both window kernels so the two cannot disagree about what an empty
 * bin means. Sum and product of nothing are the identities pandas reports;
 * every other statistic of nothing is missing, and a `least` above zero makes
 * even those two missing, which is what a rolling `min_periods` asks for. */
inline float finish_moment(uint agg, float seen, float rows, float total, float squares,
                           float product, float truthy, uint least, thread bool& defined) {
    defined = true;
    if (agg == TS_SIZE) return rows;
    if (agg == TS_COUNT) return seen;
    if (agg == TS_ANY) return (truthy > 0.0f) ? 1.0f : 0.0f;
    if (agg == TS_ALL) return (truthy == seen) ? 1.0f : 0.0f;
    if (seen < (float)least || seen == 0.0f) {
        if (agg == TS_SUM && least == 0u) return 0.0f;
        if (agg == TS_PROD && least == 0u) return 1.0f;
        defined = false;
        return missing_value();
    }
    if (agg == TS_SUM) return total;
    if (agg == TS_PROD) return product;
    if (agg == TS_MEAN) return total / seen;
    if (seen < 2.0f) {
        defined = false;
        return missing_value();
    }
    /* One degree of freedom, taken from the two sums rather than from a second
     * pass. The cancellation that costs is bounded by the window, which is why
     * this is safe here and would not be over a whole column. */
    float mean = total / seen;
    float variance = (squares - seen * mean * mean) / (seen - 1.0f);
    if (variance < 0.0f) variance = 0.0f;
    if (agg == TS_VAR) return variance;
    if (agg == TS_STD) return sqrt(variance);
    return sqrt(variance / seen);
}

inline float fold_sum(threadgroup float* scratch, float value, uint lid) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    scratch[lid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = WIN_GROUP / 2u; stride > 0u; stride >>= 1) {
        if (lid < stride) scratch[lid] += scratch[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return scratch[0];
}

inline float fold_product(threadgroup float* scratch, float value, uint lid) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    scratch[lid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = WIN_GROUP / 2u; stride > 0u; stride >>= 1) {
        if (lid < stride) scratch[lid] *= scratch[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return scratch[0];
}

/* The winning row of a threadgroup, by an ordering key that is a long whether
 * it came from a float, a 48-bit timestamp or a row number. A tie keeps the
 * earlier row, so the tree fold agrees with a serial scan however it pairs up.
 * A thread that drew no present row passes -1 and never wins. */
inline float fold_best(threadgroup long* keys, threadgroup float* slots,
                       long key, float at, uint lid, uint want_max) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    keys[lid] = key;
    slots[lid] = at;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = WIN_GROUP / 2u; stride > 0u; stride >>= 1) {
        if (lid < stride) {
            float other_at = slots[lid + stride];
            if (other_at >= 0.0f) {
                long other_key = keys[lid + stride];
                bool take = slots[lid] < 0.0f;
                if (!take) {
                    take = (want_max != 0u) ? (other_key > keys[lid]) : (other_key < keys[lid]);
                    if (!take && other_key == keys[lid] && other_at < slots[lid]) take = true;
                }
                if (take) {
                    keys[lid] = other_key;
                    slots[lid] = other_at;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return slots[0];
}

/* One output per thread, 256 of them a group, over a tile of the values they
 * share.
 *
 * A resample bucket and a time-offset window are the same shape — a row range
 * per output — so one kernel serves both, and `resample(1D).mean()` cannot
 * drift from `rolling(7D).mean()`. Consecutive outputs read overlapping runs,
 * heavily so for a rolling window, so the group stages the union of its 256
 * ranges once and every thread then reads threadgroup memory. The bounds are
 * non-decreasing in both uses, which is what makes that union a contiguous
 * run; when it does not fit the tile the threads fall back to device reads and
 * the answer is the same. */
kernel void ts_reduce_tiled(device const float* values [[buffer(0)]],
                            device const float* wide [[buffer(1)]],
                            device const float* valid [[buffer(2)]],
                            device const float* starts [[buffer(3)]],
                            device const float* ends [[buffer(4)]],
                            device float* out [[buffer(5)]],
                            device float* out_valid [[buffer(6)]],
                            constant uint& outputs [[buffer(7)]],
                            constant uint& has_valid [[buffer(8)]],
                            constant uint& has_wide [[buffer(9)]],
                            constant uint& agg [[buffer(10)]],
                            constant uint& least [[buffer(11)]],
                            uint gid [[thread_position_in_grid]],
                            uint lid [[thread_index_in_threadgroup]],
                            uint wid [[threadgroup_position_in_grid]]) {
    threadgroup float tile[WIN_TILE];
    threadgroup float tile_valid[WIN_TILE];
    threadgroup uint tile_from;
    threadgroup uint tile_span;
    uint base = wid * WIN_GROUP;
    if (lid == 0u) {
        uint last = min(base + WIN_GROUP, outputs) - 1u;
        uint first = (uint)starts[base];
        uint top = (uint)ends[last];
        bool fits = (top > first) && ((top - first) <= WIN_TILE) && (has_wide == 0u);
        tile_from = first;
        tile_span = fits ? (top - first) : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint from_row = tile_from;
    uint span = tile_span;
    for (uint i = lid; i < span; i += WIN_GROUP) {
        uint at = from_row + i;
        float v = values[at];
        bool present = ((has_valid == 0u) || (valid[at] != 0.0f)) && !isnan(v);
        tile[i] = present ? v : 0.0f;
        tile_valid[i] = present ? 1.0f : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gid >= outputs) return;
    uint from = (uint)starts[gid];
    uint to = (uint)ends[gid];
    bool staged = span > 0u;
    if (agg >= TS_ARGMIN) {
        bool want_max = (agg == TS_ARGMAX) || (agg == TS_ARGLAST);
        bool by_position = (agg == TS_ARGFIRST) || (agg == TS_ARGLAST);
        long best_key = 0;
        float best_at = -1.0f;
        for (uint i = from; i < to; i++) {
            bool present;
            float v = 0.0f;
            if (staged) {
                uint k = i - from_row;
                present = tile_valid[k] != 0.0f;
                v = tile[k];
            } else if (has_wide != 0u) {
                present = (has_valid == 0u) || (valid[i] != 0.0f);
            } else {
                v = values[i];
                present = ((has_valid == 0u) || (valid[i] != 0.0f)) && !isnan(v);
            }
            if (!present) continue;
            long key;
            if (by_position) key = (long)i;
            else if (has_wide != 0u) key = wide_value(values[i], wide[i]);
            else key = order_key(v);
            bool take = best_at < 0.0f;
            if (!take) take = (want_max) ? (key > best_key) : (key < best_key);
            if (take) {
                best_key = key;
                best_at = (float)i;
            }
        }
        out[gid] = best_at;
        out_valid[gid] = (best_at < 0.0f) ? 0.0f : 1.0f;
        return;
    }
    float seen = 0.0f;
    float total = 0.0f;
    float squares = 0.0f;
    float product = 1.0f;
    float truthy = 0.0f;
    for (uint i = from; i < to; i++) {
        float v;
        bool present;
        if (staged) {
            uint k = i - from_row;
            present = tile_valid[k] != 0.0f;
            v = tile[k];
        } else {
            v = values[i];
            present = ((has_valid == 0u) || (valid[i] != 0.0f)) && !isnan(v);
        }
        if (!present) continue;
        seen += 1.0f;
        total += v;
        squares += v * v;
        product *= v;
        if (v != 0.0f) truthy += 1.0f;
    }
    bool defined = true;
    float answer = finish_moment(agg, seen, (float)(to - from), total, squares, product,
                                 truthy, least, defined);
    out[gid] = answer;
    out_valid[gid] = defined ? 1.0f : 0.0f;
}

/* One threadgroup per output, for when there are few outputs and each covers a
 * great many rows — a yearly resample of a decade of ticks. One thread per
 * output would run that on twelve threads; this runs it on all of them and
 * pays eight barriers a fold for the privilege. */
kernel void ts_reduce_group(device const float* values [[buffer(0)]],
                            device const float* wide [[buffer(1)]],
                            device const float* valid [[buffer(2)]],
                            device const float* starts [[buffer(3)]],
                            device const float* ends [[buffer(4)]],
                            device float* out [[buffer(5)]],
                            device float* out_valid [[buffer(6)]],
                            constant uint& outputs [[buffer(7)]],
                            constant uint& has_valid [[buffer(8)]],
                            constant uint& has_wide [[buffer(9)]],
                            constant uint& agg [[buffer(10)]],
                            constant uint& least [[buffer(11)]],
                            uint lid [[thread_index_in_threadgroup]],
                            uint wid [[threadgroup_position_in_grid]]) {
    threadgroup float scratch[WIN_GROUP];
    threadgroup long keys[WIN_GROUP];
    threadgroup float slots[WIN_GROUP];
    if (wid >= outputs) return;
    uint from = (uint)starts[wid];
    uint to = (uint)ends[wid];
    if (agg >= TS_ARGMIN) {
        bool want_max = (agg == TS_ARGMAX) || (agg == TS_ARGLAST);
        bool by_position = (agg == TS_ARGFIRST) || (agg == TS_ARGLAST);
        long best_key = 0;
        float best_at = -1.0f;
        for (uint i = from + lid; i < to; i += WIN_GROUP) {
            bool present;
            float v = 0.0f;
            if (has_wide != 0u) {
                present = (has_valid == 0u) || (valid[i] != 0.0f);
            } else {
                v = values[i];
                present = ((has_valid == 0u) || (valid[i] != 0.0f)) && !isnan(v);
            }
            if (!present) continue;
            long key;
            if (by_position) key = (long)i;
            else if (has_wide != 0u) key = wide_value(values[i], wide[i]);
            else key = order_key(v);
            bool take = best_at < 0.0f;
            if (!take) take = (want_max) ? (key > best_key) : (key < best_key);
            if (take) {
                best_key = key;
                best_at = (float)i;
            }
        }
        float winner = fold_best(keys, slots, best_key, best_at, lid, want_max ? 1u : 0u);
        if (lid == 0u) {
            out[wid] = winner;
            out_valid[wid] = (winner < 0.0f) ? 0.0f : 1.0f;
        }
        return;
    }
    float seen = 0.0f;
    float total = 0.0f;
    float squares = 0.0f;
    float product = 1.0f;
    float truthy = 0.0f;
    for (uint i = from + lid; i < to; i += WIN_GROUP) {
        float v = values[i];
        bool present = ((has_valid == 0u) || (valid[i] != 0.0f)) && !isnan(v);
        if (!present) continue;
        seen += 1.0f;
        total += v;
        squares += v * v;
        product *= v;
        if (v != 0.0f) truthy += 1.0f;
    }
    seen = fold_sum(scratch, seen, lid);
    total = fold_sum(scratch, total, lid);
    squares = fold_sum(scratch, squares, lid);
    truthy = fold_sum(scratch, truthy, lid);
    product = fold_product(scratch, product, lid);
    if (lid != 0u) return;
    bool defined = true;
    float answer = finish_moment(agg, seen, (float)(to - from), total, squares, product,
                                 truthy, least, defined);
    out[wid] = answer;
    out_valid[wid] = defined ? 1.0f : 0.0f;
}
"""

var _kernels: dict[str, Kernel] = {}

fn kernel_for(entry: str) -> Kernel {
    let existing = _kernels.get(entry, null)
    if existing is not null { return existing }
    let compiled = Kernel.compile(SOURCE, entry)
    _kernels[entry] = compiled
    return compiled
}

fn flag(value: bool) -> int { return if value { 1 } else { 0 } }

fn u32(value: int) -> int { return value & 0xFFFFFFFF }

fn require_device(what: str) -> void {
    if not gpu.is_available() {
        throw RuntimeError(f"jaiframe needs a GPU to {what}")
    }
}

#: A 48-bit value as the `(low, high)` pair a kernel scalar carries it in.
fn wide_args(value: int) -> (int, int) {
    let split = wide_pack(value)
    return (u32(int(split[0])), u32(int(split[1])))
}

let WEEKDAY_NAMES = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]

let MONTH_NAMES = [
    "JAN",
    "FEB",
    "MAR",
    "APR",
    "MAY",
    "JUN",
    "JUL",
    "AUG",
    "SEP",
    "OCT",
    "NOV",
    "DEC",
]

#: What a frequency counts.
#:
#: The split that matters is `is_fixed`: the first six are a constant number of
#: milliseconds and bucket by division, the last three are calendar periods and
#: bucket by month index. A month is not 30 days, and a library that treats it
#: as one loses a day every February and two every leap year.
pub enum FreqUnit {
    #: Milliseconds — the resolution the store itself has.
    Milli,
    Second,
    Minute,
    Hour,
    Day,
    #: Seven days, phased to a weekday rather than to the epoch.
    Week,
    Month,
    Quarter,
    #: A year. pandas spells the alias `A` and, since 2.0, `Y`.
    Year,

    #: The pandas alias, which `parse_freq` reads back.
    pub fn name(self) -> str {
        return match self {
            FreqUnit.Milli => "ms",
            FreqUnit.Second => "s",
            FreqUnit.Minute => "min",
            FreqUnit.Hour => "H",
            FreqUnit.Day => "D",
            FreqUnit.Week => "W",
            FreqUnit.Month => "M",
            FreqUnit.Quarter => "Q",
            FreqUnit.Year => "A",
        }
    }

    #: Whether one period is always the same number of milliseconds.
    pub fn is_fixed(self) -> bool {
        return match self {
            FreqUnit.Month | FreqUnit.Quarter | FreqUnit.Year => false,
            _ => true,
        }
    }

    #: Milliseconds in one period, for the fixed units.
    pub fn millis(self) -> int {
        return match self {
            FreqUnit.Milli => 1,
            FreqUnit.Second => MILLIS_PER_SECOND,
            FreqUnit.Minute => MILLIS_PER_MINUTE,
            FreqUnit.Hour => MILLIS_PER_HOUR,
            FreqUnit.Day => MILLIS_PER_DAY,
            FreqUnit.Week => MILLIS_PER_WEEK,
            _ => throw ValueError(f"{self.name()} is a calendar period, not a fixed span"),
        }
    }

    #: Months in one period, for the calendar units.
    pub fn months(self) -> int {
        return match self {
            FreqUnit.Month => 1,
            FreqUnit.Quarter => 3,
            FreqUnit.Year => 12,
            _ => throw ValueError(f"{self.name()} is a fixed span, not a whole number of months"),
        }
    }
}

#: A frequency: how long a period is, and where its edges fall.
#:
#: `count` multiplies the unit, so `5min` and `2W` are one of these. `anchor`
#: is read differently by unit and by nothing else: for `Week` it is the
#: weekday a bin ends on, 0 for Monday, so `W` is `W-SUN` and bins run Monday
#: to Sunday; for `Month`, `Quarter` and `Year` it is the month a period ends
#: in, so `Q` is `Q-DEC` and `A-JUN` is a fiscal year closing in June.
#:
#: `at_end` records which edge the alias named — `M` is a month end and `MS` a
#: month start. It changes no bin, only which instant the bin is labelled with,
#: which is the one thing about resampling that surprises people.
pub class Freq {
    #: What a period counts.
    pub let unit: FreqUnit
    #: How many of them one period spans.
    pub let count: int
    #: The weekday a `Week` bin ends on, or the month a calendar period ends
    #: in. Unread for the other units.
    pub let anchor: int
    #: Whether the alias named the end of a period rather than its start.
    pub let at_end: bool

    #: A frequency of `count` units. The anchor defaults per unit, to Sunday
    #: for a week and December for a calendar period, as pandas' do.
    pub fn init(self, unit: FreqUnit, count: int = 1, anchor: int = -1, at_end: bool = true) {
        if count <= 0 {
            throw ValueError(f"a frequency spans at least one period, got {count}")
        }
        self.unit = unit
        self.count = count
        self.at_end = at_end
        if anchor >= 0 {
            self.anchor = anchor
            return
        }
        self.anchor = if unit == FreqUnit.Week { 6 } else { 12 }
    }

    #: Whether one period is always the same number of milliseconds.
    pub fn is_fixed(self) -> bool { return self.unit.is_fixed() }

    #: Milliseconds in one period, for a fixed frequency.
    pub fn millis(self) -> int { return self.count * self.unit.millis() }

    #: Months in one period, for a calendar frequency.
    pub fn total_months(self) -> int { return self.count * self.unit.months() }

    #: The month index a period boundary is offset by.
    #:
    #: Periods run `[p * total + shift, (p + 1) * total + shift)` over the
    #: 0-based month index `year * 12 + month - 1`. Solving for the shift that
    #: puts the last month of a period at `anchor` leaves the anchor's own
    #: residue, which is why `Q-DEC` shifts by nothing and `Q-JAN` by one.
    pub fn shift(self) -> int {
        let total = self.total_months()
        let residue = self.anchor - self.anchor // total * total
        return residue
    }

    #: The instant a fixed frequency's grid is phased from.
    #:
    #: Zero for everything but a week, whose bins are anchored to a weekday and
    #: not to the epoch — which is what makes `W` mean the same seven days in
    #: two frames that start on different days. 1970-01-01 was a Thursday, so
    #: the phase is the first day of 1970 whose weekday opens the bin.
    pub fn phase(self) -> int {
        if self.unit != FreqUnit.Week { return 0 }
        let opens = self.anchor + 1 - 7 * ((self.anchor + 1) // 7)
        let day = opens - 3 - 7 * ((opens - 3) // 7)
        return day * MILLIS_PER_DAY
    }

    #: The alias this frequency parses back from.
    pub fn to_str(self) -> str {
        var alias = self.unit.name()
        if not self.at_end and not self.unit.is_fixed() { alias = alias + "S" }
        var suffix = ""
        if self.unit == FreqUnit.Week and self.anchor != 6 {
            suffix = "-" + WEEKDAY_NAMES[self.anchor]
        }
        if not self.unit.is_fixed() and self.unit != FreqUnit.Month and self.anchor != 12 {
            suffix = "-" + MONTH_NAMES[self.anchor - 1]
        }
        let head = if self.count == 1 { "" } else { f"{self.count}" }
        return head + alias + suffix
    }
}

#: A pandas frequency alias — `5min`, `H`, `2W-MON`, `QS`, `A-JUN`.
#:
#: A leading integer multiplies, an alias names the unit, and a `-` suffix
#: anchors it. The alias table is case-sensitive where pandas' is: `ms` is
#: milliseconds and `MS` is a month start, and letting one stand for the other
#: is a two-order-of-magnitude bug in a resample nobody would look at twice.
pub fn parse_freq(text: str) -> Freq {
    let source = text.strip()
    if source.len() == 0 { throw ValueError("a frequency alias cannot be empty") }
    var at = 0
    while at < source.len() and source[at] >= "0" and source[at] <= "9" { at += 1 }
    let count = if at == 0 { 1 } else { source[0:at].to_int() ?? 0 }
    if count <= 0 {
        throw ValueError(f"a frequency spans at least one period, got {repr(text)}")
    }
    let rest = source[at:source.len()]
    if rest.len() == 0 { throw ValueError(f"{repr(text)} names no unit") }
    let dash = rest.find("-")
    let alias = if dash < 0 { rest } else { rest[0:dash] }
    let suffix = if dash < 0 { "" } else { rest[dash + 1:rest.len()].upper() }
    let named = unit_of_alias(alias)
    let unit = named[0]
    let at_end = named[1]
    if suffix.len() == 0 { return Freq(unit, count: count, at_end: at_end) }
    if unit == FreqUnit.Week {
        return Freq(unit, count: count, anchor: weekday_of_name(suffix, text), at_end: at_end)
    }
    if unit.is_fixed() {
        throw ValueError(f"{repr(text)} anchors a fixed frequency, which has nothing to anchor to")
    }
    return Freq(unit, count: count, anchor: month_of_name(suffix, text), at_end: at_end)
}

#: A frequency however it was spelled — an alias or an already-parsed `Freq`.
pub fn to_freq(spec: any) -> Freq {
    if isinstance(spec, Freq) { return spec }
    if isinstance(spec, str) { return parse_freq(spec) }
    throw TypeError("a frequency is an alias string or a Freq")
}

fn unit_of_alias(alias: str) -> (FreqUnit, bool) {
    return match alias {
        "ms" | "L" | "l" => (FreqUnit.Milli, true),
        "s" | "S" | "sec" | "secs" => (FreqUnit.Second, true),
        "min" | "T" | "t" | "mins" => (FreqUnit.Minute, true),
        "H" | "h" | "hr" | "hrs" => (FreqUnit.Hour, true),
        "D" | "d" => (FreqUnit.Day, true),
        "W" | "w" => (FreqUnit.Week, true),
        "M" | "ME" => (FreqUnit.Month, true),
        "MS" => (FreqUnit.Month, false),
        "Q" | "QE" => (FreqUnit.Quarter, true),
        "QS" => (FreqUnit.Quarter, false),
        "A" | "Y" | "AE" | "YE" => (FreqUnit.Year, true),
        "AS" | "YS" => (FreqUnit.Year, false),
        _ => match alias.lower() {
            "milli" | "millis" | "millisecond" | "milliseconds" => (FreqUnit.Milli, true),
            "second" | "seconds" => (FreqUnit.Second, true),
            "minute" | "minutes" => (FreqUnit.Minute, true),
            "hour" | "hours" => (FreqUnit.Hour, true),
            "day" | "days" => (FreqUnit.Day, true),
            "week" | "weeks" => (FreqUnit.Week, true),
            "month" | "months" => (FreqUnit.Month, true),
            "quarter" | "quarters" => (FreqUnit.Quarter, true),
            "year" | "years" | "annual" => (FreqUnit.Year, true),
            _ => throw ValueError(f"unknown frequency alias {repr(alias)}"),
        },
    }
}

fn weekday_of_name(name: str, whole: str) -> int {
    for (at, day) in WEEKDAY_NAMES.enumerate() {
        if name.startswith(day) { return at }
    }
    throw ValueError(f"{repr(whole)} anchors to no weekday this calendar has")
}

fn month_of_name(name: str, whole: str) -> int {
    for (at, month) in MONTH_NAMES.enumerate() {
        if name.startswith(month) { return at + 1 }
    }
    throw ValueError(f"{repr(whole)} anchors to no month this calendar has")
}
