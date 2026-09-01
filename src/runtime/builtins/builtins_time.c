/* builtins_time.c — the process's ambient OS-facing state: the random number
 * stream and both clocks. */

/* Feature macros must precede every include: -std=c11 alone does not expose
 * nanosleep, clock_gettime, strptime, localtime_r or getentropy. */
#if !defined(_POSIX_C_SOURCE)
#  define _POSIX_C_SOURCE 200809L
#endif
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#  define _DARWIN_C_SOURCE
#endif

#include <errno.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

#if defined(__APPLE__)
#  include <sys/random.h>
#  define JAI_HAVE_GETENTROPY 1
#elif defined(__GLIBC__) &&                                                    \
    (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 25))
#  include <sys/random.h>
#  define JAI_HAVE_GETENTROPY 1
#endif

#include "runtime/builtins/builtins_math.h"
#include "vm/gc.h"

/* ------------------------------------------------------------------ */
/* Random — xoshiro256**                                                */
/* ------------------------------------------------------------------ */

/* One stream for the process. std.random layers distributions on top of it;
 * nothing here is thread-safe, which matches the rest of the runtime. */
static uint64_t gRngState[4];
static bool     gRngSeeded;

static inline uint64_t splitMix64(uint64_t *state) {
    *state += 0x9E3779B97F4A7C15ULL;
    return jaiHashU64(*state);
}

/* An all-zero state is a fixed point of xoshiro, so the seed is expanded
 * through splitmix64, which cannot produce four zero words in a row. */
static void seedFromWord(uint64_t seed) {
    uint64_t state = seed;
    for (int i = 0; i < 4; i++) gRngState[i] = splitMix64(&state);
    gRngSeeded = true;
}

static bool readSystemEntropy(unsigned char *bytes, size_t length) {
#if defined(JAI_HAVE_GETENTROPY)
    /* getentropy is capped at 256 bytes per call, which is far above what the
     * 32-byte state needs. */
    if (getentropy(bytes, length) == 0) return true;
#endif
    FILE *urandom = fopen("/dev/urandom", "rb");
    if (urandom == NULL) return false;
    size_t got = fread(bytes, 1, length, urandom);
    (void)fclose(urandom);
    return got == length;
}

static void seedFromEntropy(void) {
    unsigned char bytes[sizeof gRngState];
    if (readSystemEntropy(bytes, sizeof bytes)) {
        memcpy(gRngState, bytes, sizeof gRngState);
        gRngSeeded = true;
        if ((gRngState[0] | gRngState[1] | gRngState[2] | gRngState[3]) == 0)
            seedFromWord(0x2545F4914F6CDD1DULL);
        return;
    }
    /* No entropy source. A fixed seed would replay the same stream on every
     * run, so mix in whatever this process can say about itself instead. */
    const void *here = (const void *)&bytes;
    uint64_t mixed = (uint64_t)(jaiClockMonotonic() * 1e9);
    mixed ^= jaiHashBytes(&here, sizeof here);
    seedFromWord(mixed);
}

static inline void ensureSeeded(void) {
    if (!gRngSeeded) seedFromEntropy();
}

static inline uint64_t rotateLeft(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static inline uint64_t randomNext(void) {
    uint64_t s0 = gRngState[0];
    uint64_t s1 = gRngState[1];
    uint64_t s2 = gRngState[2];
    uint64_t s3 = gRngState[3];

    const uint64_t result = rotateLeft(s1 * 5u, 7) * 9u;
    const uint64_t t = s1 << 17;

    s2 ^= s0;
    s3 ^= s1;
    s1 ^= s2;
    s0 ^= s3;
    s2 ^= t;
    s3 = rotateLeft(s3, 45);

    gRngState[0] = s0;
    gRngState[1] = s1;
    gRngState[2] = s2;
    gRngState[3] = s3;

    return result;
}

static bool nRandomU64(int argc, Value *args, Value *out) {
    (void)argc;
    (void)args;
    ensureSeeded();
    /* The full 64-bit pattern is handed over as an int; the top bit reads as a
     * sign, which is what `& INT_MAX` in std.random strips off. */
    *out = INT_VAL((int64_t)randomNext());
    return true;
}

static bool nRandomSeed(int argc, Value *args, Value *out) {
    if (argc == 0 || IS_NULL(args[0])) {
        seedFromEntropy();
        *out = NULL_VAL;
        return true;
    }
    int64_t seed;
    if (!argIntFast(args[0], 1, "random_seed", &seed)) return false;
    seedFromWord((uint64_t)seed);
    *out = NULL_VAL;
    return true;
}

/* ------------------------------------------------------------------ */
/* Time                                                                 */
/* ------------------------------------------------------------------ */

static inline bool secondsToNanos(double seconds, const char *fnName,
                                  int64_t *out) {
    const double nanos = seconds * 1e9;

    if (isnan(nanos) || nanos >= kTwoPow63 || nanos < -kTwoPow63)
        return jaiThrow(vm.cOverflowError,
                        "%s(): %g seconds does not fit in an int of nanoseconds",
                        fnName, seconds);

    *out = (int64_t)nanos;
    return true;
}

/* Both clocks report integer nanoseconds: the monotonic one from an arbitrary
 * origin, the wall one from the Unix epoch. std.time builds its Duration and
 * Instant directly on those counts. */
static bool nTimeMono(int argc, Value *args, Value *out) {
    (void)argc;
    (void)args;
    int64_t nanos;
    if (!secondsToNanos(jaiClockMonotonic(), "time_mono", &nanos)) return false;
    *out = INT_VAL(nanos);
    return true;
}

static bool nTimeMonoSeconds(int argc, Value *args, Value *out) {
    (void)argc;
    (void)args;
    *out = FLOAT_VAL(jaiClockMonotonic());
    return true;
}

static bool wallClock(const char *fnName, struct timespec *ts) {
    if (clock_gettime(CLOCK_REALTIME, ts) == 0) return true;
    return jaiThrow(vm.cOSError, "%s(): the system clock is unavailable: %s",
                    fnName, strerror(errno));
}

static bool nTimeWall(int argc, Value *args, Value *out) {
    (void)argc;
    (void)args;
    struct timespec ts;
    if (!wallClock("time_wall", &ts)) return false;
    if ((int64_t)ts.tv_sec > INT64_MAX / 1000000000)
        return jaiThrow(vm.cOverflowError,
                        "time_wall(): the clock is past the range of an int of "
                        "nanoseconds");
    *out = INT_VAL((int64_t)ts.tv_sec * 1000000000 + (int64_t)ts.tv_nsec);
    return true;
}

static bool nSleep(int argc, Value *args, Value *out) {
    (void)argc;
    double seconds;
    if (!argNumberFast(args[0], 1, "sleep", &seconds)) return false;
    if (isnan(seconds) || isinf(seconds))
        return domainError("sleep", "a finite duration in seconds", seconds);
    if (seconds < 0.0)
        return domainError("sleep", "a non-negative duration in seconds", seconds);

    double whole = floor(seconds);
    if (whole >= kTwoPow63)
        return jaiThrow(vm.cOverflowError, "sleep(): %g seconds is too long",
                        seconds);

    struct timespec request;
    request.tv_sec = (time_t)whole;
    long fraction = (long)((seconds - whole) * 1e9);
    request.tv_nsec = fraction > 999999999L ? 999999999L
                                            : (fraction < 0 ? 0 : fraction);

    /* A signal cuts the sleep short and reports what is left; the request is
     * restarted so that the caller waits for the duration it asked for. */
    struct timespec remaining;
    while (nanosleep(&request, &remaining) != 0) {
        if (errno != EINTR)
            return jaiThrow(vm.cOSError, "sleep(): %s", strerror(errno));
        request = remaining;
    }
    *out = NULL_VAL;
    return true;
}

/* Days from 1970-01-01 to a proleptic Gregorian date (Hinnant's
 * days_from_civil) -- avoids timegm (nonstandard) and mktime (applies the
 * local time zone). */
static inline int64_t daysFromCivil(int64_t year, int64_t month, int64_t day) {
    year -= month <= 2;
    const int64_t era = (year >= 0 ? year : year - 399) / 400;
    const int64_t yearOfEra = year - era * 400;
    const int64_t dayOfYear =
        (153 * (month + (month > 2 ? -3 : 9)) + 2) / 5 + day - 1;
    const int64_t dayOfEra =
        yearOfEra * 365 + yearOfEra / 4 - yearOfEra / 100 + dayOfYear;

    return era * 146097 + dayOfEra - 719468;
}

static inline bool tmToUnix(const struct tm *parts, int64_t *out) {
    const int64_t days =
        daysFromCivil((int64_t)parts->tm_year + 1900,
                      (int64_t)parts->tm_mon + 1,
                      (int64_t)parts->tm_mday);

    if (days > 106751991167LL || days < -106751991167LL)
        return false;

    *out = days * 86400 +
           (int64_t)parts->tm_hour * 3600 +
           (int64_t)parts->tm_min * 60 +
           (int64_t)parts->tm_sec;

    return true;
}

static bool nTimeFormat(int argc, Value *args, Value *out) {
    double seconds;
    ObjString *format;
    bool utc = false;

    if (!argNumberFast(args[0], 1, "time_format", &seconds)) return false;
    if (!jaiArgString(args[1], 2, "time_format", &format)) return false;
    if (argc >= 3 && !jaiArgBool(args[2], 3, "time_format", &utc)) return false;

    if (isnan(seconds) || isinf(seconds))
        return domainError("time_format", "a finite Unix timestamp", seconds);

    const double whole = floor(seconds);
    if (whole >= kTwoPow63 || whole < -kTwoPow63)
        return jaiThrow(vm.cOverflowError,
                        "time_format(): %g is out of range for a timestamp",
                        seconds);

    const time_t when = (time_t)whole;
    struct tm parts;
    struct tm *filled =
        utc ? gmtime_r(&when, &parts) : localtime_r(&when, &parts);

    if (filled == NULL)
        return jaiThrow(vm.cValueError,
                        "time_format(): %g is not a representable date",
                        seconds);

    if (format->length == 0) {
        ObjString *empty = jaiStringIntern("", 0);
        if (empty == NULL) return false;
        *out = OBJ_VAL(empty);
        return true;
    }

    /* `format` reaches here as an argument, not necessarily interned or a
     * literal, and strftime scans it to a NUL. The terminated form is reused
     * below across the fast path, the growth loop -- which allocates on the
     * JAI heap via jaiStringNew on every successful strftime -- and the final
     * error message, so it is rooted for the whole function rather than
     * trusted to outlive those allocations unrooted, exactly as nGpuCompile
     * roots jaiStringTerminated's result. */
    ObjString *formatTerm = jaiStringTerminated(format);
    if (formatTerm == NULL) return false;
    jaiGCPushRoot(OBJ_VAL(formatTerm));
    const char *cformat = formatTerm->chars;

    /* Most formatted timestamps fit here: avoid allocator traffic entirely. */
    char stackBuffer[128];
    size_t written = strftime(stackBuffer, sizeof stackBuffer, cformat, &parts);

    if (written > 0) {
        ObjString *text = jaiStringNew(stackBuffer, written);
        jaiGCPopRoot();
        if (text == NULL) return false;
        *out = OBJ_VAL(text);
        return true;
    }

    size_t capacity = 256;
    for (;;) {
        char *buffer = JAI_ALLOC(char, capacity);
        written = strftime(buffer, capacity, cformat, &parts);

        if (written > 0) {
            ObjString *text = jaiStringNew(buffer, written);
            JAI_FREE_ARRAY(char, buffer, capacity);
            jaiGCPopRoot();
            if (text == NULL) return false;
            *out = OBJ_VAL(text);
            return true;
        }

        JAI_FREE_ARRAY(char, buffer, capacity);

        if (capacity >= 65536) {
            bool thrown = jaiThrow(vm.cValueError,
                            "time_format(): '%s' produces more than 64 KiB",
                            cformat);
            jaiGCPopRoot();
            return thrown;
        }

        capacity <<= 1;
    }
}

static bool nTimeParse(int argc, Value *args, Value *out) {
    ObjString *text, *format;
    bool utc = false;
    if (!jaiArgString(args[0], 1, "time_parse", &text)) return false;
    if (!jaiArgString(args[1], 2, "time_parse", &format)) return false;
    if (argc >= 3 && !jaiArgBool(args[2], 3, "time_parse", &utc)) return false;

    struct tm parts;
    memset(&parts, 0, sizeof parts);
    /* strptime leaves untouched fields alone, and a zero day-of-month is not a
     * date; -1 lets mktime work out whether daylight saving is in effect. */
    parts.tm_mday = 1;
    parts.tm_isdst = -1;

    /* Both `text` and `format` are caller arguments, not necessarily interned,
     * and strptime scans both to a NUL in one call -- the same shape as
     * nGpuCompile's source/entry pair. Both rooted, and rooted BEFORE the
     * second is made: a terminated copy is reachable from nothing, so making
     * format's copy could collect text's. */
    ObjString *textTerm = jaiStringTerminated(text);
    if (textTerm == NULL) return false;
    jaiGCPushRoot(OBJ_VAL(textTerm));
    ObjString *formatTerm = jaiStringTerminated(format);
    if (formatTerm == NULL) { jaiGCPopRoot(); return false; }
    jaiGCPushRoot(OBJ_VAL(formatTerm));
    const char *ctext = textTerm->chars;
    const char *cformat = formatTerm->chars;

    bool matched = strptime(ctext, cformat, &parts) != NULL;
    if (!matched) {
        bool thrown = jaiThrow(vm.cValueError,
                        "time_parse(): '%s' does not match '%s'", ctext, cformat);
        jaiGCPopRoots(2);
        return thrown;
    }

    int64_t seconds;
    if (utc) {
        if (!tmToUnix(&parts, &seconds)) {
            bool thrown = jaiThrow(vm.cOverflowError,
                            "time_parse(): '%s' is out of range for a timestamp",
                            ctext);
            jaiGCPopRoots(2);
            return thrown;
        }
    } else {
        time_t local = mktime(&parts);
        if (local == (time_t)-1) {
            bool thrown = jaiThrow(vm.cValueError,
                            "time_parse(): '%s' is not a valid local time", ctext);
            jaiGCPopRoots(2);
            return thrown;
        }
        seconds = (int64_t)local;
    }
    jaiGCPopRoots(2);
    *out = INT_VAL(seconds);
    return true;
}

void jaiRegisterRandomPrimitives(void) {
    if (vm.builtins == NULL) return;

    jaiDefineNative("__prim__.random_u64",  nRandomU64,  0, 0);
    jaiDefineNative("__prim__.random_seed", nRandomSeed, 0, 1);
}

void jaiRegisterTimePrimitives(void) {
    if (vm.builtins == NULL) return;

    jaiDefineNative("__prim__.time_mono", nTimeMono, 0, 0);
    jaiDefineNative("__prim__.time_wall", nTimeWall, 0, 0);
    jaiDefineNative("__prim__.time_mono_seconds", nTimeMonoSeconds, 0, 0);

    /* Appendix C calls this `sleep`; `time_sleep` is the same native kept only
     * because lib/std/time.jai still calls it under that name. */
    jaiDefineNative("__prim__.sleep",      nSleep, 1, 1);
    jaiDefineNative("__prim__.time_sleep", nSleep, 1, 1);

    jaiDefineNative("__prim__.time_format", nTimeFormat, 2, 3);
    jaiDefineNative("__prim__.time_parse",  nTimeParse,  2, 3);
}
