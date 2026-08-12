// Same program as bitops.jai.
public class Bitops {
    // BENCH_LEVEL picks how much work this benchmark does: easy a sixteenth of it,
    // medium a quarter, hard (the default, and anything unrecognised) all of it.
    static long scale() {
        String l = System.getenv("BENCH_LEVEL");
        if ("easy".equals(l)) return 16;
        if ("medium".equals(l)) return 4;
        return 1;
    }

    static final long SCALE = scale();
    static final long ITERATIONS = 1000000L / SCALE;

    static long popcount(long value) {
        long v = value;
        long bits = 0;
        while (v != 0) {
            v = v & (v - 1);
            bits += 1;
        }
        return bits;
    }

    public static void main(String[] args) {
        final long mask = 4294967295L;
        long seed = 7;
        long ones = 0;
        long checksum = 0;
        for (long i = 0; i < ITERATIONS; i++) {
            seed = (seed * 1103515245L + 12345L) % 2147483648L;
            long v = seed ^ checksum;
            v = v ^ (v >> 7);
            v = (v ^ (v << 3)) & mask;
            v = v ^ (v >> 11);
            long bits = popcount(v);
            ones += bits;
            checksum = (checksum + v + bits) & mask;
            checksum = ((checksum << 1) | (checksum >> 31)) & mask;
        }
        System.out.println(ones);
        System.out.println(checksum);
    }
}
