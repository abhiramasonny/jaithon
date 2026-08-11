// Same program as closure_calls.jai.
import java.util.function.LongUnaryOperator;

public class ClosureCalls {
    // BENCH_LEVEL picks how much work this benchmark does: easy a sixteenth of it,
    // medium a quarter, hard (the default, and anything unrecognised) all of it.
    static long scale() {
        String l = System.getenv("BENCH_LEVEL");
        if ("easy".equals(l)) return 16;
        if ("medium".equals(l)) return 4;
        return 1;
    }

    static final long SCALE = scale();
    // Repetitions of the fixed 100000-call kernel.
    static final long ROUNDS = 1000L / SCALE;

    static LongUnaryOperator adder(long step) {
        return x -> x + step;
    }

    static long applyN(LongUnaryOperator f, long start, long times) {
        long acc = start;
        for (long i = 0; i < times; i++) {
            acc = f.applyAsLong(acc);
        }
        return acc;
    }

    public static void main(String[] args) {
        long total = 0;
        for (long k = 1; k <= ROUNDS; k++) {
            LongUnaryOperator bump = adder(k);
            total = (total + applyN(bump, 0, 100000L)) % 1000000007L;
        }
        System.out.println(total);
    }
}
