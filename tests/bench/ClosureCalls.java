// Same program as closure_calls.jai.
import java.util.function.LongUnaryOperator;

public class ClosureCalls {
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
        for (long k = 1; k <= 20L; k++) {
            LongUnaryOperator bump = adder(k);
            total = (total + applyN(bump, 0, 100000L)) % 1000000007L;
        }
        System.out.println(total);
    }
}
