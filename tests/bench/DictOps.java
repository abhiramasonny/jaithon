// Same program as dict_ops.jai.
import java.util.HashMap;
import java.util.Map;

public class DictOps {
    // BENCH_LEVEL picks how much work this benchmark does: easy a sixteenth of it,
    // medium a quarter, hard (the default, and anything unrecognised) all of it.
    static long scale() {
        String l = System.getenv("BENCH_LEVEL");
        if ("easy".equals(l)) return 16;
        if ("medium".equals(l)) return 4;
        return 1;
    }

    static final long SCALE = scale();
    static final long ITERATIONS = 30000000L / SCALE;

    public static void main(String[] args) {
        Map<String, Long> counts = new HashMap<>();
        for (long i = 0; i < ITERATIONS; i++) {
            String key = "k" + (i % 10000L);
            counts.put(key, counts.getOrDefault(key, 0L) + 1L);
        }

        long total = 0;
        for (Map.Entry<String, Long> entry : counts.entrySet()) {
            total += entry.getValue();
        }
        System.out.println(counts.size());
        System.out.println(total);
    }
}
