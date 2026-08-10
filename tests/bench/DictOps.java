// Same program as dict_ops.jai.
import java.util.HashMap;
import java.util.Map;

public class DictOps {
    public static void main(String[] args) {
        Map<String, Long> counts = new HashMap<>();
        for (long i = 0; i < 500000L; i++) {
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
