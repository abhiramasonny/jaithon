// Same program as word_freq.jai.
import java.util.HashMap;
import java.util.Map;

public class WordFreq {
    public static void main(String[] args) {
        StringBuilder builder = new StringBuilder();
        long seed = 7;
        for (int i = 0; i < 200000; i++) {
            seed = (seed * 1103515245L + 12345L) % 2147483648L;
            builder.append("w").append(seed % 500L).append(" ");
        }
        String text = builder.toString();

        Map<String, Long> counts = new HashMap<>();
        int start = 0;
        int at = 0;
        int n = text.length();
        while (at < n) {
            if (text.charAt(at) == ' ') {
                if (at > start) {
                    String word = text.substring(start, at);
                    counts.put(word, counts.getOrDefault(word, 0L) + 1L);
                }
                start = at + 1;
            }
            at++;
        }

        long total = 0;
        for (Map.Entry<String, Long> entry : counts.entrySet()) {
            total += entry.getValue();
        }
        System.out.println(counts.size());
        System.out.println(total);
    }
}
