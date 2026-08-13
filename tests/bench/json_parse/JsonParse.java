// Same program as json_parse.jai. Parsed by hand, like every other port.
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class JsonParse {
    static long scale() {
        String l = System.getenv("BENCH_LEVEL");
        if ("easy".equals(l)) return 16;
        if ("medium".equals(l)) return 4;
        return 1;
    }

    static final long SCALE = scale();
    static final int RECORDS = (int) Math.max(2500 / SCALE, 1);
    static final int REPS = 6;

    static final int K_NULL = 0;
    static final int K_BOOL = 1;
    static final int K_INT = 2;
    static final int K_STR = 3;
    static final int K_LIST = 4;
    static final int K_DICT = 5;

    static final class JNode {
        int kind;
        long num;
        String text = "";
        List<JNode> items = new ArrayList<>();
        Map<String, JNode> fields = new HashMap<>();
        JNode(int kind) { this.kind = kind; }
    }

    static final class Parser {
        final String src;
        int at;
        final int n;
        Parser(String src) { this.src = src; this.at = 0; this.n = src.length(); }

        void skip() {
            while (at < n && src.charAt(at) == ' ') at += 1;
        }

        JNode value() {
            skip();
            char c = src.charAt(at);
            if (c == '{') return object();
            if (c == '[') return array();
            if (c == '"') {
                JNode node = new JNode(K_STR);
                node.text = text();
                return node;
            }
            if (c == 't') {
                at += 4;
                JNode node = new JNode(K_BOOL);
                node.num = 1;
                return node;
            }
            if (c == 'f') {
                at += 5;
                JNode node = new JNode(K_BOOL);
                node.num = 0;
                return node;
            }
            if (c == 'n') {
                at += 4;
                return new JNode(K_NULL);
            }
            JNode node = new JNode(K_INT);
            node.num = integer();
            return node;
        }

        String text() {
            at += 1;
            int start = at;
            while (src.charAt(at) != '"') at += 1;
            String out = src.substring(start, at);
            at += 1;
            return out;
        }

        long integer() {
            long acc = 0;
            while (at < n) {
                char c = src.charAt(at);
                if (c < '0' || c > '9') break;
                acc = acc * 10 + (c - 48);
                at += 1;
            }
            return acc;
        }

        JNode array() {
            JNode node = new JNode(K_LIST);
            at += 1;
            skip();
            if (src.charAt(at) == ']') {
                at += 1;
                return node;
            }
            while (true) {
                node.items.add(value());
                skip();
                char c = src.charAt(at);
                at += 1;
                if (c == ']') break;
            }
            return node;
        }

        JNode object() {
            JNode node = new JNode(K_DICT);
            at += 1;
            skip();
            if (src.charAt(at) == '}') {
                at += 1;
                return node;
            }
            while (true) {
                skip();
                String key = text();
                skip();
                at += 1;
                node.fields.put(key, value());
                skip();
                char c = src.charAt(at);
                at += 1;
                if (c == '}') break;
            }
            return node;
        }
    }

    static long walk(JNode node) {
        int k = node.kind;
        if (k == K_INT) return node.num;
        if (k == K_BOOL) return node.num;
        if (k == K_STR) return node.text.length();
        if (k == K_LIST) {
            long total = 0;
            for (JNode item : node.items) total += walk(item);
            return total;
        }
        if (k == K_DICT) {
            long total = 0;
            for (Map.Entry<String, JNode> e : node.fields.entrySet()) {
                total += e.getKey().length() + walk(e.getValue());
            }
            return total;
        }
        return 0;
    }

    static String document(int records) {
        List<String> parts = new ArrayList<>();
        parts.add("[");
        long s = 7;
        int i = 0;
        while (i < records) {
            if (i > 0) parts.add(", ");
            s = (s * 1103515245L + 12345L) % 2147483648L;
            long a = (s / 65536) % 1000;
            long b = s % 997;
            String flag = (a % 2 == 0) ? "true" : "false";
            parts.add("{\"id\": " + i + ", \"name\": \"item" + a + "\", \"tags\": ["
                      + a + ", " + b + ", " + (a + b) + "], \"ok\": " + flag
                      + ", \"note\": null}");
            i += 1;
        }
        parts.add("]");
        StringBuilder sb = new StringBuilder();
        for (String p : parts) sb.append(p);
        return sb.toString();
    }

    public static void main(String[] args) {
        String src = document(RECORDS);
        long total = 0;
        for (int r = 0; r < REPS; r++) {
            Parser p = new Parser(src);
            JNode root = p.value();
            total = walk(root);
        }
        System.out.println(src.length());
        System.out.println(total);
    }
}
