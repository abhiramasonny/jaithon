// Same program as json_parse.jai. See tests/bench/README.md for why the C++ and
// Java rows are here and what they are not.
//
// Parsed by hand, like every other port: reaching for a JSON library here would
// measure the library, not the language.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

static long bench_scale() {
    const char *l = std::getenv("BENCH_LEVEL");
    if (l && std::strcmp(l, "easy") == 0) return 16;
    if (l && std::strcmp(l, "medium") == 0) return 4;
    return 1;
}
static const long SCALE = bench_scale();
static const int REPS = 6;

enum { K_NULL = 0, K_BOOL = 1, K_INT = 2, K_STR = 3, K_LIST = 4, K_DICT = 5 };

struct JNode {
    int kind;
    int64_t num;
    std::string text;
    std::vector<std::unique_ptr<JNode>> items;
    std::unordered_map<std::string, std::unique_ptr<JNode>> fields;
    explicit JNode(int k) : kind(k), num(0) {}
};

typedef std::unique_ptr<JNode> Node;

struct Parser {
    const std::string &src;
    size_t at;
    size_t n;
    explicit Parser(const std::string &s) : src(s), at(0), n(s.size()) {}

    void skip() {
        while (at < n && src[at] == ' ') at += 1;
    }

    Node value() {
        skip();
        char c = src[at];
        if (c == '{') return object();
        if (c == '[') return array();
        if (c == '"') {
            Node node(new JNode(K_STR));
            node->text = text();
            return node;
        }
        if (c == 't') {
            at += 4;
            Node node(new JNode(K_BOOL));
            node->num = 1;
            return node;
        }
        if (c == 'f') {
            at += 5;
            Node node(new JNode(K_BOOL));
            node->num = 0;
            return node;
        }
        if (c == 'n') {
            at += 4;
            return Node(new JNode(K_NULL));
        }
        Node node(new JNode(K_INT));
        node->num = integer();
        return node;
    }

    std::string text() {
        at += 1;
        size_t start = at;
        while (src[at] != '"') at += 1;
        std::string out = src.substr(start, at - start);
        at += 1;
        return out;
    }

    int64_t integer() {
        int64_t acc = 0;
        while (at < n) {
            char c = src[at];
            if (c < '0' || c > '9') break;
            acc = acc * 10 + (int64_t)(c - 48);
            at += 1;
        }
        return acc;
    }

    Node array() {
        Node node(new JNode(K_LIST));
        at += 1;
        skip();
        if (src[at] == ']') {
            at += 1;
            return node;
        }
        for (;;) {
            node->items.push_back(value());
            skip();
            char c = src[at];
            at += 1;
            if (c == ']') break;
        }
        return node;
    }

    Node object() {
        Node node(new JNode(K_DICT));
        at += 1;
        skip();
        if (src[at] == '}') {
            at += 1;
            return node;
        }
        for (;;) {
            skip();
            std::string key = text();
            skip();
            at += 1;
            node->fields[key] = value();
            skip();
            char c = src[at];
            at += 1;
            if (c == '}') break;
        }
        return node;
    }
};

static int64_t walk(const JNode *node) {
    int k = node->kind;
    if (k == K_INT) return node->num;
    if (k == K_BOOL) return node->num;
    if (k == K_STR) return (int64_t)node->text.size();
    if (k == K_LIST) {
        int64_t total = 0;
        for (const Node &item : node->items) total += walk(item.get());
        return total;
    }
    if (k == K_DICT) {
        int64_t total = 0;
        for (const auto &kv : node->fields) {
            total += (int64_t)kv.first.size() + walk(kv.second.get());
        }
        return total;
    }
    return 0;
}

static std::string document(int records) {
    std::vector<std::string> parts;
    parts.push_back("[");
    int64_t s = 7;
    int i = 0;
    char buf[256];
    while (i < records) {
        if (i > 0) parts.push_back(", ");
        s = (s * 1103515245LL + 12345LL) % 2147483648LL;
        int64_t a = (s / 65536) % 1000;
        int64_t b = s % 997;
        const char *flag = (a % 2 == 0) ? "true" : "false";
        std::snprintf(buf, sizeof buf,
                      "{\"id\": %d, \"name\": \"item%lld\", \"tags\": [%lld, %lld, %lld],"
                      " \"ok\": %s, \"note\": null}",
                      i, (long long)a, (long long)a, (long long)b,
                      (long long)(a + b), flag);
        parts.push_back(buf);
        i += 1;
    }
    parts.push_back("]");
    size_t total = 0;
    for (const std::string &p : parts) total += p.size();
    std::string out;
    out.reserve(total);
    for (const std::string &p : parts) out += p;
    return out;
}

int main() {
    int records = (int)(2500 / SCALE);
    if (records < 1) records = 1;
    std::string src = document(records);
    int64_t total = 0;
    for (int r = 0; r < REPS; r++) {
        Parser p(src);
        Node root = p.value();
        total = walk(root.get());
    }
    std::printf("%d\n", (int)src.size());
    std::printf("%lld\n", (long long)total);
    return 0;
}
