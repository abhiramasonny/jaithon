// Same program as fib_recursive.jai.
#include <cstdint>
#include <cstdio>

static int64_t fib(int64_t n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); }

int main() {
    std::printf("%lld\n", (long long)fib(30));
    return 0;
}
