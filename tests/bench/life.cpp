// Same program as life.jai. See tests/bench/README.md for why the C++ and Java
// rows are here and what they are not.
#include <cstdint>
#include <cstdio>
#include <vector>

static std::vector<int64_t> zeros(int n) {
    std::vector<int64_t> row;
    for (int i = 0; i < n; i++) row.push_back(0);
    return row;
}

static std::vector<std::vector<int64_t>> seed_board(int h, int w, int64_t seed) {
    std::vector<std::vector<int64_t>> board;
    int64_t s = seed;
    board.push_back(zeros(w + 2));
    for (int r = 0; r < h; r++) {
        std::vector<int64_t> row;
        row.push_back(0);
        for (int c = 0; c < w; c++) {
            s = (s * 1103515245 + 12345) % 2147483648;
            int64_t bits = (s / 65536) % 100;
            int64_t alive = 0;
            if (bits < 35) alive = 1;
            row.push_back(alive);
        }
        row.push_back(0);
        board.push_back(row);
    }
    board.push_back(zeros(w + 2));
    return board;
}

static std::vector<std::vector<int64_t>> step(
    const std::vector<std::vector<int64_t>> &board, int h, int w) {
    std::vector<std::vector<int64_t>> out;
    out.push_back(zeros(w + 2));
    for (int r = 1; r < h + 1; r++) {
        const std::vector<int64_t> &up = board[r - 1];
        const std::vector<int64_t> &mid = board[r];
        const std::vector<int64_t> &down = board[r + 1];
        std::vector<int64_t> row;
        row.push_back(0);
        for (int c = 1; c < w + 1; c++) {
            int64_t n = up[c - 1] + up[c] + up[c + 1];
            n += mid[c - 1] + mid[c + 1];
            n += down[c - 1] + down[c] + down[c + 1];
            int64_t alive = 0;
            if (n == 3) {
                alive = 1;
            } else if (n == 2) {
                alive = mid[c];
            }
            row.push_back(alive);
        }
        row.push_back(0);
        out.push_back(row);
    }
    out.push_back(zeros(w + 2));
    return out;
}

int main() {
    const int h = 100;
    const int w = 100;
    const int generations = 50;

    std::vector<std::vector<int64_t>> board = seed_board(h, w, 7);
    for (int g = 0; g < generations; g++) board = step(board, h, w);

    int64_t live = 0;
    for (int r = 1; r < h + 1; r++) {
        const std::vector<int64_t> &row = board[r];
        for (int c = 1; c < w + 1; c++) live += row[c];
    }
    std::printf("%d\n", generations);
    std::printf("%lld\n", (long long)live);
    return 0;
}
