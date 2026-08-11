def zeros(n):
    row = []
    for _i in range(n):
        row.append(0)
    return row


def seed_board(h, w, seed):
    board = []
    s = seed
    board.append(zeros(w + 2))
    for _r in range(h):
        row = []
        row.append(0)
        for _c in range(w):
            s = (s * 1103515245 + 12345) % 2147483648
            bits = (s // 65536) % 100
            alive = 0
            if bits < 35:
                alive = 1
            row.append(alive)
        row.append(0)
        board.append(row)
    board.append(zeros(w + 2))
    return board


def step(board, h, w):
    out = []
    out.append(zeros(w + 2))
    for r in range(1, h + 1):
        up = board[r - 1]
        mid = board[r]
        down = board[r + 1]
        row = []
        row.append(0)
        for c in range(1, w + 1):
            n = up[c - 1] + up[c] + up[c + 1]
            n += mid[c - 1] + mid[c + 1]
            n += down[c - 1] + down[c] + down[c + 1]
            alive = 0
            if n == 3:
                alive = 1
            elif n == 2:
                alive = mid[c]
            row.append(alive)
        row.append(0)
        out.append(row)
    out.append(zeros(w + 2))
    return out


h = 500
w = 500
generations = 70

board = seed_board(h, w, 7)
for _g in range(generations):
    board = step(board, h, w)

live = 0
for r in range(1, h + 1):
    row = board[r]
    for c in range(1, w + 1):
        live += row[c]
print(generations)
print(live)
