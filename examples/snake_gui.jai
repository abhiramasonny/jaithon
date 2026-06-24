import std
import gui/window
import gui/keyboard

# Loads snake_weights.txt (produced by snake_es.jai) and displays
# the trained AI playing snake in a live GUI window.

# ---- GRID / WINDOW ----
var GRID_W = 12
var GRID_H = 12
var CELL = 40
var MARGIN = 30
var HEADER = 50

var WIN_W = GRID_W * CELL + MARGIN * 2
var WIN_H = GRID_H * CELL + MARGIN + HEADER
var GRID_X = MARGIN
var GRID_Y = HEADER

# ---- NETWORK (must match snake_es.jai) ----
var N_IN = 24
var H1 = 16
var H2 = 8
var N_OUT = 4
var _W1 = 0
var _B1 = 384
var _W2 = 400
var _B2 = 528
var _W3 = 536
var _B3 = 568
var GSIZE = 572

var MAX_NO_FOOD = 200

var _DX4 = [0, 1, 0, -1]
var _DY4 = [-1, 0, 1, 0]
var _DX8 = [0, 1, 1, 1, 0, -1, -1, -1]
var _DY8 = [-1, -1, 0, 1, 1, 1, 0, -1]

# ---- NETWORK FORWARD PASS ----
func _tanh(x)
    if x > 10.0
        return 1.0
    end
    if x < -10.0
        return -1.0
    end
    var e2x = exp(2.0 * x)
    return (e2x - 1.0) / (e2x + 1.0)
end

func _forward(w, ins)
    var h1 = []
    var j = 0
    while j < H1
        var s = w[_B1 + j]
        var i = 0
        while i < N_IN
            s = s + ins[i] * w[_W1 + i * H1 + j]
            i = i + 1
        end
        _push(h1, _tanh(s))
        j = j + 1
    end
    var h2 = []
    j = 0
    while j < H2
        var s = w[_B2 + j]
        var i = 0
        while i < H1
            s = s + h1[i] * w[_W2 + i * H2 + j]
            i = i + 1
        end
        _push(h2, _tanh(s))
        j = j + 1
    end
    var out = []
    j = 0
    while j < N_OUT
        var s = w[_B3 + j]
        var i = 0
        while i < H2
            s = s + h2[i] * w[_W3 + i * N_OUT + j]
            i = i + 1
        end
        _push(out, s)
        j = j + 1
    end
    return out
end

# ---- SNAKE ----
func _foodPos(body)
    var tries = 0
    while tries < 200
        var fx = _int(_rand() * GRID_W)
        var fy = _int(_rand() * GRID_H)
        var ok = true
        var bi = 0
        while bi < _alen(body)
            if body[bi][0] == fx and body[bi][1] == fy
                ok = false
                break
            end
            bi = bi + 1
        end
        if ok
            return [fx, fy]
        end
        tries = tries + 1
    end
    return [0, 0]
end

func _inputs(s)
    var head = s["body"][0]
    var hx = head[0]
    var hy = head[1]
    var food = s["food"]
    var body = s["body"]
    # flat occupancy grid (1 = body cell), indexed cy*GRID_W+cx
    var occ = _buf(GRID_W * GRID_H)
    var bi = 1
    while bi < _alen(body)
        _bset(occ, body[bi][1] * GRID_W + body[bi][0], 1)
        bi = bi + 1
    end
    var ins = []
    var d = 0
    while d < 8
        var vx = _DX8[d]
        var vy = _DY8[d]
        var cx = hx + vx
        var cy = hy + vy
        var dist = 1
        var foodF = 0
        var bodyF = 0
        while cx >= 0 and cx < GRID_W and cy >= 0 and cy < GRID_H
            if foodF == 0 and cx == food[0] and cy == food[1]
                foodF = 1
            end
            if bodyF == 0 and _bget(occ, cy * GRID_W + cx) == 1
                bodyF = 1
            end
            cx = cx + vx
            cy = cy + vy
            dist = dist + 1
        end
        _push(ins, 1.0 / dist)
        _push(ins, _num(foodF))
        _push(ins, _num(bodyF))
        d = d + 1
    end
    return ins
end

# ---- WEIGHT LOADING ----
func loadWeights(path)
    var content = readFile(path)
    if content == null
        return null
    end
    var lns = lines(content)
    var w = []
    var i = 0
    while i < _alen(lns)
        var l = trim(lns[i])
        if len(l) > 0
            _push(w, _num(l))
        end
        i = i + 1
    end
    if _alen(w) != GSIZE
        print concat("Weight count mismatch: got ", concat(_str(_alen(w)), concat(" need ", _str(GSIZE))))
        return null
    end
    return w
end

# ---- PIXEL DIGIT RENDERER ----
# Each digit encoded as 21-char string (3 cols x 7 rows), '1' = lit pixel
var DIGS = ["111101101101101101111", "010110010010010010111", "111001001111100100111", "111001001111001001111", "101101101111001001001", "111100100111001001111", "111100100111101101111", "111001001001001001001", "111101101111101101111", "111101101111001001111"]

func drawDigit(win, d, x, y, ps, color)
    var pat = DIGS[d]
    var row = 0
    while row < 7
        var col = 0
        while col < 3
            if charAt(pat, row * 3 + col) == "1"
                win.fillRect(x + col * ps, y + row * ps, ps, ps, color)
            end
            col = col + 1
        end
        row = row + 1
    end
end

func drawNumber(win, n, x, y, ps, color)
    if n < 0
        n = 0
    end
    var s = _str(_int(n))
    var i = 0
    while i < len(s)
        var d = _int(_num(charAt(s, i)))
        drawDigit(win, d, x + i * (4 * ps), y, ps, color)
        i = i + 1
    end
end

# ---- MAIN ----
var weights = loadWeights("snake_weights.txt")
if weights == null
    weights = loadWeights("examples/snake_weights.txt")
end
if weights == null
    print "snake_weights.txt not found — run snake_es.jai first"
end

var win = new Window()
win.init(WIN_W, WIN_H, "Snake AI")

var totalGames = 0
var bestScore = 0
var stepDelay = 0.1
var accumTime = 0.0
var restartTimer = 0.0
var RESTART_DELAY = 0.8

var snake = []
var food = [0, 0]
var gameScore = 0
var noFood = 0
var alive = true

func resetGame()
    var cx = _int(GRID_W / 2)
    var cy = _int(GRID_H / 2)
    snake = [[cx, cy], [cx - 1, cy], [cx - 2, cy]]
    food = _foodPos(snake)
    gameScore = 0
    noFood = 0
    alive = true
end

resetGame()

while true
    win.poll()
    var dt = win.getDeltaTime()

    if win.keyDown(KEY_ESC)
        break
    end

    # speed: +/= faster, - slower, space toggle
    if win.keyDown(KEY_SPACE)
        if stepDelay > 0.05
            stepDelay = 0.03
        else
            stepDelay = 0.1
        end
    end

    if restartTimer > 0.0
        restartTimer = restartTimer - dt
        if restartTimer <= 0.0
            restartTimer = 0.0
            resetGame()
        end
    else if alive and weights != null
        accumTime = accumTime + dt
        while accumTime >= stepDelay
            accumTime = accumTime - stepDelay

            # build state dict for _inputs
            var state = _dict()
            state["body"] = snake
            state["food"] = food

            var ins = _inputs(state)
            var outs = _forward(weights, ins)
            var best = outs[0]
            var act = 0
            var oi = 1
            while oi < 4
                if outs[oi] > best
                    best = outs[oi]
                    act = oi
                end
                oi = oi + 1
            end

            var dx = _DX4[act]
            var dy = _DY4[act]
            var head = snake[0]
            var nx = head[0] + dx
            var ny = head[1] + dy

            if nx < 0 or nx >= GRID_W or ny < 0 or ny >= GRID_H
                alive = false
            else
                var i = 0
                while i < _alen(snake) - 1
                    if snake[i][0] == nx and snake[i][1] == ny
                        alive = false
                        break
                    end
                    i = i + 1
                end
            end

            if alive
                var nb = [[nx, ny]]
                var i = 0
                while i < _alen(snake) - 1
                    _push(nb, snake[i])
                    i = i + 1
                end
                noFood = noFood + 1
                if nx == food[0] and ny == food[1]
                    gameScore = gameScore + 1
                    noFood = 0
                    _push(nb, snake[_alen(snake) - 1])
                    food = _foodPos(nb)
                    if gameScore > bestScore
                        bestScore = gameScore
                    end
                end
                snake = nb
                if noFood >= MAX_NO_FOOD
                    alive = false
                end
            end

            if not alive
                totalGames = totalGames + 1
                restartTimer = RESTART_DELAY
                accumTime = 0.0
                break
            end
        end
    end

    # ---- RENDER ----
    win.clear(rgb(15, 15, 20))

    # header background
    win.fillRect(0, 0, WIN_W, HEADER - 2, rgb(25, 25, 35))

    # score label box (left)
    win.fillRect(MARGIN, 9, 6, 32, rgb(80, 200, 100))

    # best label box (right)
    win.fillRect(WIN_W - MARGIN - 70, 9, 6, 32, rgb(220, 180, 30))

    # score digits
    drawNumber(win, gameScore, MARGIN + 12, 10, 4, rgb(80, 200, 100))

    # best score digits
    drawNumber(win, bestScore, WIN_W - MARGIN - 58, 10, 4, rgb(220, 180, 30))

    # games played (small, top center)
    drawNumber(win, totalGames, WIN_W / 2 - 8, 18, 2, rgb(120, 120, 140))

    # grid border
    win.fillRect(GRID_X - 3, GRID_Y - 3, GRID_W * CELL + 6, GRID_H * CELL + 6, rgb(60, 60, 80))

    # grid background
    win.fillRect(GRID_X, GRID_Y, GRID_W * CELL, GRID_H * CELL, rgb(22, 22, 30))

    # subtle grid lines
    var gx = 0
    while gx <= GRID_W
        win.fillRect(GRID_X + gx * CELL, GRID_Y, 1, GRID_H * CELL, rgb(30, 30, 42))
        gx = gx + 1
    end
    var gy = 0
    while gy <= GRID_H
        win.fillRect(GRID_X, GRID_Y + gy * CELL, GRID_W * CELL, 1, rgb(30, 30, 42))
        gy = gy + 1
    end

    # food (red with glow effect)
    win.fillRect(GRID_X + food[0] * CELL + 4, GRID_Y + food[1] * CELL + 4, CELL - 8, CELL - 8, rgb(200, 50, 50))
    win.fillRect(GRID_X + food[0] * CELL + 8, GRID_Y + food[1] * CELL + 8, CELL - 16, CELL - 16, rgb(255, 100, 100))

    # snake body
    var bodyColor = rgb(40, 140, 70)
    var headColor = rgb(80, 220, 110)
    if not alive
        bodyColor = rgb(80, 80, 80)
        headColor = rgb(120, 120, 120)
    end

    var i = _alen(snake) - 1
    while i >= 0
        var seg = snake[i]
        var col = bodyColor
        if i == 0
            col = headColor
        end
        win.fillRect(GRID_X + seg[0] * CELL + 2, GRID_Y + seg[1] * CELL + 2, CELL - 4, CELL - 4, col)
        i = i - 1
    end

    # head eye dots
    if alive and _alen(snake) > 0
        var h = snake[0]
        win.fillRect(GRID_X + h[0] * CELL + 10, GRID_Y + h[1] * CELL + 10, 5, 5, rgb(20, 20, 20))
        win.fillRect(GRID_X + h[0] * CELL + 24, GRID_Y + h[1] * CELL + 10, 5, 5, rgb(20, 20, 20))
    end

    win.display()
    if not win.update()
        break
    end
end
