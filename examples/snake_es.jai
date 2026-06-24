import std

# Snake trained with Deep MLP + Truncation Evolution Strategy
# Architecture: 24 -> 16 -> 8 -> 4  (tanh hidden, linear output)
# Far faster convergence than NEAT: no wasted topology search,
# Gaussian noise in full 572-dim weight space, parallel eval.

var GRID_W = 12
var GRID_H = 12
var N_IN = 24
var H1 = 16
var H2 = 8
var N_OUT = 4

# flat genome layout (row-major)
# W1: N_IN x H1 = 384   offset 0
# b1: H1        = 16    offset 384
# W2: H1 x H2  = 128   offset 400
# b2: H2        = 8     offset 528
# W3: H2 x N_OUT = 32  offset 536
# b3: N_OUT     = 4     offset 568
var _W1 = 0
var _B1 = 384
var _W2 = 400
var _B2 = 528
var _W3 = 536
var _B3 = 568
var GSIZE = 572

var POP = 100
var GENS = 150
var N_EVAL = 5
var MAX_NO_FOOD = 150
var TOP_K = 10
var SIGMA = 0.5
var SIGMA_DECAY = 0.992
var SIGMA_MIN = 0.02
var STAG_LIMIT = 12

var _DX4 = [0, 1, 0, -1]
var _DY4 = [-1, 0, 1, 0]
var _DX8 = [0, 1, 1, 1, 0, -1, -1, -1]
var _DY8 = [-1, -1, 0, 1, 1, 1, 0, -1]

# CLT approx for N(0,1): sum of 12 uniform[0,1] - 6
func _randN()
    var s = 0.0
    var i = 0
    while i < 12
        s = s + _rand()
        i = i + 1
    end
    return s - 6.0
end

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

func _mkGenome()
    var g = _dict()
    var w = []
    var scale = 0.3
    var i = 0
    while i < GSIZE
        _push(w, _randN() * scale)
        i = i + 1
    end
    g["w"] = w
    g["fit"] = 0.0
    return g
end

func _mkGenomeFrom(w)
    var g = _dict()
    g["w"] = w
    g["fit"] = 0.0
    return g
end

func saveWeights(w, path)
    var s = ""
    var i = 0
    while i < _alen(w)
        s = concat(s, concat(_str(w[i]), "\n"))
        i = i + 1
    end
    writeFile(path, s)
end

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
        return null
    end
    return w
end

func _mutate(parent, sigma)
    var g = _dict()
    var pw = parent["w"]
    var w = []
    var i = 0
    while i < GSIZE
        _push(w, pw[i] + _randN() * sigma)
        i = i + 1
    end
    g["w"] = w
    g["fit"] = 0.0
    return g
end

func _blend(pa, pb, sigma)
    var g = _dict()
    var wa = pa["w"]
    var wb = pb["w"]
    var w = []
    var i = 0
    while i < GSIZE
        var t = _rand()
        _push(w, wa[i] * t + wb[i] * (1.0 - t) + _randN() * sigma)
        i = i + 1
    end
    g["w"] = w
    g["fit"] = 0.0
    return g
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

func _mkSnake()
    var cx = _int(GRID_W / 2)
    var cy = _int(GRID_H / 2)
    var body = [[cx, cy], [cx - 1, cy], [cx - 2, cy]]
    var s = _dict()
    s["body"] = body
    s["food"] = _foodPos(body)
    s["score"] = 0
    s["steps"] = 0
    s["noFood"] = 0
    s["alive"] = true
    return s
end

func _step(s, act)
    if not s["alive"]
        return
    end
    var dx = _DX4[act]
    var dy = _DY4[act]
    var head = s["body"][0]
    var nx = head[0] + dx
    var ny = head[1] + dy
    if nx < 0 or nx >= GRID_W or ny < 0 or ny >= GRID_H
        s["alive"] = false
        return
    end
    var body = s["body"]
    var i = 0
    while i < _alen(body) - 1
        if body[i][0] == nx and body[i][1] == ny
            s["alive"] = false
            return
        end
        i = i + 1
    end
    var nb = [[nx, ny]]
    i = 0
    while i < _alen(body) - 1
        _push(nb, body[i])
        i = i + 1
    end
    s["body"] = nb
    s["steps"] = s["steps"] + 1
    s["noFood"] = s["noFood"] + 1
    if nx == s["food"][0] and ny == s["food"][1]
        s["score"] = s["score"] + 1
        s["noFood"] = 0
        _push(s["body"], body[_alen(body) - 1])
        s["food"] = _foodPos(s["body"])
    end
    if s["noFood"] >= MAX_NO_FOOD
        s["alive"] = false
    end
end

func _inputs(s)
    var head = s["body"][0]
    var hx = head[0]
    var hy = head[1]
    var food = s["food"]
    var body = s["body"]
    # flat occupancy grid (1 = body cell), indexed cy*GRID_W+cx.
    # Replaces the old string-key dict + per-cell concat/hash in the raycast.
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

func _eval(g)
    var total = 0.0
    var ei = 0
    while ei < N_EVAL
        var s = _mkSnake()
        while s["alive"]
            var ins = _inputs(s)
            var outs = _forward(g["w"], ins)
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
            _step(s, act)
        end
        var head = s["body"][0]
        var food = s["food"]
        var ddx = abs(head[0] - food[0])
        var ddy = abs(head[1] - food[1])
        total = total + s["steps"] + s["score"] * 500.0 + (GRID_W + GRID_H - ddx - ddy)
        ei = ei + 1
    end
    g["fit"] = total / N_EVAL
    return g["fit"]
end

func _render(s)
    var body = s["body"]
    var food = s["food"]
    var bar = ""
    var i = 0
    while i < GRID_W + 2
        bar = concat(bar, "#")
        i = i + 1
    end
    print bar
    var ry = 0
    while ry < GRID_H
        var row = "#"
        var rx = 0
        while rx < GRID_W
            var cell = " "
            if food[0] == rx and food[1] == ry
                cell = "*"
            end
            var bi = 0
            while bi < _alen(body)
                if body[bi][0] == rx and body[bi][1] == ry
                    if bi == 0
                        cell = "O"
                    else
                        cell = "o"
                    end
                    break
                end
                bi = bi + 1
            end
            row = concat(row, cell)
            rx = rx + 1
        end
        print concat(row, "#")
        ry = ry + 1
    end
    print bar
end

func _playGame(g)
    var s = _mkSnake()
    while s["alive"]
        var ins = _inputs(s)
        var outs = _forward(g["w"], ins)
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
        _step(s, act)
    end
    return s
end

func _runDemo(g)
    var totalScore = 0
    var bestScore = 0
    var bestState = null
    var di = 0
    while di < 10
        var s = _playGame(g)
        totalScore = totalScore + s["score"]
        if s["score"] > bestScore
            bestScore = s["score"]
            bestState = s
        end
        di = di + 1
    end
    if bestState == null
        bestState = _playGame(g)
    end
    _render(bestState)
    print concat("Best game  : Score=", concat(_str(bestState["score"]), concat("  Steps=", _str(bestState["steps"]))))
    print concat("Avg score  : ", _str(totalScore / 10))
    return totalScore / 10
end

print "================================================="
print "  Snake -- Deep MLP + Evolution Strategy"
print "================================================="
print concat("  Grid  : ", concat(_str(GRID_W), concat("x", _str(GRID_H))))
print concat("  Pop   : ", _str(POP))
print concat("  Arch  : 24->16->8->4  params=", _str(GSIZE))
print concat("  Gens  : ", concat(_str(GENS), concat("  evals/genome=", _str(N_EVAL))))
print concat("  Top-K : ", concat(_str(TOP_K), "  sigma0=0.5"))
print "================================================="
print ""

var savedW = loadWeights("snake_weights.txt")
var pop = []
var gi = 0
if savedW != null
    print "  Resuming from snake_weights.txt"
    while gi < POP
        _push(pop, _mutate(_mkGenomeFrom(savedW), SIGMA * 0.3))
        gi = gi + 1
    end
    _set(pop, 0, _mkGenomeFrom(savedW))
else
    while gi < POP
        _push(pop, _mkGenome())
        gi = gi + 1
    end
end

var bestG = pop[0]
var bestFit = -1.0
var sigma = SIGMA
var noImprove = 0

var gen = 0
while gen < GENS
    if gen < 20
        MAX_NO_FOOD = 80
    else if gen < 60
        MAX_NO_FOOD = 130
    else
        MAX_NO_FOOD = 200
    end

    peval_pop(pop, _eval)

    # sort indices by fitness descending (insertion sort)
    var sorted = []
    gi = 0
    while gi < POP
        _push(sorted, gi)
        gi = gi + 1
    end
    var k = 1
    while k < _alen(sorted)
        var kv = sorted[k]
        var j = k - 1
        while j >= 0
            if pop[sorted[j]]["fit"] < pop[kv]["fit"]
                _set(sorted, j + 1, sorted[j])
                j = j - 1
            else
                break
            end
        end
        _set(sorted, j + 1, kv)
        k = k + 1
    end

    var genBest = pop[sorted[0]]
    var genAvg = 0.0
    gi = 0
    while gi < POP
        genAvg = genAvg + pop[gi]["fit"]
        gi = gi + 1
    end
    genAvg = genAvg / POP

    if genBest["fit"] > bestFit
        bestFit = genBest["fit"]
        bestG = genBest
        noImprove = 0
        sigma = sigma * SIGMA_DECAY
        if sigma < SIGMA_MIN
            sigma = SIGMA_MIN
        end
    else
        noImprove = noImprove + 1
        if noImprove >= STAG_LIMIT
            sigma = sigma * 2.0
            if sigma > SIGMA
                sigma = SIGMA
            end
            noImprove = 0
        end
    end

    var sigmaDisp = _int(sigma * 1000)
    print concat("Gen ", concat(_str(gen + 1), concat(" | Best:", concat(_str(_int(genBest["fit"])), concat(" Avg:", concat(_str(_int(genAvg)), concat(" sig:", _str(sigmaDisp))))))))

    # breed next generation
    var np = []
    _push(np, bestG)
    var cutoff = min(TOP_K, _alen(sorted))
    gi = 1
    while gi < POP
        var pidx = sorted[_int(_rand() * cutoff)]
        var child = null
        if _rand() < 0.3 and cutoff >= 2
            var pidx2 = sorted[_int(_rand() * cutoff)]
            child = _blend(pop[pidx], pop[pidx2], sigma * 0.3)
        else
            child = _mutate(pop[pidx], sigma)
        end
        _push(np, child)
        gi = gi + 1
    end
    pop = np

    gen = gen + 1
end

MAX_NO_FOOD = 200
print ""
print "================================================="
print concat("  Training done — best fitness: ", _str(_int(bestFit)))
print "================================================="
print ""
_runDemo(bestG)
saveWeights(bestG["w"], "snake_weights.txt")
print "  Weights saved -> snake_weights.txt"
print "  Run snake_gui.jai to watch the AI play"
