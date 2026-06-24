import std

# Snake trained with a *GPU-batched* Evolution Strategy.
#
# The whole population plays in lockstep. At every step the forward pass of ALL
# genomes is fused into one batched matmul per layer (_gpu_matmul_batched), so
# the M2 Max GPU does the heavy linear algebra instead of the interpreter.
# This makes a much larger net (24->64->64->4, 6020 params) trainable -- a
# net this size is ~500x too slow to forward on the CPU interpreter (see
# gpu_bench.jai), so the GPU is what makes it feasible.

var GRID_W = 12
var GRID_H = 12
var N_IN = 24
var H1 = 64
var H2 = 64
var N_OUT = 4

# flat genome layout (row-major), all weights of one network
var _W1 = 0
var _B1 = N_IN * H1
var _W2 = _B1 + H1
var _B2 = _W2 + H1 * H2
var _W3 = _B2 + H2
var _B3 = _W3 + H2 * N_OUT
var GSIZE = _B3 + N_OUT

var POP = 64
var GENS = 30
var MAX_STEPS = 200
var TOP_K = 8
var SIGMA = 0.5
var SIGMA_DECAY = 0.99
var SIGMA_MIN = 0.03

var _DX4 = [0, 1, 0, -1]
var _DY4 = [-1, 0, 1, 0]
var _DX8 = [0, 1, 1, 1, 0, -1, -1, -1]
var _DY8 = [-1, -1, 0, 1, 1, 1, 0, -1]

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

func _mkGenome()
    var g = _dict()
    var w = []
    var i = 0
    while i < GSIZE
        _push(w, _randN() * 0.3)
        i = i + 1
    end
    g["w"] = w
    g["fit"] = 0.0
    return g
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
    s["fit"] = 0.0
    return s
end

func _inputs(s)
    var head = s["body"][0]
    var hx = head[0]
    var hy = head[1]
    var food = s["food"]
    var body = s["body"]
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

func _stepGame(s, act)
    if not s["alive"]
        return null
    end
    var head = s["body"][0]
    var nx = head[0] + _DX4[act]
    var ny = head[1] + _DY4[act]
    if nx < 0 or nx >= GRID_W or ny < 0 or ny >= GRID_H
        s["alive"] = false
        return null
    end
    var body = s["body"]
    var i = 0
    while i < _alen(body) - 1
        if body[i][0] == nx and body[i][1] == ny
            s["alive"] = false
            return null
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
    if s["noFood"] >= MAX_STEPS
        s["alive"] = false
    end
    return null
end

# ---- batched weight packing: build per-layer batched matrices once per gen ----
func _packLayer(pop, off, count)
    var out = []
    var p = 0
    while p < POP
        var w = pop[p]["w"]
        var k = 0
        while k < count
            _push(out, w[off + k])
            k = k + 1
        end
        p = p + 1
    end
    return out
end

# batched forward for the whole population. inBatch is flat POP*N_IN.
# returns flat POP*N_OUT.
func _forwardBatch(inBatch, W1b, B1b, W2b, B2b, W3b, B3b)
    var z1 = _gpu_matmul_batched(inBatch, W1b, POP, 1, N_IN, H1)
    var h1 = []
    var p = 0
    while p < POP
        var j = 0
        while j < H1
            _push(h1, _tanh(z1[p * H1 + j] + B1b[p * H1 + j]))
            j = j + 1
        end
        p = p + 1
    end
    var z2 = _gpu_matmul_batched(h1, W2b, POP, 1, H1, H2)
    var h2 = []
    p = 0
    while p < POP
        var j = 0
        while j < H2
            _push(h2, _tanh(z2[p * H2 + j] + B2b[p * H2 + j]))
            j = j + 1
        end
        p = p + 1
    end
    var z3 = _gpu_matmul_batched(h2, W3b, POP, 1, H2, N_OUT)
    var out = []
    p = 0
    while p < POP
        var j = 0
        while j < N_OUT
            _push(out, z3[p * N_OUT + j] + B3b[p * N_OUT + j])
            j = j + 1
        end
        p = p + 1
    end
    return out
end

# evaluate the whole population in lockstep on the GPU
func _evalPopGPU(pop)
    var W1b = _packLayer(pop, _W1, N_IN * H1)
    var B1b = _packLayer(pop, _B1, H1)
    var W2b = _packLayer(pop, _W2, H1 * H2)
    var B2b = _packLayer(pop, _B2, H2)
    var W3b = _packLayer(pop, _W3, H2 * N_OUT)
    var B3b = _packLayer(pop, _B3, N_OUT)

    var games = []
    var p = 0
    while p < POP
        _push(games, _mkSnake())
        p = p + 1
    end

    var step = 0
    var anyAlive = true
    while step < MAX_STEPS and anyAlive
        # build input batch (dead genomes contribute zeros)
        var inBatch = []
        p = 0
        while p < POP
            var s = games[p]
            if s["alive"]
                var ins = _inputs(s)
                var j = 0
                while j < N_IN
                    _push(inBatch, ins[j])
                    j = j + 1
                end
            else
                var j2 = 0
                while j2 < N_IN
                    _push(inBatch, 0.0)
                    j2 = j2 + 1
                end
            end
            p = p + 1
        end

        var out = _forwardBatch(inBatch, W1b, B1b, W2b, B2b, W3b, B3b)

        # argmax per genome -> action -> step
        anyAlive = false
        p = 0
        while p < POP
            var s = games[p]
            if s["alive"]
                var base = p * N_OUT
                var best = out[base]
                var act = 0
                var oi = 1
                while oi < N_OUT
                    if out[base + oi] > best
                        best = out[base + oi]
                        act = oi
                    end
                    oi = oi + 1
                end
                _stepGame(s, act)
                if s["alive"]
                    anyAlive = true
                end
            end
            p = p + 1
        end
        step = step + 1
    end

    # fitness -> back onto genomes
    p = 0
    while p < POP
        var s = games[p]
        var head = s["body"][0]
        var food = s["food"]
        var ddx = head[0] - food[0]
        if ddx < 0
            ddx = 0 - ddx
        end
        var ddy = head[1] - food[1]
        if ddy < 0
            ddy = 0 - ddy
        end
        pop[p]["fit"] = s["steps"] + s["score"] * 500.0 + (GRID_W + GRID_H - ddx - ddy)
        p = p + 1
    end
    return null
end

# ---- ES loop ----
print "================================================="
print "  Snake -- GPU-batched Evolution Strategy"
print "================================================="
print concat("  GPU    : ", _str(_gpu_available()))
print concat("  Arch   : 24->64->64->4  params=", _str(GSIZE))
print concat("  Pop    : ", concat(_str(POP), concat("  Gens=", _str(GENS))))
print "================================================="
print ""

var pop = []
var gi = 0
while gi < POP
    _push(pop, _mkGenome())
    gi = gi + 1
end

var bestG = pop[0]
var bestFit = 0.0 - 1.0
var sigma = SIGMA

var gen = 0
while gen < GENS
    var t0 = _time()
    _evalPopGPU(pop)
    var t1 = _time()

    # sort indices by fitness desc (insertion sort)
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
    if genBest["fit"] > bestFit
        bestFit = genBest["fit"]
        bestG = genBest
        sigma = sigma * SIGMA_DECAY
        if sigma < SIGMA_MIN
            sigma = SIGMA_MIN
        end
    end

    var ms = _int((t1 - t0) * 1000.0)
    var line = concat("Gen ", _str(gen + 1))
    line = concat(line, concat(" | genBest:", _str(_int(genBest["fit"]))))
    line = concat(line, concat(" allBest:", _str(_int(bestFit))))
    line = concat(line, concat(" eval=", concat(_str(ms), "ms")))
    print line

    # breed: elitism + mutated top-K
    var np = []
    _push(np, bestG)
    gi = 1
    while gi < POP
        var pidx = sorted[_int(_rand() * TOP_K)]
        _push(np, _mutate(pop[pidx], sigma))
        gi = gi + 1
    end
    pop = np
    gen = gen + 1
end

print ""
print concat("  Done. Best fitness: ", _str(_int(bestFit)))
print "  (GPU did every forward pass as a batched matmul)"
