import std

#config vars
var GRID_W = 12
var GRID_H = 12
var N_IN = 24          #8 dirs × (wall, food, body)
var N_OUT = 4          #up / right / down / left
var BIAS_ID = N_IN     #id of bias node
var OUT0 = N_IN + 1    #first output node id

var POP = 15
var GENS = 40
var N_EVAL = 5         #games per genome for fitness evaluation
var MAX_NO_FOOD = 200  #more steps to find food
var C1 = 2.0           # amplify structural distance signal
var C2 = 2.0
var C3 = 0.5
var COMPAT_T = 0.5     # was 2.0 — far too loose; 0.5 gives real species
var P_W = 0.8
var P_PERTURB = 0.85
var P_CONN = 0.2       # was 0.1
var P_NODE = 0.15      # was 0.08 — need hidden nodes to appear in winner
var P_DISABLE = 0.75
var STALE = 10         # was 15 — prune dead species faster


var _innov = 0
var _imap = _dict()
var _ncount = OUT0 + N_OUT - 1

#direction vectors for movement
var _DX4 = [0, 1, 0, -1]
var _DY4 = [-1, 0, 1, 0]

#for sensors
var _DX8 = [0, 1, 1, 1, 0, -1, -1, -1]
var _DY8 = [-1, -1, 0, 1, 1, 1, 0, -1]

func _iv(i, o)
    var k = concat(_str(i), concat("-", _str(o)))
    if _dict_has(_imap, k)
        return _imap[k]
    end
    _innov = _innov + 1
    _imap[k] = _innov
    return _innov
end

#     genome struct
# node : [id, type]          type: 0=in 1=hidden 2=out 3=bias
# conn : [in, out, w, en, innov]
# genome: {nodes, conns, fit, adjFit, sp}

func _genome()
    var g = _dict()
    g["nodes"] = []
    g["conns"] = []
    g["fit"] = 0
    g["adjFit"] = 0
    g["sp"] = -1
    return g
end

func _minGenome()
    var g = _genome()
    var i = 0
    while i < N_IN
        _push(g["nodes"], [i, 0])
        i = i + 1
    end
    _push(g["nodes"], [BIAS_ID, 3])
    i = 0
    while i < N_OUT
        _push(g["nodes"], [OUT0 + i, 2])
        i = i + 1
    end
    i = 0
    while i <= BIAS_ID
        var o = 0
        while o < N_OUT
            var iv = _iv(i, OUT0 + o)
            _push(g["conns"], [i, OUT0 + o, _rand() * 2 - 1, true, iv])
            o = o + 1
        end
        i = i + 1
    end
    return g
end

# ---- NETWORK ----
func _sig(x)
    return 1 / (1 + exp(0 - x))
end

func _activate(g, inputs)
    var sig = _dict()
    var nodes = g["nodes"]
    var i = 0
    while i < _alen(nodes)
        sig[_str(nodes[i][0])] = 0
        i = i + 1
    end
    i = 0
    while i < N_IN
        sig[_str(i)] = inputs[i]
        i = i + 1
    end
    sig[_str(BIAS_ID)] = 1

    var pass = 0
    while pass < 3
        var acc = _dict()
        var conns = g["conns"]
        i = 0
        while i < _alen(conns)
            var c = conns[i]
            if c[3]
                var sk = _str(c[0])
                var tk = _str(c[1])
                if _dict_has(sig, sk) and _dict_has(sig, tk)
                    if not _dict_has(acc, tk)
                        acc[tk] = 0
                    end
                    acc[tk] = acc[tk] + sig[sk] * c[2]
                end
            end
            i = i + 1
        end
        var j = 0
        while j < _alen(nodes)
            var n = nodes[j]
            if n[1] == 1 or n[1] == 2
                var k = _str(n[0])
                if _dict_has(acc, k)
                    sig[k] = _sig(acc[k])
                end
            end
            j = j + 1
        end
        pass = pass + 1
    end

    var out = []
    i = 0
    while i < N_OUT
        var k = _str(OUT0 + i)
        if _dict_has(sig, k)
            _push(out, sig[k])
        else
            _push(out, 0.5)
        end
        i = i + 1
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
    var bset = _dict()
    var bi = 1
    while bi < _alen(body)
        bset[concat(_str(body[bi][0]), concat("_", _str(body[bi][1])))] = true
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
            if bodyF == 0
                var bk = concat(_str(cx), concat("_", _str(cy)))
                if _dict_has(bset, bk)
                    bodyF = 1
                end
            end
            cx = cx + vx
            cy = cy + vy
            dist = dist + 1
        end
        _push(ins, 1 / dist)
        _push(ins, _num(foodF))
        _push(ins, _num(bodyF))
        d = d + 1
    end
    return ins
end

func _eval(g)
    var total = 0
    var ei = 0
    while ei < N_EVAL
        var s = _mkSnake()
        while s["alive"]
            var ins = _inputs(s)
            var outs = _activate(g, ins)
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
        var dx = abs(head[0] - food[0])
        var dy = abs(head[1] - food[1])
        var distBonus = GRID_W + GRID_H - dx - dy
        total = total + s["steps"] + s["score"] * 500 + distBonus
        ei = ei + 1
    end
    g["fit"] = total / N_EVAL
    return g["fit"]
end

# ---- MUTATION ----
func _mutW(g)
    var conns = g["conns"]
    var i = 0
    while i < _alen(conns)
        if _rand() < P_W
            if _rand() < P_PERTURB
                conns[i][2] = conns[i][2] + (_rand() * 0.4 - 0.2)
                if conns[i][2] > 4
                    conns[i][2] = 4
                end
                if conns[i][2] < -4
                    conns[i][2] = -4
                end
            else
                conns[i][2] = _rand() * 4 - 2
            end
        end
        i = i + 1
    end
end

func _mutConn(g)
    var nodes = g["nodes"]
    if _alen(nodes) < 2
        return
    end
    var t = 0
    while t < 20
        var ni = nodes[_int(_rand() * _alen(nodes))][0]
        var no = nodes[_int(_rand() * _alen(nodes))][0]
        if ni != no
            var dup = false
            var conns = g["conns"]
            var k = 0
            while k < _alen(conns)
                if conns[k][0] == ni and conns[k][1] == no
                    dup = true
                    break
                end
                k = k + 1
            end
            if not dup
                _push(g["conns"], [ni, no, _rand() * 4 - 2, true, _iv(ni, no)])
                return
            end
        end
        t = t + 1
    end
end

func _mutNode(g)
    var conns = g["conns"]
    var enabled = []
    var i = 0
    while i < _alen(conns)
        if conns[i][3]
            _push(enabled, i)
        end
        i = i + 1
    end
    if _alen(enabled) == 0
        return
    end
    var ci = enabled[_int(_rand() * _alen(enabled))]
    var c = conns[ci]
    _ncount = _ncount + 1
    var nid = _ncount
    _push(g["nodes"], [nid, 1])
    conns[ci][3] = false
    _push(conns, [c[0], nid, 1, true, _iv(c[0], nid)])
    _push(conns, [nid, c[1], c[2], true, _iv(nid, c[1])])
end

func _mutate(g)
    _mutW(g)
    if _rand() < P_CONN
        _mutConn(g)
    end
    if _rand() < P_NODE
        _mutNode(g)
    end
end

# ---- CROSSOVER ----
func _cross(p1, p2)
    var child = _genome()
    var nodeSet = _dict()
    var i = 0
    while i < _alen(p1["nodes"])
        _push(child["nodes"], [p1["nodes"][i][0], p1["nodes"][i][1]])
        nodeSet[_str(p1["nodes"][i][0])] = true
        i = i + 1
    end
    # inherit hidden nodes from p2; track which are p2-only so we can pull their connections
    var p2Hidden = _dict()
    i = 0
    while i < _alen(p2["nodes"])
        var pn = p2["nodes"][i]
        if pn[1] == 1
            p2Hidden[_str(pn[0])] = true
            if not _dict_has(nodeSet, _str(pn[0]))
                _push(child["nodes"], [pn[0], pn[1]])
                nodeSet[_str(pn[0])] = true
            end
        end
        i = i + 1
    end
    var c2m = _dict()
    i = 0
    while i < _alen(p2["conns"])
        c2m[_str(p2["conns"][i][4])] = i
        i = i + 1
    end
    var childInnov = _dict()
    i = 0
    while i < _alen(p1["conns"])
        var c = p1["conns"][i]
        var key = _str(c[4])
        childInnov[key] = true
        if _dict_has(c2m, key)
            var c2 = p2["conns"][c2m[key]]
            var pick = c
            if _rand() < 0.5
                pick = c2
            end
            var en = pick[3]
            if not c[3] or not c2[3]
                if _rand() < P_DISABLE
                    en = false
                end
            end
            _push(child["conns"], [pick[0], pick[1], pick[2], en, pick[4]])
        else
            _push(child["conns"], [c[0], c[1], c[2], c[3], c[4]])
        end
        i = i + 1
    end
    # critical: pull in p2 connections for inherited p2 hidden nodes
    # without this, hidden nodes arrive with zero connections and do nothing
    i = 0
    while i < _alen(p2["conns"])
        var c = p2["conns"][i]
        if not _dict_has(childInnov, _str(c[4]))
            if _dict_has(p2Hidden, _str(c[0])) or _dict_has(p2Hidden, _str(c[1]))
                if _dict_has(nodeSet, _str(c[0])) and _dict_has(nodeSet, _str(c[1]))
                    _push(child["conns"], [c[0], c[1], c[2], c[3], c[4]])
                    childInnov[_str(c[4])] = true
                end
            end
        end
        i = i + 1
    end
    return child
end

# ---- SPECIATION ----
func _compat(g1, g2)
    var c1 = g1["conns"]
    var c2 = g2["conns"]
    var c1m = _dict()
    var i = 0
    while i < _alen(c1)
        c1m[_str(c1[i][4])] = i
        i = i + 1
    end
    var c2m = _dict()
    i = 0
    while i < _alen(c2)
        c2m[_str(c2[i][4])] = i
        i = i + 1
    end
    var match = 0
    var wsum = 0
    var nonMatch = 0
    i = 0
    while i < _alen(c2)
        var k = _str(c2[i][4])
        if _dict_has(c1m, k)
            match = match + 1
            wsum = wsum + abs(c2[i][2] - c1[c1m[k]][2])
        else
            nonMatch = nonMatch + 1
        end
        i = i + 1
    end
    i = 0
    while i < _alen(c1)
        if not _dict_has(c2m, _str(c1[i][4]))
            nonMatch = nonMatch + 1
        end
        i = i + 1
    end
    var n = max(1, max(_alen(c1), _alen(c2)))
    var avgW = 0
    if match > 0
        avgW = wsum / match
    end
    return (C1 + C2) * nonMatch / n + C3 * avgW
end

func _speciate(pop, specs)
    var reps = []
    var ns = []
    var si = 0
    while si < _alen(specs)
        _push(reps, specs[si]["rep"])
        var sp = _dict()
        sp["rep"] = specs[si]["rep"]
        sp["members"] = []
        sp["best"] = specs[si]["best"]
        sp["stale"] = specs[si]["stale"]
        _push(ns, sp)
        si = si + 1
    end
    var gi = 0
    while gi < _alen(pop)
        var g = pop[gi]
        var placed = false
        si = 0
        while si < _alen(reps)
            if _compat(g, reps[si]) < COMPAT_T
                _push(ns[si]["members"], gi)
                placed = true
                break
            end
            si = si + 1
        end
        if not placed
            var sp = _dict()
            sp["rep"] = g
            sp["members"] = [gi]
            sp["best"] = 0
            sp["stale"] = 0
            _push(ns, sp)
            _push(reps, g)
        end
        gi = gi + 1
    end
    var pruned = []
    si = 0
    while si < _alen(ns)
        if _alen(ns[si]["members"]) > 0
            _push(pruned, ns[si])
        end
        si = si + 1
    end
    return pruned
end

func _adjFit(pop, specs)
    var si = 0
    while si < _alen(specs)
        var members = specs[si]["members"]
        var sz = _alen(members)
        var bestH = 0
        var mi = 0
        while mi < sz
            var g = pop[members[mi]]
            g["adjFit"] = g["fit"] / sz
            if g["fit"] > bestH
                bestH = g["fit"]
            end
            mi = mi + 1
        end
        if bestH > specs[si]["best"]
            specs[si]["best"] = bestH
            specs[si]["stale"] = 0
        else
            specs[si]["stale"] = specs[si]["stale"] + 1
        end
        si = si + 1
    end
end

func _nextGen(pop, specs)
    _adjFit(pop, specs)
    var totAdj = 0
    var si = 0
    while si < _alen(specs)
        var members = specs[si]["members"]
        var mi = 0
        while mi < _alen(members)
            totAdj = totAdj + pop[members[mi]]["adjFit"]
            mi = mi + 1
        end
        si = si + 1
    end
    var np = []
    si = 0
    while si < _alen(specs)
        var sp = specs[si]
        if sp["stale"] > STALE
            si = si + 1
            continue
        end
        var members = sp["members"]
        # sort members by fitness desc (insertion sort)
        var sorted = arrayCopy(members)
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
        # 20% elites per species, min 1, max 3
        var eliteCnt = max(1, min(3, _int(_alen(sorted) / 5)))
        var eli = 0
        while eli < eliteCnt
            _push(np, pop[sorted[eli]])
            eli = eli + 1
        end
        # allocation
        var specAdj = 0
        var mi = 0
        while mi < _alen(members)
            specAdj = specAdj + pop[members[mi]]["adjFit"]
            mi = mi + 1
        end
        var nOff = 0
        if totAdj > 0
            nOff = _int(specAdj / totAdj * POP)
        end
        # only top 1/3 of species breed — creates real selection pressure
        var cutoff = max(2, _int(_alen(sorted) / 3))
        var oi = 0
        while oi < nOff - eliteCnt and _alen(np) < POP
            var child = null
            if _alen(sorted) >= 2 and _rand() > 0.25
                var ai = sorted[_int(_rand() * cutoff)]
                var bi = sorted[_int(_rand() * cutoff)]
                if pop[ai]["fit"] >= pop[bi]["fit"]
                    child = _cross(pop[ai], pop[bi])
                else
                    child = _cross(pop[bi], pop[ai])
                end
            else
                child = _cross(pop[sorted[0]], pop[sorted[0]])
            end
            _mutate(child)
            _push(np, child)
            oi = oi + 1
        end
        si = si + 1
    end
    while _alen(np) < POP
        var g = _minGenome()
        _mutate(g)
        _push(np, g)
    end
    while _alen(np) > POP
        _pop(np)
    end
    return np
end

# ---- VISUALIZATION ----
func _render(s)
    var body = s["body"]
    var food = s["food"]
    var topBar = ""
    var i = 0
    while i < GRID_W + 2
        topBar = concat(topBar, "#")
        i = i + 1
    end
    print topBar
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
    print topBar
end

func _playGame(g)
    var s = _mkSnake()
    while s["alive"]
        var ins = _inputs(s)
        var outs = _activate(g, ins)
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
    print concat("Avg score  : ", concat(_str(totalScore / 10), "  (over 10 games)"))
    return totalScore / 10
end

print "================================================="
print "  NEAT Snake"
print "================================================="
print concat("  Grid   : ", concat(_str(GRID_W), concat("x", _str(GRID_H))))
print concat("  Pop    : ", _str(POP))
print concat("  Gens   : ", concat(_str(GENS), concat("  (", concat(_str(N_EVAL), " evals/genome)"))))
print concat("  Inputs : ", concat(_str(N_IN), concat("  Outputs: ", _str(N_OUT))))
print concat("  MaxSteps (no food): ", _str(MAX_NO_FOOD))
print "================================================="
print ""

var pop = []
var _gi = 0
while _gi < POP
    _push(pop, _minGenome())
    _gi = _gi + 1
end

var specs = []
var bestG = pop[0]
var bestFit = -1

var gen = 0
while gen < GENS
    # adaptive starvation: bad early snakes die fast → 3x speedup gen 1-20
    if gen < 20
        MAX_NO_FOOD = 80
    else if gen < 45
        MAX_NO_FOOD = 130
    else
        MAX_NO_FOOD = 200
    end

    # evaluate fitness (N_EVAL games each, averaged) — parallel across all cores
    _peval_pop(pop, _eval)

    # track best
    _gi = 0
    var genBest = pop[0]
    var genAvg = 0
    while _gi < _alen(pop)
        genAvg = genAvg + pop[_gi]["fit"]
        if pop[_gi]["fit"] > genBest["fit"]
            genBest = pop[_gi]
        end
        _gi = _gi + 1
    end
    genAvg = genAvg / _alen(pop)
    if genBest["fit"] > bestFit
        bestFit = genBest["fit"]
        bestG = genBest
    end

    specs = _speciate(pop, specs)
    var nConns = _alen(genBest["conns"])
    var nNodes = _alen(genBest["nodes"])

    print concat("Gen ", concat(_str(gen + 1), concat(" | Best:", concat(_str(_int(genBest["fit"])), concat(" Avg:", concat(_str(_int(genAvg)), concat(" Sp:", concat(_str(_alen(specs)), concat(" N:", concat(_str(nNodes), concat(" C:", _str(nConns))))))))))))

    pop = _nextGen(pop, specs)
    # always inject global best so it can never be mutated away
    _set(pop, 0, bestG)
    specs = _speciate(pop, specs)
    gen = gen + 1
end

MAX_NO_FOOD = 200
print ""
print "================================================="
print concat("  Training done: ", _str(_int(bestFit)))
print "  Best genome, final demo game:"
print "================================================="
print ""
_runDemo(bestG)
print ""
print concat("  Hidden nodes grown: ", _str(_alen(bestG["nodes"]) - N_IN - 1 - N_OUT))
print concat("  Connections: ", _str(_alen(bestG["conns"])))
