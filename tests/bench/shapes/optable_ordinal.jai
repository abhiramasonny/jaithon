enum K { A, B, C, D, E, F, G, H }

enum Op { Lt, Le, Gt, Ge, Eq, Ne, In }

fn by_match(k: K) -> Op? {
    return match k {
        K.A => Op.Lt,
        K.B => Op.Le,
        K.C => Op.Gt,
        K.D => Op.Ge,
        K.E => Op.Eq,
        K.F => Op.Ne,
        K.G => Op.In,
        _ => null,
    }
}
let table: list[Op?] = [Op.Lt, Op.Le, Op.Gt, Op.Ge, Op.Eq, Op.Ne, Op.In, null]
let ks = [K.A, K.B, K.C, K.D, K.E, K.F, K.G, K.H]
var hits = 0
var i = 0
while i < 400000 {
    if 1 == 0 {
        if by_match(ks[i % 8]) != null { hits = hits + 1 }
    } else {
        if table[i % 8] != null { hits = hits + 1 }
    }
    i = i + 1
}
print(hits)
