#: Sampling and cached decoding, shared by `generate.jai`, `chat.jai` and the
#: samples `pretrain.jai` prints.
#:
#: The logits of each step come down to the host and the pick is made here:
#: temperature, then a repetition penalty over the recent tokens, then the
#: `top_k` likeliest, then the smallest set of those whose probability adds
#: up to `top_p`. A vocabulary of 4096 is 16 KB a step to download, which is
#: nothing next to the forward that made it.
import std.math
from std.random import Random
from .model import GPT

pub class Sampling {
    pub var temperature: float
    pub var top_k: int
    pub var top_p: float
    pub var repetition_penalty: float
    #: How many of the latest tokens the penalty looks at.
    pub var window: int

    pub fn init(
        self,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        window: int = 64
    ) {
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.window = window
    }
}

fn argmax(row: list[float]) -> int {
    var best = 0
    for i in 1..row.len() {
        if row[i] > row[best] { best = i }
    }
    return best
}

#: Pick the next token from one row of logits. Temperature 0 is greedy, and
#: then the penalty is the only thing applied.
pub fn pick(row: list[float], s: Sampling, recent: list[int], rng: Random) -> int {
    for value in row {
        if value != value { throw RuntimeError("the model produced NaN logits") }
    }
    var logits = row
    if s.repetition_penalty != 1.0 and recent.len() > 0 {
        logits = row.copy()
        let first = max(0, recent.len() - s.window)
        for t in recent[first:] {
            let v = logits[t]
            logits[t] = if v > 0.0 { v / s.repetition_penalty } else { v * s.repetition_penalty }
        }
    }
    if s.temperature <= 0.0 { return argmax(logits) }
    let k = if s.top_k > 0 and s.top_k < logits.len() { s.top_k } else { logits.len() }
    let ordered = logits.sorted()
    let floor = ordered[logits.len() - k]
    var ids: list[int] = []
    var vals: list[float] = []
    for (i, v) in logits.enumerate() {
        if v >= floor and ids.len() < k {
            ids.push(i)
            vals.push(v)
        }
    }
    #: Largest first, by insertion: k is small.
    for i in 1..ids.len() {
        var j = i
        while j > 0 and vals[j] > vals[j - 1] {
            let tv = vals[j]
            vals[j] = vals[j - 1]
            vals[j - 1] = tv
            let ti = ids[j]
            ids[j] = ids[j - 1]
            ids[j - 1] = ti
            j -= 1
        }
    }
    var weights: list[float] = []
    var total = 0.0
    for v in vals {
        let w = math.exp((v - vals[0]) / s.temperature)
        weights.push(w)
        total += w
    }
    #: Keep the head of the list whose mass reaches top_p.
    var keep = weights.len()
    if s.top_p < 1.0 {
        var mass = 0.0
        for (i, w) in weights.enumerate() {
            mass += w / total
            if mass >= s.top_p {
                keep = i + 1
                break
            }
        }
    }
    var kept_total = 0.0
    for i in 0..keep { kept_total += weights[i] }
    var target = rng.next_float() * kept_total
    for i in 0..keep {
        target -= weights[i]
        if target <= 0.0 { return ids[i] }
    }
    return ids[keep - 1]
}

#: Generates tokens through the attention cache, one sequence at a time. The
#: cache holds the model's whole context; when it is full, the most recent
#: half is fed through again and decoding goes on.
pub class Decoder {
    pub let model: GPT
    #: Every token the cache holds, in order.
    pub var context: list[int]
    var at: int

    pub fn init(self, model: GPT) {
        self.model = model
        self.context = []
        self.at = 0
        model.start_cache(model.cfg.seq)
    }

    pub fn reset(self) -> void {
        self.model.reset_cache()
        self.context = []
        self.at = 0
    }

    #: Feed `ids` and return the logits that follow them, downloaded.
    pub fn feed(self, ids: list[int]) -> list[float] {
        let limit = self.model.cfg.seq
        var pending = ids
        if self.at + pending.len() > limit {
            #: Start again from the recent half of everything so far.
            var all = self.context.copy()
            for t in ids { all.push(t) }
            let keep = min(all.len(), limit // 2)
            self.model.reset_cache()
            self.context = []
            self.at = 0
            pending = all[all.len() - keep:]
        }
        let logits = self.model.forward_cached(pending, self.at)
        self.at += pending.len()
        for t in pending { self.context.push(t) }
        let row = logits.download()
        logits.free()
        return row
    }

    pub fn free(self) -> void { self.model.free_cache() }
}

#: `count` tokens after `prompt`, stopping early at any of `stops` (which is
#: not included). `each` sees every token as it is made.
pub fn generate(
    decoder: Decoder,
    prompt: list[int],
    count: int,
    s: Sampling,
    rng: Random,
    stops: list[int],
    each: fn(int) -> void
) -> list[int] {
    var out: list[int] = []
    var recent = prompt.copy()
    var row = decoder.feed(prompt)
    for _n in 0..count {
        let next = pick(row, s, recent, rng)
        if stops.contains(next) { break }
        out.push(next)
        recent.push(next)
        each(next)
        row = decoder.feed([next])
    }
    return out
}
