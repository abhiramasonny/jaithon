#: DailyDialog as training rows for the chat model, and the chat format.
#:
#: A conversation is its turns, each opened by a role token, the whole
#: closed by `<|endoftext|>`:
#:
#:     <|user|>Say, Jim, how about going for a few beers?<|bot|>You know that is tempting...<|user|>...<|endoftext|>
#:
#: Fine-tuning scores only what the bot says: the tokens of its turns, and
#: the token after each -- the next `<|user|>` or the `<|endoftext|>` -- so it
#: learns where its turn stops. The user's words and the `<|bot|>` that hands
#: over are context, never targets.
import jaitensor as jt
from jaitensor import Tensor
from std.random import Random
from .tokenizer import BOT, EOT, USER, Tokenizer

#: One formatted conversation: ids, which of them are targets, and where
#: each turn ends.
pub class Formatted {
    pub let ids: list[int]
    pub let scored: list[bool]
    pub let turn_ends: list[int]

    pub fn init(self, ids: list[int], scored: list[bool], turn_ends: list[int]) {
        self.ids = ids
        self.scored = scored
        self.turn_ends = turn_ends
    }
}

#: `turns` with the first speaker as the user, the second as the bot, and so
#: on alternately.
pub fn format_turns(tok: Tokenizer, turns: list[str]) -> Formatted {
    var ids: list[int] = []
    var scored: list[bool] = []
    var ends: list[int] = []
    var bot_spoke = false
    for (i, turn) in turns.enumerate() {
        let is_bot = i % 2 == 1
        ids.push(if is_bot { BOT } else { USER })
        scored.push(bot_spoke)
        for t in tok.encode(turn) {
            ids.push(t)
            scored.push(is_bot)
        }
        ends.push(ids.len())
        bot_spoke = is_bot
    }
    ids.push(EOT)
    scored.push(bot_spoke)
    return Formatted(ids, scored, ends)
}

#: A training row: `ids` of at most `seq + 1` tokens and the targets in it.
pub class Row {
    pub let ids: list[int]
    pub let scored: list[bool]

    pub fn init(self, ids: list[int], scored: list[bool]) {
        self.ids = ids
        self.scored = scored
    }

    pub fn targets(self) -> int {
        var n = 0
        for i in 1..self.scored.len() { if self.scored[i] { n += 1 } }
        return n
    }
}

#: The rows of one dialogue: each speaker takes a turn as the bot, and a
#: dialogue longer than the context is cut into windows that each start at a
#: user turn and overlap the last by a couple of turns.
pub fn dialogue_rows(tok: Tokenizer, turns: list[str], seq: int) -> list[Row] {
    var out: list[Row] = []
    for shift in 0..2 {
        if turns.len() - shift < 2 { continue }
        var k = shift
        while k < turns.len() - 1 {
            let f = format_turns(tok, turns[k:])
            if f.ids.len() <= seq + 1 {
                out.push(Row(f.ids, f.scored))
                break
            }
            out.push(Row(f.ids[0:seq + 1], f.scored[0:seq + 1]))
            var covered = 0
            for e in f.turn_ends { if e <= seq + 1 { covered += 1 } }
            var step = covered - 2
            step -= step % 2
            k += max(2, step)
        }
    }
    return out
}

#: Every row of a DailyDialog split.
pub fn split_rows(tok: Tokenizer, dialogues: list[list[str]], seq: int) -> list[Row] {
    var rows: list[Row] = []
    for turns in dialogues {
        for r in dialogue_rows(tok, turns, seq) {
            if r.targets() > 0 { rows.push(r) }
        }
    }
    return rows
}

#: The padded length a row trains at: a multiple of `quantum`, so a batch of
#: short dialogues is not padded out to the whole context.
pub fn bucket_length(row: Row, seq: int, quantum: int) -> int {
    let need = row.ids.len() - 1
    let up = (need + quantum - 1) // quantum * quantum
    return min(seq, max(quantum, up))
}

#: A batch of rows padded to `length + 1` tokens: inputs `[length, batch]`,
#: sequence-first, the flat logits rows the loss is over, and their targets.
pub class Batch {
    pub let x: Tensor
    pub let rows: Tensor
    pub let y: Tensor
    pub let targets: int

    pub fn init(self, picked: list[Row], length: int) {
        let batch = picked.len()
        var xs: list[float] = []
        var where: list[float] = []
        var ys: list[float] = []
        for t in 0..length {
            for (b, r) in picked.enumerate() {
                xs.push(float(if t < r.ids.len() { r.ids[t] } else { EOT }))
                if t + 1 < r.ids.len() and r.scored[t + 1] {
                    where.push(float(t * batch + b))
                    ys.push(float(r.ids[t + 1]))
                }
            }
        }
        self.x = Tensor([length, batch], xs)
        self.rows = Tensor([where.len()], where)
        self.y = Tensor([ys.len()], ys)
        self.targets = ys.len()
    }
}

#: The batches of one epoch: rows grouped by padded length, shuffled within
#: each length, cut into batches, and the batches shuffled.
pub fn epoch_batches(
    rows: list[Row],
    batch: int,
    seq: int,
    quantum: int,
    rng: Random?
) -> list[(int, list[Row])] {
    var buckets: dict[int, list[Row]] = {}
    for r in rows {
        let length = bucket_length(r, seq, quantum)
        if not buckets.has(length) { buckets[length] = [] }
        buckets[length].push(r)
    }
    var out: list[(int, list[Row])] = []
    for length in buckets.keys().sorted() {
        var members = buckets[length]
        if rng is not null { rng.shuffle(members) }
        var i = 0
        while i < members.len() {
            out.push((length, members[i:min(i + batch, members.len())]))
            i += batch
        }
    }
    if rng is not null { rng.shuffle(out) }
    return out
}

#: The token ids of a conversation so far, ready for the bot to answer:
#: every turn, then `<|bot|>`. `turns` alternate user and bot, the user's
#: first and last. The oldest exchanges are dropped, a user turn and its
#: reply at a time, until it fits `budget` tokens.
pub fn chat_prompt(tok: Tokenizer, turns: list[str], budget: int) -> list[int] {
    var first = 0
    while true {
        var ids: list[int] = []
        for (i, turn) in turns[first:].enumerate() {
            let is_bot = (first + i) % 2 == 1
            ids.push(if is_bot { BOT } else { USER })
            for t in tok.encode(turn) { ids.push(t) }
        }
        ids.push(BOT)
        if ids.len() <= budget or first + 2 >= turns.len() {
            if ids.len() > budget { return ids[ids.len() - budget:] }
            return ids
        }
        first += 2
    }
    return []
}
