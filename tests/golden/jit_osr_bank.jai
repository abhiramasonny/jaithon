#: Which BANK a hot loop's locals land in, and whether every home reads back
#: what was written to it.
#:
#: jit_osr_locals covers the three homes -- an X register, one of d8..d15, the
#: frame -- and that a loop naming more locals than there are registers keeps
#: working. This file is about the two questions that decide WHICH home a slot
#: gets, both of which changed when the OSR tier stopped ranking its slots by
#: how often the body names them and started ranking them, as the function tier
#: already did, by the instructions a register would actually save.
#:
#: The first is that a write is not worth the same as a read here. An OSR slot's
#: memory home is the interpreter's own Value, so a write to it stores the tag
#: as well as the payload and costs three instructions where a read costs one. A
#: use count cannot tell the two apart, so a variable assigned every iteration
#: and a variable read every iteration made the same claim on the same register,
#: and the assigned one -- which is to say the loop variable -- lost ties. The
#: floats and `ia`/`ib`/`ic` below are written every iteration; `id` is read
#: every iteration and written on one in a hundred and one. They rank in the
#: opposite order under the two rules, so the plan here is not the plan this
#: file would have got before, and the answer has to be the same either way.
#:
#: The second is that a use count is one number answering for two register
#: banks, which are not worth the same to the same slot: a float read is an
#: `ldr d` from the frame and an `fmov` from an X register, so an X home saves
#: it nothing, while an integer read is an `ldr x` against no instruction at
#: all. Five floats and four integers compete here, over eight of d8..d15 and
#: whatever is left of the ten callee-saved X registers -- the plan this gets is
#: nine X locals and six FP locals, so both banks are genuinely full.
#:
#: What must NOT change is that only a float may take one of d8..d15 under OSR.
#: The bank is not chosen by kind in the function tier -- jitArgIn marshals
#: every incoming argument, so an FP home there only ever sees a clean value --
#: but an OSR home is a slot the interpreter itself wrote, and the OSR prologue
#: fills an FP home with a full-width `ldr d`. A bool is a one-byte `strb` in
#: that slot, so the seven bytes above it are whatever the slot held before, and
#: an eight-byte load puts them in the register; every consumer of a bool tests
#: the whole word, so `if flag {` then takes the wrong arm, and the program runs
#: on and prints a different number rather than crashing. The X arm of that
#: prologue narrows to a byte for exactly this reason and the FP arm has nowhere
#: to narrow, which is why the ranking may reorder the slots but may not be let
#: near the choice of bank. `lo`, `hi` and `mid` are bools a ranking would want
#: to place: read on every iteration, written on only some, so they are live
#: across the back edge and the first read of each is of the value the prologue
#: loaded rather than one this loop has written.
#:
#: The printed number is built to show all of that. `acc` only ever takes whole
#: numbers and quarters, so it is exact in a double however long the loop runs,
#: and one bool read wrongly moves it by at least 0.25 -- visible at six places
#: rather than lost in a rounding. The five floats are contractions, so they
#: stay under one and land in the fraction, where a wrong FP home shows up
#: instead of being swamped.
#:
#: Under `--gc-stress` the list the caller builds collects with floats live in
#: d8..d15; under `JAITHON_JIT_DEOPT_STRESS=1` every guard fails, so each of the
#: three homes is written back into the record the interpreter resumes from at
#: every guard rather than never.
fn tally(n: int) -> float {
    var fa = 1.0
    var fb = 2.0
    var fc = 3.0
    var fd = 4.0
    var fe = 5.0

    var lo = false
    var hi = true
    var mid = false

    var ia = 0
    var ib = 1
    var ic = 2
    var id = 3

    var wsl = 0
    var rdo = 5

    var acc = 0.0
    var k = 0
    while k < n {
        # Written every iteration and each fed by the last, so all five want the
        # FP bank at once. Every update is a contraction, so they stay under one
        # however long the loop runs.
        fa = fa * 0.5 + float(ia % 8) * 0.125
        fb = fb * 0.5 + fa * 0.25
        fc = fc * 0.5 + fb * 0.25
        fd = fd * 0.5 + fc * 0.25
        fe = fe * 0.5 + fd * 0.25

        # Written every iteration and read by each other, so they want X.
        ia += 1
        ib = (ib + ia % 3) % 9973
        ic = (ic + ib % 5) % 9973

        # Read every iteration, written on one in a hundred and one: the slot
        # whose rank the two rules disagree about most.
        if ia % 101 == 0 { id = (id + 1) % 97 }
        acc += float(id)

        # The pair the two rules order OPPOSITELY, which is what makes this file
        # test the ranking and not just the homes. `wsl` is written three times
        # and read none: three sites to a use count, six instructions saved to
        # the ledger, because an OSR write to the frame stores a tag as well as
        # a payload. `rdo` is read three times and written once: four sites,
        # five instructions. So a use count puts `rdo` ahead of `wsl` and the
        # ledger puts `wsl` ahead of `rdo`.
        #
        # Eleven slots then want the nine X registers this loop can hand out --
        # `n`, these two, the three bools, `ia`/`ib`/`ic`/`id` and `k` -- so two
        # are left in the frame, and reordering the pair above moves which two.
        # It is not `wsl` and `rdo` themselves that end up displaced: the plan
        # this file gets keeps the loop counter `k` in a register and leaves the
        # bool `mid` in the frame, and the plan it got before did the opposite,
        # which is the whole point -- the assigned variable used to lose. The
        # printed number has to be the same under either.
        wsl = ia % 5
        wsl = ic % 7
        wsl = ib % 3
        if rdo > 0 { acc += 0.25 }
        if ia % 2 == 0 { acc += float(rdo) }
        acc += float(rdo % 3)
        if ia % 997 == 0 { rdo = (rdo + 1) % 11 }

        # Live across the back edge, so the first read of each is of the value
        # the OSR prologue loaded. Quarters, so acc stays exact.
        if lo { acc += 0.25 }
        if hi { acc += 0.5 }
        if mid { acc += 0.75 }
        if ia % 7 == 0 { lo = not lo }
        if ia % 11 == 0 { hi = not hi }
        if ia % 13 == 0 { mid = not mid }

        k += 1
    }
    return acc + fa + fb + fc + fd + fe + float(ia + ib + ic + id + wsl + rdo)
}

#: A second and a third loop in the same function, over slots the compiler is
#: free to reuse for one another: a bool landing where an integer has been is
#: how the seven bytes above a `strb` come to be dirty in the first place. The
#: list is here so a collection can happen while the loop above is holding
#: floats in callee-saved FP registers.
fn recount(rounds: int, n: int) -> float {
    var total = 0.0
    var keep: list[float] = []
    var r = 0
    while r < rounds {
        var big = 0
        var j = 0
        while j < 64 {
            big = (big * 3 + j) % 1000003
            j += 1
        }
        var seen = big % 2 == 0
        var bump = 0.0
        var m = 0
        while m < 64 {
            if seen { bump += 1.5 } else { bump += 0.5 }
            if m % 5 == 0 { seen = not seen }
            m += 1
        }
        keep.push(bump)
        total += tally(n) + bump + float(big % 97)
        r += 1
    }
    var s = 0.0
    for v in keep { s += v }
    return total + s
}

#: Long enough that the tier actually reaches the loop: OSR is entered from a
#: timer tick that lands on a back edge, so a loop finishing in a millisecond is
#: compiled on some runs and not others.
fn main() -> int {
    print(f"{recount(8, 40000):.6f}")
    return 0
}
