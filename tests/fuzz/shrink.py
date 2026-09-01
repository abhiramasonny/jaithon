#!/usr/bin/env python3
"""Reduces a failing generated program to a minimal one that still disagrees.

A fuzz hit is worthless until someone can act on it, and a 200-line random
program with four probe functions is not something anyone can act on. This
turns one into the few lines that actually matter.

The method is the ordinary delta debugging loop -- propose a smaller program,
keep it if it still fails, otherwise put it back -- and every bit of the
cleverness is in making "smaller" always mean "still legal". Two properties of
the generator do that, and neither is an accident:

  - a whole Node is deleted, never part of one, so a loop's counter increment
    (a literal line of the loop's own Node, not a child) cannot be separated
    from the loop and no reduction can produce a program that fails to
    terminate;
  - a reduction that IS illegal -- dropping the class a later statement
    constructs, say -- fails to compile identically under every configuration,
    which reads as "agrees", so it is rejected on its own. Validity needs no
    separate check.

The one judgement call is what "still fails" means. Requiring the exact same
set of disagreeing configurations would stall the reduction, since a hit found
under both `tick` and `deopt` legitimately narrows to just `deopt` as the
program shrinks. Requiring merely "something still disagrees" is too loose --
it can walk off onto a different bug entirely. So the test is that the new
disagreement INTERSECTS the original one: at least one configuration that was
wrong before is still wrong.

Passes run to a fixpoint, cheapest and highest-yield first: whole probe
functions, then statements from the end backwards (deleting late leaves earlier
indices valid), then helpers, then classes, then the warm-up count -- lowered
last, and only as far as the failure survives, since dropping under the tier's
64-call threshold stops it compiling at all and would "fix" every bug.

Not run directly; differential.py --shrink SEED drives it.
"""

import sys


def walk(nodes):
    """Every deletable Node in pre-order, parents before children.

    A plain string in a block is a line the generator marked as not removable
    -- a loop's counter increment, the break that consumes it -- so it is never
    offered as a candidate. See Node in progen.py for why that matters.
    """
    for node in nodes:
        if isinstance(node, str):
            continue
        yield node
        for part in node.parts:
            if isinstance(part, list):
                yield from walk(part)


def remove(nodes, target):
    """Delete `target` wherever it sits in the tree. Identity, not equality."""
    for i, node in enumerate(nodes):
        if isinstance(node, str):
            continue
        if node is target:
            del nodes[i]
            return True
        for part in node.parts:
            if isinstance(part, list) and remove(part, target):
                return True
    return False


def confirm(prog, oracle, original, trials):
    """How many of `trials` runs of an unchanged program still disagree.

    The whole point of running this on the FINAL program: the OSR tier is
    driven by a SIGPROF sampler, so whether a loop compiles at all depends on
    where the ticks land, and a reduction accepted on a single lucky check can
    be a program that fails one run in ten. A shrinker that reports such a
    thing as "the minimal reproducer" is worse than no shrinker -- it sends
    someone to debug a program that mostly works. Report the rate instead.
    """
    source = prog.render()
    return sum(1 for _ in range(trials)
               if set(original) & set(oracle(source)))


def reduce(prog, oracle, original, repeat=2):
    """Shrink `prog` while `oracle` keeps naming a mode from `original`.

    `oracle` takes source text and returns the tuple of disagreeing mode names.
    A candidate is adopted only after `repeat` consecutive checks agree it
    still fails: one check is not enough, and accepting on one is how a
    reduction walks off a solid 15-of-15 segfault onto a program that never
    crashes again. Returns (smallest program, how many oracle runs were spent).
    """
    original = set(original)
    tried = [0]

    def still_fails(candidate):
        source = candidate.render()
        for _ in range(repeat):
            tried[0] += 1
            if not (original & set(oracle(source))):
                return False
        return True

    def attempt(build):
        """Apply `build` to a clone; adopt it only if the failure survives."""
        candidate = prog.clone()
        if not build(candidate):
            return None
        if still_fails(candidate):
            return candidate
        return None

    changed = True
    while changed:
        changed = False

        # 1. Whole probe functions, last first.
        for idx in range(len(prog.probes) - 1, -1, -1):
            if len(prog.probes) <= 1:
                break
            got = attempt(lambda c, i=idx: (c.probes.pop(i), True)[1])
            if got is not None:
                prog, changed = got, True

        # 2. Statements, last first, so earlier positions stay valid.
        for probe_idx in range(len(prog.probes)):
            while True:
                count = len(list(walk(prog.probes[probe_idx].body)))
                for k in range(count - 1, -1, -1):
                    def build(c, pi=probe_idx, kk=k):
                        flat = list(walk(c.probes[pi].body))
                        if kk >= len(flat):
                            return False
                        return remove(c.probes[pi].body, flat[kk])
                    got = attempt(build)
                    if got is not None:
                        prog, changed = got, True
                        break
                else:
                    break

        # 3. Helpers, then classes. A statement still naming one of them makes
        #    the candidate fail to compile everywhere, which reads as agreement.
        for idx in range(len(prog.helpers) - 1, -1, -1):
            got = attempt(lambda c, i=idx: (c.helpers.pop(i), True)[1])
            if got is not None:
                prog, changed = got, True
        for idx in range(len(prog.classes) - 1, -1, -1):
            got = attempt(lambda c, i=idx: (c.classes.pop(i), True)[1])
            if got is not None:
                prog, changed = got, True

    # 4. The warm-up count, last. Below the tier's 64-call threshold nothing
    #    compiles and every bug "goes away", so this is a real reduction only
    #    while the failure survives it.
    for warm in (1000, 500, 300, 200, 150, 120, 100, 90, 80, 70):
        if warm >= prog.warm:
            continue
        got = attempt(lambda c, w=warm: (setattr(c, "warm", w), True)[1])
        if got is not None:
            prog = got

    return prog, tried[0]


if __name__ == "__main__":
    print(__doc__.strip().splitlines()[-1], file=sys.stderr)
    sys.exit(2)
