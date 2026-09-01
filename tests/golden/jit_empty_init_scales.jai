#: jit-cold-compile: yes
#:
#: An empty initializer must yield the object, with the arena full.
#:
#: `LinearScale.init` in `jaiplot` is empty, so its whole chunk is one
#: `OP_RETURN_NULL`, and the JIT had a fast path for exactly that shape that
#: returned null instead of the receiver. An initializer's `OP_RETURN_NULL` is
#: not a null return: the interpreter rewrites it to `frame->slots[0]`, which is
#: what makes `LinearScale()` an expression yielding an object.
#:
#: This is a golden rather than a unit test because it has to run the way
#: `jaithon run` runs. The fast path is a FALLBACK, reached only once the
#: whole-function tier declines, and for an empty initializer the only decline
#: it can reach is the shared code arena being full. That is a property of how
#: much the process has compiled: under `jaithon test` the same source leaves
#: too much headroom and the initializer compiles normally, so it never gets
#: there. The `(jit-first-call)` pass in scripts/run_tests.sh is what makes this
#: bite -- see `jaiJitThreshold` in src/vm/jit/jit.h and the named refusal
#: `the code arena is full` in src/vm/jit/jit_func.c.
#:
#: `Axes.init` builds two scales on consecutive lines. The call that COMPILES a
#: body does not use it, so `x_scale` came back an object and `y_scale` null,
#: and the program died in `_fallback` with `'null' object has no method
#: 'accepts'` -- hundreds of frames from the initializer that lost the value.
from jaiplot.figure import Figure

fn main() {
    let figure = Figure()
    let grid = figure.subplots(1, 2)
    for row in grid {
        for panel in row {
            print(panel.x_scale_of() != null and panel.y_scale_of() != null)
        }
    }
}
main()
