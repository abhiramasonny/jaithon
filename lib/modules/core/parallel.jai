# General parallel primitives.
# The C core exposes two general threading builtins (not eval-specific):
#   _pmap(array, fn) -> new array of fn(elem) for each element, across all cores
#   _pfor(n, fn)     -> runs fn(i) for i in [0, n) across all cores (side effects)
# Everything higher-level is bootstrapped here in Jaithon.

func pmap(arr, fn)
    return _pmap(arr, fn)
end

func pfor(n, fn)
    _pfor(n, fn)
    return null
end

# Population evaluator, bootstrapped: a parallel map whose result is ignored
# because fn mutates each genome in place (e.g. genome["fit"] = score).
# Previously a dedicated C builtin; now just _pmap.
func peval_pop(pop, fn)
    _pmap(pop, fn)
    return null
end
