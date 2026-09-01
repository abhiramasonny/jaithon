#: An estimator's parameter dictionary, and the argument checks every fit repeats.
from jaitensor import Tensor

#: A parameter dictionary, as `get_params` returns it and `set_params` takes
#: it. Read values back through `param_int` and friends, which accept an
#: integer where a float was meant rather than making every caller remember
#: which one they wrote.
pub type Params = dict[str, any]

#: Read an integer parameter, accepting the float a caller may have written.
pub fn param_int(params: Params, name: str, fallback: int) -> int {
    let value = params.get(name, null)
    if value is int { return value }
    if value is float { return int(value) }
    return fallback
}

#: Read a float parameter, accepting the integer a caller may have written --
#: `{"gamma": 1}` is the mistake everybody makes once.
pub fn param_float(params: Params, name: str, fallback: float) -> float {
    let value = params.get(name, null)
    if value is float { return value }
    if value is int { return float(value) }
    return fallback
}

#: Read a boolean parameter.
pub fn param_bool(params: Params, name: str, fallback: bool) -> bool {
    let value = params.get(name, null)
    if value is bool { return value }
    return fallback
}

#: Read a string parameter.
pub fn param_str(params: Params, name: str, fallback: str) -> str {
    let value = params.get(name, null)
    if value is str { return value }
    return fallback
}

#: A parameter dictionary from pairs, which reads better inside a search grid
#: than a dictionary literal in a list does.
pub fn params_of(pairs: list[tuple[str, any]]) -> Params {
    var out: Params = {}
    for (name, value) in pairs { out[name] = value }
    return out
}

#: Insist that `x` is a `[n_samples, n_features]` matrix with something in it.
pub fn check_matrix(x: Tensor, name: str = "X") -> void {
    if x.rank() != 2 { throw ValueError(f"{name} must be [n_samples, n_features], got {x.shape}") }
    if x.shape[0] == 0 or x.shape[1] == 0 { throw ValueError(f"{name} is empty, shape {x.shape}") }
}

#: Insist that `x` and `y` describe the same samples.
pub fn check_pair(x: Tensor, y: Tensor, name: str = "X") -> void {
    check_matrix(x, name)
    if y.rank() != 1 { throw ValueError(f"y must be [n_samples], got {y.shape}") }
    if y.shape[0] != x.shape[0] {
        throw ValueError(f"y has {y.shape[0]} values for {x.shape[0]} samples")
    }
}

#: Unwrap the target of a supervised `fit`, naming the estimator that needs it.
pub fn require_target(y: Tensor?, what: str) -> Tensor {
    let target = y
    if target is null { throw ValueError(f"{what} needs a target vector") }
    return target
}

#: Insist that `fit` has run before whatever is about to read its results.
pub fn check_fitted(fitted: bool, what: str) -> void {
    if not fitted { throw RuntimeError(f"{what} is not fitted; call fit first") }
}

#: Insist that `x` is as wide as `fit` saw. An `expected` of 0 means the
#: estimator records no width, and the call goes through.
pub fn check_width(x: Tensor, expected: int, what: str) -> void {
    check_matrix(x)
    if expected == 0 { return }
    if x.shape[1] != expected {
        throw ValueError(f"{what} was fitted on {expected} features, got {x.shape[1]}")
    }
}
