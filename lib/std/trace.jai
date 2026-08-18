#: Compile-time graph tracing hooks used by `@trace` functions.
#:
#: `@trace` marks a function for a lower JIT compile threshold and records
#: tensor-op metadata while the function runs. ML libraries such as jaitensor
#: call `record_op` from hot kernels so a second invocation with matching
#: shapes sets `is_replay()`. The JIT compiles traced functions after 8 calls.
pub const FN_TRACE: int = 512

#: Whether a `@trace` function is executing.
pub fn is_active() -> bool { return __prim__.trace_active() }

#: Whether the current invocation is replaying a cached graph.
pub fn is_replay() -> bool { return __prim__.trace_replay() }

#: How many times the active traced function has run with a cached graph.
pub fn run_id() -> int { return __prim__.trace_run_id() }

#: Record one op and its shape while tracing is active.
pub fn record_op(name: str, shape: list[int]) -> void {
    __prim__.trace_record_op(name, shape)
}

#: Whether `target` was compiled with `@trace`.
pub fn is_traced(target: any) -> bool { return __prim__.function_has_flag(target, FN_TRACE) }
