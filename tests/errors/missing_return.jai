# expect: E0601
# A non-void function must not fall off the end.

fn pick(flag: bool) -> int {
    if flag { return 1 }
}
print(pick(false))
