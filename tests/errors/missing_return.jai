# expect: E0601
fn pick(flag: bool) -> int {
    if flag { return 1 }
}
print(pick(false))
