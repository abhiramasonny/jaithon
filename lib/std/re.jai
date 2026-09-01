#: Regular expressions, simulated rather than backtracked.
#:
#: A pattern compiles to a Thompson NFA and runs on a Pike VM: the simulation
#: walks the input once, carrying every NFA state alive at the current
#: position. Cost is O(program size * text length), always — no input makes
#: it worse.
#:
#: That is the whole reason this engine exists. A backtracking engine — Perl,
#: Python, Java, JavaScript — tries one path at a time and unwinds, so
#: `(a+)+$` against thirty `a`s and a `!` explores 2^30 ways of splitting the
#: `a`s between the two quantifiers, and the process stops responding. The
#: same pattern here is a few thousand steps — the difference between a
#: service and an outage once the pattern meets a user or a log line instead
#: of a test case.
#:
#: The price: backreferences and lookaround are absent, since neither is a
#: regular language and neither can be simulated in one pass. Everything else
#: behaves as expected, including leftmost-first semantics — a greedy `.*`
#: still takes as much as it can and `.*?` as little.
#:
#: Supported syntax:
#:
#:   literals, `.`               any character except a newline
#:   `[a-z0-9_]`, `[^abc]`       classes, ranges, negation
#:   `\d \w \s \D \W \S`         digits, word characters, whitespace
#:   `\b \B`                     word boundary, and its negation
#:   `^ $`                       start and end of the whole text
#:   `(...)` `(?:...)`           capturing and non-capturing groups
#:   `a|b`                       alternation
#:   `* + ? {n} {n,} {n,m}`      quantifiers, each with a lazy `?` suffix
#:   `\n \t \r \f \v \0 \\ \.`   escapes; any punctuation may be escaped
#:
#: Compiling is the expensive half, so compile once and keep the `Regex`:
#:
#: ```
#: let word = Regex.compile(r"\w+")
#: word.is_match("hello world")          # true
#: word.find_all("hello world").len()    # 2
#: ```
#:
#: Facade: imports and re-exports every sibling module in `re/`.
from .re.matcher import Match, Regex

export { Match, Regex }
