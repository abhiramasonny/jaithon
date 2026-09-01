# Data-representation probes

Each pair (or triple) computes the same answer two ways and differs only in
how the data is laid out. Run with `./jaithon run --stats <file>` and compare
the `vm:` line. These are the measurements behind ROADMAP.md's "3.4, measured"
section: the compiler's hot paths are written in the slow shape of each pair.

    optable_match / optable_dict / optable_ordinal   enum -> value lookup
    node_dict / node_slots / node_class              AST field storage
    token_three_objects / token_flat                 one token, how many objects

The lexer pair (`list[str]` of characters versus `str.bytes()`) is not here
because it needs a large input; ROADMAP.md says how it was generated.
