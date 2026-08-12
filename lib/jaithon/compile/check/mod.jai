#: The semantic type system: the Jaithon half of `src/sema`. It is the stage
#: between `resolve` and `emit`, and the reason the self-hosted front end
#: cannot yet match the C's bytecode: no `OP_TYPE_GUARD`, no `OP_TO_FLOAT`, no
#: constant folding. Nothing here is wired into the pipeline yet.
#:
#: Two properties are load-bearing. **Identity is interning**: two structurally
#: equal types must be the same object, so build types through `Universe`, not
#: `Type` directly. **The rendered string is a wire format**: the union sort
#: order — and therefore `kinds.ORDINAL` — decides whether a program diffs clean.

#: There is deliberately nothing else in this file: Jaithon does not re-export
#: a name a module merely imported, so importing each name here would only
#: raise `unused import`. Import the file that owns the name:
#:
#: ```text
#: from jaithon.compile.check.ty import Type
#: from jaithon.compile.check.universe import Universe, list_of
#: from jaithon.compile.check.assign import assignable
#: ```
