# Jaithon for VS Code

Language support for [Jaithon](https://github.com/abhiramasonny/jaithon), driven
entirely by the real compiler. There is no reimplemented parser here and no
language server: `jaithon` already emits positioned diagnostics, a JSON syntax
tree, the type it inferred for every node, and a canonical formatter, so the
extension's job is to run it and index the answers.

That is why the editor never disagrees with `jaithon check`.

## What it does

**Problems, as you type.** Errors and warnings appear on a 300 ms debounce
without saving, with the compiler's own span, `help:` and `note:` text, and
secondary spans — in other files too — as related information. An unused binding
greys out rather than shouting.

**Navigation.** Go to definition, type definition, implementations, find all
references, call hierarchy, document and workspace symbols. Definitions follow
`import` into other modules, resolved the way `src/runtime/module.c` resolves
them, so cmd-clicking `std.math` opens the file the compiler would have loaded.

**Rename.** Across files, driven by the same resolution as go-to-definition, so
a local stays local and a `pub fn` reaches its importers. It refuses names that
are not identifiers, keywords, and anything the runtime owns.

**Hover.** The declaration's signature, its `#:` doc comment, and the type the
checker gave it. Hovering the middle of an expression shows that
subexpression's type, because the checker recorded one for every node.

**Completion.** Scope-aware: locals shadow globals, `self.` reaches fields,
`xs.` offers the methods the runtime gives a `list`, and a module name offers
what that module actually exports. `import ` offers module paths found on the
search path.

**Signature help, inlay hints, code lenses.** Inferred types after a `let`,
parameter names in front of literal arguments, Run above `main`, and a test lens
above every `test_` function.

**Quick fixes.** `did you mean` becomes a one-click fix, an unused binding gets
its `_`, an uninitialised `let` becomes a `var`, an undefined name offers the
import that would supply it, and a class that names a trait can have the methods
it still owes generated.

**Formatting.** `jaithon fmt` is canonical and takes no options. The extension
runs it on a scratch copy and applies only the lines that changed, so your
cursor, folds and decorations survive.

**Semantic highlighting.** The TextMate grammar colours by shape; semantic
tokens then recolour by what the compiler decided each name actually is, so a
class and a variable spelled alike stop looking alike.

**`.jaic` viewer.** Opening a bytecode image renders `jaithon disasm` instead of
binary garbage.

## Installing

The extension has no dependencies and no build step.

```bash
ln -s "$PWD/editors/vscode" ~/.vscode/extensions/jaithon
```

Then reload VS Code. For the desktop app's other flavours, use
`~/.vscode-insiders/extensions` or `~/.vscode-server/extensions`.

## Settings

| Setting | Default | |
|---|---|---|
| `jaithon.path` | `jaithon` | Path to the compiler. `${workspaceFolder}` and `~` expand. A `jaithon` binary in the workspace root beats one on `PATH`, so working on the compiler tests the compiler you just built. |
| `jaithon.checkOnType` | `true` | Report problems without saving. |
| `jaithon.checkOnSave` | `true` | Report problems on save. |
| `jaithon.checkDelay` | `300` | Milliseconds of quiet typing before the checker runs. |
| `jaithon.strict` | `false` | Pass `--strict`: an unannotated parameter becomes an error. |
| `jaithon.includePaths` | `[]` | Extra module search directories, passed as `-I`. |
| `jaithon.excludeGlob` | build and cache directories | What workspace-wide search skips. |
| `jaithon.inlayHints.types` | `true` | Inferred types after `let`, `var` and unannotated parameters. |
| `jaithon.inlayHints.parameterNames` | `true` | Parameter names in front of literal arguments. |
| `jaithon.codeLens.run` | `true` | Run and test lenses. |
| `jaithon.codeLens.references` | `false` | Reference counts. Off by default: each one searches the workspace. |

Format-on-save is VS Code's own `editor.formatOnSave`, scoped to the language:

```json
"[jaithon]": { "editor.formatOnSave": true }
```

## How it is put together

| File | |
|---|---|
| `extension.js` | Activation, commands, tasks, status bar |
| `src/tool.js` | Running `jaithon`; binary resolution; the module search path; scratch copies of unsaved buffers |
| `src/diagnostics.js` | Parsing the compiler's rustc-style diagnostic blocks |
| `src/analysis.js` | The language service: `ast --json` plus `--dump-sema`, indexed into scopes, symbols and references |
| `src/builtins.js` | The names the runtime registers in C, which no `.jai` file declares |
| `src/navigation.js` | Definition, references, rename, outlines, folding, call hierarchy |
| `src/completion.js` | Completion and signature help |
| `src/hints.js` | Hover, inlay hints, code lenses, semantic tokens |
| `src/actions.js` | Formatting and code actions |
| `src/jaic.js` | The bytecode viewer |

Two facts do most of the work. `jaithon ast --json` puts a byte span on every
node, and `jaithon check --dump-sema` reports the checker's type for each of
those spans — so `p.norm` is answered by the type checker rather than guessed.
Only lexical scoping is reimplemented here, because neither dump records which
declaration a bare name resolved to.

Spans are byte offsets and VS Code positions are UTF-16; `src/analysis.js`
builds a conversion table only for files that are not pure ASCII.

## When the file does not parse

Completion fires on `xs.` and signature help on `f(`, and neither parses. The
analyser keeps the last tree that did, and every byte before the cursor still
has the offset it had then — so the receiver's type is still on record. What
cannot come from the tree is read off the line directly.

## Adding syntax

`CONTRIBUTING.md` asks for `syntaxes/jaithon.tmLanguage.json` to be updated
whenever the language grows. Two things are worth knowing:

- The keyword list mirrors `TOK_KW_*` in `src/lang/token.h`.
- `src/builtins.js` mirrors `src/runtime/builtins.c` (methods on primitive
  receivers), `builtins_core.c` (free functions) and `errors.c` (the exception
  hierarchy). Nothing depends on these being exhaustive; a name that has drifted
  simply stops being offered.
