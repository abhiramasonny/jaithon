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
`import` into other modules, resolved the way `src/runtime/modules/module.c` resolves
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
class and a variable spelled alike stop looking alike. Across `lib/`,
`examples/` and `tests/`, 107,664 of 107,666 identifiers outside strings and
comments carry a semantic token — a name is coloured by what it *is*, not by
what it looks like, essentially everywhere.

**`.jaic` viewer.** Opening a bytecode image renders `jaithon disasm` instead of
binary garbage.

## Installing

```bash
editors/vscode/install.sh
```

Then reload the window. `EDITOR_CLI=cursor editors/vscode/install.sh` installs
it elsewhere.

Symlinking the folder into `~/.vscode/extensions` does **not** work, however
much it looks like it should. The registry in `extensions.json` is
authoritative: a folder that is not listed there is logged as

```
Marked extension as removed jaithon.jaithon-3.1.0
```

and never scanned, never activated, and absent from `code --list-extensions` —
with nothing shown in the UI to say so. The script packages a `.vsix` and
installs it through the CLI, which is what writes that entry.

The install is therefore a copy. Run the script again after changing anything
here. To iterate without reinstalling, open this folder in VS Code and press F5
(Extension Development Host), which runs the source in place.

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
| `src/analysis.js` | The language service: `ast --json`, indexed into scopes, symbols and references |
| `src/builtins.js` | The names the runtime registers in C, which no `.jai` file declares |
| `src/navigation.js` | Definition, references, rename, outlines, folding, call hierarchy |
| `src/completion.js` | Completion and signature help |
| `src/hints.js` | Hover, inlay hints, code lenses, semantic tokens |
| `src/actions.js` | Formatting and code actions |
| `src/jaic.js` | The bytecode viewer |

One fact does most of the work: `jaithon ast --json` puts a byte span on every
node, and the whole language service is that answer indexed. Only lexical
scoping is reimplemented here, because the tree does not record which
declaration a bare name resolved to.

There was a second source. `jaithon check --dump-sema` reported the checker's
type for each span, which is what answered member access — `p.norm` came from
the type checker rather than a guess. It was an option of the C front end, which
no longer exists, so member access now falls back to the syntax tree. Reviving
it means the self-hosted front end emitting the same dump.

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
- A keyword must get **one** scope everywhere. `fn` used to be
  `storage.type.function` when it named a function and `storage.type` when it
  did not, which is what "the highlighting is inconsistent" looks like from the
  outside. `editors/vscode/syntaxes` has no context-dependent keyword scopes
  left, and nothing in the tree relies on one.

Where a scope genuinely cannot be decided from the text — a name that is a
parameter here and a property there — leave it alone and let semantic tokens
answer. Guessing is what produces the flicker.

## Where the builtin names come from

Everything written in `.jai` is read from the source: hovering `math.sqrt`
shows the `#:` comment above it in `lib/std/math.jai`, because the analyser
indexes doc comments from the syntax tree. Nothing about the standard library
is baked into this extension.

The exception is the surface implemented in C — `print`, `len`, `list.map`,
`ValueError` — which no `.jai` file declares and which therefore has no `#:`
comment to read. For those:

- **Names** come from the compiler at run time. `src/builtins.js` asks it
  `dir(0)`, `dir("")`, `dir([])` and so on in a single 6 ms process, which
  probes the real dispatch tables. That is not a convenience: the static tables
  in `src/runtime/builtins/builtins.c` list the names `dir` *tries*, and thirteen of the
  forty-six it lists for `list` are not implemented in any build. Completion
  offered all thirteen before this asked.
- **Prose** is the short table in `src/builtins.js`, because nothing in the
  tree carries it. It is used only until the compiler answers, and only for
  names the compiler itself implements.
