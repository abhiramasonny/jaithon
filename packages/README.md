# Workspace packages

Jaithon's standard library is at `lib/`. External Libraries are here as workspace packages.

Each package uses this layout:

```text
packages/name/
├── jaithon.package.json
├── src/name/
└── tests/
```

Project examples stay in the root `examples/` directory.

The source directory must remain `src`, and its public module must match the
package name. The runtime scans package directories in name order and adds each
manifest-backed `src` directory to the import path. An installed Jaithon scans
`share/jaithon/packages` by the same rule.

`jaithon.workspace.json` lists the packages in this checkout. Package manifests
declare exact versions or caret ranges in `dependencies`. Run
`make package-check` after changing a manifest. The check rejects missing
members, duplicate names, unresolved versions, and dependency cycles.

`jaitensor` is GPU-first training on `std.gpu`. `jaiplot` is Matplotlib-style
figures. See the root [`README.md`](../README.md) for what each one covers.
