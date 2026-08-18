# Source layout

`src/` contains the C and Objective-C runtime used by the `jaithon` executable.
The compiler lives in `lib/jaithon/compile/` and runs on this runtime.

```text
src/
├── cli/
│   └── commands/       command implementations for build, run, and inspection
├── common/             diagnostics, allocation, buffers, and shared definitions
├── native/
│   ├── apple/          Cocoa and Metal implementations
│   └── posix/          processes, files, and threading
├── runtime/
│   ├── builtins/
│   │   ├── collections/  list, dictionary, bytes, and sequence methods
│   │   ├── io/           files, processes, and compression
│   │   ├── platform/     canvas, GPU, GUI, and thread primitives
│   │   └── text/         strings and formatting
│   └── modules/         imports, source compilation, cache loading, and seed boot
└── vm/
    ├── bytecode/       chunks, verification, and `.jaic` serialization
    ├── jit/            code arena and arm64 JIT
    ├── object/         heap object implementations
    └── trace/          `@trace` sessions (op names, shapes, replay)
```

The files at `vm/` hold the interpreter, values, tables, and garbage collector.
The files at `runtime/` define shared runtime state, errors, and native handles.
Public subsystem headers stay at those levels; private headers sit beside their
implementation group.
