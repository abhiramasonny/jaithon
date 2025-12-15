# JAITHON

**Java + Python as a programming language**

[Documentation](docs/grammar.md)

## Instalation

### Prerequisites

- GCC compiler
- Make build tool
- readline library
- MacOS with apple silicon (windows/linux works without GUI/GPU stuff)

### make from src

```bash
git clone https://github.com/abhiramasonny/jaithon
cd jaithon
make
```

### macOS easy install

```bash
sudo bash ./scripts/install_macos.sh   # copies jaithon to /usr/local/bin and libs to /usr/local/share/jaithon
```

### Running Programs

```bash
# Run a file
jaithon test/checks/variables.jai

# Enter shell
jaithon

# view options
jaithon -h
```

---

## Standard Library

Jaithon includes a stdlib written in Jaithon (`lib/std.jai`). These functions are available automatically. 

## Documentation

Full language reference: [`docs/grammar.md`](docs/grammar.md)

---

## License

See [LICENSE](LICENSE) for details.

---

## Author

Created by [Abhirama Sonny](https://abhiramasonny.com/)
