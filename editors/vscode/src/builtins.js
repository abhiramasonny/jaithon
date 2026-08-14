// Names the compiler knows without being told.

const KEYWORDS = [
    'and', 'as', 'assert', 'break', 'case', 'catch', 'class', 'const', 'continue',
    'defer', 'elif', 'else', 'enum', 'export', 'extends', 'false', 'finally',
    'for', 'fn', 'from', 'if', 'impl', 'import', 'in', 'is', 'let', 'loop',
    'match', 'module', 'mut', 'not', 'null', 'or', 'prot', 'pub', 'return',
    'self', 'static', 'super', 'throw', 'trait', 'true', 'try', 'type', 'var',
    'while', 'yield',
];

const PRIMITIVE_TYPES = [
    'int', 'float', 'bool', 'str', 'bytes', 'any', 'void', 'never',
    'list', 'dict', 'set', 'tuple', 'range',
];

/** name -> signature, from jaiDefineNative in src/runtime/builtins/builtins_core.c. */
const FUNCTIONS = {
    print: { signature: 'print(...values) -> void', doc: 'Write each value to standard output, separated by spaces, then a newline.' },
    input: { signature: 'input(prompt: str = "") -> str', doc: 'Read one line from standard input.' },
    len: { signature: 'len(container) -> int', doc: 'Number of elements in a str, list, dict, set, tuple, bytes or range.' },
    range: { signature: 'range(start: int, stop: int = …, step: int = 1) -> range', doc: 'A half-open range. With one argument it counts from zero.' },
    type_of: { signature: 'type_of(value) -> str', doc: 'The name of the value\'s type.' },
    repr: { signature: 'repr(value) -> str', doc: 'An unambiguous rendering, quoting strings.' },
    str: { signature: 'str(value) -> str', doc: 'Convert to a string, as `print` would.' },
    int: { signature: 'int(value, base: int = 10) -> int', doc: 'Convert to an integer, truncating a float toward zero.' },
    float: { signature: 'float(value) -> float', doc: 'Convert to a float.' },
    bool: { signature: 'bool(value) -> bool', doc: 'Truthiness of the value.' },
    id: { signature: 'id(value) -> int', doc: 'A stable identity for the object.' },
    hash: { signature: 'hash(value) -> int', doc: 'The hash used by dict and set.' },
    abs: { signature: 'abs(value) -> int | float', doc: 'Magnitude, keeping the argument\'s type.' },
    min: { signature: 'min(...values)', doc: 'Smallest argument, or smallest element of a single container.' },
    max: { signature: 'max(...values)', doc: 'Largest argument, or largest element of a single container.' },
    sum: { signature: 'sum(values, start = 0)', doc: 'Add every element to `start`.' },
    sorted: { signature: 'sorted(values, key: fn = …, reverse: bool = false) -> list', doc: 'A new sorted list; the original is untouched.' },
    reversed: { signature: 'reversed(values) -> list', doc: 'A new list in reverse order.' },
    enumerate: { signature: 'enumerate(values, start: int = 0) -> Iterator[tuple[int, any]]', doc: 'Pair each element with its index.' },
    zip: { signature: 'zip(...iterables) -> Iterator[tuple]', doc: 'Walk several iterables in step, stopping at the shortest.' },
    map: { signature: 'map(fn, ...iterables) -> Iterator', doc: 'Apply `fn` to each element.' },
    filter: { signature: 'filter(predicate, values) -> Iterator', doc: 'Keep the elements `predicate` accepts.' },
    any: { signature: 'any(values) -> bool', doc: 'True when at least one element is truthy.' },
    all: { signature: 'all(values) -> bool', doc: 'True when every element is truthy.' },
    chr: { signature: 'chr(code: int) -> str', doc: 'The character with this codepoint.' },
    ord: { signature: 'ord(character: str) -> int', doc: 'The codepoint of a one-character string.' },
    isinstance: { signature: 'isinstance(value, type) -> bool', doc: 'True when `value` is an instance of `type` or a subclass of it.' },
    callable: { signature: 'callable(value) -> bool', doc: 'True when the value can be called.' },
    dir: { signature: 'dir(value) -> list[str]', doc: 'The names the value answers to.' },
    assert_eq: { signature: 'assert_eq(actual, expected, message: str = "")', doc: 'Throw an AssertionError unless the two are equal.' },
    exit: { signature: 'exit(status: int = 0) -> never', doc: 'End the process.' },
    list: { signature: 'list(values = []) -> list', doc: 'Build a list from any iterable.' },
    dict: { signature: 'dict(pairs = []) -> dict', doc: 'Build a dict from pairs.' },
    set: { signature: 'set(values = []) -> set', doc: 'Build a set from any iterable.' },
    tuple: { signature: 'tuple(values = []) -> tuple', doc: 'Build a tuple from any iterable.' },
    bytes: { signature: 'bytes(values = []) -> bytes', doc: 'Build a byte string.' },
};

const FALLBACK_METHODS = {
    str: ['at', 'capitalize', 'center', 'chars', 'code_at', 'contains', 'count',
          'encode', 'ends_with', 'find', 'format', 'index', 'is_alnum', 'is_alpha',
          'is_digit', 'is_empty', 'is_lower', 'is_space', 'is_upper', 'iter', 'join',
          'len', 'lines', 'lower', 'lstrip', 'pad_left', 'pad_right', 'repeat',
          'replace', 'reversed', 'rfind', 'rstrip', 'slice', 'split', 'splitlines',
          'starts_with', 'strip', 'title', 'to_float', 'to_int', 'to_str', 'upper'],
    list: ['all', 'any', 'append', 'at', 'chunks', 'clear', 'clone', 'concat',
           'contains', 'copy', 'count', 'drop', 'enumerate', 'extend', 'filter',
           'find', 'first', 'flatten', 'fold', 'index', 'insert', 'is_empty', 'iter',
           'join', 'last', 'len', 'map', 'max', 'min', 'pop', 'position', 'push',
           'reduce', 'remove', 'reverse', 'reversed', 'set', 'shuffle', 'slice',
           'sort', 'sorted', 'sum', 'take', 'to_list', 'unique', 'zip'],
    dict: ['clear', 'contains', 'copy', 'filter', 'get', 'get_or_insert', 'has',
           'items', 'iter', 'keys', 'len', 'map_values', 'merge', 'pop', 'remove',
           'set', 'update', 'values'],
    set: ['add', 'clear', 'contains', 'copy', 'difference', 'discard', 'has',
          'intersection', 'is_subset', 'is_superset', 'iter', 'len', 'remove',
          'symmetric_difference', 'to_list', 'union'],
    tuple: ['at', 'contains', 'count', 'first', 'get', 'index', 'is_empty', 'iter',
            'last', 'len', 'second', 'slice', 'to_list'],
    range: ['contains', 'is_empty', 'iter', 'len', 'reversed', 'start', 'step',
            'stop', 'to_list'],
    bytes: ['at', 'concat', 'contains', 'decode', 'get', 'hex', 'is_empty', 'iter',
            'len', 'slice', 'to_list'],
    int: ['abs', 'bit_length', 'clamp', 'hash', 'is_even', 'is_odd', 'max', 'min',
          'pow', 'sign', 'to_float', 'to_int', 'to_str'],
    float: ['abs', 'ceil', 'clamp', 'floor', 'hash', 'is_finite', 'is_inf', 'is_nan',
            'round', 'sign', 'sqrt', 'to_float', 'to_int', 'to_str', 'trunc'],
    File: ['close', 'flush', 'is_closed', 'iter', 'lines', 'read', 'read_line',
           'read_lines', 'seek', 'tell', 'write', 'write_line'],
    Iterator: ['all', 'any', 'chain', 'collect', 'count', 'drop', 'enumerate',
               'filter', 'find', 'fold', 'last', 'map', 'next', 'nth', 'peekable',
               'take', 'to_list', 'zip'],
};

/** The hierarchy from jaiRegisterErrorClasses in src/runtime/errors.c. */
const EXCEPTIONS = {
    Error: null,
    AssertionError: 'Error',
    ArithmeticError: 'Error',
    DivisionByZeroError: 'ArithmeticError',
    OverflowError: 'ArithmeticError',
    LookupError: 'Error',
    IndexError: 'LookupError',
    KeyError: 'LookupError',
    NameError: 'Error',
    TypeError: 'Error',
    ValueError: 'Error',
    ParseError: 'ValueError',
    AttributeError: 'Error',
    IOError: 'Error',
    FileNotFoundError: 'IOError',
    PermissionError: 'IOError',
    OSError: 'Error',
    RuntimeError: 'Error',
    RecursionError: 'RuntimeError',
    StopIteration: 'RuntimeError',
    ImportError: 'Error',
};

/** Every exception carries these; see addErrorFields. Also replaced by `refresh`. */
const FALLBACK_EXCEPTION_MEMBERS = ['message', 'traceback', 'traceback_string'];

// What the editor actually consults. Starts as the tables above and is replaced
// wholesale the first time the compiler answers.
let METHODS = { ...FALLBACK_METHODS };
let EXCEPTION_MEMBERS = [...FALLBACK_EXCEPTION_MEMBERS];
let live = false;

const PROBE = '{"int": dir(0), "float": dir(0.0), "str": dir(""), "list": dir([]), '
    + '"dict": dir({}), "set": dir(set()), "tuple": dir((1, 2)), "range": dir(0..1), '
    + '"bytes": dir(bytes()), "Iterator": dir([].iter()), "Error": dir(Error(""))}';

async function refresh(tool, output) {
    const result = await tool.run(['--eval', PROBE], {});
    if (result.spawnFailed || result.code !== 0) return false;

    let parsed;
    try {
        parsed = JSON.parse(result.stdout.trim());
    } catch {
        output?.appendLine('could not read the builtin method tables; using the bundled copy');
        return false;
    }
    if (!parsed || typeof parsed !== 'object') return false;

    const next = { ...FALLBACK_METHODS };
    for (const [type, names] of Object.entries(parsed)) {
        if (!Array.isArray(names) || !names.every((n) => typeof n === 'string')) continue;
        if (type === 'Error') {
            EXCEPTION_MEMBERS = names.filter((name) => !name.startsWith('__') && name !== 'init');
            continue;
        }
        next[type] = names;
    }
    METHODS = next;
    live = true;
    return true;
}

/** True once the tables came from the compiler rather than from this file. */
function isLive() {
    return live;
}

/** Traits std.prelude re-exports into every module. */
const PRELUDE_TRAITS = [
    'Iterable', 'Iterator', 'Comparable', 'Hashable', 'Printable',
    'Cloneable', 'Default', 'Container', 'Ordering',
];

/** Methods that give a class its behaviour under operators, `print` and `for`. */
const DUNDERS = [
    '__init__', '__str__', '__repr__', '__eq__', '__ne__', '__lt__', '__le__',
    '__gt__', '__ge__', '__hash__', '__add__', '__sub__', '__mul__', '__div__',
    '__floordiv__', '__mod__', '__pow__', '__neg__', '__len__', '__iter__',
    '__next__', '__contains__', '__getitem__', '__setitem__', '__call__',
    '__format__', '__clone__',
];

/** The methods a receiver of this checker type answers to. */
function methodsFor(type) {
    if (!type) return [];
    const text = type.trim().replace(/\?+$/, '');
    const base = text.split('[')[0].split('.').pop();
    if (METHODS[base]) return METHODS[base];
    if (EXCEPTIONS[base] !== undefined) return EXCEPTION_MEMBERS;
    return [];
}

function exceptionMembers() {
    return EXCEPTION_MEMBERS;
}

/** The namespace the runtime's primitive operations live under. */
const PRIMITIVE_NAMESPACE = '__prim__';

function isException(name) {
    return Object.prototype.hasOwnProperty.call(EXCEPTIONS, name);
}

/** `ParseError` -> `ValueError` -> `Error`. */
function exceptionChain(name) {
    const out = [];
    let current = name;
    while (current && EXCEPTIONS[current] !== undefined) {
        out.push(current);
        current = EXCEPTIONS[current];
    }
    return out;
}

module.exports = {
    KEYWORDS, PRIMITIVE_TYPES, FUNCTIONS, EXCEPTIONS,
    PRELUDE_TRAITS, DUNDERS, PRIMITIVE_NAMESPACE,
    methodsFor, isException, exceptionChain, exceptionMembers, refresh, isLive,
};
