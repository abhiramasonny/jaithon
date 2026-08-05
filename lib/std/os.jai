#: The environment, the working directory, and child processes.
#:
#: Everything a program needs to know about the machine it is running on and
#: everything it needs to run another program. Paths and files are `std.io`'s
#: job; this module is about the process.
#:
#: Running a command comes in three shapes, in increasing order of control:
#:
#: - `run(command)` runs it to completion and hands back a `Completed` with the
#:   exit code and the captured output.
#: - `run_checked(command)` does the same but raises `OSError` when the command
#:   fails, which is what a build step or a deploy script wants.
#: - `spawn(command)` returns a live `Process` whose input and output are open
#:   pipes, for a command that streams or that outlives the call.
#:
#: ```
#: run(["echo", "hello"]).stdout        # "hello\n"
#: run(["echo", "hello"]).exit_code     # 0
#: which("no-such-program-xyzzy")       # null
#: ```
#:
#: A command is a list of arguments, never a string, so nothing is ever handed
#: to a shell to re-parse: an argument containing a space, a quote or a `$` is
#: passed through untouched. `shell` exists for the cases that genuinely want a
#: shell, and says in its own documentation why you should not reach for it
#: with untrusted input.
#:
#: ## The primitive contract
#:
#: This module reaches the operating system through the `os` primitives of the
#: language specification, and only through the private wrappers at the bottom
#: of this file:
#:
#: - `os_env()` returns the whole environment as a `dict[str, str]`;
#:   `os_env(name)` returns one value or null; `os_env(name, value)` sets it,
#:   and a null `value` removes it.
#: - `os_argv() -> list[str]` is the command line, program name first.
#: - `os_cwd() -> str`, `os_chdir(path)`, `os_exit(code)`.
#: - `os_spawn(argv, options) -> dict` starts `argv[0]`, searching `PATH` when
#:   it holds no `/`, and raises `FileNotFoundError` when it is not there. The
#:   options are `cwd`, `env`, `stdin` and `stream`. Without `stream` the call
#:   waits and returns `exit_code`, `stdout` and `stderr`. With `stream` it
#:   returns at once with `pid`, the three pipe handles, and the `wait`, `poll`
#:   and `signal` callables that a `Process` is built from.
#: - `cpu_count() -> int`.
#:
#: There is no primitive for the platform name, the host name or this process's
#: own id, so those three are derived by asking a child process and are cached
#: after the first call.
from std.core import Iterator, Printable
from std.io import BufferedReader, File, Path

#: The interrupt signal, as sent by Ctrl-C.
pub const SIGINT = 2

#: The kill signal, which a process cannot catch or ignore.
pub const SIGKILL = 9

#: The termination signal, which asks a process to shut down cleanly.
pub const SIGTERM = 15

const _SHELL = "/bin/sh"
const _MESSAGE_LIMIT = 200

var _platform_name: str? = null
var _host_name: str? = null
var _user_name: str? = null
var _process_id: int? = null

#: A finished command: how it ended and what it printed.
pub class Completed: Printable {
    #: The command that was run, program name first.
    pub let command: list[str]

    #: The exit status, 0 on success.
    pub let exit_code: int

    #: Everything the command wrote to standard output, decoded as UTF-8.
    pub let stdout: str

    #: Everything the command wrote to standard error, decoded as UTF-8.
    pub let stderr: str

    #: Record the outcome of running `command`.
    pub fn init(self, command: list[str], exit_code: int, stdout: str, stderr: str) {
        self.command = command
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
    }

    #: Whether the command exited with status 0.
    pub fn ok(self) -> bool { return self.exit_code == 0 }

    #: Render the outcome as `Completed(<command>, exit_code=<code>)`.
    pub fn to_str(self) -> str {
        return f"Completed({_render(self.command)}, exit_code={self.exit_code})"
    }

    fn __str__(self) -> str { return self.to_str() }
}

#: A running child process, with its three streams open as pipes.
#:
#: Use this when `run` will not do: when the output has to be consumed as it
#: arrives, when the child expects input that depends on its own output, or
#: when the child should outlive the call that started it.
#:
#: A pipe holds a finite amount of data. If the child fills its output pipe
#: while this process is blocked writing to its input pipe, both stop for good.
#: The way out is to read the output as you go, or to close `stdin` before
#: waiting — a child reading to end of input needs that close to finish at all.
#:
#: ```
#: let process = spawn(["cat"])
#: process.write("one\ntwo\n")
#: process.close_stdin()
#: process.read_line()      # "one"
#: process.read_line()      # "two"
#: process.wait()           # 0
#: ```
pub class Process: Printable {
    #: The command that was started, program name first.
    pub let command: list[str]

    #: The process id of the child.
    pub let pid: int

    #: The write end of the child's standard input.
    pub let stdin: File

    #: The read end of the child's standard output.
    pub let stdout: File

    #: The read end of the child's standard error.
    pub let stderr: File

    var status: int?
    var reader: BufferedReader?
    let waiter: fn() -> int
    let poller: fn() -> int?
    let signaller: fn(int) -> void

    #: Adopt the live process described by `handles`.
    #:
    #: `spawn` is what a caller wants; this constructor turns the record the
    #: spawn primitive returns into a `Process`.
    pub fn init(self, command: list[str], handles: dict[str, any]) {
        self.command = command
        self.pid = handles["pid"]
        self.stdin = File(handles["stdin"], _render(command) + " <stdin>", "w")
        self.stdout = File(handles["stdout"], _render(command) + " <stdout>", "r")
        self.stderr = File(handles["stderr"], _render(command) + " <stderr>", "r")
        self.status = null
        self.reader = null
        self.waiter = handles["wait"]
        self.poller = handles["poll"]
        self.signaller = handles["signal"]
    }

    #: Write `text` to the child's standard input.
    pub fn write(self, text: str) -> void {
        let _written = self.stdin.write(text)
        self.stdin.flush()
    }

    #: Write `text` and a newline to the child's standard input.
    pub fn write_line(self, text: str = "") -> void { self.write(text + "\n") }

    #: Close the child's standard input, so a child reading to end of input stops.
    pub fn close_stdin(self) -> void { self.stdin.close() }

    #: Read one line of the child's standard output, or null once it ends.
    pub fn read_line(self) -> str? { return self._reader().read_line() }

    #: Iterate the child's standard output line by line, as it arrives.
    pub fn lines(self) -> Iterator[str] { return self._reader().lines() }

    #: Read the rest of the child's standard output.
    #:
    #: This blocks until the child closes the stream, which usually means until
    #: it exits.
    pub fn read_stdout(self) -> str { return self._reader().read_all() }

    #: Read the rest of the child's standard error.
    pub fn read_stderr(self) -> str { return self.stderr.read() }

    #: Return the exit status if the child has finished, or null while it runs.
    pub fn poll(self) -> int? {
        let known = self.status
        if known is not null { return known }
        let result = self.poller()
        if result is not null { self.status = result }
        return result
    }

    #: Whether the child is still running.
    pub fn is_running(self) -> bool { return self.poll() is null }

    #: Wait for the child to finish and return its exit status.
    #:
    #: Waiting twice returns the same status without waiting again.
    pub fn wait(self) -> int {
        let known = self.status
        if known is not null { return known }
        let result = self.waiter()
        self.status = result
        return result
    }

    #: Send signal `number` to the child.
    #:
    #: Signalling a child that has already exited does nothing.
    pub fn signal(self, number: int) -> void {
        if self.poll() is not null { return }
        self.signaller(number)
    }

    #: Ask the child to shut down, by sending `SIGTERM`.
    pub fn terminate(self) -> void { self.signal(SIGTERM) }

    #: Stop the child immediately, by sending `SIGKILL`.
    pub fn kill(self) -> void { self.signal(SIGKILL) }

    #: Render the process as `Process(<command>, pid=<pid>)`.
    pub fn to_str(self) -> str { return f"Process({_render(self.command)}, pid={self.pid})" }

    # One buffered reader over standard output, shared by every read method, so
    # that read-ahead in one of them is visible to the others.
    fn _reader(self) -> BufferedReader {
        let existing = self.reader
        if existing is not null { return existing }
        let created = BufferedReader(self.stdout)
        self.reader = created
        return created
    }

    fn __str__(self) -> str { return self.to_str() }
}

#: Return the value of environment variable `name`, or null when it is unset.
#:
#: ```
#: env("JAITHON_DOC_UNSET_VARIABLE")    # null
#: ```
pub fn env(name: str) -> str? {
    _check_variable_name(name)
    return __prim__.os_env(name)
}

#: Return the value of environment variable `name`, or `fallback` when unset.
#:
#: An empty value counts as set, because a variable deliberately set to nothing
#: means something different from one that was never set at all.
#:
#: ```
#: env_or("JAITHON_DOC_UNSET_VARIABLE", "default")    # "default"
#: ```
pub fn env_or(name: str, fallback: str) -> str { return env(name) ?? fallback }

#: Set environment variable `name` to `value` for this process and its children.
#:
#: ```
#: set_env("JAITHON_DOC_VARIABLE", "recorded")
#: env("JAITHON_DOC_VARIABLE")                              # "recorded"
#: run(["sh", "-c", "echo $JAITHON_DOC_VARIABLE"]).stdout  # "recorded\n"
#: unset_env("JAITHON_DOC_VARIABLE")
#: env("JAITHON_DOC_VARIABLE")                              # null
#: ```
pub fn set_env(name: str, value: str) -> void {
    _check_variable_name(name)
    __prim__.os_env(name, value)
}

#: Remove environment variable `name`. Removing an unset variable does nothing.
pub fn unset_env(name: str) -> void {
    _check_variable_name(name)
    __prim__.os_env(name, null)
}

#: Return the whole environment as a dictionary.
#:
#: The result is a snapshot: changing it does not change the environment, and
#: it does not see a later `set_env`.
pub fn environ() -> dict[str, str] { return __prim__.os_env() }

#: Return the command line, with the program name first.
#:
#: This is the same list an entry-point `main(args)` receives, so `argv()[0]` is
#: the program and the arguments start at index 1.
pub fn argv() -> list[str] { return __prim__.os_argv() }

#: End the process immediately with status `code`.
#:
#: Nothing after the call runs: no `defer` block, no `finally` block, and no
#: buffered writer is flushed. Return from `main` instead wherever you can.
pub fn exit(code: int = 0) -> void { __prim__.os_exit(code) }

#: Return the current working directory as an absolute path.
#:
#: ```
#: Path(cwd()).is_absolute()    # true
#: ```
pub fn cwd() -> str { return __prim__.os_cwd() }

#: Change the current working directory to `path`.
#:
#: The working directory is process-wide, so this affects every thread and
#: every relative path the program later resolves. Raises `FileNotFoundError`
#: when `path` is not a directory this process can enter.
pub fn chdir(path: str) -> void { __prim__.os_chdir(path) }

#: Return the current user's home directory.
#:
#: Raises `OSError` when `HOME` is unset, which happens in stripped-down
#: environments such as a bare `cron` job or an empty container.
#:
#: ```
#: Path(home_dir()).is_dir()    # true
#: ```
pub fn home_dir() -> str {
    let home = env("HOME")
    if home is null or home.len() == 0 {
        throw OSError("HOME is not set, so there is no home directory to report")
    }
    return home
}

#: Return the directory for temporary files, without a trailing separator.
#:
#: `TMPDIR`, `TMP` and `TEMP` are consulted in that order, falling back to
#: `/tmp`.
#:
#: ```
#: Path(temp_dir()).is_dir()    # true
#: ```
pub fn temp_dir() -> str {
    let named = ((env("TMPDIR") ?? env("TMP")) ?? env("TEMP")) ?? "/tmp"
    var text = if named.len() == 0 { "/tmp" } else { named }
    # `TMPDIR` conventionally ends in a separator; drop it so that callers can
    # join onto the result without thinking about it.
    while text.len() > 1 and text[text.len() - 1] == "/" { text = text[0:text.len() - 1] }
    return text
}

#: Return the operating system name in lower case.
#:
#: The known answers are `"macos"`, `"linux"`, `"freebsd"`, `"openbsd"` and
#: `"netbsd"`; anything else is the lower-cased output of `uname -s`. The
#: primitive surface has no platform query, so the first call runs `uname` and
#: every later call returns the cached answer.
pub fn platform() -> str {
    let cached = _platform_name
    if cached is not null { return cached }
    # A fresh binding rather than a store into `cached`: a store ends the
    # narrowing fact (spec §2.1), so the assigned value has to carry its own.
    let detected = _detect_platform()
    _platform_name = detected
    return detected
}

#: Return the number of hardware threads the machine reports.
#:
#: ```
#: cpu_count() >= 1    # true
#: ```
pub fn cpu_count() -> int { return __prim__.cpu_count() }

#: Return the machine's host name, from `uname -n`, cached after the first call.
pub fn hostname() -> str {
    let cached = _host_name
    if cached is not null { return cached }
    # A fresh binding rather than a store into `cached`: a store ends the
    # narrowing fact (spec §2.1), so the assigned value has to carry its own.
    let detected = _uname("-n")
    _host_name = detected
    return detected
}

#: Return the name of the user this process runs as, cached after the first call.
#:
#: `USER` and `LOGNAME` are consulted first, and `id -un` asked only when
#: neither is set.
pub fn username() -> str {
    let cached = _user_name
    if cached is not null { return cached }
    # A fresh binding rather than a store into `cached`: a store ends the
    # narrowing fact (spec §2.1), so the assigned value has to carry its own.
    let detected = _detect_username()
    _user_name = detected
    return detected
}

#: Return this process's own id.
#:
#: ```
#: pid() > 0    # true
#: ```
pub fn pid() -> int {
    let cached = _process_id
    if cached is not null { return cached }
    # A fresh binding rather than a store into `cached`: a store ends the
    # narrowing fact (spec §2.1), so the assigned value has to carry its own.
    let detected = _detect_pid()
    _process_id = detected
    return detected
}

#: Run `command` to completion and return how it went.
#:
#: `command[0]` is looked up on `PATH` unless it contains a `/`. Standard output
#: and standard error are captured separately and decoded as UTF-8, so this is
#: for commands that produce text; a command producing binary output wants
#: `spawn` and `read_bytes` on the pipe. `stdin_text`, when given, is written to
#: the child and the stream is then closed.
#:
#: A command that fails is not an error here — inspect `exit_code`, or use
#: `run_checked`. Raises `FileNotFoundError` when the program does not exist.
#:
#: ```
#: run(["echo", "one two"]).stdout                       # "one two\n"
#: run(["cat"], stdin_text: "piped\n").stdout            # "piped\n"
#: run(["sh", "-c", "exit 3"]).exit_code                 # 3
#: run(["sh", "-c", "exit 3"]).ok()                      # false
#: ```
pub fn run(
    command: list[str],
    cwd: str? = null,
    environment: dict[str, str]? = null,
    stdin_text: str? = null
) -> Completed {
    if command.len() == 0 { throw ValueError("command must name a program to run") }
    let options: dict[str, any] = {
        "cwd": cwd,
        "env": environment,
        "stdin": stdin_text,
        "stream": false,
    }
    let result = __prim__.os_spawn(command, options)
    return Completed(command, result["exit_code"], result["stdout"], result["stderr"])
}

#: Run `command` to completion and raise `OSError` unless it succeeds.
#:
#: The exception message names the command, the status and the beginning of
#: what the command wrote to standard error, because a build step that fails
#: silently is worse than one that stops.
#:
#: ```
#: run_checked(["echo", "fine"]).stdout    # "fine\n"
#: ```
pub fn run_checked(
    command: list[str],
    cwd: str? = null,
    environment: dict[str, str]? = null,
    stdin_text: str? = null
) -> Completed {
    let result = run(command, cwd, environment, stdin_text)
    if result.exit_code != 0 {
        let reason = _truncate(result.stderr.strip())
        throw OSError(f"{_render(command)} exited with status {result.exit_code}: {reason}")
    }
    return result
}

#: Run `command` through `/bin/sh -c` and return how it went.
#:
#: The shell brings pipes, redirections and globbing, and it brings the whole
#: shell grammar with them: any value interpolated into `command` is code, not
#: data. Never build the string from anything a user supplied — use `run` with
#: a list of arguments, which cannot be re-parsed.
#:
#: ```
#: shell("echo one; echo two").stdout    # "one\ntwo\n"
#: ```
pub fn shell(command: str, cwd: str? = null) -> Completed {
    return run([_SHELL, "-c", command], cwd)
}

#: Start `command` and return it as a live `Process`.
#:
#: The call returns as soon as the child exists; nothing waits for it. The
#: caller owns the child's three pipes and must `wait` for it, or the finished
#: child stays in the process table.
pub fn spawn(command: list[str], cwd: str? = null, environment: dict[str, str]? = null) -> Process {
    if command.len() == 0 { throw ValueError("command must name a program to run") }
    let options: dict[str, any] = {"cwd": cwd, "env": environment, "stdin": null, "stream": true}
    return Process(command, __prim__.os_spawn(command, options))
}

#: Return the path `program` would run from, or null when it is not on `PATH`.
#:
#: A `program` containing a `/` is checked as it stands rather than searched
#: for. An empty entry in `PATH` means the current directory, as the shell
#: reads it.
#:
#: ```
#: which("no-such-program-xyzzy")    # null
#: which("sh") is not null           # true
#: ```
pub fn which(program: str) -> str? {
    if program.len() == 0 { throw ValueError("program name must not be empty") }
    if _has_separator(program) {
        let direct = Path(program)
        if direct.is_executable() { return direct.text }
        return null
    }
    for entry in env_or("PATH", "").split(":") {
        let directory = if entry.len() == 0 { "." } else { entry }
        let candidate = Path(directory).join(program)
        if candidate.is_executable() { return candidate.text }
    }
    return null
}

fn _detect_platform() -> str {
    let system = _uname("-s")
    return match system {
        "Darwin" => "macos",
        "Linux" => "linux",
        "FreeBSD" => "freebsd",
        "OpenBSD" => "openbsd",
        "NetBSD" => "netbsd",
        _ => system.lower(),
    }
}

fn _detect_username() -> str {
    let named = env("USER") ?? env("LOGNAME")
    if named is not null and named.len() > 0 { return named }
    let result = run(["id", "-un"])
    if result.exit_code != 0 {
        throw OSError("USER and LOGNAME are unset and `id -un` failed")
    }
    return result.stdout.strip()
}

# There is no primitive that reports this process's own id, but a child shell
# knows its parent's, and `os_spawn` makes the shell a direct child.
fn _detect_pid() -> int {
    let result = run([_SHELL, "-c", "echo $PPID"])
    let parsed = result.stdout.strip().to_int()
    if result.exit_code != 0 or parsed is null {
        throw OSError("cannot determine this process id: `sh -c 'echo $PPID'` failed")
    }
    return parsed
}

fn _uname(flag: str) -> str {
    let result = run(["uname", flag])
    if result.exit_code != 0 {
        throw OSError(f"`uname {flag}` exited with status {result.exit_code}")
    }
    return result.stdout.strip()
}

fn _check_variable_name(name: str) -> void {
    if name.len() == 0 { throw ValueError("environment variable name must not be empty") }
    for index in 0..name.len() {
        if name[index] == "=" or name[index] == "\0" {
            throw ValueError(f"environment variable name may not contain {repr(name[index])}")
        }
    }
}

fn _has_separator(text: str) -> bool {
    for index in 0..text.len() {
        if text[index] == "/" { return true }
    }
    return false
}

fn _render(command: list[str]) -> str { return " ".join(command) }

fn _truncate(text: str) -> str {
    if text.len() <= _MESSAGE_LIMIT { return text }
    return text[0:_MESSAGE_LIMIT] + "..."
}
