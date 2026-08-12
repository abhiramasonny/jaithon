/* process.c — subprocesses, directory listing, stat, and the running binary's
 * path; backs __prim__.os_spawn, __prim__.io_listdir, __prim__.io_stat.
 * Failures are a status code or false — turning that into IOError is the wrapper's job, not this layer's. */

/* Feature macros must precede every include: popen, readdir and readlink are
 * POSIX, and getprogname needs the full Darwin surface. */
#if !defined(_POSIX_C_SOURCE)
#  define _POSIX_C_SOURCE 200809L
#endif
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#  define _DARWIN_C_SOURCE
#endif

#include "native/native.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#if defined(__APPLE__)
#  include <mach-o/dyld.h>
#endif

#define JAI_PROC_READ_BLOCK 4096u

/* Runs `command` through the shell; returns the exit code, 128 + signal for a
 * killed child, or -1 (no output handed back) on failure. *outStdout is a
 * NUL-terminated heap string that may hold embedded NULs — free with
 * JAI_FREE_ARRAY(char, s, len + 1). stderr stays attached to the parent's. */
int jaiProcessRun(const char *command, char **outStdout, size_t *outLen) {
    if (outStdout != NULL) *outStdout = NULL;
    if (outLen != NULL) *outLen = 0;
    if (command == NULL) return -1;

    FILE *proc = popen(command, "r");
    if (proc == NULL) return -1;

    JaiBuf out;
    jaiBufInit(&out);

    char block[JAI_PROC_READ_BLOCK];
    for (;;) {
        size_t got = fread(block, 1, sizeof block, proc);
        if (got > 0) jaiBufAppend(&out, block, got);
        if (got < sizeof block) break;   /* EOF or error; ferror tells which */
    }
    bool readFailed = ferror(proc) != 0;

    /* pclose waits for the child, so the status is only valid after this. */
    int status = pclose(proc);
    if (readFailed || status == -1) {
        jaiBufFree(&out);
        return -1;
    }

    if (outStdout != NULL) {
        size_t len = 0;
        *outStdout = jaiBufTakeCString(&out, &len);
        if (outLen != NULL) *outLen = len;
    } else {
        jaiBufFree(&out);
    }

    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return status;   /* stopped or unknown: hand the raw wait status back */
}

extern char **environ;

/* The child reports exec failure by writing errno down a close-on-exec pipe;
 * a successful exec closes it, so EOF there tells "not found" from "ran and exited 127". */
#define JAI_EXEC_FAILED 127

static void closeFd(int *fd) {
    if (*fd >= 0) { (void)close(*fd); *fd = -1; }
}

static bool setCloexec(int fd) {
    int flags = fcntl(fd, F_GETFD);
    return flags >= 0 && fcntl(fd, F_SETFD, flags | FD_CLOEXEC) == 0;
}

/* Returns false at end of stream; that's what drops the descriptor from the poll set. */
static bool drainFd(int fd, JaiBuf *buf, bool *outFailed) {
    char block[JAI_PROC_READ_BLOCK];
    for (;;) {
        ssize_t got = read(fd, block, sizeof block);
        if (got > 0) {
            jaiBufAppend(buf, block, (size_t)got);
            if ((size_t)got < sizeof block) return true;   /* drained for now */
            continue;
        }
        if (got == 0) return false;
        if (errno == EINTR) continue;
        if (errno == EAGAIN || errno == EWOULDBLOCK) return true;
        *outFailed = true;
        return false;
    }
}

/* All three pipes are non-blocking and polled together: a child that fills
 * stdout while this process is blocked writing its stdin would deadlock a naive write-then-read implementation. */
static bool pumpChild(int inFd, int outFd, int errFd,
                      const char *stdinText, size_t stdinLen,
                      JaiBuf *outBuf, JaiBuf *errBuf) {
    size_t written = 0;
    bool failed = false;

    if (inFd >= 0 && (stdinText == NULL || stdinLen == 0)) closeFd(&inFd);

    while (outFd >= 0 || errFd >= 0 || inFd >= 0) {
        struct pollfd fds[3];
        int slots[3];
        int n = 0;
        if (outFd >= 0) { fds[n].fd = outFd; fds[n].events = POLLIN;  slots[n] = 0; n++; }
        if (errFd >= 0) { fds[n].fd = errFd; fds[n].events = POLLIN;  slots[n] = 1; n++; }
        if (inFd  >= 0) { fds[n].fd = inFd;  fds[n].events = POLLOUT; slots[n] = 2; n++; }

        int ready = poll(fds, (nfds_t)n, -1);
        if (ready < 0) {
            if (errno == EINTR) continue;
            failed = true;
            break;
        }

        for (int i = 0; i < n; i++) {
            if (fds[i].revents == 0) continue;
            switch (slots[i]) {
            case 0:
                if (!drainFd(outFd, outBuf, &failed)) closeFd(&outFd);
                break;
            case 1:
                if (!drainFd(errFd, errBuf, &failed)) closeFd(&errFd);
                break;
            case 2: {
                if (fds[i].revents & (POLLERR | POLLHUP | POLLNVAL)) {
                    /* The child closed its stdin, or died. Not an error: a
                     * command that reads none of its input is normal. */
                    closeFd(&inFd);
                    break;
                }
                ssize_t put = write(inFd, stdinText + written, stdinLen - written);
                if (put > 0) written += (size_t)put;
                else if (put < 0 && errno != EINTR && errno != EAGAIN &&
                         errno != EWOULDBLOCK)
                    closeFd(&inFd);      /* EPIPE: the child is not reading */
                if (inFd >= 0 && written >= stdinLen) closeFd(&inFd);
                break;
            }
            default: break;
            }
        }
        if (failed) break;
    }

    closeFd(&inFd);
    closeFd(&outFd);
    closeFd(&errFd);
    return !failed;
}

static int waitStatusToExit(int status) {
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return status;
}

bool jaiProcessWait(int pid, bool block, int *outExit) {
    if (pid <= 0) { errno = ESRCH; return false; }
    for (;;) {
        int status = 0;
        pid_t got = waitpid((pid_t)pid, &status, block ? 0 : WNOHANG);
        if (got == (pid_t)pid) {
            if (outExit != NULL) *outExit = waitStatusToExit(status);
            return true;
        }
        if (got == 0) { errno = 0; return false; }   /* still running */
        if (errno == EINTR) continue;
        return false;
    }
}

bool jaiProcessSignal(int pid, int sig) {
    if (pid <= 0) { errno = ESRCH; return false; }
    return kill((pid_t)pid, sig) == 0;
}

JaiSpawnStatus jaiProcessSpawn(const char *const *argv, const char *cwd,
                               const char *const *envp,
                               const char *stdinText, size_t stdinLen,
                               bool stream, JaiSpawnResult *result,
                               int *outErrno) {
    if (outErrno != NULL) *outErrno = 0;
    if (result == NULL || argv == NULL || argv[0] == NULL) {
        if (outErrno != NULL) *outErrno = EINVAL;
        return JAI_SPAWN_SETUP;
    }

    result->pid = -1;
    result->exitCode = -1;
    result->stdinFd = result->stdoutFd = result->stderrFd = -1;
    result->out = NULL; result->outLen = 0;
    result->err = NULL; result->errLen = 0;

    int inPipe[2]  = { -1, -1 };
    int outPipe[2] = { -1, -1 };
    int errPipe[2] = { -1, -1 };
    int report[2]  = { -1, -1 };

    if (pipe(inPipe) != 0 || pipe(outPipe) != 0 || pipe(errPipe) != 0 ||
        pipe(report) != 0) {
        if (outErrno != NULL) *outErrno = errno;
        goto setupFailed;
    }
    /* Only the report pipe's write end is close-on-exec — closing it signals a successful exec. */
    if (!setCloexec(report[1])) {
        if (outErrno != NULL) *outErrno = errno;
        goto setupFailed;
    }

    pid_t child = fork();
    if (child < 0) {
        if (outErrno != NULL) *outErrno = errno;
        goto setupFailed;
    }

    if (child == 0) {
        /* Child. Only async-signal-safe calls from here to the exec. */
        int failure = 0;
        if (dup2(inPipe[0], STDIN_FILENO) < 0 ||
            dup2(outPipe[1], STDOUT_FILENO) < 0 ||
            dup2(errPipe[1], STDERR_FILENO) < 0) {
            failure = errno;
        }
        (void)close(inPipe[0]);  (void)close(inPipe[1]);
        (void)close(outPipe[0]); (void)close(outPipe[1]);
        (void)close(errPipe[0]); (void)close(errPipe[1]);
        (void)close(report[0]);

        if (failure == 0 && cwd != NULL && chdir(cwd) != 0) failure = errno;
        if (failure == 0) {
            /* SIGPIPE is ignored in the parent so a dead-reader write becomes
             * EPIPE, not death; the child must not inherit that, or it behaves unlike a shell pipeline. */
            struct sigaction restore;
            memset(&restore, 0, sizeof restore);
            restore.sa_handler = SIG_DFL;
            (void)sigaction(SIGPIPE, &restore, NULL);

            if (envp != NULL) environ = (char **)(uintptr_t)(const void *)envp;
            execvp(argv[0], (char *const *)(uintptr_t)(const void *)argv);
            failure = errno;
        }
        {
            /* The parent reads this to learn why; a short write leaves it at
             * end-of-file and the exit status still says 127. */
            ssize_t ignored = write(report[1], &failure, sizeof failure);
            (void)ignored;
        }
        _exit(JAI_EXEC_FAILED);
    }

    closeFd(&inPipe[0]);
    closeFd(&outPipe[1]);
    closeFd(&errPipe[1]);
    closeFd(&report[1]);

    int childErrno = 0;
    {
        ssize_t got;
        do { got = read(report[0], &childErrno, sizeof childErrno); }
        while (got < 0 && errno == EINTR);
        if (got != (ssize_t)sizeof childErrno) childErrno = 0;
    }
    closeFd(&report[0]);

    if (childErrno != 0) {
        closeFd(&inPipe[1]);
        closeFd(&outPipe[0]);
        closeFd(&errPipe[0]);
        (void)jaiProcessWait((int)child, true, NULL);
        if (outErrno != NULL) *outErrno = childErrno;
        return JAI_SPAWN_EXEC;
    }

    result->pid = (int)child;

    if (stream) {
        result->stdinFd  = inPipe[1];
        result->stdoutFd = outPipe[0];
        result->stderrFd = errPipe[0];
        return JAI_SPAWN_OK;
    }

    for (int i = 0; i < 3; i++) {
        int fd = i == 0 ? inPipe[1] : (i == 1 ? outPipe[0] : errPipe[0]);
        int flags = fcntl(fd, F_GETFL);
        if (flags >= 0) (void)fcntl(fd, F_SETFL, flags | O_NONBLOCK);
    }

    struct sigaction ignore, previous;
    memset(&ignore, 0, sizeof ignore);
    ignore.sa_handler = SIG_IGN;
    bool restoreSigpipe = sigaction(SIGPIPE, &ignore, &previous) == 0;

    JaiBuf outBuf, errBuf;
    jaiBufInit(&outBuf);
    jaiBufInit(&errBuf);
    bool pumped = pumpChild(inPipe[1], outPipe[0], errPipe[0],
                            stdinText, stdinLen, &outBuf, &errBuf);
    inPipe[1] = outPipe[0] = errPipe[0] = -1;   /* pumpChild closed them */

    if (restoreSigpipe) (void)sigaction(SIGPIPE, &previous, NULL);

    int exitCode = -1;
    bool reaped = jaiProcessWait((int)child, true, &exitCode);
    if (!pumped || !reaped) {
        jaiBufFree(&outBuf);
        jaiBufFree(&errBuf);
        if (outErrno != NULL) *outErrno = errno;
        return JAI_SPAWN_IO;
    }

    result->exitCode = exitCode;
    result->out = jaiBufTakeCString(&outBuf, &result->outLen);
    result->err = jaiBufTakeCString(&errBuf, &result->errLen);
    return JAI_SPAWN_OK;

setupFailed:
    closeFd(&inPipe[0]);  closeFd(&inPipe[1]);
    closeFd(&outPipe[0]); closeFd(&outPipe[1]);
    closeFd(&errPipe[0]); closeFd(&errPipe[1]);
    closeFd(&report[0]);  closeFd(&report[1]);
    return JAI_SPAWN_SETUP;
}

static void freeNameList(char **names, int count, int capacity) {
    for (int i = 0; i < count; i++) {
        JAI_FREE_ARRAY(char, names[i], strlen(names[i]) + 1);
    }
    JAI_FREE_ARRAY(char *, names, capacity);
}

/* Excludes "." and ".."; an empty directory still yields a one-element array
 * holding the NULL terminator. Caller frees each name, then the array, with
 * JAI_FREE_ARRAY (sizes: strlen(name) + 1 and count + 1). */
char **jaiListDir(const char *path, int *outCount) {
    if (outCount != NULL) *outCount = 0;
    if (path == NULL) return NULL;

    DIR *dir = opendir(path);
    if (dir == NULL) return NULL;

    char **names   = NULL;
    int    count    = 0;
    int    capacity = 0;
    bool   failed   = false;

    for (;;) {
        /* readdir returns NULL for both end-of-directory and error; errno is
         * the only way to tell a truncated listing from a complete one. */
        errno = 0;
        struct dirent *entry = readdir(dir);
        if (entry == NULL) {
            failed = errno != 0;
            break;
        }
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }

        /* count + 1 so the NULL terminator always has a slot of its own. */
        if (count + 1 >= capacity) {
            int oldCap = capacity;
            capacity = JAI_GROW_CAP(oldCap);
            names = JAI_GROW_ARRAY(char *, names, oldCap, capacity);
        }
        names[count++] = jaiStrdup(entry->d_name);
    }
    closedir(dir);

    if (failed) {
        freeNameList(names, count, capacity);
        return NULL;
    }

    /* Shrink to exactly count + 1: the caller frees the array by count and has
     * no way to learn the capacity we grew to. */
    if (capacity != count + 1) {
        names = JAI_GROW_ARRAY(char *, names, capacity, count + 1);
    }
    names[count] = NULL;

    if (outCount != NULL) *outCount = count;
    return names;
}

/* Follows symlinks. mtime is whole seconds (std.time's timestamps are
 * second-grained); every out param is optional and cleared up front, so a
 * false return leaves no stale values. */
bool jaiStatPath(const char *path, int64_t *size, int64_t *mtime, bool *isDir) {
    if (size != NULL) *size = 0;
    if (mtime != NULL) *mtime = 0;
    if (isDir != NULL) *isDir = false;
    if (path == NULL) return false;

    struct stat st;
    if (stat(path, &st) != 0) return false;

    if (size != NULL) *size = (int64_t)st.st_size;
    if (mtime != NULL) *mtime = (int64_t)st.st_mtime;
    if (isDir != NULL) *isDir = S_ISDIR(st.st_mode) != 0;
    return true;
}

static char            gExecPath[JAI_MAX_PATH];
static bool            gExecPathOk   = false;
static pthread_once_t  gExecPathOnce = PTHREAD_ONCE_INIT;

/* A bare argv[0] names something the shell found on PATH; resolve it the same
 * way, taking the first executable regular file. */
static bool searchPath(const char *name, char *out, size_t outSize) {
    const char *pathEnv = getenv("PATH");
    if (pathEnv == NULL || pathEnv[0] == '\0') return false;

    size_t nameLen = strlen(name);
    const char *p = pathEnv;

    for (;;) {
        const char *sep = strchr(p, ':');
        size_t dirLen = sep != NULL ? (size_t)(sep - p) : strlen(p);

        char candidate[JAI_MAX_PATH];
        if (dirLen > 0 && dirLen + 1 + nameLen + 1 <= sizeof candidate) {
            memcpy(candidate, p, dirLen);
            size_t pos = dirLen;
            if (candidate[pos - 1] != '/') candidate[pos++] = '/';
            memcpy(candidate + pos, name, nameLen + 1);

            if (access(candidate, X_OK) == 0 && !jaiPathIsDir(candidate)) {
                return jaiPathAbsolute(out, outSize, candidate);
            }
        }

        if (sep == NULL) return false;
        p = sep + 1;
    }
}

static void execPathInit(void) {
    char raw[JAI_MAX_PATH];
    raw[0] = '\0';

#if defined(__APPLE__)
    uint32_t size = (uint32_t)sizeof raw;
    if (_NSGetExecutablePath(raw, &size) != 0) raw[0] = '\0';   /* would not fit */
#elif defined(__linux__)
    ssize_t n = readlink("/proc/self/exe", raw, sizeof raw - 1);
    raw[n > 0 ? (size_t)n : 0] = '\0';   /* readlink does not NUL-terminate */
#endif

    if (raw[0] == '\0') {
        const char *argv0 = NULL;
#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__NetBSD__) || \
    defined(__OpenBSD__) || defined(__DragonFly__)
        argv0 = getprogname();
#endif
        if (argv0 == NULL || argv0[0] == '\0') return;

        if (strchr(argv0, '/') == NULL) {
            gExecPathOk = searchPath(argv0, gExecPath, sizeof gExecPath);
            return;
        }

        size_t len = strlen(argv0);
        if (len + 1 > sizeof raw) return;
        memcpy(raw, argv0, len + 1);
    }

    /* Both _NSGetExecutablePath and argv[0] can be relative or run through
     * symlinks; callers were promised a resolved absolute path. */
    gExecPathOk = jaiPathAbsolute(gExecPath, sizeof gExecPath, raw);
}

/* Absolute, symlink-resolved path of the running binary, or NULL if unknown.
 * Computed once; the string is owned by this module and valid for the process's life. */
const char *jaiExecutablePath(void) {
    pthread_once(&gExecPathOnce, execPathInit);
    return gExecPathOk ? gExecPath : NULL;
}
