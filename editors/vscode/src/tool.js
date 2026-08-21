// Talking to the `jaithon` binary.

const vscode = require('vscode');
const fs = require('fs');
const os = require('os');
const path = require('path');
const crypto = require('crypto');
const { execFile } = require('child_process');

const MAX_BUFFER = 32 * 1024 * 1024;

function config() {
    return vscode.workspace.getConfiguration('jaithon');
}

function workspaceDir(document) {
    if (document) {
        const folder = vscode.workspace.getWorkspaceFolder(document.uri);
        if (folder) return folder.uri.fsPath;
        if (document.uri.scheme === 'file') return path.dirname(document.uri.fsPath);
    }
    return vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || process.cwd();
}

function binary(document) {
    const root = workspaceDir(document);
    const configured = config().get('path');

    if (configured && configured !== 'jaithon') {
        return configured
            .replace(/\$\{workspaceFolder\}/g, root)
            .replace(/^~(?=$|\/)/, os.homedir());
    }
    const local = path.join(root, 'jaithon');
    try {
        fs.accessSync(local, fs.constants.X_OK);
        return local;
    } catch {
        return 'jaithon';
    }
}

let missingBinaryReported = false;

function resetBinaryWarning() {
    missingBinaryReported = false;
}

function reportMissingBinary(document) {
    if (missingBinaryReported) return;
    missingBinaryReported = true;
    const settings = 'Open Settings';
    vscode.window.showErrorMessage(
        `Jaithon: '${binary(document)}' not found. Build it with 'make', or set "jaithon.path".`,
        settings,
    ).then((choice) => {
        if (choice === settings) {
            vscode.commands.executeCommand('workbench.action.openSettings', 'jaithon.path');
        }
    });
}

/**
 * Run the compiler. Never rejects: a failed spawn and a non-zero exit are both
 * results, because every caller wants to inspect the output either way.
 *
 * @returns {Promise<{code:number, stdout:string, stderr:string, spawnFailed:boolean}>}
 */
function run(args, options = {}) {
    const { cwd, document, token, timeout = 20000 } = options;
    return new Promise((resolve) => {
        const child = execFile(
            binary(document),
            args,
            { cwd: cwd || workspaceDir(document), maxBuffer: MAX_BUFFER, timeout },
            (error, stdout, stderr) => {
                const spawnFailed = Boolean(error && (error.code === 'ENOENT' || error.code === 'EACCES'));
                if (spawnFailed) reportMissingBinary(document);
                resolve({
                    code: error && typeof error.code === 'number' ? error.code : 0,
                    stdout: stdout || '',
                    stderr: stderr || '',
                    spawnFailed,
                });
            },
        );
        if (token) {
            token.onCancellationRequested(() => {
                try { child.kill(); } catch { }
            });
        }
    });
}

// ---------------------------------------------------------------------------
// Search path
// ---------------------------------------------------------------------------

function extraSearchDirs(document) {
    const root = workspaceDir(document);
    const dirs = [];
    for (const entry of config().get('includePaths') || []) {
        dirs.push(path.isAbsolute(entry) ? entry : path.join(root, entry));
    }
    return dirs;
}

const JAI_PACKAGE_MANIFEST = 'jaithon.package.json';

/**
 * The `src` dir of every subdirectory of `packagesDir` that has a package
 * manifest — e.g. `packages/jaicv/jaithon.package.json` makes `jaicv` a
 * module root at `packages/jaicv/src`. Mirrors addPackageSourceDirs() in
 * src/runtime/modules/module_path.c, which is how `import jaicv` actually
 * resolves at runtime.
 */
function packageSourceDirs(packagesDir) {
    let entries;
    try { entries = fs.readdirSync(packagesDir, { withFileTypes: true }); } catch { return []; }

    const names = entries
        .filter((entry) => entry.isDirectory() && !entry.name.startsWith('.'))
        .map((entry) => entry.name)
        .sort();

    const dirs = [];
    for (const name of names) {
        const manifest = path.join(packagesDir, name, JAI_PACKAGE_MANIFEST);
        try {
            if (fs.statSync(manifest).isFile()) dirs.push(path.join(packagesDir, name, 'src'));
        } catch { }
    }
    return dirs;
}

function libraryDirs(document) {
    const exe = binary(document);
    const dirs = [];

    let base = null;
    if (exe.includes(path.sep)) {
        base = path.dirname(path.resolve(workspaceDir(document), exe));
    } else {
        for (const dir of (process.env.PATH || '').split(path.delimiter)) {
            const candidate = path.join(dir, exe);
            try {
                fs.accessSync(candidate, fs.constants.X_OK);
                base = dir;
                break;
            } catch { }
        }
    }
    if (base) {
        dirs.push(path.join(base, 'lib'),
                  path.join(base, '..', 'lib'),
                  path.join(base, '..', 'share', 'jaithon', 'lib'),
                  path.join(base, '..', 'share', 'jaithon'));
        dirs.push(...packageSourceDirs(path.join(base, 'packages')));
        dirs.push(...packageSourceDirs(path.join(base, '..', 'packages')));
        dirs.push(...packageSourceDirs(path.join(base, '..', 'share', 'jaithon', 'packages')));
    }
    dirs.push('/usr/local/share/jaithon/lib', '/usr/local/share/jaithon',
              '/opt/homebrew/share/jaithon/lib', '/opt/homebrew/share/jaithon');
    dirs.push(...packageSourceDirs('/usr/local/share/jaithon/packages'));
    dirs.push(...packageSourceDirs('/opt/homebrew/share/jaithon/packages'));

    for (const entry of (process.env.JAITHON_PATH || '').split(':')) {
        if (entry) dirs.unshift(entry);
    }
    return dirs.filter((dir) => { try { return fs.statSync(dir).isDirectory(); } catch { return false; } });
}

function searchPath(document, fileDir) {
    const seen = new Set();
    const out = [];
    for (const dir of [fileDir, workspaceDir(document), ...extraSearchDirs(document), ...libraryDirs(document)]) {
        if (!dir) continue;
        const resolved = path.resolve(dir);
        if (seen.has(resolved)) continue;
        seen.add(resolved);
        out.push(resolved);
    }
    return out;
}

function includeArgs(document, fileDir) {
    const args = [];
    for (const dir of [fileDir, workspaceDir(document), ...extraSearchDirs(document)]) {
        if (dir) args.push('-I', dir);
    }
    return args;
}

function resolveModule(dotted, fromFile, document) {
    let dots = 0;
    while (dots < dotted.length && dotted[dots] === '.') dots++;
    const relative = dotted.slice(dots).split('.').join(path.sep);
    if (!relative) return null;

    let dirs;
    if (dots > 0) {
        let base = path.dirname(fromFile);
        for (let i = 1; i < dots; i++) base = path.dirname(base);
        dirs = [base];
    } else {
        dirs = searchPath(document, path.dirname(fromFile));
    }

    for (const dir of dirs) {
        for (const candidate of [path.join(dir, `${relative}.jai`), path.join(dir, relative, 'mod.jai')]) {
            try {
                if (fs.statSync(candidate).isFile()) return candidate;
            } catch { }
        }
    }
    return null;
}

// ---------------------------------------------------------------------------
// Temporary mirrors
// ---------------------------------------------------------------------------

let mirrorRoot = null;

function mirrorDir() {
    if (!mirrorRoot) {
        mirrorRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'jaithon-vscode-'));
    }
    return mirrorRoot;
}

function snapshot(document) {
    const real = document.uri.fsPath;
    const dir = path.dirname(real);

    if (document.uri.scheme === 'file' && !document.isDirty) {
        return { path: real, mirrored: false, dir, dispose() {} };
    }

    const stamp = crypto.createHash('sha1').update(document.uri.toString()).digest('hex').slice(0, 8);
    const scratch = path.join(mirrorDir(), stamp);
    fs.mkdirSync(scratch, { recursive: true });
    const target = path.join(scratch, path.basename(real) || 'buffer.jai');
    fs.writeFileSync(target, document.getText(), 'utf8');

    return {
        path: target,
        mirrored: true,
        dir,
        dispose() { try { fs.unlinkSync(target); } catch { } },
    };
}

function cleanup() {
    if (!mirrorRoot) return;
    try { fs.rmSync(mirrorRoot, { recursive: true, force: true }); } catch { }
    mirrorRoot = null;
}

async function version(document) {
    const result = await run(['version'], { document });
    if (result.spawnFailed) return null;
    const match = /(\d+\.\d+\.\d+)/.exec(result.stdout + result.stderr);
    return match ? match[1] : (result.stdout.trim() || null);
}

module.exports = {
    config, workspaceDir, binary, run, resetBinaryWarning,
    searchPath, includeArgs, resolveModule,
    snapshot, cleanup, version,
};
