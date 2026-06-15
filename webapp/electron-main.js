// Electron main process for Virtual Try-On.
//
// Modes:
//   - dev:   the caller has already started `next dev` on localhost:3000 (via
//            the `electron:dev` script). We just open a kiosk window pointing
//            at it.
//   - prod:  spawn `next start` ourselves as a child process and wait for it
//            to be ready, then open the window. Used by the packaged app.
//
// Window behaviour:
//   - Fullscreen kiosk. No menu bar. Camera permission auto-granted.
//   - Reload-on-error so a transient backend hiccup doesn't brick the kiosk.

const { app, BrowserWindow, session } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const http = require('http');

const PORT = process.env.PORT || '3000';
const DEV = process.env.ELECTRON_DEV === '1';
const SERVER_URL = `http://localhost:${PORT}`;

let serverProc = null;
let win = null;

function waitForServer(url, timeoutMs = 30_000) {
  const start = Date.now();
  return new Promise((resolve, reject) => {
    const tick = () => {
      const req = http.get(url, (res) => {
        res.resume();
        resolve();
      });
      req.on('error', () => {
        if (Date.now() - start > timeoutMs) return reject(new Error(`Server at ${url} not ready after ${timeoutMs}ms`));
        setTimeout(tick, 250);
      });
      req.setTimeout(2000, () => req.destroy());
    };
    tick();
  });
}

function startServerIfNeeded() {
  if (DEV) return Promise.resolve();   // user runs `next dev` themselves
  // Production: spawn `next start` from this app directory.
  const cwd = __dirname;
  serverProc = spawn(process.execPath, [path.join(cwd, 'node_modules', 'next', 'dist', 'bin', 'next'), 'start', '-p', PORT], {
    cwd,
    env: { ...process.env, NODE_ENV: 'production' },
    stdio: 'inherit',
  });
  serverProc.on('exit', (code) => {
    console.log(`[electron] next start exited with code ${code}`);
    app.quit();
  });
  return waitForServer(SERVER_URL);
}

function createWindow() {
  win = new BrowserWindow({
    width: 1280,
    height: 800,
    fullscreen: true,
    backgroundColor: '#000000',
    autoHideMenuBar: true,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      // Camera + microphone are granted via the permission handler below; no
      // preload required for the current photo-booth UX.
    },
  });
  // No menu — kiosk mode
  win.setMenu(null);
  win.loadURL(SERVER_URL);

  // Auto-recover from blank-screen if the dev server hiccups
  win.webContents.on('render-process-gone', (_e, details) => {
    console.warn('[electron] render gone:', details.reason);
    setTimeout(() => win?.loadURL(SERVER_URL), 1000);
  });
  win.webContents.on('did-fail-load', (_e, code, desc) => {
    console.warn(`[electron] did-fail-load: ${code} ${desc}`);
    setTimeout(() => win?.loadURL(SERVER_URL), 1500);
  });
}

app.whenReady().then(async () => {
  // Auto-grant camera permission so the kiosk doesn't show a prompt.
  session.defaultSession.setPermissionRequestHandler((_wc, permission, cb) => {
    if (permission === 'media' || permission === 'mediaKeySystem') return cb(true);
    cb(false);
  });

  try {
    await startServerIfNeeded();
  } catch (e) {
    console.error('[electron] server failed to start:', e);
    app.quit();
    return;
  }
  createWindow();
});

app.on('window-all-closed', () => {
  if (serverProc) {
    try { serverProc.kill(); } catch (_) {}
  }
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});
