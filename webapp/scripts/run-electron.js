#!/usr/bin/env node
// Launch Electron with ELECTRON_RUN_AS_NODE explicitly unset.
//
// Some shells (and some MCP/dev environments) export ELECTRON_RUN_AS_NODE=1
// globally, which forces every Electron binary to behave as a plain Node
// runtime. That makes `require('electron')` return a path string instead of
// the API object, and the main process script crashes immediately. We strip
// that variable before spawning so Electron always runs as a GUI app.
const { spawn } = require('child_process');
const path = require('path');

const env = { ...process.env };
delete env.ELECTRON_RUN_AS_NODE;

const electron = require('electron');
const appPath = path.resolve(__dirname, '..');

const child = spawn(electron, [appPath], { env, stdio: 'inherit' });
child.on('exit', (code, signal) => {
  if (code !== null) process.exit(code);
  if (signal) process.exit(1);
});
process.on('SIGINT',  () => child.kill('SIGINT'));
process.on('SIGTERM', () => child.kill('SIGTERM'));
