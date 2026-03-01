const { app, BrowserWindow, ipcMain, screen } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

const PROJECT_ROOT = path.join(__dirname, '..');
const CONFIG_PATH = path.join(PROJECT_ROOT, 'config.json');
// gui.py watches this file; when you send a task from Electron we write here and gui.py runs it.
const TASK_PENDING_PATH = path.join(PROJECT_ROOT, 'task_pending.txt');

let dotWindow = null;
let taskWindow = null;
let settingsWindow = null;
let borderWindow = null;
let agentProcess = null;

const DOT_SIZE = 80;
const EXPANDED_SIZE = 480;

// ── Config (reads/writes the same config.json the Python code uses) ──

function loadConfig() {
  try {
    if (fs.existsSync(CONFIG_PATH)) {
      return JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf-8'));
    }
  } catch (_) {}
  return {};
}

function saveConfig(cfg) {
  const merged = { ...loadConfig(), ...cfg };
  fs.writeFileSync(CONFIG_PATH, JSON.stringify(merged, null, 2), 'utf-8');
  return merged;
}

// ── Windows ──

function createDotWindow() {
  const { width: sw, height: sh } = screen.getPrimaryDisplay().workAreaSize;

  dotWindow = new BrowserWindow({
    width: DOT_SIZE,
    height: DOT_SIZE,
    x: Math.round((sw - DOT_SIZE) / 2),
    y: Math.round((sh - DOT_SIZE) / 2),
    frame: false,
    transparent: true,
    alwaysOnTop: true,
    resizable: false,
    skipTaskbar: true,
    hasShadow: false,
    show: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  dotWindow.once('ready-to-show', () => {
    if (dotWindow && !dotWindow.isDestroyed()) dotWindow.show();
  });
  dotWindow.loadFile(path.join(__dirname, 'renderer', 'index.html'));
  dotWindow.setVisibleOnAllWorkspaces(true);
  dotWindow.setMovable(true);

  dotWindow.on('closed', () => {
    dotWindow = null;
    if (taskWindow) taskWindow.close();
    if (settingsWindow) settingsWindow.close();
    if (borderWindow) borderWindow.close();
    killAgent();
    app.quit();
  });
}

function createTaskWindow() {
  if (taskWindow && !taskWindow.isDestroyed()) {
    taskWindow.focus();
    return;
  }

  const { width: sw, height: sh } = screen.getPrimaryDisplay().workAreaSize;
  const w = 700, h = 560;

  taskWindow = new BrowserWindow({
    width: w,
    height: h,
    x: Math.round((sw - w) / 2),
    y: Math.round((sh - h) / 2) - 40,
    frame: false,
    transparent: true,
    alwaysOnTop: true,
    resizable: true,
    skipTaskbar: false,
    hasShadow: false,
    minWidth: 520,
    minHeight: 400,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  taskWindow.loadFile(path.join(__dirname, 'renderer', 'task.html'));
  taskWindow.on('closed', () => { taskWindow = null; });
}

function createSettingsWindow() {
  if (settingsWindow && !settingsWindow.isDestroyed()) {
    settingsWindow.focus();
    return;
  }

  const { width: sw, height: sh } = screen.getPrimaryDisplay().workAreaSize;
  const w = 480, h = 580;

  settingsWindow = new BrowserWindow({
    width: w,
    height: h,
    x: Math.round((sw - w) / 2),
    y: Math.round((sh - h) / 2) - 30,
    frame: false,
    transparent: true,
    alwaysOnTop: true,
    resizable: false,
    skipTaskbar: false,
    hasShadow: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  settingsWindow.loadFile(path.join(__dirname, 'renderer', 'settings.html'));
  settingsWindow.on('closed', () => {
    settingsWindow = null;
    if (dotWindow && !dotWindow.isDestroyed()) {
      dotWindow.show();
      dotWindow.webContents.send('settings-closed');
    }
  });
}

// ── Rainbow screen border overlay ──

function createBorderWindow() {
  if (borderWindow && !borderWindow.isDestroyed()) return;

  const { width: sw, height: sh } = screen.getPrimaryDisplay().size;

  borderWindow = new BrowserWindow({
    width: sw,
    height: sh,
    x: 0,
    y: 0,
    frame: false,
    transparent: true,
    alwaysOnTop: true,
    resizable: false,
    skipTaskbar: true,
    hasShadow: false,
    focusable: false,
    webPreferences: {
      contextIsolation: false,
      nodeIntegration: true,
    },
  });

  borderWindow.loadFile(path.join(__dirname, 'renderer', 'border.html'));
  borderWindow.setIgnoreMouseEvents(true);
  borderWindow.setVisibleOnAllWorkspaces(true);
  borderWindow.on('closed', () => { borderWindow = null; });
}

function showBorder() {
  if (!borderWindow || borderWindow.isDestroyed()) createBorderWindow();
  if (borderWindow) {
    borderWindow.show();
    borderWindow.webContents.send('show-border');
  }
}

function hideBorder() {
  if (borderWindow && !borderWindow.isDestroyed()) {
    borderWindow.webContents.send('hide-border');
    setTimeout(() => {
      if (borderWindow && !borderWindow.isDestroyed()) borderWindow.hide();
    }, 700);
  }
}

// ── Agent process (spawns agent_backend.py — headless, no Python GUI) ──
// Hides task + dot windows before spawn so screenshots don’t include the app; shows them again on exit.

function sendToTask(data) {
  if (taskWindow && !taskWindow.isDestroyed()) {
    taskWindow.webContents.send('log-update', data);
  }
}

function showAgentWindows() {
  if (taskWindow && !taskWindow.isDestroyed()) taskWindow.show();
  if (dotWindow && !dotWindow.isDestroyed()) dotWindow.show();
}

function hideAgentWindows() {
  if (taskWindow && !taskWindow.isDestroyed()) taskWindow.hide();
  if (dotWindow && !dotWindow.isDestroyed()) dotWindow.hide();
}

// No process spawn: you run gui.py yourself. We just write the task to a file gui.py watches.
function startAgent(task) {
  try {
    fs.writeFileSync(TASK_PENDING_PATH, (task || '').trim() + '\n', 'utf8');
    sendToTask({ type: 'log', msg: 'Task sent to gui.py. Make sure gui.py is running.', tag: 'info' });
    sendToTask({ type: 'done', message: 'Sent' });
  } catch (e) {
    sendToTask({ type: 'log', msg: 'Failed to write task file: ' + e.message, tag: 'error' });
    sendToTask({ type: 'done', message: 'Error' });
  }
}

function ensureAgentProcess() {
  if (agentProcess) return;
  const venvExe = path.join(PROJECT_ROOT, 'venv', 'Scripts', 'python.exe');
  const dotVenvExe = path.join(PROJECT_ROOT, '.venv', 'Scripts', 'python.exe');
  const script = path.join(PROJECT_ROOT, 'gui.py');
  const exe = fs.existsSync(dotVenvExe) ? dotVenvExe : (fs.existsSync(venvExe) ? venvExe : 'python');
  agentProcess = spawn(exe, [script], {
    cwd: PROJECT_ROOT,
    env: { ...process.env },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  agentProcess.stdout.on('data', (buf) => { process.stdout.write(buf); });
  agentProcess.stderr.on('data', (buf) => { process.stderr.write(buf); });
  agentProcess.on('close', (code) => { agentProcess = null; });
}

function killAgent() {
  if (agentProcess) {
    agentProcess.kill();
    agentProcess = null;
  }
  try {
    if (fs.existsSync(TASK_PENDING_PATH)) fs.unlinkSync(TASK_PENDING_PATH);
  } catch (_) {}
}

// ── IPC ──

ipcMain.on('dot-expand', () => {
  if (!dotWindow) return;
  const { width: sw, height: sh } = screen.getPrimaryDisplay().workAreaSize;
  const bounds = dotWindow.getBounds();
  const cx = bounds.x + bounds.width / 2;
  const cy = bounds.y + bounds.height / 2;
  let nx = Math.round(cx - EXPANDED_SIZE / 2);
  let ny = Math.round(cy - EXPANDED_SIZE / 2);
  nx = Math.max(0, Math.min(nx, sw - EXPANDED_SIZE));
  ny = Math.max(0, Math.min(ny, sh - EXPANDED_SIZE));
  dotWindow.setBounds({ x: nx, y: ny, width: EXPANDED_SIZE, height: EXPANDED_SIZE });
});

ipcMain.on('dot-collapse', () => {
  if (!dotWindow) return;
  const { width: sw, height: sh } = screen.getPrimaryDisplay().workAreaSize;
  const bounds = dotWindow.getBounds();
  const cx = bounds.x + bounds.width / 2;
  const cy = bounds.y + bounds.height / 2;
  let nx = Math.round(cx - DOT_SIZE / 2);
  let ny = Math.round(cy - DOT_SIZE / 2);
  nx = Math.max(0, Math.min(nx, sw - DOT_SIZE));
  ny = Math.max(0, Math.min(ny, sh - DOT_SIZE));
  dotWindow.setBounds({ x: nx, y: ny, width: DOT_SIZE, height: DOT_SIZE });
});

ipcMain.on('drag-dot', (_e, dx, dy) => {
  if (!dotWindow) return;
  const [x, y] = dotWindow.getPosition();
  dotWindow.setPosition(x + dx, y + dy);
});

ipcMain.on('open-task', () => createTaskWindow());
ipcMain.on('open-settings', () => {
  createSettingsWindow();
});

ipcMain.on('icons-hidden', () => {
  if (dotWindow && !dotWindow.isDestroyed()) dotWindow.hide();
});

ipcMain.on('show-border', () => showBorder());
ipcMain.on('hide-border', () => hideBorder());

ipcMain.on('quit-app', () => {
  if (borderWindow && !borderWindow.isDestroyed()) borderWindow.close();
  if (taskWindow && !taskWindow.isDestroyed()) taskWindow.close();
  if (settingsWindow && !settingsWindow.isDestroyed()) settingsWindow.close();
  if (dotWindow && !dotWindow.isDestroyed()) dotWindow.close();
  killAgent();
  app.quit();
});

ipcMain.on('close-window', (event) => {
  const win = BrowserWindow.fromWebContents(event.sender);
  if (win) win.close();
});

ipcMain.handle('get-config', () => loadConfig());
ipcMain.handle('save-config', (_e, cfg) => {
  const saved = saveConfig(cfg);
  if (dotWindow && !dotWindow.isDestroyed() && saved.display_contrast != null) {
    dotWindow.webContents.send('set-contrast', saved.display_contrast);
  }
  return saved;
});

ipcMain.on('start-agent', (_e, task) => startAgent(task));

ipcMain.on('stop-agent', () => {
  killAgent();
  sendToTask({ type: 'done', message: 'Stopped' });
});

// ── App lifecycle ──

app.whenReady().then(() => {
  createBorderWindow();
  createDotWindow();
  ensureAgentProcess();
});

app.on('window-all-closed', () => {
  killAgent();
  app.quit();
});
