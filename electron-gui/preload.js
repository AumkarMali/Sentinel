const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('api', {
  // Dot window
  expandDot: () => ipcRenderer.send('dot-expand'),
  collapseDot: () => ipcRenderer.send('dot-collapse'),

  // Open windows
  openTask: () => ipcRenderer.send('open-task'),
  openSettings: () => ipcRenderer.send('open-settings'),
  closeWindow: () => ipcRenderer.send('close-window'),

  // Drag
  dragDot: (dx, dy) => ipcRenderer.send('drag-dot', dx, dy),

  // Rainbow screen border
  showBorder: () => ipcRenderer.send('show-border'),
  hideBorder: () => ipcRenderer.send('hide-border'),

  // Quit
  quitApp: () => ipcRenderer.send('quit-app'),

  // Config
  getConfig: () => ipcRenderer.invoke('get-config'),
  saveConfig: (cfg) => ipcRenderer.invoke('save-config', cfg),

  // Agent
  startAgent: (task) => ipcRenderer.send('start-agent', task),
  stopAgent: () => ipcRenderer.send('stop-agent'),

  // Receive log stream from main process
  onLogUpdate: (cb) => {
    ipcRenderer.on('log-update', (_e, data) => cb(data));
  },

  // Contrast for icons (dot window)
  onContrastChange: (cb) => {
    ipcRenderer.on('set-contrast', (_e, value) => cb(value));
  },

  // Settings lifecycle
  hideIcons: () => ipcRenderer.send('icons-hidden'),
  onSettingsClosed: (cb) => {
    ipcRenderer.on('settings-closed', () => cb());
  },
});
