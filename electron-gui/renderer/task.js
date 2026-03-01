const taskInput = document.getElementById('taskInput');
const goBtn = document.getElementById('goBtn');
const closeBtn = document.getElementById('closeBtn');
const statusBadge = document.getElementById('statusBadge');
const statusText = document.getElementById('statusText');
const statMode = document.getElementById('statMode');
const statCycles = document.getElementById('statCycles');
const statActions = document.getElementById('statActions');
const actionMove = document.getElementById('actionMove');
const actionEval = document.getElementById('actionEval');
const logArea = document.getElementById('logArea');
const clearLogBtn = document.getElementById('clearLogBtn');
const screenshotArea = document.getElementById('screenshotArea');

let running = false;

// ── Status ──

function setStatus(text, type = 'ready') {
  statusText.textContent = text;
  statusBadge.className = `status-badge ${type}`;
}

function updateStats(data) {
  if (data.mode !== undefined) statMode.textContent = data.mode || '—';
  if (data.cycles !== undefined) statCycles.textContent = data.cycles;
  if (data.actions !== undefined) statActions.textContent = data.actions;
}

function setAction(move, evalText = '') {
  actionMove.textContent = move || '—';
  actionEval.textContent = evalText;
}

// ── Log ──

const TAG_CLASSES = {
  info: 'log-info', error: 'log-error', warning: 'log-warning',
  action: 'log-action', header: 'log-header', thought: 'log-thought',
  dim: 'log-dim', move: 'log-move', piece: 'log-thought',
  board: 'log-dim', result: 'log-move',
};

function appendLog(msg, tag = '') {
  const ts = new Date().toLocaleTimeString('en-US', { hour12: false });
  const line = document.createElement('div');
  line.textContent = `[${ts}] ${msg}`;
  if (tag && TAG_CLASSES[tag]) line.classList.add(TAG_CLASSES[tag]);
  logArea.appendChild(line);
  logArea.scrollTop = logArea.scrollHeight;
}

clearLogBtn.addEventListener('click', () => { logArea.innerHTML = ''; });

// ── Agent control ──

function startAgent() {
  const task = taskInput.value.trim();
  if (!task) {
    appendLog('Enter a task first!', 'error');
    return;
  }

  running = true;
  goBtn.textContent = 'Stop';
  goBtn.className = 'btn btn-stop';
  setStatus('Running...', 'running');
  appendLog(`Starting task: ${task}`, 'header');

  window.api.startAgent(task);
}

function stopAgent() {
  running = false;
  goBtn.textContent = 'Start';
  goBtn.className = 'btn btn-primary';
  setStatus('Stopped', 'stopped');
  appendLog('Agent stopped.', 'warning');
  window.api.stopAgent();
}

// Form submit: Enter or clicking Start sends the task to the agent
const taskForm = document.getElementById('taskForm');
taskForm.addEventListener('submit', (e) => {
  e.preventDefault();
  if (running) stopAgent();
  else startAgent();
});

closeBtn.addEventListener('click', () => {
  if (running) stopAgent();
  window.api.closeWindow();
});

// ── Receive log stream from main process ──

window.api.onLogUpdate((data) => {
  if (data.type === 'log') {
    appendLog(data.msg, data.tag);
  } else if (data.type === 'status') {
    setStatus(data.text, data.badge || 'ready');
  } else if (data.type === 'stats') {
    updateStats(data);
  } else if (data.type === 'action') {
    setAction(data.move, data.eval);
  } else if (data.type === 'screenshot') {
    screenshotArea.innerHTML = `<img src="data:image/png;base64,${data.b64}" alt="screenshot">`;
  } else if (data.type === 'done') {
    running = false;
    goBtn.textContent = 'Start';
    goBtn.className = 'btn btn-primary';
    setStatus(data.message || 'Done', 'ready');
  }
});

taskInput.focus();
taskInput.select();
