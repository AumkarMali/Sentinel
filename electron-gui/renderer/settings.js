const closeBtn = document.getElementById('closeBtn');
const apiKeyInput = document.getElementById('apiKeyInput');
const showKeyBtn = document.getElementById('showKeyBtn');
const modelSelect = document.getElementById('modelSelect');
const confSlider = document.getElementById('confSlider');
const confValue = document.getElementById('confValue');
const depthSlider = document.getElementById('depthSlider');
const depthValue = document.getElementById('depthValue');
const intervalSlider = document.getElementById('intervalSlider');
const intervalValue = document.getElementById('intervalValue');
const clickDelaySlider = document.getElementById('clickDelaySlider');
const clickDelayValue = document.getElementById('clickDelayValue');
const contrastSlider = document.getElementById('contrastSlider');
const contrastValue = document.getElementById('contrastValue');
const contrastPreview = document.getElementById('contrastPreview');
const saveBtn = document.getElementById('saveBtn');

let keyVisible = false;

// ── Load config (same keys as gui.py / agent_backend) ──

async function loadSettings() {
  const cfg = await window.api.getConfig();

  apiKeyInput.value = cfg.gemini_api_key || cfg.api_key || '';
  modelSelect.value = cfg.gemini_model || cfg.model || 'gemini-2.0-flash';

  if (cfg.yolo_confidence != null) confSlider.value = cfg.yolo_confidence;
  if (cfg.stockfish_depth != null) depthSlider.value = cfg.stockfish_depth;
  if (cfg.scan_interval != null) intervalSlider.value = cfg.scan_interval;
  if (cfg.click_delay != null) clickDelaySlider.value = cfg.click_delay;
  if (cfg.display_contrast != null) contrastSlider.value = cfg.display_contrast;

  updateAllSliderValues();
  updateContrastPreview();
}

loadSettings();

// ── Show/hide API key ──

showKeyBtn.addEventListener('click', () => {
  keyVisible = !keyVisible;
  apiKeyInput.type = keyVisible ? 'text' : 'password';
  showKeyBtn.textContent = keyVisible ? 'Hide' : 'Show';
});

// ── Sliders ──

function bindSlider(slider, display, suffix = '') {
  slider.addEventListener('input', () => {
    display.textContent = slider.value + suffix;
  });
}

function updateAllSliderValues() {
  confValue.textContent = confSlider.value;
  depthValue.textContent = depthSlider.value;
  intervalValue.textContent = intervalSlider.value;
  clickDelayValue.textContent = clickDelaySlider.value;
  contrastValue.textContent = contrastSlider.value + '%';
}

bindSlider(confSlider, confValue);
bindSlider(depthSlider, depthValue);
bindSlider(intervalSlider, intervalValue);
bindSlider(clickDelaySlider, clickDelayValue);
bindSlider(contrastSlider, contrastValue, '%');

contrastSlider.addEventListener('input', updateContrastPreview);

function updateContrastPreview() {
  const v = parseInt(contrastSlider.value);
  const lo = Math.round(30 + (v / 100) * 80);
  const hi = Math.round(80 + (v / 100) * 175);
  const darkEl = contrastPreview.querySelector('.cp-dark');
  const lightEl = contrastPreview.querySelector('.cp-light');
  darkEl.style.background = `rgb(${lo},${lo},${lo})`;
  darkEl.style.color = `rgb(${hi},${hi},${hi})`;
  lightEl.style.background = `rgb(${hi},${hi},${hi})`;
  lightEl.style.color = `rgb(${lo},${lo},${lo})`;
}

updateContrastPreview();

// ── Save ──

saveBtn.addEventListener('click', async () => {
  await window.api.saveConfig({
    gemini_api_key: apiKeyInput.value.trim(),
    gemini_model: modelSelect.value,
    yolo_confidence: parseFloat(confSlider.value),
    stockfish_depth: parseInt(depthSlider.value),
    scan_interval: parseFloat(intervalSlider.value),
    click_delay: parseFloat(clickDelaySlider.value),
    display_contrast: parseInt(contrastSlider.value),
  });

  saveBtn.textContent = 'Saved ✓';
  saveBtn.style.opacity = '0.7';
  setTimeout(() => {
    saveBtn.textContent = 'Save';
    saveBtn.style.opacity = '1';
  }, 1200);
});

// ── Close ──

closeBtn.addEventListener('click', () => window.api.closeWindow());
