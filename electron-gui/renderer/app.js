const mainOrb = document.getElementById('mainOrb');
const subDotsRing = document.getElementById('subDotsRing');
const subDots = document.querySelectorAll('.sub-dot');
const dotContainer = document.getElementById('dotContainer');

let expanded = false;

function applyIconContrast(value) {
  const num = value == null ? 50 : Math.max(0, Math.min(100, Number(value)));
  const contrast = 0.5 + (num / 100) * 1.3;
  if (dotContainer) dotContainer.style.filter = `contrast(${contrast})`;
}

applyIconContrast(50);
if (window.api.onContrastChange) window.api.onContrastChange(applyIconContrast);
let dragging = false;
let didDragThisGesture = false;
let dragStart = null;
const DRAG_THRESHOLD = 5;

const SUB_DOT_POSITIONS = [
  { angle: -90 },
  { angle: 30 },
  { angle: 150 },
];

const RING_RADIUS = 95;

function positionSubDots(show) {
  subDots.forEach((dot, i) => {
    const angleDeg = SUB_DOT_POSITIONS[i].angle;
    const angleRad = (angleDeg * Math.PI) / 180;
    const x = Math.cos(angleRad) * RING_RADIUS;
    const y = Math.sin(angleRad) * RING_RADIUS;

    dot.style.left = `calc(50% + ${x}px)`;
    dot.style.top = `calc(50% + ${y}px)`;

    if (show) {
      setTimeout(() => dot.classList.add('show'), 80 * i);
    } else {
      dot.classList.remove('show');
    }
  });
}

function doExpand() {
  if (expanded) return;
  expanded = true;
  window.api.expandDot();
  window.api.showBorder();

  mainOrb.classList.remove('idle-pulse');
  subDotsRing.classList.add('visible');
  positionSubDots(true);
}

function doCollapse() {
  if (!expanded) return;
  expanded = false;
  collapseTaskBar();

  window.api.hideBorder();
  positionSubDots(false);
  subDotsRing.classList.remove('visible');

  setTimeout(() => {
    window.api.collapseDot();
    mainOrb.classList.add('idle-pulse');
  }, 350);
}

/* ── Dragging (mousedown → mousemove → mouseup) — works when collapsed or expanded ── */
mainOrb.addEventListener('mousedown', (e) => {
  dragging = false;
  didDragThisGesture = false;
  dragStart = { x: e.screenX, y: e.screenY };
});

window.addEventListener('mousemove', (e) => {
  if (!dragStart) return;
  const dx = e.screenX - dragStart.x;
  const dy = e.screenY - dragStart.y;
  if (!dragging && (Math.abs(dx) > DRAG_THRESHOLD || Math.abs(dy) > DRAG_THRESHOLD)) {
    dragging = true;
    didDragThisGesture = true;
  }
  if (dragging) {
    window.api.dragDot(dx, dy);
    dragStart = { x: e.screenX, y: e.screenY };
  }
});

window.addEventListener('mouseup', (e) => {
  if (!dragging && dragStart && !mainOrb.contains(e.target)) {
    expanded ? doCollapse() : doExpand();
  }
  dragStart = null;
  setTimeout(() => { dragging = false; didDragThisGesture = false; }, 10);
});

let clickTimer = null;
const DOUBLE_CLICK_MS = 300;

mainOrb.addEventListener('dblclick', (e) => {
  e.stopPropagation();
  e.preventDefault();
  if (clickTimer) { clearTimeout(clickTimer); clickTimer = null; }
  window.api.quitApp();
});

const taskDot = document.getElementById('sub-task');
const taskInput = document.getElementById('taskInput');

function isTaskExpanded() {
  return taskDot && taskDot.classList.contains('task-expanded');
}

const TASK_ANGLE_RAD = 30 * Math.PI / 180;
const TASK_EXPANDED_RADIUS = 115;

function expandTaskBar() {
  if (!taskDot) return;
  taskDot.classList.add('task-expanded');
  const x = Math.cos(TASK_ANGLE_RAD) * TASK_EXPANDED_RADIUS;
  const y = Math.sin(TASK_ANGLE_RAD) * TASK_EXPANDED_RADIUS;
  taskDot.style.left = `calc(50% + ${x}px)`;
  taskDot.style.top = `calc(50% + ${y}px)`;
  setTimeout(() => taskInput && taskInput.focus(), 100);
}

function collapseTaskBar() {
  if (!taskDot) return;
  taskDot.classList.remove('task-expanded');
  const x = Math.cos(TASK_ANGLE_RAD) * RING_RADIUS;
  const y = Math.sin(TASK_ANGLE_RAD) * RING_RADIUS;
  taskDot.style.left = `calc(50% + ${x}px)`;
  taskDot.style.top = `calc(50% + ${y}px)`;
  if (taskInput) taskInput.blur();
}

taskInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    const text = (taskInput.value || '').trim();
    if (text) {
      window.api.startAgent(text);
      taskInput.value = '';
      collapseTaskBar();
    }
    e.preventDefault();
  }
  if (e.key === 'Escape') {
    collapseTaskBar();
    e.preventDefault();
  }
});

function animateOutThenSettings() {
  collapseTaskBar();
  expanded = false;

  // Scatter sub-dots
  subDots.forEach((d, i) => {
    d.classList.remove('show');
    setTimeout(() => {
      d.classList.add('scatter');
    }, i * 40);
  });
  subDotsRing.classList.remove('visible');

  // Shrink main orb
  mainOrb.style.transition = 'transform 0.32s ease, opacity 0.32s ease';
  mainOrb.style.transform = 'scale(0)';
  mainOrb.style.opacity = '0';

  setTimeout(() => {
    window.api.hideIcons();
    window.api.openSettings();
    // Silently reset for when the window comes back
    setTimeout(() => {
      subDots.forEach(d => { d.classList.remove('scatter'); d.classList.remove('show'); });
      mainOrb.style.transition = '';
      mainOrb.style.transform = '';
      mainOrb.style.opacity = '';
      mainOrb.classList.add('idle-pulse');
    }, 80);
    setTimeout(() => window.api.collapseDot(), 50);
  }, 370);
}

if (window.api.onSettingsClosed) {
  window.api.onSettingsClosed(() => {
    // Expand dot window back to show sub-dots
    doExpand();

    // Animate main orb popping back in
    mainOrb.style.transition = 'none';
    mainOrb.style.transform = 'scale(0.5)';
    mainOrb.style.opacity = '0';
    mainOrb.classList.remove('idle-pulse');
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        mainOrb.style.transition = 'transform 0.5s cubic-bezier(0.34, 1.56, 0.64, 1), opacity 0.4s ease';
        mainOrb.style.transform = 'scale(1)';
        mainOrb.style.opacity = '1';
        setTimeout(() => {
          mainOrb.style.transition = '';
          mainOrb.style.transform = '';
          mainOrb.style.opacity = '';
        }, 520);
      });
    });

    // Animate sub-dots popping in with stagger
    subDots.forEach((d, i) => {
      d.classList.remove('scatter');
      d.classList.remove('show');
      d.classList.remove('popin');
      setTimeout(() => {
        d.classList.add('popin');
        setTimeout(() => {
          d.classList.remove('popin');
          d.classList.add('show');
        }, 460);
      }, 80 + i * 60);
    });
  });
}

subDots.forEach((dot) => {
  dot.addEventListener('click', (e) => {
    e.stopPropagation();
    const action = dot.dataset.action;
    if (action === 'task') {
      if (isTaskExpanded()) {
        collapseTaskBar();
      } else {
        expandTaskBar();
      }
      return;
    }
    if (action === 'settings') {
      animateOutThenSettings();
    } else {
      doCollapse();
      setTimeout(() => {
        if (action === 'mic') {
          const icon = dot.querySelector('.sub-dot-icon');
          icon.textContent = icon.textContent === '🎙' ? '🔴' : '🎙';
        }
      }, 400);
    }
  });
});

document.getElementById('dotContainer').addEventListener('click', (e) => {
  if (expanded && e.target.id === 'dotContainer') {
    collapseTaskBar();
    doCollapse();
  }
});

mainOrb.addEventListener('click', (e) => {
  e.stopPropagation();
  if (didDragThisGesture) return;
  if (clickTimer) { clearTimeout(clickTimer); clickTimer = null; }
  clickTimer = setTimeout(() => {
    clickTimer = null;
    if (expanded) {
      collapseTaskBar();
      doCollapse();
    } else {
      doExpand();
    }
  }, DOUBLE_CLICK_MS);
});

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    if (isTaskExpanded()) {
      collapseTaskBar();
      e.preventDefault();
    } else {
      window.api.quitApp();
    }
  }
});
