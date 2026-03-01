# Project Overview

---

## Inspiration

> **One agent** that sees your screen, does real tasks (including chess), and improves from your corrections.

---

## What It Does

- **Vision-based task routing** — Decides what to do from the current screen.
- **Screen control** — Click, type, and send keys (PyAutoGUI + PyWinAuto).
- **Chess autoplay** — YOLO piece detection + Stockfish for moves.
- **Feedback learning** — The model follows your rules via `prompt_history.json`.

---

## How We Built It

| Layer        | Tech |
|-------------|------|
| **GUI**     | Tkinter (reference) / Electron frontend |
| **Vision & reasoning** | Gemini API with system instructions for feedback |
| **Automation** | PyAutoGUI, PyWinAuto |
| **Chess**   | YOLOv8, Stockfish, `python-chess` |
| **Learning** | `prompt_history.json` + system-instruction injection |

---

## Challenges

| Issue | Fix |
|-------|-----|
| Thinking-only replies (no action JSON) | JSON retry + fallback parsing |
| Feedback being ignored | Moved corrections into **system instructions** |
| Chess "playing as" / wrong side | Clearer rules + orientation-from-board detection |
| Screenshot / context limits | Trimmed history, single prior screenshot, long-edge resize |

---

## Accomplishments

- **Single agent** for chess + general desktop tasks.
- **Feedback that actually affects behavior** (system-instruction integration).
- **Retry logic** so the loop doesn’t stop on bad or empty API replies.
- **Working Windows pipeline** from vision → actions → learning.

---

## What We Learned

- **System vs user prompt** — Where you put feedback changes how strongly the model follows it.
- **One retry** — Often enough to recover from thinking-only or empty responses.
- **Vision + automation** — Making screen control reliable needs grid/cursor hints and strict action formats.

---

## What’s Next

- Richer feedback and lessons
- Smarter screenshot history
- Multi-monitor / DPI handling
- Voice input
- Saved presets and macros
