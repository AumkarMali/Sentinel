"""
AI Agent - Unified GUI.

Task input + screenshot → Gemini decides:
  - Chess task + board visible → start chess bot (YOLO + Stockfish)
  - Feedback/correction → store for learning
  - Other tasks → pywinauto + keyboard screen control (click, type, key press)

Vision: Gemini only. API key at https://aistudio.google.com/apikey
"""
import os
import sys

# Windows DPI awareness — MUST be set before any GUI/screenshot imports.
# Without this, pyautogui coordinates are in logical (scaled) space while
# screenshots may be physical pixels, causing coordinate mismatches.
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)   # per-monitor aware
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()     # fallback
        except Exception:
            pass

import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import time
import io
import json
import re
import tempfile
import pyautogui
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

from chess_agent import ChessEngine, is_chess_task


def _move_cursor(x: int, y: int) -> bool:
    """Move the mouse cursor to (x, y). Use Windows SetCursorPos when available."""
    x, y = int(x), int(y)
    if sys.platform == "win32":
        try:
            ctypes = __import__("ctypes")
            ok = ctypes.windll.user32.SetCursorPos(x, y)
            return bool(ok)
        except Exception:
            pass
    pyautogui.moveTo(x, y, duration=0.1)
    return True

try:
    from pywinauto_actions import get_ui_elements, execute_action as _execute_pywinauto
    _PYWINAUTO_OK = True
except ImportError:
    _PYWINAUTO_OK = False

# pynput for keyboard (sending keys to focused window)
try:
    from pynput.keyboard import Controller as PynputController, Key as PynputKey
    _PYNPUT_AVAILABLE = True
except ImportError:
    PynputController = PynputKey = None
    _PYNPUT_AVAILABLE = False

# ── config (persisted API key) ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")


def _load_config():
    """Load config.json. Returns dict with gemini_api_key, gemini_model."""
    try:
        if os.path.isfile(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def _save_config(gemini_api_key: str = "", gemini_model: str = ""):
    """Save Gemini settings to config.json."""
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "gemini_api_key": (gemini_api_key or "").strip(),
                "gemini_model": (gemini_model or "").strip(),
            }, f, indent=2)
    except OSError:
        pass


# ── prompt history (learning from mistakes) ──
PROMPT_HISTORY_PATH = os.path.join(BASE_DIR, "prompt_history.json")
MAX_HISTORY_ENTRIES = 100


def _load_prompt_history():
    """Load prompt history for learning. Returns list of entries."""
    try:
        if os.path.isfile(PROMPT_HISTORY_PATH):
            with open(PROMPT_HISTORY_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        pass
    return []


def _save_prompt_entry(entry: dict):
    """Append one entry to prompt history. Trims to MAX_HISTORY_ENTRIES."""
    history = _load_prompt_history()
    history.append(entry)
    if len(history) > MAX_HISTORY_ENTRIES:
        history = history[-MAX_HISTORY_ENTRIES:]
    try:
        with open(PROMPT_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except OSError:
        pass


def _parse_json(text: str):
    """Parse JSON from model response. Tolerates leading/trailing text and single quotes. Returns dict or None."""
    text = text.strip()
    # If model wrote reasoning before the JSON, find the JSON object (prefer last { so we get the real payload)
    start_candidates = [i for i, c in enumerate(text) if c == "{"]
    if not start_candidates:
        return None
    # Try from last { first (model often writes reasoning then JSON)
    for start in reversed(start_candidates):
        depth = 0
        chunk = None
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    chunk = text[start : i + 1]
                    break
        if chunk is None:
            continue
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            pass
        fixed = re.sub(r"([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:\s*)", r'\1"\2"\3', chunk)
        fixed = re.sub(r",\s*}", "}", fixed)
        fixed = re.sub(r",\s*]", "]", fixed)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        fixed = re.sub(r"'([^']*)'\s*:", r'"\1":', chunk)
        fixed = re.sub(r":\s*'([^']*)'", r': "\1"', fixed)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            continue
    # Fallback: original first-{ extraction and fixes
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                text = text[start : i + 1]
                break
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fix common LLM output: unquoted keys (only after { or ,), trailing commas
    fixed = re.sub(r"([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:\s*)", r'\1"\2"\3', text)
    fixed = re.sub(r",\s*}", "}", fixed)  # trailing comma before }
    fixed = re.sub(r",\s*]", "]", fixed)  # trailing comma before ]
    # Fix single-quoted keys/values (simple: replace ' with " where it looks like JSON)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    # Last resort: try replacing ' with " for keys (key':  -> "key":)
    fixed = re.sub(r"'([^']*)'\s*:", r'"\1":', text)
    fixed = re.sub(r":\s*'([^']*)'", r': "\1"', fixed)
    fixed = re.sub(r",\s*}", "}", fixed)
    fixed = re.sub(r",\s*]", "]", fixed)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        return None


# Router system prompt (tail) — used by both GUI and process worker
ROUTER_SYSTEM_TAIL = """You control the screen. Reply with <thinking>brief reasoning + grid coordinates if clicking</thinking> then ONE JSON object.

━━ COORDINATE GRID ━━
The screenshot has grid lines every 50px labeled along top (x) and left (y).
To click a target:
1. Find the grid line just LEFT of the target → base x. Estimate pixels right → x = base + offset.
2. Find the grid line just ABOVE the target → base y. Estimate pixels down → y = base + offset.
3. Use CLICK_XY with those coordinates — it moves AND clicks in one step.
NEVER guess coordinates. Read them from the grid lines you see in the image.

If a UI element name is listed below, prefer CLICK by name (more reliable):
{"action": "CLICK", "parameters": {"element": "Play"}}

Chess: pieces at BOTTOM = user's color. Light at bottom → "white", dark → "black".
- Board visible: {"action": "start_chess", "parameters": {"reason": "...", "playing_as": "white"}}
- No board: {"action": "comment", "message": "No chess board visible."}

Actions (one per turn):
- CLICK_XY (move+click at grid coordinates): {"action": "CLICK_XY", "parameters": {"x": 400, "y": 300}}
- CLICK by element name: {"action": "CLICK", "parameters": {"element": "Open"}}
- OPEN_APP (Windows programs ONLY, NOT websites): {"action": "OPEN_APP", "parameters": {"program": "Chrome"}}
- OPEN_URL (websites, browser must be open): {"action": "OPEN_URL", "parameters": {"url": "chess.com"}}
- KEYS: {"action": "KEYS", "parameters": {"keys": ["ctrl", "t"]}}
- TYPE_TEXT: {"action": "TYPE_TEXT", "parameters": {"text": "hello"}}
- TYPE_IN: {"action": "TYPE_IN", "parameters": {"element": "Search", "text": "query"}}
- TASK_COMPLETE: {"action": "TASK_COMPLETE", "parameters": {"message": "Done"}}
- Observation only: {"action": "comment", "message": "..."}

NEVER use KEYS to open apps. OPEN_APP for programs, OPEN_URL for websites."""


def _parse_router_response(data, text: str, raw_response: str):
    """Given parsed data and raw text, return (action, payload). Used by router and process worker."""
    if data is None:
        raw_preview = (text.strip() or "(empty)")[:120]
        return ("comment", f"Model did not return valid JSON (reply was: '{raw_preview}'). Reply with a single JSON object only, e.g. {{\"action\": \"KEYS\", \"parameters\": {{\"keys\": [\"win\"]}}}}.")
    raw_action = data.get("action") or "comment"
    action = str(raw_action).lower().strip().replace(" ", "_").replace("-", "_")
    a_norm = str(raw_action).upper().strip().replace(" ", "_").replace("-", "_")
    if a_norm == "SCREEN_LOADING":
        return ("screen_loading", None)
    if a_norm in ("CLICK", "CLICK_CURRENT", "CLICK_XY", "MOVE_TO", "DOUBLE_CLICK", "TYPE_IN", "MENU", "KEYS", "KEY_PRESS", "TYPE_TEXT", "TASK_COMPLETE", "OPEN_APP", "OPEN_URL"):
        params = data.get("parameters", data)
        act = "KEYS" if a_norm == "KEY_PRESS" else a_norm
        return ("execute", {"action": act, "parameters": params if isinstance(params, dict) else {}})
    if action == "comment":
        msg = data.get("message") or data.get("reason") or ""
        if isinstance(msg, str) and msg.strip().startswith("{"):
            inner = _parse_json(msg.strip())
            if isinstance(inner, dict):
                ia = (inner.get("action") or "").lower().strip().replace(" ", "_").replace("-", "_")
                if ia == "store_feedback":
                    fb = inner.get("feedback", {})
                    return ("store_feedback", fb if isinstance(fb, dict) else {"raw": str(fb)})
                if ia.upper().replace(" ", "_") in ("CLICK", "CLICK_CURRENT", "CLICK_XY", "MOVE_TO", "DOUBLE_CLICK", "TYPE_IN", "MENU", "KEYS", "KEY_PRESS", "TYPE_TEXT", "TASK_COMPLETE", "OPEN_APP", "OPEN_URL"):
                    params = inner.get("parameters", inner)
                    return ("execute", {"action": ia.upper().replace(" ", "_"), "parameters": params if isinstance(params, dict) else {}})
    if action == "storefeedback":
        action = "store_feedback"
    if action == "keypress":
        action = "keys"
    elif action == "typetext":
        action = "type_text"
    elif action == "taskcomplete":
        action = "task_complete"
    if action == "start_chess":
        reason = data.get("reason", "Chess board detected.")
        playing_as = (data.get("playing_as") or "white").lower()
        if playing_as not in ("white", "black"):
            playing_as = "white"
        return ("start_chess", {"reason": reason, "playing_as": playing_as})
    if action == "store_feedback":
        feedback = data.get("feedback", {})
        if not isinstance(feedback, dict):
            feedback = {"raw": str(feedback)}
        return ("store_feedback", feedback)
    if action in ("click", "click_xy", "double_click", "type_in", "menu", "keys", "type_text", "task_complete", "open_app", "open_url"):
        params = data.get("parameters", data)
        act = "KEYS" if action == "keys" else action.upper().replace(" ", "_")
        return ("execute", {"action": act, "parameters": params if isinstance(params, dict) else {}})
    params = data.get("parameters")
    if isinstance(params, dict) or "parameters" in data:
        a_upper = str(raw_action).upper().replace(" ", "_").replace("-", "_")
        if a_upper in ("CLICK", "CLICK_CURRENT", "CLICK_XY", "MOVE_TO", "DOUBLE_CLICK", "TYPE_IN", "MENU", "KEYS", "KEY_PRESS", "TYPE_TEXT", "TASK_COMPLETE", "OPEN_APP", "OPEN_URL"):
            act = "KEYS" if a_upper == "KEY_PRESS" else a_upper
            return ("execute", {"action": act, "parameters": params if isinstance(params, dict) else {}})
    for start in [i for i, c in enumerate(text) if c == "{"]:
        depth, end = 0, None
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is None:
            continue
        try:
            obj = json.loads(text[start:end])
        except json.JSONDecodeError:
            continue
        a = (obj.get("action") or "").upper().replace(" ", "_").replace("-", "_")
        if a in ("CLICK", "CLICK_CURRENT", "CLICK_XY", "MOVE_TO", "DOUBLE_CLICK", "TYPE_IN", "MENU", "KEYS", "KEY_PRESS", "TYPE_TEXT", "TASK_COMPLETE", "OPEN_APP", "OPEN_URL"):
            p = obj.get("parameters", obj)
            act = "KEYS" if a == "KEY_PRESS" else a
            return ("execute", {"action": act, "parameters": p if isinstance(p, dict) else {}})
        if a == "STORE_FEEDBACK":
            fb = obj.get("feedback", {})
            return ("store_feedback", fb if isinstance(fb, dict) else {"raw": str(fb)})
    for scan_text in (text, raw_response):
        if "store_feedback" not in scan_text and "store_feedback" not in str(data):
            continue
        for start in [i for i, c in enumerate(scan_text) if c == "{"]:
            depth, end = 0, None
            for i in range(start, len(scan_text)):
                if scan_text[i] == "{":
                    depth += 1
                elif scan_text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end is None:
                continue
            try:
                obj = json.loads(scan_text[start:end])
            except json.JSONDecodeError:
                continue
            if (obj.get("action") or "").lower().replace(" ", "_") == "store_feedback":
                fb = obj.get("feedback", {})
                return ("store_feedback", fb if isinstance(fb, dict) else {"raw": str(fb)})
    msg = data.get("message", data.get("reason", text))
    return ("comment", msg or "No message from model.")


def _build_learning_context(max_entries: int = 15):
    """Build a string of past user feedback only (type feedback) for the model."""
    history = _load_prompt_history()
    out = []
    feedback_entries = [e for e in history if e.get("type") == "feedback"][-max_entries:]
    if feedback_entries:
        out.append("LEARNING FROM USER FEEDBACK — you MUST apply these corrections. Ignoring them has caused repeated user corrections:")
        for e in feedback_entries:
            user_msg = (e.get("user_message") or e.get("task") or "?")[:80]
            fb = e.get("feedback", {})
            if isinstance(fb, dict):
                wrong = (fb.get("what_was_wrong") or "").strip()[:400]
                approach = (fb.get("correct_approach") or "").strip()[:400]
                correction = (fb.get("user_correction") or "").strip()[:200]
                block = [f"User: \"{user_msg}\""]
                if correction:
                    block.append(f"Correction: {correction}")
                if wrong:
                    block.append(f"What was wrong: {wrong}")
                if approach:
                    block.append(f"Correct approach: {approach}")
                out.append("\n".join(block))
            else:
                out.append(f"User: \"{user_msg}\" → {str(fb)[:200]}")
        out.append("Apply the correct_approach from the feedback above when making decisions (e.g. for chess, determine playing_as only from the bottom two rows of the board).")
    return "\n".join(out) if out else ""


# ── action executor (mouse + keyboard) ──
def _pynput_key(name):
    """Map key name to pynput Key or single-char string. Returns None if unknown."""
    if not _PYNPUT_AVAILABLE or PynputKey is None:
        return None
    n = str(name).lower()
    m = {
        "win": PynputKey.cmd, "winleft": PynputKey.cmd, "winright": PynputKey.cmd,
        "ctrl": PynputKey.ctrl, "control": PynputKey.ctrl,
        "alt": PynputKey.alt, "shift": PynputKey.shift,
        "enter": PynputKey.enter, "return": PynputKey.enter,
        "tab": PynputKey.tab, "space": PynputKey.space,
        "backspace": PynputKey.backspace, "esc": PynputKey.esc, "escape": PynputKey.esc,
    }
    for i in range(1, 13):
        m[f"f{i}"] = getattr(PynputKey, f"f{i}", None)
    m = {k: v for k, v in m.items() if v is not None}
    if n in m:
        return m[n]
    if len(n) == 1:
        return n
    return None


def _open_app_via_start(element_name: str) -> tuple:
    """Fallback: open app via Run dialog (Win+R). Returns (True, msg) or (False, msg)."""
    return _open_program(element_name)


def _open_program(program_name: str) -> tuple:
    """Open a program instantly: Win+R, type name, Enter. ONLY for Windows executable programs."""
    if not program_name or len(program_name) > 80:
        return False, "Invalid program name"
    name = program_name.strip()
    search = name.split(",")[0].split(" - ")[0].split(" (")[0].strip() or name
    search_lower = search.lower()

    # Reject anything that looks like a website or URL — OPEN_APP is for executables only.
    # If the LLM tried to open a website, tell it to use OPEN_URL or navigate in the browser.
    website_indicators = (
        ".com", ".org", ".net", ".io", ".gg", ".tv", ".co",
        "http://", "https://", "www.",
        "lichess", "chess.com", "youtube", "google", "github",
        "twitch", "reddit", "twitter", "instagram", "facebook",
        "netflix", "amazon", "spotify",
    )
    if any(ind in search_lower for ind in website_indicators):
        return False, (
            f"OPEN_APP cannot open websites. '{search}' is a website. "
            "To go to a website: if a browser is open use OPEN_URL, "
            "otherwise first use OPEN_APP with program 'Chrome' to open the browser, "
            "then use OPEN_URL with the URL."
        )
    # Map to what Run dialog expects (no blocking check—just run)
    if "microsoft edge" in search_lower or search_lower == "edge":
        search = "msedge"
    elif "google chrome" in search_lower or search_lower == "chrome":
        search = "chrome"
    elif "notepad" in search_lower:
        search = "notepad"
    elif "code" in search_lower or "vscode" in search_lower or "visual studio code" in search_lower:
        search = "code"
    elif "calculator" in search_lower or "calc" in search_lower:
        search = "calc"
    elif "command prompt" in search_lower or "cmd" in search_lower:
        search = "cmd"
    elif "spotify" in search_lower:
        search = "Spotify"
    elif "discord" in search_lower:
        search = "Discord"
    elif "slack" in search_lower:
        search = "Slack"
    elif "outlook" in search_lower:
        search = "outlook"
    elif "excel" in search_lower:
        search = "excel"
    elif "word" in search_lower:
        search = "winword"
    # Else: use name as-is

    # Win+R → type → Enter (single sequence, no checks that could hang or block)
    try:
        pyautogui.hotkey("win", "r")
        time.sleep(1.0)
        pyautogui.write(search, interval=0.04)
        time.sleep(0.3)
        pyautogui.press("enter")
        return True, f"Opened via Run (Win+R): {search}"
    except Exception as e:
        return False, str(e)


def _open_url(url: str) -> tuple:
    """Focus browser address bar (Ctrl+L), type URL, Enter. Use when a browser is already open. Returns (True, msg) or (False, msg)."""
    if not url or len(url) > 500:
        return False, "Invalid URL"
    url = url.strip()
    if not url:
        return False, "Empty URL"
    # Add https:// if no scheme so "youtube.com" works
    if "://" not in url and not url.startswith("file"):
        url = "https://" + url
    try:
        pyautogui.hotkey("ctrl", "l")
        time.sleep(0.5)
        pyautogui.write(url, interval=0.03)
        time.sleep(0.2)
        pyautogui.press("enter")
        return True, f"Opened URL: {url}"
    except Exception as e:
        return False, str(e)


def _execute_action(action_dict: dict, screen_w: int, screen_h: int) -> tuple:
    """Execute pywinauto actions (CLICK, DOUBLE_CLICK, TYPE_IN, MENU) and pynput (KEYS, TYPE_TEXT). Returns (success, message)."""
    try:
        action = (action_dict.get("action") or "").upper().replace(" ", "_")
        params = action_dict.get("parameters") or action_dict
        if isinstance(params, dict) and "action" in params:
            params = params.get("parameters", params)

        # OPEN_APP: open any program by name (Start menu, Run dialog, or shell)
        if action == "OPEN_APP":
            program = (params.get("program") or params.get("app_name") or params.get("element") or params.get("name") or "").strip()
            if not program:
                return False, "OPEN_APP requires 'program' or 'app_name'"
            return _open_program(program)

        # OPEN_URL: focus address bar, type URL, Enter (browser must already be open)
        if action == "OPEN_URL":
            url = (params.get("url") or params.get("website") or params.get("site") or "").strip()
            if not url:
                return False, "OPEN_URL requires 'url' (e.g. youtube.com)"
            return _open_url(url)

        # CLICK with no element = click at current cursor position (after MOVE_TO)
        if action == "CLICK" and not (params.get("element") or "").strip():
            pyautogui.click()
            return True, "Clicked at current cursor position"

        # pywinauto actions (element-based)
        if action in ("CLICK", "DOUBLE_CLICK", "TYPE_IN", "MENU", "TASK_COMPLETE") and _PYWINAUTO_OK:
            ok, msg = _execute_pywinauto(action_dict)
            if ok:
                return True, msg
            # Fallback: TYPE_IN failed → type into focused control (user may have clicked field first)
            if action == "TYPE_IN" and params.get("text"):
                if sys.platform == "win32" and _PYNPUT_AVAILABLE:
                    kb = PynputController()
                    kb.type(params.get("text", ""))
                    return True, f"Typed into focused control: {params.get('text', '')[:30]}..."
                try:
                    pyautogui.write(params.get("text", ""), interval=0.05)
                    return True, f"Typed into focused control: {params.get('text', '')[:30]}..."
                except Exception:
                    pass
            # Fallback: if CLICK/DOUBLE_CLICK on an app-like name failed, try opening via Start menu
            if action in ("CLICK", "DOUBLE_CLICK"):
                el = (params.get("element") or params.get("name") or "").strip()
                if el and not any(x in el.lower() for x in ["button", "open", "save", "cancel", "search", "menu", "file", "edit"]):
                    return _open_app_via_start(el)
            return False, msg

        if action in ("MOVE_TO", "CLICK_XY"):
            x = params.get("x")
            y = params.get("y")
            if x is None or y is None:
                return False, f"{action} requires 'x' and 'y' parameters"
            try:
                x, y = int(x), int(y)
            except (TypeError, ValueError):
                return False, f"{action} x,y must be numbers"
            scale = 1024 / max(screen_w, screen_h) if max(screen_w, screen_h) > 1024 else 1.0
            model_w = max(1, int(screen_w * scale))
            model_h = max(1, int(screen_h * scale))
            actual_x = max(0, min(screen_w - 1, int(round(x * screen_w / model_w))))
            actual_y = max(0, min(screen_h - 1, int(round(y * screen_h / model_h))))
            print(f"[{action}] model({x},{y}) -> screen({actual_x},{actual_y}) size={screen_w}x{screen_h}", flush=True)
            _move_cursor(actual_x, actual_y)
            if action == "CLICK_XY":
                import time as _t
                _t.sleep(0.05)
                pyautogui.click()
                return True, f"Moved to ({actual_x}, {actual_y}) and clicked"
            return True, f"Moved cursor to ({actual_x}, {actual_y})"

        if action == "CLICK_CURRENT":
            pos = pyautogui.position()
            pyautogui.click()
            return True, f"Clicked at current cursor position ({pos.x}, {pos.y})"

        if action in ("KEYS", "KEY_PRESS"):
            keys = params.get("keys") or params.get("keys_list") or []
            if isinstance(keys, str):
                keys = [keys]
            if not keys:
                return False, "No keys specified"
            # Guard: block Win-key combos used to open apps — the model must use OPEN_APP instead
            keys_lower = [str(k).lower() for k in keys]
            win_keys = {"win", "winleft", "winright", "super", "windows", "command"}
            if any(k in win_keys for k in keys_lower):
                return False, (
                    "BLOCKED: Do not use KEYS to press Win/Win+R to open apps. "
                    "Use OPEN_APP action instead (e.g. {\"action\": \"OPEN_APP\", \"parameters\": {\"program\": \"Chrome\"}})."
                )
            key_map = {"control": "ctrl", "windows": "win", "command": "win", "win": "winleft", "super": "winleft"}
            keys = [key_map.get(str(k).lower(), k) for k in keys]
            # On Windows use pynput for reliable key delivery to foreground window
            if sys.platform == "win32" and _PYNPUT_AVAILABLE:
                pynput_keys = []
                for k in keys:
                    pk = _pynput_key(k)
                    if pk is None and len(str(k)) == 1:
                        pk = str(k).lower()
                    pynput_keys.append(pk)
                if all(pk is not None for pk in pynput_keys):
                    kb = PynputController()
                    for pk in pynput_keys:
                        kb.press(pk)
                    for pk in reversed(pynput_keys):
                        kb.release(pk)
                    return True, f"Pressed {'+'.join(str(k) for k in keys)}"
            # Fallback: pyautogui
            mods = {"winleft", "winright", "ctrl", "alt", "shift"}
            if len(keys) == 1 and str(keys[0]).lower() in mods:
                pyautogui.keyDown(keys[0])
                pyautogui.keyUp(keys[0])
            elif len(keys) == 1:
                pyautogui.press(keys[0])
            else:
                pyautogui.hotkey(*keys)
            return True, f"Pressed {'+'.join(str(k) for k in keys)}"

        if action == "TYPE_TEXT":
            text = params.get("text", "")
            if sys.platform == "win32" and _PYNPUT_AVAILABLE:
                kb = PynputController()
                kb.type(text)
            else:
                pyautogui.write(text, interval=params.get("interval", 0.05))
            return True, f"Typed: {text[:50]}..."

        if action == "TASK_COMPLETE":
            return True, params.get("message", "Task complete")

        if action in ("CLICK", "DOUBLE_CLICK", "TYPE_IN", "MENU") and not _PYWINAUTO_OK:
            return False, "pywinauto not available (Windows + pip install pywinauto)"

        return False, f"Unknown action: {action}"
    except Exception as e:
        import traceback
        return False, f"{e}\n{traceback.format_exc()}"


# ── theme ──
BG      = "#1a1a2e"
BG_DARK = "#0f0f1a"
FG      = "#ffffff"
ACCENT  = "#e94560"

def _draw_grid_on_image(img, spacing: int = 50):
    """Overlay a labeled coordinate grid every `spacing` pixels so the model can read off exact coordinates."""
    try:
        draw = ImageDraw.Draw(img)
        w, h = img.size
        minor_col = (50, 50, 50)     # faint lines every 50px
        major_col = (100, 100, 100)  # brighter every 100px
        label_col = (180, 220, 255)  # bright blue labels
        try:
            font = ImageFont.truetype("arial.ttf", 11)
        except Exception:
            font = ImageFont.load_default()

        for x in range(0, w, spacing):
            is_major = x % 100 == 0
            col = major_col if is_major else minor_col
            draw.line([(x, 0), (x, h)], fill=col, width=1)
            if is_major and x > 0:
                draw.text((x + 2, 1), str(x), fill=label_col, font=font)

        for y in range(0, h, spacing):
            is_major = y % 100 == 0
            col = major_col if is_major else minor_col
            draw.line([(0, y), (w, y)], fill=col, width=1)
            if is_major and y > 0:
                draw.text((2, y + 1), str(y), fill=label_col, font=font)
    except Exception:
        pass


def _draw_cursor_on_image(img, cursor_x_img: int, cursor_y_img: int):
    """Draw a visible cursor marker so the model can see cursor position clearly."""
    try:
        mw, mh = img.size
        cx = max(0, min(mw - 1, cursor_x_img))
        cy = max(0, min(mh - 1, cursor_y_img))
        draw = ImageDraw.Draw(img)
        r, arm, dot, lw = 4, 12, 2, 2
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline="#ff0000", width=lw)
        draw.line([cx - arm, cy, cx + arm, cy], fill="#ff0000", width=lw)
        draw.line([cx, cy - arm, cx, cy + arm], fill="#ff0000", width=lw)
        draw.ellipse([cx - dot, cy - dot, cx + dot, cy + dot], fill="#ff0000", outline="#ffffff")
        try:
            font = ImageFont.truetype("arial.ttf", 13)
        except Exception:
            font = ImageFont.load_default()
        label = f"CURSOR ({cx},{cy})"
        tx = cx + arm + 6
        ty = cy - 12
        if tx + 100 > mw:
            tx = cx - arm - 110
        if ty < 0:
            ty = cy + arm + 4
        for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            draw.text((tx + dx, ty + dy), label, fill="#000000", font=font)
        draw.text((tx, ty), label, fill="#ffff00", font=font)
    except Exception:
        pass


def _take_screenshot_with_cursor():
    """Capture screenshot AND cursor position atomically (before the GUI re-appears).

    Returns (screenshot: PIL.Image, cursor_xy: tuple[int,int], screen_size: tuple[int,int]).
    """
    cursor = pyautogui.position()
    screenshot = pyautogui.screenshot()
    screen_size = pyautogui.size()
    return screenshot, (cursor.x, cursor.y), (screen_size[0], screen_size[1])


def _prepare_image_for_model(screenshot, cursor_xy, screen_size):
    """Resize screenshot, draw cursor marker, return (img, model_w, model_h, cursor_in_model).

    The cursor marker is drawn at the position the cursor was when the screenshot was taken.

    Coordinate chain:
      cursor_xy (pyautogui screen coords)
        -> screenshot pixel coords:  cursor * (screenshot.size / screen_size)
        -> model pixel coords:       ss_cursor * scale
    """
    img = screenshot.convert("RGB")
    ss_w, ss_h = img.size          # screenshot pixel dimensions (physical)
    scr_w, scr_h = screen_size     # pyautogui logical screen size

    # Step 1: map cursor from pyautogui screen space → screenshot pixel space
    # (These spaces differ when Windows display scaling > 100% and no DPI awareness.)
    cur_ss_x = cursor_xy[0] * ss_w / max(1, scr_w)
    cur_ss_y = cursor_xy[1] * ss_h / max(1, scr_h)

    # Step 2: resize image for model (long edge ≤ 1024)
    scale = 1024 / max(ss_w, ss_h) if max(ss_w, ss_h) > 1024 else 1.0
    if scale < 1.0:
        nw, nh = int(ss_w * scale), int(ss_h * scale)
        img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    model_w, model_h = img.size

    # Step 3: map cursor from screenshot space → model-image space
    cx_img = int(cur_ss_x * scale)
    cy_img = int(cur_ss_y * scale)
    cx_img = max(0, min(model_w - 1, cx_img))
    cy_img = max(0, min(model_h - 1, cy_img))

    # Draw grid first so cursor appears on top of it
    _draw_grid_on_image(img)
    _draw_cursor_on_image(img, cx_img, cy_img)

    return img, model_w, model_h, (cx_img, cy_img)


# Gemini vision models
GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]


class AgentGUI:
    """Single-window AI agent with chess and screen-control modes."""

    def __init__(self, root):
        self.root = root
        root.title("AI Agent")
        root.geometry("1400x900")
        root.configure(bg=BG)

        # ── settings variables (Gemini only) ──
        cfg = _load_config()
        default_gemini_key = (cfg.get("gemini_api_key") or "").strip() or os.environ.get("GEMINI_API_KEY", "")
        self.gemini_api_key_var = tk.StringVar(value=default_gemini_key)
        default_gemini_model = (cfg.get("gemini_model") or "").strip() or GEMINI_MODELS[0]
        self.gemini_model_var = tk.StringVar(value=default_gemini_model if default_gemini_model in GEMINI_MODELS else GEMINI_MODELS[0])
        # chess
        self.turn_var        = tk.StringVar(value="white")
        self.conf_var        = tk.DoubleVar(value=0.15)
        self.depth_var       = tk.IntVar(value=18)
        self.interval_var    = tk.DoubleVar(value=3.0)
        self.click_delay_var = tk.DoubleVar(value=0.15)

        # ── state ──
        self._running = False
        self._mode = None          # "chess", "screen_control", or None
        self._thread = None
        self._screen_action_count = 0
        self._hide_event = threading.Event()
        self._show_event = threading.Event()

        # ── chess engine (headless) ──
        self.chess = ChessEngine(log_fn=self.log)

        self._apply_styles()
        self._build_ui()

        # start loading chess models in background
        self.log("Loading chess engine...", "warning")
        threading.Thread(target=self._load_chess, daemon=True).start()

    # ==================================================================
    #  Styles
    # ==================================================================
    def _apply_styles(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("TFrame",           background=BG)
        s.configure("TLabel",           background=BG, foreground=FG)
        s.configure("TLabelframe",      background=BG, foreground=FG)
        s.configure("TLabelframe.Label", background=BG, foreground=FG)

    # ==================================================================
    #  UI
    # ==================================================================
    def _build_ui(self):
        outer = ttk.Frame(self.root, padding="10")
        outer.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # ── left panel ──
        left = ttk.Frame(outer)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        # title
        tk.Label(left, text="AI Agent", font=("Arial", 24, "bold"),
                 bg=BG, fg=ACCENT).grid(row=0, column=0, pady=(0, 2))
        tk.Label(left, text="Type a task and press Start",
                 font=("Arial", 10, "italic"),
                 bg=BG, fg="#9e9e9e").grid(row=1, column=0, pady=(0, 8))

        # status
        self.status_label = tk.Label(left, text="Loading...",
                                     font=("Arial", 12), bg=BG, fg="#FF9800")
        self.status_label.grid(row=2, column=0, pady=(0, 10))

        # task input
        task_frame = ttk.LabelFrame(left, text="Task", padding="6")
        task_frame.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        self.task_text = tk.Text(
            task_frame, height=3, width=32, font=("Consolas", 10),
            bg=BG_DARK, fg="#d4d4d4", insertbackground="#fff",
            wrap=tk.WORD, relief=tk.FLAT,
            highlightthickness=1, highlightcolor="#444")
        self.task_text.pack(fill=tk.X)
        self.task_text.insert("1.0", "play chess")

        # buttons row
        btn_row = tk.Frame(left, bg=BG)
        btn_row.grid(row=4, column=0, pady=(0, 8))

        self.start_btn = tk.Button(
            btn_row, text="Start", font=("Arial", 14, "bold"),
            bg="#4CAF50", fg="#fff", activebackground="#388E3C",
            relief=tk.FLAT, padx=25, pady=10, width=10,
            command=self._toggle, cursor="hand2")
        self.start_btn.pack(side=tk.LEFT, padx=(0, 8))

        tk.Button(
            btn_row, text="Settings", font=("Arial", 11),
            bg="#555", fg="#fff", activebackground="#666",
            relief=tk.FLAT, padx=15, pady=10,
            command=self._open_settings, cursor="hand2",
        ).pack(side=tk.LEFT)

        tk.Button(
            left, text="Clear Log", font=("Arial", 10),
            bg="#444", fg="#fff", relief=tk.FLAT, padx=15, pady=5,
            command=lambda: self.log_text.delete("1.0", tk.END),
            cursor="hand2",
        ).grid(row=5, column=0, pady=(0, 10))

        # stats
        stats_frame = ttk.LabelFrame(left, text="Stats", padding="8")
        stats_frame.grid(row=6, column=0, sticky="ew")
        self.stats_label = tk.Label(
            stats_frame,
            text="Mode: --\nCycles: 0\nActions: 0\nStatus: idle",
            font=("Consolas", 10), bg=BG, fg="#fff", justify=tk.LEFT)
        self.stats_label.pack(anchor="w")

        # ── right panel ──
        right = ttk.Frame(outer)
        right.grid(row=0, column=1, sticky="nsew")
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(0, weight=1)

        img_frame = ttk.LabelFrame(right, text="Screen", padding="8")
        img_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        right.rowconfigure(0, weight=2)
        right.columnconfigure(0, weight=1)
        self.screenshot_label = tk.Label(
            img_frame,
            text="No screenshot yet\n\nPosition this window so\nit doesn't cover the target.",
            bg="#2a2a3e", fg="#888", font=("Arial", 12))
        self.screenshot_label.pack(expand=True, fill=tk.BOTH)

        log_frame = ttk.LabelFrame(right, text="Log", padding="8")
        log_frame.grid(row=1, column=0, sticky="nsew")
        right.rowconfigure(1, weight=1)
        self.log_text = scrolledtext.ScrolledText(
            log_frame, height=12, font=("Consolas", 9),
            bg=BG_DARK, fg="#d4d4d4", insertbackground="#fff", wrap=tk.WORD)
        self.log_text.pack(expand=True, fill=tk.BOTH)

        for tag, color in [("info", "#4CAF50"), ("error", "#f44336"),
                           ("action", "#2196F3"), ("warning", "#FF9800"),
                           ("header", "#e94560"), ("piece", "#80CBC4"),
                           ("move", "#FFD700"), ("board", "#B0BEC5"),
                           ("dim", "#666666"), ("thought", "#80CBC4"),
                           ("result", "#FFD700")]:
            self.log_text.tag_config(tag, foreground=color)

    # ==================================================================
    #  Settings Popup
    # ==================================================================
    def _open_settings(self):
        win = tk.Toplevel(self.root)
        win.title("Settings")
        win.geometry("640x1020")
        win.configure(bg=BG)
        win.transient(self.root)
        win.grab_set()

        px = 10
        py = 2

        # ── Gemini (vision model) ──
        tk.Label(win, text="Gemini (vision model)",
                 font=("Arial", 12, "bold"), bg=BG, fg=ACCENT).pack(
            anchor="w", padx=px, pady=(12, 4))
        tk.Label(win, text="API key: https://aistudio.google.com/apikey", bg=BG, fg="#9e9e9e", font=("Arial", 9)).pack(anchor="w", padx=px, pady=py)
        tk.Label(win, text="Gemini API Key:", bg=BG, fg=FG).pack(anchor="w", padx=px, pady=py)
        gkey_f = tk.Frame(win, bg=BG)
        gkey_f.pack(anchor="w", fill=tk.X, padx=px, pady=py)
        self._gemini_key_entry = tk.Entry(
            gkey_f, textvariable=self.gemini_api_key_var, show="*",
            font=("Consolas", 9), bg=BG_DARK, fg="#d4d4d4",
            insertbackground="#fff", relief=tk.FLAT)
        self._gemini_key_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._gemini_key_vis = False
        tk.Button(gkey_f, text="Show", font=("Arial", 8), bg="#444",
                  fg="#fff", relief=tk.FLAT, padx=6,
                  command=self._toggle_gemini_key).pack(side=tk.RIGHT, padx=(4, 0))
        tk.Label(win, text="Model:", bg=BG, fg=FG).pack(anchor="w", padx=px, pady=py)
        ttk.Combobox(win, textvariable=self.gemini_model_var,
                     values=GEMINI_MODELS, state="readonly",
                     width=40).pack(anchor="w", padx=px, pady=py)

        # ── Chess section ──
        tk.Label(win, text="Chess Agent", font=("Arial", 12, "bold"),
                 bg=BG, fg=ACCENT).pack(anchor="w", padx=px, pady=(16, 4))
        tk.Label(win, text="Playing as: auto-detected from the board",
                 bg=BG, fg="#9e9e9e", font=("Arial", 9)).pack(anchor="w", padx=px, pady=py)

        for label, var, lo, hi, res in [
            ("YOLO confidence:",  self.conf_var,        0.10, 0.95, 0.05),
            ("Stockfish depth:",  self.depth_var,       5,    25,   1),
            ("Scan interval (s):",self.interval_var,    1.0,  15.0, 0.5),
            ("Click delay (s):",  self.click_delay_var, 0.05, 1.0,  0.05),
        ]:
            tk.Label(win, text=label, bg=BG, fg=FG).pack(anchor="w", padx=px, pady=py)
            tk.Scale(win, from_=lo, to=hi, resolution=res, orient=tk.HORIZONTAL,
                     variable=var, bg=BG, fg=FG,
                     highlightthickness=0, troughcolor="#333",
                     length=300).pack(anchor="w", padx=px, pady=py)

        # close button (saves config)
        def _close_settings():
            _save_config(
                self.gemini_api_key_var.get().strip(),
                self.gemini_model_var.get(),
            )
            win.destroy()
        tk.Button(win, text="Close", font=("Arial", 11, "bold"),
                  bg=ACCENT, fg="#fff", relief=tk.FLAT, padx=30, pady=8,
                  command=_close_settings, cursor="hand2",
                  ).pack(pady=(16, 10))

    def _toggle_gemini_key(self):
        if hasattr(self, "_gemini_key_entry"):
            self._gemini_key_vis = getattr(self, "_gemini_key_vis", False)
            self._gemini_key_vis = not self._gemini_key_vis
            self._gemini_key_entry.config(show="" if self._gemini_key_vis else "*")

    # ==================================================================
    #  Logging (thread-safe)
    # ==================================================================
    def log(self, msg, tag=""):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"
        self.root.after(0, self._log_insert, line, tag)

    def _log_insert(self, line, tag):
        self.log_text.insert(tk.END, line, tag)
        self.log_text.see(tk.END)

    # ==================================================================
    #  Image display
    # ==================================================================
    def _show_image(self, img):
        try:
            w = 640
            r = w / img.width
            h = int(img.height * r)
            resized = img.resize((w, h), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(resized)
            self.screenshot_label.config(image=photo, text="")
            self.screenshot_label.image = photo
        except Exception:
            pass

    # ==================================================================
    #  Gemini (screenshot + task -> start_chess or comment)
    # ==================================================================
    def _run_router(self, task, screenshot, ui_elements=None, cursor_xy=None, screen_size=None):
        """Run router using Gemini API. Returns (action, payload, raw_response)."""
        if screen_size is None:
            screen_size = pyautogui.size()
        if cursor_xy is None:
            cursor_xy = tuple(pyautogui.position())
        img, model_w, model_h, (cur_mx, cur_my) = _prepare_image_for_model(
            screenshot, cursor_xy, screen_size
        )
        # Show annotated image (with cursor marker) in GUI preview
        self.root.after(0, self._show_image, img)
        learning_context = _build_learning_context()
        system = "You are a router for an AI agent. The user will send a task and a screenshot of their screen.\n"
        if learning_context:
            system += learning_context + "\n\n"
        system += ROUTER_SYSTEM_TAIL
        user_text = f"Task: {task}\n\nLook at the screenshot and respond with JSON."
        user_text += f"\n\nScreenshot dimensions: width={model_w} height={model_h}. Coords from 0 to {model_w-1}, 0 to {model_h-1}."
        user_text += f"\n\n>>> CURRENT CURSOR POSITION: ({cur_mx}, {cur_my}) — the CLICK happens at this point only. Only CLICK_CURRENT if this exact point is on the button. <<<"
        user_text += (
            "\n\nGrid lines every 50px. Use CLICK_XY with grid-derived coordinates to click targets in one step."
        )
        if ui_elements:
            user_text += "\n\nUI elements (use CLICK with element name when available):\n"
            for item in ui_elements[:40]:
                name = (item.get("name") or "").strip()
                ctype = item.get("control_type", "")
                if name:
                    user_text += f"  \"{name}\" ({ctype})\n"
        from gemini_vl import call_gemini
        raw_response = call_gemini(
            system, user_text, img, conversation_messages=None, max_tokens=1024,
            api_key=self.gemini_api_key_var.get().strip(),
            model=self.gemini_model_var.get().strip() or None,
        )
        text = raw_response
        think_match = re.search(r"<thinking\s*>.*?</thinking\s*>", text, re.DOTALL | re.IGNORECASE)
        if think_match:
            text = re.sub(r"<thinking\s*>.*?</thinking\s*>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
        if "```" in text:
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if m:
                text = m.group(1)
        data = _parse_json(text)
        if data is None and "store_feedback" in raw_response:
            data = _parse_json(raw_response)
        action, payload = _parse_router_response(data, text, raw_response)
        return (action, payload, raw_response)

    def _ask_next_action(self, task, screenshot, action_history: list, conversation_messages: list = None, thought_history: list = None, ui_elements=None, cursor_xy=None, screen_size=None, prior_screenshot_parts: list = None, stuck_hint: str = None, move_to_hint: str = None):
        """Ask Gemini for the next screen-control action. Returns (action, payload, messages, thought_history, prepared_img)."""
        if not self.gemini_api_key_var.get().strip():
            return ("comment", "Set Gemini API key in Settings.", conversation_messages or [], thought_history or [], None)
        thought_history = list(thought_history) if thought_history else []
        if screen_size is None:
            screen_size = pyautogui.size()
        if cursor_xy is None:
            cursor_xy = tuple(pyautogui.position())
        img, _, _, (cur_mx, cur_my) = _prepare_image_for_model(
            screenshot, cursor_xy, screen_size
        )
        # Show annotated image (with cursor marker) in GUI preview
        self.root.after(0, self._show_image, img)
        history_str = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(action_history[-12:]))
        learning = _build_learning_context()

        system = "You control the user's computer. Output one action per turn. Be fast and precise."
        if move_to_hint:
            system = f"OVERRIDE: {move_to_hint}\n\n" + system
        if thought_history:
            system += "\nPrevious reasoning:\n"
            for prev in thought_history[-5:]:
                system += f"  {prev[:400]}\n"
        if learning:
            system += "\n" + learning
        system += """

Reply: <thinking>brief reasoning + EXACT grid-derived coordinates if clicking</thinking> then ONE JSON object.

━━ COORDINATE GRID — HOW TO READ IT ━━
The screenshot has a grid overlay: faint lines every 50px, labeled 0, 50, 100, 150, 200... along the top (x) and left (y).
To find a target's coordinates:
1. Find the vertical grid line just LEFT of the target → that's the base x (e.g. 500).
2. Estimate how many pixels RIGHT of that line the target center is → add to base (e.g. +25 → x=525).
3. Find the horizontal grid line just ABOVE the target → base y (e.g. 250).
4. Estimate pixels BELOW that line → add to base (e.g. +25 → y=275).
5. State these in <thinking> BEFORE outputting the action.

CLICKING: Use CLICK_XY — it moves the cursor to (x,y) AND clicks in one step.
{"action": "CLICK_XY", "parameters": {"x": 525, "y": 275}}
NEVER guess coordinates. ALWAYS read them from the grid lines in the image.

If a clickable UI element name is listed, prefer CLICK by name (more reliable than coordinates):
{"action": "CLICK", "parameters": {"element": "Play"}}

Actions:
- CLICK_XY: move+click at grid-derived (x,y): {"action": "CLICK_XY", "parameters": {"x": 400, "y": 300}}
- CLICK by element name: {"action": "CLICK", "parameters": {"element": "Open"}}
- OPEN_APP (Windows programs ONLY — NOT websites): {"action": "OPEN_APP", "parameters": {"program": "Chrome"}}
- OPEN_URL (websites, browser must be open): {"action": "OPEN_URL", "parameters": {"url": "chess.com"}}
- KEYS: {"action": "KEYS", "parameters": {"keys": ["ctrl", "t"]}}
- TYPE_TEXT: {"action": "TYPE_TEXT", "parameters": {"text": "hello"}}
- TYPE_IN: {"action": "TYPE_IN", "parameters": {"element": "Search", "text": "query"}}
- TASK_COMPLETE: {"action": "TASK_COMPLETE", "parameters": {"message": "Done"}}
- SCREEN_LOADING (page still loading): {"action": "SCREEN_LOADING", "parameters": {}}
- start_chess (board visible): {"action": "start_chess", "parameters": {"reason": "Board visible", "playing_as": "white"}}

NEVER use KEYS to open apps. OPEN_APP for programs, OPEN_URL for websites.
NEVER click blindly. If CLICK_XY missed the target (check next screenshot), try CLICK by element name or re-read the grid more carefully."""

        model_w, model_h = img.size
        last_action_str = action_history[-1] if action_history else "(none)"
        user_text = f"Task: {task}\n\nActions so far:\n{history_str or '  (none yet)'}"
        user_text += f"\n\nLast action result: {last_action_str}"
        user_text += "\nLook at this NEW screenshot — did your last action work? If the screen didn't change after a click, the click MISSED. Try a different approach."
        user_text += f"\n\nScreenshot: {model_w}x{model_h}. Cursor currently at ({cur_mx},{cur_my})."
        user_text += f"\nGrid: lines every 50px labeled 0,50,100,... along top (x) and left (y). Range: x=0..{model_w-1}, y=0..{model_h-1}."
        user_text += "\nTo click: read target center from nearest grid lines → CLICK_XY. If a UI element name matches, prefer CLICK by name."
        user_text += "\n\nWhat is the next action?"
        if move_to_hint:
            user_text += f"\n{move_to_hint}"
        if stuck_hint:
            user_text += f"\n{stuck_hint}"
        if ui_elements:
            user_text += "\n\nUI elements:\n"
            for item in ui_elements[:40]:
                name = (item.get("name") or "").strip()
                ctype = item.get("control_type", "")
                if name:
                    user_text += f"  \"{name}\" ({ctype})\n"
        messages = list(conversation_messages) if conversation_messages else []

        try:
            from gemini_vl import call_gemini
            local_history = []
            for m in messages:
                role, content = m.get("role"), m.get("content")
                if isinstance(content, list):
                    text = next((c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"), "")
                else:
                    text = content or ""
                if role and (text or role == "assistant"):
                    local_history.append({"role": role, "content": text})
            prior_parts = None
            if prior_screenshot_parts:
                prior_parts = [(cap, pimg) for pimg, cap in prior_screenshot_parts]
            last_err = None
            for api_attempt in range(3):
                try:
                    text = call_gemini(
                        system, user_text, img,
                        conversation_messages=None if prior_parts else local_history,
                        max_tokens=1024,
                        api_key=self.gemini_api_key_var.get().strip(),
                        model=self.gemini_model_var.get().strip() or None,
                        prior_screenshot_parts=prior_parts,
                    )
                    break
                except Exception as e:
                    last_err = e
                    err_str = str(e).lower()
                    if api_attempt < 2:
                        if "timed out" in err_str or "timeout" in err_str:
                            self.log(f"  Request timed out ({api_attempt + 1}/3); retrying in 2s...", "dim")
                        time.sleep(2)
                    else:
                        # After retries: if "no text" or empty response, return screen_loading so loop continues with fresh screenshot
                        if "no text" in err_str or "returned no text" in err_str or "empty" in err_str:
                            self.log("  Response still empty after 3 tries; taking fresh screenshot and continuing (no stop).", "dim")
                            return ("screen_loading", None, messages, thought_history, img)
                        raise
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": text})
            if len(messages) > 6:
                messages = messages[-6:]

            # Extract and log thinking phase; append to thought history for next turn
            think_match = re.search(r"<thinking\s*>.*?</thinking\s*>", text, re.DOTALL | re.IGNORECASE)
            if think_match:
                thinking = think_match.group(0)
                inner = re.search(r"<thinking\s*>(.*?)</thinking\s*>", thinking, re.DOTALL | re.IGNORECASE)
                thinking = inner.group(1).strip() if inner else thinking
                thought_history.append(thinking)
                if len(thought_history) > 10:
                    thought_history = thought_history[-10:]
                self.log("  ━━━ MODEL THINKING (next action) ━━━", "header")
                self.log("  Reasoning:", "header")
                for line in thinking.split("\n"):
                    if line.strip():
                        self.log(f"    {line.strip()}", "thought")
                self.log("  ━━━ End thinking → action ━━━", "dim")
                # Parse only the part AFTER </thinking> so we don't treat thinking text as JSON
                text = re.sub(r"<thinking\s*>.*?</thinking\s*>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
            else:
                self.log("  ━━━ MODEL THINKING (next action) ━━━", "header")
                before_brace = text.split("{")[0].strip() if "{" in text else ""
                if before_brace and len(before_brace) > 2:
                    self.log("  (reasoning before JSON):", "header")
                    for line in before_brace.split("\n")[:8]:
                        if line.strip():
                            self.log(f"    {line.strip()}", "thought")
                else:
                    self.log("  (no <thinking> block)", "dim")
                self.log("  ━━━ End → action ━━━", "dim")

            if not text or "{" not in text:
                # Retry once: ask for JSON only so the loop doesn't stop
                self.log("  Model sent only thinking; asking for JSON only...", "dim")
                nudge = "You only sent thinking. Reply with exactly one JSON object for the next action. No other text."
                messages.append({"role": "user", "content": nudge})
                retry_text = call_gemini(
                    system, nudge, img,
                    conversation_messages=None if prior_parts else messages,
                    max_tokens=512,
                    api_key=self.gemini_api_key_var.get().strip(),
                    model=self.gemini_model_var.get().strip() or None,
                    prior_screenshot_parts=prior_parts,
                )
                messages.append({"role": "assistant", "content": retry_text})
                retry_text = re.sub(r"<thinking\s*>.*?</thinking\s*>", "", retry_text, flags=re.DOTALL | re.IGNORECASE).strip()
                if retry_text and "{" in retry_text:
                    text = retry_text
                else:
                    return ("comment", "Model only sent thinking, no action JSON. Output JSON right after </thinking>.", messages, thought_history, img)
            if "```" in text:
                m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
                if m:
                    text = m.group(1)
            data = _parse_json(text)
            # Fallback: model returned plain action name (e.g. "CLICK_CURRENT" or "CLICK_CURRENT {}") — treat as execute so we never stop
            _EXECUTE_ACTIONS = ("CLICK", "CLICK_CURRENT", "CLICK_XY", "MOVE_TO", "DOUBLE_CLICK", "TYPE_IN", "MENU", "KEYS", "KEY_PRESS", "TYPE_TEXT", "TASK_COMPLETE", "OPEN_APP", "OPEN_URL")
            def _parse_bare_action(raw: str):
                s = (raw or "").strip().upper().replace("-", "_")
                # Match "CLICK_CURRENT", "CLICK_CURRENT {}", "CLICK_CURRENT {}" (with space), etc.
                for a in _EXECUTE_ACTIONS:
                    if s == a or s.startswith(a + " ") or s.startswith(a + "{}") or s == a + "{}":
                        return "KEYS" if a == "KEY_PRESS" else a
                s = s.replace(" ", "_")  # "CLICK_CURRENT {}" -> "CLICK_CURRENT_{}"
                for a in _EXECUTE_ACTIONS:
                    if s == a or s.startswith(a + "_"):
                        return "KEYS" if a == "KEY_PRESS" else a
                return None
            if data is None:
                bare = _parse_bare_action(text)
                if bare:
                    return ("execute", {"action": bare, "parameters": {}}, messages, thought_history, img)
                return ("comment", f"Model returned invalid JSON. Ask again or rephrase. Raw: {text[:120]}...", messages, thought_history, img)
            raw_action = data.get("action") or "comment"
            action = str(raw_action).lower().strip().replace(" ", "_").replace("-", "_")
            a_norm = str(raw_action).upper().strip().replace(" ", "_").replace("-", "_")
            if a_norm == "SCREEN_LOADING":
                return ("screen_loading", None, messages, thought_history, img)
            if a_norm == "ADD_FEEDBACK":
                params = data.get("parameters", data) or {}
                if isinstance(params, dict):
                    return ("add_feedback", params, messages, thought_history, img)
                return ("add_feedback", {"message": str(params)}, messages, thought_history, img)
            if a_norm == "START_CHESS":
                params = data.get("parameters", data) or {}
                if isinstance(params, dict):
                    playing_as = (params.get("playing_as") or "white").lower()
                    if playing_as not in ("white", "black"):
                        playing_as = "white"
                    return ("start_chess", {"reason": params.get("reason", "Chess board visible."), "playing_as": playing_as}, messages, thought_history, img)
                return ("start_chess", {"reason": "Chess board visible.", "playing_as": "white"}, messages, thought_history, img)
            if a_norm in ("CLICK", "CLICK_CURRENT", "CLICK_XY", "MOVE_TO", "DOUBLE_CLICK", "TYPE_IN", "MENU", "KEYS", "KEY_PRESS", "TYPE_TEXT", "TASK_COMPLETE", "OPEN_APP", "OPEN_URL"):
                params = data.get("parameters", data)
                act = "KEYS" if a_norm == "KEY_PRESS" else a_norm
                return ("execute", {"action": act, "parameters": params if isinstance(params, dict) else {}}, messages, thought_history, img)
            if action == "comment":
                msg = data.get("message") or data.get("reason") or ""
                if isinstance(msg, str) and msg.strip().startswith("{"):
                    inner = _parse_json(msg.strip())
                    if isinstance(inner, dict):
                        ia = (inner.get("action") or "").upper().replace(" ", "_").replace("-", "_")
                        if ia in ("CLICK", "CLICK_CURRENT", "CLICK_XY", "MOVE_TO", "DOUBLE_CLICK", "TYPE_IN", "MENU", "KEYS", "KEY_PRESS", "TYPE_TEXT", "TASK_COMPLETE", "OPEN_APP", "OPEN_URL"):
                            params = inner.get("parameters", inner)
                            act = "KEYS" if ia == "KEY_PRESS" else ia
                            return ("execute", {"action": act, "parameters": params if isinstance(params, dict) else {}}, messages, thought_history, img)
            if action == "keypress":
                action = "keys"
            elif action == "typetext":
                action = "type_text"
            elif action == "taskcomplete":
                action = "task_complete"
            if action in ("click", "click_xy", "double_click", "type_in", "menu", "keys", "type_text", "task_complete", "open_app", "open_url"):
                params = data.get("parameters", data)
                act = "KEYS" if action == "keys" else action.upper().replace(" ", "_")
                return ("execute", {"action": act, "parameters": params if isinstance(params, dict) else {}}, messages, thought_history, img)
            a_upper = str(raw_action).upper().replace(" ", "_").replace("-", "_")
            if a_upper in ("CLICK", "CLICK_CURRENT", "CLICK_XY", "MOVE_TO", "DOUBLE_CLICK", "TYPE_IN", "MENU", "KEYS", "KEY_PRESS", "TYPE_TEXT", "TASK_COMPLETE", "OPEN_APP", "OPEN_URL"):
                params = data.get("parameters", data)
                act = "KEYS" if a_upper == "KEY_PRESS" else a_upper
                return ("execute", {"action": act, "parameters": params if isinstance(params, dict) else {}}, messages, thought_history, img)
            # Last resort: text might be the raw action JSON (e.g. parser returned wrong object). Find any action JSON and run it.
            for start in [i for i, c in enumerate(text) if c == "{"]:
                depth, end = 0, None
                for i in range(start, len(text)):
                    if text[i] == "{":
                        depth += 1
                    elif text[i] == "}":
                        depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
                if end is None:
                    continue
                try:
                    obj = json.loads(text[start:end])
                except json.JSONDecodeError:
                    continue
                a = (obj.get("action") or "").upper().replace(" ", "_").replace("-", "_")
                if a == "SCREEN_LOADING":
                    return ("screen_loading", None, messages, thought_history, img)
                if a == "START_CHESS":
                    p = obj.get("parameters", obj) or {}
                    playing_as = (p.get("playing_as") if isinstance(p, dict) else "white") or "white"
                    if str(playing_as).lower() not in ("white", "black"):
                        playing_as = "white"
                    return ("start_chess", {"reason": p.get("reason", "Chess board visible.") if isinstance(p, dict) else "Chess board visible.", "playing_as": str(playing_as).lower()}, messages, thought_history, img)
                if a in ("CLICK", "CLICK_CURRENT", "CLICK_XY", "MOVE_TO", "DOUBLE_CLICK", "TYPE_IN", "MENU", "KEYS", "KEY_PRESS", "TYPE_TEXT", "TASK_COMPLETE", "OPEN_APP", "OPEN_URL"):
                    p = obj.get("parameters", obj)
                    act = "KEYS" if a == "KEY_PRESS" else a
                    return ("execute", {"action": act, "parameters": p if isinstance(p, dict) else {}}, messages, thought_history, img)
            # If raw text looks like SCREEN_LOADING, treat as screen_loading so we continue instead of stopping
            if text and "SCREEN_LOADING" in text.upper() and "action" in text.lower():
                return ("screen_loading", None, messages, thought_history, img)
            # Fallback: model put action in message (e.g. {"action":"comment","message":"CLICK_CURRENT {}"}) — run it, don't stop
            msg = data.get("message", text) if data else text
            bare = _parse_bare_action(str(msg))
            if bare:
                return ("execute", {"action": bare, "parameters": {}}, messages, thought_history, img)
            return ("comment", data.get("message", text), messages, thought_history, img)
        except Exception as e:
            return ("comment", f"API error: {e}", conversation_messages or [], thought_history, None)

    # ==================================================================
    #  Stats
    # ==================================================================
    def _update_stats(self, status):
        mode = self._mode or "--"
        cycles = self.chess.cycle_count if self._mode == "chess" else 0
        actions = self.chess.move_count if self._mode == "chess" else self._screen_action_count
        self.root.after(0, lambda: self.stats_label.config(
            text=f"Mode: {mode}\n"
                 f"Cycles: {cycles}\n"
                 f"Actions: {actions}\n"
                 f"Status: {status}"))

    # ==================================================================
    #  Chess model loading
    # ==================================================================
    def _load_chess(self):
        ok = self.chess.load_models()

        def _done():
            if ok:
                self.status_label.config(text="Ready", fg="#4CAF50")
                self.log("Chess engine ready.", "info")
                self.log("Type a task and press Start.\n", "info")
            else:
                self.status_label.config(text="Chess load error", fg="#f44336")

        self.root.after(0, _done)

    # ==================================================================
    #  Start / Stop
    # ==================================================================
    def _toggle(self):
        if self._running:
            self._stop()
        else:
            self._start()

    def _start(self):
        task = self.task_text.get("1.0", tk.END).strip()
        if not task:
            self.log("Enter a task first!", "error")
            return

        self._running = True
        self.start_btn.config(text="Stop", bg="#f44336",
                              activebackground="#d32f2f")
        self.log("Workflow: Screenshot → Router (chess/execute/comment) → if execute: loop [run action → screenshot → Gemini next] until done.", "dim")

        if self.gemini_api_key_var.get().strip():
            self._thread = threading.Thread(
                target=self._start_vision_flow, daemon=True, args=(task,))
            self._thread.start()
        else:
            if is_chess_task(task):
                self._mode = "chess"
                self._start_chess()
            else:
                self.log("Set Gemini API key in Settings.", "warning")
                self._running = False
                self.start_btn.config(text="Start", bg="#4CAF50",
                                      activebackground="#388E3C")

    def _stop(self):
        self._running = False
        self.start_btn.config(text="Start", bg="#4CAF50",
                              activebackground="#388E3C")
        self.status_label.config(text="Stopped", fg="#FF9800")
        self.log("\nAGENT STOPPED", "header")
        if self._mode == "chess":
            self.log(f"  Moves played: {self.chess.move_count}", "info")
            self.log(f"  Scans: {self.chess.cycle_count}\n", "info")
        elif self._mode == "screen_control":
            self.log(f"  Screen actions: {self._screen_action_count}\n", "info")

    def _start_vision_flow(self, task):
        """Take screenshot, run Gemini router, then _on_router_done on main thread."""
        try:
            # "1234" or "1234, <feedback text>" triggers feedback storage.
            task_stripped = task.strip()
            is_feedback_code = task_stripped.startswith("1234")
            feedback_text = task_stripped[4:].lstrip().lstrip(",").strip() if is_feedback_code else ""

            # If user typed "1234, <their message>", store it directly without calling the model.
            if is_feedback_code and feedback_text:
                self.log("Storing feedback...", "action")
                payload = {
                    "user_correction": feedback_text,
                    "what_was_wrong": feedback_text,
                    "correct_approach": feedback_text,
                }
                self._on_router_done(feedback_text, "store_feedback", payload, "", None)
                return

            # "open X" / "launch X" → run Win+R instantly only when X is a single app (no "and", no comma)
            task_lower = task_stripped.lower()
            if not is_feedback_code and (task_lower.startswith("open ") or task_lower.startswith("launch ")):
                app_name = task_stripped[5:].strip() if task_lower.startswith("open ") else task_stripped[7:].strip()
                # Compound tasks like "open google and play chess" go to the router so the model can open Chrome then handle chess
                if app_name and " and " not in app_name.lower() and "," not in app_name:
                    self.log("Opening app directly (Win+R)...", "action")
                    ok, msg = _open_program(app_name)
                    self.root.after(0, lambda: self._on_open_app_done(ok, msg))
                    return

            # Just "1234" with no text: use model to infer feedback from screenshot.
            router_task = (
                "User requested feedback from this screenshot. Look at the screenshot and return exactly one JSON: "
                '{"action": "store_feedback", "feedback": {"user_correction": "...", "what_was_wrong": "...", "correct_approach": "..."}} '
                "with the three fields filled from what you see (e.g. what the user might be correcting)."
                if is_feedback_code
                else task
            )

            self.log("\n--- PHASE 1: Router (screenshot + first decision) ---", "header")
            self.log("Taking screenshot...", "action")
            self._hide_for_screenshot()
            time.sleep(0.3)
            screenshot, cursor_xy, screen_size = _take_screenshot_with_cursor()
            ui_elements = get_ui_elements() if _PYWINAUTO_OK else []
            self._show_after_screenshot()
            # Annotated image (with cursor marker) will be shown by _run_router

            self.log("Asking Gemini (router): chess / execute one action / comment?", "action")
            self.root.after(0, lambda: self.status_label.config(text="Calling Gemini...", fg="#FF9800"))

            action, payload, raw_response = "comment", "Router did not run.", ""
            MAX_ROUTER_ATTEMPTS = 5   # retries for API errors
            MAX_LOADING_RETRIES = 5   # retries for SCREEN_LOADING
            loading_retries = 0
            router_attempt = 0

            while self._running:
                try:
                    action, payload, raw_response = self._run_router(
                        router_task, screenshot, ui_elements,
                        cursor_xy=cursor_xy, screen_size=screen_size,
                    )
                    # Successful call — check if screen is still loading
                    if action == "screen_loading":
                        if loading_retries >= MAX_LOADING_RETRIES:
                            self.log("  Screen still loading after max retries; stopping.", "dim")
                            break
                        loading_retries += 1
                        self.log(f"  Screen still loading (router, retry {loading_retries}/{MAX_LOADING_RETRIES}); taking another screenshot in 2s...", "action")
                        time.sleep(2)
                        self._hide_for_screenshot()
                        time.sleep(0.3)
                        screenshot, cursor_xy, screen_size = _take_screenshot_with_cursor()
                        ui_elements = get_ui_elements() if _PYWINAUTO_OK else []
                        self._show_after_screenshot()
                        router_attempt = 0  # reset API retry counter after a fresh screenshot
                        continue
                    break  # Got a valid action
                except Exception as e:
                    router_attempt += 1
                    err_str = str(e)
                    if router_attempt < MAX_ROUTER_ATTEMPTS:
                        wait = [3, 5, 8, 15][min(router_attempt - 1, 3)]
                        self.log(f"  Router attempt {router_attempt}/{MAX_ROUTER_ATTEMPTS} failed ({err_str[:80]}); retrying in {wait}s...", "dim")
                        time.sleep(wait)
                        continue
                    # All retries exhausted
                    self.log(f"Router error after {MAX_ROUTER_ATTEMPTS} attempts: {err_str}", "error")
                    action, payload, raw_response = "comment", f"Error: {err_str}", ""
                    break

            self.root.after(0, lambda: self._on_router_done(task, action, payload, raw_response, screenshot))
        except Exception as e:
            self.log(f"Vision flow error: {e}", "error")
            import traceback
            self.log(traceback.format_exc(), "error")
            self._running = False
            self.root.after(0, lambda: self.start_btn.config(
                text="Start", bg="#4CAF50", activebackground="#388E3C"))
            self.root.after(0, lambda: self.status_label.config(text="Ready", fg="#4CAF50"))

    def _on_open_app_done(self, ok: bool, msg: str):
        """Called after instant open-app (Win+R); log result and stop."""
        self._running = False
        if ok:
            self.log(f"  {msg}", "info")
            self.root.after(0, lambda: self._update_stats("Opened"))
        else:
            self.log(f"  Failed: {msg}", "error")
            self.root.after(0, lambda: self._update_stats(f"Error: {msg[:30]}"))
        self.root.after(0, lambda: self.start_btn.config(text="Start", bg="#4CAF50", activebackground="#388E3C"))
        self.root.after(0, lambda: self.status_label.config(text="Ready", fg="#4CAF50"))

    def _on_router_done(self, task, action, payload, raw_response, screenshot):
        """Run on main thread: log thinking from raw_response, then branch on action."""
        self.status_label.config(text="Ready", fg="#4CAF50")
        if raw_response:
            think_match = re.search(r"<thinking\s*>.*?</thinking\s*>", raw_response, re.DOTALL | re.IGNORECASE)
            if think_match:
                inner = re.search(r"<thinking\s*>(.*?)</thinking\s*>", think_match.group(0), re.DOTALL | re.IGNORECASE)
                thinking = inner.group(1).strip() if inner else think_match.group(0)
                self.log("  ━━━ MODEL THINKING (router) ━━━", "header")
                self.log("  [Router] Reasoning:", "header")
                for line in thinking.split("\n"):
                    if line.strip():
                        self.log(f"    {line.strip()}", "thought")
                self.log("  ━━━ End thinking → action ━━━", "dim")
            else:
                # No <thinking> block: show any text before the JSON as reasoning preview
                self.log("  ━━━ MODEL THINKING (router) ━━━", "header")
                before_brace = raw_response.split("{")[0].strip() if "{" in raw_response else raw_response.strip()
                if before_brace and len(before_brace) > 2:
                    self.log("  [Router] (reasoning before JSON):", "header")
                    for line in before_brace.split("\n")[:8]:
                        if line.strip():
                            self.log(f"    {line.strip()}", "thought")
                else:
                    self.log("  [Router] (no <thinking> block; model went straight to JSON)", "dim")
                self.log("  ━━━ End → action ━━━", "dim")

        if action == "store_feedback":
            # Never store "1234" in prompt history: strip it from user_message
            user_message = (task or "").strip()
            if user_message.lower().startswith("1234"):
                user_message = user_message[4:].lstrip().lstrip(",").strip()
            if not user_message:
                user_message = "Feedback from screenshot"
            entry = {
                "type": "feedback",
                "user_message": user_message,
                "feedback": payload if isinstance(payload, dict) else {"raw": str(payload)},
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            _save_prompt_entry(entry)
            self.log("  Feedback stored for learning.", "info")
            fb_str = json.dumps(payload, indent=2)
            self.log(f"  {fb_str[:300]}{'...' if len(fb_str) > 300 else ''}", "dim")
            self._running = False
            self.start_btn.config(text="Start", bg="#4CAF50", activebackground="#388E3C")
            self.status_label.config(text="Ready", fg="#4CAF50")
            return

        if action == "start_chess":
            reason = payload.get("reason", "Chess board detected.") if isinstance(payload, dict) else payload
            playing_as = payload.get("playing_as", "white") if isinstance(payload, dict) else "white"
            self.log(f"  Claude: {reason}", "info")
            self.log(f"  Claude detected: playing as {playing_as}", "info")
            self.turn_var.set(playing_as)
            self._mode = "chess"
            self._start_chess()
            return

        if action == "execute":
            act = payload.get("action", "") if isinstance(payload, dict) else ""
            self.log(f"  Router → execute first action: {act}", "info")
            self.log("--- PHASE 2: Screen control loop (execute → screenshot → next action) ---", "header")
            self._mode = "screen_control"
            self._start_screen_control(task, screenshot, payload)
            return

        if action == "screen_loading":
            self.log("  Page still loading after retries; try again in a moment.", "warning")
            self._running = False
            self.start_btn.config(text="Start", bg="#4CAF50", activebackground="#388E3C")
            self.status_label.config(text="Ready", fg="#4CAF50")
            self._update_stats("Screen still loading")
            return

        # comment path
        msg = payload.get("message", payload) if isinstance(payload, dict) else payload
        msg_str = str(msg).strip() if msg else "(no message)"
        self.log(f"[Gemini] {msg_str}", "thought")
        self._running = False
        self.start_btn.config(text="Start", bg="#4CAF50", activebackground="#388E3C")
        self.status_label.config(text="Ready", fg="#4CAF50")
        if msg_str.startswith("{"):
            display = "Gemini replied (see log)"
        else:
            display = (msg_str[:28] + "…") if len(msg_str) > 28 else msg_str
        self._update_stats(display)

    # ------------------------------------------------------------------
    #  Chess mode
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    #  Screen control mode (mouse + keyboard)
    # ------------------------------------------------------------------
    def _start_screen_control(self, task, initial_screenshot, first_action):
        """Start screen control loop: execute first action, then loop for more."""
        self._screen_action_count = 0
        self.status_label.config(text="Screen control - running", fg="#4CAF50")
        self.log("Loop: will execute action → take screenshot → ask Gemini for next → repeat.", "info")
        self.log(f"  Task: {task}", "info")
        self.log("", "info")
        self._thread = threading.Thread(
            target=self._screen_control_loop,
            daemon=True,
            args=(task, first_action, initial_screenshot),
        )
        self._thread.start()

    def _screen_control_loop(self, task, first_action=None, initial_screenshot=None):
        """Loop: execute current action -> screenshot -> ask Gemini for next -> repeat until TASK_COMPLETE or stop."""
        handoff_to_chess = False
        try:
            # Use the screenshot dimensions that produced the current action (so model coords map correctly).
            if initial_screenshot is not None:
                sw, sh = initial_screenshot.size[0], initial_screenshot.size[1]
            else:
                _test_ss = pyautogui.screenshot()
                sw, sh = _test_ss.size
                del _test_ss
            self.log(f"  [DEBUG] coordinate mapping uses screenshot size: {sw}x{sh}", "dim")
            action_history = []
            conversation_messages = []
            thought_history = []
            screenshot_history = []  # list of (prepared_img, caption) — last 1 prior screenshot sent to model
            current_action = first_action
            turn = 0
            _last_click_pos = None   # (x, y) of last CLICK_CURRENT
            _same_click_count = 0    # consecutive CLICK_CURRENT at same position
            _last_was_move_to = None  # (x, y) target if last action was MOVE_TO, else None

            while self._running:
                # 1) Execute current action (if we have one and it's not TASK_COMPLETE)
                if current_action:
                    act = current_action.get("action", "")
                    params = current_action.get("parameters", current_action)
                    action_dict = {"action": act, "parameters": params}

                    if act == "TASK_COMPLETE":
                        msg = params.get("message", "Task complete")
                        self.log(f"  Done: {msg}", "info")
                        display = (str(msg)[:36] + "…") if len(str(msg)) > 36 else str(msg)
                        self.root.after(0, lambda m=msg: self._update_stats(m))
                        break

                    self.log(f"  Executing: {act} {params}", "action")
                    self._hide_for_screenshot()
                    time.sleep(0.05)
                    ok, result = _execute_action(action_dict, sw, sh)
                    time.sleep(0.3)
                    self._show_after_screenshot()

                    action_history.append(f"{act}: {result}")
                    self._screen_action_count += 1
                    self.root.after(0, lambda: self._update_stats(f"last: {act}"))
                    if ok:
                        self.log(f"  -> OK: {result}", "info")
                    else:
                        self.log(f"  -> Failed: {result}", "error")

                    # ── Repeated-click guard ──
                    import re as _re
                    _click_pos_match = _re.search(r"\((\d+),\s*(\d+)\)", result or "")
                    if act in ("CLICK_CURRENT", "CLICK_XY") and _click_pos_match:
                        _pos = (int(_click_pos_match.group(1)), int(_click_pos_match.group(2)))
                        if _last_click_pos and abs(_pos[0] - _last_click_pos[0]) < 30 and abs(_pos[1] - _last_click_pos[1]) < 30:
                            _same_click_count += 1
                        else:
                            _same_click_count = 1
                            _last_click_pos = _pos
                    else:
                        _same_click_count = 0
                        _last_click_pos = None

                    # ── Auto-click after MOVE_TO ──
                    # If cursor landed close enough to the intended target, click immediately
                    # instead of wasting a round-trip asking the model to verify.
                    if act == "MOVE_TO" and ok:
                        _mt_params = params if isinstance(params, dict) else {}
                        _mt_x, _mt_y = _mt_params.get("x"), _mt_params.get("y")
                        if _mt_x is not None and _mt_y is not None:
                            # Verify cursor actually landed near the target
                            cur_pos = pyautogui.position()
                            scale = 1024 / max(sw, sh) if max(sw, sh) > 1024 else 1.0
                            m_w = max(1, int(sw * scale))
                            m_h = max(1, int(sh * scale))
                            target_sx = int(round(int(_mt_x) * sw / m_w))
                            target_sy = int(round(int(_mt_y) * sh / m_h))
                            dist = ((cur_pos.x - target_sx)**2 + (cur_pos.y - target_sy)**2) ** 0.5
                            if dist < 25:
                                time.sleep(0.05)
                                pyautogui.click()
                                self.log(f"  -> Auto-clicked at ({cur_pos.x},{cur_pos.y}) (cursor within {int(dist)}px of target)", "info")
                                action_history.append(f"AUTO_CLICK: at ({cur_pos.x},{cur_pos.y})")
                                _last_was_move_to = None
                            else:
                                _last_was_move_to = (int(_mt_x), int(_mt_y))
                        else:
                            _last_was_move_to = None
                    elif act == "MOVE_TO":
                        _mt_params = params if isinstance(params, dict) else {}
                        _last_was_move_to = (_mt_params.get("x"), _mt_params.get("y"))
                    else:
                        _last_was_move_to = None

                if not self._running:
                    break

                # 2) Screenshot again (so Gemini sees the result of the action)
                self.log("  Taking screenshot...", "action")
                self._hide_for_screenshot()
                time.sleep(0.3)
                screenshot, cursor_xy, screen_size = _take_screenshot_with_cursor()
                ui_elements = get_ui_elements() if _PYWINAUTO_OK else []
                self._show_after_screenshot()
                # Annotated image (with cursor marker) will be shown by _ask_next_action

                if not self._running:
                    break

                # 3) Ask Gemini for next action (include prior screenshots so model can compare)
                turn += 1
                self.log(f"  --- Turn {turn}: asking Gemini for next action ---", "header")
                self.log("  Asking Gemini for next action...", "action")

                # ── Repeated-click nudge ──
                _stuck_hint = None
                if _same_click_count >= 2 and _last_click_pos:
                    _stuck_hint = (
                        f"⚠️ WARNING: You clicked at {_last_click_pos} {_same_click_count} times but the screen did NOT change. "
                        "That click is missing the target. STOP and try a DIFFERENT approach:\n"
                        "  • Use CLICK with the element name from the UI elements list (most reliable).\n"
                        "  • Re-read the grid lines carefully and use CLICK_XY with corrected coordinates.\n"
                        "  • Use KEYS (Tab, Enter, F5) to interact without clicking.\n"
                        "  • Use OPEN_URL if you need to navigate to a website.\n"
                        "Do NOT click the same position again."
                    )
                    self.log(f"  ⚠ Stuck: CLICK_CURRENT at {_last_click_pos} x{_same_click_count} — injecting corrective hint.", "error")

                # ── MOVE_TO commitment nudge ──
                # After a MOVE_TO the model MUST follow up with CLICK_CURRENT or
                # another MOVE_TO — nothing else.  Inject this as an override hint.
                _move_to_hint = None
                if _last_was_move_to is not None:
                    _move_to_hint = (
                        f"🎯 YOUR LAST ACTION WAS MOVE_TO {_last_was_move_to}. "
                        "The cursor is now at that position. The CLICK happens at the center point only.\n"
                        "YOU MUST NOW do ONE of only TWO things:\n"
                        "  1. CLICK_CURRENT — ONLY if the center point (CURSOR x,y) is on the button, not just near it.\n"
                        "  2. MOVE_TO {x, y} — if the crosshair missed or is only close to the target; use grid to aim at the button center.\n"
                        "DO NOT take any other action. Close/near is NOT enough — use CLICK_CURRENT only when the crosshair is clearly on the target center."
                    )

                while True:
                    action, payload, conversation_messages, thought_history, prepared_img = self._ask_next_action(
                        task, screenshot, action_history, conversation_messages, thought_history, ui_elements,
                        cursor_xy=cursor_xy, screen_size=screen_size,
                        prior_screenshot_parts=screenshot_history if screenshot_history else None,
                        stuck_hint=_stuck_hint,
                        move_to_hint=_move_to_hint,
                    )
                    if not self._running:
                        break
                    if action == "screen_loading":
                        self.log("  Screen still loading, taking another screenshot in 2s...", "action")
                        time.sleep(2)
                        self._hide_for_screenshot()
                        time.sleep(0.3)
                        screenshot, cursor_xy, screen_size = _take_screenshot_with_cursor()
                        ui_elements = get_ui_elements() if _PYWINAUTO_OK else []
                        self._show_after_screenshot()
                        continue
                    if action == "add_feedback":
                        p = payload if isinstance(payload, dict) else {}
                        msg = p.get("message") or p.get("user_correction") or str(p.get("feedback", "")) or "Self-feedback"
                        wrong = p.get("what_was_wrong") or msg
                        approach = p.get("correct_approach") or "Apply the lesson next time."
                        entry = {
                            "type": "feedback",
                            "user_message": f"Agent self-feedback: {msg[:200]}",
                            "feedback": {"user_correction": msg, "what_was_wrong": wrong, "correct_approach": approach},
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        }
                        _save_prompt_entry(entry)
                        self.log("  Self-feedback stored for next time.", "info")
                        continue
                    break
                if not self._running:
                    break

                if action == "execute":
                    current_action = payload
                    next_act = payload.get("action", "") if isinstance(payload, dict) else ""
                    self.log(f"  → Will execute: {next_act} (loop continues)", "info")
                    # Reset stuck counter if the model chose something other than CLICK_CURRENT
                    if next_act != "CLICK_CURRENT":
                        _same_click_count = 0
                        _last_click_pos = None
                    # Reset MOVE_TO commitment once the model correctly follows through
                    if next_act in ("CLICK_CURRENT", "MOVE_TO"):
                        pass  # keep _last_was_move_to for MOVE_TO; clear after CLICK_CURRENT
                    if next_act == "CLICK_CURRENT":
                        _last_was_move_to = None
                    # Use this screenshot's dimensions for the next execution (action was generated from this image).
                    sw, sh = screenshot.size[0], screenshot.size[1]
                    # Keep 1 prior screenshot so model can compare before/after.
                    if prepared_img is not None:
                        act = payload.get("action", "")
                        params = payload.get("parameters", {})
                        caption = f"Previous turn: you saw this, then did: {act} {params}."
                        screenshot_history = [(prepared_img, caption)]
                elif action == "start_chess":
                    # Chess board is visible — hand off to chess agent (YOLO + Stockfish); do not set _running = False
                    handoff_to_chess = True
                    reason = payload.get("reason", "Chess board visible.") if isinstance(payload, dict) else "Chess board visible."
                    playing_as = (payload.get("playing_as", "white") if isinstance(payload, dict) else "white")
                    if str(playing_as).lower() not in ("white", "black"):
                        playing_as = "white"
                    self.log(f"  Chess board detected: {reason}; playing as {playing_as}. Starting chess agent.", "info")
                    self.root.after(0, lambda p=playing_as: self.turn_var.set(p))
                    self.root.after(0, lambda: setattr(self, "_mode", "chess"))
                    self.root.after(0, self._start_chess)
                    break
                elif action == "screen_loading":
                    # Already handled above (continue with new screenshot); should not reach here
                    self.log("  Screen still loading, taking another screenshot in 2s...", "action")
                    time.sleep(2)
                    self._hide_for_screenshot()
                    time.sleep(0.3)
                    screenshot, cursor_xy, screen_size = _take_screenshot_with_cursor()
                    ui_elements = get_ui_elements() if _PYWINAUTO_OK else []
                    self._show_after_screenshot()
                    continue
                else:
                    # Check if payload is actually SCREEN_LOADING (e.g. mis-parsed as comment) — continue instead of stopping
                    payload_str = str(payload).strip() if payload else "(empty)"
                    if "SCREEN_LOADING" in payload_str.upper() and "action" in payload_str.lower():
                        self.log("  Screen still loading (detected from response), taking another screenshot in 2s...", "action")
                        time.sleep(2)
                        self._hide_for_screenshot()
                        time.sleep(0.3)
                        screenshot, cursor_xy, screen_size = _take_screenshot_with_cursor()
                        ui_elements = get_ui_elements() if _PYWINAUTO_OK else []
                        self._show_after_screenshot()
                        continue
                    # Gemini returned a comment instead of an action - log and stop
                    self.log(f"  Gemini replied with comment (stopping): {payload_str[:200]}", "warning")
                    if not payload_str or payload_str.lower() in ("action", "ok", "done"):
                        self.log("  (Gemini may have returned non-JSON or vague text; check model output.)", "dim")
                    if payload_str.startswith("{"):
                        display = "Stopped (see log)"
                    else:
                        display = (payload_str[:28] + "…") if len(payload_str) > 28 else payload_str or "Stopped"
                    self.root.after(0, lambda d=display: self._update_stats(f"Stopped: {d}"))
                    break

        except Exception as e:
            self.log(f"SCREEN CONTROL ERROR: {e}", "error")
            import traceback
            self.log(traceback.format_exc(), "error")
            err_msg = str(e)[:40]
            self.root.after(0, lambda m=err_msg: self._update_stats(f"Error: {m}"))
        finally:
            if not handoff_to_chess:
                self._running = False
                self.root.after(0, lambda: self.start_btn.config(
                    text="Start", bg="#4CAF50", activebackground="#388E3C"))
                self.root.after(0, lambda: self.status_label.config(
                    text="Stopped", fg="#FF9800"))

    # ------------------------------------------------------------------
    #  Chess mode
    # ------------------------------------------------------------------
    def _start_chess(self):
        if not self.chess.ready:
            self.log("Chess engine not loaded yet!", "error")
            self._running = False
            self.start_btn.config(text="Start", bg="#4CAF50",
                                  activebackground="#388E3C")
            return

        self.chess.reset()
        self.status_label.config(text="Chess - running", fg="#4CAF50")

        self.log(f"\n{'='*55}", "header")
        self.log("CHESS AGENT STARTED", "header")
        self.log(f"  Playing as: {self.turn_var.get()}", "info")
        self.log(f"  Scan interval: {self.interval_var.get()}s", "info")
        self.log(f"  Stockfish depth: {self.depth_var.get()}", "info")
        self.log(f"{'='*55}\n", "header")

        self._thread = threading.Thread(target=self._chess_loop, daemon=True)
        self._thread.start()

    def _chess_loop(self):
        """Background thread for chess auto-play."""
        _last_move_attempted = None   # tracks last Stockfish move we tried to play
        _same_move_attempts  = 0      # how many times we've tried the same move in a row
        try:
            while self._running:
                self.log(f"-- Scan #{self.chess.cycle_count + 1} --", "action")

                # screenshot
                self._hide_for_screenshot()
                t0 = time.time()
                ss = pyautogui.screenshot()
                cap_t = time.time() - t0
                self._show_after_screenshot()

                # analyse
                r = self.chess.analyze(
                    ss,
                    conf=self.conf_var.get(),
                    depth=self.depth_var.get(),
                    turn=self.turn_var.get(),
                    force_move=False)

                if r["annotated"]:
                    self.root.after(0, self._show_image, r["annotated"])

                # --- act on result ---
                if r["status"] == "move":
                    best = r["best_move"]
                    fsq, tsq = best[:2], best[2:4]
                    promo = f"={best[4:].upper()}" if len(best) > 4 else ""

                    # ── Same-move retry guard ──
                    # If Stockfish keeps suggesting the same move and the board
                    # isn't changing (drag not landing), give up after 5 attempts
                    # and force a full re-analysis by clearing _last_fen.
                    if best == _last_move_attempted:
                        _same_move_attempts += 1
                    else:
                        _same_move_attempts = 1
                        _last_move_attempted = best

                    if _same_move_attempts > 5:
                        self.log(
                            f"  Move {best} attempted {_same_move_attempts} times without "
                            "board change — resetting board state and waiting for next scan.",
                            "error"
                        )
                        self.chess._last_fen = None
                        _same_move_attempts = 0
                        _last_move_attempted = None
                        # Skip the drag this turn; next scan will re-evaluate.
                        self._update_stats(f"stuck on {fsq}->{tsq} — resetting")
                    else:
                        # Save the board state BEFORE our move so we can detect
                        # whether the drag was actually registered by chess.com.
                        fen_before_move = r["fen"].split(" ")[0]

                        # Execute drag (GUI hidden so chess.com board is exposed)
                        self._hide_for_screenshot()
                        self.chess.execute_move(
                            best, r["board_box"], r["orientation"],
                            click_delay=self.click_delay_var.get())
                        self._show_after_screenshot()

                        self.log(f"  Move #{self.chess.move_count} played!", "info")
                        self._update_stats(f"played {fsq}->{tsq}{promo}")

                        # Wait for chess.com / lichess animation to complete
                        time.sleep(1.2)

                        # Re-scan to track board state
                        self._hide_for_screenshot()
                        ss2 = pyautogui.screenshot()
                        self._show_after_screenshot()
                        self.chess.capture_post_move_fen(ss2, self.conf_var.get())

                        # ── Move-registration check ──
                        # If the board looks identical to before our drag, the site
                        # didn't register it.  Reset _last_fen so the next scan
                        # retries instead of waiting for a phantom opponent move.
                        if (self.chess._last_fen is None or
                                self.chess._last_fen == fen_before_move):
                            self.log(
                                "  Board unchanged after drag — move may not have "
                                "registered; will retry next scan.",
                                "dim"
                            )
                            self.chess._last_fen = None
                        else:
                            # Move registered — reset the retry counter.
                            _same_move_attempts = 0
                            _last_move_attempted = None

                elif r["status"] == "game_over":
                    self.log(f"  GAME OVER: {r.get('message', '')}", "header")
                    self._update_stats(f"game over: {r.get('message','')}")
                    break

                elif r["status"] == "waiting":
                    self._update_stats("waiting for opponent")

                elif r["status"] == "no_board":
                    # Board temporarily invisible (e.g. animation between moves).
                    # Don't stop — just retry on the next cycle.
                    self.log("  No board detected — retrying next scan", "dim")
                    self._update_stats("no board — retrying")

                else:
                    self._update_stats("detection error")

                # After playing a move the opponent often responds within 1–2s,
                # so use a shorter poll interval (1s) in that case; otherwise use
                # the full user-configured interval to avoid hammering the CPU.
                if r["status"] == "move":
                    wait = 1.5
                elif r["status"] == "waiting":
                    wait = self.interval_var.get()
                else:
                    wait = max(1.0, self.interval_var.get() * 0.5)

                end = time.time() + wait
                while time.time() < end and self._running:
                    time.sleep(0.05)

        except Exception as e:
            self.log(f"CHESS LOOP ERROR: {e}", "error")
            import traceback
            self.log(traceback.format_exc(), "error")
        finally:
            self._running = False
            self.root.after(0, lambda: self.start_btn.config(
                text="Start", bg="#4CAF50", activebackground="#388E3C"))
            self.root.after(0, lambda: self.status_label.config(
                text="Stopped", fg="#FF9800"))

    # ==================================================================
    #  Thread-safe hide / show (for screenshots & clicks)
    # ==================================================================
    def _hide_and_signal(self):
        self.root.withdraw()
        self.root.update()
        self._hide_event.set()

    def _show_and_signal(self):
        self.root.deiconify()
        self.root.update()
        self._show_event.set()

    def _hide_for_screenshot(self):
        self._hide_event.clear()
        self.root.after(0, self._hide_and_signal)
        self._hide_event.wait(timeout=2.0)

    def _show_after_screenshot(self):
        self._show_event.clear()
        self.root.after(0, self._show_and_signal)
        self._show_event.wait(timeout=2.0)

    def _minimize_for_execution(self):
        """Minimize window (alternative to full hide). Screen control now uses _hide_for_screenshot so GUI is fully closed during actions."""
        self._hide_event.clear()
        self.root.after(0, lambda: (self.root.iconify(), self.root.update(), self._hide_event.set()))
        self._hide_event.wait(timeout=2.0)

    def _restore_after_execution(self):
        self._show_event.clear()
        self.root.after(0, lambda: (self.root.deiconify(), self.root.update(), self._show_event.set()))
        self._show_event.wait(timeout=2.0)


# ======================================================================
def main():
    root = tk.Tk()
    AgentGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
