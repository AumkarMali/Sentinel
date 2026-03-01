"""
Shared agent logic (no tkinter). Used by gui.py (reference) and agent_backend.py (headless runner).
"""
import os
import sys
import time
import json
import re

# Windows DPI awareness — MUST be set before any GUI/screenshot imports.
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

import pyautogui
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from pywinauto_actions import get_ui_elements, execute_action as _execute_pywinauto
    _PYWINAUTO_OK = True
except ImportError:
    _PYWINAUTO_OK = False
    get_ui_elements = lambda: []

try:
    from pynput.keyboard import Controller as PynputController, Key as PynputKey
    _PYNPUT_AVAILABLE = True
except ImportError:
    PynputController = PynputKey = None
    _PYNPUT_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
PROMPT_HISTORY_PATH = os.path.join(BASE_DIR, "prompt_history.json")
MAX_HISTORY_ENTRIES = 100


def _load_config():
    try:
        if os.path.isfile(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def _save_config(gemini_api_key: str = "", gemini_model: str = "", **kwargs):
    try:
        data = {"gemini_api_key": (gemini_api_key or "").strip(), "gemini_model": (gemini_model or "").strip()}
        for k, v in kwargs.items():
            if v is not None:
                data[k] = v
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass


def _load_prompt_history():
    try:
        if os.path.isfile(PROMPT_HISTORY_PATH):
            with open(PROMPT_HISTORY_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        pass
    return []


def _save_prompt_entry(entry: dict):
    history = _load_prompt_history()
    history.append(entry)
    if len(history) > MAX_HISTORY_ENTRIES:
        history = history[-MAX_HISTORY_ENTRIES:]
    try:
        with open(PROMPT_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except OSError:
        pass


def _build_learning_context(max_entries: int = 15):
    history = _load_prompt_history()
    out = []
    feedback_entries = [e for e in history if e.get("type") == "feedback"][-max_entries:]
    if feedback_entries:
        out.append("LEARNING FROM USER FEEDBACK — you MUST apply these corrections.")
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
        out.append("Apply the correct_approach from the feedback above.")
    return "\n".join(out) if out else ""


def _parse_json(text: str):
    text = text.strip()
    start_candidates = [i for i, c in enumerate(text) if c == "{"]
    if not start_candidates:
        return None
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
        fixed = re.sub(r",\s*}", "}", fixed).replace(r",\s*]", "]")
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
    return None


ROUTER_SYSTEM_TAIL = """Prequisite knowledge:
 CRITICAL - How to determine playing_as: The human player's pieces are ALWAYS at the BOTTOM of the board.
  - DARK/GREY/BROWN pieces at the bottom → playing_as: "black"
  - LIGHT/WHITE/CREAM pieces at the bottom → playing_as: "white"

Respond with exactly one JSON object, no markdown:
- Chess task + board visible: {"action": "start_chess", "reason": "...", "playing_as": "white" or "black"}
- Chess task, NO board: {"action": "comment", "message": "No chess board visible."}
- Task requires CONTROLLING the screen: return ONE execute action (OPEN_APP, OPEN_URL, MOVE_TO, CLICK_CURRENT, etc.).
  * OPEN_APP — ONLY for Windows programs (Chrome, Edge, Notepad). NEVER for websites.
  * OPEN_URL — For navigating when a browser is open. {"action": "OPEN_URL", "parameters": {"url": "lichess.org"}}
  * MOVE_TO: {"action": "MOVE_TO", "parameters": {"x": 400, "y": 300}}
  * CLICK_CURRENT: {"action": "CLICK_CURRENT", "parameters": {}}
- Observational only: {"action": "comment", "message": "your observation"}
CRITICAL — NEVER use KEYS ["win"] or ["win", "r"] to open apps. Use OPEN_APP only.
Your reply MUST start with <thinking>...</thinking> then one JSON line."""


def _parse_router_response(data, text: str, raw_response: str):
    if data is None:
        return ("comment", "Model did not return valid JSON.")
    raw_action = data.get("action") or "comment"
    action = str(raw_action).lower().strip().replace(" ", "_").replace("-", "_")
    a_norm = str(raw_action).upper().strip().replace(" ", "_").replace("-", "_")
    if a_norm == "SCREEN_LOADING":
        return ("screen_loading", None)
    if a_norm in ("CLICK", "CLICK_CURRENT", "MOVE_TO", "DOUBLE_CLICK", "TYPE_IN", "MENU", "KEYS", "KEY_PRESS", "TYPE_TEXT", "TASK_COMPLETE", "OPEN_APP", "OPEN_URL"):
        params = data.get("parameters", data)
        act = "KEYS" if a_norm == "KEY_PRESS" else a_norm
        return ("execute", {"action": act, "parameters": params if isinstance(params, dict) else {}})
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
    msg = data.get("message", data.get("reason", text))
    return ("comment", msg or "No message from model.")


def _move_cursor(x: int, y: int) -> bool:
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


def _pynput_key(name):
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
    return _open_program(element_name)


def _open_program(program_name: str) -> tuple:
    if not program_name or len(program_name) > 80:
        return False, "Invalid program name"
    name = program_name.strip()
    search = name.split(",")[0].split(" - ")[0].split(" (")[0].strip() or name
    search_lower = search.lower()
    website_indicators = (
        ".com", ".org", ".net", ".io", "http://", "https://", "www.",
        "lichess", "chess.com", "youtube", "google", "github",
    )
    if any(ind in search_lower for ind in website_indicators):
        return False, "OPEN_APP cannot open websites. Use OPEN_URL for sites."
    if "microsoft edge" in search_lower or search_lower == "edge":
        search = "msedge"
    elif "google chrome" in search_lower or search_lower == "chrome":
        search = "chrome"
    elif "notepad" in search_lower:
        search = "notepad"
    elif "code" in search_lower or "vscode" in search_lower:
        search = "code"
    elif "calculator" in search_lower or "calc" in search_lower:
        search = "calc"
    elif "cmd" in search_lower:
        search = "cmd"
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
    if not url or len(url) > 500:
        return False, "Invalid URL"
    url = url.strip()
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
    try:
        action = (action_dict.get("action") or "").upper().replace(" ", "_")
        params = action_dict.get("parameters") or action_dict
        if isinstance(params, dict) and "action" in params:
            params = params.get("parameters", params)

        if action == "OPEN_APP":
            program = (params.get("program") or params.get("app_name") or "").strip()
            if not program:
                return False, "OPEN_APP requires 'program'"
            return _open_program(program)
        if action == "OPEN_URL":
            url = (params.get("url") or "").strip()
            if not url:
                return False, "OPEN_URL requires 'url'"
            return _open_url(url)
        if action == "CLICK" and not (params.get("element") or "").strip():
            pyautogui.click()
            return True, "Clicked at current cursor position"
        if action in ("CLICK", "DOUBLE_CLICK", "TYPE_IN", "MENU", "TASK_COMPLETE") and _PYWINAUTO_OK:
            ok, msg = _execute_pywinauto(action_dict)
            if ok:
                return True, msg
            if action in ("CLICK", "DOUBLE_CLICK"):
                el = (params.get("element") or "").strip()
                if el:
                    return _open_app_via_start(el)
            return False, msg
        if action == "MOVE_TO":
            x, y = params.get("x"), params.get("y")
            if x is None or y is None:
                return False, "MOVE_TO requires x and y"
            try:
                x, y = int(x), int(y)
            except (TypeError, ValueError):
                return False, "MOVE_TO x,y must be numbers"
            scale = 1024 / max(screen_w, screen_h) if max(screen_w, screen_h) > 1024 else 1.0
            model_w = max(1, int(screen_w * scale))
            model_h = max(1, int(screen_h * scale))
            actual_x = max(0, min(screen_w - 1, int(round(x * screen_w / model_w))))
            actual_y = max(0, min(screen_h - 1, int(round(y * screen_h / model_h))))
            ok = _move_cursor(actual_x, actual_y)
            return ok, f"Moved cursor to ({actual_x}, {actual_y})"
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
            keys_lower = [str(k).lower() for k in keys]
            win_keys = {"win", "winleft", "winright", "super", "windows", "command"}
            if any(k in win_keys for k in keys_lower):
                return False, "BLOCKED: Use OPEN_APP to open apps, not KEYS."
            key_map = {"control": "ctrl", "windows": "win", "command": "win", "win": "winleft", "super": "winleft"}
            keys = [key_map.get(str(k).lower(), k) for k in keys]
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
            if len(keys) == 1:
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
            return False, "pywinauto not available"
        return False, f"Unknown action: {action}"
    except Exception as e:
        return False, str(e)


def _draw_grid_on_image(img, spacing: int = 100):
    try:
        draw = ImageDraw.Draw(img)
        w, h = img.size
        minor_col, major_col, label_col = (70, 70, 70), (110, 110, 110), (160, 200, 255)
        try:
            font = ImageFont.truetype("arial.ttf", 9)
        except Exception:
            font = ImageFont.load_default()
        for x in range(0, w, spacing):
            col = major_col if x % (spacing * 2) == 0 else minor_col
            draw.line([(x, 0), (x, h)], fill=col, width=1)
            if x > 0:
                draw.text((x + 2, 2), str(x), fill=label_col, font=font)
        for y in range(0, h, spacing):
            col = major_col if y % (spacing * 2) == 0 else minor_col
            draw.line([(0, y), (w, y)], fill=col, width=1)
            if y > 0:
                draw.text((2, y + 2), str(y), fill=label_col, font=font)
    except Exception:
        pass


def _draw_cursor_on_image(img, cursor_x_img: int, cursor_y_img: int):
    try:
        mw, mh = img.size
        cx = max(0, min(mw - 1, cursor_x_img))
        cy = max(0, min(mh - 1, cursor_y_img))
        draw = ImageDraw.Draw(img)
        r, arm, dot, lw = 2, 5, 1, 1
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline="#ff0000", width=lw)
        draw.line([cx - arm, cy, cx + arm, cy], fill="#ff0000", width=lw)
        draw.line([cx, cy - arm, cx, cy + arm], fill="#ff0000", width=lw)
        draw.ellipse([cx - dot, cy - dot, cx + dot, cy + dot], fill="#ff0000", outline="#ffffff")
        try:
            font = ImageFont.truetype("arial.ttf", 11)
        except Exception:
            font = ImageFont.load_default()
        label = f"CURSOR ({cx},{cy})"
        tx, ty = cx + arm + 4, cy - 10
        if tx + 80 > mw:
            tx = cx - arm - 85
        if ty < 0:
            ty = cy + arm + 2
        for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            draw.text((tx + dx, ty + dy), label, fill="#000000", font=font)
        draw.text((tx, ty), label, fill="#ffff00", font=font)
    except Exception:
        pass


def _take_screenshot_with_cursor():
    cursor = pyautogui.position()
    screenshot = pyautogui.screenshot()
    screen_size = pyautogui.size()
    return screenshot, (cursor.x, cursor.y), (screen_size[0], screen_size[1])


def _prepare_image_for_model(screenshot, cursor_xy, screen_size):
    img = screenshot.convert("RGB")
    ss_w, ss_h = img.size
    scr_w, scr_h = screen_size
    cur_ss_x = cursor_xy[0] * ss_w / max(1, scr_w)
    cur_ss_y = cursor_xy[1] * ss_h / max(1, scr_h)
    scale = 1024 / max(ss_w, ss_h) if max(ss_w, ss_h) > 1024 else 1.0
    if scale < 1.0:
        nw, nh = int(ss_w * scale), int(ss_h * scale)
        img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    model_w, model_h = img.size
    cx_img = int(cur_ss_x * scale)
    cy_img = int(cur_ss_y * scale)
    cx_img = max(0, min(model_w - 1, cx_img))
    cy_img = max(0, min(model_h - 1, cy_img))
    _draw_grid_on_image(img)
    _draw_cursor_on_image(img, cx_img, cy_img)
    return img, model_w, model_h, (cx_img, cy_img)


GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

# Action-loop system prompt (tail) — used by agent_backend for _ask_next_action
ACTION_SYSTEM_TAIL = """
CRITICAL - Your reply MUST start with <thinking>...</thinking> then one JSON line.
COORDINATE GRID: Use grid labels to derive MOVE_TO coordinates. CLICK_CURRENT only when crosshair is centered ON the target.
OPEN_APP — Windows programs only. OPEN_URL — when browser is open. MOVE_TO then CLICK_CURRENT for buttons.
When a chess BOARD is visible: return {"action": "start_chess", "parameters": {"reason": "Board visible", "playing_as": "white" or "black"}}. playing_as = color at BOTTOM of board.
Actions: OPEN_APP, OPEN_URL, MOVE_TO, CLICK_CURRENT, CLICK, KEYS, TYPE_TEXT, TASK_COMPLETE, SCREEN_LOADING, start_chess.
CRITICAL — Never use KEYS to open apps. Use OPEN_APP."""
