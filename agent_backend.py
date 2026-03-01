"""
Headless agent runner (no tkinter). Used by Electron: takes a task, runs the full flow,
logs JSON lines to stdout. Electron hides/shows its windows around screenshots via protocol messages.
"""
import json
import re
import sys
import time

import pyautogui


def _signal_hide():
    """Tell Electron to hide its overlay windows before a screenshot."""
    print(json.dumps({"type": "hide_windows"}), flush=True)
    time.sleep(0.15)


def _signal_show():
    """Tell Electron to show its overlay windows after a screenshot."""
    print(json.dumps({"type": "show_windows"}), flush=True)


from agent_core import (
    _build_learning_context,
    _execute_action,
    _load_config,
    _open_program,
    _open_url,
    _parse_json,
    _prepare_image_for_model,
    _save_prompt_entry,
    _take_screenshot_with_cursor,
    ACTION_SYSTEM_TAIL,
    get_ui_elements,
    GEMINI_MODELS,
)
from chess_agent import ChessEngine


def _ask_next_action(state, task, screenshot, action_history, conversation_messages, thought_history,
                     ui_elements=None, cursor_xy=None, screen_size=None, prior_screenshot_parts=None,
                     stuck_hint=None, move_to_hint=None):
    if not (state["api_key"] or "").strip():
        return ("comment", "Set Gemini API key in Settings.", conversation_messages or [], thought_history or [], None)
    thought_history = list(thought_history or [])
    if screen_size is None:
        screen_size = pyautogui.size()
    if cursor_xy is None:
        cursor_xy = tuple(pyautogui.position())
    img, model_w, model_h, (cur_mx, cur_my) = _prepare_image_for_model(
        screenshot, cursor_xy, screen_size
    )
    history_str = "\n".join(f"  {i+1}. {a}" for i, a in enumerate((action_history or [])[-12:]))
    learning = _build_learning_context()
    system = "You control the user's computer. One action per message."
    if move_to_hint:
        system = f"OVERRIDE: {move_to_hint}\n\n" + system
    if thought_history:
        for prev in thought_history[-3:]:
            system += "\n\n[Previous thinking]\n" + (prev[:800] + "..." if len(prev) > 800 else prev)
    if learning:
        system += "\n" + learning
    system += ACTION_SYSTEM_TAIL
    user_text = f"Task: {task}\n\nActions so far:\n{history_str or '  (none yet)'}\n\nWhat's the next action?"
    user_text += f"\n\nScreenshot dimensions: width={model_w} height={model_h}. Coords 0 to {model_w-1}, 0 to {model_h-1}."
    user_text += f"\n\n>>> CURRENT CURSOR POSITION: ({cur_mx}, {cur_my}) <<<"
    if stuck_hint:
        user_text += f"\n\n{stuck_hint}"
    if ui_elements:
        user_text += "\n\nClickable UI elements:\n"
        for item in (ui_elements or [])[:60]:
            name = (item.get("name") or "").strip()
            if name:
                user_text += f'  "{name}"\n'
    messages = list(conversation_messages or [])
    _EXECUTE_ACTIONS = ("CLICK", "CLICK_CURRENT", "CLICK_XY", "MOVE_TO", "DOUBLE_CLICK", "TYPE_IN", "TYPE_TEXT", "MENU", "KEYS", "KEY_PRESS", "TASK_COMPLETE", "OPEN_APP", "OPEN_URL")
    def _parse_bare_action(raw):
        s = (raw or "").strip().upper().replace("-", "_")
        for a in _EXECUTE_ACTIONS:
            if s == a or s.startswith(a + " ") or s.startswith(a + "{}") or s == a + "{}":
                return "KEYS" if a == "KEY_PRESS" else a
        for a in _EXECUTE_ACTIONS:
            if s == a or s.startswith(a + "_"):
                return "KEYS" if a == "KEY_PRESS" else a
        return None
    try:
        from gemini_vl import call_gemini
        prior_parts = [(cap, pimg) for pimg, cap in (prior_screenshot_parts or [])] if prior_screenshot_parts else None
        for api_attempt in range(3):
            try:
                text = call_gemini(
                    system, user_text, img,
                    conversation_messages=messages[-8:] if messages else None,
                    max_tokens=4096,
                    api_key=state["api_key"],
                    model=state["model"] or None,
                    prior_screenshot_parts=prior_parts,
                )
                break
            except Exception as e:
                if api_attempt < 2:
                    time.sleep(2)
                else:
                    if "no text" in str(e).lower() or "empty" in str(e).lower():
                        return ("screen_loading", None, messages, thought_history, img)
                    raise
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": text})
        if len(messages) > 8:
            messages = messages[-8:]
        text = re.sub(r"<thinking\s*>.*?</thinking\s*>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
        if not text or "{" not in text:
            return ("comment", "Model sent only thinking, no JSON.", messages, thought_history, img)
        if "```" in text:
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if m:
                text = m.group(1)
        data = _parse_json(text)
        if data is None:
            bare = _parse_bare_action(text)
            if bare:
                return ("execute", {"action": bare, "parameters": {}}, messages, thought_history, img)
            return ("comment", f"Invalid JSON: {text[:120]}", messages, thought_history, img)
        raw_action = data.get("action") or "comment"
        action = str(raw_action).lower().strip().replace(" ", "_").replace("-", "_")
        a_norm = str(raw_action).upper().strip().replace(" ", "_").replace("-", "_")
        if a_norm == "SCREEN_LOADING":
            return ("screen_loading", None, messages, thought_history, img)
        if a_norm == "ADD_FEEDBACK":
            p = data.get("parameters", data) or {}
            return ("add_feedback", p if isinstance(p, dict) else {"message": str(p)}, messages, thought_history, img)
        if a_norm == "START_CHESS":
            p = data.get("parameters", data) or {}
            playing_as = (p.get("playing_as") or "white").lower() if isinstance(p, dict) else "white"
            if playing_as not in ("white", "black"):
                playing_as = "white"
            return ("start_chess", {"reason": p.get("reason", "Board visible.") if isinstance(p, dict) else "Board visible.", "playing_as": playing_as}, messages, thought_history, img)
        if a_norm in _EXECUTE_ACTIONS:
            params = data.get("parameters", data)
            act = "KEYS" if a_norm == "KEY_PRESS" else a_norm
            return ("execute", {"action": act, "parameters": params if isinstance(params, dict) else {}}, messages, thought_history, img)
        msg = data.get("message", data.get("reason", text))
        bare = _parse_bare_action(str(msg))
        if bare:
            return ("execute", {"action": bare, "parameters": {}}, messages, thought_history, img)
        return ("comment", data.get("message", text), messages, thought_history, img)
    except Exception as e:
        return ("comment", f"API error: {e}", conversation_messages or [], thought_history or [], None)


WEBSITE_MAP = {
    "lichess": "lichess.org",
    "chess.com": "chess.com",
    "youtube": "youtube.com",
    "google": "google.com",
    "google docs": "docs.google.com",
    "google doc": "docs.google.com",
    "google slides": "slides.google.com",
    "google slide": "slides.google.com",
    "google sheets": "sheets.google.com",
    "google sheet": "sheets.google.com",
    "google drive": "drive.google.com",
    "github": "github.com",
    "gmail": "mail.google.com",
    "reddit": "reddit.com",
    "twitter": "twitter.com",
    "twitch": "twitch.tv",
    "spotify": "open.spotify.com",
    "netflix": "netflix.com",
    "chatgpt": "chat.openai.com",
    "claude": "claude.ai",
}

APP_NAMES = {"chrome", "edge", "firefox", "notepad", "calculator", "calc", "code", "vscode", "explorer", "terminal", "cmd", "powershell", "word", "excel", "powerpoint"}


def _try_quick_commands(task: str, task_lower: str, log) -> bool:
    """Handle common tasks directly without a Gemini call.
    Returns True if the task was handled, False to fall through to screen control."""
    import webbrowser

    # Split compound tasks: "open chrome and play lichess" → ["open chrome", "play lichess"]
    parts = [p.strip() for p in re.split(r'\band\b|,', task_lower) if p.strip()]

    actions_done = []
    browser_opened = False

    for part in parts:
        part_stripped = part.strip()

        # "open <app>" or "launch <app>"
        if part_stripped.startswith("open ") or part_stripped.startswith("launch "):
            target = part_stripped.split(" ", 1)[1].strip()

            # Check if it's a known website
            url = None
            for name, site in WEBSITE_MAP.items():
                if name in target:
                    url = site
                    break
            if any(ind in target for ind in (".com", ".org", ".net", ".io", ".gg", ".tv", ".ai", "http")):
                url = target if "://" in target else target

            if url:
                if not browser_opened:
                    log(f"Opening Chrome...", "action")
                    _open_program("chrome")
                    browser_opened = True
                    time.sleep(2.0)
                log(f"Navigating to {url}...", "action")
                ok, msg = _open_url(url)
                log(f"  {msg}" if ok else f"  Failed: {msg}", "info" if ok else "error")
                actions_done.append(f"opened {url}")
                time.sleep(1.0)
            elif target.lower() in APP_NAMES or target.lower() in ("chrome", "browser"):
                log(f"Opening {target}...", "action")
                ok, msg = _open_program(target)
                log(f"  {msg}" if ok else f"  Failed: {msg}", "info" if ok else "error")
                if "chrome" in target.lower() or "browser" in target.lower():
                    browser_opened = True
                actions_done.append(f"opened {target}")
                time.sleep(1.5)
            else:
                # Unknown target — try as program first, then as website
                log(f"Trying to open {target}...", "action")
                ok, msg = _open_program(target)
                if ok:
                    log(f"  {msg}", "info")
                    actions_done.append(f"opened {target}")
                else:
                    # Try as URL
                    if not browser_opened:
                        _open_program("chrome")
                        browser_opened = True
                        time.sleep(2.0)
                    ok2, msg2 = _open_url(target)
                    log(f"  {msg2}" if ok2 else f"  Failed: {msg2}", "info" if ok2 else "error")
                    actions_done.append(f"opened {target}")
                time.sleep(1.0)

        # "play lichess", "go to youtube", "play chess"
        elif part_stripped.startswith("play ") or part_stripped.startswith("go to ") or part_stripped.startswith("visit "):
            target = re.sub(r'^(play|go to|visit)\s+', '', part_stripped).strip()

            url = None
            for name, site in WEBSITE_MAP.items():
                if name in target:
                    url = site
                    break
            if url is None and any(ind in target for ind in (".com", ".org", ".net", ".io", ".gg", ".tv", ".ai")):
                url = target
            if url is None and "chess" in target:
                url = "lichess.org"

            if url:
                if not browser_opened:
                    log(f"Opening Chrome...", "action")
                    _open_program("chrome")
                    browser_opened = True
                    time.sleep(2.0)
                log(f"Navigating to {url}...", "action")
                ok, msg = _open_url(url)
                log(f"  {msg}" if ok else f"  Failed: {msg}", "info" if ok else "error")
                actions_done.append(f"opened {url}")
                time.sleep(1.0)
            else:
                return False  # Can't handle this part — fall through to router

        # "search for X", "google X"
        elif part_stripped.startswith("search ") or part_stripped.startswith("google "):
            query = re.sub(r'^(search\s+for|search|google)\s+', '', part_stripped).strip()
            if not browser_opened:
                log(f"Opening Chrome...", "action")
                _open_program("chrome")
                browser_opened = True
                time.sleep(2.0)
            url = f"google.com/search?q={query.replace(' ', '+')}"
            log(f"Searching: {query}...", "action")
            ok, msg = _open_url(url)
            log(f"  {msg}" if ok else f"  Failed: {msg}", "info" if ok else "error")
            actions_done.append(f"searched {query}")
            time.sleep(1.0)

        else:
            # Unrecognized part — can't handle without router
            return False

    if actions_done:
        log(f"Quick commands done: {', '.join(actions_done)}", "info")
        return True

    return False


def run_task(task: str) -> None:
    """Run the full agent flow for one task. Logs JSON lines to stdout. No tkinter."""
    cfg = _load_config()
    api_key = (cfg.get("gemini_api_key") or "").strip() or __import__("os").environ.get("GEMINI_API_KEY", "")
    model = (cfg.get("gemini_model") or "").strip() or (GEMINI_MODELS[0] if GEMINI_MODELS else "gemini-2.0-flash")
    conf = float(cfg.get("yolo_confidence", 0.15))
    depth = int(cfg.get("stockfish_depth", 18))
    interval = float(cfg.get("scan_interval", 3.0))
    click_delay = float(cfg.get("click_delay", 0.15))

    def log(msg, tag="info"):
        print(json.dumps({"type": "log", "msg": msg, "tag": (tag or "info").strip() or "info"}), flush=True)

    state = {
        "api_key": api_key,
        "model": model,
        "conf": conf,
        "depth": depth,
        "interval": interval,
        "click_delay": click_delay,
        "running": True,
        "turn": "white",
    }
    chess = ChessEngine(log_fn=log)
    state["chess"] = chess

    task_stripped = (task or "").strip()
    if task_stripped.startswith("1234") and task_stripped[4:].lstrip().lstrip(",").strip():
        feedback_text = task_stripped[4:].lstrip().lstrip(",").strip()
        log("Storing feedback...", "action")
        _save_prompt_entry({
            "type": "feedback",
            "user_message": feedback_text,
            "feedback": {"user_correction": feedback_text, "what_was_wrong": feedback_text, "correct_approach": feedback_text},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        log("Feedback stored.", "info")
        print(json.dumps({"type": "done", "message": "Done"}), flush=True)
        return

    task_lower = task_stripped.lower()

    # ── Quick commands that don't need a Gemini call ──
    handled = _try_quick_commands(task_stripped, task_lower, log)
    if handled:
        print(json.dumps({"type": "done", "message": "Done"}), flush=True)
        return

    if not api_key:
        log("No Gemini API key. Set it in Settings (click dot → ⚙) or set GEMINI_API_KEY env var.", "error")
        print(json.dumps({"type": "done", "message": "No API key"}), flush=True)
        return

    log("--- Starting screen control ---", "header")
    log("Taking screenshot...", "action")
    _signal_hide()
    time.sleep(0.3)
    screenshot, cursor_xy, screen_size = _take_screenshot_with_cursor()
    _signal_show()

    _run_screen_control_loop(state, task_stripped or task, screenshot, None, log)
    print(json.dumps({"type": "done", "message": "Done"}), flush=True)


def _run_screen_control_loop(state, task, initial_screenshot, first_action, log):
    sw, sh = initial_screenshot.size[0], initial_screenshot.size[1]
    action_history = []
    conversation_messages = []
    thought_history = []
    screenshot_history = []
    current_action = first_action
    _last_click_pos = None
    _same_click_count = 0
    _last_was_move_to = None
    screen_action_count = 0
    _consecutive_noaction = 0

    while state["running"]:
        if current_action:
            act = current_action.get("action", "")
            params = current_action.get("parameters", current_action)
            action_dict = {"action": act, "parameters": params}

            if act == "TASK_COMPLETE":
                msg = params.get("message", "Task complete")
                log(f"  Done: {msg}", "info")
                return

            log(f"  Executing: {act} {params}", "action")
            time.sleep(0.1)
            ok, result = _execute_action(action_dict, sw, sh)
            time.sleep(1.0)

            action_history.append(f"{act}: {result}")
            screen_action_count += 1
            if ok:
                log(f"  -> OK: {result}", "info")
            else:
                log(f"  -> Failed: {result}", "error")

            _click_pos_match = re.search(r"\((\d+),\s*(\d+)\)", result or "")
            if act == "CLICK_CURRENT" and _click_pos_match:
                _pos = (int(_click_pos_match.group(1)), int(_click_pos_match.group(2)))
                if _pos == _last_click_pos:
                    _same_click_count += 1
                else:
                    _same_click_count = 1
                    _last_click_pos = _pos
            else:
                _same_click_count = 0
                _last_click_pos = None
            if act == "MOVE_TO":
                _mt_params = params if isinstance(params, dict) else {}
                _last_was_move_to = (_mt_params.get("x"), _mt_params.get("y"))
            else:
                _last_was_move_to = None

        if not state["running"]:
            break

        log("  Taking screenshot...", "action")
        _signal_hide()
        time.sleep(0.3)
        screenshot, cursor_xy, screen_size = _take_screenshot_with_cursor()
        ui_elements = get_ui_elements()
        _signal_show()
        sw, sh = screenshot.size[0], screenshot.size[1]

        if not state["running"]:
            break

        _stuck_hint = None
        if _same_click_count >= 3 and _last_click_pos:
            _stuck_hint = f"WARNING: CLICK_CURRENT at {_last_click_pos} {_same_click_count} times with no change. Try a different approach (OPEN_URL, KEYS, or MOVE_TO elsewhere)."
        _move_to_hint = None
        if _last_was_move_to is not None:
            _move_to_hint = f"Your last action was MOVE_TO {_last_was_move_to}. Now do CLICK_CURRENT (if cursor is on target) or MOVE_TO again."

        prepared_img = None
        for _ in range(10):
            action, payload, conversation_messages, thought_history, prepared_img = _ask_next_action(
                state, task, screenshot, action_history, conversation_messages, thought_history,
                ui_elements, cursor_xy, screen_size,
                prior_screenshot_parts=screenshot_history if screenshot_history else None,
                stuck_hint=_stuck_hint, move_to_hint=_move_to_hint,
            )
            if not state["running"]:
                break
            if action == "screen_loading":
                time.sleep(2)
                _signal_hide()
                screenshot, cursor_xy, screen_size = _take_screenshot_with_cursor()
                ui_elements = get_ui_elements()
                _signal_show()
                continue
            if action == "add_feedback":
                p = payload if isinstance(payload, dict) else {}
                _save_prompt_entry({
                    "type": "feedback",
                    "user_message": p.get("message", "Self-feedback")[:200],
                    "feedback": p,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                })
                log("  Self-feedback stored.", "info")
                continue
            break

        if not state["running"]:
            break

        if action == "execute":
            current_action = payload
            _consecutive_noaction = 0
            if prepared_img is not None:
                act = payload.get("action", "")
                params = payload.get("parameters", {})
                caption = f"Previous: {act} {params}"
                screenshot_history = [(prepared_img, caption)]
            if payload.get("action") != "CLICK_CURRENT":
                _same_click_count = 0
                _last_click_pos = None
            if payload.get("action") == "CLICK_CURRENT":
                _last_was_move_to = None
        elif action == "start_chess":
            playing_as = (payload.get("playing_as", "white") if isinstance(payload, dict) else "white")
            state["turn"] = playing_as
            log("Chess board detected; starting chess agent.", "info")
            _run_chess_loop(state, log)
            return
        elif action == "comment":
            msg = str(payload)[:200] if payload else ""
            _consecutive_noaction += 1
            if _consecutive_noaction >= 5:
                log(f"  Agent gave up after 5 consecutive non-actions. Last: {msg}", "warning")
                return
            log(f"  [Gemini] {msg}", "thought")
            current_action = None  # No action — loop back and re-screenshot
        else:
            # Unknown action type — treat as comment and keep looping
            _consecutive_noaction += 1
            if _consecutive_noaction >= 5:
                return
            current_action = None


def _run_chess_loop(state, log):
    chess = state["chess"]
    if not chess.ready:
        log("Loading chess engine...", "warning")
        if not chess.load_models():
            log("Chess engine failed to load — cannot play chess.", "error")
            return
        log("Chess engine ready.", "info")
    chess.reset()
    turn = state["turn"]
    conf = state["conf"]
    depth = state["depth"]
    interval = state["interval"]
    click_delay = state["click_delay"]
    log(f"CHESS AGENT STARTED — Playing as: {turn}", "header")
    _last_move_attempted = None
    _same_move_attempts = 0

    while state["running"]:
        log(f"-- Scan #{chess.cycle_count + 1} --", "action")
        _signal_hide()
        time.sleep(0.2)
        ss = pyautogui.screenshot()
        _signal_show()
        r = chess.analyze(ss, conf=conf, depth=depth, turn=turn, force_move=False)

        if r["status"] == "move":
            best = r["best_move"]
            fsq, tsq = best[:2], best[2:4]
            promo = f"={best[4:].upper()}" if len(best) > 4 else ""
            if best == _last_move_attempted:
                _same_move_attempts += 1
            else:
                _same_move_attempts = 1
                _last_move_attempted = best

            if _same_move_attempts > 5:
                log("Move attempted 5 times without board change — resetting.", "error")
                chess._last_fen = None
                _same_move_attempts = 0
                _last_move_attempted = None
            else:
                fen_before_move = r["fen"].split(" ")[0]
                _signal_hide()
                chess.execute_move(best, r["board_box"], r["orientation"], click_delay=click_delay)
                log(f"  Move #{chess.move_count} played!", "info")
                time.sleep(1.2)
                ss2 = pyautogui.screenshot()
                _signal_show()
                chess.capture_post_move_fen(ss2, conf)
                if chess._last_fen == fen_before_move or chess._last_fen is None:
                    chess._last_fen = None
                else:
                    _same_move_attempts = 0
                    _last_move_attempted = None

        elif r["status"] == "game_over":
            log(f"  GAME OVER: {r.get('message', '')}", "header")
            return
        elif r["status"] == "no_board":
            log("  No board detected — retrying next scan", "dim")

        if r["status"] == "move":
            wait = 1.5
        elif r["status"] == "waiting":
            wait = interval
        else:
            wait = max(1.0, interval * 0.5)
        end = time.time() + wait
        while time.time() < end and state["running"]:
            time.sleep(0.05)


def _run_daemon():
    """Long-running mode: read one task per line from stdin, run it, then wait for the next. No process restart."""
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            task = line.strip()
            if not task or task.lower() in ("exit", "quit"):
                break
            try:
                run_task(task)
            except Exception as e:
                import traceback
                print(json.dumps({"type": "log", "msg": f"Error: {e}\n{traceback.format_exc()}", "tag": "error"}), flush=True)
                print(json.dumps({"type": "done", "message": str(e)}), flush=True)
        except (KeyboardInterrupt, EOFError):
            break


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--daemon":
        _run_daemon()
        sys.exit(0)
    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else ""
    if not task:
        print(json.dumps({"type": "log", "msg": "Usage: python agent_backend.py <task> OR python agent_backend.py --daemon", "tag": "error"}), flush=True)
        sys.exit(1)
    try:
        run_task(task)
    except Exception as e:
        import traceback
        print(json.dumps({"type": "log", "msg": f"Error: {e}\n{traceback.format_exc()}", "tag": "error"}), flush=True)
        print(json.dumps({"type": "done", "message": str(e)}), flush=True)
        sys.exit(1)
