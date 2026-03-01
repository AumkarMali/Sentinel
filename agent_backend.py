"""
Headless agent runner (no tkinter). Used by Electron: takes a task, runs the full flow,
logs JSON lines to stdout. Electron hides its windows before spawning so screenshots are clean.
"""
import json
import re
import sys
import time

import pyautogui

from agent_core import (
    _build_learning_context,
    _execute_action,
    _load_config,
    _open_program,
    _open_url,
    _parse_json,
    _parse_router_response,
    _prepare_image_for_model,
    _save_prompt_entry,
    _take_screenshot_with_cursor,
    ACTION_SYSTEM_TAIL,
    get_ui_elements,
    GEMINI_MODELS,
    ROUTER_SYSTEM_TAIL,
)
from chess_agent import ChessEngine


def _run_router(state, task, screenshot, ui_elements=None, cursor_xy=None, screen_size=None):
    if screen_size is None:
        screen_size = pyautogui.size()
    if cursor_xy is None:
        cursor_xy = tuple(pyautogui.position())
    img, model_w, model_h, (cur_mx, cur_my) = _prepare_image_for_model(
        screenshot, cursor_xy, screen_size
    )
    learning_context = _build_learning_context()
    system = "You are a router for an AI agent. The user will send a task and a screenshot.\n"
    if learning_context:
        system += learning_context + "\n\n"
    system += ROUTER_SYSTEM_TAIL
    user_text = f"Task: {task}\n\nLook at the screenshot and respond with JSON."
    user_text += f"\n\nScreenshot dimensions: width={model_w} height={model_h}. Coords from 0 to {model_w-1}, 0 to {model_h-1}."
    user_text += f"\n\n>>> CURRENT CURSOR POSITION: ({cur_mx}, {cur_my}) <<<"
    if ui_elements:
        user_text += "\n\nClickable UI elements:\n"
        for item in (ui_elements or [])[:60]:
            name = (item.get("name") or "").strip()
            if name:
                user_text += f'  "{name}"\n'
    from gemini_vl import call_gemini
    raw_response = call_gemini(
        system, user_text, img, conversation_messages=None, max_tokens=4096,
        api_key=state["api_key"], model=state["model"] or None,
    )
    text = raw_response
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
    _EXECUTE_ACTIONS = ("CLICK", "CLICK_CURRENT", "MOVE_TO", "DOUBLE_CLICK", "TYPE_IN", "MENU", "KEYS", "KEY_PRESS", "TYPE_TEXT", "TASK_COMPLETE", "OPEN_APP", "OPEN_URL")
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
    log("Loading chess engine...", "warning")
    if not chess.load_models():
        log("Chess engine failed to load.", "error")
        print(json.dumps({"type": "done", "message": "Chess load error"}), flush=True)
        return
    log("Chess engine ready.", "info")

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
    if (task_lower.startswith("open ") or task_lower.startswith("launch ")) and " and " not in task_lower and "," not in task_stripped:
        app_name = task_stripped[5:].strip() if task_lower.startswith("open ") else task_stripped[7:].strip()
        if app_name:
            log("Opening app directly (Win+R)...", "action")
            ok, msg = _open_program(app_name)
            log(f"  {msg}" if ok else f"  Failed: {msg}", "info" if ok else "error")
            print(json.dumps({"type": "done", "message": msg if ok else "Failed"}), flush=True)
            return

    log("--- PHASE 1: Router ---", "header")
    log("Taking screenshot...", "action")
    time.sleep(0.3)
    screenshot, cursor_xy, screen_size = _take_screenshot_with_cursor()
    ui_elements = get_ui_elements()

    log("Asking Gemini (router)...", "action")
    action, payload, raw_response = "comment", "Router did not run.", ""
    MAX_ROUTER_ATTEMPTS = 5
    MAX_LOADING_RETRIES = 5
    loading_retries = 0
    router_attempt = 0

    while state["running"]:
        try:
            action, payload, raw_response = _run_router(state, task_stripped or task, screenshot, ui_elements, cursor_xy, screen_size)
            if action == "screen_loading":
                if loading_retries >= MAX_LOADING_RETRIES:
                    log("Screen still loading after max retries; stopping.", "dim")
                    break
                loading_retries += 1
                log(f"Screen still loading (retry {loading_retries}/{MAX_LOADING_RETRIES}); waiting 2s...", "action")
                time.sleep(2)
                screenshot, cursor_xy, screen_size = _take_screenshot_with_cursor()
                ui_elements = get_ui_elements()
                router_attempt = 0
                continue
            break
        except Exception as e:
            router_attempt += 1
            err_str = str(e)
            if router_attempt < MAX_ROUTER_ATTEMPTS:
                wait = [3, 5, 8, 15][min(router_attempt - 1, 3)]
                log(f"Router attempt {router_attempt}/{MAX_ROUTER_ATTEMPTS} failed; retrying in {wait}s...", "dim")
                time.sleep(wait)
                continue
            log(f"Router error: {err_str}", "error")
            action, payload, raw_response = "comment", f"Error: {err_str}", ""
            break

    # On router done
    if action == "store_feedback":
        user_message = (task_stripped or "").strip()
        if user_message.lower().startswith("1234"):
            user_message = user_message[4:].lstrip().lstrip(",").strip()
        if not user_message:
            user_message = "Feedback from screenshot"
        _save_prompt_entry({
            "type": "feedback",
            "user_message": user_message,
            "feedback": payload if isinstance(payload, dict) else {"raw": str(payload)},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        log("Feedback stored.", "info")
        print(json.dumps({"type": "done", "message": "Done"}), flush=True)
        return

    if action == "start_chess":
        playing_as = payload.get("playing_as", "white") if isinstance(payload, dict) else "white"
        state["turn"] = playing_as
        log(f"Chess board detected; playing as {playing_as}. Starting chess agent.", "info")
        _run_chess_loop(state, log)
        print(json.dumps({"type": "done", "message": "Chess finished"}), flush=True)
        return

    if action == "execute":
        act = payload.get("action", "") if isinstance(payload, dict) else ""
        log(f"Router → execute first action: {act}", "info")
        log("--- PHASE 2: Screen control loop ---", "header")
        _run_screen_control_loop(state, task_stripped or task, screenshot, payload, log)
        print(json.dumps({"type": "done", "message": "Done"}), flush=True)
        return

    if action == "screen_loading":
        log("Page still loading; try again in a moment.", "warning")
        print(json.dumps({"type": "done", "message": "Screen loading"}), flush=True)
        return

    msg = payload.get("message", payload) if isinstance(payload, dict) else payload
    log(f"[Gemini] {msg}", "thought")
    print(json.dumps({"type": "done", "message": str(msg)[:200]}), flush=True)


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
        time.sleep(0.3)
        screenshot, cursor_xy, screen_size = _take_screenshot_with_cursor()
        ui_elements = get_ui_elements()
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
                screenshot, cursor_xy, screen_size = _take_screenshot_with_cursor()
                ui_elements = get_ui_elements()
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
        else:
            log(f"  Gemini replied (stopping): {str(payload)[:200]}", "warning")
            return


def _run_chess_loop(state, log):
    chess = state["chess"]
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
        time.sleep(0.2)
        ss = pyautogui.screenshot()
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
                chess.execute_move(best, r["board_box"], r["orientation"], click_delay=click_delay)
                log(f"  Move #{chess.move_count} played!", "info")
                time.sleep(1.2)
                ss2 = pyautogui.screenshot()
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
