"""
pywinauto-based screen control.

Exposes UI elements to the agent and executes element-based instructions
(CLICK, DOUBLE_CLICK, TYPE_IN, MENU, KEYS, TYPE_TEXT) instead of raw mouse/keyboard.
Windows only.
"""
import re
import sys

if sys.platform != "win32":
    _PYWINAUTO_AVAILABLE = False
    Application = Desktop = None
else:
    try:
        from pywinauto import Application, Desktop
        from pywinauto.findwindows import ElementNotFoundError
        import ctypes
        _PYWINAUTO_AVAILABLE = True
    except ImportError:
        _PYWINAUTO_AVAILABLE = False
        Application = Desktop = None


# Control types that are typically clickable/actionable
CLICKABLE_TYPES = {"Button", "ListItem", "MenuItem", "Hyperlink", "DataItem", "TreeItem", "TabItem"}


def get_ui_elements(max_items: int = 80) -> list:
    """
    Get list of actionable UI elements from the foreground window and taskbar.
    Returns [{"name": str, "control_type": str}, ...] for the agent to choose from.
    """
    
    if not _PYWINAUTO_AVAILABLE:
        return []
    out = []
    seen = set()
    try:
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        if hwnd:
            app = Application(backend="uia").connect(handle=hwnd)
            win = app.window(handle=hwnd)
            for ctrl in win.descendants():
                try:
                    name = (ctrl.window_text() or "").strip()
                    if not name or len(name) > 60:
                        continue
                    ctype = getattr(ctrl.element_info, "control_type", None) or "Unknown"
                    key = (name, ctype)
                    if key in seen:
                        continue
                    seen.add(key)
                    if ctype in CLICKABLE_TYPES or "Edit" in ctype or "Combo" in ctype:
                        out.append({"name": name, "control_type": str(ctype)})
                        if len(out) >= max_items:
                            return out
                except Exception:
                    continue
        # Taskbar via Desktop
        try:
            desktop = Desktop(backend="uia")
            taskbar = desktop.child_window(title="Taskbar", control_type="ToolBar")
            if taskbar.exists(timeout=1):
                for ctrl in taskbar.descendants():
                    try:
                        name = (ctrl.window_text() or "").strip()
                        if not name or len(name) > 60:
                            continue
                        ctype = getattr(ctrl.element_info, "control_type", None) or "Unknown"
                        key = (name, ctype)
                        if key in seen:
                            continue
                        seen.add(key)
                        if ctype in CLICKABLE_TYPES or "Button" in ctype:
                            out.append({"name": name, "control_type": str(ctype)})
                            if len(out) >= max_items:
                                return out
                    except Exception:
                        continue
        except Exception:
            pass
        # Taskbar via explorer.exe (Windows 10/11 – often exposes taskbar icons better)
        try:
            app = Application(backend="uia").connect(path="explorer.exe")
            for w in [app.window(class_name="Shell_TrayWnd"), app.window(title="Taskbar")]:
                try:
                    if not w.exists(timeout=1):
                        continue
                    for ctrl in w.descendants():
                        try:
                            name = (ctrl.window_text() or "").strip()
                            if not name or len(name) > 60:
                                continue
                            ctype = getattr(ctrl.element_info, "control_type", None) or "Unknown"
                            key = (name, ctype)
                            if key in seen:
                                continue
                            seen.add(key)
                            out.append({"name": name, "control_type": str(ctype)})
                            if len(out) >= max_items:
                                return out
                        except Exception:
                            continue
                except Exception:
                    continue
        except Exception:
            pass
    except Exception:
        pass
    return out


def execute_action(action_dict: dict) -> tuple:
    """
    Execute a pywinauto-compatible action. Returns (success: bool, message: str).

    Supported actions:
    - CLICK: {"action": "CLICK", "parameters": {"element": "Open"}}
    - DOUBLE_CLICK: {"action": "DOUBLE_CLICK", "parameters": {"element": "Google Chrome"}}
    - TYPE_IN: {"action": "TYPE_IN", "parameters": {"element": "Search", "text": "chess.com"}}
    - MENU: {"action": "MENU", "parameters": {"path": "File->Open"}}
    - TASK_COMPLETE: {"action": "TASK_COMPLETE", "parameters": {"message": "Done"}}

    KEYS and TYPE_TEXT are handled by the caller (pynput) - not pywinauto.
    """
    if not _PYWINAUTO_AVAILABLE and sys.platform == "win32":
        return False, "pywinauto not available"
    action = (action_dict.get("action") or "").upper().replace(" ", "_")
    params = action_dict.get("parameters") or action_dict
    if isinstance(params, dict) and "action" in params:
        params = params.get("parameters", params) or params

    if action == "TASK_COMPLETE":
        return True, params.get("message", "Task complete")

    # pywinauto actions need Windows
    if sys.platform != "win32":
        return False, "pywinauto actions require Windows"

    element = params.get("element") or params.get("name")
    if not element and action in ("CLICK", "DOUBLE_CLICK", "TYPE_IN"):
        return False, f"Missing 'element' for action {action}"

    try:
        import ctypes
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        if not hwnd:
            return False, "No foreground window"
        app = Application(backend="uia").connect(handle=hwnd)
        win = app.window(handle=hwnd)

        def _find_and_click(container, double=False):
            # Try full match first, then shorter variants (Start menu items often have long suffixes)
            candidates = [element.strip()]
            if len(element) > 30:
                candidates.append(element[:30].strip())
            if "," in element:
                candidates.append(element.split(",")[0].strip())
            if " - " in element:
                candidates.append(element.split(" - ")[0].strip())
            if " (" in element:
                candidates.append(element.split(" (")[0].strip())
            for search_text in candidates:
                if not search_text:
                    continue
                try:
                    ctrl = container.child_window(title_re=f".*{_re_escape(search_text)}.*")
                    if not ctrl.exists(timeout=1):
                        ctrl = container.child_window(title=search_text)
                    if ctrl.exists(timeout=1):
                        if double:
                            ctrl.double_click_input()
                        else:
                            ctrl.click_input()
                        return True
                except Exception:
                    continue
            return False

        def _find_by_scan(container, double=False, max_scan=300, strict_type=True):
            """Walk descendants; click first control whose title contains element (taskbar/desktop)."""
            search_lower = element.strip().lower()
            if not search_lower:
                return False
            parts = [search_lower]
            if "," in element:
                parts.append(element.split(",")[0].strip().lower())
            if " - " in element:
                parts.append(element.split(" - ")[0].strip().lower())
            # "Chrome" alone can match "Google Chrome"
            if "google chrome" in search_lower or "chrome" in search_lower:
                parts.append("chrome")
            n = 0
            try:
                for ctrl in container.descendants():
                    if n >= max_scan:
                        break
                    n += 1
                    try:
                        title = (ctrl.window_text() or "").strip()
                        if not title or not any(p in title.lower() for p in parts):
                            continue
                        if strict_type:
                            ctype = getattr(ctrl.element_info, "control_type", None) or ""
                            if ctype not in CLICKABLE_TYPES and "Button" not in ctype:
                                continue
                        if double:
                            ctrl.double_click_input()
                        else:
                            ctrl.click_input()
                        return True
                    except Exception:
                        continue
            except Exception:
                pass
            return False

        def _try_taskbar_via_explorer():
            """Connect to explorer.exe and find taskbar / Running applications; click matching item."""
            try:
                app = Application(backend="uia").connect(path="explorer.exe")
                # Try Shell_TrayWnd (taskbar container)
                for w in [app.window(class_name="Shell_TrayWnd"), app.window(title="Taskbar")]:
                    try:
                        if w.exists(timeout=1) and _find_by_scan(w, double=False, strict_type=False):
                            return True
                    except Exception:
                        continue
                # Try "Running applications" toolbar (Windows 10/11)
                try:
                    tb = app.child_window(title="Taskbar", control_type="ToolBar")
                    if tb.exists(timeout=1):
                        run_apps = tb.child_window(title="Running applications", control_type="ToolBar")
                        if run_apps.exists(timeout=1) and _find_by_scan(run_apps, double=False, strict_type=False):
                            return True
                        if _find_by_scan(tb, double=False, strict_type=False):
                            return True
                except Exception:
                    pass
            except Exception:
                pass
            return False

        if action == "CLICK":
            if _find_and_click(win, double=False):
                return True, f"Clicked '{element}'"
            try:
                desktop = Desktop(backend="uia")
                if _find_and_click(desktop, double=False):
                    return True, f"Clicked '{element}'"
                if _find_by_scan(desktop, double=False):
                    return True, f"Clicked '{element}'"
                taskbar = desktop.child_window(title="Taskbar", control_type="ToolBar")
                if taskbar.exists(timeout=1) and _find_by_scan(taskbar, double=False):
                    return True, f"Clicked '{element}'"
                if _try_taskbar_via_explorer():
                    return True, f"Clicked '{element}'"
            except Exception:
                pass
            return False, f"Element '{element}' not found"

        if action == "DOUBLE_CLICK":
            if _find_and_click(win, double=True):
                return True, f"Double-clicked '{element}'"
            try:
                desktop = Desktop(backend="uia")
                if _find_and_click(desktop, double=True):
                    return True, f"Double-clicked '{element}'"
                if _find_by_scan(desktop, double=True):
                    return True, f"Double-clicked '{element}'"
                taskbar = desktop.child_window(title="Taskbar", control_type="ToolBar")
                if taskbar.exists(timeout=1) and _find_by_scan(taskbar, double=True):
                    return True, f"Double-clicked '{element}'"
                # Explorer taskbar (double-click to launch from taskbar if pinned)
                try:
                    app = Application(backend="uia").connect(path="explorer.exe")
                    for w in [app.window(class_name="Shell_TrayWnd"), app.window(title="Taskbar")]:
                        try:
                            if w.exists(timeout=1) and _find_by_scan(w, double=True, strict_type=False):
                                return True, f"Double-clicked '{element}'"
                        except Exception:
                            continue
                except Exception:
                    pass
            except Exception:
                pass
            return False, f"Element '{element}' not found"

        if action == "TYPE_IN":
            text = params.get("text", "")
            if not text:
                return False, "TYPE_IN requires 'text' parameter"
            el_strip = (element or "").strip()
            # Try to find an Edit or ComboBox: by name, then common browser/OS labels
            candidates = []
            if el_strip:
                candidates.append((el_strip, "Edit"))
                candidates.append((el_strip, "ComboBox"))
            candidates.extend([
                ("Address and search bar", "Edit"), ("Address", "Edit"), ("Search", "Edit"),
                ("Search or type a URL", "Edit"), ("Search the web", "Edit"),
                ("Type here to search", "Edit"), ("Search box", "Edit"),
            ])
            ctrl = None
            for name, ctype in candidates:
                try:
                    c = win.child_window(title_re=f".*{_re_escape(name)}.*", control_type=ctype)
                    if c.exists(timeout=1):
                        ctrl = c
                        break
                    c = win.child_window(title=name, control_type=ctype)
                    if c.exists(timeout=1):
                        ctrl = c
                        break
                except Exception:
                    continue
            if not ctrl or not ctrl.exists(timeout=1):
                # Last resort: first Edit in window (e.g. single search field)
                try:
                    for c in win.descendants():
                        ct = getattr(c.element_info, "control_type", None) or ""
                        if ct in ("Edit", "ComboBox"):
                            ctrl = c
                            break
                except Exception:
                    pass
            if not ctrl or not ctrl.exists(timeout=1):
                return False, f"Edit/field for '{element or 'text'}' not found"
            ctrl.set_focus()
            try:
                ctrl.click_input()
            except Exception:
                pass
            import time
            time.sleep(0.15)
            try:
                ctrl.set_edit_text(text)
                return True, f"Typed into field: {text[:30]}..."
            except Exception:
                pass
            # set_edit_text failed (e.g. contenteditable); type via keyboard into focused control
            try:
                from pynput.keyboard import Controller
                Controller().type(text)
                return True, f"Typed (keyboard) into field: {text[:30]}..."
            except Exception:
                try:
                    import pyautogui
                    pyautogui.write(text, interval=0.02)
                    return True, f"Typed (keyboard) into field: {text[:30]}..."
                except Exception as e:
                    return False, f"TYPE_IN failed: {e}"

        if action == "MENU":
            path = params.get("path") or params.get("menu") or ""
            if not path:
                return False, "MENU requires 'path' (e.g. 'File->Open')"
            win.menu_select(path)
            return True, f"Menu: {path}"

        return False, f"Unknown action: {action}"
    except Exception as e:
        return False, str(e)


def _re_escape(s: str) -> str:
    """Escape string for use in regex."""
    return re.escape(str(s))
