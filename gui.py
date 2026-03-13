"""
Sentinel - Chrome Job Automator GUI.

Simple tkinter interface: upload resume, configure API key, click Start.
The agent opens Chrome, scrapes internship listings from GitHub, and auto-applies.
"""
import os
import sys
import json
import time
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

BG = "#1a1a2e"
BG_DARK = "#0f0f1a"
FG = "#ffffff"
ACCENT = "#e94560"
GREEN = "#4CAF50"
ORANGE = "#FF9800"


def _load_config():
    try:
        if os.path.isfile(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def _save_config(data: dict):
    try:
        existing = _load_config()
        existing.update(data)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
    except OSError:
        pass


GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]


class JobAutomatorGUI:
    def __init__(self, root):
        self.root = root
        root.title("Sentinel — Job Automator")
        root.geometry("860x680")
        root.configure(bg=BG)
        root.minsize(700, 550)

        cfg = _load_config()
        self.gemini_api_key_var = tk.StringVar(
            value=(cfg.get("gemini_api_key") or "").strip()
            or os.environ.get("GEMINI_API_KEY", "")
        )
        self.gemini_model_var = tk.StringVar(
            value=(cfg.get("gemini_model") or "").strip() or GEMINI_MODELS[0]
        )
        self.resume_path = cfg.get("resume_path") or ""

        self._running = False
        self._thread = None
        self._stats = {"found": 0, "applied": 0, "skipped": 0, "failed": 0}

        self._apply_styles()
        self._build_ui()

        if self.resume_path and os.path.isfile(self.resume_path):
            self.resume_label.config(text=os.path.basename(self.resume_path))
        self.log("Ready. Upload your resume and click Start.", "info")

    def _apply_styles(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("TFrame", background=BG)
        s.configure("TLabel", background=BG, foreground=FG)
        s.configure("TLabelframe", background=BG, foreground=FG)
        s.configure("TLabelframe.Label", background=BG, foreground=FG)

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding="16")
        outer.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)

        # Title
        title_frame = tk.Frame(outer, bg=BG)
        title_frame.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        tk.Label(
            title_frame, text="Sentinel", font=("Arial", 28, "bold"),
            bg=BG, fg=ACCENT,
        ).pack(side=tk.LEFT)
        tk.Label(
            title_frame, text="  Chrome Job Automator",
            font=("Arial", 14), bg=BG, fg="#9e9e9e",
        ).pack(side=tk.LEFT, pady=(8, 0))

        # Status
        self.status_label = tk.Label(
            outer, text="Idle", font=("Arial", 11), bg=BG, fg=GREEN,
        )
        self.status_label.grid(row=1, column=0, sticky="w", pady=(0, 8))

        # Resume upload row
        resume_frame = ttk.LabelFrame(outer, text="Resume (PDF)", padding="8")
        resume_frame.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        resume_frame.columnconfigure(0, weight=1)

        self.resume_label = tk.Label(
            resume_frame, text="No resume selected",
            font=("Consolas", 10), bg=BG_DARK, fg="#888",
            anchor="w", padx=8, pady=6, relief=tk.FLAT,
        )
        self.resume_label.grid(row=0, column=0, sticky="ew", padx=(0, 8))

        tk.Button(
            resume_frame, text="Browse", font=("Arial", 10, "bold"),
            bg="#555", fg="#fff", activebackground="#666",
            relief=tk.FLAT, padx=16, pady=4,
            command=self._browse_resume, cursor="hand2",
        ).grid(row=0, column=1)

        # Controls row
        ctrl_frame = tk.Frame(outer, bg=BG)
        ctrl_frame.grid(row=3, column=0, sticky="ew", pady=(0, 8))

        self.start_btn = tk.Button(
            ctrl_frame, text="Start Applying", font=("Arial", 14, "bold"),
            bg=GREEN, fg="#fff", activebackground="#388E3C",
            relief=tk.FLAT, padx=30, pady=10,
            command=self._toggle, cursor="hand2",
        )
        self.start_btn.pack(side=tk.LEFT, padx=(0, 12))

        tk.Button(
            ctrl_frame, text="Settings", font=("Arial", 11),
            bg="#555", fg="#fff", activebackground="#666",
            relief=tk.FLAT, padx=16, pady=10,
            command=self._open_settings, cursor="hand2",
        ).pack(side=tk.LEFT)

        # Progress stats
        stats_frame = ttk.LabelFrame(outer, text="Progress", padding="6")
        stats_frame.grid(row=4, column=0, sticky="ew", pady=(0, 8))
        self.stats_label = tk.Label(
            stats_frame,
            text="Found: 0  |  Applied: 0  |  Skipped: 0  |  Failed: 0",
            font=("Consolas", 10), bg=BG, fg="#d4d4d4", anchor="w",
        )
        self.stats_label.pack(fill=tk.X)

        # Log area
        log_frame = ttk.LabelFrame(outer, text="Log", padding="8")
        log_frame.grid(row=5, column=0, sticky="nsew")
        outer.rowconfigure(5, weight=1)

        btn_row = tk.Frame(log_frame, bg=BG)
        btn_row.pack(fill=tk.X, pady=(0, 4))
        tk.Button(
            btn_row, text="Clear Log", font=("Arial", 9),
            bg="#444", fg="#fff", relief=tk.FLAT, padx=10, pady=2,
            command=lambda: self.log_text.delete("1.0", tk.END),
            cursor="hand2",
        ).pack(side=tk.RIGHT)

        self.log_text = scrolledtext.ScrolledText(
            log_frame, height=14, font=("Consolas", 9),
            bg=BG_DARK, fg="#d4d4d4", insertbackground="#fff",
            wrap=tk.WORD, relief=tk.FLAT,
        )
        self.log_text.pack(expand=True, fill=tk.BOTH)

        for tag, color in [
            ("info", GREEN), ("error", "#f44336"), ("action", "#2196F3"),
            ("warning", ORANGE), ("header", ACCENT), ("dim", "#666"),
            ("success", "#00E676"),
        ]:
            self.log_text.tag_config(tag, foreground=color)

    def _browse_resume(self):
        path = filedialog.askopenfilename(
            title="Select Resume",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        )
        if path:
            self.resume_path = path
            self.resume_label.config(
                text=os.path.basename(path), fg="#d4d4d4",
            )
            _save_config({"resume_path": path})
            self.log(f"Resume selected: {os.path.basename(path)}", "info")

    def _open_settings(self):
        win = tk.Toplevel(self.root)
        win.title("Settings")
        win.geometry("500x300")
        win.configure(bg=BG)
        win.transient(self.root)
        win.grab_set()

        px, py = 12, 4

        tk.Label(
            win, text="Gemini API Key",
            font=("Arial", 11, "bold"), bg=BG, fg=ACCENT,
        ).pack(anchor="w", padx=px, pady=(16, 2))
        tk.Label(
            win, text="Get a key at https://aistudio.google.com/apikey",
            bg=BG, fg="#9e9e9e", font=("Arial", 9),
        ).pack(anchor="w", padx=px, pady=py)

        key_frame = tk.Frame(win, bg=BG)
        key_frame.pack(fill=tk.X, padx=px, pady=py)
        self._settings_key_entry = tk.Entry(
            key_frame, textvariable=self.gemini_api_key_var, show="*",
            font=("Consolas", 10), bg=BG_DARK, fg="#d4d4d4",
            insertbackground="#fff", relief=tk.FLAT,
        )
        self._settings_key_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._key_visible = False

        def toggle_key():
            self._key_visible = not self._key_visible
            self._settings_key_entry.config(show="" if self._key_visible else "*")

        tk.Button(
            key_frame, text="Show", font=("Arial", 8),
            bg="#444", fg="#fff", relief=tk.FLAT, padx=6,
            command=toggle_key,
        ).pack(side=tk.RIGHT, padx=(4, 0))

        tk.Label(
            win, text="Model", font=("Arial", 11, "bold"),
            bg=BG, fg=ACCENT,
        ).pack(anchor="w", padx=px, pady=(12, 2))
        ttk.Combobox(
            win, textvariable=self.gemini_model_var,
            values=GEMINI_MODELS, state="readonly", width=30,
        ).pack(anchor="w", padx=px, pady=py)

        def close():
            _save_config({
                "gemini_api_key": self.gemini_api_key_var.get().strip(),
                "gemini_model": self.gemini_model_var.get(),
            })
            win.destroy()

        tk.Button(
            win, text="Save & Close", font=("Arial", 11, "bold"),
            bg=ACCENT, fg="#fff", relief=tk.FLAT, padx=24, pady=8,
            command=close, cursor="hand2",
        ).pack(pady=(20, 10))

    def _toggle(self):
        if self._running:
            self._stop()
        else:
            self._start()

    def _start(self):
        if not self.resume_path or not os.path.isfile(self.resume_path):
            self.log("Please select a valid resume PDF first.", "error")
            return
        if not self.gemini_api_key_var.get().strip():
            self.log("Set your Gemini API key in Settings first.", "error")
            return

        self._running = True
        self._stats = {"found": 0, "applied": 0, "skipped": 0, "failed": 0}
        self._update_stats()
        self.start_btn.config(
            text="Stop", bg="#f44336", activebackground="#d32f2f",
        )
        self.status_label.config(text="Running...", fg=ORANGE)
        self.log("--- Starting Job Automator ---", "header")

        self._thread = threading.Thread(
            target=self._run_automator, daemon=True,
        )
        self._thread.start()

    def _stop(self):
        self._running = False
        self.start_btn.config(
            text="Start Applying", bg=GREEN, activebackground="#388E3C",
        )
        self.status_label.config(text="Stopped", fg=ORANGE)
        self.log("--- Stopped ---", "header")

    def _run_automator(self):
        try:
            from job_automator import JobAutomator

            automator = JobAutomator(
                resume_path=self.resume_path,
                api_key=self.gemini_api_key_var.get().strip(),
                model=self.gemini_model_var.get().strip() or GEMINI_MODELS[0],
                log_fn=self.log,
                running_fn=lambda: self._running,
                stats_fn=self._on_stat_update,
            )
            automator.run()
        except Exception as e:
            import traceback
            self.log(f"Fatal error: {e}", "error")
            self.log(traceback.format_exc(), "dim")
        finally:
            self._running = False
            self.root.after(0, lambda: self.start_btn.config(
                text="Start Applying", bg=GREEN, activebackground="#388E3C",
            ))
            self.root.after(0, lambda: self.status_label.config(
                text="Done", fg=GREEN,
            ))

    def _on_stat_update(self, key, value):
        self._stats[key] = value
        self.root.after(0, self._update_stats)

    def _update_stats(self):
        s = self._stats
        self.stats_label.config(
            text=f"Found: {s['found']}  |  Applied: {s['applied']}  |  "
                 f"Skipped: {s['skipped']}  |  Failed: {s['failed']}",
        )

    def log(self, msg, tag="info"):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"
        self.root.after(0, self._log_insert, line, tag)

    def _log_insert(self, line, tag):
        self.log_text.insert(tk.END, line, tag)
        self.log_text.see(tk.END)


def main():
    root = tk.Tk()
    JobAutomatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
