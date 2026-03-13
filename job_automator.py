"""
Core job application automation engine.

Opens Chrome via Selenium, scrapes internship listings from the SimplifyJobs
GitHub repo, and attempts to auto-apply using Gemini vision for form analysis.

Hybrid approach:
  1. Selenium DOM access (primary — fast, doesn't move the mouse)
  2. pyautogui + Gemini vision (fallback — screenshots the screen, asks Gemini
     for coordinates, moves mouse and clicks/types physically)
"""
import io
import os
import re
import sys
import json
import time
from urllib.parse import urlparse

from PIL import Image, ImageDraw, ImageFont
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException, NoSuchElementException, ElementNotInteractableException,
    StaleElementReferenceException,
)

import pyautogui

# DPI awareness on Windows so pyautogui coordinates match physical pixels
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1

try:
    from webdriver_manager.chrome import ChromeDriverManager
    _WDM_AVAILABLE = True
except ImportError:
    _WDM_AVAILABLE = False

LISTINGS_URL = "https://github.com/SimplifyJobs/Summer2026-Internships"

GRID_SPACING = 100


# ── Vision helpers (screen-level fallback) ──────────────────────────────

def _draw_grid(img, spacing=GRID_SPACING):
    """Overlay a labeled coordinate grid so Gemini can read exact pixel positions."""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    minor = (60, 60, 60)
    major = (110, 110, 110)
    label_col = (160, 210, 255)
    try:
        font = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
    for x in range(0, w, spacing):
        col = major if x % (spacing * 2) == 0 else minor
        draw.line([(x, 0), (x, h)], fill=col, width=1)
        if x > 0:
            draw.text((x + 2, 2), str(x), fill=label_col, font=font)
    for y in range(0, h, spacing):
        col = major if y % (spacing * 2) == 0 else minor
        draw.line([(0, y), (w, y)], fill=col, width=1)
        if y > 0:
            draw.text((2, y + 2), str(y), fill=label_col, font=font)


def _take_screen_screenshot():
    """Capture full screen via pyautogui, return (PIL image, screen_w, screen_h)."""
    ss = pyautogui.screenshot()
    screen_size = pyautogui.size()
    return ss, screen_size[0], screen_size[1]


def _prepare_for_model(screenshot, screen_w, screen_h):
    """Resize screenshot to <=1024px long edge, draw grid. Returns (img, scale)."""
    img = screenshot.convert("RGB")
    ss_w, ss_h = img.size
    scale = 1024 / max(ss_w, ss_h) if max(ss_w, ss_h) > 1024 else 1.0
    if scale < 1.0:
        img = img.resize(
            (int(ss_w * scale), int(ss_h * scale)), Image.Resampling.LANCZOS,
        )
    _draw_grid(img)
    return img, scale


def _model_to_screen(mx, my, scale, screen_w, screen_h, ss_w, ss_h):
    """Convert model-image coordinates back to pyautogui screen coordinates."""
    img_x = mx / scale
    img_y = my / scale
    sx = int(round(img_x * screen_w / ss_w))
    sy = int(round(img_y * screen_h / ss_h))
    sx = max(0, min(screen_w - 1, sx))
    sy = max(0, min(screen_h - 1, sy))
    return sx, sy


class JobAutomator:
    """Scrapes the SimplifyJobs internship list and auto-applies via Chrome."""

    def __init__(self, resume_path, api_key, model, log_fn, running_fn, stats_fn):
        self.resume_path = os.path.abspath(resume_path)
        self.api_key = api_key
        self.model = model
        self.log = log_fn
        self.is_running = running_fn
        self.stats_fn = stats_fn
        self.driver = None
        self.resume_data = None

    # ── Main entry ──────────────────────────────────────────────────────

    def run(self):
        try:
            self._parse_resume()
            if not self.is_running():
                return
            self._setup_chrome()
            if not self.is_running():
                return
            self._navigate_to_listings()
            if not self.is_running():
                return
            jobs = self._scrape_jobs()
            self.stats_fn("found", len(jobs))
            self.log(f"Found {len(jobs)} open positions with apply links.", "success")
            if not jobs:
                self.log("No jobs found. The page format may have changed.", "warning")
                return
            self._apply_to_jobs(jobs)
        except Exception as e:
            import traceback
            self.log(f"Error: {e}", "error")
            self.log(traceback.format_exc(), "dim")
        finally:
            if self.driver:
                try:
                    self.driver.quit()
                except Exception:
                    pass
                self.log("Chrome closed.", "info")

    # ── Setup ───────────────────────────────────────────────────────────

    def _parse_resume(self):
        self.log("Parsing resume...", "action")
        from resume_parser import parse_resume
        self.resume_data = parse_resume(self.resume_path, self.api_key, self.model)
        name = self.resume_data.get("name", "Unknown")
        email = self.resume_data.get("email", "")
        self.log(f"Resume parsed: {name} ({email})", "info")

    def _setup_chrome(self):
        self.log("Starting Chrome...", "action")
        options = webdriver.ChromeOptions()
        options.add_argument("--start-maximized")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)

        if _WDM_AVAILABLE:
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=options)
        else:
            self.driver = webdriver.Chrome(options=options)

        self.driver.implicitly_wait(5)
        self.log("Chrome started.", "success")

    def _navigate_to_listings(self):
        self.log(f"Navigating to {LISTINGS_URL}", "action")
        self.driver.get(LISTINGS_URL)
        time.sleep(3)
        self.log("Loaded internship listings page.", "info")

    # ── Job scraping ────────────────────────────────────────────────────

    def _scrape_jobs(self):
        """Parse the GitHub README table for job listings with apply links."""
        self.log("Scraping job listings...", "action")
        jobs = []

        try:
            readme = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "article.markdown-body")
                )
            )
        except TimeoutException:
            self.log("Could not find README content on the page.", "error")
            return jobs

        tables = readme.find_elements(By.TAG_NAME, "table")
        if not tables:
            self.log("No tables found in README.", "warning")
            return jobs

        table = tables[0]
        rows = table.find_elements(By.TAG_NAME, "tr")

        for row in rows[1:]:
            try:
                cells = row.find_elements(By.TAG_NAME, "td")
                if len(cells) < 4:
                    continue

                company = cells[0].text.strip()
                role = cells[1].text.strip()
                location = cells[2].text.strip()

                row_text = row.text
                if "\U0001f512" in row_text or "\U0001f512" in row_text:
                    continue

                apply_links = cells[3].find_elements(By.TAG_NAME, "a")
                if not apply_links:
                    continue

                apply_url = None
                for link in apply_links:
                    href = link.get_attribute("href") or ""
                    if href and "simplify.jobs" not in href.lower():
                        apply_url = href
                        break
                if not apply_url and apply_links:
                    apply_url = apply_links[0].get_attribute("href")

                if apply_url:
                    jobs.append({
                        "company": company,
                        "role": role,
                        "location": location,
                        "url": apply_url,
                    })
            except StaleElementReferenceException:
                continue
            except Exception:
                continue

        return jobs

    # ── Apply loop ──────────────────────────────────────────────────────

    def _apply_to_jobs(self, jobs):
        applied = 0
        skipped = 0
        failed = 0

        for i, job in enumerate(jobs):
            if not self.is_running():
                self.log("Stopped by user.", "warning")
                break

            company = job["company"]
            role = job["role"]
            url = job["url"]
            self.log(f"\n[{i+1}/{len(jobs)}] {company} — {role}", "header")
            self.log(f"  URL: {url}", "dim")

            try:
                result = self._apply_to_single_job(job)
                if result == "applied":
                    applied += 1
                    self.stats_fn("applied", applied)
                    self.log("  Applied successfully!", "success")
                elif result == "skipped":
                    skipped += 1
                    self.stats_fn("skipped", skipped)
                    self.log(
                        "  Skipped (login required or unsupported form).", "warning",
                    )
                else:
                    failed += 1
                    self.stats_fn("failed", failed)
                    self.log("  Could not complete application.", "error")
            except Exception as e:
                failed += 1
                self.stats_fn("failed", failed)
                self.log(f"  Error: {e}", "error")

            time.sleep(2)

        self.log(
            f"\nDone! Applied: {applied}, Skipped: {skipped}, Failed: {failed}",
            "header",
        )

    def _apply_to_single_job(self, job):
        """Open apply link and attempt to fill the application form."""
        url = job["url"]

        self.driver.execute_script("window.open('');")
        self.driver.switch_to.window(self.driver.window_handles[-1])

        try:
            self.driver.get(url)
            time.sleep(4)

            page_analysis = self._analyze_page()
            if not page_analysis:
                return "failed"

            page_type = page_analysis.get("page_type", "unknown")
            self.log(f"  Page type: {page_type}", "dim")

            if page_type == "login_required":
                return "skipped"
            if page_type == "job_description_only":
                apply_btn = page_analysis.get("apply_button")
                if apply_btn:
                    clicked = self._click_element_by_analysis(apply_btn)
                    if clicked:
                        time.sleep(3)
                        page_analysis = self._analyze_page()
                        if not page_analysis:
                            return "failed"
                        page_type = page_analysis.get("page_type", "unknown")
                    else:
                        return "skipped"
                else:
                    return "skipped"

            if page_type in ("application_form", "multi_step_form"):
                return self._fill_and_submit_form(page_analysis)

            return "skipped"
        finally:
            self.driver.close()
            self.driver.switch_to.window(self.driver.window_handles[0])

    # ── Screenshots ─────────────────────────────────────────────────────

    def _take_page_screenshot(self):
        """Capture current page via Selenium (no mouse needed)."""
        png = self.driver.get_screenshot_as_png()
        return Image.open(io.BytesIO(png)).convert("RGB")

    def _take_screen_for_vision(self):
        """Full-screen pyautogui screenshot with grid for vision fallback.

        Returns (grid_image, scale, screen_w, screen_h, raw_ss_w, raw_ss_h).
        """
        self._bring_chrome_to_front()
        time.sleep(0.3)
        ss, scr_w, scr_h = _take_screen_screenshot()
        raw_w, raw_h = ss.size
        img, scale = _prepare_for_model(ss, scr_w, scr_h)
        return img, scale, scr_w, scr_h, raw_w, raw_h

    def _bring_chrome_to_front(self):
        """Try to bring the Chrome window to the foreground."""
        try:
            self.driver.switch_to.window(self.driver.current_window_handle)
            self.driver.execute_script(
                "window.focus(); document.title = document.title;"
            )
        except Exception:
            pass
        if sys.platform == "win32":
            try:
                import ctypes
                hwnd = ctypes.windll.user32.FindWindowW(
                    "Chrome_WidgetWin_1", None,
                )
                if hwnd:
                    ctypes.windll.user32.SetForegroundWindow(hwnd)
            except Exception:
                pass

    # ── Page analysis (Gemini) ──────────────────────────────────────────

    def _analyze_page(self):
        """Use Gemini vision to analyze the current page."""
        self.log("  Analyzing page with Gemini...", "action")
        screenshot = self._take_page_screenshot()

        system = (
            "You are a web page analyzer for a job application bot. "
            "Analyze the screenshot of a job application page and return a JSON object."
        )

        user_text = """Analyze this web page screenshot. Determine what type of page this is and identify actionable elements.

Return ONLY a JSON object with this structure:
{
    "page_type": "application_form" | "job_description_only" | "login_required" | "multi_step_form" | "redirect" | "error" | "unknown",
    "fields": [
        {"label": "field label", "type": "text|email|tel|select|file|textarea|checkbox|radio", "required": true/false, "value_hint": "what kind of data goes here"}
    ],
    "apply_button": {"text": "button text", "description": "where it is on the page"} or null,
    "submit_button": {"text": "button text", "description": "where it is on the page"} or null,
    "notes": "any relevant observations about the page"
}

Important:
- "login_required" means a login/signup wall blocks the form
- "job_description_only" means it shows job details with an Apply button to click
- "application_form" means there are fillable form fields visible
- Include ALL visible form fields in the fields array"""

        try:
            from gemini_vl import call_gemini
            response = call_gemini(
                system, user_text, screenshot,
                max_tokens=2048,
                api_key=self.api_key,
                model=self.model,
            )
            return self._parse_gemini_json(response)
        except Exception as e:
            self.log(f"  Gemini analysis error: {e}", "error")
            return None

    def _parse_gemini_json(self, response):
        """Extract JSON from a Gemini response (strip thinking, code fences)."""
        response = re.sub(
            r"<thinking\s*>.*?</thinking\s*>", "", response,
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()
        if "```" in response:
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
            if m:
                response = m.group(1)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            self.log("  Could not parse Gemini response as JSON.", "warning")
            return None

    # ── Vision fallback (pyautogui + Gemini) ────────────────────────────

    def _vision_click(self, description):
        """Fallback: screenshot the screen, ask Gemini for coordinates, click with pyautogui.

        Returns True if click was performed.
        """
        self.log(f"  [Fallback] Vision-clicking: {description}", "action")
        img, scale, scr_w, scr_h, raw_w, raw_h = self._take_screen_for_vision()
        model_w, model_h = img.size

        system = (
            "You are a screen coordinate finder. The image has a grid overlay "
            "with labels every 100px along the edges. Use the grid to determine "
            "exact pixel coordinates."
        )
        user_text = (
            f"Find the element described below on this screenshot and return "
            f"its CENTER coordinates as JSON.\n\n"
            f"Element to find: {description}\n\n"
            f"Image size: {model_w}x{model_h}. "
            f"Grid lines are labeled 0, 100, 200, ... along top (x) and left (y).\n\n"
            f"Return ONLY: {{\"x\": <number>, \"y\": <number>, \"found\": true/false}}\n"
            f"If you cannot find the element, return {{\"found\": false, \"x\": 0, \"y\": 0}}"
        )

        try:
            from gemini_vl import call_gemini
            response = call_gemini(
                system, user_text, img,
                max_tokens=256,
                api_key=self.api_key,
                model=self.model,
            )
            data = self._parse_gemini_json(response)
            if not data or not data.get("found", False):
                self.log("  [Fallback] Element not found on screen.", "warning")
                return False

            mx, my = int(data["x"]), int(data["y"])
            sx, sy = _model_to_screen(mx, my, scale, scr_w, scr_h, raw_w, raw_h)
            self.log(f"  [Fallback] Clicking at screen ({sx}, {sy})", "action")
            pyautogui.click(sx, sy)
            time.sleep(0.5)
            return True
        except Exception as e:
            self.log(f"  [Fallback] Vision click error: {e}", "error")
            return False

    def _vision_type(self, description, text):
        """Fallback: vision-click a field, then type into it with pyautogui.

        Returns True if text was typed.
        """
        clicked = self._vision_click(description)
        if not clicked:
            return False
        time.sleep(0.3)
        # Select all existing text and overwrite
        pyautogui.hotkey("ctrl", "a")
        time.sleep(0.1)
        pyautogui.write(text, interval=0.02)
        self.log(f"  [Fallback] Typed into '{description}': {text[:30]}...", "dim")
        return True

    def _vision_upload_file(self, file_path):
        """Fallback: vision-click an upload button/area, handle the OS file dialog."""
        clicked = self._vision_click(
            "file upload button OR 'Choose File' OR 'Upload Resume' OR 'Attach' button"
        )
        if not clicked:
            return False
        time.sleep(2)
        # OS file dialog: type the file path and press Enter
        pyautogui.write(file_path, interval=0.02)
        time.sleep(0.3)
        pyautogui.press("enter")
        time.sleep(1)
        self.log("  [Fallback] Uploaded file via OS dialog.", "info")
        return True

    # ── Form filling ────────────────────────────────────────────────────

    def _fill_and_submit_form(self, page_analysis):
        """Fill form fields using resume data and Gemini guidance."""
        fields = page_analysis.get("fields", [])
        if not fields:
            self.log("  No form fields identified.", "warning")
            return "skipped"

        filled_count = 0
        for field_info in fields:
            if not self.is_running():
                return "failed"

            label = field_info.get("label", "").lower()
            field_type = field_info.get("type", "text")
            value = self._get_field_value(label, field_type)

            if not value and field_type != "file":
                continue

            try:
                if field_type == "file":
                    filled = self._upload_resume_to_field(label)
                else:
                    filled = self._fill_field(label, value, field_type)

                if filled:
                    filled_count += 1
                    self.log(f"  Filled: {field_info.get('label', 'field')}", "dim")
            except Exception as e:
                self.log(f"  Could not fill {label}: {e}", "dim")

        if filled_count == 0:
            self.log("  Could not fill any fields.", "warning")
            return "failed"

        self.log(f"  Filled {filled_count} fields.", "info")

        submit = page_analysis.get("submit_button")
        if submit:
            clicked = self._click_element_by_analysis(submit)
            if clicked:
                time.sleep(3)
                self.log("  Form submitted.", "info")
                return "applied"

        return "applied" if filled_count > 0 else "failed"

    def _get_field_value(self, label, field_type):
        """Map a form field label to a value from the parsed resume."""
        if not self.resume_data:
            return ""

        label = label.lower().strip()
        rd = self.resume_data

        first_name_kw = ["first name", "first", "given name"]
        last_name_kw = ["last name", "last", "surname", "family name"]
        name_keywords = ["name", "full name", "your name", "applicant name"]
        email_kw = ["email", "e-mail", "email address"]
        phone_kw = ["phone", "telephone", "mobile", "cell", "phone number"]
        linkedin_kw = ["linkedin", "linkedin url", "linkedin profile"]
        github_kw = ["github", "github url", "github profile"]
        website_kw = ["website", "portfolio", "personal website", "url"]
        location_kw = ["location", "city", "address", "where are you located"]
        university_kw = ["university", "school", "college", "education", "institution"]
        degree_kw = ["degree", "major", "field of study"]
        gpa_kw = ["gpa", "grade", "cgpa"]
        grad_kw = ["graduation", "grad date", "expected graduation", "graduation date"]

        for kw in first_name_kw:
            if kw in label:
                name = rd.get("name", "")
                return name.split()[0] if name else ""
        for kw in last_name_kw:
            if kw in label:
                name = rd.get("name", "")
                parts = name.split()
                return parts[-1] if len(parts) > 1 else ""
        for kw in name_keywords:
            if kw in label:
                return rd.get("name", "")
        for kw in email_kw:
            if kw in label:
                return rd.get("email", "")
        for kw in phone_kw:
            if kw in label:
                return rd.get("phone", "")
        for kw in linkedin_kw:
            if kw in label:
                return rd.get("linkedin", "")
        for kw in github_kw:
            if kw in label:
                return rd.get("github", "")
        for kw in website_kw:
            if kw in label:
                return rd.get("website", rd.get("github", ""))
        for kw in location_kw:
            if kw in label:
                return rd.get("location", "")
        for kw in university_kw:
            if kw in label:
                return rd.get("university", "")
        for kw in degree_kw:
            if kw in label:
                return rd.get("degree", "")
        for kw in gpa_kw:
            if kw in label:
                return rd.get("gpa", "")
        for kw in grad_kw:
            if kw in label:
                return rd.get("graduation_date", "")

        return ""

    # ── Selenium element finders ────────────────────────────────────────

    def _find_by_label_text(self, label):
        labels = self.driver.find_elements(By.TAG_NAME, "label")
        for lbl in labels:
            if label.lower() in lbl.text.lower():
                for_attr = lbl.get_attribute("for")
                if for_attr:
                    try:
                        return self.driver.find_element(By.ID, for_attr)
                    except NoSuchElementException:
                        pass
                inputs = lbl.find_elements(
                    By.CSS_SELECTOR, "input, textarea, select",
                )
                if inputs:
                    return inputs[0]
        return None

    def _find_by_placeholder(self, label):
        inputs = self.driver.find_elements(
            By.CSS_SELECTOR, "input[placeholder], textarea[placeholder]",
        )
        for inp in inputs:
            ph = (inp.get_attribute("placeholder") or "").lower()
            if label.lower() in ph or any(w in ph for w in label.lower().split()):
                return inp
        return None

    def _find_by_aria_label(self, label):
        inputs = self.driver.find_elements(
            By.CSS_SELECTOR, "input[aria-label], textarea[aria-label]",
        )
        for inp in inputs:
            al = (inp.get_attribute("aria-label") or "").lower()
            if label.lower() in al:
                return inp
        return None

    def _find_by_nearby_text(self, label):
        try:
            label_lower = label.lower().replace("'", "\\'")
            xpath = (
                f"//*[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', "
                f"'abcdefghijklmnopqrstuvwxyz'), '{label_lower}')]"
                f"/following::input[1] | "
                f"//*[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', "
                f"'abcdefghijklmnopqrstuvwxyz'), '{label_lower}')]"
                f"/following::textarea[1]"
            )
            elements = self.driver.find_elements(By.XPATH, xpath)
            return elements[0] if elements else None
        except Exception:
            return None

    def _fill_select(self, element, value):
        try:
            select = Select(element)
            for option in select.options:
                if value.lower() in option.text.lower():
                    select.select_by_visible_text(option.text)
                    return True
            if len(select.options) > 1:
                select.select_by_index(1)
                return True
        except Exception:
            pass
        return False

    # ── Fill field (Selenium first, vision fallback) ────────────────────

    def _fill_field(self, label, value, field_type):
        """Find a form field and fill it. Tries Selenium DOM, falls back to vision+mouse."""
        if not value:
            return False

        # Strategy 1: Selenium DOM access
        strategies = [
            lambda: self._find_by_label_text(label),
            lambda: self._find_by_placeholder(label),
            lambda: self._find_by_aria_label(label),
            lambda: self._find_by_nearby_text(label),
        ]

        for strategy in strategies:
            try:
                element = strategy()
                if element and element.is_displayed():
                    if field_type == "select":
                        return self._fill_select(element, value)
                    element.clear()
                    element.send_keys(value)
                    return True
            except (NoSuchElementException, ElementNotInteractableException,
                    StaleElementReferenceException):
                continue
            except Exception:
                continue

        # Strategy 2: vision fallback — screenshot screen, find field, click + type
        self.log(f"  Selenium couldn't reach '{label}', trying vision fallback...", "warning")
        return self._vision_type(
            f"text input field labeled '{label}' on the web page", value,
        )

    def _upload_resume_to_field(self, label):
        """Upload resume PDF. Selenium first, vision fallback for tricky upload widgets."""
        # Strategy 1: Selenium file input
        file_inputs = self.driver.find_elements(
            By.CSS_SELECTOR, "input[type='file']",
        )
        for inp in file_inputs:
            try:
                accept = (inp.get_attribute("accept") or "").lower()
                if (not accept or "pdf" in accept
                        or "document" in accept or "*" in accept):
                    inp.send_keys(self.resume_path)
                    self.log("  Uploaded resume PDF (Selenium).", "info")
                    return True
            except Exception:
                continue

        # Strategy 2: vision fallback for custom upload widgets
        self.log("  No standard file input found, trying vision fallback...", "warning")
        return self._vision_upload_file(self.resume_path)

    def _click_element_by_analysis(self, button_info):
        """Click a button described by Gemini's analysis.
        Selenium first, then vision fallback."""
        text = (button_info.get("text") or "").strip()
        if not text:
            return False

        # Strategy 1: Selenium XPath
        strategies = [
            lambda: self.driver.find_element(
                By.XPATH,
                f"//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', "
                f"'abcdefghijklmnopqrstuvwxyz'), '{text.lower()}')]",
            ),
            lambda: self.driver.find_element(
                By.XPATH,
                f"//a[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', "
                f"'abcdefghijklmnopqrstuvwxyz'), '{text.lower()}')]",
            ),
            lambda: self.driver.find_element(
                By.XPATH,
                f"//input[@type='submit' and contains(translate(@value, "
                f"'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), "
                f"'{text.lower()}')]",
            ),
            lambda: self.driver.find_element(
                By.XPATH,
                f"//*[@role='button' and contains(translate(., "
                f"'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), "
                f"'{text.lower()}')]",
            ),
        ]

        for strategy in strategies:
            try:
                el = strategy()
                if el and el.is_displayed():
                    el.click()
                    return True
            except (NoSuchElementException, ElementNotInteractableException):
                continue
            except Exception:
                continue

        # Strategy 2: Selenium JavaScript click (bypasses overlay issues)
        for strategy in strategies:
            try:
                el = strategy()
                if el:
                    self.driver.execute_script("arguments[0].click();", el)
                    return True
            except Exception:
                continue

        # Strategy 3: vision fallback — find button on screen and pyautogui click
        desc = button_info.get("description", "")
        self.log(f"  Selenium couldn't click '{text}', trying vision fallback...", "warning")
        return self._vision_click(
            f"button or link with text '{text}'"
            + (f" ({desc})" if desc else "")
            + " on the web page",
        )
