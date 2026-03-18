"""
Core job application automation engine.

Opens Chrome via Selenium and attempts to auto-apply using:
  1. Direct Selenium field matching (primary, especially for Workday)
  2. Gemini vision page analysis (secondary)
  3. pyautogui + Gemini vision coordinate fallback (last resort)
"""
import io
import os
import re
import sys
import json
import time

from PIL import Image, ImageDraw, ImageFont
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    ElementNotInteractableException,
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

WORKDAY_TEST_URL = "https://resmed.wd3.myworkdayjobs.com/en-US/ResMed_External_Careers/job/Halifax-Canada/Software-Engineer-Intern_JR_047978/apply/applyManually"

GRID_SPACING = 100


# ── Vision helpers ──────────────────────────────────────────────────────

def _draw_grid(img, spacing=GRID_SPACING):
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
    ss = pyautogui.screenshot()
    screen_size = pyautogui.size()
    return ss, screen_size[0], screen_size[1]


def _prepare_for_model(screenshot, screen_w, screen_h):
    img = screenshot.convert("RGB")
    ss_w, ss_h = img.size
    scale = 1024 / max(ss_w, ss_h) if max(ss_w, ss_h) > 1024 else 1.0
    if scale < 1.0:
        img = img.resize(
            (int(ss_w * scale), int(ss_h * scale)),
            Image.Resampling.LANCZOS,
        )
    _draw_grid(img)
    return img, scale


def _model_to_screen(mx, my, scale, screen_w, screen_h, ss_w, ss_h):
    img_x = mx / scale
    img_y = my / scale
    sx = int(round(img_x * screen_w / ss_w))
    sy = int(round(img_y * screen_h / ss_h))
    sx = max(0, min(screen_w - 1, sx))
    sy = max(0, min(screen_h - 1, sy))
    return sx, sy


class JobAutomator:
    def __init__(self, resume_path, api_key, model, log_fn, running_fn, stats_fn):
        self.resume_path = os.path.abspath(resume_path)
        self.api_key = (api_key or "").strip()
        self.model = model
        self.log = log_fn
        self.is_running = running_fn
        self.stats_fn = stats_fn
        self.driver = None
        self.resume_data = None

    # ── Main ───────────────────────────────────────────────────────────

    def run(self):
        try:
            self._parse_resume()
            if not self.is_running():
                return

            self._setup_chrome()
            if not self.is_running():
                return

            jobs = [{
                "company": "Resmed",
                "role": "Test Job",
                "location": "Halifax, Canada",
                "url": WORKDAY_TEST_URL,
            }]

            self.stats_fn("found", len(jobs))
            self.log(f"Found {len(jobs)} open positions with apply links.", "success")
            self._apply_to_jobs(jobs)

        except Exception as e:
            import traceback
            self.log(f"Error: {e}", "error")
            self.log(traceback.format_exc(), "dim")
        finally:
            if self.driver:
                try:
                    self.driver.quit()
                except Exception as e:
                    self.log(f"Cleanup warning during quit: {e}", "dim")
                self.log("Chrome closed.", "info")

    # ── Setup ──────────────────────────────────────────────────────────

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

    # ── Apply loop ─────────────────────────────────────────────────────

    def _apply_to_jobs(self, jobs):
        applied = 0
        skipped = 0
        failed = 0

        for i, job in enumerate(jobs):
            if not self.is_running():
                self.log("Stopped by user.", "warning")
                break

            self.log(f"\n[{i+1}/{len(jobs)}] {job['company']} — {job['role']}", "header")
            self.log(f"  URL: {job['url']}", "dim")

            try:
                result = self._apply_to_single_job(job)
                if result == "applied":
                    applied += 1
                    self.stats_fn("applied", applied)
                    self.log("  Applied successfully!", "success")
                elif result == "skipped":
                    skipped += 1
                    self.stats_fn("skipped", skipped)
                    self.log("  Skipped.", "warning")
                else:
                    failed += 1
                    self.stats_fn("failed", failed)
                    self.log("  Could not complete application.", "error")
            except Exception as e:
                failed += 1
                self.stats_fn("failed", failed)
                self.log(f"  Error: {e}", "error")

            time.sleep(2)

        self.log(f"\nDone! Applied: {applied}, Skipped: {skipped}, Failed: {failed}", "header")

    def _apply_to_single_job(self, job):
        url = job["url"]

        try:
            self.driver.execute_script("window.open('');")
            self.driver.switch_to.window(self.driver.window_handles[-1])
            self.driver.get(url)
            time.sleep(5)

            # First: try direct Workday filling without Gemini
            filled = self._workday_direct_fill()
            if filled:
                self.log("  Direct Workday fill completed.", "success")
                return "applied"

            # Second: try Gemini page analysis if direct fill didn't work
            page_analysis = self._analyze_page()
            if not page_analysis:
                return "failed"

            page_type = page_analysis.get("page_type", "unknown")
            self.log(f"  Page type: {page_type}", "dim")

            if page_type == "login_required":
                return "skipped"

            if page_type == "job_description_only":
                apply_btn = page_analysis.get("apply_button")
                if apply_btn and self._click_element_by_analysis(apply_btn):
                    time.sleep(3)
                    page_analysis = self._analyze_page()
                    if not page_analysis:
                        return "failed"
                    page_type = page_analysis.get("page_type", "unknown")
                else:
                    return "skipped"

            if page_type in ("application_form", "multi_step_form"):
                return self._fill_and_submit_form(page_analysis)

            return "skipped"

        except Exception as e:
            self.log(f"  Error inside _apply_to_single_job: {e}", "error")
            return "failed"

        finally:
            try:
                handles = self.driver.window_handles
                if len(handles) > 1:
                    self.driver.close()
                    self.driver.switch_to.window(handles[0])
            except Exception as e:
                self.log(f"  Cleanup warning: {e}", "dim")

    # ── Workday direct fill ────────────────────────────────────────────

    def _workday_direct_fill(self):
        """
        Fill common Workday fields without Gemini.
        This is the important fallback when Gemini API is broken.
        """
        self.log("  Trying direct Workday fill...", "action")

        time.sleep(2)
        self._click_apply_or_continue_if_present()
        time.sleep(2)

        filled_any = False

        field_map = [
            (["first name", "given name"], "text", self._resume("first_name")),
            (["last name", "family name", "surname"], "text", self._resume("last_name")),
            (["full name", "legal name", "name"], "text", self._resume("name")),
            (["email", "email address"], "text", self._resume("email")),
            (["phone", "mobile", "phone number"], "text", self._resume("phone")),
            (["linkedin"], "text", self._resume("linkedin")),
            (["github"], "text", self._resume("github")),
            (["website", "portfolio", "personal website"], "text", self._resume("website")),
            (["city", "location", "address"], "text", self._resume("location")),
            (["school", "university", "college"], "text", self._resume("university")),
            (["degree", "major", "field of study"], "text", self._resume("degree")),
            (["gpa"], "text", self._resume("gpa")),
            (["graduation", "graduation date", "expected graduation"], "text", self._resume("graduation_date")),
        ]

        for labels, field_type, value in field_map:
            if not value:
                continue
            for label in labels:
                if self._fill_field(label, value, field_type):
                    self.log(f"  Filled direct field for '{label}'", "dim")
                    filled_any = True
                    break

        if self._upload_resume_to_field("resume"):
            filled_any = True

        self._click_apply_or_continue_if_present()
        self._click_submit_if_present()

        return filled_any

    def _resume(self, key):
        if not self.resume_data:
            return ""
        if key == "first_name":
            name = self.resume_data.get("name", "")
            return name.split()[0] if name else ""
        if key == "last_name":
            name = self.resume_data.get("name", "")
            parts = name.split()
            return parts[-1] if len(parts) > 1 else ""
        if key == "website":
            return self.resume_data.get("website", self.resume_data.get("github", ""))
        return self.resume_data.get(key, "")

    def _click_apply_or_continue_if_present(self):
        texts = [
            "apply manually",
            "apply",
            "continue",
            "next",
            "continue with application",
        ]
        for text in texts:
            try:
                btn = self.driver.find_element(
                    By.XPATH,
                    f"//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{text}')] | "
                    f"//a[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{text}')]"
                )
                if btn.is_displayed():
                    btn.click()
                    self.log(f"  Clicked '{text}'", "dim")
                    time.sleep(2)
                    return True
            except Exception:
                continue
        return False

    def _click_submit_if_present(self):
        texts = ["submit", "send application", "review and submit"]
        for text in texts:
            try:
                btn = self.driver.find_element(
                    By.XPATH,
                    f"//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{text}')]"
                )
                if btn.is_displayed():
                    self.log(f"  Submit button found: '{text}'", "info")
                    return True
            except Exception:
                continue
        return False

    # ── Screenshots ────────────────────────────────────────────────────

    def _take_page_screenshot(self):
        png = self.driver.get_screenshot_as_png()
        return Image.open(io.BytesIO(png)).convert("RGB")

    def _take_screen_for_vision(self):
        self._bring_chrome_to_front()
        time.sleep(0.3)
        ss, scr_w, scr_h = _take_screen_screenshot()
        raw_w, raw_h = ss.size
        img, scale = _prepare_for_model(ss, scr_w, scr_h)
        return img, scale, scr_w, scr_h, raw_w, raw_h

    def _bring_chrome_to_front(self):
        try:
            self.driver.switch_to.window(self.driver.current_window_handle)
            self.driver.execute_script("window.focus(); document.title = document.title;")
        except Exception:
            pass

    # ── Gemini analysis ────────────────────────────────────────────────

    def _analyze_page(self):
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
}"""

        try:
            self.log(f"  DEBUG model: {repr(self.model)}", "dim")
            self.log(f"  DEBUG api_key prefix: {repr(str(self.api_key)[:12])}", "dim")

            from gemini_vl import call_gemini
            response = call_gemini(
                system,
                user_text,
                screenshot,
                max_tokens=2048,
                api_key=self.api_key,
                model=self.model,
            )
            return self._parse_gemini_json(response)
        except Exception as e:
            self.log(f"  Gemini analysis error: {e}", "error")
            return None

    def _parse_gemini_json(self, response):
        response = re.sub(
            r"<thinking\s*>.*?</thinking\s*>",
            "",
            response,
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()
        if "```" in response:
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
            if m:
                response = m.group(1)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            self.log("  Could not parse Gemini response as JSON.", "warning")
            return None

    # ── Vision fallback ────────────────────────────────────────────────

    def _vision_click(self, description):
        self.log(f"  [Fallback] Vision-clicking: {description}", "action")
        img, scale, scr_w, scr_h, raw_w, raw_h = self._take_screen_for_vision()
        model_w, model_h = img.size

        system = (
            "You are a screen coordinate finder. The image has a grid overlay "
            "with labels every 100px along the edges."
        )
        user_text = (
            f"Find the element described below on this screenshot and return "
            f"its CENTER coordinates as JSON.\n\n"
            f"Element to find: {description}\n\n"
            f"Image size: {model_w}x{model_h}.\n\n"
            f"Return ONLY: {{\"x\": <number>, \"y\": <number>, \"found\": true/false}}"
        )

        try:
            from gemini_vl import call_gemini
            response = call_gemini(
                system,
                user_text,
                img,
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
            pyautogui.click(sx, sy)
            time.sleep(0.5)
            return True
        except Exception as e:
            self.log(f"  [Fallback] Vision click error: {e}", "error")
            return False

    def _vision_type(self, description, text):
        clicked = self._vision_click(description)
        if not clicked:
            return False
        time.sleep(0.3)
        if sys.platform == "darwin":
            pyautogui.hotkey("command", "a")
        else:
            pyautogui.hotkey("ctrl", "a")
        time.sleep(0.1)
        pyautogui.write(text, interval=0.02)
        self.log(f"  [Fallback] Typed into '{description}': {text[:30]}...", "dim")
        return True

    def _vision_upload_file(self, file_path):
        clicked = self._vision_click(
            "file upload button OR 'Choose File' OR 'Upload Resume' OR 'Attach' button"
        )
        if not clicked:
            return False
        time.sleep(2)
        pyautogui.write(file_path, interval=0.02)
        time.sleep(0.3)
        pyautogui.press("enter")
        time.sleep(1)
        self.log("  [Fallback] Uploaded file via OS dialog.", "info")
        return True

    # ── Form filling ───────────────────────────────────────────────────

    def _fill_and_submit_form(self, page_analysis):
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

        submit = page_analysis.get("submit_button")
        if submit and self._click_element_by_analysis(submit):
            time.sleep(3)
            self.log("  Form submitted.", "info")

        return "applied"

    def _get_field_value(self, label, field_type):
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

    # ── Selenium element finders ───────────────────────────────────────

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
                inputs = lbl.find_elements(By.CSS_SELECTOR, "input, textarea, select")
                if inputs:
                    return inputs[0]
        return None

    def _find_by_placeholder(self, label):
        inputs = self.driver.find_elements(
            By.CSS_SELECTOR, "input[placeholder], textarea[placeholder]"
        )
        for inp in inputs:
            ph = (inp.get_attribute("placeholder") or "").lower()
            if label.lower() in ph or any(w in ph for w in label.lower().split()):
                return inp
        return None

    def _find_by_aria_label(self, label):
        inputs = self.driver.find_elements(
            By.CSS_SELECTOR, "input[aria-label], textarea[aria-label]"
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
                f"//*[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{label_lower}')]/following::input[1] | "
                f"//*[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{label_lower}')]/following::textarea[1]"
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

    def _fill_field(self, label, value, field_type):
        if not value:
            return False

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
            except (NoSuchElementException, ElementNotInteractableException, StaleElementReferenceException):
                continue
            except Exception:
                continue

        self.log(f"  Selenium couldn't reach '{label}', trying vision fallback...", "warning")
        return self._vision_type(f"text input field labeled '{label}' on the web page", value)

    def _upload_resume_to_field(self, label):
        file_inputs = self.driver.find_elements(By.CSS_SELECTOR, "input[type='file']")
        for inp in file_inputs:
            try:
                accept = (inp.get_attribute("accept") or "").lower()
                if not accept or "pdf" in accept or "document" in accept or "*" in accept:
                    inp.send_keys(self.resume_path)
                    self.log("  Uploaded resume PDF (Selenium).", "info")
                    return True
            except Exception:
                continue

        self.log("  No standard file input found, trying vision fallback...", "warning")
        return self._vision_upload_file(self.resume_path)

    def _click_element_by_analysis(self, button_info):
        text = (button_info.get("text") or "").strip()
        if not text:
            return False

        strategies = [
            lambda: self.driver.find_element(
                By.XPATH,
                f"//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{text.lower()}')]",
            ),
            lambda: self.driver.find_element(
                By.XPATH,
                f"//a[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{text.lower()}')]",
            ),
            lambda: self.driver.find_element(
                By.XPATH,
                f"//input[@type='submit' and contains(translate(@value, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{text.lower()}')]",
            ),
            lambda: self.driver.find_element(
                By.XPATH,
                f"//*[@role='button' and contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{text.lower()}')]",
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

        for strategy in strategies:
            try:
                el = strategy()
                if el:
                    self.driver.execute_script("arguments[0].click();", el)
                    return True
            except Exception:
                continue

        desc = button_info.get("description", "")
        self.log(f"  Selenium couldn't click '{text}', trying vision fallback...", "warning")
        return self._vision_click(
            f"button or link with text '{text}'" + (f" ({desc})" if desc else "") + " on the web page"
        )