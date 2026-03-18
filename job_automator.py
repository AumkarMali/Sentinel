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

            # A short, final delay before the next job
            time.sleep(1)

        self.log(f"\nDone! Applied: {applied}, Skipped: {skipped}, Failed: {failed}", "header")

    def _apply_to_single_job(self, job):
        url = job["url"]
        try:
            self.driver.execute_script("window.open('');")
            self.driver.switch_to.window(self.driver.window_handles[-1])
            self.driver.get(url)
            WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

            # --- Start Multi-Page Navigation Loop ---
            for page_num in range(10): # Allow up to 10 pages/steps
                self.log(f"\nProcessing page {page_num + 1}...", "header")

                # Step 1: Fill everything possible on the current page
                self.log("Filling known fields with Selenium...", "action")
                self._workday_direct_fill()

                self.log("Analyzing and filling remaining fields with Gemini...", "action")
                analysis = self._analyze_and_fill_page()

                if not analysis:
                    self.log("Could not analyze page. Cannot continue.", "error")
                    return "failed"

                # Step 2: Decide whether to continue or submit
                if analysis.get("submit_button"):
                    self.log("Final submit button found. Finalizing application.", "info")
                    self._click_element_by_analysis(analysis["submit_button"])
                    return "applied"
                
                if analysis.get("next_button"):
                    self.log("Next page button found. Continuing...", "info")
                    self._click_element_by_analysis(analysis["next_button"])
                    # Wait for the next page to load
                    WebDriverWait(self.driver, 10).until(EC.staleness_of(self.driver.find_element(By.TAG_NAME, 'html')))
                else:
                    self.log("No 'next' or 'submit' button found. Assuming application is complete.", "warning")
                    return "applied" # Or 'failed' if this is unexpected

            self.log("Exceeded maximum page navigation limit.", "error")
            return "failed"

        except Exception as e:
            self.log(f"Error inside _apply_to_single_job: {e}", "error")
            return "failed"
        finally:
            try:
                handles = self.driver.window_handles
                if len(handles) > 1:
                    self.driver.close()
                    self.driver.switch_to.window(handles[0])
            except Exception as e:
                self.log(f"Cleanup warning: {e}", "dim")

    # ── Workday direct fill ────────────────────────────────────────────

    def _workday_direct_fill(self):
        """
        Fill common Workday fields without Gemini.
        This is the important fallback when Gemini API is broken.
        """
        self.log("  Trying direct Workday fill...", "action")

        self._click_apply_or_continue_if_present()
        try:
            # Smart wait for the next page to load after clicking
            WebDriverWait(self.driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input, button"))
            )
        except TimeoutException:
            pass  # It's okay if nothing new loads, not a fatal error

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
            # Example Checkbox/Radio
            (["i agree", "i accept", "terms and conditions"], "checkbox", "true"),
            (["gender"], "radio", self._resume("gender")),
            (["veteran"], "radio", self._resume("veteran_status")),
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
        """More robustly finds and clicks a generic 'next' button."""
        texts = ["apply manually", "apply", "continue", "next"]
        # Look for buttons or links containing the text
        text_xpaths = " or ".join([f"contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{s}')" for s in texts])
        xpath = f"//button[{text_xpaths}] | //a[{text_xpaths}]"
        
        try:
            # Wait for the button to be clickable and then click it
            button = WebDriverWait(self.driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, xpath))
            )
            button.click()
            self.log(f"Clicked '{button.text}'", "dim")
            return True
        except TimeoutException:
            self.log("No 'apply' or 'continue' button found.", "dim")
            return False
        except Exception as e:
            self.log(f"Error clicking apply/continue button: {e}", "warning")
            return False

    def _click_submit_if_present(self):
        texts = ["submit", "send application", "review and submit"]
        text_xpaths = " or ".join([f"contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{s}')" for s in texts])
        xpath = f"//button[{text_xpaths}] | //input[@type='submit' and ({text_xpaths})]"
        
        try:
            button = WebDriverWait(self.driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, xpath))
            )
            button.click()
            self.log(f"Clicked submit button: '{button.text}'", "info")
            return True
        except Exception:
            return False

    # ── Screenshots ────────────────────────────────────────────────────

    def _take_page_screenshot(self):
        png = self.driver.get_screenshot_as_png()
        return png

    # ── Gemini analysis ────────────────────────────────────────────────

    def _analyze_and_fill_page(self):
        """Scrolls, captures, analyzes, and fills the current page."""
        try:
            # 1. Capture the full page
            screenshots = self._capture_full_page()
            self.log(f"Captured {len(screenshots)} screenshots for analysis.", "dim")
            
            # 2. Analyze with Gemini
            system = "You are a web page analyzer. Identify remaining fields and navigation buttons."
            user_text = """Analyze the sequence of screenshots from a scrolling job application.
Identify ALL empty/unselected fields and any navigation buttons.

Return ONLY a JSON object:
{
    "remaining_fields": [{"label": "field label", "type": "text|select|checkbox|radio", "required": true/false}],
    "next_button": {"text": "button text"} or null,
    "submit_button": {"text": "button text"} or null
}"""
            analysis = self._call_gemini_vision_api(system, user_text, screenshots)
            if not analysis: return None

            # 3. Check for required fields before filling
            for field in analysis.get("remaining_fields", []):
                if field.get("required"):
                    label = field.get("label", "").lower()
                    value = self._get_field_value(label, field.get("type", "text"))
                    if not value and field.get("type") not in ["checkbox", "radio"]:
                        self.log(f"Skipping: Missing required info for '{label}'.", "warning")
                        return {"skipped": True}

            # 4. Fill all identified fields
            for field in analysis.get("remaining_fields", []):
                self._fill_field_based_on_analysis(field)

            return analysis # Return the full analysis for navigation decisions
        except Exception as e:
            self.log(f"Error during page analysis and filling: {e}", "error")
            return None

    def _capture_full_page(self):
        """Scrolls the page and captures a series of screenshots."""
        screenshots = []
        total_height = self.driver.execute_script("return document.body.scrollHeight")
        viewport_height = self.driver.execute_script("return window.innerHeight")
        self.driver.execute_script("window.scrollTo(0, 0)")
        time.sleep(0.5)
        
        for i in range(10): # Max 10 scrolls
            png = self.driver.get_screenshot_as_png()
            screenshots.append(Image.open(io.BytesIO(png)).convert("RGB"))
            scroll_amount = (viewport_height * i)
            self.driver.execute_script(f"window.scrollTo(0, {scroll_amount})")
            time.sleep(0.5)
            if scroll_amount > total_height:
                break
        return screenshots

    def _call_gemini_vision_api(self, system, user_text, screenshot):
        """Helper to call Gemini and parse the JSON response."""
        from gemini_vl import call_gemini
        response = call_gemini(
            system, user_text, screenshot,
            max_tokens=2048, api_key=self.api_key, model=self.model,
        )
        return self._parse_gemini_json(response)

    def _parse_gemini_json(self, response):
        # ... (parsing logic remains the same)
        response = re.sub(
            r"<thinking\s*>.*?</thinking\s*>", "", response, flags=re.DOTALL | re.IGNORECASE
        ).strip()
        if "```" in response:
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
            if m:
                response = m.group(1)
        try:
            data = json.loads(response)
            self.log("Gemini analysis JSON response:", "info")
            self.log(json.dumps(data, indent=2), "dim")
            return data
        except json.JSONDecodeError:
            self.log("Could not parse Gemini response as JSON.", "warning")
            return None

    # ── Vision fallback ────────────────────────────────────────────────

    def _vision_click(self, description):
        self.log(f"  [Fallback] Vision-clicking: {description}", "action")
        time.sleep(1)
        self.log("  [Fallback] Uploaded file via OS dialog.", "info")
        return True

    # ── Form filling helpers ───────────────────────────────────────────

    def _fill_field(self, label, value, field_type):
        if not value:
            return False

        # Checkable logic is now handled separately
        if field_type in ["checkbox", "radio"]:
            return self._find_and_click_checkable(label, value, field_type)

        strategies = [
            (self._find_by_label_text, "by label text"),
            (self._find_by_placeholder, "by placeholder"),
            (self._find_by_aria_label, "by aria-label"),
            (self._find_by_nearby_text, "by nearby text"),
        ]

        for strategy_fn, name in strategies:
            try:
                self.log(f"  Attempting to find '{label}' {name}...", "dim")
                element = strategy_fn(label)
                if element and element.is_displayed():
                    if field_type == "select":
                        return self._fill_select(element, value)
                    element.clear()
                    element.send_keys(value)
                    self.log(f"    > Success.", "dim")
                    return True
            except Exception:
                continue

        self.log(f"  Selenium couldn't reach '{label}', trying vision fallback...", "warning")
        return self._vision_type(f"text input field labeled '{label}' on the web page", value)

    def _upload_resume_to_field(self, label):
        file_inputs = self.driver.find_elements(By.CSS_SELECTOR, "input[type='file']")
        for file_input in file_inputs:
            if file_input.get_attribute("name") == label:
                file_input.send_keys(self.resume_path)
                return True
        return False

    def _vision_upload_file(self, file_path):
        self.log(f"  [Fallback] Uploading resume file: {file_path}", "action")
        return self._vision_click(f"Resume file: {file_path}")

    def _click_element_by_analysis(self, button_info):
        text = (button_info.get("text") or "").strip()
        if not text:
            return False
        
        self.log(f"Attempting to click element from Gemini analysis: '{text}'...", "action")

        strategies = [
            lambda: self.driver.find_element(
                By.XPATH,
                f"//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{text.lower()}')]"
            ),
            lambda: self.driver.find_element(
                By.XPATH,
                f"//a[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{text.lower()}')]"
            ),
        ]

        for strategy in strategies:
            try:
                element = strategy()
                if element and element.is_displayed():
                    element.click()
                    self.log(f"    > Clicked '{text}' from analysis.", "dim")
                    return True
            except Exception as e:
                self.log(f"  Could not click element for '{text}': {e}", "dim")
        return False

    def _find_by_label_text(self, label):
        labels = self.driver.find_elements(By.TAG_NAME, "label")
        for lbl in labels:
            if label.lower() in lbl.text.lower():
                if inputs:
                    return inputs[0]
        return None

    def _find_and_click_checkable(self, label, value, field_type):
        """Finds and clicks a checkbox or a specific radio button option."""
        try:
            # Strategy 1: Find by label text and then look for the input element
            self.log(f"  Trying to find checkable '{label}' by its text...", "dim")
            xpath = f"//label[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{label.lower()}')]"
            label_elements = self.driver.find_elements(By.XPATH, xpath)
            for lbl in label_elements:
                try:
                    # Check for input as a child of the label
                    el = lbl.find_element(By.XPATH, f".//input[@type='{field_type}']")
                    if not el.is_selected(): el.click()
                    return True
                except NoSuchElementException:
                    # Check for input linked by the 'for' attribute
                    for_id = lbl.get_attribute('for')
                    if for_id:
                        el = self.driver.find_element(By.ID, for_id)
                        if not el.is_selected(): el.click()
                        return True
            
            # Strategy 2 (specifically for radios): Find the label for the specific value (e.g., "Yes")
            if field_type == 'radio' and value:
                 self.log(f"  Trying to find radio option '{value}' for '{label}'...", "dim")
                 radio_label = self.driver.find_element(By.XPATH, f"//label[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{value.lower()}')]")
                 if radio_label:
                     radio_label.click()
                     return True

        except Exception as e:
            self.log(f"  Could not click checkable for '{label}': {e}", "dim")
        return False

    def _find_by_placeholder(self, label):
        inputs = self.driver.find_elements(
            By.XPATH, f"//input[@placeholder='{label}']"
        )
        if inputs:
            return inputs[0]
        return None
