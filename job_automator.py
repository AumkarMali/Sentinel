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

import datetime
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
    ElementClickInterceptedException,
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

WORKDAY_TEST_URL = "https://synchronyfinancial.wd5.myworkdayjobs.com/University/job/Other-Remote-NY/Technology-Intern_2404484-1?utm_source=Simplify&ref=Simplify"

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
                "company": "Synchrony Financial",
                "role": "Technology Intern",
                "location": "Remote, NY",
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
                self.log("Keeping Chrome open for manual review.", "info")

    # ── Setup ──────────────────────────────────────────────────────────

    def _parse_resume(self):
        self.log("Parsing resume...", "action")
        from resume_parser import parse_resume
        self.resume_data = parse_resume(self.resume_path, self.api_key, self.model)
        self.resume_data["password"] = "AM20060305!_ilovesushi" # Hardcoded password
        self.resume_data["phone"] = "6575697753" # Hardcoded phone
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
        options.add_experimental_option("detach", True)

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
            self.log(f"Opened URL: {url}", "dim")
            
            for step in range(15): # Allow up to 15 steps/pages
                if not self.is_running(): return "failed"
                
                self.log(f"\n[Step {step+1}] Analyzing page...", "header")
                self._wait_for_page_load()
                time.sleep(3) # Buffer for dynamic content

                html = self._get_cleaned_html()
                
                instructions = self._get_instructions_from_gemini(html)
                if not instructions:
                    self.log("No instructions received or error parsing. Retrying...", "warning")
                    time.sleep(2)
                    continue

                if instructions.get("status") == "complete":
                    self.log("Application marked as complete by agent.", "success")
                    return "applied"

                actions = instructions.get("actions", [])
                if not actions:
                    self.log("No actions to perform. Checking for navigation fallback...", "warning")
                    if not self._click_apply_or_continue_if_present():
                         return "failed"
                    continue

                self.log(f"Performing {len(actions)} actions...", "action")
                for action in actions:
                    if not self.is_running(): return "failed"
                    self._execute_action(action)
                
                time.sleep(2) # Wait for any transitions

            return "failed"

        except Exception as e:
            self.log(f"Error inside _apply_to_single_job: {e}", "error")
            return "failed"

    def _get_cleaned_html(self):
        """Extracts and cleans HTML to reduce token usage."""
        script = """
        var body = document.body.cloneNode(true);
        var junk = body.querySelectorAll('script, style, svg, path, link, meta, noscript, iframe, video');
        junk.forEach(n => n.remove());
        return body.innerHTML;
        """
        html = self.driver.execute_script(script)
        html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)
        html = re.sub(r'\s+', ' ', html).strip()
        
        # If we are stuck on the same page, we need to ensure Gemini sees the errors.
        # Workday errors are often in <p data-automation-id="inputAlert">
        # The DOM should naturally contain them now, but we just need to ensure they aren't stripped.
        
        return html[:100000] # Cap at 100k chars for Gemini Flash

    def _get_instructions_from_gemini(self, html):
        system = "You are an expert web automation agent. Your goal is to fill out job applications."
        
        resume_summary = json.dumps(self.resume_data, indent=2)
        
        user_text = f"""
I am applying for a job. Here is my resume data:
{resume_summary}

The current date is: {datetime.date.today().strftime('%B %d, %Y')}

And here is the HTML of the current page:
{html}

Provide a JSON object with the next steps to take.
Identify all visible form fields (text, select, radio, checkbox, file) and buttons.
Pay special attention to any error messages (e.g., `<p data-automation-id="inputAlert">`) which indicate fields that were missed or filled incorrectly.
If this is a login or 'create account' page, use my email to sign in or create one.
If it's an application form, fill everything.

Your entire response MUST be ONLY the JSON object, starting with `{{` and ending with `}}`. Do not include ```json or any other text.
Return ONLY a JSON object:
{{
  "status": "continue" | "complete" | "error",
  "actions": [
    {{
      "action": "type" | "click" | "select" | "upload",
      "xpath": "the most specific and robust XPath for the element",
      "value": "the value to type/select/upload",
      "description": "briefly what this does"
    }}
  ]
}}

Notes:
- For Workday, always prioritize elements with 'data-automation-id'.
- **Navigation Buttons**: The 'Save and Continue', 'Next', or 'Submit' buttons in Workday almost always have `data-automation-id='pageFooterNextButton'`. Use this XPath whenever possible to progress or submit the final application.
- **Dates**: If a field asks for today's date (like an electronic signature date), use the current date provided above.
- **Education/Degree**: If the field asks for "Field of Study" or "Major", use "Aerospace Engineering".
- **Search Boxes**: Some dropdowns require you to type a value into an `<input data-automation-id='searchBox'>` and then select a result. For these, use `"action": "select"` and target the search input itself. The engine will handle the typing.
- **Custom Dropdowns**: Workday often uses `<button>` elements with `aria-haspopup="listbox"` instead of standard `<select>` tags. You should still use the `"action": "select"` for these, targeting the button element.
- **Error Handling**: If you see error messages (like `<p data-automation-id="inputAlert">Error...</p>` or a div saying 'Must end after start date'), prioritize actions to fix those specific fields. If an end date is before a start date, increase the end date by 1 year to make it valid.
- **Checkboxes**: Workday checkboxes are often hidden or intercepted. Always target the actual `<input type="checkbox">` tag. Use `action: "click"`. Good XPaths are `//input[@type='checkbox']` or specific IDs like `//input[@id='termsAndConditions--acceptTermsAndAgreements']`. Do NOT target the `<label>`.
- **Click Interception**: Sometimes a `<div>` with `role='button'` and a `data-automation-id` like 'click_filter' will cover the actual form submission button. When you see this pattern for 'Create Account' or 'Submit', target the covering `<div>` for the click action.
- **IMPORTANT**: On 'Create Account' pages, the agreement checkbox is an `<input>` tag. Target it directly with `//input[@data-automation-id='createAccountCheckbox']`.
- **IMPORTANT**: On 'Create Account' pages, use the `password` from the resume data for both 'Password' and 'Verify Password' fields. Generate exactly one action for each. Do not duplicate actions.
- For 'upload', the value should be 'RESUME_PATH'.
- For 'select', the value should be the exact text of the option to pick.
- For 'click', include navigation buttons like 'Next', 'Continue', 'Submit', 'Apply'. Remember to use `//button[@data-automation-id='pageFooterNextButton']` for these.
- If the application is finished or it says 'Application Submitted' or 'Congratulations', set status to 'complete'.
- Be extremely precise with XPaths. Prefer IDs or unique text.
"""
        from gemini_vl import call_gemini
        try:
            response = call_gemini(
                system=system,
                user_text=user_text,
                api_key=self.api_key,
                model=self.model,
                max_tokens=4096  # Increased to prevent truncation
            )
            return self._parse_gemini_json(response)
        except Exception as e:
            self.log(f"Gemini agent error: {e}", "error")
            return None

    def _execute_action(self, action):
        act_type = action.get("action")
        xpath = action.get("xpath")
        value = action.get("value")
        desc = action.get("description", "action")

        self.log(f"  > {act_type}: {desc}", "dim")

        try:
            element = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.XPATH, xpath))
            )
            # Better scrolling to avoid headers/footers
            self.driver.execute_script("arguments[0].scrollIntoView(true);", element)
            time.sleep(0.5)
            self.driver.execute_script("window.scrollBy(0, -150);") 
            time.sleep(0.3)

            # --- Skip if already filled ---
            if act_type == "type":
                # Check value attribute AND text content for inputs/textareas
                curr_val = element.get_attribute("value") or element.text or ""
                if curr_val.strip() and curr_val.strip().lower() == str(value).strip().lower():
                    self.log(f"    Skipping '{desc}': Already filled with '{curr_val}'", "dim")
                    return True
                element.clear()
                element.send_keys(value)

            elif act_type == "click":
                # If it's a radio or checkbox, check if already selected
                el_type = (element.get_attribute("type") or "").lower()
                if el_type in ["radio", "checkbox"]:
                    if element.is_selected():
                        self.log(f"    Skipping '{desc}': Already selected", "dim")
                        return True
                
                try:
                    # Some checkboxes in Workday require scrolling further down to not be obscured
                    self.driver.execute_script("window.scrollBy(0, 150);")
                    time.sleep(0.3)
                    element.click()
                except:
                    # Aggressive fallback for intercepted radio buttons/checkboxes
                    try:
                        self.driver.execute_script("arguments[0].click();", element)
                    except:
                        self.log(f"    Warning: Could not click {desc} via standard or JS method.", "warning")

            elif act_type == "select":
                # Robust check for current selection in Workday buttons
                curr_text = ""
                if element.tag_name == "select":
                    try:
                        curr_text = Select(element).first_selected_option.text
                    except: pass
                else:
                    curr_text = element.text or element.get_attribute("value") or element.get_attribute("title") or ""

                if curr_text.strip() and curr_text.strip().lower() not in ["select one", "select", ""]:
                    # It's already filled with something.
                    self.log(f"    Skipping '{desc}': Already selected '{curr_text}'", "dim")
                    return True

                if element.tag_name == "select":
                    select = Select(element)
                    try:
                        select.select_by_visible_text(value)
                    except:
                        found = False
                        for option in select.options:
                            if value.lower() in option.text.lower():
                                select.select_by_visible_text(option.text)
                                found = True
                                break
                        if not found: self.log(f"    Failed to select '{value}'", "warning")
                else:
                    # Workday custom dropdown OR Search Box
                    is_input = element.tag_name == "input"
                    if is_input:
                        self.log(f"    Search-box detected, typing '{value}'...", "dim")
                        element.clear()
                        element.send_keys(value)
                        time.sleep(2) # Wait for search results
                    else:
                        self.log(f"    Custom dropdown detected, clicking to open...", "dim")
                        self.driver.execute_script("arguments[0].click();", element)
                        time.sleep(1) # wait for overlay
                        
                        # Sometimes workday requires two clicks or focus+click
                        if element.get_attribute("aria-expanded") != "true":
                            self.driver.execute_script("arguments[0].focus(); arguments[0].click();", element)
                            time.sleep(1)

                    # Search for options in the entire body (Workday uses overlays)
                    option_xpath = f"//*[(@role='option' or @role='menuitem' or @data-automation-id='promptOption') and contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{value.lower()}')]"
                    try:
                        option_el = WebDriverWait(self.driver, 5).until(
                            EC.presence_of_element_located((By.XPATH, option_xpath))
                        )
                        self.driver.execute_script("arguments[0].click();", option_el)
                        self.log(f"    Successfully selected custom option '{value}'", "dim")
                    except:
                        if is_input:
                            element.send_keys("\n") # Fallback for search boxes
                            self.log(f"    No option found, hit Enter on input.", "dim")
                        else:
                            # Fallback: type and enter for regular buttons acting as search
                            element.send_keys(value)
                            time.sleep(0.5)
                            element.send_keys("\n")
                            self.log(f"    Failed to find custom option '{value}', sent keys as fallback.", "warning")
            elif act_type == "upload":
                if value == "RESUME_PATH":
                    element.send_keys(self.resume_path)
                else:
                    element.send_keys(value)
            
            time.sleep(0.5)
            return True
        except Exception as e:
            self.log(f"  Error executing {act_type} on {xpath}: {e}", "warning")
            return False

    def _parse_gemini_json(self, response):
        self.log(f"Raw Gemini Response: {response}", "dim")
        
        # Aggressively find the JSON block
        json_str = response
        if "```" in json_str:
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", json_str, re.DOTALL)
            if m:
                json_str = m.group(1)

        # Fallback: find first { and last }
        start = json_str.find("{")
        end = json_str.rfind("}")
        if start != -1 and end != -1:
            json_str = json_str[start:end+1]
        else:
            self.log("Could not find a JSON block in the response.", "warning")
            return None

        # Sometimes Gemini cuts off mid-generation (e.g. max tokens hit).
        # We can try to cleanly close the JSON string if it looks truncated.
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            # Let's try to fix truncated arrays/objects
            self.log(f"Initial JSON parse failed: {e}. Attempting auto-fix for truncated JSON...", "dim")
            try:
                # If it ended while building an action object
                if json_str.rfind("{") > json_str.rfind("}"):
                    # Strip back to the last complete action
                    last_complete = json_str.rfind("}")
                    json_str = json_str[:last_complete+1] + "\n  ]\n}"
                # If it ended while listing actions
                elif json_str.rstrip().endswith(","):
                    json_str = json_str.rstrip()[:-1] + "\n  ]\n}"
                
                return json.loads(json_str)
            except Exception as e2:
                self.log(f"Could not parse or auto-fix Gemini response as JSON: {e2}", "warning")
                self.log(f"Cleaned JSON String that failed:\n---\n{json_str}\n---", "dim")
                return None

    def _wait_for_page_load(self, timeout=25):
        """More robustly waits for dynamic pages like Workday."""
        try:
            # 1. Wait for document.readyState
            WebDriverWait(self.driver, timeout).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
            
            # 2. Specifically wait for common interactive elements or Workday-specific containers
            # This is key for dynamic apps that might be 'ready' but empty
            WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input, button, [role='button'], [role='textbox'], [data-automation-id]"))
            )
            
            # 3. Small extra buffer for JS to finish rendering/animations
            time.sleep(3)
        except Exception:
            # It's better to continue and try to analyze what *is* there than fail.
            pass

    def _click_apply_or_continue_if_present(self):
        """Fallback navigation if Gemini fails to provide a click action."""
        texts = ["apply", "continue", "next", "submit", "save"]
        for t in texts:
            try:
                xpath = f"//*[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{t.lower()}')]"
                btn = self.driver.find_element(By.XPATH, xpath)
                if btn.is_displayed():
                    btn.click()
                    self.log(f"Fallback clicked: '{btn.text}'", "dim")
                    return True
            except:
                continue
        return False

