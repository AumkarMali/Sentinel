"""
Core job application automation engine.

Opens Chrome via Selenium, scrapes internship listings from the SimplifyJobs
GitHub repo, and attempts to auto-apply using Gemini vision for form analysis.
"""
import io
import os
import re
import json
import time
from urllib.parse import urlparse

from PIL import Image
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

try:
    from webdriver_manager.chrome import ChromeDriverManager
    _WDM_AVAILABLE = True
except ImportError:
    _WDM_AVAILABLE = False

LISTINGS_URL = "https://github.com/SimplifyJobs/Summer2026-Internships"


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

    def _scrape_jobs(self):
        """Parse the GitHub README table for job listings with apply links."""
        self.log("Scraping job listings...", "action")
        jobs = []

        try:
            readme = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "article.markdown-body"))
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

                # Check if this row is closed (lock emoji)
                row_text = row.text
                if "\U0001f512" in row_text or "🔒" in row_text:
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
                    self.log(f"  Applied successfully!", "success")
                elif result == "skipped":
                    skipped += 1
                    self.stats_fn("skipped", skipped)
                    self.log(f"  Skipped (login required or unsupported form).", "warning")
                else:
                    failed += 1
                    self.stats_fn("failed", failed)
                    self.log(f"  Could not complete application.", "error")
            except Exception as e:
                failed += 1
                self.stats_fn("failed", failed)
                self.log(f"  Error: {e}", "error")

            time.sleep(2)

        self.log(f"\nDone! Applied: {applied}, Skipped: {skipped}, Failed: {failed}", "header")

    def _apply_to_single_job(self, job):
        """Open apply link and attempt to fill the application form."""
        url = job["url"]

        # Open in a new tab
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

    def _take_screenshot(self):
        """Capture current page as a PIL Image."""
        png = self.driver.get_screenshot_as_png()
        return Image.open(io.BytesIO(png)).convert("RGB")

    def _analyze_page(self):
        """Use Gemini vision to analyze the current page."""
        self.log("  Analyzing page with Gemini...", "action")
        screenshot = self._take_screenshot()

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

            response = re.sub(
                r"<thinking\s*>.*?</thinking\s*>", "", response,
                flags=re.DOTALL | re.IGNORECASE,
            ).strip()

            if "```" in response:
                m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
                if m:
                    response = m.group(1)

            data = json.loads(response)
            return data
        except json.JSONDecodeError:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            self.log("  Could not parse Gemini response as JSON.", "warning")
            return None
        except Exception as e:
            self.log(f"  Gemini analysis error: {e}", "error")
            return None

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

        name_keywords = ["name", "full name", "your name", "applicant name"]
        first_name_kw = ["first name", "first", "given name"]
        last_name_kw = ["last name", "last", "surname", "family name"]
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

    def _fill_field(self, label, value, field_type):
        """Find a form field by label text and fill it."""
        if not value:
            return False

        # Try finding input/textarea by various strategies
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

        return False

    def _find_by_label_text(self, label):
        """Find input associated with a <label> element."""
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
        """Find input by placeholder text."""
        inputs = self.driver.find_elements(
            By.CSS_SELECTOR, "input[placeholder], textarea[placeholder]"
        )
        for inp in inputs:
            ph = (inp.get_attribute("placeholder") or "").lower()
            if label.lower() in ph or any(w in ph for w in label.lower().split()):
                return inp
        return None

    def _find_by_aria_label(self, label):
        """Find input by aria-label."""
        inputs = self.driver.find_elements(
            By.CSS_SELECTOR, "input[aria-label], textarea[aria-label]"
        )
        for inp in inputs:
            al = (inp.get_attribute("aria-label") or "").lower()
            if label.lower() in al:
                return inp
        return None

    def _find_by_nearby_text(self, label):
        """Find input near text that matches the label (XPath-based)."""
        try:
            xpath = (
                f"//*[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', "
                f"'abcdefghijklmnopqrstuvwxyz'), '{label.lower()}')]"
                f"/following::input[1] | "
                f"//*[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', "
                f"'abcdefghijklmnopqrstuvwxyz'), '{label.lower()}')]"
                f"/following::textarea[1]"
            )
            elements = self.driver.find_elements(By.XPATH, xpath)
            return elements[0] if elements else None
        except Exception:
            return None

    def _fill_select(self, element, value):
        """Fill a <select> dropdown."""
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

    def _upload_resume_to_field(self, label):
        """Find file input and upload the resume PDF."""
        file_inputs = self.driver.find_elements(By.CSS_SELECTOR, "input[type='file']")
        for inp in file_inputs:
            try:
                accept = (inp.get_attribute("accept") or "").lower()
                if not accept or "pdf" in accept or "document" in accept or "*" in accept:
                    inp.send_keys(self.resume_path)
                    self.log("  Uploaded resume PDF.", "info")
                    return True
            except Exception:
                continue
        return False

    def _click_element_by_analysis(self, button_info):
        """Click a button described by Gemini's analysis."""
        text = (button_info.get("text") or "").strip()
        if not text:
            return False

        # Try finding by button text
        strategies = [
            lambda: self.driver.find_element(
                By.XPATH,
                f"//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', "
                f"'abcdefghijklmnopqrstuvwxyz'), '{text.lower()}')]"
            ),
            lambda: self.driver.find_element(
                By.XPATH,
                f"//a[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', "
                f"'abcdefghijklmnopqrstuvwxyz'), '{text.lower()}')]"
            ),
            lambda: self.driver.find_element(
                By.XPATH,
                f"//input[@type='submit' and contains(translate(@value, "
                f"'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), "
                f"'{text.lower()}')]"
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

        return False
