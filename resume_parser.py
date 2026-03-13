"""
Resume PDF parser.

Extracts text from a PDF resume and uses Gemini to parse structured fields
(name, email, phone, education, skills, etc.).
"""
import io
import os
import re
import json

import pdfplumber


def _extract_text(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


def _extract_email(text: str) -> str:
    match = re.search(r'[\w.+-]+@[\w-]+\.[\w.-]+', text)
    return match.group(0) if match else ""


def _extract_phone(text: str) -> str:
    match = re.search(
        r'(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}', text
    )
    return match.group(0).strip() if match else ""


def _extract_linkedin(text: str) -> str:
    match = re.search(r'(?:https?://)?(?:www\.)?linkedin\.com/in/[\w-]+/?', text, re.I)
    return match.group(0) if match else ""


def _extract_github(text: str) -> str:
    match = re.search(r'(?:https?://)?(?:www\.)?github\.com/[\w-]+/?', text, re.I)
    return match.group(0) if match else ""


def parse_resume(pdf_path: str, api_key: str = "", model: str = "") -> dict:
    """
    Parse a resume PDF into structured data.

    First extracts text, then tries Gemini for rich parsing.
    Falls back to regex extraction if Gemini is unavailable.
    """
    raw_text = _extract_text(pdf_path)
    if not raw_text:
        return {"raw_text": "", "name": "", "email": "", "phone": ""}

    base = {
        "raw_text": raw_text,
        "email": _extract_email(raw_text),
        "phone": _extract_phone(raw_text),
        "linkedin": _extract_linkedin(raw_text),
        "github": _extract_github(raw_text),
    }

    if api_key:
        gemini_data = _parse_with_gemini(raw_text, api_key, model)
        if gemini_data:
            for k, v in gemini_data.items():
                if v and k != "raw_text":
                    base[k] = v

    if not base.get("name"):
        lines = raw_text.strip().split("\n")
        base["name"] = lines[0].strip() if lines else ""

    return base


def _parse_with_gemini(text: str, api_key: str, model: str) -> dict:
    """Use Gemini to extract structured resume data from text."""
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        prompt = f"""Extract structured information from this resume text. Return ONLY a JSON object.

Resume text:
---
{text[:4000]}
---

Return this exact JSON structure (fill in what you can find, leave empty string for missing):
{{
    "name": "Full Name",
    "email": "email@example.com",
    "phone": "phone number",
    "linkedin": "linkedin URL",
    "github": "github URL",
    "website": "personal website URL",
    "location": "City, State",
    "university": "University Name",
    "degree": "Degree and Major",
    "gpa": "GPA value",
    "graduation_date": "Month Year",
    "skills": "comma-separated list of skills",
    "work_authorization": "yes/no/unknown"
}}"""

        config = types.GenerateContentConfig(
            max_output_tokens=1024,
            system_instruction="Extract resume information. Return only valid JSON.",
        )

        response = client.models.generate_content(
            model=model or "gemini-2.0-flash",
            contents=[prompt],
            config=config,
        )

        resp_text = ""
        try:
            resp_text = response.text
        except (ValueError, AttributeError):
            if response.candidates and response.candidates[0].content.parts:
                resp_text = response.candidates[0].content.parts[0].text

        if not resp_text:
            return {}

        resp_text = resp_text.strip()
        if "```" in resp_text:
            m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', resp_text, re.DOTALL)
            if m:
                resp_text = m.group(1)

        return json.loads(resp_text)

    except json.JSONDecodeError:
        json_match = re.search(r'\{.*\}', resp_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return {}
    except Exception:
        return {}
