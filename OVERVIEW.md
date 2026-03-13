# Project Overview

## What It Does

**Sentinel** is a Chrome-based job application automator. It scrapes internship listings from the [SimplifyJobs Summer 2026 Internships](https://github.com/SimplifyJobs/Summer2026-Internships) GitHub repository and auto-applies to open positions using your resume.

## Architecture

| Layer | Tech |
|-------|------|
| **GUI** | Tkinter — Start button, resume upload, log |
| **Browser** | Selenium WebDriver (Chrome) |
| **Vision & reasoning** | Gemini API — analyzes application pages, identifies form fields |
| **Resume parsing** | pdfplumber for text extraction, Gemini for structured field extraction |

## Flow

1. User uploads resume PDF and clicks Start
2. Chrome opens → navigates to GitHub internship listings
3. Scrapes the README table for companies, roles, and apply links
4. For each open position:
   - Opens the apply link in a new tab
   - Screenshots the page → sends to Gemini for form analysis
   - Maps form fields to resume data (name, email, phone, education, etc.)
   - Fills fields via Selenium and uploads resume PDF
   - Logs result (applied / skipped / failed)
5. Reports summary when done

## Key Design Decisions

- **Selenium over pyautogui**: Direct DOM access is far more reliable for web forms than screenshot-based clicking
- **Gemini for page analysis**: ATS systems (Workday, Greenhouse, Lever, etc.) have wildly different form layouts; vision-based analysis adapts to any format
- **Hybrid resume parsing**: Regex for reliable fields (email, phone), Gemini for context-dependent fields (name, education, skills)
