# Sentinel — Chrome Job Automator

Auto-apply to internship positions from [SimplifyJobs/Summer2026-Internships](https://github.com/SimplifyJobs/Summer2026-Internships).

Upload your resume, click Start, and Sentinel will:
1. Open Chrome and navigate to the internship listings
2. Scrape open positions with apply links
3. Visit each application page
4. Use Gemini vision to analyze forms and fill them with your resume data
5. Upload your resume PDF where file inputs are available

## Setup

```bash
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

## Gemini API Key

Get a free key at [Google AI Studio](https://aistudio.google.com/apikey). Enter it in **Settings** inside the app, or set the `GEMINI_API_KEY` environment variable.

## Run

```bash
python gui.py
```

1. Click **Browse** to select your resume PDF
2. Click **Settings** to enter your Gemini API key (first time only)
3. Click **Start Applying**

## How It Works

| Component | Tech |
|-----------|------|
| **GUI** | Tkinter |
| **Browser automation** | Selenium + ChromeDriver |
| **Page analysis** | Gemini vision API |
| **Resume parsing** | pdfplumber + Gemini text extraction |

## Files

| File | Purpose |
|------|---------|
| `gui.py` | Main application GUI |
| `job_automator.py` | Selenium automation engine |
| `resume_parser.py` | PDF resume parser |
| `gemini_vl.py` | Gemini API client |
| `config.json` | Stored settings (API key, model, resume path) |
