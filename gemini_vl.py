"""
Vision chat via Google Gemini API (cloud).
Install: pip install google-genai
API key: https://aistudio.google.com/apikey
Best vision models: gemini-2.5-flash, gemini-2.5-pro
"""
import io
import os
import ssl
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

def _disable_ssl_verification():
    """Globally disables SSL cert verification to handle corporate proxies."""
    os.environ["PYTHONHTTPSVERIFY"] = "0"
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["REQUESTS_CA_BUNDLE"] = ""
    os.environ["SSL_CERT_FILE"] = ""
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
    except AttributeError:
        pass

# Call it on module import
_disable_ssl_verification()

GEMINI_DEFAULT_MODEL = "gemini-2.5-flash"

MAX_RETRIES = 3
RETRY_BACKOFF = [1, 3, 6]

REQUEST_TIMEOUT_SEC = 30


def call_gemini(
    system: str,
    user_text: str,
    pil_image=None,
    conversation_messages: list = None,
    max_tokens: int = 4096,
    model: str = None,
    api_key: str = None,
) -> str:
    """
    Send system + user text + optional image to Gemini. Returns assistant text.
    conversation_messages: optional list of {"role": "user"|"assistant", "content": str}.
    """
    from google import genai
    from google.genai import types

    model = (model or GEMINI_DEFAULT_MODEL).strip()
    api_key = (api_key or "").strip() or os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise ValueError("Gemini API key required. Set in Settings or GEMINI_API_KEY env.")

    client = genai.Client(api_key=api_key)

    generation_config = types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        system_instruction=system,
    )

    content_parts = [user_text]
    if pil_image:
        if isinstance(pil_image, list):
            content_parts.extend(pil_image)
        else:
            content_parts.append(pil_image)

    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=model,
                contents=content_parts,
                config=generation_config,
            )
            
            if response.text:
                return response.text.strip()
            
            # Handle cases where response might be empty but not an error
            last_err = RuntimeError(f"Gemini returned no text (safety filter or empty response). {response}".strip())
            time.sleep(RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)])
            continue

        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
                time.sleep(wait)
                continue
    
    raise RuntimeError(f"Gemini API error after {MAX_RETRIES} retries: {last_err}") from last_err
