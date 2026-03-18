"""
Vision chat via Google Gemini API (cloud).
Install: pip install google-genai
API key: https://aistudio.google.com/apikey
Best vision models: gemini-2.0-flash, gemini-1.5-flash, gemini-1.5-pro
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
    # Patch httpx if available
    try:
        import httpx
        httpx.AsyncClient.__init__ = lambda self, *a, **kw: _orig_async_init(self, *a, **kw, verify=False)
        httpx.Client.__init__ = lambda self, *a, **kw: _orig_init(self, *a, **kw, verify=False)
    except (ImportError, NameError):
        pass

# Call it on module import
_disable_ssl_verification()

GEMINI_DEFAULT_MODEL = "gemini-1.5-flash"

MAX_RETRIES = 3
RETRY_BACKOFF = [1, 3, 6]

REQUEST_TIMEOUT_SEC = 30


def call_gemini(
    system: str,
    user_text: str,
    pil_image,
    conversation_messages: list = None,
    max_tokens: int = 4096,
    model: str = None,
    api_key: str = None,
    prior_screenshot_parts: list = None,
) -> str:
    """
    Send system + user text + image to Gemini. Returns assistant text.
    conversation_messages: optional list of {"role": "user"|"assistant", "content": str}.
    prior_screenshot_parts: optional list of (caption_str, pil_image) for previous turns.
    """
    from google import genai
    from google.genai import types

    model = (model or GEMINI_DEFAULT_MODEL).strip()
    api_key = (api_key or "").strip() or os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise ValueError("Gemini API key required. Set in Settings or GEMINI_API_KEY env.")

    genai.configure(api_key=api_key)

    generation_config = genai.types.GenerationConfig(
        max_output_tokens=max_tokens,
    )
    
    safety_settings = {
        key: val for key, val in {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        }.items()
    }

    model_instance = genai.GenerativeModel(
        model_name=model,
        generation_config=generation_config,
        system_instruction=system,
        safety_settings=safety_settings
    )

    # Handle single or multiple images
    content_parts = [user_text]
    if isinstance(pil_image, list):
        # This is a list of PIL images for scrolling analysis
        content_parts.extend(pil_image)
    else:
        # Standard single image
        content_parts.append(pil_image)

    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            # The genai library now handles timeouts internally, but we use a thread for safety
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(model_instance.generate_content, content_parts)
            response = future.result(timeout=REQUEST_TIMEOUT_SEC)
            
            if response.text:
                return response.text.strip()
            
            # Handle cases where response might be empty but not an error
            reason = str(response.prompt_feedback) if hasattr(response, "prompt_feedback") else "No text returned."
            last_err = RuntimeError(f"Gemini returned no text (safety filter or empty response). {reason}".strip())
            time.sleep(RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)])
            continue

        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
                time.sleep(wait)
                continue
    
    raise RuntimeError(f"Gemini API error after {MAX_RETRIES} retries: {last_err}") from last_err
