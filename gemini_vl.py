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

# Disable SSL verification globally for corporate proxies / self-signed certs
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["SSL_CERT_FILE"] = ""

try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

# Patch httpx (used by google-genai) to skip SSL verification
try:
    import httpx
    _orig_init = httpx.Client.__init__
    def _patched_init(self, *a, **kw):
        kw.setdefault("verify", False)
        _orig_init(self, *a, **kw)
    httpx.Client.__init__ = _patched_init

    _orig_async_init = httpx.AsyncClient.__init__
    def _patched_async_init(self, *a, **kw):
        kw.setdefault("verify", False)
        _orig_async_init(self, *a, **kw)
    httpx.AsyncClient.__init__ = _patched_async_init
except Exception:
    pass

GEMINI_DEFAULT_MODEL = "gemini-2.0-flash"

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

    # Build client with SSL verification disabled via httpx transport
    try:
        import httpx
        transport = httpx.HTTPTransport(verify=False)
        http_client = httpx.Client(transport=transport, verify=False)
        client = genai.Client(api_key=api_key, http_client=http_client)
    except Exception:
        client = genai.Client(api_key=api_key)

    system = (system or "").strip()

    if hasattr(pil_image, "save"):
        img = pil_image
    else:
        from PIL import Image
        img = Image.open(io.BytesIO(pil_image)).convert("RGB")

    # Convert PIL image to bytes for the new API
    def _img_to_bytes(pil_img):
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return buf.getvalue()

    def _make_contents():
        parts = []

        if prior_screenshot_parts:
            from PIL import Image as PILImage
            for cap, part_img in prior_screenshot_parts:
                parts.append(str(cap))
                pimg = part_img if hasattr(part_img, "save") else PILImage.open(io.BytesIO(part_img)).convert("RGB")
                parts.append(types.Part.from_bytes(data=_img_to_bytes(pimg), mime_type="image/png"))
            parts.append("=== Current screenshot (most recent). Compare with above and decide next action. ===")

        parts.append(types.Part.from_bytes(data=_img_to_bytes(img), mime_type="image/png"))
        parts.append(user_text)
        return parts

    config = types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        system_instruction=system if system else None,
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
        ],
    )

    def _do_request():
        contents = []

        if conversation_messages:
            for m in conversation_messages:
                role = "user" if (m.get("role") or "user") == "user" else "model"
                content = m.get("content")
                if isinstance(content, list):
                    content = next((c.get("text", "") for c in content if isinstance(c, dict) and "text" in c), str(content))
                if content and str(content).strip():
                    contents.append(types.Content(role=role, parts=[types.Part.from_text(text=str(content))]))

        contents.append(types.Content(role="user", parts=[
            types.Part.from_bytes(data=_img_to_bytes(img), mime_type="image/png"),
            types.Part.from_text(text=user_text),
        ]))

        if prior_screenshot_parts:
            from PIL import Image as PILImage
            prior_parts = []
            for cap, part_img in prior_screenshot_parts:
                prior_parts.append(types.Part.from_text(text=str(cap)))
                pimg = part_img if hasattr(part_img, "save") else PILImage.open(io.BytesIO(part_img)).convert("RGB")
                prior_parts.append(types.Part.from_bytes(data=_img_to_bytes(pimg), mime_type="image/png"))
            prior_parts.append(types.Part.from_text(text="=== Current screenshot. ==="))
            prior_parts.append(types.Part.from_bytes(data=_img_to_bytes(img), mime_type="image/png"))
            prior_parts.append(types.Part.from_text(text=user_text))
            contents[-1] = types.Content(role="user", parts=prior_parts)

        return client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(_do_request)
            try:
                response = future.result(timeout=REQUEST_TIMEOUT_SEC)
            except FuturesTimeoutError:
                raise RuntimeError(f"Request timed out after {REQUEST_TIMEOUT_SEC}s")
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
                time.sleep(wait)
                continue
            raise RuntimeError(f"Gemini API error: {e}") from e

        text = None
        try:
            text = response.text
        except (ValueError, AttributeError, IndexError):
            pass
        if not text:
            try:
                if response.candidates and response.candidates[0].content.parts:
                    text = response.candidates[0].content.parts[0].text
            except (AttributeError, IndexError):
                pass

        if text:
            return text.strip()

        if attempt < MAX_RETRIES - 1:
            wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
            time.sleep(wait)
            continue

        reason = ""
        try:
            reason = str(response.prompt_feedback) if hasattr(response, "prompt_feedback") else ""
        except Exception:
            pass
        raise RuntimeError(f"Gemini returned no text (safety filter or empty response). {reason}".strip())
