"""
Vision chat via Google Gemini API (cloud).
Install: pip install google-generativeai
API key: https://aistudio.google.com/apikey
Best vision models: gemini-2.0-flash, gemini-1.5-flash, gemini-1.5-pro
"""
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# Best balance of speed and quality for vision/screen understanding
GEMINI_DEFAULT_MODEL = "gemini-2.0-flash"

# Retry on rate-limit / empty-response with exponential backoff
MAX_RETRIES = 3
RETRY_BACKOFF = [1, 3, 6]  # seconds

# Stop waiting after this many seconds so we never hang (e.g. when API stalls)
REQUEST_TIMEOUT_SEC = 10


def _is_rate_limit(err):
    msg = (getattr(err, "message", None) or str(err)).lower()
    return "429" in str(getattr(err, "code", "")) or "resource exhausted" in msg or "rate" in msg or "quota" in msg


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
    conversation_messages: optional list of {"role": "user"|"assistant", "content": str} for prior turns (text only).
    prior_screenshot_parts: optional list of (caption_str, pil_image) for previous turns so the model can compare.
    """
    import google.generativeai as genai

    model = (model or GEMINI_DEFAULT_MODEL).strip()
    api_key = (api_key or "").strip() or os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise ValueError("Gemini API key required. Set in Settings or GEMINI_API_KEY env.")

    genai.configure(api_key=api_key)
    # Use Gemini's system_instruction so the model treats instructions (and user feedback) as rules to follow, not just context.
    system = (system or "").strip()
    user_only = user_text  # content we send as the user message (no system mixed in)
    # Relax safety so screenshots aren't blocked
    safety = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    if system:
        gemini = genai.GenerativeModel(model, system_instruction=system, safety_settings=safety)
    else:
        gemini = genai.GenerativeModel(model, safety_settings=safety)

    # Build user content: image + text (PIL accepted by SDK)
    if hasattr(pil_image, "save"):
        img = pil_image
    else:
        from PIL import Image
        img = Image.open(io.BytesIO(pil_image)).convert("RGB")

    gen_cfg = genai.types.GenerationConfig(max_output_tokens=max_tokens)

    def _do_request():
        if prior_screenshot_parts:
            from PIL import Image as PILImage
            content_parts = []
            for cap, part_img in prior_screenshot_parts:
                content_parts.append(str(cap))
                pimg = part_img if hasattr(part_img, "save") else PILImage.open(io.BytesIO(part_img)).convert("RGB")
                content_parts.append(pimg)
            content_parts.append("=== Current screenshot (most recent). Compare with above and decide next action. ===")
            content_parts.append(img)
            content_parts.append(user_only)
            return gemini.generate_content(content_parts, generation_config=gen_cfg)
        elif conversation_messages:
            history = []
            for m in conversation_messages:
                role = "user" if (m.get("role") or "user") == "user" else "model"
                content = m.get("content")
                if isinstance(content, list):
                    content = next((c.get("text", "") for c in content if isinstance(c, dict) and "text" in c), str(content))
                if content is not None and str(content).strip():
                    history.append({"role": role, "parts": [str(content)]})
            history.append({"role": "user", "parts": [img, user_only]})
            chat = gemini.start_chat(history=history[:-1])
            return chat.send_message([img, user_only], generation_config=gen_cfg)
        else:
            return gemini.generate_content([img, user_only], generation_config=gen_cfg)

    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(_do_request)
            try:
                response = future.result(timeout=REQUEST_TIMEOUT_SEC)
            except FuturesTimeoutError:
                raise RuntimeError(f"Request timed out after {REQUEST_TIMEOUT_SEC}s — retrying")
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
                time.sleep(wait)
                continue
            raise RuntimeError(f"Gemini API error (429 rate limit? try again in a minute): {e}") from e

        # ── Extract text from response ──
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

        # No text returned — often transient (rate limit, safety hold-off).
        # Retry before giving up.
        reason = ""
        try:
            if hasattr(response, "prompt_feedback"):
                reason = str(response.prompt_feedback)
        except Exception:
            pass
        if attempt < MAX_RETRIES - 1:
            wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
            time.sleep(wait)
            continue
        raise RuntimeError(
            f"Gemini returned no text (rate limit, safety filter, or empty response). {reason}".strip()
        )
