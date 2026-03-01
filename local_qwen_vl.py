"""
Local inference with Qwen3-VL-8B-Instruct (Hugging Face).
Free, no API key. Uses NVIDIA GPU when available (faster); else CPU (~16GB+ RAM).
For GPU: install PyTorch with CUDA, e.g. pip install torch --index-url https://download.pytorch.org/whl/cu121
https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
"""
import os
import tempfile
import threading

_model = None
_processor = None
_load_lock = threading.Lock()
_LOADING = False
_load_done = threading.Event()  # set when load_model() has finished (success or not)
_device_name = None  # "cuda" or "cpu" after first load


def _get_device_and_dtype():
    """Return (use_cuda, dtype) for loading. Prefer GPU + bfloat16 when supported."""
    import torch
    if not torch.cuda.is_available():
        return False, torch.float32
    # Prefer bfloat16 on GPU (Ampere+); fallback to float16 for older GPUs
    try:
        # bfloat16 is faster and stable on modern NVIDIA GPUs
        torch.zeros(1, device="cuda", dtype=torch.bfloat16)
        return True, torch.bfloat16
    except (RuntimeError, TypeError):
        return True, torch.float16


def load_model():
    """Load model and processor once. Call from background thread on first use. Uses NVIDIA GPU when available."""
    global _model, _processor, _LOADING, _load_done
    with _load_lock:
        if _model is not None and _processor is not None:
            _load_done.set()
            return True
        if _LOADING:
            return False
        _LOADING = True
        _load_done.clear()
    try:
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        import torch
        use_cuda, dtype = _get_device_and_dtype()
        if use_cuda:
            # Faster matmuls on Ampere+ (RTX 30xx, 40xx, A100, etc.)
            if hasattr(torch.backends.cuda, "matmul") and hasattr(torch.backends.cuda.matmul, "allow_tf32"):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = True
        with _load_lock:
            if _model is not None:
                _LOADING = False
                return True
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct",
            torch_dtype=dtype,
            device_map="auto" if use_cuda else None,
        )
        if not use_cuda:
            model = model.to("cpu")
        processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
        global _device_name
        _device_name = "cuda" if use_cuda else "cpu"
        with _load_lock:
            _model = model
            _processor = processor
            _LOADING = False
        _load_done.set()
        return True
    except Exception:
        with _load_lock:
            _LOADING = False
        _load_done.set()
        raise


def get_device_name():
    """Return 'cuda' or 'cpu' depending on where the model is (or will be) loaded. None if not loaded yet."""
    return _device_name


def _content_as_parts(content):
    """Ensure content is a list of parts (processor expects list, not raw string)."""
    if isinstance(content, list):
        return content
    text = str(content) if content else ""
    return [{"type": "text", "text": text}] if text else []


def call_local_qwen(system: str, user_text: str, pil_image, conversation_messages: list = None, max_tokens: int = 4096) -> str:
    """
    Run inference with Qwen3-VL-8B-Instruct. Returns response text.
    conversation_messages: list of {"role": "user"|"assistant", "content": str} for prior turns (text only).
    """
    if _model is None or _processor is None:
        load_model()
        if _model is None or _processor is None:
            # Background load may still be running; wait up to 5 minutes
            _load_done.wait(timeout=300)
    if _model is None or _processor is None:
        raise RuntimeError("Local Qwen3-VL model failed to load. Check logs and try again.")
    import torch
    # Build messages: processor expects each message content to be a list of parts (not a string)
    messages = [{"role": "system", "content": _content_as_parts(system)}]
    if conversation_messages:
        for m in conversation_messages:
            role = m.get("role")
            content = m.get("content")
            if isinstance(content, list):
                text_parts = [c.get("text", "") for c in content if isinstance(c, dict) and "text" in c]
                content = "\n".join(text_parts) if text_parts else str(content)
            if role and content is not None:
                messages.append({"role": role, "content": _content_as_parts(content)})
    # Current user message with image and text (processor often expects URL/path; save PIL to temp file)
    tmp_path = None
    try:
        if hasattr(pil_image, "save"):
            fd, tmp_path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            pil_image.save(tmp_path)
            image_input = tmp_path
        else:
            image_input = pil_image
        content = [
            {"type": "image", "image": image_input},
            {"type": "text", "text": user_text},
        ]
        messages.append({"role": "user", "content": content})
        # Apply chat template and generate
        inputs = _processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        # Processor may return BatchEncoding (UserDict), not a plain dict
        if inputs is None:
            raise TypeError("apply_chat_template returned None")
        if not isinstance(inputs, dict):
            inputs = dict(inputs)
        # Move to model device; remove token_type_ids if present (not used by generate)
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]
        dev = next(_model.parameters()).device
        inputs = {k: v.to(dev) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.inference_mode():
            generated = _model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=_processor.tokenizer.pad_token_id or _processor.tokenizer.eos_token_id,
            )
        # Decode only the new tokens
        input_len = inputs["input_ids"].shape[1]
        output_ids = generated[0][input_len:]
        output_text = _processor.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return output_text.strip()
    finally:
        if tmp_path and os.path.isfile(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
