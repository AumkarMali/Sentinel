"""
Test script for local Qwen3-VL inference.
Run from project root: python test_qwen_local.py
Prints timing for load + one inference so you can see where it hangs.
"""
import sys
import time

# Small test image (100x100 red square) so we don't need a real screenshot
def make_test_image():
    from PIL import Image
    w, h = 100, 100
    img = Image.new("RGB", (w, h), color=(200, 50, 50))
    return img


def main():
    print("=== Test: local Qwen3-VL ===\n")

    # 1) Load model
    print("[1] Loading model...")
    t0 = time.perf_counter()
    try:
        from local_qwen_vl import load_model
        load_model()
    except Exception as e:
        print(f"Load failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    load_time = time.perf_counter() - t0
    print(f"    Loaded in {load_time:.1f}s\n")

    # 2) One inference with a tiny image and short prompt
    print("[2] Running one inference (small image, short prompt)...")
    img = make_test_image()
    system = "You are a helpful assistant. Reply in one short sentence."
    user_text = "What color is this image? One word only."
    t0 = time.perf_counter()
    try:
        from local_qwen_vl import call_local_qwen
        out = call_local_qwen(
            system=system,
            user_text=user_text,
            pil_image=img,
            conversation_messages=None,
            max_tokens=64,
        )
    except Exception as e:
        print(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    inf_time = time.perf_counter() - t0
    print(f"    Inference took {inf_time:.1f}s")
    print(f"    Response: {out!r}\n")

    print("=== Done. If you see this, local Qwen is working. ===")


if __name__ == "__main__":
    main()
