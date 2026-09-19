# Full-stack integration test: posts the sample workflow (API format) to a running
# ComfyUI server, waits for execution, and checks the prompt output.
# Usage: python_embeded\python.exe test\test_workflow_api.py [--comfy http://127.0.0.1:8188] [--clip 1]
import argparse
import json
import time
import urllib.request

API_FORMAT = {
    "1": {"class_type": "LoadVideo", "inputs": {"file": ""}},
    "2": {"class_type": "H3VLMModelLoader", "inputs": {
        "model": "Qwen 2.5 VL 7B Instruct",
        "custom_model_id": "",
        "memory_mode": "ComfyUI managed (BF16)",
        "attention_mode": "Auto (SDPA)"}},
    "3": {"class_type": "MiniMaxH3VideoToPrompt", "inputs": {
        "video": ["1", 0], "vlm": ["2", 0], "mode": "T2VA", "duration": 0.0,
        "max_seconds": 0.0,
        "keep_fade": False, "frame_fps": 1.0, "max_frames": 24, "frame_max_side": 512,
        "use_asr": True, "asr_model": "small", "use_audio_tags": True,
        "tags_threshold": 0.05, "tags_max": 12, "max_new_tokens": 8192,
        "unload_vlm": True}},
    "4": {"class_type": "PreviewAny", "inputs": {"source": ["3", 0]}},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", default="http://127.0.0.1:8188")
    ap.add_argument("--clip", default="16")
    ap.add_argument("--mode", default="T2VA")
    args = ap.parse_args()

    prompt = json.loads(json.dumps(API_FORMAT))
    prompt["1"]["inputs"]["file"] = f"Sample Media Clip {args.clip}.mp4"
    prompt["3"]["inputs"]["mode"] = args.mode

    req = urllib.request.Request(
        f"{args.comfy}/prompt",
        data=json.dumps({"prompt": prompt}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        resp = json.loads(r.read())
    pid = resp["prompt_id"]
    print(f"queued {pid}, clip {args.clip}")

    for _ in range(600):
        time.sleep(2)
        with urllib.request.urlopen(f"{args.comfy}/history/{pid}", timeout=30) as r:
            hist = json.loads(r.read())
        if pid not in hist:
            continue
        entry = hist[pid]
        status = entry.get("status", {})
        if status.get("completed"):
            text = entry["outputs"]["4"]["text"][0]
            if args.mode == "LTX":
                ok = len(text.split()) >= 20 and "integrated_multimodal_description" not in text
            else:
                ok = all(f in text for f in ("integrated_multimodal_description",
                                             "overall_soundscape", "non_diegetic_music"))
            print(("PASS" if ok else "FAIL") + f" ({len(text)} chars)")
            print("---")
            print(text)
            return
        if status.get("status_str") == "error":
            print("FAIL: execution error")
            for det in status.get("messages", []):
                if det[0] == "execution_error":
                    print(json.dumps(det[1], indent=2)[:3000])
            return
    print("FAIL: timeout")


if __name__ == "__main__":
    main()
