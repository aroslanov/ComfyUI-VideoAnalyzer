# Direct e2e test of the pipeline used by MiniMaxH3VideoToPrompt.
# Usage (embedded python):
#   python_embeded\python.exe test\test_e2e.py [--mode T2VA] [--videos 1,2,3] [--max-seconds 30]
# Requires llama-server (or any OpenAI-compatible VLM) running; override with --api-base.
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import H3PipelineError, video_to_prompt  # noqa: E402

DEFAULT_VIDEOS = Path(r"C:\Users\Public\Documents\Adobe\Premiere Pro\26.0\Sample Media")
OUT_DIR = Path(__file__).resolve().parents[1] / "out"
H3_FIELDS = ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music")


def check_prompt(mode: str, prompt: str) -> list:
    errs = []
    if not prompt or not prompt.strip():
        return ["empty prompt"]
    if mode == "LTX":
        if len(prompt.split()) < 20:
            errs.append(f"too short for LTX ({len(prompt.split())} words)")
        return errs
    positions = []
    for f in H3_FIELDS:
        pos = prompt.find(f + ":")
        if pos < 0:
            errs.append(f"missing field '{f}:'")
        else:
            positions.append(pos)
    if len(positions) == 3 and positions != sorted(positions):
        errs.append(f"fields out of order: {positions}")
    if mode in ("I2VA", "FL2VA", "L2VA") and not prompt.startswith(("For the target video", "How the reference pictures align")):
        errs.append("missing alignment line prefix")
    return errs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="T2VA")
    ap.add_argument("--api-base", default="http://127.0.0.1:8777/v1")
    ap.add_argument("--model", default="qwen2.5-vl-3b-instruct")
    ap.add_argument("--videos", default="", help="comma separated clip numbers, e.g. 1,16")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--analysis-input", default="frames", choices=["frames", "video"])
    ap.add_argument("--no-asr", action="store_true")
    ap.add_argument("--no-tags", action="store_true")
    args = ap.parse_args()

    if args.videos:
        clips = [DEFAULT_VIDEOS / f"Sample Media Clip {n.strip()}.mp4" for n in args.videos.split(",")]
    else:
        clips = sorted(DEFAULT_VIDEOS.glob("Sample Media Clip *.mp4"))
    clips = [c for c in clips if c.is_file()]
    assert clips, "no test videos found"

    failures = 0
    for clip in clips:
        name = clip.stem
        t0 = time.time()
        try:
            prompt = video_to_prompt(
                str(clip), OUT_DIR / name,
                api_base=args.api_base, model=args.model,
                mode=args.mode, max_seconds=args.max_seconds,
                analysis_input=args.analysis_input,
                use_asr=not args.no_asr, use_audio_tags=not args.no_tags,
            )
        except H3PipelineError as e:
            print(f"FAIL {name}: {e}", flush=True)
            failures += 1
            continue
        errs = check_prompt(args.mode, prompt)
        status = "PASS" if not errs else "FAIL"
        if errs:
            failures += 1
        print(f"{status} {name} ({time.time() - t0:.0f}s, {len(prompt)} chars)"
              + (f" errs={errs}" if errs else ""), flush=True)
    print(f"\n{len(clips) - failures}/{len(clips)} passed, mode={args.mode}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
