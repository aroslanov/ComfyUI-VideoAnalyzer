# Comprehensive pre-release test matrix (20+ cases): modes, options, parameters,
# synthetic edge-case videos, negative/validation cases.
# Run: python_embeded\python.exe test\test_matrix.py [--quick]
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(os.environ.get("COMFYUI_ROOT", "F:/ComfyUI/ComfyUI"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(COMFY_ROOT / "custom_nodes"))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import H3PipelineError, video_to_prompt  # noqa: E402
from test_e2e import DEFAULT_VIDEOS, build_vlm, check_prompt  # noqa: E402

SYNTH_DIR = ROOT / "out" / "synthetic"
H3_FIELDS = ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music")


def synth(name: str, args: list) -> Path:
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    out = SYNTH_DIR / name
    src = str(DEFAULT_VIDEOS / "Sample Media Clip 16.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, *args, str(out)], check=True)
    return out


def build_synthetic_videos() -> dict:
    d = {}
    d["no_audio"] = synth("no_audio.mp4", ["-an", "-t", "6"])
    d["vertical"] = synth("vertical.mp4", ["-t", "6", "-vf", "crop=480:854"])
    d["fades"] = synth("fades.mp4", ["-t", "8",
                                     "-vf", "fade=t=in:st=0:d=1.5,fade=t=out:st=6.5:d=1.5"])
    d["one_second"] = synth("one_second.mp4", ["-t", "1"])
    d["hd_ready"] = synth("hd_ready.mp4", ["-t", "6", "-vf", "scale=1280:720"])
    return d


CASES = [
    # --- modes ---
    ("T2VA_basic_16", dict(video="Sample Media Clip 16.mp4", mode="T2VA"),
     lambda p, c: check_prompt("T2VA", p)),
    ("T2VA_multishot_7", dict(video="Sample Media Clip 7.mp4", mode="T2VA"),
     lambda p, c: check_prompt("T2VA", p)),
    ("I2VA_clip1", dict(video="Sample Media Clip 1.mp4", mode="I2VA"),
     lambda p, c: check_prompt("I2VA", p)),
    ("FL2VA_clip16", dict(video="Sample Media Clip 16.mp4", mode="FL2VA"),
     lambda p, c: check_prompt("FL2VA", p)),
    ("L2VA_clip8", dict(video="Sample Media Clip 8.mp4", mode="L2VA"),
     lambda p, c: check_prompt("L2VA", p)),
    ("LTX_clip2", dict(video="Sample Media Clip 2.mp4", mode="LTX"),
     lambda p, c: check_prompt("LTX", p)),
    ("LTX_multishot_11", dict(video="Sample Media Clip 11.mp4", mode="LTX"),
     lambda p, c: check_prompt("LTX", p)),
    # --- duration handling ---
    ("duration_explicit_8", dict(video="Sample Media Clip 16.mp4", mode="T2VA", duration=8.0),
     lambda p, c: [] if p else ["empty"]),
    ("LTX_duration_snap_7", dict(video="Sample Media Clip 16.mp4", mode="LTX", duration=7.0),
     lambda p, c: check_prompt("LTX", p)),
    ("H3_duration_too_low", dict(video="Sample Media Clip 16.mp4", mode="T2VA", duration=2.0),
     None),
    ("LTX_duration_too_high", dict(video="Sample Media Clip 16.mp4", mode="LTX", duration=25.0),
     None),
    # --- sampling parameters ---
    ("frame_fps_2", dict(video="Sample Media Clip 16.mp4", frame_fps=2.0),
     lambda p, c: [] if len(list((c / "frames").glob("frame_*.jpg"))) == 10 else ["expected 10 frames"]),
    ("max_frames_4", dict(video="Sample Media Clip 7.mp4", max_frames=4),
     lambda p, c: [] if len(list((c / "frames").glob("frame_*.jpg"))) <= 4 else ["more than 4 frames"]),
    ("frame_max_side_256", dict(video="Sample Media Clip 16.mp4", frame_max_side=256),
     lambda p, c: _check_frame_side(c, 256)),
    ("max_seconds_4", dict(video="Sample Media Clip 7.mp4", max_seconds=4.0),
     lambda p, c: check_prompt("T2VA", p)),
    # --- audio options ---
    ("no_asr", dict(video="Sample Media Clip 2.mp4", use_asr=False),
     lambda p, c: [] if not (c / "transcript.txt").exists() else ["transcript written with use_asr=False"]),
    ("no_audio_tags", dict(video="Sample Media Clip 2.mp4", use_audio_tags=False),
     lambda p, c: [] if not (c / "audio_tags.txt").exists() else ["tags written with use_audio_tags=False"]),
    ("tags_threshold_05_max_3", dict(video="Sample Media Clip 2.mp4", tags_threshold=0.5, tags_max=3),
     lambda p, c: _check_tag_lines(c, 3)),
    ("asr_model_tiny", dict(video="Sample Media Clip 2.mp4", asr_model="tiny"),
     lambda p, c: check_prompt("T2VA", p)),
    # --- fade detection ---
    ("keep_fade_on_fades", dict(video="fades", keep_fade=True),
     lambda p, c: check_prompt("T2VA", p)),
    ("fade_detection_trims", dict(video="fades"),
     lambda p, c: check_prompt("T2VA", p)),
    # --- synthetic inputs ---
    ("no_audio_video", dict(video="no_audio", use_asr=True, use_audio_tags=True),
     lambda p, c: [] if not (c / "transcript.txt").exists() else ["transcript for silent video"]),
    ("vertical_video", dict(video="vertical"),
     lambda p, c: _check_frame_side(c, 512)),
    ("one_second_video", dict(video="one_second"),
     lambda p, c: check_prompt("T2VA", p)),
    ("hd_ready_video", dict(video="hd_ready"),
     lambda p, c: check_prompt("T2VA", p)),
]


def _check_frame_side(case_dir: Path, max_side: int) -> list:
    from PIL import Image
    for f in (case_dir / "frames").glob("frame_*.jpg"):
        w, h = Image.open(f).size
        if max(w, h) > max_side:
            return [f"frame {f.name} is {w}x{h}, exceeds {max_side}"]
    return []


def _check_tag_lines(case_dir: Path, max_lines: int) -> list:
    f = case_dir / "audio_tags.txt"
    if f.exists() and len(f.read_text(encoding="utf-8").strip().splitlines()) > max_lines:
        return ["more tags than tags_max"]
    return []


def run_case(vlm, name, kwargs, check, videos) -> tuple:
    kwargs = dict(kwargs)
    video_name = kwargs.pop("video")
    video_path = videos[video_name] if video_name in videos else DEFAULT_VIDEOS / video_name
    case_dir = ROOT / "out" / "matrix" / name
    expect_fail = check is None
    t0 = time.time()
    try:
        prompt = video_to_prompt(str(video_path), case_dir, vlm=vlm, **kwargs)
    except H3PipelineError as e:
        return (name, "PASS" if expect_fail else "FAIL", time.time() - t0,
                None if expect_fail else str(e)[:200])
    except Exception as e:
        return (name, "FAIL", time.time() - t0, f"unexpected {type(e).__name__}: {e}")
    if expect_fail:
        return (name, "FAIL", time.time() - t0, "expected error, got prompt")
    errs = check(prompt, case_dir)
    return (name, "PASS" if not errs else "FAIL", time.time() - t0, errs or None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="only the first 12 cases")
    args = ap.parse_args()

    videos = build_synthetic_videos()
    print(f"synthetic videos: {sorted(videos)}", flush=True)
    vlm = build_vlm("Qwen 2.5 VL 7B Instruct (legacy workflows)")

    cases = CASES[:12] if args.quick else CASES
    failures = 0
    rows = []
    for name, kwargs, check in cases:
        row = run_case(vlm, name, kwargs, check, videos)
        rows.append(row)
        if row[1] == "FAIL":
            failures += 1
        print(f"{row[1]} {name} ({row[2]:.0f}s)" + (f" errs={row[3]}" if row[3] else ""), flush=True)

    print("\n=== summary ===")
    for name, status, secs, err in rows:
        print(f"{status:4} {name:28} {secs:5.0f}s" + (f"  {err}" if err else ""))
    print(f"\n{len(rows) - failures}/{len(rows)} passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
