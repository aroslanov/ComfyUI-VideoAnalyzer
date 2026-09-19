# pipeline.py - video -> MiniMax H3 / LTX-2.5 prompt pipeline for ComfyUI.
# Ported from https://github.com/knishika62/video-analyzer (h3_video2prompt.py +
# h3_video2prompt_frames.py). Changes for ComfyUI/Windows:
#   - die() -> H3PipelineError (ComfyUI shows it on the node)
#   - ASR: mlx-whisper (Apple Silicon only) -> faster-whisper (CPU int8)
#   - PANNs audio tagging: subprocess isolation (for mlx/numba clashes) -> in-process
#   - CLI main() -> video_to_prompt() driven by node inputs
#   - LLM: OpenAI-compatible HTTP endpoint -> in-process VLM from ComfyUI_VLM_nodes
#     (frames go through the model's native video pathway, one call per pass)

import json
import os
import re
import shutil
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

KEYFRAME_MODES = ("I2VA", "FL2VA", "L2VA")
H3_MIN_DURATION, H3_MAX_DURATION = 4, 15
LTX_MIN_DURATION, LTX_MAX_DURATION = 6, 20
LTX_VALID_DURATIONS = (6, 8, 10, 12, 14, 16, 18, 20)
DEFAULT_ASR_MODEL = "small"  # faster-whisper: tiny/base/small/medium/large-v3-turbo or HF repo id
DEFAULT_TAG_THRESHOLD = 0.05
DEFAULT_TAG_MAX = 12
PANNS_WEIGHTS = Path(__file__).resolve().parent / ".panns" / "Cnn14_mAP=0.431.pth"
PANNS_WEIGHTS_URL = ("https://huggingface.co/thelou1s/panns-inference/"
                     "resolve/main/Cnn14_mAP%3D0.431.pth")
TEMPERATURE = 0.2
TOP_P = 0.9


class H3PipelineError(RuntimeError):
    pass


def _device_hint(e: RuntimeError) -> H3PipelineError:
    from .vlm import VRAM_HINT

    s = str(e)
    if "same device" in s or "index_select" in s or VRAM_HINT in s:
        return H3PipelineError(VRAM_HINT)
    return H3PipelineError(s)


def log(msg: str) -> None:
    print(f"[h3] {msg}", flush=True)


def die(msg: str) -> None:
    raise H3PipelineError(msg)


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        die(f"{name} not found in PATH. Install ffmpeg (https://www.gyan.dev/ffmpeg/builds/) "
            "and restart ComfyUI.")
    return path


def run_cmd(cmd: list, desc: str) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        die(f"{desc} failed (returncode={proc.returncode}):\n{proc.stderr.strip()[-2000:]}")
    return proc


# ---------------------------------------------------------------------------
# ffmpeg / ffprobe
# ---------------------------------------------------------------------------

def probe_video(video: Path) -> dict:
    out = run_cmd(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate",
         "-show_entries", "format=duration",
         "-of", "json", str(video)],
        "ffprobe",
    )
    data = json.loads(out.stdout)
    stream = data.get("streams", [{}])[0]
    duration = float(data.get("format", {}).get("duration", 0.0))
    if not stream.get("width") or not stream.get("height") or duration <= 0:
        die("Could not read video stream or duration.")
    num, _, den = (stream.get("avg_frame_rate") or "30/1").partition("/")
    try:
        fps = float(num) / float(den)
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    if not (1.0 <= fps <= 240.0):
        fps = 30.0
    return {"duration": duration, "width": int(stream["width"]),
            "height": int(stream["height"]), "fps": fps}


def scan_yavg(video: Path) -> list:
    out = run_cmd(
        ["ffmpeg", "-i", str(video),
         "-vf", "scale=192:108,signalstats,metadata=print:key=lavfi.signalstats.YAVG",
         "-f", "null", "-"],
        "YAVG scan",
    )
    return [float(v) for v in re.findall(r"YAVG=([0-9.]+)", out.stderr)]


def black_intervals(video: Path, pix_th: float = 0.20, min_dur: float = 0.1) -> list:
    out = run_cmd(
        ["ffmpeg", "-i", str(video),
         "-vf", f"blackdetect=pix_th={pix_th}:black_min_duration={min_dur}",
         "-f", "null", "-"],
        "black interval detection",
    )
    return [(float(a), float(b))
            for a, b in re.findall(r"black_start:([0-9.]+) black_end:([0-9.]+)", out.stderr)]


def detect_content_range(video: Path, duration: float, fps: float) -> tuple:
    """Content range excluding leading/trailing fade-in/out (black) segments."""
    yavg = scan_yavg(video)
    if len(yavg) < max(10, int(fps * 0.5)):
        return (0.0, duration, "skipped detection (too short)")

    med = statistics.median(yavg)
    if med < 24.0:
        log("video is almost black - skipping fade detection")
        return (0.0, duration, "all-black")

    target = 0.9 * med
    intervals = black_intervals(video)
    lead = next((iv for iv in intervals if iv[0] <= 0.05), None)
    trail = next((iv for iv in intervals if iv[1] + 1.0 / fps >= duration - 0.03), None)
    if not lead and not trail:
        return (0.0, duration, "no black intervals (no fades)")

    start, end = 0.0, duration
    notes = []
    max_advance = 3.0

    if lead:
        i0 = max(0, int(lead[0] * fps))
        for j in range(i0, len(yavg)):
            if yavg[j] >= target:
                start = min(j / fps, lead[1] + max_advance)
                notes.append(f"lead black {lead[0]:.2f}s->{lead[1]:.2f}s => content start {start:.2f}s")
                break
        else:
            log("warning: no bright frame found after leading black interval; keeping start at 0.0s")
    if trail:
        i0 = min(len(yavg) - 1, int(trail[0] * fps))
        for j in range(i0, -1, -1):
            if yavg[j] >= target:
                end = max((j + 1) / fps, trail[0] - max_advance)
                notes.append(f"trailing black {trail[0]:.2f}s->{trail[1]:.2f}s => content end {end:.2f}s")
                break
        else:
            log("warning: no bright frame found before trailing black interval; using full tail")

    if end - start < 0.5 * duration:
        log("warning: detected content range < 50% of video; falling back to full range")
        return (0.0, duration, "range too short, fallback")
    return (start, end, "; ".join(notes))


def extract_keyframes(src: Path, out_dir: Path, fps: float,
                      first_at: float = 0.0, last_at: float | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    first = out_dir / "first.jpg"
    last = out_dir / "last.jpg"
    run_cmd(["ffmpeg", "-y", "-v", "error", "-ss", f"{first_at:.3f}", "-i", str(src),
             "-frames:v", "1", "-q:v", "2", str(first)], "first keyframe extraction")
    if last_at is None:
        last_cmd = ["ffmpeg", "-y", "-v", "error", "-sseof", "-0.15", "-i", str(src),
                    "-frames:v", "1", "-q:v", "2", str(last)]
    else:
        seek = max(0.0, last_at - 1.0 / fps)
        last_cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{seek:.3f}", "-i", str(src),
                    "-frames:v", "1", "-q:v", "2", str(last)]
    run_cmd(last_cmd, "last keyframe extraction")
    return {"first": str(first), "last": str(last)}


# ---------------------------------------------------------------------------
# VLM (in-process, via ComfyUI_VLM_nodes ModernVLMPredictor)
# ---------------------------------------------------------------------------

def frames_to_tensor(frames: list):
    """[(jpg_path, abs_seconds), ...] -> torch batch [N,H,W,3] float 0-1 (ComfyUI IMAGE layout)."""
    import numpy as np
    import torch
    from PIL import Image
    arrays = [np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
              for p, _ in frames]
    return torch.from_numpy(np.stack(arrays))


def text_generate(predictor, prompt: str, max_new_tokens: int,
                  temperature: float = TEMPERATURE) -> str:
    """Text-only chat completion on the loaded model (see vlm.VLMPredictor)."""
    return predictor.generate_text(prompt, max_new_tokens, temperature, TOP_P)


def extract_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    start = text.find("{")
    if start == -1:
        raise ValueError("JSON block not found")
    text = text[start:]
    if not text.rstrip().endswith("}"):
        # truncated generation: keep the open tail, json_repair closes it below
        pass
    else:
        text = text[:text.rfind("}") + 1]
    # common small-model slip: trailing commas
    text = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # last resort for small models: unescaped quotes, truncated output etc.
        import json_repair
        repaired = json_repair.loads(text)
        if isinstance(repaired, dict):
            return repaired
        raise


# ---------------------------------------------------------------------------
# Pass 1: video analysis
# ---------------------------------------------------------------------------

ANALYSIS_JSON_SCHEMA = """{
  "style": "live-action, cinematic, ...",
  "shots": [
    {"index": 1, "time_range": "0.0s-3.5s",
     "composition": "shot size / framing / subject position",
     "subjects": "appearance, clothing, objects, spatial relations",
     "actions": "what happens, in order",
     "camera_motion": "e.g. push in with small amplitude at slow speed; 'static shot' if still",
     "environment": "scene, lighting, props"}
  ],
  "speakers": [{"id": "S1", "description": "age, gender, on/off-screen, pitch, timbre, rate, accent"}],
  "dialogue": [{"time": "~2.0s", "speaker": "S1", "text": "...", "language": "English",
                 "confidence": "low"}],
  "on_screen_text": ["exact visible text, verbatim"],
  "sounds": "ambient / physical / non-verbal human sounds heard or implied"
}"""

ANALYSIS_PROMPT = """You are a video analysis expert for video-generation prompting.
Observe the attached video closely and output structured JSON usable only for the downstream prompt generation.

Rules:
- All JSON values must be written in English (on_screen_text and dialogue.text keep the original language as shown/heard)
- Split shots in playback order and assign index for each cut. time_range is absolute seconds within the video
- Describe camera_motion as "motion type + (if meaningful) amplitude + speed". Use "static shot" if still
- Assign stable speaker IDs S1, S2... including off-screen narration
- Audio cannot be inferred from visuals, so dialogue.text is a best guess from visual cues (lip movement, subtitles, context) marked confidence "low"/"medium". Prefer readable subtitles with confidence "medium"
- Omit unclear or invisible dialogue rather than guessing, or mark it [unclear]
- Output JSON body only, no code fences

JSON schema (follow exactly):
""" + ANALYSIS_JSON_SCHEMA


# ---------------------------------------------------------------------------
# ASR (faster-whisper)
# ---------------------------------------------------------------------------

def has_audio_stream(video: Path) -> bool:
    out = run_cmd(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=codec_type", "-of", "csv", str(video)],
        "ffprobe(audio)",
    )
    return bool(out.stdout.strip())


def extract_audio(src: Path, dst: Path, start: float, length: float,
                  sample_rate: int = 16000) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-v", "error"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(src), "-t", f"{float(length):.3f}",
            "-vn", "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(dst)]
    run_cmd(cmd, "audio extraction")


def transcribe(wav: Path, model: str, language: str | None) -> dict:
    # ponytail: CPU int8 keeps install simple (no cuDNN/cuBLAS DLL dance); move to
    # device="cuda", compute_type="float16" if ASR latency ever matters
    from faster_whisper import WhisperModel
    log(f"ASR: {model} (first run downloads the model)")
    t0 = time.time()
    m = WhisperModel(model, device="cpu", compute_type="int8")
    segments, info = m.transcribe(str(wav), language=language)
    segments = list(segments)
    # whisper sometimes decodes non-Latin speech as pure '?' runs; that junk
    # also derails the VLM, so drop such segments
    clean = []
    for s in segments:
        alnum = sum(c.isalnum() for c in s.text)
        if alnum == 0 or s.text.count("?") > alnum / 2:
            continue
        clean.append(s)
    log(f"ASR done: {time.time() - t0:.1f} s (language={info.language}, "
        f"{len(clean)}/{len(segments)} segments kept)")
    return {
        "segments": [{"start": s.start, "end": s.end, "text": s.text} for s in clean],
        "text": "".join(s.text for s in clean),
        "language": info.language,
    }


def format_transcript(result: dict) -> str:
    def ts(sec: float) -> str:
        m, s = divmod(float(sec), 60)
        return f"{int(m):02d}:{s:06.3f}"

    lines = []
    for seg in result.get("segments", []):
        text = seg.get("text", "").strip()
        if text:
            lines.append(f"[{ts(seg.get('start', 0))}-{ts(seg.get('end', 0))}] {text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Audio tags (PANNs / AudioSet)
# ---------------------------------------------------------------------------

def _ensure_panns_weights() -> Path:
    if PANNS_WEIGHTS.is_file() and PANNS_WEIGHTS.stat().st_size > 3e8:
        return PANNS_WEIGHTS
    PANNS_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    part = PANNS_WEIGHTS.parent / (PANNS_WEIGHTS.name + ".part")
    log(f"downloading PANNs weights (~315MB): {PANNS_WEIGHTS}")
    import urllib.request
    urllib.request.urlretrieve(PANNS_WEIGHTS_URL, part)
    size = part.stat().st_size
    if size <= 3e8:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"PANNs weights download incomplete ({size} bytes)")
    part.replace(PANNS_WEIGHTS)
    return PANNS_WEIGHTS


def audio_tags(wav_32k: Path, threshold: float = DEFAULT_TAG_THRESHOLD,
               max_tags: int = DEFAULT_TAG_MAX) -> list:
    os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
    os.environ.setdefault("NUMBA_CACHE_DIR", tempfile.gettempdir())
    import wave
    import numpy as np
    import torch
    from panns_inference.models import Cnn14
    from panns_inference.config import labels, classes_num

    log("computing audio tags (PANNs)...")
    t0 = time.time()
    weights = _ensure_panns_weights()
    model = Cnn14(sample_rate=32000, window_size=1024, hop_size=320,
                  mel_bins=64, fmin=50, fmax=14000, classes_num=classes_num)
    ckpt = torch.load(str(weights), map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    with wave.open(str(wav_32k), "rb") as w:
        data = np.frombuffer(w.readframes(w.getnframes()),
                             dtype=np.int16).astype(np.float32) / 32768.0
    audio = torch.from_numpy(data).unsqueeze(0)
    with torch.no_grad():
        out = model(audio, None)["clipwise_output"]

    probs = out.squeeze(0).numpy()
    order = np.argsort(probs)[::-1]
    tags = [[labels[i], float(probs[i])] for i in order if probs[i] >= threshold][:max_tags]
    log(f"audio tags: {time.time() - t0:.1f} s")
    return [tuple(t) for t in tags]


# ---------------------------------------------------------------------------
# Pass 1: frame sampling + single-call analysis over the model's video pathway
# ---------------------------------------------------------------------------

def extract_frames(video: Path, out_dir: Path, start: float, length: float,
                   fps: float, max_side: int, max_frames: int) -> list:
    """Sample jpg frames at fps over [start, start+length]. Returns [(path, abs_seconds)]."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.glob("frame_*.jpg"):
        p.unlink()
    vf = (f"fps={fps},scale='min({max_side},iw)':'min({max_side},ih)'"
          ":force_original_aspect_ratio=decrease")
    cmd = ["ffmpeg", "-y", "-v", "error"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(video), "-t", f"{length:.3f}", "-vf", vf,
            "-frames:v", str(max_frames), "-q:v", "3", str(out_dir / "frame_%04d.jpg")]
    run_cmd(cmd, "frame extraction")
    paths = sorted(out_dir.glob("frame_*.jpg"))
    if not paths:
        die("frame extraction produced 0 frames. Check frame_fps / max_frames")
    return [(p, start + i / fps) for i, p in enumerate(paths)]


def run_analysis(predictor, frames: list, transcript: str | None,
                 tags: list | None, frame_fps: float, max_new_tokens: int) -> dict:
    """Pass 1: all sampled frames in one call through the model's video pathway."""
    text = ANALYSIS_PROMPT
    if transcript:
        text += (
            "\n\n[Transcript (ground truth for dialogue/lyrics)]\n"
            "A timestamped transcript is provided (speech or sung lyrics).\n"
            "Match dialogue text/language/time to this transcript and set confidence to \"high\".\n"
            "Treat sung lyrics as dialogue too.\n"
            + transcript
        )
    if tags:
        text += (
            "\n\n[Audio tags (computed from the real audio: PANNs/AudioSet, with confidence)]\n"
            "The following music/instrument/voice/ambience labels were detected in the real audio.\n"
            "Keep the sounds field and any music description consistent with these tags.\n"
            + "\n".join(tags)
        )
    video_batch = frames_to_tensor(frames)
    log(f"analyzing {len(frames)} frames (native video input)...")
    try:
        raw = predictor.generate(
            images=None,
            prompt=text,
            system_prompt="",
            max_new_tokens=max_new_tokens,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            video_frames=video_batch,
            fps=frame_fps,
        )
    except RuntimeError as e:
        raise _device_hint(e)
    try:
        return extract_json(raw)
    except (ValueError, json.JSONDecodeError) as e:
        fail_path = frames[0][0].parent / "analysis_failed_1.txt"
        fail_path.write_text(raw + "\n", encoding="utf-8")
        log(f"JSON parse failed ({e}); raw saved to {fail_path.name}. Retrying...")
        raw2 = predictor.generate(
            images=None,
            prompt=text + "\n\nIMPORTANT: Output only valid JSON following the schema, no code fences.",
            system_prompt="",
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            top_p=TOP_P,
            video_frames=video_batch,
            fps=frame_fps,
        )
        try:
            return extract_json(raw2)
        except (ValueError, json.JSONDecodeError):
            (frames[0][0].parent / "analysis_failed_2.txt").write_text(raw2 + "\n", encoding="utf-8")
            raise


# ---------------------------------------------------------------------------
# Pass 2: H3 prompt rewrite
# ---------------------------------------------------------------------------

def build_alignment_line(mode: str, duration: float) -> str | None:
    dur = f"{duration:.2f}"
    if mode == "I2VA":
        return ("For the target video, at 0.00 seconds into the target video, "
                "<Picture 1> (from [Shot 1]) is fully referenced.")
    if mode == "FL2VA":
        return (f"How the reference pictures align with the target video — "
                f"Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; "
                f"Picture 2 (from Shot 1) aligns with the {dur}-second mark of the target video.")
    if mode == "L2VA":
        return (f"How the reference pictures align with the target video — "
                f"<Picture 1> (from [Shot 1]) aligns with the {dur}-second mark of the target video.")
    return None  # T2VA


def build_rewrite_prompt(mode: str, duration: float, analysis: dict,
                         guide_text: str, transcript: str | None,
                         tags: list | None = None,
                         output_style: str = "markers") -> str:
    shots = analysis.get("shots", [])
    single_shot = len(shots) <= 1
    is_ltx = (mode == "LTX")

    if is_ltx:
        mode_rule = (
            "Mode: LTX (LTX-2.5 text-to-video). Output ONE natural-language prompt in English, following the LTX guide.\n"
            "There are NO structured fields (no integrated_multimodal_description / overall_soundscape / non_diegetic_music) "
            "and NO image-alignment line — output ONLY the prompt body.\n"
            "Structure: a single flowing paragraph for a continuous take, or screenplay-style (scene header, character cues, "
            "quoted dialogue) for dialogue / multi-beat / multi-shot content. Aim for roughly 4-8 descriptive sentences.\n"
            "Cover the guide's key elements: (1) establish the shot (genre + shot scale), (2) set the scene "
            "(lighting, color palette, textures, atmosphere), (3) describe the action as a natural sequence in PRESENT TENSE, "
            "(4) define the character(s) with concrete features and express emotion through physical cues, not labels, "
            "(5) identify camera movement and how subjects appear AFTER it, (6) describe the audio "
            "(ambient sound, music, speech/singing — spoken lines in double quotation marks, with language/accent if needed).\n"
            "If the analysis has multiple shots, write 2-4 shots as ONE chronological paragraph, naming each cut in plain language "
            "(hard cut / match cut / dissolve), re-establishing the new framing, keeping re-appearing subjects' identity consistent, "
            "and stating whether audio continues or changes across each cut.\n"
            "Japanese dialogue/lyrics must be written in HIRAGANA ONLY (no kanji, no katakana), e.g. "
            "\"今日は六本木をブラブラしてる\" -> \"きょうはろっぽんぎをぶらぶらしてる\". Other languages stay verbatim."
        )
    elif mode == "T2VA":
        if output_style == "canonical":
            mode_rule = (
                "Mode: T2VA (text only). Output the three core fields in this exact order:\n"
                "1. integrated_multimodal_description: [Shot 1] ...\n"
                "2. overall_soundscape: ...\n"
                "3. non_diegetic_music: ...\n"
                "There is no image-alignment line in T2VA; start directly with integrated_multimodal_description."
            )
        else:
            mode_rule = (
                "Mode: T2VA (text only). Output three sections in this exact order, each starting "
                "with its marker on its own line, content on the following lines:\n"
                "[DESCRIPTION]\n<the integrated multimodal description, starting with [Shot 1] ...>\n"
                "[SOUNDSCAPE]\n<the overall soundscape>\n"
                "[MUSIC]\n<the non-diegetic music, or N/A>\n"
                "Do NOT write the long field names (integrated_multimodal_description etc.) — "
                "use only these markers. There is no image-alignment line in T2VA."
            )
    elif mode == "I2VA":
        if output_style == "canonical":
            mode_rule = (
                "Mode: I2VA. The user will supply <Picture 1> as the actual first frame of the target video.\n"
                "Output ONLY the three core fields (integrated_multimodal_description / overall_soundscape / non_diegetic_music) in that order, "
                "separated by one blank line. Do NOT output the image-alignment instruction line — the tool adds it automatically.\n"
                "Content rule: Shot 1 must start from the state shown in <Picture 1> (establish style, subjects, composition, scene anchors first) and develop forward. "
                "Keep character identity, clothing, colors, key objects, and spatial relations consistent with the reference."
            )
        else:
            mode_rule = (
                "Mode: I2VA. The user will supply <Picture 1> as the actual first frame of the target video.\n"
                "Output three sections in this exact order, each starting with its marker on its own line, "
                "content on the following lines: [DESCRIPTION], [SOUNDSCAPE], [MUSIC]. "
                "Do NOT write the long field names — use only these markers, and do NOT output the "
                "image-alignment instruction line (the tool adds it automatically).\n"
                "Content rule: the [DESCRIPTION] section must start from the state shown in <Picture 1> "
                "(establish style, subjects, composition, scene anchors first) and develop forward. "
                "Keep character identity, clothing, colors, key objects, and spatial relations consistent with the reference."
            )
    elif mode == "FL2VA":
        if output_style == "canonical":
            mode_rule = (
                "Mode: FL2VA. The user will supply Picture 1 (first frame) and Picture 2 (last frame).\n"
                "Output ONLY the three core fields in that order. Do NOT output the image-alignment instruction line — the tool adds it automatically.\n"
                "Content rule: write a single continuous shot describing the motion path from Picture 1 to Picture 2: "
                "first-frame state -> observable intermediate changes -> progressively narrowing differences -> last-frame state. "
                "Do not repeat two static image descriptions; supply the connecting motion. The last frame must be reached at the end of the shot."
            )
        else:
            mode_rule = (
                "Mode: FL2VA. The user will supply Picture 1 (first frame) and Picture 2 (last frame).\n"
                "Output three sections in this exact order, each starting with its marker on its own line, "
                "content on the following lines: [DESCRIPTION], [SOUNDSCAPE], [MUSIC]. "
                "Do NOT write the long field names — use only these markers, and do NOT output the "
                "image-alignment instruction line (the tool adds it automatically).\n"
                "Content rule: the [DESCRIPTION] section is a single continuous shot describing the motion path "
                "from Picture 1 to Picture 2: first-frame state -> observable intermediate changes -> "
                "progressively narrowing differences -> last-frame state. "
                "Do not repeat two static image descriptions; supply the connecting motion. The last frame must be reached at the end of the shot."
            )
    else:  # L2VA
        if output_style == "canonical":
            mode_rule = (
                "Mode: L2VA. The user will supply <Picture 1> as the actual LAST frame of the target video.\n"
                "Output ONLY the three core fields in that order. Do NOT output the image-alignment instruction line — the tool adds it automatically.\n"
                "Content rule: write a single shot that infers a plausible preceding state, then lets actions, object states, and composition "
                "gradually converge and land exactly on <Picture 1> in the final moment: "
                "plausible preceding state -> explicit action and transition path -> gradual convergence -> last-frame landing."
            )
        else:
            mode_rule = (
                "Mode: L2VA. The user will supply <Picture 1> as the actual LAST frame of the target video.\n"
                "Output three sections in this exact order, each starting with its marker on its own line, "
                "content on the following lines: [DESCRIPTION], [SOUNDSCAPE], [MUSIC]. "
                "Do NOT write the long field names — use only these markers, and do NOT output the "
                "image-alignment instruction line (the tool adds it automatically).\n"
                "Content rule: the [DESCRIPTION] section is a single shot that infers a plausible preceding state, "
                "then lets actions, object states, and composition gradually converge and land exactly on "
                "<Picture 1> in the final moment: plausible preceding state -> explicit action and transition path "
                "-> gradual convergence -> last-frame landing."
            )

    extra = ""
    if transcript:
        if is_ltx:
            extra += (
                "\n## Ground-truth transcript (with timestamps)\n"
                "Use this transcript for the spoken words and sung lyrics. Put each line in double quotation marks "
                "inside the prompt, preserving its exact words and original language, and place each line at its "
                "timestamp position in the timeline. For Japanese, write each line in hiragana only (no kanji, no katakana):\n"
                + transcript + "\n"
            )
        else:
            extra += (
                "\n## Ground-truth transcript (with timestamps)\n"
                "Use this transcript for the spoken words and sung lyrics inside <d>. "
                "Preserve its exact words and language, and place each line at its timestamp position in the timeline:\n"
                + transcript + "\n"
            )
    if tags:
        if is_ltx:
            extra += (
                "\n## Audio tags (from the actual audio: PANNs/AudioSet, with confidence)\n"
                "The following music / instrument / voice / ambience labels were detected in the real audio. "
                "Use them to make the prompt's audio description (ambient sound, music, voice) accurate and consistent "
                "(instrumentation, genre, crowd/ambience). Do not invent music that these tags contradict:\n"
                + "\n".join(tags) + "\n"
            )
        else:
            extra += (
                "\n## Audio tags (from the actual audio: PANNs/AudioSet, with confidence)\n"
                "The following music / instrument / voice / ambience labels were detected in the real audio. "
                "Use them to make overall_soundscape and any diegetic-music description accurate and consistent "
                "(instrumentation, genre, crowd/ambience). Do not invent music that these tags contradict:\n"
                + "\n".join(tags) + "\n"
            )

    if is_ltx:
        return f"""You are writing an LTX-2.5 video-generation prompt.
Read the LTX prompt-writing guide below, then use the video analysis JSON to produce the FINAL prompt.

## Target parameters
- Mode: LTX (LTX-2.5 text-to-video)
- Target video duration: {duration:.2f} seconds (keep every action/cut inside this duration)
- Shot plan from analysis: {len(shots)} shot(s)

## Mode rule (must follow exactly)
{mode_rule}

{extra}
## LTX prompt-writing guide
{guide_text}

## Video analysis JSON (from vision LLM; dialogue marked "low" confidence is an inference — use it if it fits, and you may drop it if implausible; "high"/"medium" should be kept)
{json.dumps(analysis, ensure_ascii=False, indent=2)}

## Output rules
1. Write the prompt in English: ONE flowing paragraph for a single continuous take, or screenplay-style (scene header, character cues, quoted dialogue) for dialogue / multi-beat / multi-shot content.
2. No markdown fences, no commentary, no explanations before or after the prompt.
3. No structured field names, no image-alignment instruction lines — output ONLY the prompt body.
4. Use present tense for actions/movement; express emotion through physical cues, not abstract labels.
5. Dialogue / lyrics / singing go in double quotation marks in the original language; specify language/accent when needed.
   Japanese lines must be in HIRAGANA ONLY (no kanji, no katakana).
6. If there are multiple shots (2-4), keep them in one chronological paragraph with explicit transition language at each cut and stated audio continuity.
7. Do not invent on-screen text, brands, or logos. If the source has readable text, keep it short.
8. If there is no dialogue, no singing, and no on-screen speaking source, do not invent any.
"""

    if output_style == "canonical":
        output_rules = """1. Write ONLY the three core fields, in English, separated by one blank line, in the order required by the mode rule above.
2. No markdown fences, no commentary, no explanations before or after the fields.
3. Preserve the exact field names: integrated_multimodal_description, overall_soundscape, non_diegetic_music.
4. Never output the image-alignment instruction line ("For the target video..." or "How the reference pictures align...") — the tool adds it.
5. For keyframe modes, reference <Picture 1>/Picture 2 in the shot descriptions.
6. Dialogue inside <d>[Language] ... </d> must keep the original language verbatim.
7. If there is no dialogue, no singing, and no on-screen speaking source, do not invent any <d> blocks.
8. overall_soundscape: 1-4 sentences; non_diegetic_music: 1-3 sentences or "N/A".
9. For overall_soundscape, infer plausible ambient/physical/non-verbal sounds from the visual content (weather, locations, actions, crowds). Use "N/A" only when the scene truly has no plausible sound source.
"""
    else:
        output_rules = """1. Write ONLY the three marker sections ([DESCRIPTION], [SOUNDSCAPE], [MUSIC]), in English, in that exact order. Each marker stands alone on its own line; its content follows on the next lines.
2. No markdown fences, no commentary, no explanations before or after the sections.
3. Never write the long field names (integrated_multimodal_description, overall_soundscape, non_diegetic_music) — the tool converts markers to the final format.
4. Never output the image-alignment instruction line ("For the target video..." or "How the reference pictures align...") — the tool adds it.
5. For keyframe modes, reference <Picture 1>/Picture 2 in the [DESCRIPTION] section.
6. Dialogue inside <d>[Language] ... </d> must keep the original language verbatim.
7. If there is no dialogue, no singing, and no on-screen speaking source, do not invent any <d> blocks.
8. [SOUNDSCAPE]: 1-4 sentences; [MUSIC]: 1-3 sentences or "N/A".
9. For [SOUNDSCAPE], infer plausible ambient/physical/non-verbal sounds from the visual content (weather, locations, actions, crowds). Use "N/A" only when the scene truly has no plausible sound source.
"""

    return f"""You are writing a MiniMax H3 video-generation prompt.
Read the H3 prompt-writing guide below, then use the video analysis JSON to produce the FINAL prompt.

## Target parameters
- Mode: {mode}
- Target video duration: {duration:.2f} seconds (every cut time must be strictly inside this duration; format the alignment line with the duration to exactly two decimal places)
- Shot plan from analysis: {len(shots)} shot(s){" -> keep as a single shot" if single_shot and mode in ("FL2VA", "L2VA") else ""}

## Mode rule (must follow exactly)
{mode_rule}

{extra}
## H3 prompt-writing guide
{guide_text}

## Video analysis JSON (from vision LLM; dialogue marked "low" confidence is an inference — use it if it fits, and you may drop it if implausible; "high"/"medium" should be kept)
{json.dumps(analysis, ensure_ascii=False, indent=2)}

## Output rules
{output_rules}"""


def format_shot_breaks(text: str) -> str:
    label = "integrated_multimodal_description:"
    idx = text.find(label)
    if idx < 0:
        return text
    head, rest = text[: idx + len(label)], text[idx + len(label):]
    rest = re.sub(r"\s*(\[(?:Shot|shot) \d+\])", r"\n\1", rest)
    return head + rest


def strip_alignment_line(text: str) -> str:
    lines = text.split("\n")
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i < len(lines) and (
        lines[i].startswith("For the target video,")
        or lines[i].startswith("How the reference pictures align")
    ):
        lines[i] = ""
    return "\n".join(lines).lstrip("\n")


def strip_stray_tags(text: str) -> str:
    def _keep(m: "re.Match") -> str:
        tag = m.group(0)
        if tag.startswith("</"):
            return tag if tag == "</d>" else ""
        return tag if re.fullmatch(r"<(d|Picture \d+)>", tag) else ""
    return re.sub(r"</?[a-zA-Z][a-zA-Z0-9 _-]*>", _keep, text)


def _fix_field_labels(text: str) -> str:
    """Repair near-miss field labels (e.g. 'overall_soundsscape', 'Non-Diegetic-Music')
    before validation."""
    import difflib
    fields = ("integrated_multimodal_description", "overall_soundscape",
              "non_diegetic_music")
    # pass 1: separator/case variants ("non-diegetic music", "non non diegetic music")
    for expected in fields:
        first, rest = expected.split("_", 1)
        flex = r"[\s_-]*".join(map(re.escape, rest.split("_")))
        pattern = re.compile(
            r"(?<![A-Za-z_])" + re.escape(first) + r"[\s_-]*"
            + rf"(?:{re.escape(first)}[\s_-]*)?" + flex + r"\b",
            re.IGNORECASE,
        )
        text = pattern.sub(expected, text)
    # pass 2: letter-level typos ("overall_soundsscape")
    tokens = set(re.findall(r"[a-zA-Z_]{8,}", text))
    for expected in fields:
        if expected + ":" in text:
            continue
        close = difflib.get_close_matches(expected, tokens, n=1, cutoff=0.85)
        if close and close[0] != expected:
            log(f"fixed field label typo: {close[0]!r} -> {expected!r}")
            text = text.replace(close[0], expected)
    # pass 3: label reduced to its last word at line start ("Soundscape: ...")
    for expected in fields:
        if expected + ":" in text:
            continue
        last = expected.rsplit("_", 1)[-1]
        text, n = re.subn(rf"(?m)^(\s*){last}\s*:", rf"\1{expected}:", text,
                          count=1, flags=re.IGNORECASE)
        if n:
            log(f"restored shortened field label: {last!r} -> {expected!r}")
    return text


_MARKERS = (("DESCRIPTION", "integrated_multimodal_description"),
            ("SOUNDSCAPE", "overall_soundscape"),
            ("MUSIC", "non_diegetic_music"))


def _markers_to_fields(raw: str) -> str | None:
    """Reassemble canonical H3 fields from [DESCRIPTION]/[SOUNDSCAPE]/[MUSIC] sections."""
    if not all(f"[{name}]" in raw for name, _ in _MARKERS):
        return None
    parts = re.split(r"\[(DESCRIPTION|SOUNDSCAPE|MUSIC)\]", raw)
    sections = {}
    for i in range(1, len(parts) - 1, 2):
        sections[parts[i]] = parts[i + 1].strip()
    if len(sections) != 3 or not all(sections.get(name) for name, _ in _MARKERS):
        return None
    return "\n\n".join(f"{field}: {sections[name]}"
                       for name, field in _MARKERS)


def _validate_rewrite(raw: str, mode: str) -> str:
    # only unwrap a code fence when it wraps the whole output; trailing junk
    # fences after the fields must not swallow the prompt itself
    s = raw.strip()
    m = re.match(r"^```[a-zA-Z]*\s*(.*?)\s*```\s*$", s, re.DOTALL)
    if m:
        raw = m.group(1).strip()
    else:
        raw = re.sub(r"^(?:\s*```[a-zA-Z]*\s*)+", "", s)   # stray opening fence
        raw = re.sub(r"(?:\s*```\s*)+$", "", raw)          # trailing fence runs
    raw = raw.strip()
    if mode != "LTX":
        assembled = _markers_to_fields(raw)
        if assembled is not None:
            raw = assembled
    if mode == "LTX":
        if len(raw.split()) < 20:
            die(f"LLM output too short for an LTX prompt: {len(raw.split())} words")
        quoted = (re.findall(r'"([^"]*)"', raw)
                  + re.findall(r"'([^']*)'", raw)
                  + re.findall(r"「([^」]*)」", raw))
        bad = [q for q in quoted if re.search(r"[\u30A0-\u30FF\u3400-\u4DBF\u4E00-\u9FFF]", q)]
        if bad:
            log(f"warning: LTX dialogue contains kanji/katakana (spec: hiragana only): {bad[:3]}")
        return raw
    raw = strip_alignment_line(raw)
    raw = strip_stray_tags(raw)
    raw = _fix_field_labels(raw)
    fields = ("integrated_multimodal_description", "overall_soundscape",
              "non_diegetic_music")
    positions = []
    for field in fields:
        pos = raw.find(field + ":")
        if pos < 0:
            die(f"LLM output missing '{field}:'")
        positions.append(pos)
    if positions != sorted(positions):
        die(f"LLM output has wrong 3-field order (pos={positions})")
    return format_shot_breaks(raw)


def run_rewrite(predictor, mode: str, duration: float, analysis: dict,
                guide_text: str, transcript: str | None,
                tags: list | None = None, max_new_tokens: int = 8192,
                out_dir: Path | None = None) -> str:
    prompt = build_rewrite_prompt(mode, duration, analysis, guide_text, transcript,
                                  tags=tags)
    last_err = None
    # ~2-4% of generations from a 7B model hit a stochastic glitch (early EOS,
    # injected garbage tokens, label dodging); retry hotter and alternate the
    # output format (markers / canonical labels) since degeneration is
    # prompt-dependent
    for attempt in range(3):
        output_style = "canonical" if attempt == 2 else "markers"
        prompt = build_rewrite_prompt(mode, duration, analysis, guide_text,
                                      transcript, tags=tags,
                                      output_style=output_style)
        raw = text_generate(predictor, prompt, max_new_tokens,
                            temperature=TEMPERATURE if attempt == 0 else 0.6)
        try:
            return _validate_rewrite(raw, mode)
        except H3PipelineError as e:
            last_err = e
            if out_dir is not None:
                (out_dir / f"rewrite_failed_{attempt + 1}.txt").write_text(
                    raw + "\n", encoding="utf-8")
            log(f"rewrite validation failed ({e}); retrying with correction")
    die(f"LLM rewrite failed validation 3 times: {last_err}")


def resolve_duration(mode: str, requested: float | None, effective_len: float) -> float:
    if mode == "LTX":
        raw = round(effective_len) if requested is None else requested
        if not (LTX_MIN_DURATION <= raw <= LTX_MAX_DURATION):
            die(f"LTX mode duration must be 6-20 s (got: {requested})")
        return float(min(LTX_VALID_DURATIONS, key=lambda v: abs(v - raw)))
    dur = min(H3_MAX_DURATION, max(H3_MIN_DURATION, round(effective_len))) if requested is None else requested
    if not (H3_MIN_DURATION <= dur <= H3_MAX_DURATION):
        die(f"duration must be {H3_MIN_DURATION}-{H3_MAX_DURATION} s (got: {requested})")
    return float(dur)


# ---------------------------------------------------------------------------
# main entry used by the ComfyUI node
# ---------------------------------------------------------------------------

def video_to_prompt(video_path: str, out_dir: Path, vlm: dict,
                    mode: str = "T2VA", duration: float | None = None,
                    max_seconds: float | None = None,
                    keep_fade: bool = False,
                    frame_fps: float = 1.0, max_frames: int = 24,
                    frame_max_side: int = 512,
                    use_asr: bool = True, asr_model: str = DEFAULT_ASR_MODEL,
                    use_audio_tags: bool = True,
                    tags_threshold: float = DEFAULT_TAG_THRESHOLD,
                    tags_max: int = DEFAULT_TAG_MAX,
                    max_new_tokens: int = 8192) -> str:
    """Analyze a video file and return the final H3/LTX prompt. Writes
    analysis.json / prompt.txt (+ frames/, keyframes/, transcript.txt, audio_tags.txt)
    under out_dir for debugging."""
    require_tool("ffmpeg")
    require_tool("ffprobe")
    predictor = vlm["predictor"]
    video = Path(video_path)
    if not video.is_file():
        die(f"video not found: {video}")

    out_dir.mkdir(parents=True, exist_ok=True)
    keyframes_dir = out_dir / "keyframes"
    frames_dir = out_dir / "frames"

    info = probe_video(video)
    log(f"input video: {info['width']}x{info['height']}, {info['duration']:.2f} s, {info['fps']:.2f} fps")

    if keep_fade:
        start, end = 0.0, info["duration"]
        log("fade detection skipped (--keep-fade)")
    else:
        start, end, note = detect_content_range(video, info["duration"], info["fps"])
        log(f"content range: {start:.2f}s-{end:.2f}s ({note})")

    analysis_start = start
    analysis_len = (end - start) if max_seconds is None else min(end - start, max_seconds)
    if analysis_len <= 0.2:
        die("analysis range too short (check max_seconds)")

    duration = resolve_duration(mode, duration, analysis_len)
    log(f"target duration: {duration:.2f} s ({mode})")

    keyframe_paths = {}
    if mode in KEYFRAME_MODES:
        keyframe_paths = extract_keyframes(video, keyframes_dir, info["fps"],
                                           first_at=analysis_start,
                                           last_at=analysis_start + analysis_len)
        log(f"keyframes: {keyframe_paths['first']} / {keyframe_paths['last']}")

    default_guide = ("ltx-prompt-guide-base.md" if mode == "LTX"
                     else "minimax-h3-prompt-guide-base.md")
    guide_path = Path(__file__).resolve().parent / "md" / default_guide
    if not guide_path.is_file():
        die(f"prompt guide not found: {guide_path}")
    guide_text = guide_path.read_text(encoding="utf-8")

    transcript = None
    audio_present = has_audio_stream(video)
    if use_asr and audio_present:
        wav = out_dir / "audio_16k.wav"
        extract_audio(video, wav, analysis_start, analysis_len)
        try:
            result = transcribe(wav, asr_model, None)
        except ImportError:
            log("warning: faster_whisper not installed, skipping ASR (pip install faster-whisper)")
        except Exception as e:
            log(f"warning: ASR failed, skipping ({e})")
        else:
            transcript = format_transcript(result)
            if transcript:
                (out_dir / "transcript.txt").write_text(transcript + "\n", encoding="utf-8")
                log(f"transcript saved: {out_dir / 'transcript.txt'}")
            else:
                log("ASR: no speech detected")
                transcript = None

    tag_lines: list = []
    if use_audio_tags and audio_present:
        wav32k = out_dir / "audio_32k.wav"
        extract_audio(video, wav32k, analysis_start, analysis_len, sample_rate=32000)
        try:
            tags = audio_tags(wav32k, tags_threshold, tags_max)
        except Exception as e:
            log(f"warning: audio tagging failed, skipping ({e})")
        else:
            if tags:
                tag_lines = [f"{conf:.3f}  {label}" for label, conf in tags]
                (out_dir / "audio_tags.txt").write_text("\n".join(tag_lines) + "\n",
                                                        encoding="utf-8")
                log(f"audio tags saved (top3: {', '.join(l.split('  ')[1] for l in tag_lines[:3])})")
            else:
                log("audio tags: none above threshold")

    frames = extract_frames(video, frames_dir, analysis_start, analysis_len,
                            fps=frame_fps, max_side=frame_max_side,
                            max_frames=max_frames)
    log(f"frames extracted: {len(frames)} ({frames_dir})")
    analysis = run_analysis(predictor, frames, transcript, tag_lines,
                            frame_fps, max_new_tokens)

    (out_dir / "analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")

    prompt_text = run_rewrite(predictor, mode, duration, analysis, guide_text,
                              transcript, tags=tag_lines, max_new_tokens=max_new_tokens,
                              out_dir=out_dir)
    alignment = build_alignment_line(mode, duration)
    if alignment:
        prompt_text = alignment + "\n\n" + prompt_text
        log(f"alignment line added: {alignment[:60]}...")
    (out_dir / "prompt.txt").write_text(prompt_text + "\n", encoding="utf-8")
    return prompt_text
