# pipeline.py - video -> MiniMax H3 / LTX-2.5 prompt pipeline for ComfyUI.
# Ported from https://github.com/knishika62/video-analyzer (h3_video2prompt.py +
# h3_video2prompt_frames.py). Changes for ComfyUI/Windows:
#   - die() -> H3PipelineError (ComfyUI shows it on the node)
#   - ASR: mlx-whisper (Apple Silicon only) -> faster-whisper (CPU int8)
#   - PANNs audio tagging: subprocess isolation (for mlx/numba clashes) -> in-process
#   - CLI main() -> video_to_prompt() function driven by node inputs

import base64
import json
import os
import re
import shutil
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import requests

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


class H3PipelineError(RuntimeError):
    pass


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


def make_analysis_video(src: Path, dst: Path, max_side: int,
                        start: float = 0.0, length: float | None = None,
                        info: dict | None = None) -> None:
    """[start, start+length] trimmed to max_side, silent h264 mp4 for LLM video_url mode."""
    info = info or probe_video(src)
    w, h = info["width"], info["height"]
    cmd = ["ffmpeg", "-y", "-v", "error"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(src)]
    if length:
        cmd += ["-t", f"{float(length):.3f}"]
    cmd += ["-an"]

    if max(w, h) > max_side:
        if w >= h:
            scale = f"scale={max_side}:-2"
        else:
            scale = f"scale=-2:{max_side}"
        cmd += ["-vf", scale]
        log(f"downscale: {w}x{h} -> max side {max_side}")

    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", str(dst)]
    run_cmd(cmd, "analysis video creation")


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
# LLM
# ---------------------------------------------------------------------------

class LLMClient:
    def __init__(self, api_base: str, model: str, temperature: float = 0.2, api_key: str = ""):
        self.url = api_base.rstrip("/") + "/chat/completions"
        self.model = model
        self.temperature = temperature
        self.api_key = api_key

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def chat(self, messages: list, max_tokens: int, retries: int = 2,
             timeout: tuple = (15, 900)) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }
        last_err = None
        for attempt in range(1, retries + 1):
            try:
                log(f"LLM call... (attempt {attempt}/{retries})")
                t0 = time.time()
                resp = requests.post(self.url, json=payload, headers=self._headers(), timeout=timeout)
                if 400 <= resp.status_code < 500 and resp.status_code != 429:
                    die(f"LLM endpoint returned HTTP {resp.status_code} (no retry):\n{resp.text[:500]}")
                if 500 <= resp.status_code < 600:
                    raise requests.HTTPError(f"HTTP {resp.status_code}: {resp.text[:500]}")
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                log(f"LLM response: {time.time() - t0:.1f} s, {len(content)} chars")
                return content
            except (requests.RequestException, KeyError, json.JSONDecodeError) as e:
                last_err = e
                log(f"response error: {e}")
                if attempt < retries:
                    time.sleep(3)
        die(f"LLM call failed {retries} times: {last_err}")


def extract_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("JSON block not found")
        text = text[start:end + 1]
    return json.loads(text)


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
    log(f"ASR done: {time.time() - t0:.1f} s (language={info.language})")
    return {
        "segments": [{"start": s.start, "end": s.end, "text": s.text} for s in segments],
        "text": "".join(s.text for s in segments),
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
# Pass 1 input builders (video_url mode / frames mode)
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
        die("frame extraction produced 0 frames. Check --frame-fps / --max-frames")
    return [(p, start + i / fps) for i, p in enumerate(paths)]


def run_analysis(client: LLMClient, video_b64: str, transcript: str | None,
                 tags: list | None = None) -> dict:
    """Pass 1 with the downscaled mp4 as video_url (needs a video-vision endpoint, e.g. vLLM)."""
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
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "video_url",
             "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}},
        ],
    }]
    raw = client.chat(messages, max_tokens=8192)
    try:
        return extract_json(raw)
    except (ValueError, json.JSONDecodeError) as e:
        log(f"JSON parse failed ({e}). Asking the LLM to fix it...")
        retry_messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user",
             "content": "The output above is not valid JSON. Re-output only valid JSON following the schema. No code fences."},
        ]
        raw2 = client.chat(retry_messages, max_tokens=8192)
        return extract_json(raw2)


def run_analysis_frames(client: LLMClient, frames: list, transcript: str | None,
                        tags: list | None = None) -> dict:
    """Pass 1 with timestamped jpg frames as image_url array (works with llama.cpp/LM Studio)."""
    text = ANALYSIS_PROMPT
    text += (
        "\n\n[Note on input format]\n"
        f"Instead of the video itself, {len(frames)} frame images sampled chronologically are attached. "
        "Each image is preceded by a \"Frame at T.Ts\" heading giving its absolute seconds in the video. "
        "Use these timestamps as the clue for shot boundaries and time_range (do not interpolate content "
        "between frames; base your analysis only on what is visible in the frames)."
    )
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

    content: list = [{"type": "text", "text": text}]
    for path, t in frames:
        b64 = base64.b64encode(path.read_bytes()).decode()
        content.append({"type": "text", "text": f"Frame at {t:.2f}s"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    messages = [{"role": "user", "content": content}]
    raw = client.chat(messages, max_tokens=8192)
    try:
        return extract_json(raw)
    except (ValueError, json.JSONDecodeError) as e:
        log(f"JSON parse failed ({e}). Asking the LLM to fix it...")
        retry_messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user",
             "content": "The output above is not valid JSON. Re-output only valid JSON following the schema. No code fences."},
        ]
        raw2 = client.chat(retry_messages, max_tokens=8192)
        return extract_json(raw2)


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
                         tags: list | None = None) -> str:
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
        mode_rule = (
            "Mode: T2VA (text only). Output the three core fields in this exact order:\n"
            "1. integrated_multimodal_description: [Shot 1] ...\n"
            "2. overall_soundscape: ...\n"
            "3. non_diegetic_music: ...\n"
            "There is no image-alignment line in T2VA; start directly with integrated_multimodal_description."
        )
    elif mode == "I2VA":
        mode_rule = (
            "Mode: I2VA. The user will supply <Picture 1> as the actual first frame of the target video.\n"
            "Output ONLY the three core fields (integrated_multimodal_description / overall_soundscape / non_diegetic_music) in that order, "
            "separated by one blank line. Do NOT output the image-alignment instruction line — the tool adds it automatically.\n"
            "Content rule: Shot 1 must start from the state shown in <Picture 1> (establish style, subjects, composition, scene anchors first) and develop forward. "
            "Keep character identity, clothing, colors, key objects, and spatial relations consistent with the reference."
        )
    elif mode == "FL2VA":
        mode_rule = (
            "Mode: FL2VA. The user will supply Picture 1 (first frame) and Picture 2 (last frame).\n"
            "Output ONLY the three core fields in that order. Do NOT output the image-alignment instruction line — the tool adds it automatically.\n"
            "Content rule: write a single continuous shot describing the motion path from Picture 1 to Picture 2: "
            "first-frame state -> observable intermediate changes -> progressively narrowing differences -> last-frame state. "
            "Do not repeat two static image descriptions; supply the connecting motion. The last frame must be reached at the end of the shot."
        )
    else:  # L2VA
        mode_rule = (
            "Mode: L2VA. The user will supply <Picture 1> as the actual LAST frame of the target video.\n"
            "Output ONLY the three core fields in that order. Do NOT output the image-alignment instruction line — the tool adds it automatically.\n"
            "Content rule: write a single shot that infers a plausible preceding state, then lets actions, object states, and composition "
            "gradually converge and land exactly on <Picture 1> in the final moment: "
            "plausible preceding state -> explicit action and transition path -> gradual convergence -> last-frame landing."
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
1. Write ONLY the three core fields, in English, separated by one blank line, in the order required by the mode rule above.
2. No markdown fences, no commentary, no explanations before or after the fields.
3. Preserve the exact field names: integrated_multimodal_description, overall_soundscape, non_diegetic_music.
4. Never output the image-alignment instruction line ("For the target video..." or "How the reference pictures align...") — the tool adds it.
5. For keyframe modes, reference <Picture 1>/Picture 2 in the shot descriptions.
6. Dialogue inside <d>[Language] ... </d> must keep the original language verbatim.
7. If there is no dialogue, no singing, and no on-screen speaking source, do not invent any <d> blocks.
8. overall_soundscape: 1-4 sentences; non_diegetic_music: 1-3 sentences or "N/A".
9. For overall_soundscape, infer plausible ambient/physical/non-verbal sounds from the visual content (weather, locations, actions, crowds). Use "N/A" only when the scene truly has no plausible sound source.
"""


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


def run_rewrite(client: LLMClient, mode: str, duration: float, analysis: dict,
                guide_text: str, transcript: str | None,
                tags: list | None = None) -> str:
    prompt = build_rewrite_prompt(mode, duration, analysis, guide_text, transcript,
                                  tags=tags)
    messages = [{"role": "user", "content": prompt}]
    raw = client.chat(messages, max_tokens=8192)
    m = re.search(r"```(?:text|prompt)?\s*(.*?)\s*```", raw, re.DOTALL)
    if m:
        raw = m.group(1)
    raw = raw.strip()
    if mode == "LTX":
        if len(raw.split()) < 20:
            die(f"LLM output too short for an LTX prompt. Re-run the node.\n---\n{raw[:800]}")
        quoted = (re.findall(r'"([^"]*)"', raw)
                  + re.findall(r"'([^']*)'", raw)
                  + re.findall(r"「([^」]*)」", raw))
        bad = [q for q in quoted if re.search(r"[\u30A0-\u30FF\u3400-\u4DBF\u4E00-\u9FFF]", q)]
        if bad:
            log(f"warning: LTX dialogue contains kanji/katakana (spec: hiragana only): {bad[:3]}")
        return raw
    raw = strip_alignment_line(raw)
    raw = strip_stray_tags(raw)
    fields = ("integrated_multimodal_description", "overall_soundscape",
              "non_diegetic_music")
    positions = []
    for field in fields:
        pos = raw.find(field + ":")
        if pos < 0:
            die(f"LLM output missing '{field}:'. Re-run the node.\n---\n{raw[:800]}")
        positions.append(pos)
    if positions != sorted(positions):
        die(f"LLM output has wrong 3-field order (pos={positions}). Re-run the node.\n---\n{raw[:800]}")
    raw = format_shot_breaks(raw)
    return raw


def resolve_duration(mode: str, requested: float | None, effective_len: float) -> float:
    if mode == "LTX":
        raw = round(effective_len) if requested is None else requested
        if not (LTX_MIN_DURATION <= raw <= LTX_MAX_DURATION):
            die(f"LTX mode --duration must be 6-20 s (got: {requested})")
        return float(min(LTX_VALID_DURATIONS, key=lambda v: abs(v - raw)))
    dur = min(H3_MAX_DURATION, max(H3_MIN_DURATION, round(effective_len))) if requested is None else requested
    if not (H3_MIN_DURATION <= dur <= H3_MAX_DURATION):
        die(f"duration must be {H3_MIN_DURATION}-{H3_MAX_DURATION} s (got: {requested})")
    return float(dur)


# ---------------------------------------------------------------------------
# main entry used by the ComfyUI node
# ---------------------------------------------------------------------------

def video_to_prompt(video_path: str, out_dir: Path,
                    api_base: str, model: str, api_key: str = "",
                    mode: str = "T2VA", duration: float | None = None,
                    analysis_input: str = "frames",
                    max_side: int = 480, max_seconds: float | None = None,
                    keep_fade: bool = False,
                    frame_fps: float = 1.0, max_frames: int = 32,
                    frame_max_side: int = 768,
                    use_asr: bool = True, asr_model: str = DEFAULT_ASR_MODEL,
                    asr_language: str | None = None,
                    use_audio_tags: bool = True,
                    tags_threshold: float = DEFAULT_TAG_THRESHOLD,
                    tags_max: int = DEFAULT_TAG_MAX,
                    temperature: float = 0.2) -> str:
    """Analyze a video file and return the final H3/LTX prompt. Writes
    analysis.json / prompt.txt (+ frames/, keyframes/, transcript.txt, audio_tags.txt)
    under out_dir for debugging."""
    require_tool("ffmpeg")
    require_tool("ffprobe")
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
            result = transcribe(wav, asr_model, asr_language)
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

    client = LLMClient(api_base, model, temperature, api_key=api_key)
    if analysis_input == "video":
        analysis_video = out_dir / "analysis_downscaled.mp4"
        make_analysis_video(video, analysis_video, max_side,
                            start=analysis_start, length=analysis_len, info=info)
        video_b64 = base64.b64encode(analysis_video.read_bytes()).decode()
        log(f"analysis video: {analysis_video.stat().st_size / 1e6:.2f} MB (base64 video_url)")
        analysis = run_analysis(client, video_b64, transcript, tags=tag_lines)
    else:
        frames = extract_frames(video, frames_dir, analysis_start, analysis_len,
                                fps=frame_fps, max_side=frame_max_side,
                                max_frames=max_frames)
        log(f"frames extracted: {len(frames)} ({frames_dir})")
        analysis = run_analysis_frames(client, frames, transcript, tags=tag_lines)

    (out_dir / "analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")

    prompt_text = run_rewrite(client, mode, duration, analysis, guide_text,
                              transcript, tags=tag_lines)
    alignment = build_alignment_line(mode, duration)
    if alignment:
        prompt_text = alignment + "\n\n" + prompt_text
        log(f"alignment line added: {alignment[:60]}...")
    (out_dir / "prompt.txt").write_text(prompt_text + "\n", encoding="utf-8")
    return prompt_text
