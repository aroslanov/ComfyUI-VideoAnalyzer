# nodes.py - ComfyUI nodes for the video analyzer (MiniMax H3 prompt generation).
#
# The VLM runs in-process via ComfyUI_VLM_nodes (https://github.com/gokayfem/ComfyUI_VLM_nodes):
# models auto-download from HuggingFace into ComfyUI's model directory on first use
# and are registered with ComfyUI's model manager (smart VRAM residency/offloading).
# No external services are used.
import re
import sys
import threading
from pathlib import Path

import folder_paths
from comfy_api.latest import io, ui, InputImpl, Types

from . import pipeline
from .pipeline import DEFAULT_ASR_MODEL, DEFAULT_TAG_THRESHOLD, DEFAULT_TAG_MAX

# Make the sibling VLM node pack importable
_VLM_PACK_PARENT = Path(__file__).resolve().parent.parent
if str(_VLM_PACK_PARENT) not in sys.path:
    sys.path.insert(0, str(_VLM_PACK_PARENT))

try:
    from ComfyUI_VLM_nodes.nodes.modern_vlm import (
        LEGACY_MODEL_LABELS,
        MODEL_CATALOG,
        ModernVLMPredictor,
        RECOMMENDED_MODEL_LABELS,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "ComfyUI-VideoAnalyzer requires the ComfyUI_VLM_nodes pack: "
        "https://github.com/gokayfem/ComfyUI_VLM_nodes "
        "(clone it into custom_nodes and install its requirements.txt)"
    ) from exc

_MODEL_LABELS = tuple(RECOMMENDED_MODEL_LABELS) + tuple(
    l for l in LEGACY_MODEL_LABELS if "Qwen 2.5 VL" in l
)
_DEFAULT_MODEL = "Qwen 2.5 VL 7B Instruct (legacy workflows)"
_MEMORY_MODES = ("ComfyUI managed (BF16)", "4-bit NF4 (bitsandbytes)",
                 "8-bit (bitsandbytes)", "CPU")
_ATTENTION_MODES = ("Auto (SDPA)", "Flash Attention 2", "Eager")

# module-level cache: ComfyUI locks v3 node classes against attribute mutation
_VLM_CACHE = {"lock": threading.RLock(), "key": None, "handle": None}


class H3VLMModelLoader(io.ComfyNode):
    """Loads a vision-language model in-process with automatic HuggingFace download."""

    @classmethod
    def _get_or_create(cls, key, factory):
        with _VLM_CACHE["lock"]:
            if _VLM_CACHE["handle"] is None or _VLM_CACHE["key"] != key:
                if _VLM_CACHE["handle"] is not None:
                    try:
                        _VLM_CACHE["handle"].close()
                    except Exception:
                        pass
                _VLM_CACHE["handle"] = factory()
                _VLM_CACHE["key"] = key
            return _VLM_CACHE["handle"]

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3VLMModelLoader",
            search_aliases=["vlm loader", "vl model loader", "vision model loader",
                            "llm loader", "qwen vl"],
            display_name="VLM Model Loader (H3)",
            description="Vision-language model for video analysis. Runs in-process via "
                        "ComfyUI_VLM_nodes; the model auto-downloads from HuggingFace on "
                        "first execution and its VRAM is managed by ComfyUI.",
            category="video_analyzer",
            essentials_category="Loaders",
            inputs=[
                io.Combo.Input("model", options=list(_MODEL_LABELS), default=_DEFAULT_MODEL,
                               tooltip="HuggingFace model, downloaded automatically on first use."),
                io.String.Input("custom_model_id", default="", optional=True,
                                tooltip="Only for 'Custom Hugging Face model': repo id like "
                                        "Qwen/Qwen2.5-VL-7B-Instruct."),
                io.Combo.Input("memory_mode", options=list(_MEMORY_MODES),
                               default="ComfyUI managed (BF16)", advanced=True,
                               tooltip="ComfyUI managed keeps the model resident with smart "
                                       "offloading; 4/8-bit reduce VRAM; CPU avoids VRAM entirely."),
                io.Combo.Input("attention_mode", options=list(_ATTENTION_MODES),
                               default="Auto (SDPA)", advanced=True),
            ],
            outputs=[io.Custom("VLM_MODEL").Output(display_name="vlm")],
        )

    @classmethod
    def execute(cls, model, custom_model_id, memory_mode, attention_mode) -> io.NodeOutput:
        effective_custom_id = custom_model_id.strip() if model == "Custom Hugging Face model" else ""
        if model == "Custom Hugging Face model" and not effective_custom_id:
            raise ValueError("custom_model_id is required for 'Custom Hugging Face model'")
        if model not in MODEL_CATALOG:
            raise ValueError(f"Unsupported model {model!r}")
        key = (model, effective_custom_id, memory_mode, attention_mode)
        predictor = cls._get_or_create(
            key,
            lambda: ModernVLMPredictor(model, effective_custom_id, memory_mode, attention_mode),
        )
        return io.NodeOutput({"kind": "local", "predictor": predictor})


class MiniMaxH3VideoToPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3VideoToPrompt",
            search_aliases=["video analyzer", "video2prompt", "h3 prompt", "hailuo",
                            "reverse prompt", "video to text", "video caption"],
            display_name="MiniMax H3 Video to Prompt",
            description="Analyzes a video with the connected vision-language model "
                        "(two passes: visual analysis, then prompt rewrite) and outputs a "
                        "MiniMax H3 prompt (T2VA / I2VA / FL2VA / L2VA) or an LTX-2.5 "
                        "natural-language prompt. Based on knishika62/video-analyzer.",
            category="video_analyzer",
            essentials_category="Video Tools",
            is_output_node=True,
            inputs=[
                io.Video.Input("video", tooltip="Video to analyze (e.g. from Load Video)."),
                io.Custom("VLM_MODEL").Input("vlm", tooltip="Vision-language model from VLM Model Loader (H3)."),
                io.Combo.Input("mode", options=["T2VA", "I2VA", "FL2VA", "L2VA", "LTX"], default="T2VA",
                               tooltip="H3: T2VA=text only, I2VA=first frame, FL2VA=first+last frame, "
                                       "L2VA=last frame. LTX=LTX-2.5 natural-language prompt."),
                io.Float.Input("duration", default=0.0, min=0.0, max=20.0, step=1.0, optional=True,
                               tooltip="Target video duration in seconds. 0 = clamp automatically "
                                       "(H3: 4-15s, LTX: snaps to 6-20s)."),
                io.Float.Input("max_seconds", default=0.0, min=0.0, max=600.0, step=1.0, advanced=True,
                               optional=True, tooltip="Analyze only the first N seconds. 0 = whole video."),
                io.Boolean.Input("keep_fade", default=False, advanced=True,
                                 tooltip="Skip leading/trailing fade (black) detection and trimming."),
                io.Float.Input("frame_fps", default=1.0, min=0.1, max=4.0, step=0.1, advanced=True,
                               tooltip="Frames sampled per second for the analysis."),
                io.Int.Input("max_frames", default=24, min=1, max=48, advanced=True,
                             tooltip="Max number of frames sampled from the video."),
                io.Int.Input("frame_max_side", default=512, min=64, max=1280, step=16, advanced=True,
                             tooltip="Max side of each sampled frame before the model's own processing."),
                io.Boolean.Input("use_asr", default=True,
                                 tooltip="Transcribe the audio with faster-whisper for dialogue/lyrics."),
                io.String.Input("asr_model", default=DEFAULT_ASR_MODEL, advanced=True,
                                tooltip="faster-whisper model name (tiny/base/small/medium/large-v3-turbo "
                                        "or a HuggingFace repo id)."),
                io.Boolean.Input("use_audio_tags", default=True,
                                 tooltip="Tag the audio with PANNs/AudioSet (requires panns-inference)."),
                io.Float.Input("tags_threshold", default=DEFAULT_TAG_THRESHOLD, min=0.0, max=1.0,
                               step=0.01, advanced=True),
                io.Int.Input("tags_max", default=DEFAULT_TAG_MAX, min=1, max=32, advanced=True),
                io.Int.Input("max_new_tokens", default=8192, min=256, max=16384, step=256, advanced=True,
                             tooltip="Generation budget per LLM pass."),
            ],
            outputs=[io.String.Output(display_name="Minimax H3 Prompt")],
        )

    @classmethod
    def execute(cls, video: io.Video.Type, vlm: dict, mode: str, duration: float,
                max_seconds: float, keep_fade: bool, frame_fps: float, max_frames: int,
                frame_max_side: int, use_asr: bool, asr_model: str, use_audio_tags: bool,
                tags_threshold: float, tags_max: int, max_new_tokens: int) -> io.NodeOutput:
        if not isinstance(vlm, dict) or vlm.get("kind") != "local":
            raise ValueError("vlm input must come from VLM Model Loader (H3)")
        work_dir = Path(folder_paths.get_temp_directory()) / "h3_video2prompt" / cls.__name__
        work_dir.mkdir(parents=True, exist_ok=True)
        # ponytail: fixed source.mp4 name; concurrent queues of this node would overwrite
        source = work_dir / "source.mp4"
        # Materialize the VIDEO input (keeps the audio track) for the ffmpeg pipeline
        video.save_to(str(source), format=Types.VideoContainer.MP4, codec=Types.VideoCodec.AUTO)
        prompt = pipeline.video_to_prompt(
            str(source), work_dir, vlm=vlm,
            mode=mode,
            duration=duration if duration > 0 else None,
            max_seconds=max_seconds if max_seconds > 0 else None,
            keep_fade=keep_fade,
            frame_fps=frame_fps, max_frames=max_frames, frame_max_side=frame_max_side,
            use_asr=use_asr, asr_model=asr_model,
            use_audio_tags=use_audio_tags,
            tags_threshold=tags_threshold, tags_max=tags_max,
            max_new_tokens=max_new_tokens,
        )
        return io.NodeOutput(prompt, ui=ui.PreviewText(prompt))


class ShowH3Prompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ShowH3Prompt",
            search_aliases=["show text", "display prompt"],
            display_name="Show H3 Prompt",
            category="video_analyzer",
            is_output_node=True,
            inputs=[io.String.Input("prompt", force_input=True)],
            outputs=[io.String.Output()],
        )

    @classmethod
    def execute(cls, prompt: str) -> io.NodeOutput:
        return io.NodeOutput(prompt, ui=ui.PreviewText(prompt))
