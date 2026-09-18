# nodes.py - ComfyUI nodes for the video analyzer (MiniMax H3 prompt generation).
import re
from pathlib import Path

import folder_paths
from comfy_api.latest import io, ui, InputImpl, Types

from . import pipeline
from .pipeline import DEFAULT_ASR_MODEL, DEFAULT_TAG_THRESHOLD, DEFAULT_TAG_MAX

# Local source name of the analyzed video, sanitized for filesystem use
_SANITIZE = re.compile(r"[^A-Za-z0-9_-]+")


class H3VLMModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3VLMModelLoader",
            search_aliases=["vlm loader", "llm loader", "vision model loader", "vl model loader"],
            display_name="VLM Model Loader (H3)",
            description="OpenAI-compatible vision LLM endpoint used for video analysis "
                        "(llama.cpp server, LM Studio, vLLM, or any OpenAI-compatible API).",
            category="video_analyzer",
            essentials_category="Loaders",
            inputs=[
                io.String.Input("api_base", default="http://127.0.0.1:8080/v1",
                                tooltip="OpenAI-compatible API base URL."),
                io.String.Input("model", default="qwen2.5-vl-3b-instruct",
                                tooltip="Model name sent to the endpoint."),
                io.String.Input("api_key", default="", optional=True, advanced=True,
                                tooltip="Bearer token, only needed for endpoints that require one."),
                io.Float.Input("temperature", default=0.2, min=0.0, max=2.0, step=0.01, advanced=True),
            ],
            outputs=[io.Custom("VLM_MODEL").Output(display_name="vlm")],
        )

    @classmethod
    def execute(cls, api_base, model, api_key, temperature) -> io.NodeOutput:
        return io.NodeOutput({"api_base": api_base, "model": model,
                              "api_key": api_key, "temperature": temperature})


class MiniMaxH3VideoToPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3VideoToPrompt",
            search_aliases=["video analyzer", "video2prompt", "h3 prompt", "hailuo",
                            "reverse prompt", "video to text", "video caption"],
            display_name="MiniMax H3 Video to Prompt",
            description="Analyzes a video with a vision LLM (two passes: visual analysis, "
                        "then prompt rewrite) and outputs a MiniMax H3 prompt "
                        "(T2VA / I2VA / FL2VA / L2VA) or an LTX-2.5 natural-language prompt. "
                        "Based on knishika62/video-analyzer.",
            category="video_analyzer",
            essentials_category="Video Tools",
            is_output_node=True,
            inputs=[
                io.Video.Input("video", tooltip="Video to analyze (e.g. from Load Video)."),
                io.Custom("VLM_MODEL").Input("vlm", tooltip="Vision LLM from VLM Model Loader (H3)."),
                io.Combo.Input("mode", options=["T2VA", "I2VA", "FL2VA", "L2VA", "LTX"], default="T2VA",
                               tooltip="H3: T2VA=text only, I2VA=first frame, FL2VA=first+last frame, "
                                       "L2VA=last frame. LTX=LTX-2.5 natural-language prompt."),
                io.Float.Input("duration", default=0.0, min=0.0, max=20.0, step=1.0, optional=True,
                               tooltip="Target video duration in seconds. 0 = clamp automatically "
                                       "(H3: 4-15s, LTX: snaps to 6-20s)."),
                io.Combo.Input("analysis_input", options=["frames", "video"], default="frames", advanced=True,
                               tooltip="Pass 1 input: 'frames' sends sampled jpg frames (works with "
                                       "llama.cpp/LM Studio); 'video' sends a downscaled mp4 (needs a "
                                       "video-vision endpoint such as vLLM)."),
                io.Int.Input("max_side", default=480, min=64, max=1280, step=16, advanced=True,
                             tooltip="[video mode] max side of the downscaled analysis mp4."),
                io.Float.Input("max_seconds", default=0.0, min=0.0, max=600.0, step=1.0, advanced=True,
                               optional=True, tooltip="Analyze only the first N seconds. 0 = whole video."),
                io.Boolean.Input("keep_fade", default=False, advanced=True,
                                 tooltip="Skip leading/trailing fade (black) detection and trimming."),
                io.Float.Input("frame_fps", default=1.0, min=0.1, max=4.0, step=0.1, advanced=True,
                               tooltip="[frames mode] frames sampled per second."),
                io.Int.Input("max_frames", default=32, min=1, max=64, advanced=True,
                             tooltip="[frames mode] max number of frames sent to the LLM."),
                io.Int.Input("frame_max_side", default=768, min=64, max=1280, step=16, advanced=True,
                             tooltip="[frames mode] max side of each sampled frame."),
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
            ],
            outputs=[io.String.Output(display_name="Minimax H3 Prompt")],
        )

    @classmethod
    def execute(cls, video: io.Video.Type, vlm: dict, mode: str, duration: float,
                analysis_input: str, max_side: int, max_seconds: float, keep_fade: bool,
                frame_fps: float, max_frames: int, frame_max_side: int, use_asr: bool,
                asr_model: str, use_audio_tags: bool, tags_threshold: float, tags_max: int) -> io.NodeOutput:
        if not isinstance(vlm, dict) or "api_base" not in vlm:
            raise ValueError("vlm input must come from VLM Model Loader (H3)")
        work_dir = Path(folder_paths.get_temp_directory()) / "h3_video2prompt" / cls.__name__
        work_dir.mkdir(parents=True, exist_ok=True)
        # ponytail: fixed source.mp4 name; concurrent queues of this node would overwrite
        source = work_dir / "source.mp4"
        # Materialize the VIDEO input (keeps the audio track) for the ffmpeg pipeline
        video.save_to(str(source), format=Types.VideoContainer.MP4, codec=Types.VideoCodec.AUTO)
        prompt = pipeline.video_to_prompt(
            str(source), work_dir,
            api_base=vlm["api_base"], model=vlm["model"], api_key=vlm.get("api_key", ""),
            mode=mode,
            duration=duration if duration > 0 else None,
            analysis_input=analysis_input,
            max_side=max_side,
            max_seconds=max_seconds if max_seconds > 0 else None,
            keep_fade=keep_fade,
            frame_fps=frame_fps, max_frames=max_frames, frame_max_side=frame_max_side,
            use_asr=use_asr, asr_model=asr_model,
            use_audio_tags=use_audio_tags,
            tags_threshold=tags_threshold, tags_max=tags_max,
            temperature=vlm.get("temperature", 0.2),
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
