# ComfyUI-VideoAnalyzer

ComfyUI nodes that turn a video into a **MiniMax H3** video-generation prompt
(`T2VA` / `I2VA` / `FL2VA` / `L2VA`) or an **LTX-2.5** natural-language prompt,
by analyzing it with a vision-language model. Based on
[knishika62/video-analyzer](https://github.com/knishika62/video-analyzer),
ported to Windows and to ComfyUI's node graph.

**Fully self-contained**: the VLM runs inside ComfyUI via
[ComfyUI_VLM_nodes](https://github.com/gokayfem/ComfyUI_VLM_nodes) — no
llama-server / LM Studio / API keys. The model auto-downloads from HuggingFace
on first execution into `ComfyUI/models/LLavacheckpoints/` and its VRAM
residency is managed by ComfyUI's model manager (smart loading/offloading).

## Nodes

| Node | Inputs | Outputs |
|------|--------|---------|
| **VLM Model Loader (H3)** | model picker, custom_model_id, memory_mode, attention_mode | `VLM_MODEL` |
| **MiniMax H3 Video to Prompt** | `video` (VIDEO), `vlm` (VLM_MODEL), `mode`, `duration`, + advanced options | `STRING` ("Minimax H3 Prompt") |
| **Show H3 Prompt** | `prompt` (STRING) | displays the text |

Standard nodes are reused wherever possible: video input comes from the core
**Load Video** node (or any `VIDEO` source, e.g. Wan output), the VLM loader is
built on ComfyUI_VLM_nodes' `ModernVLMPredictor` (transformers, ComfyUI-managed
VRAM), and audio/frames are processed with system `ffmpeg`/`ffprobe`.

## Model

Default: `Qwen 2.5 VL 7B Instruct` (BF16, ~16 GB VRAM). Validated against all
17 Adobe Premiere sample clips. Smaller/faster picks from the loader's dropdown
(e.g. `Qwen 3 VL 4B Instruct`, `Qwen 3 VL 2B Instruct`, `SmolVLM2 2.2B Video`)
auto-download the same way; a `Custom Hugging Face model` field accepts any
image/video-to-text repo id. For cards under 16 GB VRAM use the 4-bit NF4
memory mode or a smaller catalog model.

## How it works (two passes, one loaded model)

1. **Prep (ffmpeg)**: fade-in/out (black) detection trims the clip to its
   content range, then up to `max_frames` jpg frames are sampled at `frame_fps`
   (bounded by `frame_max_side`), and keyframes (`first`/`last`, original
   resolution) are extracted for the I2VA/FL2VA/L2VA modes.
2. **Audio**: if the video has audio, it is transcribed with faster-whisper
   (`use_asr`) and tagged with PANNs/AudioSet (`use_audio_tags`); transcript and
   tags feed both passes.
3. **Pass 1 - visual analysis**: all sampled frames go to the VLM in one call
   through the model's native video pathway (timestamps included), returning a
   structured shot-by-shot JSON (saved as `analysis.json`).
4. **Pass 2 - prompt rewrite**: analysis JSON + the full MiniMax H3 prompt guide
   (`md/minimax-h3-prompt-guide-base.md`) → the three H3 core fields
   (`integrated_multimodal_description` / `overall_soundscape` /
   `non_diegetic_music`), with output validation, field-label repair and a
   corrective retry. Keyframe modes get the fixed image-alignment line added by
   code. LTX mode uses the LTX guide and outputs one natural-language prompt.

Intermediate artifacts (`analysis.json`, `prompt.txt`, `frames/`,
`keyframes/first.jpg|last.jpg`, `transcript.txt`, `audio_tags.txt`, failure
dumps) are written to `<ComfyUI temp>/h3_video2prompt/` for inspection.

## Setup

```bat
:: 1) the VLM node pack (in-process model loading + auto-download)
git clone https://github.com/gokayfem/ComfyUI_VLM_nodes ComfyUI\custom_nodes\ComfyUI_VLM_nodes

:: 2) dependencies for both packs (portable ComfyUI; does not touch ComfyUI's torch)
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI_VLM_nodes\requirements.txt
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-VideoAnalyzer\requirements.txt
```

Requires `ffmpeg`/`ffprobe` in PATH. First execution of the VLM loader
downloads the selected model (Qwen2.5-VL-7B ≈ 16 GB) — watch the console.

## Workflow

Load `workflows/h3_video2prompt_sample.json` in ComfyUI:
`Load Video -> MiniMax H3 Video to Prompt -> Show H3 Prompt`, with the
VLM Model Loader feeding the analyzer node.

## Tests

```bat
:: direct pipeline test over all sample videos (loads the VLM in-process)
python_embeded\python.exe ComfyUI\custom_nodes\ComfyUI-VideoAnalyzer\test\test_e2e.py
::    options: --videos 8,9  --mode FL2VA|I2VA|L2VA|LTX|T2VA  --model <catalog label>

:: full-stack test through a running ComfyUI server (port 8188)
python_embeded\python.exe ComfyUI\custom_nodes\ComfyUI-VideoAnalyzer\test\test_workflow_api.py --clip 8
```

Results (Adobe Premiere 26.0 sample media, 17 clips, RTX 5090, in-process
Qwen2.5-VL-7B): **17/17 T2VA**, FL2VA x2 / I2VA x1 / LTX x1 spot checks, and
ComfyUI `/prompt` API integration PASS.

## Credits

- Pipeline and prompt guides: https://github.com/knishika62/video-analyzer
- H3 prompt spec: https://github.com/MiniMax-AI/MiniMax-H3
- In-process VLM loading: https://github.com/gokayfem/ComfyUI_VLM_nodes
