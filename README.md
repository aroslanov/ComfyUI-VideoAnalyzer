# ComfyUI-VideoAnalyzer

ComfyUI nodes that turn a video into a **MiniMax H3** video-generation prompt
(`T2VA` / `I2VA` / `FL2VA` / `L2VA`) or an **LTX-2.5** natural-language prompt,
by analyzing it with a vision LLM. Based on
[knishika62/video-analyzer](https://github.com/knishika62/video-analyzer),
ported to Windows and to ComfyUI's node graph.

## Nodes

| Node | Inputs | Outputs |
|------|--------|---------|
| **VLM Model Loader (H3)** | `api_base`, `model`, `api_key` (opt), `temperature` | `VLM_MODEL` |
| **MiniMax H3 Video to Prompt** | `video` (VIDEO), `vlm` (VLM_MODEL), `mode`, `duration`, + advanced options | `STRING` ("Minimax H3 Prompt") |
| **Show H3 Prompt** | `prompt` (STRING) | displays the text |

Standard nodes are reused wherever possible: video input comes from the core
**Load Video** node (or any `VIDEO` source, e.g. Wan output), audio/frames are
processed with system `ffmpeg`/`ffprobe`.

### How it works (two LLM passes)

1. **Pass 1 - visual analysis**: ffmpeg detects fades (black lead/trail), samples
   up to `max_frames` jpg frames, and sends them with timestamps to the VLM,
   which returns a structured shot-by-shot JSON. Frames are sent in chunks of 8
   (small local VLMs degrade on large multi-image prompts). Optionally the whole
   downscaled mp4 is sent instead (`analysis_input=video`, needs a video-vision
   endpoint such as vLLM).
2. **Audio**: if the video has audio, it is transcribed with faster-whisper
   (`use_asr`) and tagged with PANNs/AudioSet (`use_audio_tags`); transcript and
   tags are fed to both passes.
3. **Pass 2 - prompt rewrite**: analysis JSON + the full MiniMax H3 prompt guide
   (`md/minimax-h3-prompt-guide-base.md`) -> the three H3 core fields
   (`integrated_multimodal_description` / `overall_soundscape` /
   `non_diegetic_music`). Keyframe modes get the fixed image-alignment line added
   by code. LTX mode uses the LTX guide and outputs one natural-language prompt.

Intermediate artifacts (`analysis.json`, `prompt.txt`, `frames/`,
`keyframes/first.jpg|last.jpg`, `transcript.txt`, `audio_tags.txt`) are written
to `<ComfyUI temp>/h3_video2prompt/` for inspection.

## Setup

```bat
:: deps (portable ComfyUI)
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-VideoAnalyzer\requirements.txt
```

Requires `ffmpeg`/`ffprobe` in PATH.

### Local vision LLM (recommended: llama.cpp + Qwen2.5-VL-7B)

The loader accepts any OpenAI-compatible endpoint (llama.cpp server, LM Studio,
vLLM, cloud APIs). Tested locally with:

- **llama.cpp b6000** (`video_analyzer_tools\llamacpp_b6000`) - do NOT use
  nightly builds (b11029 has a Qwen2.5-VL multi-image regression that
  degenerates into garbage tokens mid-generation).
- **Qwen2.5-VL-7B-Instruct** GGUF Q4_K_M + f16 mmproj
  (`video_analyzer_tools\models`). The 3B model collapses into repeated-token
  garbage on some multi-image prompts (e.g. high-texture skatepark footage);
  the 7B handles all 17 Adobe sample clips.

```powershell
# start the local server (script in F:\ComfyUI\video_analyzer_tools)
.\start_llama_server.ps1        # serves http://127.0.0.1:8777/v1
```

Loader defaults match: `api_base=http://127.0.0.1:8777/v1`,
`model=qwen2.5-vl-7b-instruct`.

If you see repeated `degenerate output` errors, restart llama-server - a
collapsed generation can leave its KV cache in a bad state.

## Workflow

Load `workflows/h3_video2prompt_sample.json` in ComfyUI:
`Load Video -> MiniMax H3 Video to Prompt -> Show H3 Prompt`, with the
VLM Model Loader feeding the analyzer node.

## Tests

```bat
:: start llama-server first (see above), then:
:: 1) direct pipeline test over all sample videos
python_embeded\python.exe ComfyUI\custom_nodes\ComfyUI-VideoAnalyzer\test\test_e2e.py
::    options: --videos 8,9  --mode FL2VA|I2VA|L2VA|LTX|T2VA  --api-base ...

:: 2) full-stack test through a running ComfyUI server (port 8188)
python_embeded\python.exe ComfyUI\custom_nodes\ComfyUI-VideoAnalyzer\test\test_workflow_api.py --clip 8
```

Results (Adobe Premiere 26.0 sample media, 17 clips, RTX 5090, local 7B VLM):
17/17 T2VA, plus FL2VA x2, I2VA x1, LTX x1 spot checks, and ComfyUI `/prompt`
API integration PASS.

## Credits

- Pipeline and prompt guides: https://github.com/knishika62/video-analyzer
- H3 prompt spec: https://github.com/MiniMax-AI/MiniMax-H3
