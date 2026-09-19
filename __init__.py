# ComfyUI-VideoAnalyzer
# Video -> MiniMax H3 / LTX-2.5 prompt generation nodes for ComfyUI.
# Based on https://github.com/knishika62/video-analyzer
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

from .nodes import H3VLMModelLoader, MiniMaxH3VideoToPrompt

NODE_CLASS_MAPPINGS = {
    "H3VLMModelLoader": H3VLMModelLoader,
    "MiniMaxH3VideoToPrompt": MiniMaxH3VideoToPrompt,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3VLMModelLoader": "VLM Model Loader (H3)",
    "MiniMaxH3VideoToPrompt": "MiniMax H3 Video to Prompt",
}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]


class VideoAnalyzerExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [H3VLMModelLoader, MiniMaxH3VideoToPrompt]


async def comfy_entrypoint() -> ComfyExtension:
    return VideoAnalyzerExtension()
