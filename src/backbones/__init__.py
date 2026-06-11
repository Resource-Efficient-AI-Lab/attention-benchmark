from .gpt2_lm import GPT2LM
from .modern_bert import ModernBert
from .vision_transformer import VisionTransformer
from .nano_vlm import NanoVLM
from .qwen_lm import QwenLM



# def __getattr__(name):
#     if name == "GPT2LM":
#         from .gpt2_lm import GPT2LM
#         return GPT2LM
#     if name == "ModernBert":
#         from .modern_bert import ModernBert
#         return ModernBert
#     if name == "VisionTransformer":
#         from .vision_transformer import VisionTransformer
#         return VisionTransformer
#     if name == "NanoVLM":
#         from .nano_vlm import NanoVLM
#         return NanoVLM
#     if name == "QwenLM":
#         from .qwen_lm import QwenLM
#         return QwenLM
#     raise AttributeError(f"module 'backbones' has no attribute {name!r}")