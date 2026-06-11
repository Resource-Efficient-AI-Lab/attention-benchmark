import timm
import torch
import torch.nn as nn


class VisionTransformer(nn.Module):
    def __init__(self, model_name, attention_module_cls, adapted_config):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        self.attn_module_class_name = attention_module_cls.__name__

        # Replace attention module in each transformer block with the custom implementation
        for block in self.model.blocks:
            old_attn = block.attn
            new_attn = attention_module_cls(adapted_config)

            # Copy pretrained qkv and projection weights into the new attention module
            if hasattr(new_attn, 'qkv') and hasattr(old_attn, 'qkv'):
                with torch.no_grad():
                    new_attn.qkv.weight.data = old_attn.qkv.weight.data.clone()
                    if hasattr(new_attn.qkv, 'bias') and old_attn.qkv.bias is not None:
                        new_attn.qkv.bias.data = old_attn.qkv.bias.data.clone()

            if hasattr(new_attn, 'proj') and hasattr(old_attn, 'proj'):
                with torch.no_grad():
                    new_attn.proj.weight.data = old_attn.proj.weight.data.clone()
                    if hasattr(new_attn.proj, 'bias') and old_attn.proj.bias is not None:
                        new_attn.proj.bias.data = old_attn.proj.bias.data.clone()

            block.attn = new_attn

    def forward(self, images):
        return self.model(images)
