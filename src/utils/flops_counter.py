from fvcore.nn import FlopCountAnalysis
import torch
import torch.nn as nn
import contextlib, io


# def measure_flops(model, input_tensor):
#     x = input_tensor.clone().detach()
#     buf = io.StringIO()
#     with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
#         return FlopCountAnalysis(model, x).total()

def measure_flops(model, input_tensor):
    x = input_tensor.clone().detach()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            return FlopCountAnalysis(model, x).total()
        except Exception as e:
            print(f"FLOPS measurement failed: {e}")
            return 0.0


class VLMFlopsWrapper(nn.Module):
    """Wrapper to allow FLOPS counting for VLM models"""
    def __init__(self, model, dummy_input_ids, dummy_attention_mask):
        super().__init__()
        self.model = model
        self.register_buffer('dummy_input_ids', dummy_input_ids)
        self.register_buffer('dummy_attention_mask', dummy_attention_mask)
    
    def forward(self, images):
        return self.model(images, self.dummy_input_ids, self.dummy_attention_mask)


def measure_flops_vlm(model, images, input_ids, attention_mask):
    """
    Measure FLOPs for VLM models that need multiple inputs.
    """
    # Use single sample for FLOPS measurement
    dummy_ids = input_ids[:1].clone().detach()
    dummy_mask = attention_mask[:1].clone().detach()
    dummy_images = images[:1].clone().detach()
    
    wrapper = VLMFlopsWrapper(model, dummy_ids, dummy_mask)
    wrapper.eval()
    
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            return FlopCountAnalysis(wrapper, dummy_images).total()
        except Exception as e:
            print(f"FLOPS measurement failed: {e}")
            return 0.0