import torch
import torch.nn as nn
import torch.nn.functional as F
from types import MethodType
from transformers import ModernBertForMaskedLM
from config import batch_size_number


class ModernBert(ModernBertForMaskedLM):
    def __init__(self, config, attention_module_cls):
        super().__init__(config)

        for i, layer in enumerate(self.model.layers):
            # Some attention modules require a layer_id argument; fall back to config-only if not
            try:
                attn_mod = attention_module_cls(config, layer_id=i)
            except TypeError:
                attn_mod = attention_module_cls(config)

            orig_forward = attn_mod.forward

            def compatible_forward(self, hidden_states, *args, _orig_forward=orig_forward, **kwargs):
                if hidden_states.dim() == 2:
                    total, embed_dim = hidden_states.size()
                    hidden_states = hidden_states.view(batch_size_number, total, embed_dim)

                    if total < 512:
                        hidden_states = F.pad(hidden_states, (0, 0, 0, 512 - total, 0, 0))

                def _to_additive_mask(mask, K, dtype):
                    if mask is None:
                        return None
                    if mask.dim() == 2:
                        mask = mask[:, None, None, :]
                    elif mask.dim() == 3 and mask.size(1) == 1:
                        mask = mask[:, None, :, :]
                    elif mask.dim() == 4 and not (mask.size(1) == 1 and mask.size(2) == 1):
                        mask = mask.max(dim=2, keepdim=True).values

                    cur = mask.size(-1)
                    if cur > K:
                        mask = mask[..., -K:]
                    elif cur < K:
                        mask = F.pad(mask, (K - cur, 0), value=0)

                    mask = mask.to(dtype=dtype)
                    if mask.dtype == torch.bool or (mask.dtype.is_floating_point and mask.max() <= 1):
                        mask = (1.0 - mask) * torch.finfo(dtype).min
                    return mask

                def _latent_causal_mask(B, L, K, dtype, device):
                    return torch.zeros((B, 1, L, K), dtype=dtype, device=device)

                is_latent_attn = getattr(self, "__class__", None).__name__ == "MultiHeadLatentAttention"
                if is_latent_attn:
                    assert hidden_states.dim() == 3, "expect [B,S,D]"
                    B, S, D = hidden_states.shape
                    dtype = hidden_states.dtype
                    device = hidden_states.device

                    L = getattr(self, "num_latents", S)
                    attn_mask_in = kwargs.get("attention_mask", None)
                    K = attn_mask_in.shape[-1] if (attn_mask_in is not None and attn_mask_in.dim() >= 2) else S

                    if attn_mask_in is not None:
                        kwargs["attention_mask"] = _to_additive_mask(attn_mask_in, K, dtype)

                    if "causal_mask" in kwargs and kwargs["causal_mask"] is not None:
                        kwargs["causal_mask"] = _latent_causal_mask(B, L, K, dtype, device)

                return _orig_forward(hidden_states, *args, **kwargs)

            attn_mod.forward = MethodType(compatible_forward, attn_mod)
            layer.attn = attn_mod

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        if labels is not None:
            return outputs.loss, outputs.logits
        return outputs
