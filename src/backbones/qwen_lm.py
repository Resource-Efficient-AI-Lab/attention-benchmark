import torch
import torch.nn as nn
import torch.nn.functional as F
from types import MethodType

try:
    from transformers import Qwen2ForCausalLM as _QWEN_BASE
except Exception:
    from transformers import QWenLMHeadModel as _QWEN_BASE


class QwenLM(_QWEN_BASE):
    def __init__(self, config, attention_module_cls):
        super().__init__(config)

        self._backbone = getattr(self, "model", None) or getattr(self, "transformer", None)
        if self._backbone is None:
            raise AttributeError("backbone not found (model/transformer)")

        self._layers = getattr(self._backbone, "layers", None) or getattr(self._backbone, "h", None)
        if self._layers is None:
            raise AttributeError("layers list not found (layers/h)")

        for i, layer in enumerate(self._layers):
            attn_attr = self._locate_attn_attr(layer)
            if attn_attr is None:
                continue

            try:
                attn_mod = attention_module_cls(config, layer_id=i)
            except TypeError:
                attn_mod = attention_module_cls(config)

            orig_forward = attn_mod.forward

            def compatible_forward(self_attn, hidden_states, attention_mask=None,
                                   *args, _orig_forward=orig_forward, **kwargs):
                assert hidden_states.dim() == 3, "expected [B,S,D]"
                B, S, D = hidden_states.shape
                dtype = hidden_states.dtype
                device = hidden_states.device

                # Keep kwargs clean: avoid positional/keyword collisions on embeddings/ids
                clean_kwargs = dict(kwargs)
                for k in ("position_embeddings", "position_ids"):
                    if k in clean_kwargs and any(
                        isinstance(a, torch.Tensor) and a.device == device for a in args
                    ):
                        clean_kwargs.pop(k, None)

                def _to_additive_mask_lastdim(mask, key_len, dtype):
                    """Normalize any mask shape to [B,1,1,key_len] and convert to additive."""
                    if mask is None:
                        return None
                    if mask.dim() == 2:
                        mask = mask[:, None, None, :]
                    elif mask.dim() == 3:
                        if mask.size(-2) != 1:
                            mask = mask.max(dim=-2, keepdim=False).values
                        else:
                            mask = mask.squeeze(-2)
                        mask = mask[:, None, None, :]
                    elif mask.dim() == 4:
                        if mask.size(1) != 1:
                            mask = mask.max(dim=1, keepdim=True).values
                        if mask.size(2) != 1:
                            mask = mask.max(dim=2, keepdim=True).values
                    else:
                        raise ValueError(f"unsupported mask dim: {mask.dim()}")

                    cur = mask.size(-1)
                    if cur > key_len:
                        mask = mask[..., :key_len]
                    elif cur < key_len:
                        mask = F.pad(mask, (0, key_len - cur), 0)

                    mask = mask.to(dtype=dtype)
                    if mask.dtype == torch.bool or (mask.dtype.is_floating_point and mask.max() <= 1):
                        mask = (1.0 - mask) * torch.finfo(dtype).min
                    return mask

                def _latent_friendly_causal_mask(B, L, K, dtype, device):
                    return torch.zeros((B, 1, L, K), dtype=dtype, device=device)

                is_mhla = (getattr(self_attn, "__class__", None).__name__ == "MultiHeadLatentAttention")

                if is_mhla:
                    L = getattr(self_attn, "num_latents", S)
                    am_in = attention_mask
                    K = am_in.shape[-1] if (am_in is not None and am_in.dim() >= 2) else S

                    if am_in is not None:
                        attention_mask = _to_additive_mask_lastdim(am_in, K, dtype)

                    if clean_kwargs.get("causal_mask", None) is not None:
                        clean_kwargs["causal_mask"] = _latent_friendly_causal_mask(B, L, K, dtype, device)

                    # Keyword-only call to avoid positional collisions
                    return _orig_forward(hidden_states, *args, attention_mask=attention_mask, **clean_kwargs)

                # Baseline path: trim/pad mask last dim to sequence length if needed
                if attention_mask is not None and attention_mask.size(-1) != S:
                    attention_mask = attention_mask[..., :S]

                # Safer during training/trace to keep L == K
                clean_kwargs.setdefault("use_cache", False)

                return _orig_forward(hidden_states, *args, attention_mask=attention_mask, **clean_kwargs)

            attn_mod.forward = MethodType(compatible_forward, attn_mod)
            setattr(layer, attn_attr, attn_mod)

    @staticmethod
    def _locate_attn_attr(layer):
        for name in ("self_attn", "attn", "attention"):
            if hasattr(layer, name):
                return name
        return None

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
        if labels is not None:
            return outputs.loss, outputs.logits
        return outputs
