import torch
import torch.nn as nn
from flash_attn.flash_attn_interface import flash_attn_func


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_embd(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1):
    """Applies rotary positional embeddings to query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class NanoVLMBaselineAttention:
    """Marker class — no attention replacement happens when this is used. Original NanoVLM weights are preserved."""
    def __init__(self, config=None):
        pass


class NanoVLMFlashAttention(nn.Module):
    """Flash Attention for NanoVLM encoder and decoder."""
    def __init__(self, config):
        super().__init__()
        self.is_causal = getattr(config, 'is_causal', False)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head

        if self.is_causal:
            # Decoder: separate Q/K/V with GQA (9 query heads, 3 KV heads)
            self.n_kv_heads = getattr(config, 'n_kv_heads', self.n_head)
            self.n_kv_groups = self.n_head // self.n_kv_heads
            self.q_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
            self.k_proj = nn.Linear(self.n_embd, self.n_kv_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(self.n_embd, self.n_kv_heads * self.head_dim, bias=False)
            self.out_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        else:
            # Encoder: fused QKV (like original SigLIP)
            self.n_kv_heads = self.n_head
            self.n_kv_groups = 1
            self.qkv_proj = nn.Linear(self.n_embd, self.n_embd * 3, bias=True)
            self.out_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)

    def forward(self, x, cos=None, sin=None, attention_mask=None, block_kv_cache=None):
        B, T_curr, D = x.shape
        orig_dtype = x.dtype

        if self.is_causal:
            # Decoder path with GQA
            q = self.q_proj(x).view(B, T_curr, self.n_head, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(B, T_curr, self.n_kv_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, T_curr, self.n_kv_heads, self.head_dim).transpose(1, 2)

            if cos is not None and sin is not None:
                q, k = apply_rotary_pos_embd(q, k, cos, sin)

            # Handle KV cache
            if block_kv_cache is not None and block_kv_cache['key'] is not None:
                k = torch.cat([block_kv_cache['key'], k], dim=2)
                v = torch.cat([block_kv_cache['value'], v], dim=2)
            block_kv_cache = {'key': k, 'value': v}

            # Repeat KV heads for GQA
            k_exp = k.repeat_interleave(self.n_kv_groups, dim=1)
            v_exp = v.repeat_interleave(self.n_kv_groups, dim=1)

            # Transpose to (B, S, n_heads, head_dim) for flash_attn
            q = q.transpose(1, 2).contiguous()
            k_exp = k_exp.transpose(1, 2).contiguous()
            v_exp = v_exp.transpose(1, 2).contiguous()

            # flash_attn requires bfloat16; use causal mask only during prefill
            T_kv = k_exp.shape[1]
            use_causal = (T_curr == T_kv and T_curr > 1)
            out = flash_attn_func(
                q.to(torch.bfloat16), k_exp.to(torch.bfloat16), v_exp.to(torch.bfloat16), causal=use_causal
            )
            return self.out_proj(out.reshape(B, T_curr, D).to(orig_dtype)), block_kv_cache

        else:
            # Encoder path with fused QKV (no RoPE, no KV cache)
            qkv = self.qkv_proj(x).view(B, T_curr, 3, self.n_head, self.head_dim)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

            out = flash_attn_func(
                q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16), causal=False
            )
            return self.out_proj(out.reshape(B, T_curr, D).to(orig_dtype))


class NanoVLMGroupQueryAttention(nn.Module):
    """Grouped Query Attention for NanoVLM encoder and decoder."""
    def __init__(self, config):
        super().__init__()
        self.is_causal = getattr(config, 'is_causal', False)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.scale = self.head_dim ** -0.5

        if self.is_causal:
            # Decoder: 9 query heads, 3 KV heads (ratio 3:1)
            self.n_kv_heads = getattr(config, 'n_kv_heads', self.n_head // 3)
            self.n_kv_groups = self.n_head // self.n_kv_heads
            self.q_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
            self.k_proj = nn.Linear(self.n_embd, self.n_kv_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(self.n_embd, self.n_kv_heads * self.head_dim, bias=False)
            self.out_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        else:
            # Encoder: full heads (no GQA reduction)
            self.n_kv_heads = self.n_head
            self.n_kv_groups = 1
            self.q_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)
            self.k_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)
            self.v_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)
            self.out_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)

    def forward(self, x, cos=None, sin=None, attention_mask=None, block_kv_cache=None):
        B, T_curr, D = x.shape

        q = self.q_proj(x).view(B, T_curr, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T_curr, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T_curr, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.is_causal and cos is not None and sin is not None:
            q, k = apply_rotary_pos_embd(q, k, cos, sin)

        # Handle KV cache for decoder
        if self.is_causal:
            if block_kv_cache is not None and block_kv_cache['key'] is not None:
                k = torch.cat([block_kv_cache['key'], k], dim=2)
                v = torch.cat([block_kv_cache['value'], v], dim=2)
            block_kv_cache = {'key': k, 'value': v}

        # Repeat KV heads to match query heads
        k_exp = k.repeat_interleave(self.n_kv_groups, dim=1)
        v_exp = v.repeat_interleave(self.n_kv_groups, dim=1)
        T_kv = k_exp.size(2)

        attn = torch.matmul(q, k_exp.transpose(-2, -1)) * self.scale

        # Causal mask applied only during prefill
        if self.is_causal and T_curr == T_kv and T_curr > 1:
            causal_mask = torch.tril(torch.ones(T_curr, T_curr, device=x.device, dtype=torch.bool)).view(1, 1, T_curr, T_curr)
            attn = attn.masked_fill(~causal_mask, float('-inf'))

        if attention_mask is not None:
            mask = (1.0 - attention_mask[:, :T_kv].unsqueeze(1).unsqueeze(2).float()) * torch.finfo(q.dtype).min
            attn = attn + mask

        out = torch.matmul(torch.softmax(attn, dim=-1), v_exp)
        out = self.out_proj(out.transpose(1, 2).contiguous().view(B, T_curr, D))

        if self.is_causal:
            return out, block_kv_cache
        return out


class NanoVLMMultiHeadLatentAttention(nn.Module):
    """Multi-Head Latent Attention for NanoVLM encoder and decoder."""
    def __init__(self, config, num_latents=64):
        super().__init__()
        self.is_causal = getattr(config, 'is_causal', False)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.num_latents = num_latents
        self.scale = self.head_dim ** -0.5

        # Learnable latent tokens
        self.latents = nn.Parameter(torch.randn(1, num_latents, self.n_embd))

        if self.is_causal:
            # Decoder: no bias, GQA support
            self.n_kv_heads = getattr(config, 'n_kv_heads', self.n_head)
            self.n_kv_groups = self.n_head // self.n_kv_heads
            self.query = nn.Linear(self.n_embd, self.n_embd, bias=False)
            self.key = nn.Linear(self.n_embd, self.n_kv_heads * self.head_dim, bias=False)
            self.value = nn.Linear(self.n_embd, self.n_kv_heads * self.head_dim, bias=False)
            self.latent_to_output = nn.Linear(self.n_embd, self.n_embd, bias=False)
            self.cross_attend_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
            self.cross_attend_k = nn.Linear(self.n_embd, self.n_kv_heads * self.head_dim, bias=False)
            self.cross_attend_v = nn.Linear(self.n_embd, self.n_kv_heads * self.head_dim, bias=False)
        else:
            # Encoder: with bias
            self.n_kv_heads = self.n_head
            self.n_kv_groups = 1
            self.query = nn.Linear(self.n_embd, self.n_embd, bias=True)
            self.key = nn.Linear(self.n_embd, self.n_embd, bias=True)
            self.value = nn.Linear(self.n_embd, self.n_embd, bias=True)
            self.latent_to_output = nn.Linear(self.n_embd, self.n_embd, bias=True)
            self.cross_attend_q = nn.Linear(self.n_embd, self.n_embd, bias=True)
            self.cross_attend_k = nn.Linear(self.n_embd, self.n_embd, bias=True)
            self.cross_attend_v = nn.Linear(self.n_embd, self.n_embd, bias=True)

    def forward(self, x, cos=None, sin=None, attention_mask=None, block_kv_cache=None):
        B, T_curr, D = x.shape
        latents = self.latents.expand(B, -1, -1)

        # === Stage 1: Latents attend to input ===
        q = self.query(latents).view(B, self.num_latents, self.n_head, self.head_dim).transpose(1, 2)

        if self.is_causal:
            k = self.key(x).view(B, T_curr, self.n_kv_heads, self.head_dim).transpose(1, 2)
            v = self.value(x).view(B, T_curr, self.n_kv_heads, self.head_dim).transpose(1, 2)
            # Apply RoPE only to keys — queries come from latents and carry no positional info
            if cos is not None and sin is not None:
                k = (k * cos.unsqueeze(1)) + (rotate_half(k) * sin.unsqueeze(1))
            k_exp = k.repeat_interleave(self.n_kv_groups, dim=1)
            v_exp = v.repeat_interleave(self.n_kv_groups, dim=1)
        else:
            k = self.key(x).view(B, T_curr, self.n_head, self.head_dim).transpose(1, 2)
            v = self.value(x).view(B, T_curr, self.n_head, self.head_dim).transpose(1, 2)
            k_exp, v_exp = k, v

        scores = torch.matmul(q, k_exp.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            mask = (1.0 - attention_mask[:, :T_curr].unsqueeze(1).unsqueeze(2).float()) * torch.finfo(q.dtype).min
            scores = scores + mask

        latent_out = torch.matmul(torch.softmax(scores, dim=-1), v_exp)
        latent_out = self.latent_to_output(
            latent_out.transpose(1, 2).contiguous().view(B, self.num_latents, D)
        )

        # === Stage 2: Input attends back to updated latents ===
        q2 = self.cross_attend_q(x).view(B, T_curr, self.n_head, self.head_dim).transpose(1, 2)
        if self.is_causal and cos is not None and sin is not None:
            q2 = (q2 * cos.unsqueeze(1)) + (rotate_half(q2) * sin.unsqueeze(1))

        if self.is_causal:
            k2 = self.cross_attend_k(latent_out).view(B, self.num_latents, self.n_kv_heads, self.head_dim).transpose(1, 2)
            v2 = self.cross_attend_v(latent_out).view(B, self.num_latents, self.n_kv_heads, self.head_dim).transpose(1, 2)
            k2_exp = k2.repeat_interleave(self.n_kv_groups, dim=1)
            v2_exp = v2.repeat_interleave(self.n_kv_groups, dim=1)
        else:
            k2 = self.cross_attend_k(latent_out).view(B, self.num_latents, self.n_head, self.head_dim).transpose(1, 2)
            v2 = self.cross_attend_v(latent_out).view(B, self.num_latents, self.n_head, self.head_dim).transpose(1, 2)
            k2_exp, v2_exp = k2, v2

        # No causal mask needed for cross-attention to latents
        out = torch.matmul(torch.softmax(torch.matmul(q2, k2_exp.transpose(-2, -1)) * self.scale, dim=-1), v2_exp)
        out = out.transpose(1, 2).contiguous().view(B, T_curr, D)

        if self.is_causal:
            # Latent attention doesn't use a traditional KV cache; return empty dict for interface compatibility
            return out, {'key': None, 'value': None}
        return out


class NanoVLMScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention for NanoVLM encoder and decoder."""
    def __init__(self, config):
        super().__init__()
        self.is_causal = getattr(config, 'is_causal', False)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.scale = self.head_dim ** -0.5

        if self.is_causal:
            # Decoder: separate projections, no bias, GQA
            self.n_kv_heads = getattr(config, 'n_kv_heads', self.n_head)
            self.n_kv_groups = self.n_head // self.n_kv_heads
            self.q_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
            self.k_proj = nn.Linear(self.n_embd, self.n_kv_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(self.n_embd, self.n_kv_heads * self.head_dim, bias=False)
            self.out_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        else:
            # Encoder: separate projections, with bias
            self.n_kv_heads = self.n_head
            self.n_kv_groups = 1
            self.q_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)
            self.k_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)
            self.v_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)
            self.out_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)

    def forward(self, x, cos=None, sin=None, attention_mask=None, block_kv_cache=None):
        B, T_curr, C = x.shape

        q = self.q_proj(x).view(B, T_curr, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T_curr, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T_curr, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.is_causal and cos is not None and sin is not None:
            q, k = apply_rotary_pos_embd(q, k, cos, sin)

        # Handle KV cache for decoder
        if self.is_causal:
            if block_kv_cache is not None and block_kv_cache['key'] is not None:
                k = torch.cat([block_kv_cache['key'], k], dim=2)
                v = torch.cat([block_kv_cache['value'], v], dim=2)
            block_kv_cache = {'key': k, 'value': v}

        # Repeat KV heads for GQA
        k_exp = k.repeat_interleave(self.n_kv_groups, dim=1)
        v_exp = v.repeat_interleave(self.n_kv_groups, dim=1)
        T_kv = k_exp.size(2)

        attn = torch.matmul(q, k_exp.transpose(2, 3)) * self.scale

        # Causal mask applied only during prefill
        if self.is_causal and T_curr == T_kv and T_curr > 1:
            causal_mask = torch.tril(torch.ones(T_curr, T_curr, device=x.device, dtype=torch.bool)).view(1, 1, T_curr, T_curr)
            attn = attn.masked_fill(~causal_mask, float('-inf'))

        if attention_mask is not None:
            mask = (1.0 - attention_mask[:, :T_kv].unsqueeze(1).unsqueeze(2).float()) * torch.finfo(q.dtype).min
            attn = attn + mask

        out = torch.matmul(torch.softmax(attn, dim=-1), v_exp)
        out = self.out_proj(out.transpose(1, 2).contiguous().view(B, T_curr, C))

        if self.is_causal:
            return out, block_kv_cache
        return out
