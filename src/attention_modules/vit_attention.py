import torch
import torch.nn as nn
from flash_attn.flash_attn_interface import flash_attn_func


class BaselineViTAttention(nn.Module):
    """Standard multi-head self-attention baseline for Vision Transformers."""
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.n_head
        self.dim = config.n_embd
        self.head_dim = self.dim // self.num_heads
        assert self.dim % self.num_heads == 0, "n_embd must be divisible by n_head"

        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.attn_drop = nn.Dropout(0.0)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, hidden_states, **kwargs):
        B, N, C = hidden_states.shape
        qkv = self.qkv(hidden_states).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ViTFlashAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.n_head
        self.dim = config.n_embd
        self.head_dim = self.dim // self.num_heads
        assert self.dim % self.num_heads == 0, "n_embd must be divisible by n_head"

        self.query = nn.Linear(self.dim, self.dim)
        self.key = nn.Linear(self.dim, self.dim)
        self.value = nn.Linear(self.dim, self.dim)
        self.proj = nn.Linear(self.dim, self.dim)

    def forward(self, hidden_states, **kwargs):
        orig_dtype = hidden_states.dtype
        B, N, C = hidden_states.size()

        q = self.query(hidden_states).view(B, N, self.num_heads, self.head_dim)
        k = self.key(hidden_states).view(B, N, self.num_heads, self.head_dim)
        v = self.value(hidden_states).view(B, N, self.num_heads, self.head_dim)

        # flash_attn requires bfloat16; restore original dtype after
        q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)
        out = flash_attn_func(q, k, v, causal=False)

        out = out.reshape(B, N, C).to(orig_dtype)
        return self.proj(out)


class ViTGroupQueryAttention(nn.Module):
    def __init__(self, config, num_groups=8):
        super().__init__()
        self.num_groups = num_groups
        self.num_heads = num_groups
        self.head_dim = config.n_embd // num_groups
        assert config.n_embd % num_groups == 0, "n_embd must be divisible by num_groups"

        self.scale = self.head_dim ** -0.5
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, hidden_states, **kwargs):
        B, N, C = hidden_states.size()

        q = self.query(hidden_states).view(B, N, self.num_groups, self.head_dim).transpose(1, 2)
        k = self.key(hidden_states).view(B, N, self.num_groups, self.head_dim).transpose(1, 2)
        v = self.value(hidden_states).view(B, N, self.num_groups, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class ViTMultiHeadLatentAttention(nn.Module):
    def __init__(self, config, num_latents=64):
        super().__init__()
        self.num_heads = config.n_head
        self.embed_dim = config.n_embd
        self.head_dim = self.embed_dim // self.num_heads
        self.num_latents = num_latents
        self.scale = self.head_dim ** -0.5

        # Learnable latent tokens
        self.latents = nn.Parameter(torch.randn(1, num_latents, self.embed_dim))

        self.query = nn.Linear(self.embed_dim, self.embed_dim)
        self.key = nn.Linear(self.embed_dim, self.embed_dim)
        self.value = nn.Linear(self.embed_dim, self.embed_dim)
        self.latent_to_output = nn.Linear(self.embed_dim, self.embed_dim)
        self.cross_attend_q = nn.Linear(self.embed_dim, self.embed_dim)
        self.cross_attend_k = nn.Linear(self.embed_dim, self.embed_dim)
        self.cross_attend_v = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states, **kwargs):
        B, N, _ = hidden_states.size()
        latents = self.latents.expand(B, -1, -1)

        # Stage 1: latents attend to input patches
        q = self.query(latents).view(B, self.num_latents, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        latent_out = torch.matmul(torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.scale, dim=-1), v)
        latent_out = self.latent_to_output(latent_out.transpose(1, 2).contiguous().view(B, self.num_latents, self.embed_dim))

        # Stage 2: patches attend back to updated latents
        q2 = self.cross_attend_q(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k2 = self.cross_attend_k(latent_out).view(B, self.num_latents, self.num_heads, self.head_dim).transpose(1, 2)
        v2 = self.cross_attend_v(latent_out).view(B, self.num_latents, self.num_heads, self.head_dim).transpose(1, 2)

        out = torch.matmul(torch.softmax(torch.matmul(q2, k2.transpose(-1, -2)) * self.scale, dim=-1), v2)
        return out.transpose(1, 2).contiguous().view(B, N, self.embed_dim)


class ViTScaledDotProductAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.n_head
        self.dim = config.n_embd
        self.head_dim = self.dim // self.num_heads
        assert self.dim % self.num_heads == 0, "n_embd must be divisible by n_head"

        self.scale = self.head_dim ** -0.5
        self.query = nn.Linear(self.dim, self.dim, bias=True)
        self.key = nn.Linear(self.dim, self.dim, bias=True)
        self.value = nn.Linear(self.dim, self.dim, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)

    def forward(self, hidden_states, **kwargs):
        B, N, C = hidden_states.shape

        q = self.query(hidden_states).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.key(hidden_states).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.value(hidden_states).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        out = torch.matmul(torch.softmax(attn, dim=-1), v)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)
