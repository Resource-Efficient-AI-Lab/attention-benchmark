import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn.flash_attn_interface import flash_attn_func
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from transformers.models.modernbert.modeling_modernbert import ModernBertAttention
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention


class BaselineAttention(GPT2Attention):
    def __init__(self, config):
        super().__init__(config)


class BaselineModernBertAttention(ModernBertAttention):
    def __init__(self, config, layer_id: int):
        super().__init__(config, layer_id=layer_id)


class FlashAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.n_embd // self.num_heads
        self.query = nn.Linear(config.n_embd, config.n_embd).to(torch.bfloat16)
        self.key = nn.Linear(config.n_embd, config.n_embd).to(torch.bfloat16)
        self.value = nn.Linear(config.n_embd, config.n_embd).to(torch.bfloat16)

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False, **kwargs):
        hidden_states = hidden_states.to(torch.bfloat16)
        batch_size, seq_length, embed_dim = hidden_states.size()

        query = self.query(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        attention_output = flash_attn_func(query, key, value, causal=True)
        attention_output = attention_output.transpose(1, 2).reshape(batch_size, seq_length, embed_dim)

        outputs = (attention_output,)
        if output_attentions:
            outputs += (attention_output,)
        if use_cache:
            outputs += (None,)

        return outputs


class GroupQueryAttention(nn.Module):
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

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False, **kwargs):
        batch_size, seq_len, hidden_dim = hidden_states.size()

        q = self.query(hidden_states).view(batch_size, seq_len, self.num_groups, self.head_dim).transpose(1, 2)
        k = self.key(hidden_states).view(batch_size, seq_len, self.num_groups, self.head_dim).transpose(1, 2)
        v = self.value(hidden_states).view(batch_size, seq_len, self.num_groups, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=hidden_states.device)).unsqueeze(0).unsqueeze(0)
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))

        attn_probs = torch.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_probs, v).transpose(1, 2).reshape(batch_size, seq_len, hidden_dim)

        outputs = (context,)
        if output_attentions:
            outputs += (attn_probs,)
        if use_cache:
            outputs += (None,)

        return outputs


class LinearFlexAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.scale = torch.sqrt(torch.tensor(config.n_embd, dtype=torch.float32))

    def feature_map(self, x):
        # ELU+1 is a common alternative; ReLU+1 used here as a simple positive feature map
        return F.relu(x) + 1

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False, **kwargs):
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat([past_key, key], dim=-2)
            value = torch.cat([past_value, value], dim=-2)

        query_prime = self.feature_map(query)
        key_prime = self.feature_map(key)

        kv = torch.matmul(key_prime.transpose(-1, -2), value)
        key_sum = key_prime.sum(dim=-2)
        numerator = torch.matmul(query_prime, kv)
        denominator = torch.matmul(query_prime, key_sum.unsqueeze(-1)).clamp(min=1e-6)
        context_layer = numerator / denominator

        present = (key, value) if use_cache else None

        outputs = (context_layer,)
        if output_attentions:
            outputs += (None,)
        if use_cache:
            outputs += (present,)

        return outputs


class LSHAttention(nn.Module):
    def __init__(self, config, num_hashes=8):
        super().__init__()
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.num_hashes = num_hashes
        self.scale = torch.sqrt(torch.tensor(config.n_embd, dtype=torch.float32))

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False, **kwargs):
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        hash_buckets = torch.randint(0, self.num_hashes, (query.size(0), query.size(1)), device=query.device)
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / self.scale
        attention_scores[hash_buckets.unsqueeze(-1) != hash_buckets.unsqueeze(-2)] = float('-inf')

        if attention_mask is not None:
            attention_mask = attention_mask.squeeze(1)
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context_layer = torch.matmul(attention_probs, value)

        present = (key, value) if use_cache else None

        outputs = (context_layer,)
        if output_attentions:
            outputs += (attention_probs,)

        return outputs if not use_cache else outputs + (present,)


class MultiHeadFlexAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False, **kwargs):
        batch_size, seq_length, _ = hidden_states.size()

        query = self.query(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat([past_key, key], dim=-2)
            value = torch.cat([past_value, value], dim=-2)

        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / self.scale

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context_layer = torch.matmul(attention_probs, value)
        context_layer = context_layer.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)

        present = (key, value) if use_cache else None

        outputs = (context_layer,)
        if output_attentions:
            outputs += (attention_probs,)

        return outputs if not use_cache else outputs + (present,)


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config, num_latents=64):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.embed_dim = config.n_embd

        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"
            )

        self.head_dim = self.embed_dim // self.num_heads
        self.num_latents = num_latents
        self.max_seq_len = getattr(config, 'n_positions', 1024)
        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        self.latents = nn.Parameter(torch.randn(1, num_latents, self.embed_dim))
        self.query = nn.Linear(self.embed_dim, self.embed_dim)
        self.key = nn.Linear(self.embed_dim, self.embed_dim)
        self.value = nn.Linear(self.embed_dim, self.embed_dim)
        # Projects from fixed latent sequence length back to the model's max sequence length
        self.output_projection = nn.Linear(self.num_latents, self.max_seq_len)

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False, **kwargs):
        batch_size, seq_len, _ = hidden_states.size()
        dtype = hidden_states.dtype

        latents = self.latents.expand(batch_size, -1, -1).to(dtype=dtype)

        # Cast inputs to each projection layer's dtype to avoid mixed-precision matmul errors
        query = self.query(latents.to(self.query.weight.dtype))
        key = self.key(hidden_states.to(self.key.weight.dtype))
        value = self.value(hidden_states.to(self.value.weight.dtype))

        def split_heads(tensor):
            b, s, _ = tensor.shape
            return tensor.view(b, s, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        query = split_heads(query)
        key = split_heads(key)
        value = split_heads(value)

        if layer_past is not None:
            past_key, past_value = layer_past
            # Cast past tensors to current dtype before concatenating
            key = torch.cat((past_key.to(key.dtype), key), dim=-2)
            value = torch.cat((past_value.to(value.dtype), value), dim=-2)

        present = (key, value) if use_cache else None
        key_seq_len = key.size(-2)

        scores = torch.matmul(query, key.transpose(-1, -2)) / self.scale.to(dtype)

        if attention_mask is not None:
            scores = scores + attention_mask[:, :, :, :key_seq_len].to(dtype=scores.dtype)

        attn_probs = torch.softmax(scores, dim=-1)

        if head_mask is not None:
            attn_probs = attn_probs * head_mask

        attn_output = torch.matmul(attn_probs.to(value.dtype), value)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, self.num_latents, self.embed_dim)

        # Project from latent sequence length back to input sequence length and slice to actual length
        projected = self.output_projection(attn_output.transpose(1, 2).to(self.output_projection.weight.dtype))
        final_output = projected[:, :, :seq_len].transpose(1, 2).contiguous()

        outputs = (final_output,)
        if output_attentions:
            outputs += (attn_probs,)
        if use_cache:
            outputs += (present,)

        return outputs


class ScaledDotProductAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scale = torch.sqrt(torch.tensor(config.n_embd, dtype=torch.float32))
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False, **kwargs):
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat([past_key, key], dim=-2)
            value = torch.cat([past_value, value], dim=-2)

        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / self.scale

        if attention_mask is not None:
            attention_mask = attention_mask.squeeze(1)
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context_layer = torch.matmul(attention_probs, value)

        present = (key, value) if use_cache else None

        outputs = (context_layer,)
        if output_attentions:
            outputs += (attention_probs,)

        return outputs if not use_cache else outputs + (present,)


class SlidingWindowAttention(nn.Module):
    def __init__(self, config, window_size: int = 3):
        super().__init__()
        self.num_heads = config.n_head
        self.head_dim = config.n_embd // self.num_heads
        self.scale = math.sqrt(self.head_dim)
        self.window_size = window_size

        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, use_cache: bool = False, output_attentions: bool = False, **kwargs):
        device = hidden_states.device
        bsz, seq_len, _ = hidden_states.size()

        q = self.query(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).transpose(-2, -1)
        v = self.value(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)

        attn_logits = torch.matmul(q, k) / self.scale

        if attention_mask is not None:
            # GPT2 may pass masks shaped (B,1,1,L) or (B,1,1,L,L); normalise to (B,H,L,L)
            if attention_mask.dim() == 5:
                attention_mask = attention_mask.squeeze(1)
            if attention_mask.dim() == 4:
                if attention_mask.size(2) == 1:
                    attention_mask = attention_mask.squeeze(2).expand(-1, -1, seq_len).unsqueeze(2)
                attention_mask = attention_mask.expand(-1, self.num_heads, -1, -1)
            else:
                raise ValueError(f"Unsupported attention_mask shape {attention_mask.shape}")
            attn_logits += attention_mask

        # Restrict each token to attend only within a local window of neighbouring tokens
        arange = torch.arange(seq_len, device=device)
        near = (arange[:, None] - arange[None, :]).abs() <= self.window_size
        window_mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        window_mask[near] = 0.0
        attn_logits += window_mask.unsqueeze(0).unsqueeze(0)

        attn_prob = F.softmax(attn_logits, dim=-1)

        if head_mask is not None:
            attn_prob *= head_mask

        context = torch.matmul(attn_prob, v.transpose(1, 2)).transpose(1, 2).contiguous().view(bsz, seq_len, -1)

        outputs = (context,)
        if output_attentions:
            outputs += (attn_prob,)
        if use_cache:
            outputs += ((k, v),)

        return outputs


class SparseFlexAttention(nn.Module):
    def __init__(self, config, sparsity=0.1):
        super().__init__()
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.sparsity = sparsity
        self.scale = torch.sqrt(torch.tensor(config.n_embd, dtype=torch.float32))

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False):
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat([past_key, key], dim=-2)
            value = torch.cat([past_value, value], dim=-2)

        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / self.scale
        sparse_mask = torch.rand(attention_scores.size(), device=hidden_states.device) < self.sparsity
        attention_scores = attention_scores.masked_fill(sparse_mask, float('-inf'))

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask.to(hidden_states.device)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context_layer = torch.matmul(attention_probs, value)

        present = (key, value) if use_cache else None

        outputs = (context_layer,)
        if output_attentions:
            outputs += (attention_probs,)

        return outputs if not use_cache else outputs + (present,)


class BaselineQwenAttention(Qwen2Attention):
    def __init__(self, config, layer_id: int):
        super().__init__(config, layer_idx=layer_id)
