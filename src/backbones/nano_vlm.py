import torch
import torch.nn as nn
import sys
import os
import importlib.util


# Add the external nanoVLM repo to sys.path so its internal dependencies resolve correctly
external_repo_path = '/mnt/DATA/temp/final/nanoVLM'
target_file = os.path.join(external_repo_path, 'models', 'vision_language_model.py')

if external_repo_path not in sys.path:
    sys.path.insert(0, external_repo_path)

spec = importlib.util.spec_from_file_location("nanovlm_module", target_file)
if spec is None:
    raise FileNotFoundError(f"Could not find file: {target_file}")

nanovlm_module = importlib.util.module_from_spec(spec)
nanovlm_module.__spec__.loader.exec_module(nanovlm_module)
VisionLanguageModel = nanovlm_module.VisionLanguageModel


class NanoVLMEncoderConfig:
    """Config for vision encoder attention (SigLIP-style)"""
    def __init__(self):
        self.n_embd = 768
        self.n_head = 12
        self.hidden_size = 768
        self.num_attention_heads = 12
        self.head_dim = 64
        self.is_causal = False
        self.use_rope = False
        self.fused_qkv = True
        self.dropout = 0.0


class NanoVLMDecoderConfig:
    """Config for decoder attention (SmolLM2-style with GQA)"""
    def __init__(self):
        self.n_embd = 576
        self.n_head = 9
        self.n_kv_heads = 3
        self.hidden_size = 576
        self.num_attention_heads = 9
        self.num_kv_heads = 3
        self.head_dim = 64
        self.is_causal = True
        self.use_rope = True
        self.fused_qkv = False
        self.dropout = 0.0


class NanoVLM(nn.Module):
    def __init__(self, attention_module_cls):
        super().__init__()

        self.model = VisionLanguageModel.from_pretrained("lusxvr/nanoVLM-222M")

        # Resize token embedding if the tokenizer vocabulary exceeds the original embedding size
        old_embedding = self.model.decoder.token_embedding
        new_vocab_size = len(self.model.tokenizer)
        if new_vocab_size > old_embedding.num_embeddings:
            new_embedding = nn.Embedding(new_vocab_size, old_embedding.embedding_dim)
            new_embedding.weight.data[:old_embedding.num_embeddings] = old_embedding.weight.data
            self.model.decoder.token_embedding = new_embedding

        # Baseline: no attention replacement, use original weights as-is
        if attention_module_cls.__name__ == "NanoVLMBaselineAttention":
            return

        encoder_config = NanoVLMEncoderConfig()
        decoder_config = NanoVLMDecoderConfig()

        # Replace vision encoder attention (12 blocks)
        for block in self.model.vision_encoder.blocks:
            old_attn = block.attn
            new_attn = attention_module_cls(encoder_config)
            self._copy_encoder_weights(old_attn, new_attn)
            block.attn = new_attn

        # Replace decoder attention (30 blocks)
        for block in self.model.decoder.blocks:
            old_attn = block.attn
            new_attn = attention_module_cls(decoder_config)
            self._copy_decoder_weights(old_attn, new_attn)
            block.attn = new_attn

    def _copy_encoder_weights(self, old_attn, new_attn):
        with torch.no_grad():
            if hasattr(old_attn, 'qkv_proj') and hasattr(new_attn, 'qkv_proj'):
                new_attn.qkv_proj.weight.data = old_attn.qkv_proj.weight.data.clone()
                if old_attn.qkv_proj.bias is not None:
                    new_attn.qkv_proj.bias.data = old_attn.qkv_proj.bias.data.clone()

            if hasattr(old_attn, 'qkv_proj') and hasattr(new_attn, 'q_proj'):
                qkv_weight = old_attn.qkv_proj.weight.data
                qkv_bias = old_attn.qkv_proj.bias.data if old_attn.qkv_proj.bias is not None else None
                dim = qkv_weight.shape[0] // 3
                new_attn.q_proj.weight.data = qkv_weight[:dim].clone()
                new_attn.k_proj.weight.data = qkv_weight[dim:2*dim].clone()
                new_attn.v_proj.weight.data = qkv_weight[2*dim:].clone()
                if qkv_bias is not None and hasattr(new_attn.q_proj, 'bias') and new_attn.q_proj.bias is not None:
                    new_attn.q_proj.bias.data = qkv_bias[:dim].clone()
                    new_attn.k_proj.bias.data = qkv_bias[dim:2*dim].clone()
                    new_attn.v_proj.bias.data = qkv_bias[2*dim:].clone()

            if hasattr(new_attn, 'out_proj') and hasattr(old_attn, 'out_proj'):
                new_attn.out_proj.weight.data = old_attn.out_proj.weight.data.clone()
                if old_attn.out_proj.bias is not None and new_attn.out_proj.bias is not None:
                    new_attn.out_proj.bias.data = old_attn.out_proj.bias.data.clone()

    def _copy_decoder_weights(self, old_attn, new_attn):
        with torch.no_grad():
            for proj_name in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
                if hasattr(new_attn, proj_name) and hasattr(old_attn, proj_name):
                    old_proj = getattr(old_attn, proj_name)
                    new_proj = getattr(new_attn, proj_name)
                    if old_proj.weight.shape == new_proj.weight.shape:
                        new_proj.weight.data = old_proj.weight.data.clone()
                        if old_proj.bias is not None and new_proj.bias is not None:
                            new_proj.bias.data = old_proj.bias.data.clone()

    def forward(self, images, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids, images, attention_mask, targets=labels)
