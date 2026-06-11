import os

os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["PYTORCH_TRITON_DISABLE"] = "1"
os.environ["DISABLE_TE_FLASH_ATTENTION"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import time, glob, gc, pickle
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from transformers import Qwen2ForCausalLM, Qwen2Tokenizer
from datasets import load_dataset
import pandas as pd
import psutil
import timm
import sys

# ── Project modules ──────────────────────────────────────────────────────────
from loaders.dataset import TextDataset, ImageDataset
from utils.resource_monitor import monitor_resource_usage
from utils.gpu_monitor import monitor_GPU_usage
from utils.flops_counter import measure_flops
from utils.plot_results import plot_result
import config as _cfg

# ── CLI arguments ─────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser()
_parser.add_argument("--model", type=str, default=None,
                     help=f"Model name. Options: {_cfg.model_name_list}")
_parser.add_argument("--mode",  type=str, choices=["fast", "full"], default=None,
                     help="Test mode: fast (quick sanity check) or full (full experiment)")
_args = _parser.parse_args()

model_name = _args.model if _args.model else _cfg.model_name
fast_test  = (_args.mode == "fast") if _args.mode else _cfg.fast_test
full_test  = (_args.mode == "full") if _args.mode else _cfg.full_test

if fast_test:
    only_use_first_several_batch = True
    number_first_several_batch   = _cfg._FAST["number_first_several_batch"]
    batch_size_number            = _cfg._FAST["batch_size_number"]
    only_use_single_attention    = False
    use_flops                    = True
    max_epoch                    = _cfg._FAST["max_epoch"]
elif full_test:
    _mcfg = _cfg._FULL[model_name]
    only_use_first_several_batch = True
    number_first_several_batch   = _mcfg["number_first_several_batch"]
    batch_size_number            = _mcfg["batch_size_number"]
    only_use_single_attention    = False
    use_flops                    = True
    max_epoch                    = _mcfg["max_epoch"]
else:
    only_use_first_several_batch = _cfg.only_use_first_several_batch
    number_first_several_batch   = _cfg.number_first_several_batch
    batch_size_number            = _cfg.batch_size_number
    only_use_single_attention    = _cfg.only_use_single_attention
    use_flops                    = _cfg.use_flops
    max_epoch                    = _cfg.max_epoch

from attention_modules import (
    GroupQueryAttention, ScaledDotProductAttention, MultiHeadFlexAttention,
    LinearFlexAttention, LSHAttention, SlidingWindowAttention,
    FlashAttention, BaselineAttention, MultiHeadLatentAttention,
    BaselineModernBertAttention, BaselineQwenAttention,
)
from backbones import (GPT2LM, ModernBert, VisionTransformer, NanoVLM, QwenLM)

torch._dynamo.config.suppress_errors = True

MONITOR_EVERY_N_BATCHES = 10


class GPT2CompatibleConfig:
    def __init__(self, config):
        self._cfg = config
        self.n_embd = config.hidden_size
        self.n_head = config.num_attention_heads
        self.n_layer = config.num_hidden_layers
        self.vocab_size = config.vocab_size
        self.num_attention_heads = config.num_attention_heads

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_cfg"), name)


class QwenCompatibleConfig:
    def __init__(self, config):
        self._cfg = config
        self.n_embd = config.hidden_size
        self.n_head = config.num_attention_heads
        self.n_layer = config.num_hidden_layers
        self.vocab_size = config.vocab_size
        self.num_attention_heads = config.num_attention_heads

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_cfg"), name)


class ViTCompatibleConfig:
    """Makes timm ViT config compatible with attention modules expecting GPT2-style attributes."""
    def __init__(self, model):
        self.model = model
        self.n_embd = model.embed_dim
        self.n_head = model.num_heads if hasattr(model, 'num_heads') else 12
        self.n_layer = len(model.blocks)
        self.vocab_size = model.num_classes
        self.num_attention_heads = self.n_head


# ── Optional: Intel RAPL power ───────────────────────────────────────────────
def read_cpu_energy_uj():
    paths = (
        glob.glob("/sys/class/powercap/intel-rapl/intel-rapl:?/energy_uj")
        + glob.glob("/sys/class/powercap/intel-rapl/energy_uj")
        + glob.glob("/sys/class/powercap/intel-rapl/intel-rapl:*/energy_uj")
    )
    for p in sorted(paths):
        try:
            with open(p) as f:
                return int(f.read().strip())
        except Exception:
            continue
    return None


def load_qwen2_eager(model_name):
    """Load Qwen2 with the eager (non-fused) attention backend for custom attention replacement."""
    m = Qwen2ForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    m.config.attn_implementation  = "eager"
    m.config._attn_implementation = "eager"

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    return m.to("cuda")


def _clear_all_forward_hooks(model):
    """Remove all forward hooks from every submodule to prevent memory leaks after FLOP counting."""
    for module in model.modules():
        module._forward_hooks.clear()
        module._forward_pre_hooks.clear()


def measure_flops_once(model, dataloader, device, is_vision_model=False, is_vlm_model=False):
    """Measure FLOPS once using a single batch. Cleans up hooks afterwards to prevent leaks."""
    model.eval()
    flops_per_batch = 0.0

    with torch.no_grad():
        for batch in dataloader:
            try:
                if is_vlm_model:
                    images        = batch["image"].to(device)
                    input_ids     = batch["input_ids"].to(device)
                    attention_mask= batch["attention_mask"].to(device)

                    from utils.flops_counter import measure_flops_vlm
                    flops_per_batch = measure_flops_vlm(model, images, input_ids, attention_mask)

                    if flops_per_batch == 0.0:
                        num_params = sum(p.numel() for p in model.parameters())
                        seq_len = input_ids.shape[1]
                        flops_per_batch = num_params * seq_len * 2

                elif is_vision_model:
                    images, labels = batch
                    images = images.to(device)
                    flops_per_batch = measure_flops(model, images)

                else:
                    inputs, masks = (t.to(device) for t in batch)
                    if inputs.shape[1] == 0:
                        continue
                    flops_per_batch = measure_flops(model, inputs)

            finally:
                # Always remove hooks regardless of whether an exception occurred
                _clear_all_forward_hooks(model)

            model.train()
            print(f"  → FLOPS per batch (measured once): {flops_per_batch:,.0f}")
            return flops_per_batch

    model.train()
    return 0.0


def cleanup_model(model, optimizer, device):
    """
    Fully release a model and optimizer from GPU memory.
    Moving to CPU before deletion forces CUDA to free the allocations immediately
    rather than waiting on the garbage collector.
    """
    # Release gradient buffers first
    optimizer.zero_grad(set_to_none=True)
    del optimizer

    # Remove any lingering hooks
    _clear_all_forward_hooks(model)

    # Move to CPU so CUDA memory is released before del
    model.cpu()
    del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize(device)

    if torch.cuda.is_available():
        allocated_mib = torch.cuda.memory_allocated(device) / 1024**2
        free_gib = (
            torch.cuda.get_device_properties(device).total_memory
            - torch.cuda.memory_allocated(device)
        ) / 1024**3



def check_gpu_headroom(device, min_gib=2.0):
    """Warn if free GPU memory is below threshold before loading the next model."""
    if not torch.cuda.is_available():
        return
    free_gib = (
        torch.cuda.get_device_properties(device).total_memory
        - torch.cuda.memory_allocated(device)
    ) / 1024**3
    if free_gib < min_gib:
        print(f"  WARNING: less than {min_gib} GiB free -- previous model may not have released fully")


def train_model(
    model_to_train, dataloader_instance, optimizer_instance,
    criterion_instance, device_instance, num_epochs_to_run,
    baseline_gb,
    is_vision_model=False,
    is_vlm_model=False,
    flops_per_batch=0.0,
):
    cpu_percent_data   = []
    cpu_usage_data     = []
    cpu_power_data     = []
    memory_usage_data  = []
    flops_data         = []
    gpu_percent_data   = []
    gpu_memory_data    = []
    gpu_power_data     = []
    disk_io_read_data  = []
    disk_io_write_data = []
    inference_time_data= []
    training_time_data = []
    loss_data          = []

    for epoch in range(num_epochs_to_run):
        print(f"Epoch {epoch+1}/{num_epochs_to_run}")
        model_to_train.train()

        epoch_start      = time.time()
        cpu_energy_start = read_cpu_energy_uj()

        num_batch            = 0
        totals               = dict(mem=0.0, cpu_pct=0.0, cpu_time=0.0,
                                    flops=0.0, gpu_pct=0.0, gpu_mem=0.0, gpu_pwr=0.0)
        prev_cpu_time        = None
        disk_read_start      = psutil.disk_io_counters().read_bytes
        disk_write_start     = psutil.disk_io_counters().write_bytes
        total_inference_time = 0.0
        total_loss           = 0.0
        monitor_samples      = 0

        for batch in dataloader_instance:
            if only_use_first_several_batch and num_batch == number_first_several_batch:
                break
            num_batch += 1

            if is_vlm_model:
                images         = batch["image"].to(device_instance)
                input_ids      = batch["input_ids"].to(device_instance)
                attention_mask = batch["attention_mask"].to(device_instance)
                labels         = batch["labels"].to(device_instance)

                if torch.cuda.is_available():
                    torch.cuda.synchronize(device_instance)

                optimizer_instance.zero_grad(set_to_none=True)
                inf_start = time.time()

                outputs = model_to_train(images, input_ids, attention_mask, labels=labels)
                logits, loss = outputs
                if loss is None:
                    loss = criterion_instance(logits.view(-1, logits.size(-1)), labels.view(-1))

                if torch.cuda.is_available():
                    torch.cuda.synchronize(device_instance)
                total_inference_time += time.time() - inf_start

            elif is_vision_model:
                images, labels = batch
                images, labels = images.to(device_instance), labels.to(device_instance)

                if torch.cuda.is_available():
                    torch.cuda.synchronize(device_instance)

                optimizer_instance.zero_grad(set_to_none=True)
                inf_start = time.time()

                outputs = model_to_train(images)
                logits  = outputs.logits if hasattr(outputs, 'logits') else outputs
                loss    = criterion_instance(logits, labels)

                if torch.cuda.is_available():
                    torch.cuda.synchronize(device_instance)
                total_inference_time += time.time() - inf_start

            else:
                inputs, masks = (t.to(device_instance) for t in batch)
                if inputs.shape[1] == 0:
                    continue

                if torch.cuda.is_available():
                    torch.cuda.synchronize(device_instance)

                optimizer_instance.zero_grad(set_to_none=True)
                inf_start = time.time()
                outputs = model_to_train(input_ids=inputs, attention_mask=masks, labels=inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize(device_instance)
                total_inference_time += time.time() - inf_start

                loss = outputs[0]

            loss.backward()
            optimizer_instance.step()
            total_loss += loss.item()

            # Free the computation graph immediately after step
            del loss

            if use_flops:
                totals["flops"] += flops_per_batch

            # Periodically release cache and sample resource usage
            if num_batch % MONITOR_EVERY_N_BATCHES == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                monitor_samples += 1

                mem_usage, cpu_percent, cpu_time_used = monitor_resource_usage()
                gpu_pct, gpu_mem, gpu_pwr = monitor_GPU_usage()

                mem_usage = max(mem_usage - baseline_gb, 0.0)

                totals["mem"]     += mem_usage
                totals["cpu_pct"] += cpu_percent

                delta_cpu_time = cpu_time_used if prev_cpu_time is None else cpu_time_used - prev_cpu_time
                prev_cpu_time   = cpu_time_used
                totals["cpu_time"] += delta_cpu_time

                totals["gpu_pct"] += gpu_pct
                totals["gpu_mem"] += gpu_mem
                totals["gpu_pwr"] += gpu_pwr

        # ── Epoch aggregates ───────────────────────────────────────────
        if torch.cuda.is_available():
            torch.cuda.synchronize(device_instance)

        if num_batch:
            loss_avg  = total_loss / num_batch
            flops_avg = totals["flops"] / num_batch if use_flops else 0.0
        else:
            loss_avg  = 0.0
            flops_avg = 0.0

        if monitor_samples > 0:
            averages = {k: totals[k] / monitor_samples
                        for k in ["mem", "cpu_pct", "cpu_time", "gpu_pct", "gpu_mem", "gpu_pwr"]}
        else:
            averages = {k: 0.0 for k in ["mem", "cpu_pct", "cpu_time", "gpu_pct", "gpu_mem", "gpu_pwr"]}

        disk_read_end  = psutil.disk_io_counters().read_bytes
        disk_write_end = psutil.disk_io_counters().write_bytes
        epoch_time     = time.time() - epoch_start

        cpu_energy_end = read_cpu_energy_uj()
        if (cpu_energy_start is not None and cpu_energy_end is not None
                and cpu_energy_end >= cpu_energy_start and epoch_time > 0):
            avg_cpu_power = (cpu_energy_end - cpu_energy_start) / 1e6 / epoch_time
        else:
            avg_cpu_power = 0.0

        # ── Log rows ──────────────────────────────────────────────────
        cpu_percent_data.append(   {"Epoch": epoch+1, "CPU_Percent":  averages["cpu_pct"]})
        cpu_usage_data.append(     {"Epoch": epoch+1, "CPU_Time":     averages["cpu_time"]})
        cpu_power_data.append(     {"Epoch": epoch+1, "CPU_Power":    avg_cpu_power})
        memory_usage_data.append(  {"Epoch": epoch+1, "Memory_Usage": averages["mem"]})
        flops_data.append(         {"Epoch": epoch+1, "FLOPS":        flops_avg})
        gpu_percent_data.append(   {"Epoch": epoch+1, "GPU_Utilization_Percentage": averages["gpu_pct"]})
        gpu_memory_data.append(    {"Epoch": epoch+1, "GPU_Memory":   averages["gpu_mem"]})
        gpu_power_data.append(     {"Epoch": epoch+1, "GPU_Power":    averages["gpu_pwr"]})
        disk_io_read_data.append(  {"Epoch": epoch+1, "Disk_IO_Read": disk_read_end - disk_read_start})
        disk_io_write_data.append( {"Epoch": epoch+1, "Disk_IO_Write":disk_write_end - disk_write_start})
        inference_time_data.append({"Epoch": epoch+1, "Inference_Time": total_inference_time})
        training_time_data.append( {"Epoch": epoch+1, "Training_Time":  epoch_time})
        loss_data.append(          {"Epoch": epoch+1, "Loss":           loss_avg})

    return tuple(
        pd.DataFrame(lst) for lst in (
            cpu_percent_data, cpu_usage_data, cpu_power_data, memory_usage_data,
            flops_data, gpu_percent_data, gpu_memory_data, gpu_power_data,
            disk_io_read_data, disk_io_write_data,
            inference_time_data, training_time_data, loss_data
        )
    )


# ── Setup ─────────────────────────────────────────────────────────────────────
overall_start = time.perf_counter()

# ── Model config loading ──────────────────────────────────────────────────────
if "gpt2" in model_name.lower():
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    gpt2_config = GPT2LMHeadModel.from_pretrained(model_name).config
    is_vision = False
    is_vlm    = False

elif "vit" in model_name.lower():
    vit_model  = timm.create_model(model_name, pretrained=False)
    vit_config = vit_model
    tokenizer  = None
    is_vision  = True
    is_vlm     = False

elif "nanovlm" in model_name.lower():
    from loaders.dataset import VLMDataset

    external_repo_path = '/mnt/DATA/temp/final/nanoVLM'
    if external_repo_path not in sys.path:
        sys.path.insert(0, external_repo_path)
    from models.vision_language_model import VisionLanguageModel

    temp_model = VisionLanguageModel.from_pretrained("lusxvr/nanoVLM-222M")
    tokenizer  = temp_model.tokenizer

    dummy_image = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        img_emb = temp_model.vision_encoder(dummy_image)
        img_emb = temp_model.MP(img_emb)
    mp_image_token_length = img_emb.shape[1]
    print(f"Using {mp_image_token_length} image tokens")
    del temp_model, dummy_image, img_emb
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    is_vision = False
    is_vlm    = True

elif "qwen" in model_name.lower():
    tokenizer = Qwen2Tokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # FIX: load config only (no full model weights) to avoid a persistent
    # GPU allocation that never gets freed before the attention loop starts
    qwen_config = AutoConfig.from_pretrained(model_name)
    setattr(qwen_config, "attn_implementation",  "sdpa")
    setattr(qwen_config, "_attn_implementation", "sdpa")

    is_vision = False
    is_vlm    = False

else:
    # ModernBERT / other text models
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    modernbert_config = AutoModelForMaskedLM.from_pretrained(model_name).config
    modernbert_config.use_compile_embeddings = False
    if hasattr(modernbert_config, "loss_type"):
        del modernbert_config.loss_type
    is_vision = False
    is_vlm    = False


# ── Dataset loading ───────────────────────────────────────────────────────────
if is_vlm:
    raw_dataset = load_dataset("HuggingFaceM4/the_cauldron", "vqav2", split="train")
    raw_dataset = raw_dataset.select(range(min(50000, len(raw_dataset))))
    vlm_dataset = VLMDataset(raw_dataset, tokenizer, image_size=224, mp_image_token_length=mp_image_token_length)
    dataloader  = DataLoader(
        vlm_dataset,
        batch_size=batch_size_number,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
        prefetch_factor=3,
        persistent_workers=True,
    )

elif is_vision:
    image_dataset = ImageDataset(model_name)
    dataloader    = DataLoader(
        image_dataset, batch_size=batch_size_number, shuffle=True,
        num_workers=16, pin_memory=True, prefetch_factor=3, persistent_workers=True,
    )
    print(f"Number of classes: {len(image_dataset.classes)}")

else:
    dataset      = load_dataset("allenai/tulu-v2-sft-mixture", split="train")
    text_dataset = TextDataset(dataset, tokenizer)
    dataloader   = DataLoader(text_dataset, batch_size=batch_size_number, shuffle=True)
    print("Vocab size:", tokenizer.vocab_size)


# ── Attention module selection ────────────────────────────────────────────────
if "gpt2" in model_name.lower():
    attention_module_classes = [
        BaselineAttention,
        ScaledDotProductAttention,
        LinearFlexAttention,
        LSHAttention,
        GroupQueryAttention,
        MultiHeadFlexAttention,
        FlashAttention,
        SlidingWindowAttention,
        MultiHeadLatentAttention,
    ]

elif "vit" in model_name.lower():
    from attention_modules import (
        BaselineViTAttention, ViTGroupQueryAttention, ViTScaledDotProductAttention,
        ViTFlashAttention, ViTMultiHeadLatentAttention,
    )
    attention_module_classes = [
        BaselineViTAttention,
        ViTScaledDotProductAttention,
        ViTGroupQueryAttention,
        ViTFlashAttention,
        ViTMultiHeadLatentAttention,
    ]

elif "nanovlm" in model_name.lower():
    from attention_modules import (
        NanoVLMBaselineAttention, NanoVLMFlashAttention, NanoVLMGroupQueryAttention,
        NanoVLMMultiHeadLatentAttention, NanoVLMScaledDotProductAttention,
    )
    attention_module_classes = [
        NanoVLMBaselineAttention,
        NanoVLMScaledDotProductAttention,
        NanoVLMFlashAttention,
        NanoVLMGroupQueryAttention,
        NanoVLMMultiHeadLatentAttention,
    ]

elif "qwen" in model_name.lower():
    attention_module_classes = [
        BaselineQwenAttention,
        ScaledDotProductAttention,
        LinearFlexAttention,
        LSHAttention,
        GroupQueryAttention,
        MultiHeadFlexAttention,
        FlashAttention,
        SlidingWindowAttention,
        MultiHeadLatentAttention,
    ]

else:
    attention_module_classes = [
        BaselineModernBertAttention,
        MultiHeadLatentAttention,
        ScaledDotProductAttention,
        LinearFlexAttention,
        LSHAttention,
        GroupQueryAttention,
        MultiHeadFlexAttention,
        FlashAttention,
        SlidingWindowAttention,
    ]

if only_use_single_attention:
    if is_vision:
        attention_module_classes = [ViTScaledDotProductAttention]
    elif is_vlm:
        attention_module_classes = [NanoVLMBaselineAttention]
    elif "qwen" in model_name.lower():
        attention_module_classes = [BaselineQwenAttention]
    else:
        attention_module_classes = [ScaledDotProductAttention]

device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion    = nn.CrossEntropyLoss()
results_dict = {}

# ── Outer loop over attention variants ───────────────────────────────────────
for attn_cls in attention_module_classes:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize(device)

    # Show free memory before loading; warn if headroom is low
    check_gpu_headroom(device, min_gib=2.0)

    baseline_gb = psutil.virtual_memory().used / (1024**3)

    print(f"\n──────────────── Training with {attn_cls.__name__} ────────────────")

    if "gpt2" in model_name.lower():
        current_model = GPT2LM(gpt2_config, attn_cls).to(device)

    elif "vit" in model_name.lower():
        adapted_config = ViTCompatibleConfig(vit_config)
        current_model  = VisionTransformer(model_name, attn_cls, adapted_config).to(device)

    elif is_vlm:
        current_model = NanoVLM(attn_cls).to(device)

    elif "qwen" in model_name.lower():
        if attn_cls is BaselineQwenAttention:
            current_model = load_qwen2_eager(model_name)
        else:
            adapted_cfg   = QwenCompatibleConfig(qwen_config)
            current_model = QwenLM(
                qwen_config,
                attention_module_cls=(lambda _cfg, _layer_id=None: attn_cls(adapted_cfg))
            ).to(device)

    else:
        if attn_cls is BaselineModernBertAttention:
            modernbert_config.attn_implementation  = "sdpa"
            modernbert_config._attn_implementation = "sdpa"
            current_model = ModernBert(
                modernbert_config,
                attention_module_cls=(lambda cfg, layer_id: BaselineModernBertAttention(cfg, layer_id))
            )
        else:
            adapted_config = GPT2CompatibleConfig(modernbert_config)
            current_model  = ModernBert(modernbert_config, lambda _: attn_cls(adapted_config))
        current_model.to('cuda')

    optimizer = optim.AdamW(current_model.parameters(), lr=5e-5)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    flops_per_batch = 0.0
    if use_flops:
        print("  Measuring FLOPS (one-time measurement)...")
        flops_per_batch = measure_flops_once(
            current_model, dataloader, device,
            is_vision_model=is_vision, is_vlm_model=is_vlm,
        )

    dfs = train_model(
        current_model, dataloader, optimizer, criterion,
        device, max_epoch, baseline_gb, is_vision, is_vlm,
        flops_per_batch=flops_per_batch,
    )
    (
        cpu_percent_df, cpu_usage_df, cpu_power_df, memory_usage_df, flops_df,
        gpu_percent_df, gpu_memory_df, gpu_power_df, disk_io_read_df,
        disk_io_write_df, inference_time_df, training_time_df, loss_df,
    ) = dfs

    # ── Model size record ─────────────────────────────────────────────
    tmp_path      = f"{attn_cls.__name__}.pth"
    torch.save(current_model.state_dict(), tmp_path)
    model_size_mb = os.path.getsize(tmp_path) / (1024**2)
    os.remove(tmp_path)
    model_size_df = pd.DataFrame({"Epoch": ["n/a"], "Model size": [model_size_mb]})

    # ── Collect results ───────────────────────────────────────────────
    key = (attn_cls.__name__,)
    results_dict[key + ("cpu_percent",               "percent")] = cpu_percent_df
    results_dict[key + ("cpu_usage",                 "seconds")] = cpu_usage_df
    results_dict[key + ("cpu_power",                 "W")]       = cpu_power_df
    results_dict[key + ("memory_usage",              "GB")]      = memory_usage_df
    results_dict[key + ("FLOPS",                     "FLOPS")]   = flops_df
    results_dict[key + ("gpu_utilization_percentage","percent")]  = gpu_percent_df
    results_dict[key + ("gpu_memory",                "MB")]      = gpu_memory_df
    results_dict[key + ("gpu_power",                 "W")]       = gpu_power_df
    results_dict[key + ("disk_io_read",              "MB")]      = disk_io_read_df
    results_dict[key + ("disk_io_write",             "MB")]      = disk_io_write_df
    results_dict[key + ("model_size",                "MB")]      = model_size_df
    results_dict[key + ("inference_time",            "s")]       = inference_time_df
    results_dict[key + ("training_time",             "s")]       = training_time_df
    results_dict[key + ("loss",                      "loss")]    = loss_df

    # ── Clean up before next module ───────────────────────────────────
    cleanup_model(current_model, optimizer, device)


# ── Plot + serialize results ──────────────────────────────────────────────────
safe_model_name = model_name.replace("/", "_")
test_type  = "full_test" if full_test else "fast_test" if fast_test else "normal_test"
run_stamp  = datetime.now().strftime("%Y-%m-%d,%H:%M:%S")
base_name  = (
    f"{run_stamp},{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'},"
    f"{test_type}_final_results_{safe_model_name}_batchNumber{number_first_several_batch}"
    f"_batchSize{batch_size_number}"
)
results_dir = os.path.join("results", base_name)
os.makedirs(results_dir, exist_ok=True)

with open(
    os.path.join(
        results_dir,
        f"final_results_{safe_model_name}_batchNumber{number_first_several_batch}"
        f"_batchSize{batch_size_number}.pkl",
    ),
    "wb",
) as f:
    pickle.dump(results_dict, f)

plot_result(results_dict, results_dir)

# ── Console summary ───────────────────────────────────────────────────────────
print("\nConfiguration:")
print(f"only_use_first_several_batch = {only_use_first_several_batch}")
print(f"number_first_several_batch    = {number_first_several_batch}")
print(f"batch_size_number              = {batch_size_number}")
print(f"only_use_single_attention      = {only_use_single_attention}")
print(f"use_flops                      = {use_flops}")
print(f"max_epoch                      = {max_epoch}")
print(f"fast_test                      = {fast_test}")
print(f"full_test                      = {full_test}")

elapsed = time.perf_counter() - overall_start
print(f"\nTotal run time: {elapsed:.2f} seconds")