# ── Active model ──────────────────────────────────────────────────────────────
model_name_list = [
    "gpt2-medium",
    "answerdotai/ModernBERT-base",
    "vit_base_patch16_224",
    "nanovlm",
    "Qwen/Qwen2.5-0.5B",
]
model_name = model_name_list[0]

# ── Test mode ─────────────────────────────────────────────────────────────────
fast_test = False
full_test = True

# ── Shared defaults (active when both fast_test and full_test are False) ──────
only_use_first_several_batch = True
number_first_several_batch   = 5
batch_size_number            = 5
only_use_single_attention    = True
use_flops                    = True
max_epoch                    = 3

# ── Fast test: common across all models ───────────────────────────────────────
_FAST = dict(
    number_first_several_batch=10,
    batch_size_number=4,
    max_epoch=3,
)

# ── Full test: per-model tuned values ─────────────────────────────────────────
_FULL = {
    "gpt2-medium":                 dict(number_first_several_batch=400, batch_size_number=8,   max_epoch=30),
    "answerdotai/ModernBERT-base": dict(number_first_several_batch=400, batch_size_number=16,  max_epoch=30),
    "vit_base_patch16_224":        dict(number_first_several_batch=390, batch_size_number=128, max_epoch=30),
    "nanovlm":                     dict(number_first_several_batch=400, batch_size_number=40,  max_epoch=30),
    "Qwen/Qwen2.5-0.5B":          dict(number_first_several_batch=400, batch_size_number=4,  max_epoch=30),
}

# ── Apply ─────────────────────────────────────────────────────────────────────
if fast_test:
    only_use_first_several_batch = True
    number_first_several_batch   = _FAST["number_first_several_batch"]
    batch_size_number            = _FAST["batch_size_number"]
    only_use_single_attention    = False
    use_flops                    = True
    max_epoch                    = _FAST["max_epoch"]

if full_test:
    _cfg = _FULL[model_name]
    only_use_first_several_batch = True
    number_first_several_batch   = _cfg["number_first_several_batch"]
    batch_size_number            = _cfg["batch_size_number"]
    only_use_single_attention    = False
    use_flops                    = True
    max_epoch                    = _cfg["max_epoch"]
