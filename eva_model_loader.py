"""
eva_model_loader.py — Model + processor bootstrap (BF16, mRoPE patch).

Extracted from eva_core.py during the P2 step-3 refactor. These helpers run
exactly once at agent boot. They have no class state and no dependency on the
rest of the eva pipeline.

Public surface (re-exported via `from eva_model_loader import *` in eva_core):
    build_processor, _patch_rope_scaling, load_model_bf16
"""

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText


__all__ = ["build_processor", "_patch_rope_scaling", "load_model_bf16"]


# ============================================================
# Processor (tokenizer + image processor)
# ============================================================
def build_processor(model_path):
    print(f"Loading processor natively from: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path)
    tok = processor.tokenizer
    if tok.pad_token is None:
        tok.pad_token = getattr(tok, "eos_token", "<|pad|>")
    tok.padding_side = "right"
    return processor


# ============================================================
# mRoPE scaling patch
# Some Qwen-VL checkpoints ship without rope_scaling on every attn layer; we
# inject the canonical {type: "mrope", mrope_section: [16, 24, 24]} dict so
# multimodal RoPE works on Blackwell sm_120 + sdpa attention.
# ============================================================
def _patch_rope_scaling(model):
    ROPE_SCALING = {"type": "mrope", "mrope_section": [16, 24, 24]}
    if not getattr(model.config, "rope_scaling", None):
        model.config.rope_scaling = ROPE_SCALING
    if hasattr(model.config, "text_config") and not getattr(
        model.config.text_config, "rope_scaling", None
    ):
        model.config.text_config.rope_scaling = ROPE_SCALING
    patched = 0
    for module in model.modules():
        if hasattr(module, "rope_scaling") and module.rope_scaling is None:
            module.rope_scaling = ROPE_SCALING
            patched += 1
    if patched > 0:
        print(f"[Fix] rope_scaling patched on {patched} attention layers.")


# ============================================================
# BF16 model boot
# ============================================================
def load_model_bf16(model_path):
    print(f"Loading model natively in BF16 from: {model_path}...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    _patch_rope_scaling(model)
    model.config.use_cache = True
    return model.eval()
