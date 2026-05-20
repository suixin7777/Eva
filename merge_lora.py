"""
LoRA → Base 合并脚本：SFT 专用健壮版

核心修复:
  1. tokenizer cache 自动查找:
     - 优先使用 TOKENIZER_CACHE
     - 如果 TOKENIZER_CACHE 不存在，但 ADAPTER_PATH 里有 tokenizer.json/tokenizer_config.json，
       自动使用 ADAPTER_PATH 作为 tokenizer cache
     - 如果都没有，默认直接报错，避免 token id 错位
     - 只有 ALLOW_TOKENIZER_RECONSTRUCT=1 时才允许从 base tokenizer 重建
  2. fallback 重建 tokenizer 时使用 add_special_tokens({"additional_special_tokens": ...})
     与训练端一致
  3. 保存前同步:
     - model.config.bos/eos/pad
     - model.config.text_config.bos/eos/pad
     - model.generation_config.bos/eos/pad
  4. 保存后重新加载验证，避免 tokenizer/config/generation_config 不一致
  5. 保留 meta tensor 修复:
     - device_map=None
     - low_cpu_mem_usage=False
  6. 保留 SFT sanity check:
     - LoRA adapter 数量
     - ReAct token embedding/lm_head 是否变化
     - visual.merger 是否变化
"""

import os
import shutil
import torch

from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    AutoModelForImageTextToText,
    GenerationConfig,
)

from peft import PeftModel


# ================= Config =================

BASE_MODEL_PATH = os.getenv(
    "MODEL_NAME",
    "huihui-ai/Huihui-Qwen3.5-9B-abliterated",
)

ADAPTER_PATH = os.getenv(
    "ADAPTER_PATH",
    "./eva-qwen3.5-vl-lora/checkpoint-1100",
)

OUTPUT_DIR = os.getenv(
    "OUTPUT_DIR",
    "./Eva-Qwen3.5-VL-9B-Merged",
)

# 可选：训练时保存的 tokenizer cache。
# 如果这里不存在，脚本会自动尝试 ADAPTER_PATH。
TOKENIZER_CACHE = os.getenv(
    "TOKENIZER_CACHE",
    "./processed_qwen_dataset_cache/tokenizer",
)

HF_TOKEN = os.getenv("HF_TOKEN", None)

# 如果 tokenizer cache 和 adapter path 都找不到 tokenizer，
# 默认报错；只有手动设置为 1 才允许重建。
ALLOW_TOKENIZER_RECONSTRUCT = os.getenv("ALLOW_TOKENIZER_RECONSTRUCT", "0") == "1"

REACT_TOKENS = [
    "<|tool_code|>",
    "<|tool_output|>",
    "<|answer|>",
    "<|end_react|>",
]

# SFT 合并建议开启，因为 SFT 阶段通常训练了新增 token 的 embedding/lm_head。
CHECK_NEW_TOKEN_EMBEDDINGS = os.getenv("CHECK_NEW_TOKEN_EMBEDDINGS", "1") == "1"

# visual.merger 检查开关
CHECK_VISUAL_MERGER = os.getenv("CHECK_VISUAL_MERGER", "1") == "1"

# LoRA adapter 数量预期。你的原脚本写的是 expected ~254。
MIN_EXPECTED_LORA = int(os.getenv("MIN_EXPECTED_LORA", "240"))

# ==========================================


def banner(s, char="="):
    print()
    print(char * 70)
    print(f"  {s}")
    print(char * 70)


def has_tokenizer_files(path: str) -> bool:
    """
    判断一个目录能否作为 tokenizer cache。
    最少应包含 tokenizer.json 或 tokenizer_config.json。
    """
    if not path or not os.path.isdir(path):
        return False

    tokenizer_json = os.path.join(path, "tokenizer.json")
    tokenizer_config = os.path.join(path, "tokenizer_config.json")

    return os.path.exists(tokenizer_json) or os.path.exists(tokenizer_config)


def assert_no_meta_tensors(model, where="model"):
    meta_params = [
        name for name, p in model.named_parameters()
        if getattr(p, "is_meta", False)
    ]

    if meta_params:
        print(f"\n❌ Meta tensors found in {where}:")
        for name in meta_params[:30]:
            print("  ", name)

        raise RuntimeError(
            f"{where} still has {len(meta_params)} meta parameters. "
            "This causes: Cannot copy out of meta tensor; no data! "
            "Load the model with device_map=None and low_cpu_mem_usage=False."
        )

    print(f"✓ No meta tensors in {where}")


def print_tokenizer_special_tokens(tokenizer, title="Tokenizer special tokens"):
    print(f"\n{title}:")
    print(f"  bos_token: {tokenizer.bos_token!r} -> {tokenizer.bos_token_id}")
    print(f"  eos_token: {tokenizer.eos_token!r} -> {tokenizer.eos_token_id}")
    print(f"  pad_token: {tokenizer.pad_token!r} -> {tokenizer.pad_token_id}")

    for tok in ["<|im_end|>", "<|endoftext|>", *REACT_TOKENS]:
        tid = tokenizer.convert_tokens_to_ids(tok)
        print(f"  {tok:20s} -> {tid}")


def ensure_chat_template_from_base(tokenizer):
    """
    如果训练 tokenizer cache 里没有 chat_template，则从基座 tokenizer 补回来。
    """
    if getattr(tokenizer, "chat_template", None):
        print("✓ tokenizer already has chat_template")
        return tokenizer

    try:
        base_tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_PATH,
            token=HF_TOKEN,
            trust_remote_code=True,
        )
        if getattr(base_tokenizer, "chat_template", None):
            tokenizer.chat_template = base_tokenizer.chat_template
            print("✓ chat_template copied from base tokenizer")
        else:
            print("⚠️ Base tokenizer has no chat_template either.")
    except Exception as e:
        print(f"⚠️ Could not load base tokenizer for chat_template: {e}")

    return tokenizer


def load_merge_tokenizer():
    """
    加载 merge 使用的 tokenizer。

    优先级:
      1. TOKENIZER_CACHE
      2. ADAPTER_PATH
      3. 如果 ALLOW_TOKENIZER_RECONSTRUCT=1，从 base tokenizer 重建
      4. 否则直接报错
    """
    banner("[1] Resolving tokenizer cache")

    candidates = []

    if TOKENIZER_CACHE:
        candidates.append(("TOKENIZER_CACHE", TOKENIZER_CACHE))

    if ADAPTER_PATH and ADAPTER_PATH != TOKENIZER_CACHE:
        candidates.append(("ADAPTER_PATH", ADAPTER_PATH))

    for name, path in candidates:
        print(f"Checking {name}: {path}")

        if has_tokenizer_files(path):
            print(f"✓ Using tokenizer from {name}: {path}")
            tokenizer = AutoTokenizer.from_pretrained(
                path,
                token=HF_TOKEN,
                trust_remote_code=True,
            )
            return tokenizer, path

        print(f"  - Not usable as tokenizer cache.")

    print("\n❌ No tokenizer cache found.")
    print(f"  TOKENIZER_CACHE = {TOKENIZER_CACHE}")
    print(f"  ADAPTER_PATH    = {ADAPTER_PATH}")

    if not ALLOW_TOKENIZER_RECONSTRUCT:
        raise FileNotFoundError(
            "Tokenizer cache not found.\n\n"
            "Do NOT merge blindly, because ReAct token IDs may differ from training.\n"
            "Fix options:\n"
            "  1. Set TOKENIZER_CACHE to the tokenizer saved during SFT training/checkpoint.\n"
            "  2. Use ADAPTER_PATH pointing to a checkpoint directory containing tokenizer.json.\n"
            "  3. If you are absolutely sure the base tokenizer and add_special_tokens order are identical,\n"
            "     rerun with ALLOW_TOKENIZER_RECONSTRUCT=1.\n"
        )

    print("\n⚠️ ALLOW_TOKENIZER_RECONSTRUCT=1 enabled.")
    print("⚠️ Reconstructing tokenizer from base model + ReAct special tokens.")
    print("⚠️ You MUST verify ReAct token IDs before trusting the merged model.")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        token=HF_TOKEN,
        trust_remote_code=True,
    )

    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": REACT_TOKENS}
    )
    print(f"Added special tokens: {added}")

    return tokenizer, "RECONSTRUCTED_FROM_BASE"


def get_new_token_ids(tokenizer):
    token_ids = {t: tokenizer.convert_tokens_to_ids(t) for t in REACT_TOKENS}

    print("\nReAct token IDs:")
    for t, tid in token_ids.items():
        ok = (
            tid is not None
            and tid >= 0
            and tid != tokenizer.unk_token_id
        )
        marker = "✓" if ok else "✗"
        print(f"  {marker} {t:20s} -> {tid}")

    bad_tokens = [
        t for t, tid in token_ids.items()
        if tid is None or tid < 0 or tid == tokenizer.unk_token_id
    ]

    if bad_tokens:
        raise RuntimeError(
            f"Some ReAct tokens are missing from tokenizer: {bad_tokens}. "
            "Use the tokenizer cache saved during training."
        )

    return list(token_ids.values())


def find_visual_merger_norm(model):
    for name, module in model.named_modules():
        if name.endswith("visual.merger") or name == "model.visual.merger":
            if hasattr(module, "norm") and hasattr(module.norm, "weight"):
                return name, module.norm.weight

    return None, None


def sync_special_token_config(model, tokenizer):
    """
    同步 tokenizer / model.config / text_config / generation_config 的特殊 token。
    """
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    def patch_config(cfg, name):
        if cfg is None:
            return

        if bos_id is not None:
            cfg.bos_token_id = bos_id
        if eos_id is not None:
            cfg.eos_token_id = eos_id
        if pad_id is not None:
            cfg.pad_token_id = pad_id

        if hasattr(cfg, "use_cache"):
            cfg.use_cache = True

        print(
            f"✓ Synced {name}: "
            f"bos={getattr(cfg, 'bos_token_id', None)}, "
            f"eos={getattr(cfg, 'eos_token_id', None)}, "
            f"pad={getattr(cfg, 'pad_token_id', None)}, "
            f"use_cache={getattr(cfg, 'use_cache', None)}"
        )

    patch_config(model.config, "model.config")

    if hasattr(model.config, "text_config"):
        patch_config(model.config.text_config, "model.config.text_config")

    if hasattr(model, "generation_config") and model.generation_config is not None:
        patch_config(model.generation_config, "model.generation_config")
    else:
        print("⚠️ model.generation_config not found. Creating a new GenerationConfig.")
        model.generation_config = GenerationConfig()
        patch_config(model.generation_config, "model.generation_config")


def verify_saved_special_token_config(output_dir):
    """
    保存后重新加载 tokenizer / config / generation_config，验证 eos/pad 是否一致。
    """
    banner("[Verify] Reloading saved config/tokenizer")

    tok = AutoTokenizer.from_pretrained(
        output_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    cfg = AutoConfig.from_pretrained(
        output_dir,
        trust_remote_code=True,
        local_files_only=True,
    )

    try:
        gen_cfg = GenerationConfig.from_pretrained(
            output_dir,
            local_files_only=True,
        )
    except Exception as e:
        print(f"⚠️ Could not load generation_config from saved dir: {e}")
        gen_cfg = None

    print("\nReloaded tokenizer:")
    print(f"  eos={tok.eos_token_id}, pad={tok.pad_token_id}, bos={tok.bos_token_id}")

    print("Reloaded model config:")
    print(
        f"  eos={getattr(cfg, 'eos_token_id', None)}, "
        f"pad={getattr(cfg, 'pad_token_id', None)}, "
        f"bos={getattr(cfg, 'bos_token_id', None)}"
    )

    if hasattr(cfg, "text_config"):
        print("Reloaded text_config:")
        print(
            f"  eos={getattr(cfg.text_config, 'eos_token_id', None)}, "
            f"pad={getattr(cfg.text_config, 'pad_token_id', None)}, "
            f"bos={getattr(cfg.text_config, 'bos_token_id', None)}"
        )

    if gen_cfg is not None:
        print("Reloaded generation_config:")
        print(
            f"  eos={getattr(gen_cfg, 'eos_token_id', None)}, "
            f"pad={getattr(gen_cfg, 'pad_token_id', None)}, "
            f"bos={getattr(gen_cfg, 'bos_token_id', None)}"
        )

    problems = []

    if getattr(cfg, "eos_token_id", None) != tok.eos_token_id:
        problems.append("config.eos_token_id != tokenizer.eos_token_id")
    if getattr(cfg, "pad_token_id", None) != tok.pad_token_id:
        problems.append("config.pad_token_id != tokenizer.pad_token_id")

    if hasattr(cfg, "text_config"):
        if getattr(cfg.text_config, "eos_token_id", None) != tok.eos_token_id:
            problems.append("text_config.eos_token_id != tokenizer.eos_token_id")
        if getattr(cfg.text_config, "pad_token_id", None) != tok.pad_token_id:
            problems.append("text_config.pad_token_id != tokenizer.pad_token_id")

    if gen_cfg is not None:
        if getattr(gen_cfg, "eos_token_id", None) != tok.eos_token_id:
            problems.append("generation_config.eos_token_id != tokenizer.eos_token_id")
        if getattr(gen_cfg, "pad_token_id", None) != tok.pad_token_id:
            problems.append("generation_config.pad_token_id != tokenizer.pad_token_id")

    for react_tok in REACT_TOKENS:
        tid = tok.convert_tokens_to_ids(react_tok)
        if tid is None or tid < 0 or tid == tok.unk_token_id:
            problems.append(f"Missing ReAct token after save: {react_tok}")

    if problems:
        print("\n❌ Saved config/tokenizer verification failed:")
        for p in problems:
            print(f"  - {p}")
        raise RuntimeError("Saved config/tokenizer are still inconsistent.")

    print("\n✅ Saved config/tokenizer/generation_config are consistent.")


def merge():
    # ------------------------------------------------------------------
    # 1) Tokenizer
    # ------------------------------------------------------------------
    tokenizer, tokenizer_source = load_merge_tokenizer()

    tokenizer = ensure_chat_template_from_base(tokenizer)

    if tokenizer.pad_token_id is None:
        print("⚠️ tokenizer.pad_token_id is None. Setting pad_token = eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    print_tokenizer_special_tokens(
        tokenizer,
        title=f"Tokenizer special tokens from {tokenizer_source}",
    )

    new_token_ids = get_new_token_ids(tokenizer)

    print(f"\nNew token id range: {min(new_token_ids)} - {max(new_token_ids)}")
    print(f"Tokenizer total vocab: {len(tokenizer)}")

    # ------------------------------------------------------------------
    # 2) Base model
    # ------------------------------------------------------------------
    banner("[2] Loading base model")

    print("Loading base model WITHOUT device_map='auto' to avoid meta tensors...")
    print("This requires enough RAM/VRAM to materialize the full model.")

    base_model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL_PATH,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=None,
        low_cpu_mem_usage=False,
        token=HF_TOKEN,
        trust_remote_code=True,
    )

    assert_no_meta_tensors(base_model, "base_model after load")

    if torch.cuda.is_available():
        print("Moving base model to CUDA...")
        base_model = base_model.to("cuda")
    else:
        print("CUDA not available; keeping base model on CPU.")

    assert_no_meta_tensors(base_model, "base_model after device move")

    print(f"Original base embedding: {base_model.get_input_embeddings().weight.shape}")

    print(f"Resizing token embeddings to {len(tokenizer)}...")
    base_model.resize_token_embeddings(len(tokenizer))
    print(f"After resize: {base_model.get_input_embeddings().weight.shape}")

    assert_no_meta_tensors(base_model, "base_model after resize_token_embeddings")

    base_in_emb_snapshot = None
    base_out_emb_snapshot = None

    if CHECK_NEW_TOKEN_EMBEDDINGS:
        with torch.no_grad():
            base_in_emb_snapshot = (
                base_model
                .get_input_embeddings()
                .weight[new_token_ids]
                .detach()
                .float()
                .cpu()
                .clone()
            )

            base_out_emb_snapshot = (
                base_model
                .get_output_embeddings()
                .weight[new_token_ids]
                .detach()
                .float()
                .cpu()
                .clone()
            )

    merger_norm_snapshot = None
    merger_name, merger_weight = find_visual_merger_norm(base_model)

    if CHECK_VISUAL_MERGER and merger_weight is not None:
        with torch.no_grad():
            merger_norm_snapshot = merger_weight.detach().float().cpu().clone()
        print(f"✓ Found visual merger norm: {merger_name}")
    elif CHECK_VISUAL_MERGER:
        print("⚠️ Could not find visual.merger.norm.weight. Skipping merger sanity check.")

    # ------------------------------------------------------------------
    # 3) Attach LoRA
    # ------------------------------------------------------------------
    banner("[3] Attaching LoRA adapter")
    print(f"Adapter path: {ADAPTER_PATH}")

    if not os.path.exists(ADAPTER_PATH):
        raise FileNotFoundError(f"ADAPTER_PATH not found: {ADAPTER_PATH}")

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
        is_trainable=False,
    )

    assert_no_meta_tensors(model, "PeftModel after adapter load")

    n_lora = sum(
        1 for n, _ in model.named_modules()
        if n.endswith(".lora_A.default")
    )

    print(f"LoRA adapters loaded: {n_lora}  (expected >= {MIN_EXPECTED_LORA})")

    if n_lora < MIN_EXPECTED_LORA:
        raise RuntimeError(
            f"Adapter count too low: {n_lora}. "
            "LoRA may not have loaded correctly."
        )

    # ------------------------------------------------------------------
    # 4) Merge
    # ------------------------------------------------------------------
    banner("[4] Merging and unloading")

    model = model.merge_and_unload()

    assert_no_meta_tensors(model, "merged model")

    # ------------------------------------------------------------------
    # 5) Sanity checks
    # ------------------------------------------------------------------
    banner("[5] Post-merge sanity checks")

    if CHECK_NEW_TOKEN_EMBEDDINGS:
        with torch.no_grad():
            merged_in_emb = (
                model
                .get_input_embeddings()
                .weight[new_token_ids]
                .detach()
                .float()
                .cpu()
            )

            merged_out_emb = (
                model
                .get_output_embeddings()
                .weight[new_token_ids]
                .detach()
                .float()
                .cpu()
            )

        in_emb_diff = (merged_in_emb - base_in_emb_snapshot).abs().mean().item()
        out_emb_diff = (merged_out_emb - base_out_emb_snapshot).abs().mean().item()

        print(f"Input embedding diff for ReAct tokens:  {in_emb_diff:.8f}")
        print(f"Output embedding diff for ReAct tokens: {out_emb_diff:.8f}")

        if in_emb_diff < 1e-5:
            print("⚠️  WARNING: Input embedding for ReAct tokens looks unchanged!")
            print("   Either trainable_token_indices/modules_to_save did not work or merge failed.")
        else:
            print("✓ Input embedding has been trained")

        if out_emb_diff < 1e-5:
            print("⚠️  WARNING: lm_head embedding for ReAct tokens looks unchanged!")
            print("   This may mean the model cannot generate the ReAct tokens correctly.")
        else:
            print("✓ lm_head embedding has been trained")
    else:
        print("Skipping ReAct token embedding sanity check.")

    if CHECK_VISUAL_MERGER and merger_norm_snapshot is not None:
        merged_merger_name, merged_merger_weight = find_visual_merger_norm(model)

        if merged_merger_weight is not None:
            with torch.no_grad():
                merged_merger_norm = merged_merger_weight.detach().float().cpu()

            merger_diff = (merged_merger_norm - merger_norm_snapshot).abs().mean().item()
            print(f"\nvisual.merger.norm diff: {merger_diff:.8f}")

            if merger_diff < 1e-7:
                print("⚠️  WARNING: visual.merger looks unchanged!")
                print("   modules_to_save may not have been applied.")
            else:
                print("✓ visual.merger has been trained")
        else:
            print("⚠️ Could not find visual.merger after merge. Skipping check.")
    elif not CHECK_VISUAL_MERGER:
        print("Skipping visual.merger sanity check.")

    # ------------------------------------------------------------------
    # 6) Config + Save
    # ------------------------------------------------------------------
    banner("[6] Sync config and save merged model")

    sync_special_token_config(model, tokenizer)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Saving model weights to {OUTPUT_DIR}...")
    model.save_pretrained(
        OUTPUT_DIR,
        safe_serialization=True,
        max_shard_size="5GB",
    )

    if hasattr(model, "generation_config") and model.generation_config is not None:
        print("Saving generation_config.json...")
        model.generation_config.save_pretrained(OUTPUT_DIR)

    # ------------------------------------------------------------------
    # 7) Processor
    # ------------------------------------------------------------------
    banner("[7] Saving processor")

    processor = AutoProcessor.from_pretrained(
        BASE_MODEL_PATH,
        token=HF_TOKEN,
        trust_remote_code=True,
    )

    processor.tokenizer = tokenizer
    processor.save_pretrained(OUTPUT_DIR)

    # ------------------------------------------------------------------
    # 8) Tokenizer
    # ------------------------------------------------------------------
    banner("[8] Saving tokenizer")

    tokenizer.save_pretrained(OUTPUT_DIR)

    # ------------------------------------------------------------------
    # 9) Re-save synced config files
    # ------------------------------------------------------------------
    banner("[9] Re-save synced config files")

    sync_special_token_config(model, tokenizer)

    model.config.save_pretrained(OUTPUT_DIR)

    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.save_pretrained(OUTPUT_DIR)

    # ------------------------------------------------------------------
    # 10) Copy adapter config
    # ------------------------------------------------------------------
    banner("[10] Copy adapter config")

    adapter_config = os.path.join(ADAPTER_PATH, "adapter_config.json")
    if os.path.exists(adapter_config):
        shutil.copy2(
            adapter_config,
            os.path.join(OUTPUT_DIR, "merged_from_adapter_config.json"),
        )
        print("✓ copied merged_from_adapter_config.json")
    else:
        print("⚠️ adapter_config.json not found; skip copy.")

    # ------------------------------------------------------------------
    # 11) Verify
    # ------------------------------------------------------------------
    verify_saved_special_token_config(OUTPUT_DIR)

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    banner(f"✅ Merged model saved to {OUTPUT_DIR}", char="=")

    print("\nFiles in output dir:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        path = os.path.join(OUTPUT_DIR, f)

        if not os.path.isfile(path):
            continue

        size = os.path.getsize(path)
        size_str = (
            f"{size / 1e9:.2f} GB" if size > 1e9
            else f"{size / 1e6:.1f} MB" if size > 1e6
            else f"{size / 1e3:.1f} KB"
        )

        print(f"  {f:55s}  {size_str}")


if __name__ == "__main__":
    merge()
