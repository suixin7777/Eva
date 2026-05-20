"""
eva_render.py — Pure ChatML / message rendering helpers.

Extracted from eva_core.py during the P2 step-2 refactor. These functions
have no class state, no global mutability, no model dependency. They turn
a (system_prompt, list[messages]) pair into a flat ChatML string plus a
parallel list of PIL images.

Public surface (re-exported via `from eva_render import *` in eva_core.py):
    wrap_chatml, _wrap
    clean_visual_tags, clean_user_text
    resize_image_if_needed, safe_load_image
    _resolve_message_image, _image_path_hint
    render_messages_to_prompt
"""

import os
import re

from PIL import Image

from eva_config import IMAGE_BLOCK


__all__ = [
    "wrap_chatml",
    "_wrap",
    "clean_visual_tags",
    "clean_user_text",
    "resize_image_if_needed",
    "safe_load_image",
    "_resolve_message_image",
    "_image_path_hint",
    "render_messages_to_prompt",
]


# ============================================================
# ChatML role wrapping
# ============================================================
def wrap_chatml(role: str, content: str, complete: bool = True) -> str:
    if role == "system":
        return f"<|im_start|>system\n{content}<|im_end|>\n"
    if role == "user":
        return f"<|im_start|>user\n{content}<|im_end|>\n"
    if role == "tool":
        return f"<|im_start|>tool\n{content}<|im_end|>\n"
    if role == "assistant":
        return (
            f"<|im_start|>assistant\n{content}<|im_end|>\n"
            if complete
            else f"<|im_start|>assistant\n{content}"
        )
    return str(content)


# Legacy short alias used in a few internal call sites.
_wrap = wrap_chatml


# ============================================================
# Visual / vision-tag cleanup
# ============================================================
def clean_visual_tags(text, replacement=""):
    if not isinstance(text, str):
        return ""
    text = re.sub(
        r"<\|vision_start\|>.*?<\|vision_end\|>",
        replacement,
        text,
        flags=re.DOTALL,
    )
    return (
        text.replace("<|image|>", replacement)
            .replace("<|image_pad|>", "")
            .strip()
    )


def clean_user_text(text):
    return clean_visual_tags(text, replacement="")


# ============================================================
# Image loading & resizing
# ============================================================
def resize_image_if_needed(img, max_pixels=1605632):
    current_pixels = img.width * img.height
    if current_pixels <= max_pixels:
        return img
    ratio = (max_pixels / current_pixels) ** 0.5
    new_w = max(1, int(img.width * ratio))
    new_h = max(1, int(img.height * ratio))
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def safe_load_image(img_path, max_pixels=1605632):
    if not img_path or not isinstance(img_path, str):
        return None
    if not os.path.exists(img_path):
        return None
    try:
        img = Image.open(img_path).convert("RGB")
        return resize_image_if_needed(img, max_pixels=max_pixels)
    except Exception:
        return None


def _resolve_message_image(message, max_pixels=1605632):
    img = message.get("image")
    if isinstance(img, Image.Image):
        return resize_image_if_needed(img.convert("RGB"), max_pixels=max_pixels)
    return safe_load_image(message.get("image_path"), max_pixels=max_pixels)


def _image_path_hint(message):
    image_path = message.get("image_path")
    image_id = message.get("image_id")
    if image_path:
        return f"[Image path for AskRemoteVision: {image_path}]"
    if image_id:
        return f"[Image path for AskRemoteVision: {image_id}]"
    return "[Image attached]"


# ============================================================
# Top-level message renderer
# ============================================================
def render_messages_to_prompt(
    system_prompt,
    messages,
    *,
    include_assistant_prefix=True,
    image_placeholder="[Image attached]",
    max_pixels=1605632,
    skip_system_messages=True,
):
    prompt_parts = [wrap_chatml("system", system_prompt, complete=True)]
    prompt_images = []
    for msg in messages:
        role = (msg.get("role") or "").lower()
        content = msg.get("content") or ""
        if role == "system":
            if skip_system_messages:
                continue
            prompt_parts.append(
                wrap_chatml(
                    "system",
                    clean_visual_tags(content, replacement=image_placeholder),
                    complete=True,
                )
            )
            continue
        if role == "user":
            body = clean_user_text(content)
            img = _resolve_message_image(msg, max_pixels=max_pixels)
            if img is not None:
                hint = _image_path_hint(msg)
                body = (
                    f"{IMAGE_BLOCK}\n{hint}\n{body}"
                    if body
                    else f"{IMAGE_BLOCK}\n{hint}"
                )
                prompt_images.append(img)
            else:
                body = clean_visual_tags(content, replacement=image_placeholder)
            prompt_parts.append(wrap_chatml("user", body, complete=True))
            continue
        if role == "assistant":
            prompt_parts.append(
                wrap_chatml(
                    "assistant",
                    clean_visual_tags(content, replacement=image_placeholder),
                    complete=msg.get("complete", True),
                )
            )
            continue
        if role == "tool":
            prompt_parts.append(
                wrap_chatml(
                    "tool",
                    clean_visual_tags(content, replacement=image_placeholder),
                    complete=True,
                )
            )
            continue
    if include_assistant_prefix:
        prompt_parts.append(wrap_chatml("assistant", "", complete=False))
    return "".join(prompt_parts), prompt_images
