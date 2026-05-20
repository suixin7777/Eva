"""
eva_tools_runtime.py — Concrete tool executors + ReAct text parsers.

Extracted from eva_core.py during the P2 step-5 refactor. These functions
own all of Eva's outbound HTTP traffic (Tavily web search, DeepSeek expert
LLM, remote vision API) plus the small local helpers that parse the
ReAct-formatted text the model emits inside <|tool_code|> ... <|end_react|>.

No class state. No model dependency. Imports only:
- re, ast, base64, io        (stdlib)
- requests, openai           (3rd-party HTTP)
- API keys + REACT/TAG_RE    (eva_config)

Public surface (re-exported via eva_tools.py shim):
    sanitize_tool_code      ReAct call-string -> (name, kwargs)
    parse_react_block       ReAct block -> [(tag, text), ...]
    clean_search_content    de-noise web snippets
    websearch_tavily        raw Tavily call
    run_websearch           Tavily wrapper that returns dual-channel output
                            (for_model / for_user)
    call_deepseek_expert    OpenAI-compatible call to DeepSeek
    call_deepseek_judge     DeepSeek call in JSON-only mode for verifier
                            semantic classifiers
    call_remote_vision      Qwen-VL-Plus (or compatible) image+text call,
                            modes: 'ocr' | 'chat'
"""

import re

import requests
from openai import OpenAI

from eva_config import (
    TAG_RE,
    REACT,
    TAVILY_API_KEY,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    VISION_API_KEY,
    VISION_BASE_URL,
    VISION_MODEL,
)


__all__ = [
    "sanitize_tool_code",
    "parse_react_block",
    "clean_search_content",
    "websearch_tavily",
    "run_websearch",
    "call_deepseek_expert",
    "call_deepseek_judge",
    "call_remote_vision",
]


# ============================================================
# ReAct parsing
# ============================================================
def sanitize_tool_code(raw):
    if not raw or not raw.strip(): return "Unknown", {}
    m = re.search(r"^\s*([A-Za-z_]\w*)\s*\((.*)\)\s*$", raw, re.S)
    if not m:
        if raw.strip().startswith("{"): return "Unknown", {}
        raise ValueError(f"Invalid format: {raw}")
    name, body = m.group(1), m.group(2).strip()
    if not body: return name, {}
    import ast
    if body.startswith("{"):
        try:
            params = ast.literal_eval(body)
            if isinstance(params, dict): return name, params
        except Exception:
            pass
    try:
        params = {}
        parts = re.split(r",\s*(?=\w+\s*=)", body)
        for part in parts:
            part = part.strip()
            if not part: continue
            eq_pos = part.find("=")
            if eq_pos == -1: continue
            key = part[:eq_pos].strip()
            value_str = part[eq_pos + 1:].strip()
            try:
                value = ast.literal_eval(value_str)
            except Exception:
                value = value_str.strip("\"'")
            params[key] = value
        if params: return name, params
    except Exception:
        pass
    return name, {}


def parse_react_block(block):
    parts = []
    last_tag, last_pos = None, 0
    for m in TAG_RE.finditer(block):
        raw_tag, start = m.group(1), m.end()
        if raw_tag == "<think>": tag = "thought"
        elif raw_tag == "</think>": tag = "thought_end"
        else: tag = raw_tag.strip("<|>")
        if last_tag is not None: parts.append((last_tag, block[last_pos:m.start()].strip()))
        last_tag, last_pos = tag, start
    end_pos = block.find(REACT["end"])
    if end_pos == -1: end_pos = len(block)
    if last_tag is not None and last_pos <= end_pos:
        parts.append((last_tag, block[last_pos:end_pos].strip()))
    return parts


# ============================================================
# Web search (Tavily)
# ============================================================
def clean_search_content(text, max_len=600):
    if not text: return ""
    cleaned = re.sub(
        r"User avatar|Skip to content|Powered by phpBB|Style by Arty|Post by|Sign in to edit|<<rtjson>>.*?<</rtjson>>|\[\.\.\.\]",
        " ", text, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    if len(cleaned) > max_len:
        stop_index = max([cleaned.find(p, max_len - 50) for p in ".!?"] + [max_len])
        cleaned = cleaned[:stop_index] + "..."
    return cleaned


def websearch_tavily(query):
    if not TAVILY_API_KEY:
        return [{"title": "Error", "link": "", "content": "Tavily API Key missing.", "age": ""}]
    url = "https://api.tavily.com/search"
    payload = {"api_key": TAVILY_API_KEY.strip(), "query": query, "search_depth": "advanced",
               "include_answer": True, "include_images": False, "include_raw_content": False,
               "max_results": 5, "topic": "general"}
    try:
        resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results = []
        answer = data.get("answer")
        if answer and answer.strip():
            results.append({"title": "Tavily AI Answer", "link": "https://tavily.com", "content": answer.strip(), "age": "Now"})
        for item in data.get("results", []):
            clean = clean_search_content(item.get("content", ""))
            if len(clean) < 20: continue
            results.append({"title": item.get("title", "No Title"), "link": item.get("url", "#"), "content": clean, "age": ""})
        return results or [{"title": "No Results", "link": "", "content": "No results found.", "age": ""}]
    except Exception as e:
        return [{"title": "Error", "link": "", "content": f"[Tavily Error] {e}", "age": ""}]


def run_websearch(params):
    query = params.get("query", "")
    results = websearch_tavily(query)
    if not isinstance(results, list) or "Error" in results[0]["title"]:
        return {"for_model": str(results), "for_user": "Search failed."}
    user_lines, ai_answers = [], []
    for res in results:
        if res['title'] == "Tavily AI Answer":
            user_lines.append(f"- **[{res['title']}]({res['link']})**\n> {res['content']}")
            ai_answers.append(res['content'])
        else:
            user_lines.append(f"- **[{res['title']}]({res['link']})**")
    return {"for_model": "\n".join(ai_answers), "for_user": "\n\n".join(user_lines)}


# ============================================================
# Remote LLM expert (DeepSeek)
# ============================================================
def call_deepseek_expert(prompt):
    """TextGenerationTool 调用的 LLM expert。
    模型走 eva_config.TEXTGEN_MODEL（默认 deepseek-v4-pro；可通过
    EVA_TEXTGEN_MODEL env var 覆盖）。
    """
    if not prompt or not prompt.strip(): return "[Error] Prompt is empty."
    # Lazy import：避免循环 import + 让 env var 改动通过 process 重启就生效
    from eva_config import TEXTGEN_MODEL
    print(
        f"\n        | --- CALLING EXPERT ---\n"
        f"        | [System] Calling DeepSeek Expert (model={TEXTGEN_MODEL})..."
    )
    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        resp = client.chat.completions.create(
            model=TEXTGEN_MODEL,
            messages=[
                {"role": "system", "content": "You are a precise expert assistant. Answer directly and professionally. Rules:\n1. No greetings or filler words.\n2. Start immediately with the solution."},
                {"role": "user", "content": prompt}
            ],
            stream=False,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"[DeepSeek API Error] {e}"


# ============================================================
# Remote LLM judge (DeepSeek, JSON-only mode)
# Used by the verifier's semantic intent classifiers as a paraphrase-
# tolerant fallback for the regex layer.
#
# Contract:
#   Input:
#     - system_instruction: rules + the JSON schema the judge must obey
#     - user_payload:       the question to classify (plain text)
#   Output (always a dict, never raises):
#     - on success: parsed JSON from the model
#     - on any error: {"ok": None, "error": "<reason>"}
#       Callers MUST distinguish "ok": False (judge said no) from
#       "ok": None (judge crashed / fell back); the latter means the
#       caller should defer to the regex path, never assume a verdict.
#
# Sampling: temperature 0 (config-driven) so the same input gives the
# same verdict turn after turn; otherwise verifier-driven repair could
# loop on stochastic flip-flops.
# ============================================================
def call_deepseek_judge(system_instruction, user_payload, *, debug=False,
                        timeout=None, model=None):
    import json as _json
    if not user_payload or not str(user_payload).strip():
        return {"ok": None, "error": "empty_payload"}
    # Lazy import — these come from eva_config but we don't want a
    # module-level dependency if a user disables the judge entirely.
    from eva_config import (
        LLM_JUDGE_TEMPERATURE,
        LLM_JUDGE_TIMEOUT_SECONDS,
    )
    effective_timeout = float(timeout) if timeout is not None else LLM_JUDGE_TIMEOUT_SECONDS
    effective_model = model or "deepseek-v4-flash"
    if debug:
        print("        | --- CALLING JUDGE ---")
        print(f"        | [System] DeepSeek judge: payload={str(user_payload)[:80]!r} "
              f"(timeout={effective_timeout}s, model={effective_model})")
    try:
        client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            timeout=effective_timeout,
        )
        resp = client.chat.completions.create(
            model=effective_model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": str(user_payload)},
            ],
            temperature=LLM_JUDGE_TEMPERATURE,
            response_format={"type": "json_object"},
            stream=False,
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            return {"ok": None, "error": "empty_response"}
        try:
            parsed = _json.loads(text)
        except Exception as e:
            return {"ok": None, "error": f"json_parse_failed: {e}",
                    "raw": text[:200]}
        if not isinstance(parsed, dict):
            return {"ok": None, "error": "non_dict_response",
                    "raw": text[:200]}
        return parsed
    except Exception as e:
        return {"ok": None, "error": f"{type(e).__name__}: {e}"}


# ============================================================
# Remote vision (Qwen-VL-Plus or compatible OpenAI-style endpoint)
# ============================================================
def call_remote_vision(query, image, mode="chat"):
    if image is None:
        return "[AskRemoteVision] No image is attached to this conversation."
    import base64, io
    print(f"\n        | --- CALLING VISION ---\n        | [System] Calling Vision ({VISION_MODEL}) in '{mode}' mode...")
    if mode.lower() == "ocr":
        system_instruction = ("You are a highly accurate OCR engine and data extractor. "
            "Your sole task is to extract text, code, math equations, or structural data from the image exactly as it appears. "
            "Preserve indentations, line breaks, and exact characters. DO NOT add conversational filler. Output ONLY the extracted content.")
    else:
        system_instruction = ("You are an expert visual analysis AI. "
            "Analyze the image deeply, describe relevant context, explain jokes, memes, or complex diagrams. "
            "Focus on answering the user's specific query thoughtfully based on the visual evidence.")
    try:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        client = OpenAI(api_key=VISION_API_KEY, base_url=VISION_BASE_URL)
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": query}
            ]}
        ]
        resp = client.chat.completions.create(model=VISION_MODEL, messages=messages, stream=False)
        return resp.choices[0].message.content
    except Exception as e:
        return f"[Vision Error] {e}"
