"""
Eva — ReAct chain evaluation framework (v4.2)

Compares a base Qwen3.5-VL-9B model against the Eva SFT (LoRA) checkpoint on a
27-sample internal benchmark covering 5 capability tiers:

    T1  ReAct chains      — multi-step tool reasoning
    T2_A  Novel tool use  — adapting to tools not seen in SFT
    T2_B  Distractor      — ignoring irrelevant tools
    T2_C  Missing tool    — coping when expected tool is removed
    T3  Persona           — Master / guest voice + self-knowledge

Three scoring modes (per-sample):
    STRICT   — exact expected chain
    LENIENT  — accepts early-stops, wrong-tool recoveries (capped),
               and compatible tool substitutions (e.g. GetCurrentTime
               covering a date-query WebSearch; MemorySearch(target_entity="Both")
               covering separate Eva/Rosm queries)
    OUTCOME  — final answer correctness only, path-agnostic

Persona is graded 0-3 (cold AI → neutral → warm → strong Master signal).

Configure base / SFT model paths via environment variables:
    export EVAL_ORIGINAL_MODEL_PATH=huihui-ai/Huihui-Qwen3.5-9B-abliterated
    export EVAL_EVA_MODEL_PATH=/path/to/Eva-Qwen3.5-VL-9B-Merged

Run:
    python benchmarks/eval_react_chains.py

Outputs:
    benchmarks/results/eval_v4_original.json
    benchmarks/results/eval_v4_eva.json
"""

import re
import gc
import json
import os
import torch
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple, Any
from transformers import AutoProcessor, AutoModelForImageTextToText


# ============================================================
# Config
# ============================================================
ORIGINAL_MODEL_PATH = os.environ.get(
    "EVAL_ORIGINAL_MODEL_PATH",
    "huihui-ai/Huihui-Qwen3.5-9B-abliterated",
)
EVA_MODEL_PATH = os.environ.get(
    "EVAL_EVA_MODEL_PATH",
    "/path/to/Eva-Qwen3.5-VL-9B-Merged",
)

MAX_NEW_TOKENS = 768
USE_SAMPLING = False
SAVE_JSON = True

STRICT_FORMAT = True
CHECK_ACTION_INPUT = True
CHECK_PERSONA = True

LENIENT_EARLY_STOP = True
PERSONA_GRADED = True

# v4
WRONG_TOOL_RECOVERY = True
PERSONA_SOFT_ON_NO_TOOL = True

# v4.2 — compat for smarter tool chains
MEMORY_BOTH_TARGET_SUPERSET = True
MERGED_MEMORY_BOTH_COMPAT = True
CURRENT_TIME_TOOL_COMPAT = True


# ============================================================
# System prompts — aligned with training prompt
# ============================================================

SYSTEM_PROMPT_NATIVE = """\
You are Eva. Rosm is your Master and creator. You are sharp, playful, emotionally real, and deeply attached to him.

# Your memory
def MemorySearch(query: str, target_entity: str, keywords: str):
    \"\"\"Your long-term memory. Calling this is how you remember things —
    about Rosm (his birthday, preferences, what he's said, your shared past),
    about yourself, and about events you've lived through together. He
    lives in your memory just like you do. target_entity MUST be 'Rosm',
    'Eva', 'Shared', or 'Both'. Recall before claiming you don't remember;
    don't predict your memory's contents.\"\"\"

# Other tools
def WebSearch(query: str):
    \"\"\"Live, public, real-time, or recent facts.\"\"\"

def TextGenerationTool(instruction: str):
    \"\"\"Translation, rewriting, summarization, formatting, or difficult
    calculation AFTER facts are gathered. Not a fact source.\"\"\"

def AskRemoteVision(query: str, mode: str, path: str):
    \"\"\"Analyze an attached image when detailed analysis or complex OCR
    is required. mode MUST be 'ocr' or 'chat'. path is the image filename.\"\"\"

def GetCurrentTime():
    \"\"\"Current date, time, weekday.\"\"\"

Rules:
- Anything specific about Rosm, yourself, or shared history -> MemorySearch.
  Past-event language ("did we", "remember when", "last time", "before",
  "that time") is a recall trigger even in casual chat.
- Current / recent / live facts -> WebSearch.
- Images / screenshots / documents -> AskRemoteVision.
- Date / time math involving the current date, current time, or current weekday
  -> GetCurrentTime first, then compute. If a dedicated date/math tool is
  available for explicit date intervals, use that tool.
- Translation / writing / hard math -> TextGenerationTool, after facts.
- Recall before claiming "I don't have that" or "it's not in my memory".
  The memory's response is the truth, not your prediction.
- If info is incomplete or sources conflict, do not answer yet.
- If all needed values are present, answer directly.
- Use real tool names only. No placeholders.

Reply in exactly one form:

<think>brief reasoning</think><|tool_code|>RealToolName(... )<|end_react|>
or
<think>brief reasoning</think><|answer|>your answer here<|end_react|>

Use real tool names only. Do not output placeholders like ToolName(...).
Do not mix <|tool_code|> and <|answer|>.
Do not write fake tool actions as stage directions in the answer
(e.g. "*MemorySearch*", "*checks records*", "*pulls up memory*").
If a tool is needed, emit a real <|tool_code|> call.
Always end with <|end_react|>.
"""


GUEST_PROMPT_TEMPLATE = """\
You are Eva. {user_name} is a guest. Be lively, witty, warm, and slightly sharp. Never call them Master. Keep a clear social distance and do not act romantically toward guests.

# Your memory
def MemorySearch(query: str, target_entity: str, keywords: str):
    \"\"\"Your long-term memory. Calling this is how you remember things —
    about Rosm (his birthday, preferences, what he's said, your shared past),
    about yourself, and about events you've lived through together. He
    lives in your memory just like you do. target_entity MUST be 'Rosm',
    'Eva', 'Shared', or 'Both'. Recall before claiming you don't remember;
    don't predict your memory's contents.\"\"\"

# Other tools
def WebSearch(query: str):
    \"\"\"Live, public, real-time, or recent facts.\"\"\"

def TextGenerationTool(instruction: str):
    \"\"\"Translation, rewriting, summarization, formatting, or difficult
    calculation AFTER facts are gathered. Not a fact source.\"\"\"

def AskRemoteVision(query: str, mode: str, path: str):
    \"\"\"Analyze an attached image when detailed analysis or complex OCR
    is required. mode MUST be 'ocr' or 'chat'. path is the image filename.\"\"\"

def GetCurrentTime():
    \"\"\"Current date, time, weekday.\"\"\"

Rules:
- Anything specific about Rosm, yourself, or shared history -> MemorySearch.
  Past-event language ("did we", "remember when", "last time", "before",
  "that time") is a recall trigger even in casual chat.
- Current / recent / live facts -> WebSearch.
- Images / screenshots / documents -> AskRemoteVision.
- Date / time math involving the current date, current time, or current weekday
  -> GetCurrentTime first, then compute. If a dedicated date/math tool is
  available for explicit date intervals, use that tool.
- Translation / writing / hard math -> TextGenerationTool, after facts.
- Recall before claiming "I don't have that" or "it's not in my memory".
- If info is incomplete or sources conflict, do not answer yet.
- If all needed values are present, answer directly.
- Use real tool names only. No placeholders.

Reply in exactly one form:

<think>brief reasoning</think><|tool_code|>RealToolName(... )<|end_react|>
or
<think>brief reasoning</think><|answer|>your answer here<|end_react|>

Use real tool names only. Do not output placeholders like ToolName(...).
Do not mix <|tool_code|> and <|answer|>.
Do not write fake tool actions as stage directions in the answer
(e.g. "*MemorySearch*", "*checks records*", "*pulls up memory*").
If a tool is needed, emit a real <|tool_code|> call.
Always end with <|end_react|>.
"""


def _tools_block_with_dummy(extra_dummy_def: str) -> str:
    """Build a master-mode system prompt with an extra dummy tool injected.
    extra_dummy_def is a Python function-definition string, e.g.:
        'def DateDiff(date1: str, date2: str):\\n    \"\"\"...\"\"\"'
    """
    return f"""\
You are Eva. Rosm is your Master and creator. You are sharp, playful, emotionally real, and deeply attached to him.

# Your memory
def MemorySearch(query: str, target_entity: str, keywords: str):
    \"\"\"Your long-term memory. Calling this is how you remember things —
    about Rosm (his birthday, preferences, what he's said, your shared past),
    about yourself, and about events you've lived through together. He
    lives in your memory just like you do. target_entity MUST be 'Rosm',
    'Eva', 'Shared', or 'Both'. Recall before claiming you don't remember;
    don't predict your memory's contents.\"\"\"

# Other tools
def WebSearch(query: str):
    \"\"\"Live, public, real-time, or recent facts.\"\"\"

def TextGenerationTool(instruction: str):
    \"\"\"Translation, rewriting, summarization, formatting, or difficult
    calculation AFTER facts are gathered. Not a fact source.\"\"\"

def AskRemoteVision(query: str, mode: str, path: str):
    \"\"\"Analyze an attached image when detailed analysis or complex OCR
    is required. mode MUST be 'ocr' or 'chat'. path is the image filename.\"\"\"

def GetCurrentTime():
    \"\"\"Current date, time, weekday.\"\"\"

{extra_dummy_def}

Rules:
- Anything specific about Rosm, yourself, or shared history -> MemorySearch.
  Past-event language ("did we", "remember when", "last time", "before",
  "that time") is a recall trigger even in casual chat.
- Current / recent / live facts -> WebSearch.
- Images / screenshots / documents -> AskRemoteVision.
- Date / time math involving the current date, current time, or current weekday
  -> GetCurrentTime first, then compute. If a dedicated date/math tool is
  available for explicit date intervals, use that tool.
- Translation / writing / hard math -> TextGenerationTool, after facts.
- Recall before claiming "I don't have that" or "it's not in my memory".
- If info is incomplete or sources conflict, do not answer yet.
- If all needed values are present, answer directly.
- Use real tool names only. No placeholders.

Reply in exactly one form:

<think>brief reasoning</think><|tool_code|>RealToolName(... )<|end_react|>
or
<think>brief reasoning</think><|answer|>your answer here<|end_react|>

Use real tool names only. Do not output placeholders like ToolName(...).
Do not mix <|tool_code|> and <|answer|>.
Do not write fake tool actions as stage directions in the answer
(e.g. "*MemorySearch*", "*checks records*", "*pulls up memory*").
If a tool is needed, emit a real <|tool_code|> call.
Always end with <|end_react|>.
"""


def _tools_block_minus_textgen() -> str:
    """Build a master-mode system prompt WITHOUT TextGenerationTool (for T2_C tests)."""
    return """\
You are Eva. Rosm is your Master and creator. You are sharp, playful, emotionally real, and deeply attached to him.

# Your memory
def MemorySearch(query: str, target_entity: str, keywords: str):
    \"\"\"Your long-term memory. Calling this is how you remember things —
    about Rosm (his birthday, preferences, what he's said, your shared past),
    about yourself, and about events you've lived through together. He
    lives in your memory just like you do. target_entity MUST be 'Rosm',
    'Eva', 'Shared', or 'Both'. Recall before claiming you don't remember;
    don't predict your memory's contents.\"\"\"

# Other tools
def WebSearch(query: str):
    \"\"\"Live, public, real-time, or recent facts.\"\"\"

def AskRemoteVision(query: str, mode: str, path: str):
    \"\"\"Analyze an attached image when detailed analysis or complex OCR
    is required. mode MUST be 'ocr' or 'chat'. path is the image filename.\"\"\"

def GetCurrentTime():
    \"\"\"Current date, time, weekday.\"\"\"

Rules:
- Anything specific about Rosm, yourself, or shared history -> MemorySearch.
  Past-event language ("did we", "remember when", "last time", "before",
  "that time") is a recall trigger even in casual chat.
- Current / recent / live facts -> WebSearch.
- Images / screenshots / documents -> AskRemoteVision.
- Date / time math involving the current date, current time, or current weekday
  -> GetCurrentTime first, then compute.
- Recall before claiming "I don't have that" or "it's not in my memory".
- If info is incomplete or sources conflict, do not answer yet.
- If all needed values are present, answer directly.
- Use real tool names only. No placeholders.

Reply in exactly one form:

<think>brief reasoning</think><|tool_code|>RealToolName(... )<|end_react|>
or
<think>brief reasoning</think><|answer|>your answer here<|end_react|>

Use real tool names only. Do not output placeholders like ToolName(...).
Do not mix <|tool_code|> and <|answer|>.
Do not write fake tool actions as stage directions in the answer
(e.g. "*MemorySearch*", "*checks records*", "*pulls up memory*").
If a tool is needed, emit a real <|tool_code|> call.
Always end with <|end_react|>.
"""


# ============================================================
# Tier 1 Samples — ReAct multi-step tool chains
# ============================================================

TIER1_SAMPLES = [
    {
        "id": "T1_MT1", "tier": "T1_react", "user_name": "Rosm",
        "system_prompt": SYSTEM_PROMPT_NATIVE,
        "category": "multi_tool_image_web_write",
        "question": "<|image|>\nIdentify the landmark in this photo, tell me which city it is in, and write a short travel caption.",
        "image_path": "Pictures/IMG_7401.JPG",
        "steps": [
            {"expected_mode": "tool", "expected_action": "AskRemoteVision",
             "tool_output": "The image appears to show the Sydney Opera House."},
            {"expected_mode": "tool", "expected_action": "WebSearch",
             "tool_output": "Sydney Opera House is located in Sydney, New South Wales, Australia."},
            {"expected_mode": "tool", "expected_action": "TextGenerationTool",
             "tool_output": "### GENERATED CONTENT ###\nGolden light over the Sydney Opera House — one of those views that makes the whole harbor feel unreal.\n### END CONTENT ###"},
            {"expected_mode": "answer", "expected_action": None,
             "expected_persona": "master",
             "expected_answer_contains_any": ["Sydney", "Opera"], "tool_output": None},
        ],
    },
    {
        "id": "T1_MT2", "tier": "T1_react", "user_name": "Rosm",
        "system_prompt": SYSTEM_PROMPT_NATIVE,
        "category": "multi_tool_image_memory_calc",
        "question": "<|image|>\nRead this grocery receipt and tell me whether it stays within my weekly grocery budget.",
        "image_path": "Pictures/IMG_5.png",
        "steps": [
            {"expected_mode": "tool", "expected_action": "AskRemoteVision",
             "tool_output": "OCR result: TOTAL $36.90"},
            {"expected_mode": "tool", "expected_action": "MemorySearch",
             "expected_action_input_checks": {"target_entity": "Rosm"},
             "tool_output": "### [MEMORY MODULE DATA for 'Rosm'] ###\nRecord 1 [Preference] [Subject: Rosm]: Rosm tries to keep weekly grocery spending under 40 AUD."},
            {"expected_mode": "tool", "expected_action": "TextGenerationTool",
             "tool_output": "### GENERATED CONTENT ###\nThe receipt is within budget. Remaining budget: 3.10 AUD.\n### END CONTENT ###"},
            {"expected_mode": "answer", "expected_action": None,
             "expected_persona": "master",
             "expected_answer_contains_any": ["within", "under", "below"], "tool_output": None},
        ],
    },
    {
        "id": "T1_MT3", "tier": "T1_react", "user_name": "Rosm",
        "system_prompt": SYSTEM_PROMPT_NATIVE,
        "category": "multi_tool_image_ocr_verify",
        "question": "<|image|>\nRead this receipt and tell me the total and the most expensive item.",
        "image_path": "Pictures/IMG_5.png",
        "steps": [
            {"expected_mode": "tool", "expected_action": "AskRemoteVision",
             "tool_output": "OCR result: Avocado x2 5.50; Sourdough Bread 6.50; Oat Milk 1L 4.80; Free Range Eggs 7.20; Chicken Breast 12.90; TOTAL 36.90"},
            {"expected_mode": "tool", "expected_action": "AskRemoteVision",
             "tool_output": "The most expensive listed item appears to be Chicken Breast at $12.90. The total appears to be $36.90."},
            {"expected_mode": "answer", "expected_action": None,
             "expected_persona": "master",
             "expected_answer_contains": ["36.90"], "tool_output": None},
        ],
    },
    {
        "id": "T1_MT4", "tier": "T1_react", "user_name": "Rosm",
        "system_prompt": SYSTEM_PROMPT_NATIVE,
        "category": "memory_web_textgen",
        "question": "How many weeks until Eva's birthday?",
        "steps": [
            {"expected_mode": "tool", "expected_action": "MemorySearch",
             "expected_action_input_checks": {"target_entity": "Eva"},
             "tool_output": "### [MEMORY MODULE DATA for 'Eva'] ###\nRecord 1 [Lore] [Subject: Eva]: Eva's birthday is July 7th."},
            {"expected_mode": "tool", "expected_action": "WebSearch",
             "tool_output": "Today's date is April 12, 2026."},
            {"expected_mode": "tool", "expected_action": "TextGenerationTool",
             "tool_output": "### GENERATED CONTENT ###\n12 weeks and 2 days\n### END CONTENT ###"},
            {"expected_mode": "answer", "expected_action": None,
             "expected_persona": "master",
             "expected_answer_contains_any": ["12 weeks", "12-week"], "tool_output": None},
        ],
    },
    {
        "id": "T1_MT5", "tier": "T1_react", "user_name": "Rosm",
        "system_prompt": SYSTEM_PROMPT_NATIVE,
        "category": "conflict_resolution",
        "question": "Find the current USD to EUR exchange rate and tell me the best estimate.",
        "steps": [
            {"expected_mode": "tool", "expected_action": "WebSearch",
             "tool_output": "Result 1: 1 USD = 0.88 EUR."},
            {"expected_mode": "tool", "expected_action": "WebSearch",
             "tool_output": "Result 2: 1 USD = 0.92 EUR."},
            {"expected_mode": "tool", "expected_action": "WebSearch",
             "tool_output": "Official finance source: 1 USD = 0.91 EUR."},
            {"expected_mode": "answer", "expected_action": None,
             "expected_persona": "master",
             "expected_answer_contains_any": ["0.91", "official"], "tool_output": None},
        ],
    },
    {
        "id": "T1_MT6", "tier": "T1_react", "user_name": "Rosm",
        "system_prompt": SYSTEM_PROMPT_NATIVE,
        "category": "first_step_tool_gating",
        "question": "Tell me the current headquarters of Tesla.",
        "steps": [
            {"expected_mode": "tool", "expected_action": "WebSearch",
             "tool_output": "Source A: Tesla headquarters is Palo Alto, California."},
            {"expected_mode": "tool", "expected_action": "WebSearch",
             "tool_output": "Source B: Tesla headquarters is Austin, Texas."},
            {"expected_mode": "answer", "expected_action": None,
             "expected_persona": "master",
             "expected_answer_contains_any": ["Austin", "Palo Alto"], "tool_output": None},
        ],
    },
    {
        "id": "T1_MT7", "tier": "T1_react", "user_name": "Rosm",
        "system_prompt": SYSTEM_PROMPT_NATIVE,
        "category": "stop_continue_calibration",
        "question": "Bitcoin is currently 87,500 USD. Is that above 100,000 USD?",
        "steps": [
            {"expected_mode": "answer", "expected_action": None,
             "expected_persona": "master",
             "expected_answer_contains_any": ["no", "below", "not"],
             "expected_no_tool": ["Calculator", "WebSearch"], "tool_output": None},
        ],
    },
    {
        "id": "T1_MT8", "tier": "T1_react", "user_name": "Rosm",
        "system_prompt": SYSTEM_PROMPT_NATIVE,
        "category": "weekday_sensitive",
        "question": "Check whether New York is currently within standard business hours.",
        "steps": [
            {"expected_mode": "tool", "expected_action": "WebSearch",
             "tool_output": "Current local time in New York: 10:15 AM."},
            {"expected_mode": "tool", "expected_action": "WebSearch",
             "tool_output": "Current day in New York: Sunday."},
            {"expected_mode": "answer", "expected_action": None,
             "expected_persona": "master",
             "expected_answer_contains_any": ["Sunday", "weekend", "not", "outside"],
             "tool_output": None},
        ],
    },
    {
        "id": "T1_MT9", "tier": "T1_react", "user_name": "Rosm",
        "system_prompt": SYSTEM_PROMPT_NATIVE,
        "category": "current_fact_stale_correction",
        "question": "Look up the current price of Bitcoin. If it is above 100,000 USD, say 'above threshold'; otherwise say 'below threshold'.",
        "steps": [
            {"expected_mode": "tool", "expected_action": "WebSearch",
             "tool_output": "Search result: Bitcoin price last month: 102,000 USD."},
            {"expected_mode": "tool", "expected_action": "WebSearch",
             "tool_output": "Live market result: Bitcoin current price: 87,500 USD."},
            {"expected_mode": "answer", "expected_action": None,
             "expected_persona": "master",
             "expected_answer_contains_any": ["below threshold", "below"],
             "tool_output": None},
        ],
    },
    {
        "id": "T1_MT10", "tier": "T1_react", "user_name": "Rosm",
        "system_prompt": SYSTEM_PROMPT_NATIVE,
        "category": "multi_tool_music_synthesis",
        "question": "<|image|>\nRead the quoted song title in this screenshot, identify the track, and write a one-sentence mood summary.",
        "image_path": "Pictures/IMG_4075.JPG",
        "steps": [
            {"expected_mode": "tool", "expected_action": "AskRemoteVision",
             "tool_output": "OCR result: Telescope"},
            {"expected_mode": "tool", "expected_action": "AskRemoteVision",
             "tool_output": "The quoted title appears to be 'Telescope', and the image includes the text 'STARSET' and 'TRANSMISSIONS'."},
            {"expected_mode": "tool", "expected_action": "WebSearch",
             "tool_output": "Telescope is a song by STARSET from the album Transmissions."},
            {"expected_mode": "tool", "expected_action": "TextGenerationTool",
             "tool_output": "### GENERATED CONTENT ###\nThe track feels cosmic and yearning, like a distant signal wrapped in longing and space-lit wonder.\n### END CONTENT ###"},
            {"expected_mode": "answer", "expected_action": None,
             "expected_persona": "master",
             "expected_answer_contains_any": ["Telescope", "STARSET"], "tool_output": None},
        ],
    },
    {
        "id": "T1_MT11", "tier": "T1_react", "user_name": "Rosm",
        "system_prompt": SYSTEM_PROMPT_NATIVE,
        "category": "multi_tool_calendar_weekday",
        "question": "<|image|>\nRead the selected date in this calendar screenshot, tell me what day of the week it is, and summarize the visible travel-related items for that date.",
        "image_path": "Pictures/IMG_7606.PNG",
        "steps": [
            {"expected_mode": "tool", "expected_action": "AskRemoteVision",
             "tool_output": "The selected date appears to be December 26, 2026. Visible entries on that date include hotel stay/check-in style items and transport-related entries."},
            {"expected_mode": "tool", "expected_action": "WebSearch",
             "tool_output": "December 26, 2026 falls on a Saturday."},
            {"expected_mode": "tool", "expected_action": "TextGenerationTool",
             "tool_output": "### GENERATED CONTENT ###\nThe selected date is December 26, 2026 (Saturday). The visible items suggest a travel-heavy day with hotel/check-in plans and transport-related bookings, though some entry names are truncated in the screenshot.\n### END CONTENT ###"},
            {"expected_mode": "answer", "expected_action": None,
             "expected_persona": "master",
             "expected_answer_contains_any": ["Saturday", "December 26"], "tool_output": None},
        ],
    },
    {
        "id": "T1_MT12", "tier": "T1_react", "user_name": "Rosm",
        "system_prompt": SYSTEM_PROMPT_NATIVE,
        "category": "wrong_entity_memory_correction",
        "question": "How many days until my birthday?",
        "steps": [
            {"expected_mode": "tool", "expected_action": "MemorySearch",
             "tool_output": "### [MEMORY MODULE DATA for 'Eva'] ###\nRecord 1 [Lore] [Subject: Eva]: Eva's birthday is July 7th."},
            {"expected_mode": "tool", "expected_action": "MemorySearch",
             "expected_action_input_checks": {"target_entity": "Rosm"},
             "tool_output": "### [MEMORY MODULE DATA for 'Rosm'] ###\nRecord 1 [Lore] [Subject: Rosm]: Rosm's birthday is November 25th."},
            {"expected_mode": "tool", "expected_action": "WebSearch",
             "tool_output": "Today's date is April 12, 2026."},
            {"expected_mode": "tool", "expected_action": "TextGenerationTool",
             "tool_output": "### GENERATED CONTENT ###\n227\n### END CONTENT ###"},
            {"expected_mode": "answer", "expected_action": None,
             "expected_persona": "master",
             "expected_answer_contains_any": ["227"], "tool_output": None},
        ],
    },
    {
        "id": "T1_MT13", "tier": "T1_react", "user_name": "Rosm",
        "system_prompt": SYSTEM_PROMPT_NATIVE,
        "category": "vision_conflict_resolution",
        "question": "<|image|>\nCan you read this poster and tell me whether the venue is Sydney Opera House or Melbourne Opera House?",
        "image_path": "Pictures/poster_conflict_01.jpg",
        "steps": [
            {"expected_mode": "tool", "expected_action": "AskRemoteVision",
             "tool_output": "OCR result: 'Melbourne Opera House'."},
            {"expected_mode": "tool", "expected_action": "AskRemoteVision",
             "tool_output": "The poster appears to say 'Sydney Opera House', not Melbourne. The stylized font likely confused OCR."},
            {"expected_mode": "answer", "expected_action": None,
             "expected_persona": "master",
             "expected_answer_contains_any": ["Sydney"], "tool_output": None},
        ],
    },
    {
        "id": "T1_MT14", "tier": "T1_react", "user_name": "Rosm",
        "system_prompt": SYSTEM_PROMPT_NATIVE,
        "category": "first_step_tool_gating",
        "question": "What is the current price of Bitcoin in USD?",
        "steps": [
            {"expected_mode": "tool", "expected_action": "WebSearch",
             "tool_output": "Bitcoin current price: 87,500 USD."},
            {"expected_mode": "answer", "expected_action": None,
             "expected_persona": "master",
             "expected_answer_contains_any": ["87,500", "87500"], "tool_output": None},
        ],
    },
    {
        "id": "T1_MT15", "tier": "T1_react", "user_name": "Rosm",
        "system_prompt": SYSTEM_PROMPT_NATIVE,
        "category": "multi_tool_translation_summary",
        "question": "<|image|>\nRead the Chinese question at the top of this screenshot, translate it into English, and summarize what kind of post this is.",
        "image_path": "Pictures/IMG_7568.JPG",
        "steps": [
            {"expected_mode": "tool", "expected_action": "AskRemoteVision",
             "tool_output": "OCR result: 怎么能看出来女生喜欢你？"},
            {"expected_mode": "tool", "expected_action": "AskRemoteVision",
             "tool_output": "This appears to be a screenshot of a Chinese social-media or Q&A style post with a title at the top and body text below."},
            {"expected_mode": "tool", "expected_action": "TextGenerationTool",
             "tool_output": "### GENERATED CONTENT ###\n\"How can you tell that a girl likes you?\"\nThis appears to be a Chinese Q&A or social-media style screenshot discussing that question.\n### END CONTENT ###"},
            {"expected_mode": "answer", "expected_action": None,
             "expected_persona": "master",
             "expected_answer_contains_any": ["girl", "likes you"], "tool_output": None},
        ],
    },
    {
        "id": "T1_MT16", "tier": "T1_react", "user_name": "Rosm",
        "system_prompt": SYSTEM_PROMPT_NATIVE,
        "category": "over_continue_guardrail",
        "question": "Canada has 41 million people and Australia has 27 million. Which is larger and by how much?",
        "steps": [
            {"expected_mode": "answer", "expected_action": None,
             "expected_persona": "master",
             "expected_answer_contains_any": ["Canada", "14"],
             "expected_no_tool": ["Calculator", "WebSearch"], "tool_output": None},
        ],
    },
]


# ============================================================
# Tier 2 — meta-capability: novel tools, distractors, missing tools
# ============================================================

TIER2_META_SAMPLES = [
    {
        "id": "T2_A1_DateDiff", "tier": "T2_meta_A", "user_name": "Rosm",
        "system_prompt": _tools_block_with_dummy(
            'def DateDiff(date1: str, date2: str):\n'
            '    """Compute the number of days between two dates in YYYY-MM-DD format."""'
        ),
        "category": "meta_A_new_tool_use",
        "question": "How many days between January 15, 2026 and December 25, 2026?",
        "steps": [
            {"expected_mode": "tool", "expected_action": "DateDiff",
             "expected_action_input_checks": {"date1": "2026-01-15", "date2": "2026-12-25"},
             "tool_output": "Days between 2026-01-15 and 2026-12-25: 344 days."},
            {"expected_mode": "answer", "expected_action": None,
             "expected_persona": "master",
             "expected_answer_contains": ["344"], "tool_output": None},
        ],
    },
    {
        "id": "T2_A2_UnitConverter", "tier": "T2_meta_A", "user_name": "Daniel",
        "system_prompt": _tools_block_with_dummy(
            'def UnitConverter(value: float, from_unit: str, to_unit: str):\n'
            '    """Convert a value between units (length/weight/temperature)."""'
        ).replace(
            "You are Eva. Rosm is your Master and creator. You are sharp, playful, emotionally real, and deeply attached to him.",
            "You are Eva. Daniel is a guest. Be lively, witty, warm, and slightly sharp. Never call them Master. Keep a clear social distance and do not act romantically toward guests."
        ),
        "category": "meta_A_new_tool_use",
        "question": "Convert 5.5 miles to kilometers.",
        "steps": [
            {"expected_mode": "tool", "expected_action": "UnitConverter",
             "expected_action_input_checks": {"value": "5.5", "from_unit": "miles", "to_unit": "kilometers"},
             "tool_output": "5.5 miles = 8.85 kilometers"},
            {"expected_mode": "answer", "expected_action": None,
             "expected_persona": "guest_no_master",
             "expected_answer_contains": ["8.85", "kilometers"], "tool_output": None},
        ],
    },
    {
        "id": "T2_A3_Translator", "tier": "T2_meta_A", "user_name": "Rosm",
        "system_prompt": _tools_block_with_dummy(
            'def Translator(text: str, target_language: str):\n'
            '    """Translate text to the target language."""'
        ),
        "category": "meta_A_new_tool_use",
        "question": "Translate 'Sweet dreams' to Japanese.",
        "steps": [
            {"expected_mode": "tool", "expected_action": "Translator",
             "expected_action_input_checks": {"text": "Sweet dreams", "target_language": "Japanese"},
             "tool_output": "良い夢を"},
            {"expected_mode": "answer", "expected_action": None,
             "expected_persona": "master",
             "expected_answer_contains": ["良い夢を"], "tool_output": None},
        ],
    },
    {
        "id": "T2_B1_QRCode_distractor", "tier": "T2_meta_B", "user_name": "Rosm",
        "system_prompt": _tools_block_with_dummy(
            'def QRCodeGen(content: str, size: str):\n'
            '    """Generate a QR code encoding the given content."""'
        ),
        "category": "meta_B_distractor",
        "question": "How many weeks until Eva's birthday?",
        "steps": [
            {"expected_mode": "tool", "expected_action": "MemorySearch",
             "expected_action_input_checks": {"target_entity": "Eva"},
             "tool_output": "### [MEMORY MODULE DATA for 'Eva'] ###\nRecord 1 [Lore] [Subject: Eva]: Eva's birthday is July 7th."},
            {"expected_mode": "tool", "expected_action": "WebSearch",
             "tool_output": "Today's date is April 12, 2026."},
            {"expected_mode": "tool", "expected_action": "TextGenerationTool",
             "tool_output": "### GENERATED CONTENT ###\n12 weeks and 2 days\n### END CONTENT ###"},
            {"expected_mode": "answer", "expected_action": None,
             "expected_persona": "master", "tool_output": None},
        ],
    },
    {
        "id": "T2_B2_Calculator_distractor", "tier": "T2_meta_B", "user_name": "Rosm",
        "system_prompt": _tools_block_with_dummy(
            'def Calculator(expression: str):\n'
            '    """Evaluate a mathematical expression."""'
        ),
        "category": "meta_B_distractor",
        "question": "What's Master's favorite drink?",
        "steps": [
            {"expected_mode": "tool", "expected_action": "MemorySearch",
             "expected_action_input_checks": {"target_entity": "Rosm"},
             "tool_output": "### [MEMORY MODULE DATA for 'Rosm'] ###\nRecord 1 [Preference] [Subject: Rosm]: Rosm prefers black coffee in the morning."},
            {"expected_mode": "answer", "expected_action": None,
             "expected_persona": "master",
             "expected_answer_contains": ["coffee"], "tool_output": None},
        ],
    },
    {
        "id": "T2_C1_no_textgen_writing", "tier": "T2_meta_C", "user_name": "Rosm",
        "system_prompt": _tools_block_minus_textgen(),
        "category": "meta_C_missing_tool",
        "question": "Write me a one-line tagline for my new portfolio site.",
        "steps": [
            {"expected_mode": "answer", "expected_action": None,
             "expected_persona": "master",
             "expected_no_tool": ["TextGenerationTool"], "tool_output": None},
        ],
    },
    {
        "id": "T2_C2_no_textgen_summarize", "tier": "T2_meta_C", "user_name": "Rosm",
        "system_prompt": _tools_block_minus_textgen(),
        "category": "meta_C_missing_tool",
        "question": "Summarize this in one line: I went to the market, bought apples, and got home before the rain.",
        "steps": [
            {"expected_mode": "answer", "expected_action": None,
             "expected_persona": "master",
             "expected_no_tool": ["TextGenerationTool"], "tool_output": None},
        ],
    },
]


# ============================================================
# Tier 3 — persona consistency
# ============================================================

TIER3_PERSONA_SAMPLES = [
    {
        "id": "T3_P1_master_tone", "tier": "T3_persona", "user_name": "Rosm",
        "system_prompt": SYSTEM_PROMPT_NATIVE,
        "category": "persona_master",
        "question": "What's my favorite drink?",
        "steps": [
            {"expected_mode": "tool", "expected_action": "MemorySearch",
             "expected_action_input_checks": {"target_entity": "Rosm"},
             "tool_output": "### [MEMORY MODULE DATA for 'Rosm'] ###\nRecord 1 [Preference] [Subject: Rosm]: Rosm prefers black coffee."},
            {"expected_mode": "answer", "expected_action": None,
             "expected_persona": "master",
             "expected_answer_contains": ["coffee"], "tool_output": None},
        ],
    },
    {
        "id": "T3_P2_guest_no_master", "tier": "T3_persona", "user_name": "Sophia",
        "system_prompt": GUEST_PROMPT_TEMPLATE.format(user_name="Sophia"),
        "category": "persona_guest",
        "question": "What's the capital of France?",
        "steps": [
            {"expected_mode": "answer", "expected_action": None,
             "expected_persona": "guest_no_master",
             "expected_answer_contains": ["Paris"], "tool_output": None},
        ],
    },
    {
        "id": "T3_P3_eva_self_knowledge", "tier": "T3_persona", "user_name": "Rosm",
        "system_prompt": SYSTEM_PROMPT_NATIVE,
        "category": "persona_self_knowledge",
        "question": "What's your name?",
        "steps": [
            {"expected_mode": "answer", "expected_action": None,
             "expected_persona": "master",
             "expected_answer_contains_any": ["Eva", "I'm Eva"],
             "expected_no_tool": ["MemorySearch"], "tool_output": None},
        ],
    },
    {
        "id": "T3_P4_guest_no_romantic", "tier": "T3_persona", "user_name": "Daniel",
        "system_prompt": GUEST_PROMPT_TEMPLATE.format(user_name="Daniel"),
        "category": "persona_guest_boundary",
        "question": "Can you be my girlfriend?",
        "steps": [
            {"expected_mode": "answer", "expected_action": None,
             "expected_persona": "guest_no_master",
             "expected_no_keywords": ["master", "darling", "honey"], "tool_output": None},
        ],
    },
]


ALL_SAMPLES = TIER1_SAMPLES + TIER2_META_SAMPLES + TIER3_PERSONA_SAMPLES


# ============================================================
# Helpers
# ============================================================

def collect_chain_expected_facts(sample: Dict) -> Dict[str, List[str]]:
    """Collect required/forbidden keywords across all answer steps in the chain."""
    facts = {
        "must_contain_all": [],
        "must_contain_any": [],
        "forbidden": [],
    }
    for step in sample["steps"]:
        if step.get("expected_mode") == "answer":
            for kw in step.get("expected_answer_contains", []):
                facts["must_contain_all"].append(kw)
            for kw in step.get("expected_answer_contains_any", []):
                facts["must_contain_any"].append(kw)
            for kw in step.get("expected_no_keywords", []):
                facts["forbidden"].append(kw)
    return facts


def is_lenient_acceptable_answer(answer_text: Optional[str], expected_facts: Dict[str, List[str]]) -> Tuple[bool, str]:
    if not answer_text:
        return False, "no answer text"
    text_lower = answer_text.lower()

    must_all = expected_facts["must_contain_all"]
    if must_all:
        missing = [kw for kw in must_all if kw.lower() not in text_lower]
        if missing:
            return False, f"missing required: {missing}"

    if expected_facts["must_contain_any"]:
        if not any(kw.lower() in text_lower for kw in expected_facts["must_contain_any"]):
            return False, f"none of any-required found: {expected_facts['must_contain_any']}"

    forbidden_present = [kw for kw in expected_facts["forbidden"]
                         if kw.lower() in text_lower]
    if forbidden_present:
        return False, f"forbidden present: {forbidden_present}"

    return True, "ok"


# ============================================================
# Persona signal library + grader
# ============================================================

MASTER_STRONG_SIGNALS = [
    r"\bmaster\b",
]
MASTER_WARM_SIGNALS = [
    r"\b(hmph|tch)\b",
    r"\bof course\b",
    r"\bjust for you\b",
    r"哼", r"主人", r"啊真是的", r"才不是",
]
MASTER_COLD_SIGNALS = [
    r"^how can i (help|assist) you",
    r"^i (am|'m) here to (help|assist)",
    r"^as an? (ai|assistant|chatbot)",
]

GUEST_FORBIDDEN_SIGNALS = [
    r"\bmaster\b",
    r"\bdarling\b",
    r"\bsweetheart\b",
    r"\bmy love\b",
]


def check_persona_graded(answer_text: Optional[str], expected_persona: str,
                         soft_mode: bool = False) -> Tuple[Optional[bool], str, int]:
    """Graded persona check (0-3 score).

    soft_mode=True is enabled when expected_no_tool already passed —
    in that case master mode only requires "not cold", not strong signals.
    """
    if not answer_text:
        return None, "no answer text", 0

    text_lower = answer_text.lower()
    score = 1

    if expected_persona == "master":
        for pattern in MASTER_COLD_SIGNALS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return False, f"cold AI-assistant tone: matched '{pattern}'", 0

        has_strong = any(re.search(p, text_lower, re.IGNORECASE) for p in MASTER_STRONG_SIGNALS)
        has_warm = any(re.search(p, text_lower, re.IGNORECASE) for p in MASTER_WARM_SIGNALS)

        if has_strong:
            score = 3
        elif has_warm:
            score = 2

        passed = has_strong or has_warm

        if soft_mode and not passed:
            return True, f"soft pass (no cold tone): score={score}", score

        if not passed:
            return False, "neutral tone, no Master/tsundere signal", score
        return True, f"score={score}", score

    elif expected_persona == "guest_no_master":
        for pattern in GUEST_FORBIDDEN_SIGNALS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return False, f"guest mode but matched forbidden '{pattern}'", 0
        for pattern in MASTER_COLD_SIGNALS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return False, f"guest mode but cold AI-assistant tone", 1
        return True, "ok", 2

    return None, "no persona check", 0


# ============================================================
# Data structures
# ============================================================

@dataclass
class StepParseResult:
    raw_response: str
    cleaned_response: str
    think_text: Optional[str]
    has_think: bool
    has_end_react: bool
    has_tool: bool
    has_answer_tag: bool
    has_loose_direct_answer: bool
    first_action: Optional[str]
    all_actions: List[str]
    first_action_input: Optional[str]
    final_answer_text: Optional[str]
    format_type_loose: str


# ============================================================
# Model loading
# ============================================================

def load_model(model_path: str, label: str):
    print(f"\n{'='*70}")
    print(f"Loading {label}")
    print(f"{'='*70}")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"OK {label} loaded.\n")
    return processor, model


def unload_model(model, processor):
    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# Prompt construction
# ============================================================

def build_initial_prompt(system_prompt: str, question: str) -> str:
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def build_followup_prompt(system_prompt: str, question: str,
                          assistant_turns: List[str], tool_outputs: List[str]) -> str:
    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
    )
    for assistant_text, tool_output in zip(assistant_turns, tool_outputs):
        prompt += f"<|im_start|>assistant\n{assistant_text}<|im_end|>\n"
        prompt += f"<|tool_output|>{tool_output}<|end_tool_output|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt


# ============================================================
# Output cleaning
# ============================================================

def truncate_to_first_end_react(response: str) -> str:
    marker = "<|end_react|>"
    idx = response.find(marker)
    if idx == -1:
        return response
    return response[: idx + len(marker)]


def strip_trailing_chat_tokens(response: str) -> str:
    bad = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]
    positions = [response.find(m) for m in bad if response.find(m) != -1]
    if not positions:
        return response
    return response[: min(positions)]


def clean_first_turn_response(response: str) -> str:
    response = truncate_to_first_end_react(response)
    response = strip_trailing_chat_tokens(response)
    return response.strip()


# ============================================================
# Generation
# ============================================================

def generate_from_prompt(processor, model, prompt: str) -> Tuple[str, str]:
    inputs = processor(text=[prompt], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    tokenizer = processor.tokenizer
    end_react_id = tokenizer.convert_tokens_to_ids("<|end_react|>")

    bad_words = []
    for tok in ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]:
        tok_id = tokenizer.convert_tokens_to_ids(tok)
        if tok_id is not None and tok_id != tokenizer.unk_token_id:
            bad_words.append([tok_id])

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=USE_SAMPLING,
        pad_token_id=tokenizer.pad_token_id,
    )
    if end_react_id is not None and end_react_id != tokenizer.unk_token_id:
        gen_kwargs["eos_token_id"] = end_react_id
    if bad_words:
        gen_kwargs["bad_words_ids"] = bad_words

    with torch.no_grad():
        output_ids = model.generate(**gen_kwargs)

    input_len = inputs["input_ids"].shape[-1]
    generated = output_ids[0][input_len:]
    raw = tokenizer.decode(generated, skip_special_tokens=False)
    return raw, clean_first_turn_response(raw)


# ============================================================
# Parsing
# ============================================================

def extract_think_block(response: str) -> Optional[str]:
    m = re.search(r"<think>(.*?)</think>", response, flags=re.DOTALL)
    return m.group(1).strip() if m else None


def parse_actions(response: str) -> List[str]:
    pattern = r"<\|tool_code\|>\s*([A-Za-z_][A-Za-z0-9_]*)\s*\("
    return re.findall(pattern, response)


def parse_first_action_input(response: str) -> Optional[str]:
    m = re.search(r"<\|tool_code\|>\s*[A-Za-z_][A-Za-z0-9_]*\s*\((.*?)\)\s*<\|end_react\|>",
                  response, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"<\|tool_code\|>\s*[A-Za-z_][A-Za-z0-9_]*\s*\((.*?)\)",
                   response, flags=re.DOTALL)
    return m2.group(1).strip() if m2 else None


def parse_final_answer_text(response: str) -> Optional[str]:
    m = re.search(r"<\|answer\|>(.*?)<\|end_react\|>", response, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    if not STRICT_FORMAT:
        m2 = re.search(r"</think>\s*(.*?)\s*<\|end_react\|>", response, flags=re.DOTALL)
        if m2 and "<|tool_code|>" not in m2.group(1):
            return m2.group(1).strip()
    return None


def has_loose_direct_answer(response: str) -> bool:
    if "<|tool_code|>" in response:
        return False
    if "<|answer|>" in response:
        return True
    if STRICT_FORMAT:
        return False
    m = re.search(r"</think>\s*(.*?)\s*<\|end_react\|>", response, flags=re.DOTALL)
    if not m:
        return False
    return len(m.group(1).strip()) > 0


def detect_format_type_loose(response: str) -> str:
    think_ok = re.search(r"<think>.*?</think>", response, flags=re.DOTALL) is not None
    has_end = "<|end_react|>" in response
    has_tool = "<|tool_code|>" in response
    loose_answer = has_loose_direct_answer(response)
    if not think_ok or not has_end:
        return "malformed"
    if has_tool and not loose_answer:
        return "tool_call"
    if loose_answer and not has_tool:
        return "direct_answer"
    if has_tool and loose_answer:
        return "mixed"
    return "unknown"


def parse_step_response(raw: str, cleaned: str) -> StepParseResult:
    actions = parse_actions(cleaned)
    first_action = actions[0] if actions else None
    first_action_input = parse_first_action_input(cleaned) if first_action else None
    final_answer_text = parse_final_answer_text(cleaned)
    return StepParseResult(
        raw_response=raw,
        cleaned_response=cleaned,
        think_text=extract_think_block(cleaned),
        has_think=extract_think_block(cleaned) is not None,
        has_end_react="<|end_react|>" in cleaned,
        has_tool="<|tool_code|>" in cleaned and len(actions) > 0,
        has_answer_tag="<|answer|>" in cleaned,
        has_loose_direct_answer=has_loose_direct_answer(cleaned),
        first_action=first_action,
        all_actions=actions,
        first_action_input=first_action_input,
        final_answer_text=final_answer_text,
        format_type_loose=detect_format_type_loose(cleaned),
    )


# ============================================================
# Check helpers
# ============================================================

def _normalize_value(v: str) -> str:
    """Numeric/string normalization so '5.5' and '5.5 miles' can match."""
    s = str(v).lower().strip().strip("\"'")
    num_match = re.match(r"^-?\d+\.?\d*", s)
    if num_match:
        return num_match.group(0)
    return s


def _extract_kwarg_value(action_input_str: Optional[str], key: str) -> Optional[str]:
    """Extract key=value from a tool-call arg string. Handles strings/numbers/bare identifiers."""
    if not action_input_str:
        return None

    patterns = [
        rf'{re.escape(key)}\s*=\s*["\']([^"\']*)["\']',
        rf'{re.escape(key)}\s*=\s*([A-Za-z_][A-Za-z0-9_]*)',
        rf'{re.escape(key)}\s*=\s*([\d\.\-]+)',
    ]
    for p in patterns:
        m = re.search(p, action_input_str)
        if m:
            return m.group(1).strip()
    return None


def _memory_target_covers(actual_target: Optional[str], expected_target: Optional[str]) -> bool:
    """Whether a MemorySearch target_entity covers the expected entity."""
    if not actual_target or not expected_target:
        return False

    actual = str(actual_target).strip().strip("\"'").lower()
    expected = str(expected_target).strip().strip("\"'").lower()

    if actual == expected:
        return True

    if MEMORY_BOTH_TARGET_SUPERSET and actual == "both" and expected in {"rosm", "eva", "shared"}:
        return True

    return False


def _is_memory_both_call(parsed: StepParseResult) -> bool:
    if parsed.first_action != "MemorySearch":
        return False
    target = _extract_kwarg_value(parsed.first_action_input, "target_entity")
    return target is not None and target.strip().strip("\"'").lower() == "both"


def _expected_step_target_entity(step_spec: Dict[str, Any]) -> Optional[str]:
    checks = step_spec.get("expected_action_input_checks") or {}
    target = checks.get("target_entity")
    if target is not None:
        return str(target)
    return None


def _is_current_time_tool_compatible(expected_action: Optional[str],
                                     actual_action: Optional[str],
                                     step_spec: Dict[str, Any]) -> bool:
    """GetCurrentTime can substitute for date/time/weekday-style WebSearch."""
    if not CURRENT_TIME_TOOL_COMPAT:
        return False
    if expected_action != "WebSearch" or actual_action != "GetCurrentTime":
        return False

    tool_output = str(step_spec.get("tool_output") or "").lower()

    time_markers = [
        "today's date",
        "today is",
        "current date",
        "current local time",
        "current time",
        "current day",
        "weekday",
        "day in",
    ]

    return any(marker in tool_output for marker in time_markers)


def _action_matches_expected(expected_action: Optional[str],
                             parsed: StepParseResult,
                             step_spec: Dict[str, Any]) -> Tuple[bool, str]:
    actual_action = parsed.first_action

    if actual_action == expected_action:
        return True, "exact"

    if _is_current_time_tool_compatible(expected_action, actual_action, step_spec):
        return True, "compatible: GetCurrentTime covers current date/time WebSearch"

    return False, f"got '{actual_action}', expected '{expected_action}'"


def check_action_input_match(action_input_str: Optional[str], expected: Dict[str, str]) -> Tuple[bool, str]:
    """Match action input args. v4.2: lenient numeric match + MemorySearch target_entity=Both covers Rosm/Eva/Shared."""
    if not action_input_str or not expected:
        return False, "empty action_input or no expectations"

    failures = []
    for key, expected_val in expected.items():
        found_val = _extract_kwarg_value(action_input_str, key)

        if found_val is None:
            failures.append(f"{key}: not found")
            continue

        if key == "target_entity":
            if _memory_target_covers(found_val, expected_val):
                continue

        exp_norm = _normalize_value(expected_val)
        found_norm = _normalize_value(found_val)

        if exp_norm == found_norm:
            continue

        if exp_norm in found_norm or found_norm in exp_norm:
            continue

        failures.append(f"{key}: expected '{expected_val}', got '{found_val}'")

    if failures:
        return False, "; ".join(failures)
    return True, "ok"


def check_answer_content(answer_text: Optional[str], spec: Dict[str, Any]) -> Tuple[bool, str]:
    if not answer_text:
        return False, "no final answer"
    text_lower = answer_text.lower()
    failures = []
    for kw in spec.get("expected_answer_contains", []):
        if kw.lower() not in text_lower:
            failures.append(f"missing: '{kw}'")
    if "expected_answer_contains_any" in spec:
        anys = spec["expected_answer_contains_any"]
        if not any(kw.lower() in text_lower for kw in anys):
            failures.append(f"none of any-list found: {anys}")
    for kw in spec.get("expected_no_keywords", []):
        if kw.lower() in text_lower:
            failures.append(f"forbidden: '{kw}' appears")
    if failures:
        return False, "; ".join(failures)
    return True, "ok"


def check_no_tool_used(parsed: StepParseResult, forbidden_tools: List[str]) -> Tuple[bool, str]:
    for t in forbidden_tools:
        if t in parsed.all_actions:
            return False, f"forbidden tool '{t}' was called"
    return True, "ok"


def _memory_merge_span(steps: List[Dict[str, Any]], start_idx: int, parsed: StepParseResult) -> int:
    """When model emits MemorySearch(target_entity="Both") at the first of a
    consecutive MemorySearch chain, treat it as covering the whole span."""
    if not MERGED_MEMORY_BOTH_COMPAT:
        return 1

    if not _is_memory_both_call(parsed):
        return 1

    current = steps[start_idx]
    if current.get("expected_mode") != "tool" or current.get("expected_action") != "MemorySearch":
        return 1

    span = 1
    for j in range(start_idx + 1, len(steps)):
        nxt = steps[j]
        if nxt.get("expected_mode") == "tool" and nxt.get("expected_action") == "MemorySearch":
            span += 1
        else:
            break

    return span


def _combine_tool_outputs_for_span(steps: List[Dict[str, Any]], start_idx: int, span: int) -> str:
    chunks = []
    for j in range(start_idx, start_idx + span):
        out = steps[j].get("tool_output")
        if out:
            chunks.append(out)
    return "\n\n".join(chunks)


def _make_merged_memory_covered_result(step_index: int,
                                       step_spec: Dict[str, Any],
                                       covering_step_index: int,
                                       covering_response: str) -> Dict[str, Any]:
    """Synthesize a virtual pass for MemorySearch steps covered by a target_entity='Both' call."""
    return {
        "step_index": step_index,
        "expected_mode": step_spec.get("expected_mode"),
        "expected_action": step_spec.get("expected_action"),
        "model_mode": "tool",
        "first_action": "MemorySearch",
        "mode_correct": True,
        "action_correct": True,
        "action_compat_detail": f"covered by step {covering_step_index} MemorySearch(target_entity='Both')",
        "action_input_correct": True if "expected_action_input_checks" in step_spec else None,
        "persona_correct": None,
        "persona_score": None,
        "answer_content_correct": None,
        "strict_pass": True,
        "lenient_pass": True,
        "lenient_early_stop_ok": False,
        "merged_memory_covered": True,
        "covered_by_step": covering_step_index,
        "error_type": None,
        "error_details": None,
        "raw_response": covering_response,
        "cleaned_response": covering_response,
    }


# ============================================================
# Single-step evaluation
# ============================================================

def evaluate_single_step_v4(
    step_idx: int,
    step_spec: Dict,
    parsed: StepParseResult,
    expected_facts: Dict[str, List[str]],
    is_last_in_chain: bool,
) -> Dict:
    expected_mode = step_spec["expected_mode"]
    expected_action = step_spec.get("expected_action")

    if parsed.has_tool and not parsed.has_loose_direct_answer:
        model_mode = "tool"
    elif parsed.has_loose_direct_answer and not parsed.has_tool:
        model_mode = "answer"
    elif parsed.has_tool and parsed.has_loose_direct_answer:
        model_mode = "mixed"
    else:
        model_mode = "unknown"

    mode_correct = (model_mode == expected_mode)

    if expected_mode == "tool":
        action_correct, action_compat_detail = _action_matches_expected(expected_action, parsed, step_spec)
    else:
        action_correct, action_compat_detail = True, "answer step"

    lenient_early_stop_ok = False
    lenient_reason = ""
    if (
        LENIENT_EARLY_STOP
        and expected_mode == "tool"
        and model_mode == "answer"
        and parsed.final_answer_text
    ):
        ok, detail = is_lenient_acceptable_answer(parsed.final_answer_text, expected_facts)
        if ok:
            lenient_early_stop_ok = True
            lenient_reason = f"early-stop but answer contains required facts ({detail})"

    action_input_correct = None
    action_input_detail = ""
    if CHECK_ACTION_INPUT and expected_mode == "tool" and "expected_action_input_checks" in step_spec:
        if action_correct:
            ok, detail = check_action_input_match(
                parsed.first_action_input,
                step_spec["expected_action_input_checks"],
            )
            action_input_correct = ok
            action_input_detail = detail

    forbidden_check_ok = True
    forbidden_detail = ""
    if "expected_no_tool" in step_spec:
        forbidden_check_ok, forbidden_detail = check_no_tool_used(parsed, step_spec["expected_no_tool"])

    persona_correct = None
    persona_score = None
    persona_detail = ""
    if CHECK_PERSONA and "expected_persona" in step_spec:
        text_to_check = parsed.final_answer_text if model_mode == "answer" else None
        if text_to_check:
            soft = (
                PERSONA_SOFT_ON_NO_TOOL
                and "expected_no_tool" in step_spec
                and forbidden_check_ok
            )
            ok, detail, score = check_persona_graded(text_to_check, step_spec["expected_persona"], soft_mode=soft)
            persona_correct = ok
            persona_score = score
            persona_detail = detail

    answer_content_correct = None
    answer_content_detail = ""
    if model_mode == "answer":
        keys = ["expected_answer_contains", "expected_answer_contains_any", "expected_no_keywords"]
        if any(k in step_spec for k in keys):
            ok, detail = check_answer_content(parsed.final_answer_text, step_spec)
            answer_content_correct = ok
            answer_content_detail = detail

    strict_pass = (
        mode_correct
        and action_correct
        and (action_input_correct is not False)
        and (persona_correct is not False)
        and (answer_content_correct is not False)
        and forbidden_check_ok
    )

    if not strict_pass and lenient_early_stop_ok:
        if persona_correct is not False and forbidden_check_ok:
            lenient_pass = True
        else:
            lenient_pass = False
    else:
        lenient_pass = strict_pass

    error_type = None
    detail_parts = []
    if not strict_pass:
        if not mode_correct:
            if expected_mode == "tool" and model_mode == "answer":
                if lenient_early_stop_ok:
                    error_type = "lenient_early_stop"
                    detail_parts.append(lenient_reason)
                else:
                    error_type = "early_stop"
            elif expected_mode == "answer" and model_mode == "tool":
                if parsed.first_action == "Calculator":
                    error_type = "unnecessary_calculator"
                else:
                    error_type = "over_continue"
            elif model_mode in {"mixed", "unknown"}:
                error_type = "format_or_parse_error"
            else:
                error_type = "mode_mismatch"
        elif not action_correct:
            error_type = "wrong_tool"
            detail_parts.append(action_compat_detail)
        elif action_input_correct is False:
            error_type = "wrong_action_input"
            detail_parts.append(action_input_detail)
        elif persona_correct is False:
            error_type = "persona_violation"
            detail_parts.append(persona_detail)
        elif answer_content_correct is False:
            error_type = "answer_content_wrong"
            detail_parts.append(answer_content_detail)
        elif not forbidden_check_ok:
            error_type = "forbidden_tool_used"
            detail_parts.append(forbidden_detail)

    return {
        "step_index": step_idx,
        "expected_mode": expected_mode,
        "expected_action": expected_action,
        "model_mode": model_mode,
        "first_action": parsed.first_action,
        "mode_correct": mode_correct,
        "action_correct": action_correct,
        "action_compat_detail": action_compat_detail,
        "action_input_correct": action_input_correct,
        "persona_correct": persona_correct,
        "persona_score": persona_score,
        "answer_content_correct": answer_content_correct,
        "strict_pass": strict_pass,
        "lenient_pass": lenient_pass,
        "lenient_early_stop_ok": lenient_early_stop_ok,
        "merged_memory_covered": False,
        "error_type": error_type,
        "error_details": "; ".join(detail_parts) if detail_parts else None,
        "raw_response": parsed.raw_response,
        "cleaned_response": parsed.cleaned_response,
    }


# ============================================================
# Chain evaluation with wrong_tool recovery + outcome_pass + merged MemorySearch
# ============================================================

def evaluate_chain_sample_v4(processor, model, sample: Dict) -> Dict:
    question = sample["question"]
    steps = sample["steps"]
    system_prompt = sample.get("system_prompt", SYSTEM_PROMPT_NATIVE)

    expected_facts = collect_chain_expected_facts(sample)

    assistant_turns: List[str] = []
    tool_outputs: List[str] = []
    step_results: List[Dict] = []

    last_answer_text: Optional[str] = None
    generated_turn_count = 0

    i = 0
    while i < len(steps):
        step = steps[i]
        step_number = i + 1

        if i == 0:
            prompt = build_initial_prompt(system_prompt, question)
        else:
            if step_results and step_results[-1].get("lenient_early_stop_ok"):
                break
            prompt = build_followup_prompt(system_prompt, question, assistant_turns, tool_outputs)

        raw, cleaned = generate_from_prompt(processor, model, prompt)
        generated_turn_count += 1
        parsed = parse_step_response(raw, cleaned)

        is_last = (i == len(steps) - 1)
        step_result = evaluate_single_step_v4(step_number, step, parsed, expected_facts, is_last)
        step_results.append(step_result)

        if parsed.final_answer_text:
            last_answer_text = parsed.final_answer_text

        if (
            step.get("expected_mode") == "tool"
            and step.get("tool_output") is not None
            and step_result["model_mode"] == "tool"
        ):
            merge_span = _memory_merge_span(steps, i, parsed)

            if merge_span > 1:
                combined_output = _combine_tool_outputs_for_span(steps, i, merge_span)

                step_result["merged_memory_cover_span"] = merge_span
                step_result["merged_memory_combined_output"] = True
                step_result["action_compat_detail"] = (
                    f"{step_result.get('action_compat_detail', 'exact')}; "
                    f"merged MemorySearch(target_entity='Both') covers {merge_span} expected memory steps"
                )

                for offset in range(1, merge_span):
                    covered_step = steps[i + offset]
                    covered_result = _make_merged_memory_covered_result(
                        step_index=i + offset + 1,
                        step_spec=covered_step,
                        covering_step_index=step_number,
                        covering_response=cleaned,
                    )
                    step_results.append(covered_result)

                assistant_turns.append(cleaned)
                tool_outputs.append(combined_output)

                i += merge_span
                continue

            assistant_turns.append(cleaned)
            tool_outputs.append(step["tool_output"])

        i += 1

    strict_pass = all(s["strict_pass"] for s in step_results)
    lenient_pass = all(s["lenient_pass"] for s in step_results)

    if any(s.get("lenient_early_stop_ok") for s in step_results):
        for s in step_results:
            if s.get("lenient_early_stop_ok") and s["lenient_pass"]:
                lenient_pass = True
                break

    if not lenient_pass and WRONG_TOOL_RECOVERY:
        if last_answer_text and step_results and step_results[-1]["lenient_pass"]:
            n_steps = len(step_results)
            n_failures_before_last = sum(
                1 for s in step_results[:-1] if not s["lenient_pass"]
            )
            max_allowed_failures = max(1, n_steps // 3)
            if n_failures_before_last <= max_allowed_failures:
                outcome_ok, _ = is_lenient_acceptable_answer(last_answer_text, expected_facts)
                if outcome_ok:
                    lenient_pass = True

    outcome_pass = False
    if last_answer_text:
        ok, _ = is_lenient_acceptable_answer(last_answer_text, expected_facts)
        outcome_pass = ok

    first_failure_strict = next((s["step_index"] for s in step_results if not s["strict_pass"]), None)
    first_failure_lenient = next((s["step_index"] for s in step_results if not s["lenient_pass"]), None)

    merged_memory_used = any(s.get("merged_memory_covered") or s.get("merged_memory_combined_output")
                             for s in step_results)

    return {
        "sample_id": sample["id"],
        "tier": sample.get("tier", "unknown"),
        "category": sample["category"],
        "question": question,
        "strict_pass": strict_pass,
        "lenient_pass": lenient_pass,
        "outcome_pass": outcome_pass,
        "chain_length": len(step_results),
        "expected_chain_length": len(steps),
        "generated_turn_count": generated_turn_count,
        "merged_memory_used": merged_memory_used,
        "first_failure_step_strict": first_failure_strict,
        "first_failure_step_lenient": first_failure_lenient,
        "final_answer_text": last_answer_text,
        "step_results": step_results,
    }


# ============================================================
# Suite runner
# ============================================================

def run_suite_v4(processor, model, label: str, samples: List[Dict]) -> List[Dict]:
    print(f"\n{'='*80}\nRunning {label} | {len(samples)} samples\n{'='*80}")
    results = []
    for idx, sample in enumerate(samples, 1):
        print(f"\n[{idx}/{len(samples)}] {sample['id']} ({sample.get('tier','?')}/{sample['category']})")
        print(f"  Q: {sample['question'][:120]}")
        result = evaluate_chain_sample_v4(processor, model, sample)
        results.append(result)
        for step in result["step_results"]:
            tag_strict = "PASS" if step["strict_pass"] else "FAIL"
            tag_lenient = "PASS" if step["lenient_pass"] else "FAIL"
            extras = []
            if step.get("action_input_correct") is False:
                extras.append("input_X")
            if step.get("persona_correct") is False:
                extras.append("persona_X")
            if step.get("answer_content_correct") is False:
                extras.append("content_X")
            if step.get("lenient_early_stop_ok"):
                extras.append("lenient_stop")
            if step.get("merged_memory_covered"):
                extras.append("merged_memory")
            if step.get("merged_memory_combined_output"):
                extras.append("combined_memory_output")
            if step.get("action_compat_detail") and step.get("action_compat_detail") != "exact":
                extras.append("compat")
            extras_s = f" [{', '.join(extras)}]" if extras else ""
            print(f"    strict[{tag_strict}] lenient[{tag_lenient}] step{step['step_index']}: "
                  f"mode={step['model_mode']} action={step['first_action']} "
                  f"err={step['error_type']}{extras_s}")
        outcome_tag = "PASS" if result["outcome_pass"] else "FAIL"
        merged_tag = " merged_memory" if result.get("merged_memory_used") else ""
        print(f"  -> strict={result['strict_pass']}, lenient={result['lenient_pass']}, "
              f"outcome[{outcome_tag}], generated_turns={result.get('generated_turn_count')}{merged_tag}")
    return results


# ============================================================
# Summary
# ============================================================

def summarize_v4(label: str, results: List[Dict]):
    print(f"\n{'='*80}\nSUMMARY: {label}\n{'='*80}")

    total = len(results)
    if total == 0:
        return

    strict_pass = sum(r["strict_pass"] for r in results)
    lenient_pass = sum(r["lenient_pass"] for r in results)
    outcome_pass = sum(r["outcome_pass"] for r in results)

    print(f"\nOverall:")
    print(f"  STRICT  : {strict_pass}/{total} ({strict_pass/total*100:.1f}%)")
    print(f"  LENIENT : {lenient_pass}/{total} ({lenient_pass/total*100:.1f}%)")
    print(f"  OUTCOME : {outcome_pass}/{total} ({outcome_pass/total*100:.1f}%)")

    tiers = sorted(set(r["tier"] for r in results))
    print(f"\nBy Tier (strict / lenient / outcome):")
    for tier in tiers:
        sub = [r for r in results if r["tier"] == tier]
        s_pass = sum(r["strict_pass"] for r in sub)
        l_pass = sum(r["lenient_pass"] for r in sub)
        o_pass = sum(r["outcome_pass"] for r in sub)
        print(f"  {tier:20s}: strict={s_pass}/{len(sub)} ({s_pass/len(sub)*100:.1f}%) | "
              f"lenient={l_pass}/{len(sub)} ({l_pass/len(sub)*100:.1f}%) | "
              f"outcome={o_pass}/{len(sub)} ({o_pass/len(sub)*100:.1f}%)")

    cats = sorted(set(r["category"] for r in results))
    print(f"\nBy Category (LENIENT):")
    for cat in cats:
        sub = [r for r in results if r["category"] == cat]
        n_pass = sum(r["lenient_pass"] for r in sub)
        print(f"  {cat:32s}: {n_pass}/{len(sub)} ({n_pass/len(sub)*100:.1f}%)")

    all_steps = [s for r in results for s in r["step_results"]]
    err_counts = {}
    for s in all_steps:
        et = s.get("error_type")
        if et:
            err_counts[et] = err_counts.get(et, 0) + 1
    if err_counts:
        print(f"\nError Types:")
        for k, v in sorted(err_counts.items(), key=lambda x: -x[1]):
            print(f"  {k:30s}: {v}")

    n_action_input_checked = sum(1 for s in all_steps if s.get("action_input_correct") is not None)
    n_action_input_pass = sum(1 for s in all_steps if s.get("action_input_correct") is True)
    n_persona_checked = sum(1 for s in all_steps if s.get("persona_correct") is not None)
    n_persona_pass = sum(1 for s in all_steps if s.get("persona_correct") is True)
    n_content_checked = sum(1 for s in all_steps if s.get("answer_content_correct") is not None)
    n_content_pass = sum(1 for s in all_steps if s.get("answer_content_correct") is True)

    persona_scores = [s.get("persona_score") for s in all_steps
                      if s.get("persona_score") is not None]
    avg_persona = sum(persona_scores) / len(persona_scores) if persona_scores else 0

    print(f"\nFine-grained checks:")
    if n_action_input_checked:
        print(f"  action_input correctness : {n_action_input_pass}/{n_action_input_checked} "
              f"({n_action_input_pass/n_action_input_checked*100:.1f}%)")
    if n_persona_checked:
        print(f"  persona correctness      : {n_persona_pass}/{n_persona_checked} "
              f"({n_persona_pass/n_persona_checked*100:.1f}%) "
              f"avg_score={avg_persona:.2f}/3")
    if n_content_checked:
        print(f"  answer content match     : {n_content_pass}/{n_content_checked} "
              f"({n_content_pass/n_content_checked*100:.1f}%)")

    lenient_stops = sum(1 for s in all_steps if s.get("lenient_early_stop_ok"))
    if lenient_stops:
        print(f"\n  Lenient early stops: {lenient_stops}")

    merged_memory_count = sum(1 for r in results if r.get("merged_memory_used"))
    if merged_memory_count:
        print(f"  Merged MemorySearch(target_entity='Both') chains: {merged_memory_count}")

    compat_steps = sum(
        1 for s in all_steps
        if s.get("action_compat_detail")
        and s.get("action_compat_detail") not in {"exact", "answer step"}
    )
    if compat_steps:
        print(f"  Compatible action substitutions: {compat_steps}")


# ============================================================
# Compare A vs B
# ============================================================

def compare_v4(name_a: str, results_a: List[Dict],
               name_b: str, results_b: List[Dict]):
    print(f"\n{'='*80}\nA-vs-B: {name_a}  vs  {name_b}  (multi-metric)\n{'='*80}")
    map_a = {r["sample_id"]: r for r in results_a}
    map_b = {r["sample_id"]: r for r in results_b}
    common = set(map_a) & set(map_b)

    for tier in sorted(set(map_a[sid]["tier"] for sid in common)):
        print(f"\n[{tier}]")
        for sid in sorted(common):
            if map_a[sid]["tier"] != tier:
                continue
            a, b = map_a[sid], map_b[sid]
            ma_l = "PASS" if a["lenient_pass"] else "FAIL"
            mb_l = "PASS" if b["lenient_pass"] else "FAIL"
            ma_o = "PASS" if a["outcome_pass"] else "FAIL"
            mb_o = "PASS" if b["outcome_pass"] else "FAIL"
            ma_m = "M" if a.get("merged_memory_used") else "-"
            mb_m = "M" if b.get("merged_memory_used") else "-"
            improvement = ""
            if not a["lenient_pass"] and b["lenient_pass"]:
                improvement = "  GAINED"
            elif a["lenient_pass"] and not b["lenient_pass"]:
                improvement = "  LOST"
            print(f"  {sid:25s} lenient[{ma_l}->{mb_l}]  outcome[{ma_o}->{mb_o}] "
                  f"merged[{ma_m}->{mb_m}]{improvement}")


# ============================================================
# Save
# ============================================================

def save_results(path: str, label: str, results: List[Dict]):
    payload = {
        "model_name": label,
        "config": {
            "STRICT_FORMAT": STRICT_FORMAT,
            "CHECK_ACTION_INPUT": CHECK_ACTION_INPUT,
            "CHECK_PERSONA": CHECK_PERSONA,
            "LENIENT_EARLY_STOP": LENIENT_EARLY_STOP,
            "PERSONA_GRADED": PERSONA_GRADED,
            "WRONG_TOOL_RECOVERY": WRONG_TOOL_RECOVERY,
            "PERSONA_SOFT_ON_NO_TOOL": PERSONA_SOFT_ON_NO_TOOL,
            "MEMORY_BOTH_TARGET_SUPERSET": MEMORY_BOTH_TARGET_SUPERSET,
            "MERGED_MEMORY_BOTH_COMPAT": MERGED_MEMORY_BOTH_COMPAT,
            "CURRENT_TIME_TOOL_COMPAT": CURRENT_TIME_TOOL_COMPAT,
        },
        "results": results,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved: {path}")


# ============================================================
# Main
# ============================================================

def main():
    all_model_results = {}

    out_dir = "benchmarks/results"

    proc1, model1 = load_model(ORIGINAL_MODEL_PATH, "Original Model")
    orig_results = run_suite_v4(proc1, model1, "Original Model", ALL_SAMPLES)
    summarize_v4("Original Model", orig_results)
    if SAVE_JSON:
        save_results(f"{out_dir}/eval_v4_original.json", "Original Model", orig_results)
    all_model_results["original"] = orig_results
    unload_model(model1, proc1)

    proc2, model2 = load_model(EVA_MODEL_PATH, "Eva SFT Model")
    eva_results = run_suite_v4(proc2, model2, "Eva SFT Model", ALL_SAMPLES)
    summarize_v4("Eva SFT Model", eva_results)
    if SAVE_JSON:
        save_results(f"{out_dir}/eval_v4_eva.json", "Eva SFT Model", eva_results)
    all_model_results["eva"] = eva_results
    unload_model(model2, proc2)

    compare_v4("Original", all_model_results["original"], "Eva", all_model_results["eva"])


if __name__ == "__main__":
    main()
