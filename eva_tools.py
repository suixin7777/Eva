"""
eva_tools.py — Public façade for Eva's ReAct tool layer.

Tool names the model emits in <|tool_code|> blocks (MemorySearch, WebSearch,
GetCurrentTime, TextGenerationTool, AskRemoteVision) are declared as
prompt-only signatures inside TOOLS_OPTIMIZED in eva_prompts. The actual
execution happens through the helpers re-exported below.

Layering:
- Pure runtime executors + ReAct text parsers live in eva_tools_runtime.
- Memory retrieval (the FAISS+BM25+rerank pipeline) still lives in
  eva_core (will move to eva_memory_legacy in step 7).
This shim collects both into one import path.
"""

from eva_tools_runtime import (
    # ReAct parsing
    sanitize_tool_code,
    parse_react_block,
    clean_search_content,
    # Concrete executors
    websearch_tavily,
    run_websearch,
    call_deepseek_expert,
    call_remote_vision,
)
from eva_core import (
    # Memory retrieval pipeline (still in legacy core, to be split in step 7)
    run_memory_search,
    load_memory_resources,
)

__all__ = [
    "run_websearch",
    "websearch_tavily",
    "call_deepseek_expert",
    "call_remote_vision",
    "run_memory_search",
    "sanitize_tool_code",
    "parse_react_block",
    "clean_search_content",
    "load_memory_resources",
]
