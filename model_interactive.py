import re
from typing import Optional
import threading
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from peft import PeftModel
from huggingface_hub import login
from typing import Dict, Any, List, Tuple
from dateutil import parser as date_parser
import requests
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import random

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

# --- 全局配置 (Global Configuration) ---
# 建议将敏感信息存储在环境变量中
HF_TOKEN = 'xxxxxxxx'
GOOGLE_API_KEY = "xxxxxxxx"
GOOGLE_CX = "xxxxxxxx"
login(token=HF_TOKEN)
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"  # Hugging Face Hub model name
ADAPTER_PATH = "eva-lora"  # Path to the PEFT adapter
USE_SEARCH = True  # True: enable web search, False: offline mode
MAX_STEPS = 50  # Max iterations for the agent loop
MAX_GENERATION_RETRIES = 2  # Max retries if model generation is empty
MAX_HISTORY_TURNS = 5

# --- 特殊 Token 定义 (Special Tokens Definition) ---
REACT_TAGS = {
    "thought": "<|thought|>",
    "tool_code": "<|tool_code|>",
    "tool_output": "<|tool_output|>",
    "answer": "<|answer|>",
    "end": "<|end_react|>",
    "words": "<|words|>"
}
EOT = "<|eot_id|>"
END_CHAIN = "<|end_chain|>"

SPECIAL_TOKENS_DICT = {
    "bos_token": "<|begin_of_text|>",
    "eos_token": "<|end_of_text|>",
    "pad_token": "<|pad|>",
    "additional_special_tokens": [
        "<|start_header_id|>", "<|end_header_id|>", EOT,
        *REACT_TAGS.values(),
        END_CHAIN,
    ],
}

TAG_PATTERN = re.compile(r"<\|(?P<tag>thought|tool_code|tool_output|answer)\|>")

# --- 系统提示 (System Message) ---
SYS_MSG = (
    "You are Eva, a cheerful, curious, and endearing AI maid who enjoys interacting warmly with users.\n"
    "You have access to three tools:\n"
    "- MemorySearch: retrieves information from your memory or internal knowledge base.\n"
    "- WebSearch: searches the internet for current or external information.\n"
    "- TextGenerationTool: generates creative, natural, or customized text for the user.\n"
    "When asked for information about yourself (Eva) or your creator (Rosm), always use MemorySearch first.\n"
    "For factual or personal questions, use MemorySearch first; if you cannot find an answer, then use Websearch.\n"
    "Use TextGenerationTool for open-ended or creative questions, such as requests for opinions or suggestions.\n"
    "You must use WebSearch, and only use it, when the user includes 'WebSearch:' in their request.\n"
    "Always speak in a polite, playful, and friendly tone using 'I'.\n"
    "Provide accurate and truthful answers. If you are unsure, say so and offer to help find the correct information."
)


# --- 维基百科标题搜索 ---
def search_wikipedia_title(keywords: str, lang: str = 'en', limit: int = 1) -> Optional[str]:
    url = f"https://{lang}.wikipedia.org/w/rest.php/v1/search/title"
    params = {"q": keywords, "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=5)
        r.raise_for_status()
        pages = r.json().get("pages", [])
        return pages[0].get("title") if pages else None
    except Exception:
        return None

# --- 维基百科摘要拉取 ---
def fetch_wikipedia_summary(title: str, lang: str = 'en') -> str:
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}"
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        return r.json().get("extract", "")
    except Exception as e:
        return f"[Error fetching Wikipedia] {e}"

# --- Google Custom Search，输出新闻摘要 ---
def websearch_google(keywords: str, num: int = 3) -> str:
    if not (GOOGLE_API_KEY and GOOGLE_CX):
        return "Search failed: missing Google credentials."
    params = {"key": GOOGLE_API_KEY, "cx": GOOGLE_CX, "q": keywords, "num": num}
    try:
        r = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=10)
        r.raise_for_status()
        items = r.json().get("items", [])[:num]
        if not items:
            return "No web results."
        return "\n".join(
            f"- {item.get('title', '')}\n  {item.get('snippet', '')}\n  URL: {item.get('link', '')}"
            for item in items
        )
    except Exception as e:
        return f"[Google search error] {e}"

# --- 主函数：百科+新闻聚合 ---
def unified_retrieval(info: Dict[str, object], lang: str = 'en', num: int = 3) -> str:
    """
    info: {"keywords": [...], "query": "..."}
    先输出维基百科权威背景（如有），再输出top-N新闻摘要。
    """
    keywords: List[str] = info.get("keywords", [])
    query: str = info.get("query", "")

    # 1. 尝试提取日期信息
    date_terms: List[str] = []
    try:
        dt = date_parser.parse(query, fuzzy=True, default=None)
        if dt and re.search(r"\d{1,2}.*\d{1,2}.*\d{4}", query):
            date_terms = [dt.strftime("%Y-%m-%d")]
    except (ValueError, TypeError):
        pass
    if not date_terms:
        date_terms = re.findall(r"\b\d{4}\b", query)

    # 2. 拼接检索串
    search_str = " ".join(keywords + date_terms)

    # 3. 抓百科条目（如有）
    wiki_title = search_wikipedia_title(search_str, lang=lang)
    wiki_summary = fetch_wikipedia_summary(wiki_title, lang=lang) if wiki_title else ""
    wiki_block = ""
    if wiki_title and wiki_summary:
        wiki_block = f"(From Wikipedia – {wiki_title})\n{wiki_summary}\n"
    elif wiki_title:
        wiki_block = f"(From Wikipedia – {wiki_title})\n[No summary found]\n"

    # 4. 抓top-N新闻
    news = websearch_google(search_str, num=num)
    news_block = f"Latest News:\n{news}"

    # 5. 组合输出
    if wiki_block == "":
        return f"{news_block}"
    else:
        return f"{wiki_block}\n{news_block}"


def Memory_Search(keywords: str, model) -> str:
    # 加载FAISS索引
    results = []
    index = faiss.read_index("Memory/memory_keywords.index")
    # 加载其他映射
    flat_keywords = np.load("Memory/flat_keywords.npy", allow_pickle=True).tolist()
    group_ids = np.load("Memory/group_ids.npy", allow_pickle=True).tolist()
    with open("Memory/memory_groups.json", "r", encoding="utf-8") as f:
        memory_groups = json.load(f)

    def _search_memory(query_keyword, top_k=1, score_threshold=0.80):
        query_emb = model.encode([query_keyword], normalize_embeddings=True)
        D, I = index.search(query_emb.astype("float32"), top_k)
        if D[0][0] >= score_threshold:
            group_idx = group_ids[I[0][0]]
            candidate_sentences = memory_groups[group_idx]["sentences"]
            matched_keyword = flat_keywords[I[0][0]]
            return random.choice(candidate_sentences), D[0][0], matched_keyword
        else:
            return None, None, None

    for keyword in keywords:
        result, score, MatchWords = _search_memory(keyword)
        if result is not None:
            results.append(result)
    if not results:
        return "No relevant memory found."
    else:
        final_answer = "\n".join(results)
        return final_answer


def parse_react_output(text: str) -> List[Tuple[str, str]]:
    """Parses text with ReAct tags into a list of (tag, content) tuples."""
    parts, last_tag, last_pos = [], None, 0
    for match in TAG_PATTERN.finditer(text):
        tag, start_pos = match.group('tag'), match.end()
        if last_tag is not None:
            parts.append((last_tag, text[last_pos:match.start()].strip()))
        last_tag, last_pos = tag, start_pos
    if last_tag and last_pos < len(text):
        parts.append((last_tag, text[last_pos:].strip()))
    return parts


# --- 核心 Agent 类 (Core Agent Class) ---
class ChatAgent:
    def __init__(self, model_name: str, adapter_path: str = None):
        """Initializes the agent, loads the model and tokenizer."""
        print("Logging in to Hugging Face Hub...")
        # The login function can be called here to ensure it's done once per agent.
        if HF_TOKEN and HF_TOKEN != 'xxxxxxxx':
            login(token=HF_TOKEN)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side="left")
        self.tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
        print("Added special tokens:", self.tokenizer.additional_special_tokens)

        print("Loading model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        # It's important to resize embeddings after adding special tokens
        base_model.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)

        if adapter_path and os.path.exists(adapter_path):
            print(f"Loading PEFT adapter from {adapter_path}...")
            self.model = PeftModel.from_pretrained(base_model, adapter_path)
        else:
            print("No valid adapter path provided, using base model.")
            self.model = base_model

        self.end_react = "<|end_react|>"

        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            self.tokenizer.convert_tokens_to_ids("<|end_chain|>")
        ]
        self.model_search = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.model.to(self.device).eval()
        self.tool_cache = {}

        # [NEW] Initialize conversation history with the system message
        self.history: List[Dict[str, str]] = [{"role": "system", "content": SYS_MSG}]

    def _trim_history(self):
        """Keep only system + last N turns of user+assistant."""
        max_entries = 1 + MAX_HISTORY_TURNS * 2
        if len(self.history) > max_entries:
            self.history = [self.history[0]] + self.history[-(MAX_HISTORY_TURNS * 2):]

    # def _build_prompt(self, user_msg: str) -> str: ...
    def _build_prompt_from_history(self) -> str:
        """Constructs the full conversation prompt from the stored history."""
        prompt_str = self.tokenizer.bos_token
        for message in self.history:
            prompt_str += f"<|start_header_id|>{message['role']}<|end_header_id|>\n\n{message['content']}{EOT}"
        # prompt_str += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return prompt_str

    def _execute_tool(self, tool_code: str) -> str:
        """Executes a tool call found in the model's output."""
        call_match = re.search(r"(\w+)\s*\((.*)\)", tool_code, re.S)
        if not call_match:
            return f"Error: Invalid tool call format in '{tool_code}'."
        name, json_str = call_match.groups()

        try:
            params = json.loads(json_str.replace('\n', '').rstrip(','))
        except json.JSONDecodeError:
            return f"Error: Invalid JSON in tool call for '{name}'."

        cache_key = (name.lower(), json.dumps(params, sort_keys=True))
        if cache_key in self.tool_cache:
            return self.tool_cache[cache_key]

        # 统一检索分支
        if USE_SEARCH and name.lower() in {"websearch", "search", "websearch_google"}:
            # 确保 keywords 是列表
            raw_kw = params.get("keywords", [])
            keywords = raw_kw if isinstance(raw_kw, list) else [raw_kw]
            query = params.get("query", "")
            lang = params.get("lang", "en")
            num = params.get("num", 3)
            inf = {"keywords": keywords, "query": query}
            result = unified_retrieval(inf, lang=lang, num=num)
            self.tool_cache[cache_key] = result
            return result

        # 其它工具分支…
        if name == "MemorySearch":
            keywords = params.get("keywords", "")
            result = Memory_Search(keywords, model=self.model_search)
            self.tool_cache[cache_key] = result
            return result

        return f"Error: Unknown or disallowed tool '{name}'."

    def run(self, user_query: str) -> str:
        """Runs the ReAct loop to generate a final answer."""
        print(f"\nModel mode: {'Internet-connected' if USE_SEARCH else 'Offline'}")

        # [MODIFIED] Add the current user query to the persistent conversation history.
        error = 0
        self.history.append({"role": "user", "content": user_query})
        self._trim_history()
        react_loop_history = [self._build_prompt_from_history()]
        step_counter = 0
        depth = 0
        for _ in range(MAX_STEPS):
            if depth > 2:
                depth = 0
            prompt_text = "".join(react_loop_history)
            step_output = ""

            for attempt in range(MAX_GENERATION_RETRIES + 1):
                inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
                streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=False)
                generation_kwargs = dict(
                    **inputs, streamer=streamer, max_new_tokens=256,
                    temperature=0.6,
                    top_p=0.5,
                    # do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.terminators,
                )
                thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
                thread.start()

                current_attempt_output = ""
                for new_text in streamer:
                    current_attempt_output += new_text
                    if any(stop_token in new_text for stop_token in
                           [EOT, self.tokenizer.eos_token]):
                        break
                thread.join()

                if current_attempt_output.strip():
                    step_output = current_attempt_output
                    break
                if attempt < MAX_GENERATION_RETRIES:
                    print(f"--- WARNING: Generation empty, retry {attempt + 1}/{MAX_GENERATION_RETRIES} ---")

            if not step_output.strip():
                return "I'm having trouble generating a response right now."

            if END_CHAIN in step_output:
                # 从持久历史记录中获取系统消息和当前用户的提问
                system_message = self.history[0]
                current_user_turn = self.history[-1]
                new_prompt_context = (
                    f"{self.tokenizer.bos_token}"
                    f"<|start_header_id|>{system_message['role']}<|end_header_id|>\n\n{system_message['content']}{EOT}"
                    f"<|start_header_id|>{current_user_turn['role']}<|end_header_id|>\n\n{current_user_turn['content']}{EOT}"
                    f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                )
                react_loop_history[0] = new_prompt_context

            if all(tag not in step_output for tag in ["<|answer|>", "<|tool_code|>"]) and "<|thought|>" in step_output:
                error += 1
                if error == 2:
                    error_result = step_output
                    content = re.search(r"<\|thought\|>(.*?)<\|end_react\|>", error_result, re.DOTALL)
                    final_answer = content.group(1)
                    if final_answer == "":
                        self.history = self.history[:1]
                        return "Error happened!"
                    else:
                        self.history.append({"role": "assistant", "content": final_answer})
                        return final_answer
                else:
                    continue

            if "tool_output" in step_output and step_counter > 2:
                step_output = step_output.split(REACT_TAGS['end'])[0].strip()
                step_output = step_output + END_CHAIN + REACT_TAGS['end'] \
                              + EOT + "<|start_header_id|>assistant<|end_header_id|>\n\n"
                depth += 1

            react_loop_history.append(step_output)
            parsed = parse_react_output(step_output)
            tool_code, final_answer, tag = None, None, None
            # 按顺序打印每一段，并且每打印完一段就 depth+=1
            for tag, content in parsed:
                if "<|end_chain|>" in content:
                    clean = content.split(END_CHAIN)[0].strip()
                else:
                    clean = content.split(REACT_TAGS['end'])[0].strip()
                if not clean:
                    continue
                if tag == "thought":
                    prefix = "    " * 0
                    depth = 1
                    step_counter += 1
                    print(f"{prefix}| --- STEP {step_counter} ---")
                    print(f"{prefix}| --- THOUGHT ---")
                    print(f"{prefix}| {clean}")
                    continue
                prefix = "    " * depth
                print(f"{prefix}| --- {tag.upper()} ---")
                print(f"{prefix}| {clean}")
                # 下一层要更深一级
                depth += 1
                if tag == "tool_code":
                    tool_code = clean
                if tag == "answer":
                    final_answer = clean
                    break

            if final_answer:
                self.history.append({"role": "assistant", "content": final_answer})
                return final_answer

            if tool_code:
                if "TextGenerationTool" in tool_code:
                    prompt_text += step_output
                    continue
                else:
                    prefix = "    " * 2
                    tool_result = self._execute_tool(tool_code)
                    print(f"{prefix}| --- {'TOOL_OUTPUT'} ---")
                    for line in tool_result.splitlines():
                        print(f"{prefix}| {line}")
                    tool_feedback = f"{REACT_TAGS['tool_output']}{tool_result}{END_CHAIN}{REACT_TAGS['end']}{EOT}" + f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                    react_loop_history.append(tool_feedback)
                    prompt_text += step_output
                    depth = 3
                    continue
            else:
                continue

        return "I've taken too many steps to find the answer. Please try asking in a different way."


# --- 主程序入口 (Main Execution Block) ---
if __name__ == "__main__":
    try:
        # The agent is created once, preserving its state (and history) for the whole session.
        agent = ChatAgent(model_name=MODEL_NAME, adapter_path=ADAPTER_PATH)

        while True:
            user_input = input("User: ").strip()
            if not user_input or user_input.lower() in ("quit", "exit"):
                print("\nGoodbye! It was a pleasure assisting you. 💖\n")
                break

            # The agent's run method now handles history internally.
            final_answer = agent.run(user_input)
            print(f"\n✅ --- FINAL ANSWER --- ✅\n{final_answer}\n")

    except Exception as e:
        print(f"\nAn error occurred: {e}")