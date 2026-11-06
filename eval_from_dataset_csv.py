import ast
import csv
import json
import math
import os
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import time
from datetime import datetime, timezone
from google import genai
from google.genai import errors as genai_errors

def safe_model_name(model: str) -> str:
    # qwen3:8b -> qwen3_8b
    return model.replace(":", "_").replace("/", "_")
import json
import os
from dataclasses import asdict

def save_repo_index_json(index, json_path: str) -> None:
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    payload = {
        "repo_root": index.repo_root,
        "entries": [asdict(e) for e in index.entries],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

def load_repo_index_json(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    idx = TempRepoIndex(payload["repo_root"])
    idx.entries = [FunctionEntry(**e) for e in payload["entries"]]
    idx._build_bm25()  # rebuild retrieval structures in memory
    return idx
def repo_json_path(base_dir: str, repo_root: str, model: str, mode: str) -> str:
    repo_name = os.path.basename(os.path.abspath(repo_root))
    model_dir = safe_model_name(model)
    return os.path.join(
        base_dir,
        model_dir,
        mode,
        f"{repo_name}.functions.json"
    )

# Optional deps (install only what you use):
# pip install openai anthropic google-genai
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from anthropic import Anthropic
except Exception:
    Anthropic = None

try:
    from google import genai
except Exception:
    genai = None


# ============================================================
# LLM Providers (OpenAI / Anthropic / Gemini / local OpenAI-compat)
# ============================================================

class LLMProvider:
    provider_key: str
    model_name: str
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    def __init__(self, model_name: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
        if OpenAI is None:
            raise RuntimeError("openai not installed. Run: pip install openai")
        self.provider_key = "openai"
        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or None,
        )

    def generate(self, prompt: str) -> str:
    # 1) Try OpenAI Responses API (works on OpenAI; some OpenAI-compatible servers may not support it)
        try:
            resp = self.client.responses.create(model=self.model_name, input=prompt)
            text = ""
            for item in getattr(resp, "output", []) or []:
                if getattr(item, "type", None) == "message":
                    for c in getattr(item, "content", []) or []:
                        if getattr(c, "type", None) == "output_text":
                            text += getattr(c, "text", "")
            out = (text or getattr(resp, "output_text", "") or "").strip()
            if out:
                return out
        except Exception:
            pass

        # 2) Fallback: Chat Completions API (supported by Ollama and many local servers)
        cc = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return (cc.choices[0].message.content or "").strip()



class AnthropicProvider(LLMProvider):
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        if Anthropic is None:
            raise RuntimeError("anthropic not installed. Run: pip install anthropic")
        self.provider_key = "anthropic"
        self.model_name = model_name
        self.client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def generate(self, prompt: str) -> str:
        msg = self.client.messages.create(
            model=self.model_name,
            max_tokens=900,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        out = []
        for b in msg.content:
            if getattr(b, "type", None) == "text":
                out.append(getattr(b, "text", ""))
        return "".join(out).strip()


class GeminiProvider(LLMProvider):
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        if genai is None:
            raise RuntimeError("google-genai not installed. Run: pip install google-genai")
        self.provider_key = "gemini"
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))

    # def generate(self, prompt: str) -> str:
    #     resp = self.client.models.generate_content(model=self.model_name, contents=prompt)
    #     return (getattr(resp, "text", "") or "").strip()
    def generate(self, prompt: str) -> str:
        max_retries = 6
        backoff = 1.5  # seconds

        for attempt in range(max_retries):
            try:
                resp = self.client.models.generate_content(model=self.model_name, contents=prompt)
                return resp.text.strip()

            except genai_errors.ServerError as e:
                if "503" in str(e) or "UNAVAILABLE" in str(e):
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    raise

def max_code_chars_for(model_name: str) -> int:
    """
    Heuristic mapping: increase code budget for larger-context models.
    Char↔token is approximate; keep a safety margin.
    """
    m = (model_name or "").lower()

    # Very large-context models
    if "gemini" in m and ("2.5" in m or "pro" in m):
        return 60000  # big, but still bounded
    if "claude" in m and "sonnet" in m:
        return 50000
    if m.startswith("gpt-5"):
        return 40000

    # Common OSS large models (often 32k context)
    if "70b" in m or "30b" in m:
        return 24000

    # Smaller models (8k–16k-ish)
    if "8b" in m or "3b" in m:
        return 8000

    return DEFAULT_MAX_CODE_CHARS

def build_provider(provider: str, model: str) -> LLMProvider:
    provider = provider.lower()
    if provider == "openai":
        return OpenAIProvider(model_name=model)
    if provider == "anthropic":
        return AnthropicProvider(model_name=model)
    if provider == "gemini":
        return GeminiProvider(model_name=model)
    if provider == "local":
        base_url = os.environ.get("LOCAL_OPENAI_BASE_URL")
        api_key = os.environ.get("LOCAL_OPENAI_API_KEY", "local-key")
        if not base_url:
            raise RuntimeError("Set LOCAL_OPENAI_BASE_URL (e.g., http://localhost:8000/v1) for provider=local")
        return OpenAIProvider(model_name=model, api_key=api_key, base_url=base_url)
    raise ValueError(f"Unknown provider: {provider}")


# ============================================================
# AST function parsing (Python)
# ============================================================

IGNORE_DIRS = {
    ".git", ".hg", ".svn", ".tox", ".venv", "venv", "__pycache__",
    "node_modules", "dist", "build", ".mypy_cache", ".pytest_cache"
}
DEFAULT_MAX_CODE_CHARS = 8000  # fallback if model unknown


@dataclass
class FunctionEntry:
    file_path: str
    qualname: str
    signature: str
    start_line: int
    end_line: int
    docstring: str
    code: str
    descriptions: Dict[str, str]  # mode -> text


def iter_py_files(repo_root: str) -> List[str]:
    out = []
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for f in files:
            if f.endswith(".py"):
                out.append(os.path.join(root, f))
    return out


class _Visitor(ast.NodeVisitor):
    def __init__(self, source_lines: List[str]):
        self.source_lines = source_lines
        self.stack: List[str] = []
        self.nodes: List[Tuple[str, ast.AST]] = []

    def visit_ClassDef(self, node: ast.ClassDef):
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        qual = ".".join(self.stack + [node.name]) if self.stack else node.name
        self.nodes.append((qual, node))
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        qual = ".".join(self.stack + [node.name]) if self.stack else node.name
        self.nodes.append((qual, node))
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()


def _end_lineno(node: ast.AST) -> int:
    return int(getattr(node, "end_lineno", None) or getattr(node, "lineno", 1))


def _signature(node: ast.AST) -> str:
    assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    args = node.args
    parts: List[str] = []

    def arg(a: ast.arg) -> str:
        return a.arg

    for a in getattr(args, "posonlyargs", []):
        parts.append(arg(a))
    if getattr(args, "posonlyargs", []):
        parts.append("/")

    for a in args.args:
        parts.append(arg(a))

    if args.vararg:
        parts.append("*" + args.vararg.arg)
    elif args.kwonlyargs:
        parts.append("*")

    for a in args.kwonlyargs:
        parts.append(arg(a))

    if args.kwarg:
        parts.append("**" + args.kwarg.arg)

    prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
    return f"{prefix}{node.name}({', '.join(parts)})"


def _slice_code(lines: List[str], start: int, end: int, max_code_chars: int) -> str:
    code = "".join(lines[start - 1:end])
    if len(code) > max_code_chars:
        code = code[:max_code_chars] + "\n# ... (truncated)\n"
    return code


def parse_file(file_path: str, max_code_chars: int) -> List[Tuple[str, str, int, int, str, str]]:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        src = f.read()
    lines = src.splitlines(keepends=True)

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []

    v = _Visitor(lines)
    v.visit(tree)

    out = []
    for qual, node in v.nodes:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        start = int(getattr(node, "lineno", 1))
        end = _end_lineno(node)
        sig = _signature(node)
        doc = ast.get_docstring(node) or ""
        code = _slice_code(lines, start, end, max_code_chars=max_code_chars)
        out.append((qual, sig, start, end, doc, code))
    return out


# ============================================================
# Prompting modes
# ============================================================

ONE_SHOT_EXAMPLE = {
        "signature": "def split_groups_balanced(group_to_items: dict[str, list[int]], k: int = 5) -> list[tuple[list[int], list[int]]]",
        "docstring": "Create K folds where each group appears in exactly one test fold, balancing total items.",
        "code": """def split_groups_balanced(group_to_items: dict[str, list[int]], k: int = 5):
    items = sorted(group_to_items.items(), key=lambda kv: len(kv[1]), reverse=True)
    fold_groups = [set() for _ in range(k)]
    fold_sizes = [0] * k

    for g, xs in items:
        f = min(range(k), key=lambda i: fold_sizes[i])
        fold_groups[f].add(g)
        fold_sizes[f] += len(xs)

    all_g = set(group_to_items)
    folds = []
    for f in range(k):
        test_g = fold_groups[f]
        train = [x for g in (all_g - test_g) for x in group_to_items[g]]
        test = [x for g in test_g for x in group_to_items[g]]
        folds.append((train, test))
    return folds
""",
        "description": "Builds K group-exclusive cross-validation folds by assigning whole groups to test folds while keeping fold sizes roughly balanced. Groups are processed largest-first and placed into the currently smallest fold. Returns a list of (train_items, test_items) pairs and performs no external I/O."
    }
TWO_SHOT_EXAMPLES = [
    ONE_SHOT_EXAMPLE,
    {
        "signature": "def merge_feature_maps(a: dict[str, list[str]], b: dict[str, list[str]]) -> dict[str, list[str]]",
        "docstring": "Merge two feature maps and deduplicate values per key.",
        "code": """def merge_feature_maps(a: dict[str, list[str]], b: dict[str, list[str]]):
    out = {k: list(v) for k, v in a.items()}
    for k, vals in b.items():
        out.setdefault(k, [])
        out[k].extend(vals)
        out[k] = list(dict.fromkeys(out[k]))  # stable dedupe
    return out
""",
        "description": "Combines two dictionaries that map keys to lists of feature strings by concatenating values for matching keys and removing duplicates while preserving order. Returns a new merged dictionary and does not mutate the input mappings."
    }
]


# TWO_SHOT_EXAMPLES = [
#     ONE_SHOT_EXAMPLE,
#     {
#         "signature": "def write_json(path, obj)",
#         "docstring": "Write obj as JSON to path.",
#         "code": "import json\n\ndef write_json(path, obj):\n    with open(path, 'w') as f:\n        json.dump(obj, f)\n",
#         "description": "Serializes the given Python object to JSON and writes it to the specified file path. Performs filesystem I/O and may raise exceptions if the path is invalid or the object is not JSON-serializable."
#     }
# ]

def make_description_prompt(mode: str, signature: str, docstring: str, code: str) -> str:
    base_instr = (
        "You generate a concise, search-friendly description of a Python function.\n"
        "Return ONLY one paragraph (2–5 sentences) describing:\n"
        "- what the function does\n"
        "- key inputs/outputs\n"
        "- side effects (filesystem/network/db) if any\n"
        "- notable edge cases or assumptions\n"
    )

    if mode == "zero_shot":
        return f"""{base_instr}
Signature:
{signature}

Docstring:
{docstring}

Code:
{code}
""".strip()

    if mode == "one_shot":
        ex = ONE_SHOT_EXAMPLE
        return f"""{base_instr}
Example:
Signature: {ex['signature']}
Docstring: {ex['docstring']}
Code:
{ex['code']}
Description: {ex['description']}

Now do the same for:
Signature:
{signature}

Docstring:
{docstring}

Code:
{code}
""".strip()

    if mode == "two_shot":
        ex1, ex2 = TWO_SHOT_EXAMPLES
        return f"""{base_instr}
Example 1:
Signature: {ex1['signature']}
Docstring: {ex1['docstring']}
Code:
{ex1['code']}
Description: {ex1['description']}

Example 2:
Signature: {ex2['signature']}
Docstring: {ex2['docstring']}
Code:
{ex2['code']}
Description: {ex2['description']}

Now do the same for:
Signature:
{signature}

Docstring:
{docstring}

Code:
{code}
""".strip()

    if mode == "cot_style":
        # "CoT-style" without storing reasoning
        return f"""{base_instr}
Think step-by-step privately to understand the function, but DO NOT output your reasoning.
Output ONLY the final one-paragraph description.

Signature:
{signature}

Docstring:
{docstring}

Code:
{code}
""".strip()

    raise ValueError(f"Unknown prompting mode: {mode}")


# ============================================================
# Temporary in-memory index + BM25 retrieval
# ============================================================

def _tokenize(text: str) -> List[str]:
    text = text.lower()
    toks = [t for t in re.split(r"[^a-z0-9_]+", text) if len(t) >= 2]
    return toks


class TempRepoIndex:
    def __init__(self, repo_root: str):
        self.repo_root = os.path.abspath(repo_root)
        self.entries: List[FunctionEntry] = []
        self._doc_tokens: List[List[str]] = []
        self._df: Dict[str, int] = {}
        self._avgdl: float = 0.0

    def build(self, max_code_chars: int):
        py_files = iter_py_files(self.repo_root)
        pbar_files = tqdm(py_files, desc=f"[AST] files ({os.path.basename(self.repo_root)})", unit="file")
        func_count = 0
        for full_path in py_files:
            rel = os.path.relpath(full_path, self.repo_root)
            funcs = parse_file(full_path, max_code_chars=max_code_chars)
            func_count += len(funcs)
            pbar_files.set_postfix(funcs=func_count)
            for qual, sig, start, end, doc, code in funcs:
                self.entries.append(FunctionEntry(
                    file_path=rel,
                    qualname=qual,
                    signature=sig,
                    start_line=start,
                    end_line=end,
                    docstring=doc,
                    code=code,
                    descriptions={}
                ))
        pbar_files.close()
        tqdm.write(f"[AST] done: {len(py_files)} files, {len(self.entries)} functions")
        self._build_bm25()

    def _build_bm25(self):
        docs = []
        df: Dict[str, int] = {}
        for e in self.entries:
            # Include whatever descriptions exist (all modes) for retrieval
            desc_blob = "\n".join([f"{k}: {v}" for k, v in e.descriptions.items()])
            text = f"{e.file_path}\n{e.qualname}\n{e.signature}\n{e.docstring}\n{desc_blob}\n"
            toks = _tokenize(text)
            docs.append(toks)
            seen = set(toks)
            for t in seen:
                df[t] = df.get(t, 0) + 1

        self._doc_tokens = docs
        self._df = df
        self._avgdl = (sum(len(d) for d in docs) / len(docs)) if docs else 0.0

    def add_description(self, idx: int, mode: str, description: str):
        self.entries[idx].descriptions[mode] = description
        # rebuild BM25 (simple; fine for moderate repos; optimize later if needed)
        self._build_bm25()

    def bm25_search(self, query: str, k: int = 25) -> List[Tuple[int, float]]:
        if not self.entries:
            return []
        q = _tokenize(query)
        if not q:
            return []

        N = len(self.entries)
        k1 = 1.5
        b = 0.75

        scores = [0.0] * N
        for i, doc in enumerate(self._doc_tokens):
            dl = len(doc) or 1
            freqs: Dict[str, int] = {}
            for t in doc:
                freqs[t] = freqs.get(t, 0) + 1

            s = 0.0
            for term in q:
                df = self._df.get(term, 0)
                if df == 0:
                    continue
                idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                denom = tf + k1 * (1 - b + b * (dl / (self._avgdl or 1.0)))
                s += idf * (tf * (k1 + 1)) / denom
            scores[i] = s

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(i, sc) for i, sc in ranked[:k] if sc > 0]


# ============================================================
# Resolver + confidence
# ============================================================

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def retrieval_gap_confidence(scores: List[float], temperature: float = 2.0) -> float:
    """
    scores: descending (top first). If fewer than 2, return moderate.
    """
    if not scores:
        return 0.0
    if len(scores) == 1:
        return 0.55
    s1, s2 = scores[0], scores[1]
    # if scores are close -> low confidence
    return float(sigmoid((s1 - s2) / max(temperature, 1e-6)))


def safe_json(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None


def make_resolver_prompt(user_query: str, candidates: List[dict]) -> str:
    """
    We ask for a confidence score too (self-assessed, 0..1), which works across all models.
    """
    schema = {
        "chosen_index": "number|null",
        "llm_confidence": "number (0..1)",
        "reason": "string",
        "answer": {
            "file_path": "string",
            "start_line": "number",
            "end_line": "number",
            "natural_description": "string"
        }
    }
    return f"""
You are a code repository resolver.

User query:
{user_query}

Pick the single best matching candidate from the list.
If none match, set chosen_index=null and answer=null.

Return STRICT JSON matching this schema (types shown):
{json.dumps(schema, indent=2)}

Guidance:
- llm_confidence should reflect how sure you are (0..1) that the chosen candidate is correct.
- Use the candidate natural_description + signature + file_path.
- Do NOT include any extra keys outside the schema.

Candidates (JSON array):
{json.dumps(candidates, ensure_ascii=False)}
""".strip()


def answer_query(
    index: TempRepoIndex,
    user_query: str,
    resolver_provider: str,
    resolver_model: str,
    k: int = 25,
    prefer_mode: str = "zero_shot"
) -> dict:
    hits = index.bm25_search(user_query, k=k)
    candidates = []
    hit_scores = []
    t_start = time.perf_counter()
    abs_start_time = datetime.now(timezone.utc).isoformat()

    for entry_idx, score in hits:
        e = index.entries[entry_idx]
        desc = e.descriptions.get(prefer_mode) or (next(iter(e.descriptions.values()), "") if e.descriptions else "")
        candidates.append({
            "entry_index": entry_idx,
            "file_path": e.file_path,
            "qualname": e.qualname,
            "start_line": e.start_line,
            "end_line": e.end_line,
            "signature": e.signature,
            "natural_description": desc,
            "retrieval_score": score,
        })
        hit_scores.append(score)

    # If no candidates, return low confidence
    if not candidates:
        return {
            "chosen_index": None,
            "llm_confidence": 0.0,
            "retrieval_confidence": 0.0,
            "confidence": 0.0,
            "reason": "No retrieval candidates found.",
            "answer": None,
            "inference_time_s": time.perf_counter() - t_start,
            "absolute_time_utc_start": abs_start_time,
            "absolute_time_utc_end": datetime.now(timezone.utc).isoformat(),
        }

    # Retrieval-based confidence (model-agnostic)
    rconf = retrieval_gap_confidence(hit_scores, temperature=2.0)

    llm = build_provider(resolver_provider, resolver_model)
    
    prompt = make_resolver_prompt(user_query, candidates)
    raw = llm.generate(prompt)
    obj = safe_json(raw)

    # Fallback to top retrieval
    if not obj or "chosen_index" not in obj:
        c0 = candidates[0]
        return {
            "chosen_index": c0["entry_index"],
            "llm_confidence": 0.5,
            "retrieval_confidence": rconf,
            "confidence": 0.7 * rconf + 0.3 * 0.5,
            "reason": "Resolver output invalid JSON; returned top retrieval match.",
            "answer": {
                "file_path": c0["file_path"],
                "start_line": c0["start_line"],
                "end_line": c0["end_line"],
                "natural_description": c0["natural_description"],
            },
            "inference_time_s": time.perf_counter() - t_start,
            "absolute_time_utc_start": abs_start_time,
            "absolute_time_utc_end": datetime.now(timezone.utc).isoformat(),
        }

    llm_conf = float(obj.get("llm_confidence", 0.5))
    llm_conf = max(0.0, min(1.0, llm_conf))

    # Hybrid confidence (works for ALL models)
    final_conf = 0.7 * rconf + 0.3 * llm_conf

    obj["retrieval_confidence"] = rconf
    obj["confidence"] = final_conf

    obj["inference_time_s"] = time.perf_counter() - t_start
    obj["absolute_time_utc_start"] = abs_start_time
    obj["absolute_time_utc_end"] = datetime.now(timezone.utc).isoformat()
    return obj


# ============================================================
# Description generation (per repo) — optional but recommended
# ============================================================

def generate_descriptions_for_repo(index: TempRepoIndex, provider: str, model: str, modes: List[str], max_functions: Optional[int] = None):
    llm = build_provider(provider, model)
    n = len(index.entries)
    limit = min(n, max_functions) if max_functions else n

    pbar = tqdm(range(limit), desc=f"[DESC] funcs ({provider}:{model})", unit="fn")
    for i in pbar:
        e = index.entries[i]
        for mode in modes:
            if mode in e.descriptions:
                continue
            pbar.set_postfix(mode=mode)
            prompt = make_description_prompt(mode, e.signature, e.docstring, e.code)
            desc = llm.generate(prompt)
            index.add_description(i, mode, desc)

    tqdm.write(f"[DESC] done: described {limit}/{n} functions")


# ============================================================
# Dataset CSV parsing + evaluation
# ============================================================

def parse_range_cell(x: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Accepts formats like:
      "[12,45]"  "(12,45)"  "12,45"  "12 45"  "12-45"
    """
    if x is None:
        return None, None
    s = str(x).strip()
    if not s:
        return None, None
    s = s.strip("[]()")
    # split by comma/space/dash
    parts = re.split(r"[\s,;\-]+", s)
    parts = [p for p in parts if p]
    if len(parts) >= 2:
        try:
            return int(parts[0]), int(parts[1])
        except Exception:
            return None, None
    return None, None


def get_field(row: dict, keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in row and row[k] is not None and str(row[k]).strip() != "":
            return row[k]
    return None


def normalize_path(p: str) -> str:
    return os.path.normpath(p)


def overlap_ratio(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    """
    line overlap ratio: intersection / union
    """
    if any(v is None for v in [a_start, a_end, b_start, b_end]):
        return 0.0
    lo = max(a_start, b_start)
    hi = min(a_end, b_end)
    inter = max(0, hi - lo + 1)
    union = (a_end - a_start + 1) + (b_end - b_start + 1) - inter
    return inter / union if union > 0 else 0.0


def evaluate_dataset(
    dataset_csv: str,
    out_csv: str,
    summarizer_provider: str,
    summarizer_model: str,
    summarizer_modes: List[str],
    resolver_provider: str,
    resolver_model: str,
    k: int = 25,
    prefer_mode: str = "zero_shot",
    max_functions_to_describe: int = 0,
    repo_cache: Optional[Dict[str, TempRepoIndex]] = None,
):
    print("current mode.  ", summarizer_modes, prefer_mode)
    """
    Caches indices per repo_path for speed within one run.
    """
    if repo_cache is None:
        repo_cache = {}
    #repo_cache: Dict[str, TempRepoIndex] = {}

    rows_out: List[dict] = []

    with open(dataset_csv, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        pbar_rows = tqdm(enumerate(rows), total=len(rows), desc="[DATASET] rows", unit="row")

        for idx, row in pbar_rows:
            repo_path = get_field(row, ["repopath", "repo", "path", "repoRoot"])

            filename = get_field(row, ["Filename","filename", "file", "file_path", "filepath"])
            query = get_field(row, ["query", "data(query)", "data", "prompt", "question"])

            startline = get_field(row, ["startline", "start_line", "start"])
            endline = get_field(row, ["endline", "end_line", "end"])

            # If start/end not present, parse the combined range cell:
            if (startline is None or endline is None):
                rng = get_field(row, ["Original Result", "original_result", "range", "start_end"])
                s, e = parse_range_cell(rng)
            else:
                try:
                    s, e = int(startline), int(endline)
                except Exception:
                    s, e = None, None

            if not repo_path or not query:
                continue
            
            repo_path = os.path.abspath(normalize_path(str(repo_path)))
            pbar_rows.set_postfix(step="load_repo", repo=os.path.basename(repo_path))

            gt_file = normalize_path(str(filename)) if filename else None
            print("repo path:",repo_path, repo_cache)
            # Build/load repo index
            # if repo_path not in repo_cache:
            #     CACHE_DIR = "repo_cache_json"  # choose your directory
            #     json_path = repo_json_path(
            #                 CACHE_DIR,
            #                 repo_path,
            #                 summarizer_model,
            #                 summarizer_modes[0]   # exactly one mode per run (as you designed)
            #             )

            #     print("json path is ",json_path)
            #     if os.path.exists(json_path):
            #         idx_obj = load_repo_index_json(json_path)
            #     else:
            #         idx_obj = TempRepoIndex(repo_path)
            #         idx_obj.build(max_code_chars_for(summarizer_model))  # builds AST + entries

            #         # generate only the modes you want (zero/one/two/cot)
            #         generate_descriptions_for_repo(
            #             idx_obj,
            #             provider=summarizer_provider,
            #             model=summarizer_model,
            #             modes=summarizer_modes,
            #             max_functions=max_functions_to_describe,
            #         )

            #         save_repo_index_json(idx_obj, json_path)

            #     repo_cache[repo_path] = idx_obj



            #     # Pre-generate descriptions once per repo (recommended)
            #     # If your repos are big, set max_functions_to_describe to limit cost.
            #     maxf = max_functions_to_describe if max_functions_to_describe > 0 else None
            #     pbar_rows.set_postfix(step="describe", repo=os.path.basename(repo_path))

            #     generate_descriptions_for_repo(
            #         idx_obj,
            #         provider=summarizer_provider,
            #         model=summarizer_model,
            #         modes=summarizer_modes,
            #         max_functions=maxf,
            #     )
            #     repo_cache[repo_path] = idx_obj

            # idx_obj = repo_cache[repo_path]
            mode = summarizer_modes[0]  # you run one mode per eval
            cache_key = (repo_path, summarizer_model, mode)

            if cache_key not in repo_cache:
                CACHE_DIR = "repo_cache_json"
                json_path = repo_json_path(CACHE_DIR, repo_path, summarizer_model, mode)

                if os.path.exists(json_path):
                    idx_obj = load_repo_index_json(json_path)
                else:
                    idx_obj = TempRepoIndex(repo_path)
                    idx_obj.build(max_code_chars=max_code_chars_for(summarizer_model))
                    maxf = max_functions_to_describe if max_functions_to_describe > 0 else None
                    pbar_rows.set_postfix(step="describe", repo=os.path.basename(repo_path))
                    
                    generate_descriptions_for_repo(
                        idx_obj,
                        provider=summarizer_provider,
                        model=summarizer_model,
                        modes=[mode],
                        max_functions=maxf,
                    )

                    save_repo_index_json(idx_obj, json_path)

                repo_cache[cache_key] = idx_obj

            idx_obj = repo_cache[cache_key]

            # Answer query
            pbar_rows.set_postfix(step="resolve", repo=os.path.basename(repo_path))

            result = answer_query(
                idx_obj,
                user_query=str(query),
                resolver_provider=resolver_provider,
                resolver_model=resolver_model,
                k=k,
                prefer_mode=prefer_mode
            )

            # Extract prediction
            pred = result.get("answer") if isinstance(result, dict) else None
            if pred:
                pred_file = normalize_path(str(pred.get("file_path", "")))
                pred_start = pred.get("start_line", None)
                pred_end = pred.get("end_line", None)
                pred_desc = pred.get("natural_description", "")
            else:
                pred_file, pred_start, pred_end, pred_desc = "", None, None, ""

            # Metrics
            file_match = int(bool(gt_file) and (pred_file == gt_file))
            ov = overlap_ratio(s, e, pred_start, pred_end) if (s is not None and e is not None and pred_start and pred_end) else 0.0
            range_ok = int(ov >= 0.3)  # adjustable threshold

            rows_out.append({
                "row_id": idx,
                "repo_path": repo_path,
                "query": query,
                "gt_file": gt_file or "",
                "gt_start": s if s is not None else "",
                "gt_end": e if e is not None else "",
                "pred_file": pred_file,
                "pred_start": pred_start if pred_start is not None else "",
                "pred_end": pred_end if pred_end is not None else "",
                "pred_description": pred_desc,
                "retrieval_confidence": result.get("retrieval_confidence", ""),
                "llm_confidence": result.get("llm_confidence", ""),
                "confidence": result.get("confidence", ""),
                "file_match": file_match,
                "overlap_ratio": ov,
                "range_ok(ov>=0.3)": range_ok,
                "reason": result.get("reason", ""),
                "inference_time": result.get("inference_time_s"),
                "absolute_time_utc_start": result.get("absolute_time_utc_start"),
                "absolute_time_utc_end": result.get("absolute_time_utc_end")

            })
            pbar_rows.set_postfix(step="done", repo=os.path.basename(repo_path))

    # Write output CSV
    fieldnames = list(rows_out[0].keys()) if rows_out else []
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    # Print quick summary
    if rows_out:
        acc_file = sum(r["file_match"] for r in rows_out) / len(rows_out)
        acc_range = sum(r["range_ok(ov>=0.3)"] for r in rows_out) / len(rows_out)
        print(f"Rows: {len(rows_out)}")
        print(f"File match accuracy: {acc_file:.3f}")
        print(f"Range overlap>=0.3 accuracy: {acc_range:.3f}")
        print(f"Wrote: {out_csv}")
    else:
        print("No rows processed (check CSV headers / content).")


# ============================================================
# CLI
# ============================================================
def main():
    import argparse

    p = argparse.ArgumentParser("Evaluate repo function search on dataset.csv (temporary in-memory index)")
    p.add_argument("--dataset_csv", required=True)
    p.add_argument("--out_csv", default="predictions.csv")
    #p.add_argument("--eval_modes", default="",help="Comma list of modes to evaluate separately. Example: zero_shot,one_shot,two_shot,cot_style. If empty, run once.")

    # Summarizer (for function descriptions)
    p.add_argument("--summarizer_provider", default="openai", help="openai|anthropic|gemini|local")
    p.add_argument("--summarizer_model", default="gpt-5")
    p.add_argument("--eval_modes", default="zero_shot", help="comma list: zero_shot,one_shot,two_shot,cot_style")
    p.add_argument("--max_functions_to_describe", type=int, default=0,
                   help="0 = describe all functions; else cap per repo to reduce cost")

    # Resolver (choosing best candidate)
    p.add_argument("--resolver_provider", default="openai", help="openai|anthropic|gemini|local")
    p.add_argument("--resolver_model", default="gpt-5")
    p.add_argument("--k", type=int, default=25)
    p.add_argument("--prefer_mode", default="zero_shot")

    args = p.parse_args()

    eval_modes = [m.strip() for m in args.eval_modes.split(",") if m.strip()]

# If user provided eval_modes, run separate evaluations per mode and write separate CSVs.
    print(f"Evaluation modes: {eval_modes}")
    shared_repo_cache = {}
    if args.eval_modes.strip():
        print("here")
        eval_modes = [m.strip() for m in args.eval_modes.split(",") if m.strip()]
        base_out = args.out_csv.rsplit(".", 1)[0] if "." in args.out_csv else args.out_csv
        
        for m in eval_modes:
            out_csv = f"{base_out}_{m}_{args.resolver_model}.csv"
            print(f"\n=== Evaluating mode: {m} -> {out_csv} ===")

            evaluate_dataset(
                dataset_csv=args.dataset_csv,
                out_csv=out_csv,
                summarizer_provider=args.summarizer_provider,
                summarizer_model=args.summarizer_model,
                summarizer_modes=[m],          # generate ONLY this mode’s descriptions
                resolver_provider=args.resolver_provider,
                resolver_model=args.resolver_model,
                k=args.k,
                prefer_mode=m,                 # resolver uses ONLY this mode’s description
                max_functions_to_describe=args.max_functions_to_describe,
                repo_cache=shared_repo_cache,
            )
    else:
        evaluate_dataset(
            dataset_csv=args.dataset_csv,
            out_csv=args.out_csv,
            summarizer_provider=args.summarizer_provider,
            summarizer_model=args.summarizer_model,
            summarizer_modes=modes,
            resolver_provider=args.resolver_provider,
            resolver_model=args.resolver_model,
            k=args.k,
            prefer_mode=args.prefer_mode,
            max_functions_to_describe=args.max_functions_to_describe,
            repo_cache=shared_repo_cache,

        )



if __name__ == "__main__":
    main()
