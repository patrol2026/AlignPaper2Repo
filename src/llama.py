#!/usr/bin/env python3
import argparse, csv, hashlib, json, re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Any
import time
import numpy as np
import faiss
import pandas as pd
from tqdm import tqdm

# ---------- Config ----------
CHUNK_LINES = 120
EMBED_MODEL = "mxbai-embed-large"
ALLOWED_EXTS = {
    ".py"
}

try:
    from ollama import Client
except Exception:
    print("Missing 'ollama' package. Install with: pip install ollama")
    raise

# ---------- Data ----------
@dataclass
class ChunkMeta:
    file: str      # absolute path on disk
    rel_file: str  # relative to code_root (reported)
    startLine: int
    endLine: int
    n_lines: int
    id: str
    preview: str   # short excerpt for picker prompt

# ---------- Helpers ----------
def get_client() -> Client:
    return Client()  # respects OLLAMA_HOST if set

def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms

def iter_code_files(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
            nm = p.name.lower()
            if any(b in nm for b in [".lock",".min.",".map",".png",".jpg",".jpeg",".gif",".pdf",".svg"]):
                continue
            yield p

def load_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return path.read_bytes().decode("utf-8", errors="ignore")

def chunk_lines(text: str, lines_per_chunk=CHUNK_LINES) -> List[Tuple[int,int,str]]:
    lines = text.splitlines()
    out = []
    for i in range(0, len(lines), lines_per_chunk):
        s = i + 1
        e = min(i + lines_per_chunk, len(lines))
        out.append((s, e, "\n".join(lines[i:e])))
    return out

def embed_texts(client: Client, texts: List[str]) -> np.ndarray:
    vecs = []
    for t in tqdm(texts, desc="Embedding", leave=False):
        r = client.embeddings(model=EMBED_MODEL, prompt=t)
        vecs.append(np.array(r["embedding"], dtype=np.float32))
    return np.vstack(vecs)

def save_index(index_dir: Path, index: faiss.Index, metas: List[ChunkMeta], dim: int, code_root: Path):
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "faiss.index"))
    with (index_dir / "meta.jsonl").open("w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(asdict(m), ensure_ascii=False) + "\n")
    (index_dir / "config.json").write_text(
        json.dumps({"dim": dim, "embed_model": EMBED_MODEL, "code_root": str(code_root)}, indent=2),
        encoding="utf-8"
    )

def load_index(index_dir: Path):
    idx = faiss.read_index(str(index_dir / "faiss.index"))
    metas: List[ChunkMeta] = []
    for line in (index_dir / "meta.jsonl").read_text(encoding="utf-8").splitlines():
        metas.append(ChunkMeta(**json.loads(line)))
    cfg = json.loads((index_dir / "config.json").read_text(encoding="utf-8"))
    return idx, metas, cfg

# ---------- Build / Retrieve ----------
def build_index_for_repo(repopath: Path, cache_root: Path, force_rebuild: bool=False) -> Path:
    """
    Build (or reuse) an index for a given repo path.
    Returns the index directory path.
    """
    abs_repo = repopath.resolve()
    slug = hashlib.sha1(str(abs_repo).encode("utf-8")).hexdigest()[:12]
    index_dir = cache_root / f"{slug}"
    index_dir.mkdir(parents=True, exist_ok=True)

    index_path = index_dir / "faiss.index"
    meta_path  = index_dir / "meta.jsonl"

    if index_path.exists() and meta_path.exists() and not force_rebuild:
        return index_dir

    client = get_client()
    files = list(iter_code_files(abs_repo))
    if not files:
        raise RuntimeError(f"No supported files under: {abs_repo}")

    texts, metas = [], []
    for f in tqdm(files, desc=f"Chunking: {repopath.name}", leave=False):
        rel = str(f.relative_to(abs_repo))
        txt = load_text(f)
        if not txt.strip():
            continue
        for (s, e, chunk) in chunk_lines(txt, CHUNK_LINES):
            cid = f"{rel}:{s}-{e}"
            metas.append(ChunkMeta(
                file=str(f), rel_file=rel, startLine=s, endLine=e,
                n_lines=e - s + 1, id=cid,
                preview="\n".join(chunk.splitlines()[:12])
            ))
            texts.append(chunk)

    if not texts:
        raise RuntimeError(f"No chunks produced for: {abs_repo}")

    emb = embed_texts(client, texts)
    dim = emb.shape[1]
    emb = normalize_rows(emb).astype(np.float32)
    index = faiss.IndexFlatIP(dim)  # cosine via dot on normalized vectors
    index.add(emb)
    save_index(index_dir, index, metas, dim, abs_repo)
    return index_dir

def retrieve_topk_loaded(idx: faiss.Index, metas: List[ChunkMeta], query_text: str, k: int, client: Client):
    qvec = client.embeddings(model=EMBED_MODEL, prompt=query_text)["embedding"]
    q = normalize_rows(np.array(qvec, dtype=np.float32)[None, :])
    scores, ids = idx.search(q, k)
    scores, ids = scores[0].tolist(), ids[0].tolist()

    items = []
    for sc, i in zip(scores, ids):
        if i < 0: continue
        m = metas[i]
        items.append({
            "file": m.rel_file,
            "startLine": m.startLine,
            "endLine": m.endLine,
            "score": float(sc),
            "preview": m.preview
        })
    return items
import os, json, requests

def llm_generate(backend: str, model: str, prompt: str, temperature: float = 0.1) -> str:
    """
    Generate a single text completion across multiple backends.
    Supported backends:
      - 'ollama'     -> local Ollama
      - 'openai'     -> OpenAI Chat Completions API
      - 'gemini'     -> Google Generative AI (gemini-1.5*)
      - 'openrouter' -> OpenRouter (Claude, GPT, etc.)
    Returns raw text string.
    """
    backend = (backend or "ollama").lower()

    if backend == "ollama":
        from ollama import Client
        client = Client()
        out = client.generate(model=model, prompt=prompt, options={"temperature": temperature}, stream=False)
        return out["response"]

    elif backend == "openai":
        # pip install openai>=1.47.0
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        client = OpenAI(api_key=api_key)
        # chat-completions style
        resp = client.chat.completions.create(
            model=model,  # e.g., "gpt-4o-mini" or "gpt-4.1-mini"
            temperature=temperature,
            messages=[
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content

    elif backend == "gemini":
        # pip install google-generativeai
        import google.generativeai as genai
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set")
        genai.configure(api_key=api_key)
        # e.g., model="gemini-1.5-flash" or "gemini-1.5-pro"
        model_obj = genai.GenerativeModel(model)
        resp = model_obj.generate_content(
            prompt,
            generation_config={"temperature": temperature},
        )
        # Google SDK sometimes returns a list of parts
        return getattr(resp, "text", None) or "".join(getattr(resp, "candidates", []) or [])

    elif backend == "openrouter":
        # pip install requests
        api_key = os.environ.get("OPENROUTER_API_KEY")
        base = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # Optional, but recommended:
            "HTTP-Referer": "https://your.app",           # your site/app
            "X-Title": "Repo chooser batch script",       # app name
        }
        # OpenRouter uses the OpenAI-compatible schema
        data = {
            "model": model,  # e.g., "anthropic/claude-3.5-sonnet" or "openai/gpt-4o-mini"
            "messages": [
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
        }
        r = requests.post(f"{base}/chat/completions", headers=headers, data=json.dumps(data), timeout=120)
        r.raise_for_status()
        j = r.json()
        return j["choices"][0]["message"]["content"]

    else:
        raise ValueError(f"Unknown backend: {backend}")

def llm_choose_and_summarize(query_text: str, candidates: List[Dict[str, Any]], model: str, backend:str="ollama",
                             temperature: float=0.1,
                             max_preview_chars: int=900, max_words: int=200) -> Dict[str, Any]:
    if not candidates:
        return {"file": None, "startLine": None, "endLine": None, "summary": "No candidates found."}

    lines = [f"Question: {query_text}", "", "Candidates:"]
    for i, c in enumerate(candidates, 1):
        prev = c["preview"]
        if len(prev) > max_preview_chars:
            prev = prev[:max_preview_chars] + "…"
        lines.append(f"{i}) {c['file']}:{c['startLine']}-{c['endLine']}\n{prev}")

    lines.append("""
Task:
- Pick the ONE candidate that best answers the question.
- Then write a concise summary (<= 200 words) explaining why that snippet is relevant.
- Respond ONLY as JSON with keys: file, startLine, endLine, summary.
""")
    prompt = "\n".join(lines)

    resp = llm_generate(backend, model, prompt, temperature)

    m = re.search(r"\{.*\}", resp, re.S)
    if not m:
        best = max(candidates, key=lambda x: x["score"])
        return {"file": best["file"], "startLine": best["startLine"], "endLine": best["endLine"],
                "summary": "(auto) Selected by score; LLM didn't return JSON."}

    try:
        data = json.loads(m.group(0))
    except Exception:
        best = max(candidates, key=lambda x: x["score"])
        return {"file": best["file"], "startLine": best["startLine"], "endLine": best["endLine"],
                "summary": "(auto) Selected by score; JSON parse failed."}

    words = str(data.get("summary","")).split()
    if len(words) > max_words:
        data["summary"] = " ".join(words[:max_words])
    # Ensure ints & strings
    try:
        data["startLine"] = int(data.get("startLine", 0))
        data["endLine"]   = int(data.get("endLine", 0))
    except Exception:
        data["startLine"], data["endLine"] = 0, 0
    data["file"] = str(data.get("file",""))
    return {
        "file": data["file"],
        "startLine": data["startLine"],
        "endLine": data["endLine"],
        "summary": data.get("summary","").strip()
    }

# ---------- Batch runner (dedupe by repopath) ----------
def process_csv(input_csv: Path, index_root: Path, output_csv: Path, top_k: int, model: str,
                temperature: float, force_rebuild: bool=False):
    df = pd.read_csv(input_csv)
    if not {"data","repopath"}.issubset(df.columns):
        raise SystemExit("Input CSV must contain columns: data, repopath")

    # Build/load each unique repo ONCE
    unique_repos = [Path(str(p)).expanduser() for p in sorted(df["repopath"].astype(str).unique())]
    repo_to_index_dir: Dict[str, Path] = {}
    for repo in tqdm(unique_repos, desc="Preparing indices"):
        #print(repo.strip())
        repo = Path(str(repo).strip())
        print(repo)
        if not repo.exists() or not repo.is_dir():
            #print('repo does not exist')
            repo_to_index_dir[str(repo)] = None
            continue
        try:
            idx_dir = build_index_for_repo(repo, index_root, force_rebuild=force_rebuild)
            repo_to_index_dir[str(repo)] = idx_dir
        except Exception as e:
            repo_to_index_dir[str(repo)] = None
            print(f"[WARN] Could not build index for {repo}: {e}")

    # Load each index once & keep in memory
    repo_cache: Dict[str, Dict[str, Any]] = {}
    for repo, idx_dir in tqdm(repo_to_index_dir.items(), desc="Loading indices"):
        print(repo_to_index_dir)
        if idx_dir is None:
            print('here1')
            repo_cache[repo] = None
        else:
            try:
                print('here2')

                idx, metas, cfg = load_index(idx_dir)
                repo_cache[repo] = {"idx": idx, "metas": metas, "cfg": cfg}
            except Exception as e:
                print('here3')
                repo_cache[repo] = None
                print(f"[WARN] Failed to load index for {repo}: {e}")

    # Process each row using in-memory index
    client = get_client()
    out_rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Answering"):
        query_text = str(r["data"])            # <-- use 'data' column as the question
        repo      = str(Path(str(r["repopath"].strip())).expanduser())  # <-- use 'repopath' column
        state = repo_cache.get(repo)
        status = "ok"
        result = {"file": None, "startLine": None, "endLine": None, "summary": ""}

        if state is None:
            status = "error: index unavailable for repopath"
        else:
            try:
                cands = retrieve_topk_loaded(state["idx"], state["metas"], query_text, top_k, client)
                result = llm_choose_and_summarize(query_text, cands, model=model, temperature=temperature)
            except Exception as e:
                status = f"error: {e}"

        out_rows.append({
            "data": query_text,
            "repopath": repo,
            "file": result.get("file"),
            "startLine": result.get("startLine"),
            "endLine": result.get("endLine"),
            "summary": result.get("summary"),
            "status": status
        })

    # Write results CSV (overwrite)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["data","repopath","file","startLine","endLine","summary","status"])
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)

# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser(description="Batch (dedup repopath): build each repo once, reuse for many data rows, write CSV.")
    p.add_argument("--input-csv", type=Path, required=True, help="CSV with columns: data, repopath")
    p.add_argument("--index-root", type=Path, required=True, help="Directory to cache per-repo indices")
    p.add_argument("--output-csv", type=Path, required=True, help="Where to write results CSV")
    p.add_argument("--top-k", type=int, default=7)
    p.add_argument("--llm-backend", type=str, default="ollama",
               choices=["ollama","openai","gemini","openrouter"],
               help="LLM backend for choosing/summarizing")
    p.add_argument("--model", type=str, default="llama3.1:8b")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--force-rebuild", action="store_true", help="Rebuild indices even if cache exists")
    args = p.parse_args()

    process_csv(args.input_csv, args.index_root, args.output_csv,
                top_k=args.top_k, model=args.model, temperature=args.temperature,
                force_rebuild=args.force_rebuild)

if __name__ == "__main__":
    time1=time.perf_counter()

    main()
    time2=time.perf_counter()
    elapsed=time2-time1
    with open("runtime_log.txt", "w") as f:
        f.write(f"Total runtime: {elapsed:.2f} seconds\n")
    print("Time written to runtime_log.txt")
    
