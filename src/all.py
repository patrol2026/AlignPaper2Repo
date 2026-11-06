#!/usr/bin/env python3
import argparse, csv, hashlib, json, os, re, time, requests
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import faiss
import pandas as pd
from tqdm import tqdm

# --------------------------- Config ---------------------------
CHUNK_LINES = 120
MAX_CHARS_PER_CHUNK = 8000
MIN_CHARS_PER_CHUNK = 10

ALLOWED_EXTS = {
    ".py",".js",".jsx",".ts",".tsx",".java",".kt",".scala",".go",".rs",
    ".c",".h",".cpp",".hpp",".m",".mm",".rb",".php",".cs",".swift",
    ".sh",".bash",".zsh",".sql",".html",".css",".scss",
    ".yml",".yaml",".toml",".cfg",".ini",".json",".md",".txt",
}
SKIP_DIRS = {"node_modules",".git","build","dist",".next","target",".venv",".mypy_cache","__pycache__"}

MANIFEST_NAME = "manifest.json"

# --------------------------- Data ----------------------------
@dataclass
class ChunkMeta:
    file: str       # absolute path
    rel_file: str   # path relative to repo root
    startLine: int
    endLine: int
    n_lines: int
    id: str
    preview: str    # short head for LLM picking

# --------------------------- IO helpers ----------------------
def canonical_repo_path(p: Path) -> Path:
    return p.expanduser().resolve()

def iter_code_files(root: Path):
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.suffix.lower() in ALLOWED_EXTS:
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

def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms

def write_rows_csv(output_csv: Path, rows: List[Dict[str, Any]], fieldnames: List[str]):
    if output_csv.suffix.lower() != ".csv":
        output_csv = output_csv.with_suffix(".csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames, delimiter=",", quotechar='"',
            quoting=csv.QUOTE_MINIMAL, lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[ok] Wrote CSV: {output_csv.resolve()}")

# --------------------- Embedding backends --------------------
def resolve_embed_choice(embed_backend: Optional[str], llm_backend: str, embed_model: Optional[str]):
    backend = (embed_backend or llm_backend or "ollama").lower()
    if embed_model:
        return backend, embed_model
    if backend == "ollama":
        return "ollama", "mxbai-embed-large"
    if backend == "openai":
        return "openai", "text-embedding-3-small"
    if backend == "openrouter":
        return "openrouter", "openai/text-embedding-3-small"
    if backend == "gemini":
        return "gemini", "models/gemini-embedding-small"
    raise ValueError(f"Unknown embedding backend: {backend}")

def embed_generate(backend: str, model: str, text: str) -> np.ndarray:
    try:
        if backend == "ollama":
            from ollama import Client
            v = Client().embeddings(model=model, prompt=text)["embedding"]
            return np.asarray(v, dtype=np.float32)
        elif backend == "openai":
            from openai import OpenAI
            api = os.environ.get("OPENAI_API_KEY");  
            #assert api, "OPENAI_API_KEY not set"
            #print("API KEY IS",model)
            try:
                c = OpenAI(api_key=api)
                r = c.embeddings.create(model="text-embedding-3-small", input=text)
                print("embeddings",r)
                v = r.data[0].embedding
                return np.asarray(v, dtype=np.float32)
            except Exception as e:
                print("exceptions is",repr(e))

        elif backend == "openrouter":
            api = os.environ.get("OPENROUTER_API_KEY"); assert api, "OPENROUTER_API_KEY not set"
            base = os.environ.get("OPENROUTER_BASE_URL","https://openrouter.ai/api/v1")
            hdr = {
                "Authorization": f"Bearer {api}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://your.app",
                "X-Title": "Repo chooser batch script"
            }
            data = {"model": model, "input": text}
            r = requests.post(f"{base}/embeddings", headers=hdr, data=json.dumps(data), timeout=120)
            r.raise_for_status()
            v = r.json()["data"][0]["embedding"]
            return np.asarray(v, dtype=np.float32)
        elif backend == "gemini":
            import google.generativeai as genai
            api = os.environ.get("GOOGLE_API_KEY"); assert api, "GOOGLE_API_KEY not set"
            genai.configure(api_key=api)
            r = genai.embed_content(model=model, content=text)
            v = r["embedding"]["values"]
            return np.asarray(v, dtype=np.float32)
        else:
            return np.array([], dtype=np.float32)
    except Exception:
        return np.array([], dtype=np.float32)

def _sanitize_chunk(s: str) -> str:
    s = s.strip()
    return s[:MAX_CHARS_PER_CHUNK] if len(s) > MAX_CHARS_PER_CHUNK else s

def embed_texts_dynamic(texts: List[str], embed_backend: str, embed_model: str):
    print(embed_model,embed_backend)
    vecs, kept, dim = [], [], None
    for i, raw in enumerate(tqdm(texts, desc="Embedding", leave=False)):
        t = _sanitize_chunk(raw)
        if len(t) < MIN_CHARS_PER_CHUNK:
            continue
        v = embed_generate(embed_backend, embed_model, t)
        if v.size == 0:
            print("size is zero")
            t2 = t[: max(MIN_CHARS_PER_CHUNK, len(t)//2)]
            v = embed_generate(embed_backend, embed_model, t2)
        if v.size == 0:
            continue
        if dim is None:
            dim = int(v.shape[0])
        if v.shape[0] != dim:
            continue
        vecs.append(v.astype(np.float32))
        kept.append(i)
    if not vecs:
        raise RuntimeError("No valid embeddings produced for this backend/model.")
    return np.vstack(vecs), kept, dim

# ----------------------- LLM backends ------------------------
def llm_generate(backend: str, model: str, prompt: str, temperature: float = 0.1) -> str:
    backend = (backend or "ollama").lower()
    if backend == "ollama":
        from ollama import Client
        out = Client().generate(model=model, prompt=prompt, options={"temperature": temperature}, stream=False)
        return out["response"]
    elif backend == "openai":
        from openai import OpenAI
        api = os.environ.get("OPENAI_API_KEY");  assert api, "OPENAI_API_KEY not set"
        client = OpenAI(api_key=api)
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{"role":"system","content":"You are concise."},
                      {"role":"user","content":prompt}]
        )
        return resp.choices[0].message.content
    elif backend == "gemini":
        import google.generativeai as genai
        api = os.environ.get("GOOGLE_API_KEY"); assert api, "GOOGLE_API_KEY not set"
        genai.configure(api_key=api)
        m = genai.GenerativeModel(model)
        resp = m.generate_content(prompt, generation_config={"temperature": temperature})
        return getattr(resp, "text", "") or ""
    elif backend == "openrouter":
        api = os.environ.get("OPENROUTER_API_KEY"); assert api, "OPENROUTER_API_KEY not set"
        base = os.environ.get("OPENROUTER_BASE_URL","https://openrouter.ai/api/v1")
        headers = {
            "Authorization": f"Bearer {api}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://your.app",
            "X-Title": "Repo chooser batch script",
        }
        data = {
            "model": model,
            "messages": [
                {"role":"system","content":"You are concise."},
                {"role":"user","content":prompt},
            ],
            "temperature": temperature,
        }
        r = requests.post(f"{base}/chat/completions", headers=headers, data=json.dumps(data), timeout=120)
        r.raise_for_status()
        j = r.json()
        return j["choices"][0]["message"]["content"]
    else:
        raise ValueError(f"Unknown LLM backend: {backend}")

# -------------------- Cache / manifest utils -----------------
def cache_slug_for_repo(repo_path: str, embed_backend: str, embed_model: str) -> str:
    root_hash = hashlib.sha1(repo_path.encode("utf-8")).hexdigest()[:12]
    sig_hash  = hashlib.sha1(f"{embed_backend}:{embed_model}".encode("utf-8")).hexdigest()[:8]
    return f"{root_hash}-{sig_hash}"

def compute_manifest(repo_root: Path) -> dict:
    entries = []
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        try:
            st = path.stat()
        except Exception:
            continue
        entries.append({"rel": str(path.relative_to(repo_root)), "size": st.st_size, "mtime": int(st.st_mtime)})
    entries.sort(key=lambda x: x["rel"])
    return {"root": str(repo_root), "entries": entries, "generated_at": int(time.time())}

def manifest_changed(index_dir: Path, manifest: dict, embed_backend: str, embed_model: str, chunk_lines: int) -> bool:
    cfg_path = index_dir / "config.json"
    man_path = index_dir / MANIFEST_NAME
    idx_path = index_dir / "faiss.index"
    meta_path= index_dir / "meta.jsonl"
    if not (idx_path.exists() and meta_path.exists() and cfg_path.exists() and man_path.exists()):
        return True
    try:
        old_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        old_man = json.loads(man_path.read_text(encoding="utf-8"))
    except Exception:
        return True
    if old_cfg.get("embed_backend") != embed_backend: return True
    if old_cfg.get("embed_model")   != embed_model:   return True
    if int(old_cfg.get("chunk_lines", chunk_lines)) != chunk_lines: return True
    old = {e["rel"]: (e["size"], e["mtime"]) for e in old_man.get("entries", [])}
    cur = {e["rel"]: (e["size"], e["mtime"]) for e in manifest.get("entries", [])}
    return old != cur

def save_manifest(index_dir: Path, manifest: dict):
    (index_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

def save_index(index_dir: Path, index: faiss.Index, metas: List[ChunkMeta], dim: int,
               code_root: Path, embed_backend: str, embed_model: str):
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "faiss.index"))
    with (index_dir / "meta.jsonl").open("w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(asdict(m), ensure_ascii=False) + "\n")
    (index_dir / "config.json").write_text(
        json.dumps({
            "dim": dim,
            "chunk_lines": CHUNK_LINES,
            "code_root": str(code_root),
            "embed_backend": embed_backend,
            "embed_model": embed_model
        }, indent=2),
        encoding="utf-8"
    )

def load_index(index_dir: Path):
    idx = faiss.read_index(str(index_dir / "faiss.index"))
    metas: List[ChunkMeta] = []
    for line in (index_dir / "meta.jsonl").read_text(encoding="utf-8").splitlines():
        metas.append(ChunkMeta(**json.loads(line)))
    cfg = json.loads((index_dir / "config.json").read_text(encoding="utf-8"))
    return idx, metas, cfg

# --------------------- Build / Retrieve ----------------------
def build_index_for_repo(repopath: Path, cache_root: Path, embed_backend: str, embed_model: str,
                         force_rebuild: bool=False) -> Path:
    print(str(repopath)+"scjwdkc")
    repo = canonical_repo_path(repopath)
    slug = cache_slug_for_repo(str(repo), embed_backend, embed_model)
    index_dir = cache_root / slug
    index_dir.mkdir(parents=True, exist_ok=True)

    manifest = compute_manifest(repo)
    needs_rebuild = force_rebuild or manifest_changed(index_dir, manifest, embed_backend, embed_model, CHUNK_LINES)

    if not needs_rebuild:
        print(f"[cache] Reusing index for {repo} ({embed_backend}:{embed_model})")
        return index_dir

    print(f"[build] (re)building index for {repo} using {embed_backend}:{embed_model} ...")
    files = list(iter_code_files(repo))
    if not files:
        raise RuntimeError(f"No supported files under: {repo}")

    texts, metas = [], []
    for f in tqdm(files, desc=f"Chunking: {repo.name}", leave=False):
        rel = str(f.relative_to(repo))
        txt = load_text(f)
        if not txt.strip():
            continue
        for (s, e, chunk) in chunk_lines(txt, CHUNK_LINES):
            cid = f"{rel}:{s}-{e}"
            metas.append(ChunkMeta(
                file=str(f), rel_file=rel, startLine=s, endLine=e,
                n_lines=e - s + 1, id=cid, preview="\n".join(chunk.splitlines()[:12])
            ))
            texts.append(chunk)
            #print(chunk)

    if not texts:
        raise RuntimeError(f"No chunks produced for: {repo}")
    #print("this one",embed_backend,embed_model)
    emb, kept, dim = embed_texts_dynamic(texts, embed_backend, embed_model)
    emb = normalize_rows(emb).astype(np.float32)
    metas = [metas[i] for i in kept]

    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    save_index(index_dir, index, metas, dim, repo, embed_backend, embed_model)
    save_manifest(index_dir, manifest)
    print(f"[build] Indexed {len(metas)} chunks for {repo}")
    return index_dir

def retrieve_topk_loaded(idx: faiss.Index, metas: List[ChunkMeta], query_text: str, k: int,
                         embed_backend: str, embed_model: str):
    q = embed_generate(embed_backend, embed_model, query_text)
    if q.size == 0:
        raise RuntimeError("Failed to embed query text.")
    q = normalize_rows(q.astype(np.float32)[None, :])
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

# ---------------- Pick + Summarize (<=200 words) -------------
def llm_choose_and_summarize(query_text: str, candidates: List[Dict[str, Any]],
                             llm_backend: str, llm_model: str, temperature: float=0.1,
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
    resp = llm_generate(llm_backend, llm_model, prompt, temperature)

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
    # ensure ints
    try:
        data["startLine"] = int(data.get("startLine", 0))
        data["endLine"]   = int(data.get("endLine", 0))
    except Exception:
        data["startLine"], data["endLine"] = 0, 0
    data["file"] = str(data.get("file",""))
    return {"file": data["file"], "startLine": data["startLine"], "endLine": data["endLine"], "summary": data.get("summary","").strip()}

# -------------------------- Batch runner ---------------------
def process_csv(input_csv: Path, index_root: Path, output_csv: Path, top_k: int,
                llm_backend: str, llm_model: str, temperature: float,
                embed_backend: Optional[str], embed_model: Optional[str],
                force_rebuild: bool=False) -> None:

    if input_csv.is_dir(): raise SystemExit(f"--input-csv must be a file: {input_csv}")
    if not input_csv.exists(): raise SystemExit(f"--input-csv not found: {input_csv}")
    index_root.mkdir(parents=True, exist_ok=True)
    if output_csv.is_dir():
        output_csv = output_csv / "answers.csv"
    else:
        output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    if not {"data","repopath"}.issubset(df.columns):
        raise SystemExit("Input CSV must contain columns: data, repopath")

    # choose embeddings (default matches LLM backend)
    emb_backend, emb_model = resolve_embed_choice(embed_backend, llm_backend, embed_model)
    print(f"[cfg] LLM = {llm_backend}:{llm_model} | Embedding = {emb_backend}:{emb_model}")

    # Build/load each unique repo once
    #print(df[0:5])
    unique_repos = sorted((str((Path(p)))) for p in df["repopath"].astype(str))
    #print(unique_repos)
    repo_to_index_dir: Dict[str, Optional[Path]] = {}
    t0 = time.perf_counter()
    for repo_str in tqdm(unique_repos, desc="Preparing indices"):
        print(str(repo_str)+"here")
        repo = Path(str(repo_str).strip())
        if not repo.exists() or not repo.is_dir():
            repo_to_index_dir[str(repo_str).strip()] = None
            print(f"[warn] repo not found: {repo}")
            continue
        try:
            idx_dir = build_index_for_repo(repo, index_root, emb_backend, emb_model, force_rebuild=force_rebuild)
            repo_to_index_dir[str(repo_str).strip()] = idx_dir
        except Exception as e:
            repo_to_index_dir[str(repo_str).strip()] = None
            print(f"[WARN] Could not build index for {repo}: {e}")
    t1 = time.perf_counter()
    print(f"[time] Index prep: {t1 - t0:.2f}s")

    # Load each index once into memory
    repo_cache: Dict[str, Optional[Dict[str, Any]]] = {}
    for repo_str, idx_dir in tqdm(repo_to_index_dir.items(), desc="Loading indices"):
        if idx_dir is None:
            repo_cache[str(repo_str).strip()] = None
            continue
        try:
            idx, metas, cfg = load_index(idx_dir)
            repo_cache[str(repo_str).strip()] = {"idx": idx, "metas": metas, "cfg": cfg}
        except Exception as e:
            repo_cache[str(repo_str).strip()] = None
            print(f"[WARN] Failed to load index for {repo_str}: {e}")

    # Answer rows
    out_rows = []
    t2 = time.perf_counter()
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Answering"):
        query_text = str(r["data"])
        repo_str   = str(canonical_repo_path(Path(str(r["repopath"]))))
        print("repo string is ",repo_str)
        state = repo_cache.get(str(repo_str).strip())
        status = "ok"
        result = {"file": None, "startLine": None, "endLine": None, "summary": ""}

        row_start = time.perf_counter()
        if state is None:
            status = "error: index unavailable for repopath"
        else:
            try:
                # use embedding backend/model from cfg to embed the query consistently
                eb, em = state["cfg"]["embed_backend"], state["cfg"]["embed_model"]
                cands = retrieve_topk_loaded(state["idx"], state["metas"], query_text, top_k, eb, em)
                result = llm_choose_and_summarize(query_text, cands, llm_backend, llm_model, temperature=temperature)
            except Exception as e:
                status = f"error: {e}"
        row_dt = time.perf_counter() - row_start

        out_rows.append({
            "data": query_text,
            "repopath": repo_str,
            "file": result.get("file"),
            "startLine": result.get("startLine"),
            "endLine": result.get("endLine"),
            "summary": result.get("summary"),
            "status": status,
            "elapsed_sec": round(row_dt, 3),
        })
    t3 = time.perf_counter()
    print(f"[time] Answering: {t3 - t2:.2f}s  |  Total: {t3 - t0:.2f}s")

    # Write results
    write_rows_csv(output_csv, out_rows,
                   ["data","repopath","file","startLine","endLine","summary","status","elapsed_sec"])

# ----------------------------- CLI ---------------------------
def main():
    p = argparse.ArgumentParser(description="Batch: per-repo embeddings index (reused), LLM picks one best chunk, writes CSV.")
    p.add_argument("--input-csv", type=Path, required=True, help="CSV with columns: data,repopath")
    p.add_argument("--index-root", type=Path, required=True, help="Directory to cache indices")
    p.add_argument("--output-csv", type=Path, required=True, help="Output CSV path (or directory)")

    p.add_argument("--top-k", type=int, default=7)

    # LLM for choosing/summarizing
    p.add_argument("--llm-backend", type=str, default="ollama",
                   choices=["ollama","openai","gemini","openrouter"])
    p.add_argument("--llm-model", type=str, default="llama3.1:8b")

    # Embeddings (optional override; default mirrors LLM backend)
    p.add_argument("--embed-backend", type=str, default=None,
                   choices=["ollama","openai","gemini","openrouter"])
    p.add_argument("--embed-model", type=str, default=None)

    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--force-rebuild", action="store_true")

    args = p.parse_args()
    process_csv(args.input_csv, args.index_root, args.output_csv, args.top_k,
                args.llm_backend, args.llm_model, args.temperature,
                args.embed_backend, args.embed_model, args.force_rebuild)

if __name__ == "__main__":
    main()
