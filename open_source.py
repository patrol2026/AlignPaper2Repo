# file: build_matrix_llama31_ollama.py
import os, re, csv, time, glob, json
from dataclasses import dataclass
from typing import List, Tuple
import requests

# ---------------- CONFIG ----------------
ROOT = "./"                 # parent folder containing your 20 dirs
MODEL = "qwen3:8b"              # Ollama model tag
RATE_LIMIT_SLEEP = 0.25
MAX_CHARS_TEXT = 12000
MAX_CHARS_CODE = 10000
OLLAMA_URL = "http://localhost:11434/api/generate"

INSTRUCTIONS = (
    "Task: Output ONE integer 1..5 indicating how strongly the Python file implements the methodology.\n"
    "Scale: 1=unrelated, 2=weak, 3=mixed, 4=good, 5=strong.\n"
    "DO NOT EXPLAIN ANYTHING.MUST Return JUST the number, or JSON like {\"score\": <int>}."
)

SYSTEM_MSG = (
    "You are an expert research/code auditor. Judge how strongly the Python file "
    "implements the specific methodology.DO NOT EXPLAIN ANYTHING. MUST Return ONE integer 1..5, or JSON {\"score\": <int>}."
)

# ------------- HELPERS ------------------
@dataclass
class Artifact:
    folder: str
    py_path: str
    txt_path: str

def collect_artifacts(root: str) -> List[Artifact]:
    if not os.path.isdir(root):
        raise FileNotFoundError(f"ROOT not found: {root}")
    arts: List[Artifact] = []
    for dirpath, _, files in os.walk(root):
        pys = sorted([f for f in files if f.endswith(".py")])
        txts = sorted([f for f in files if f.endswith(".txt")])
        if pys and txts:
            arts.append(Artifact(
                folder=os.path.relpath(dirpath, root),
                py_path=os.path.join(dirpath, pys[0]),
                txt_path=os.path.join(dirpath, txts[0]),
            ))
    arts.sort(key=lambda a: a.folder)
    if not arts:
        raise SystemExit(f"No dirs under {root} with both .py and .txt")
    return arts

def _read_trim(path: str, limit: int) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        s = f.read()
    if len(s) > limit:
        h = limit // 2
        s = s[:h] + "\n\n...[truncated]...\n\n" + s[-h:]
    return s

def _parse_score(text: str) -> int:
    print(text)
    m = re.search(r'"score"\s*:\s*([1-5])', text)
    if not m:
        m = re.search(r'\b([1-5])\b', text)
    if not m:
        raise RuntimeError(f"Could not parse 1–5 from: {text[:200]}")
    return int(m.group(1))

def llama_generate(prompt: str, system: str = SYSTEM_MSG, *, temperature: float = 0.0, num_predict: int = 64) -> str:
    """
    Calls Ollama /api/generate once (non-stream), returns the full response 'response' string.
    """
    payload = {
        "model": MODEL,
        "prompt": f"<<SYS>>\n{system}\n<</SYS>>\n{prompt}",
        "stream": False,
        "options": {
            "num_ctx":16384,
            "temperature": temperature
        },
        
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "").strip()

def score_pair_inprompt(code_path: str, paper_path: str) -> int:
    paper = _read_trim(paper_path, MAX_CHARS_TEXT)
    code  = _read_trim(code_path,  MAX_CHARS_CODE)
    prompt = f"""{INSTRUCTIONS}

(A) Methodology (paper):
-----------------------
{paper}

(B) Python file:
----------------
{code}
"""
    txt = llama_generate(prompt, num_predict=64)
    return _parse_score(txt)

def save_csv(row_labels: List[str], col_labels: List[str], M: List[List[int]], out_path: str = "matrix.csv"):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["python\\paper"] + col_labels)
        for rname, row in zip(row_labels, M):
            w.writerow([rname] + row)
    print(f"Saved {out_path}")

def main():
    arts = collect_artifacts(ROOT)
    rows = [a.folder for a in arts]   # python files by folder
    cols = [a.folder for a in arts]   # papers by folder
    M = [[0 for _ in cols] for _ in rows]

    for i, a_code in enumerate(arts):
        for j, a_paper in enumerate(arts):
            
            try:
                score = score_pair_inprompt(a_code.py_path, a_paper.txt_path)
            except Exception as e:
                print(f"Error (row={a_code.folder}, col={a_paper.folder}): {e}")
                score = 0
            M[i][j] = score
            time.sleep(RATE_LIMIT_SLEEP)

    save_csv(rows, cols, M)

if __name__ == "__main__":
    main()
