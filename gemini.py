# file: build_matrix_gemini_inprompt.py
import os, glob, csv, time, re
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv
from tqdm import tqdm
import google.generativeai as genai

# ---------- CONFIG ----------
ROOT = "./"             # parent folder with your 20 subfolders
MODEL = "gemini-2.5-pro"       # or "gemini-1.5-flash" for cheaper/faster
RATE_LIMIT_SLEEP = 0.5
MAX_CHARS_TEXT = 12000
MAX_CHARS_CODE = 20000

# ---------- INIT ----------
load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel(
    model_name=MODEL,
    generation_config={
        "temperature": 0.0,
        "top_p": 1.0       # small—only need 1–5 or tiny JSON
        # If your SDK supports it, this enforces JSON:
        # "response_mime_type": "application/json",
        # "response_schema": { "type": "OBJECT", "properties": { "score": { "type": "INTEGER" }}, "required": ["score"] }
    },
)

# ---------- HELPERS ----------
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
            arts.append(
                Artifact(
                    folder=os.path.relpath(dirpath, root),
                    py_path=os.path.join(dirpath, pys[0]),
                    txt_path=os.path.join(dirpath, txts[0]),
                )
            )
    arts.sort(key=lambda a: a.folder)
    if not arts:
        raise SystemExit(f"No dirs under {root} containing both .py and .txt")
    return arts

def _read_trim(path: str, limit: int) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        s = f.read()
    if len(s) > limit:
        h = limit // 2
        s = s[:h] + "\n\n...[truncated]...\n\n" + s[-h:]
    return s

def _parse_score(text: str) -> int:
    # Accept {"score":5}, "5", or "Score: 4"
    m = re.search(r'"score"\s*:\s*([1-5])', text)
    if not m:
        m = re.search(r'\b([1-5])\b', text)
    if not m:
        raise RuntimeError(f"Could not parse a 1–5 score from: {text[:200]}")
    return int(m.group(1))

INSTRUCTIONS = (
    "Task: Output ONE integer 1..5 indicating how strongly the Python file "
    "implements the methodology (1=unrelated, 2=weak, 3=mixed, 4=good, 5=strong). "
    "Return JUST the number or JSON like {\"score\": <int>}."
)

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
    r = model.generate_content(prompt)
    txt = (r.text or "").strip()
    return _parse_score(txt)

def save_csv(row_labels, col_labels, M, out_path="matrix_gemini.csv"):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["python\\paper"] + col_labels)
        for rname, row in zip(row_labels, M):
            w.writerow([rname] + row)
    print(f"Saved {out_path}")

def main():
    arts = collect_artifacts(ROOT)
    rows = [a.folder for a in arts]
    cols = [a.folder for a in arts]
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
