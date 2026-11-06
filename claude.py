# build_matrix_claude.py
import os, csv, time, re
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv
from tqdm import tqdm
from anthropic import Anthropic, APIStatusError

# ------------ CONFIG ------------
ROOT = "./"                 # parent folder with your 20 subfolders
MODEL = "claude-sonnet-4-5" # strong reasoning, 200k ctx
RATE_LIMIT_SLEEP = 0.5
MAX_CHARS_TEXT = 12000             # trim big paper texts
MAX_CHARS_CODE = 12000             # trim big code files
MAX_TOKENS_OUT = 32                # we only need a single digit

SYSTEM_MSG = (
    "You are an expert research/code auditor. "
    "Given a methodology and a Python file, output ONE integer 1..5 indicating "
    "how strongly the code implements that methodology (1=unrelated, 2=weak, "
    "3=mixed, 4=good, 5=strong). DO NOT EXPLAIN. MUST Return ONLY the number (or JSON {\"score\": <int>})."
)

INSTRUCTIONS = (
    "Task: Rate how strongly the Python file implements the methodology.\n"
    "Scale: 1=unrelated, 2=weak, 3=mixed, 4=good, 5=strong.\n"
    "DO NOT EXPLAIN. MUST Return ONLY one integer 1..5, or JSON like {\"score\": <int>}."
)

# ----------- INIT -----------
load_dotenv()
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# ----------- HELPERS -----------
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
        raise RuntimeError(f"Could not parse 1–5 from: {text[:200]}")
    return int(m.group(1))

def score_pair_inprompt(code_path: str, paper_path: str) -> int:
    paper = _read_trim(paper_path, MAX_CHARS_TEXT)
    code  = _read_trim(code_path,  MAX_CHARS_CODE)

    # Messages API: send a short system + two labeled text parts
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS_OUT,
            temperature=0.0,
            system=SYSTEM_MSG,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": INSTRUCTIONS},
                    {"type": "text", "text": "(A) Methodology (paper):\n" + paper},
                    {"type": "text", "text": "(B) Python file:\n" + code},
                ],
            }],
        )
        # Claude returns a list of content blocks; we concatenates any text blocks
        parts = []
        for block in resp.content:
            if block.type == "text":
                parts.append(block.text)
        text = "\n".join(parts).strip()
        return _parse_score(text)

    except APIStatusError as e:
        # Quota or rate errors -> simple rethrow with context
        raise RuntimeError(f"Claude API error: {e.status_code} {e.message}") from e

def save_csv(row_labels: List[str], col_labels: List[str], M: List[List[int]], out_path: str = "claude_matrix.csv"):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["python\\paper"] + col_labels)
        for rname, row in zip(row_labels, M):
            w.writerow([rname] + row)
    print(f"Saved {out_path}")

def main():
    arts = collect_artifacts(ROOT)
    rows = [a.folder for a in arts]  # python files (rows)
    cols = [a.folder for a in arts]  # papers (cols)
    M = [[0 for _ in cols] for _ in rows]

    print(f"Scoring {len(rows)}×{len(cols)} pairs with {MODEL} ...")
    for i, a_code in enumerate(tqdm(arts, desc="Rows (code)")):
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
