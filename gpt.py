# file: build_code_paper_matrix.py
import os
import json
import time
import glob
from typing import Dict, List, Tuple
from dataclasses import dataclass
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()

# OpenAI Responses API (GPT-5)
# Docs: Responses API + JSON schema outputs
# - Migrate/usage: https://platform.openai.com/docs/guides/migrate-to-responses
# - Models overview: https://openai.com/index/introducing-gpt-5/
from openai import OpenAI
client = OpenAI()

# -------------------------------
# CONFIG
# -------------------------------
ROOT = "./"        # <-- folder that contains 20 subfolders
MODEL = "gpt-5"         # you can also try "gpt-5.1-mini" for lower cost
MAX_CHARS_CODE = 12000     # trim safety for very large files
MAX_CHARS_TEXT = 20000     # trim safety for very large texts
RATE_LIMIT_SLEEP = 0.8     # seconds between calls; adjust as needed

# If your subfolders might have more files, you can tighten these patterns:
PY_GLOB = "*.py"
TXT_GLOB = "*.txt"

# -------------------------------
# HELPERS
# -------------------------------

@dataclass
class Artifact:
    folder: str
    py_path: str
    txt_path: str
    py_name: str
    txt_name: str

def collect_artifacts(root: str) -> List[Artifact]:
    artifacts: List[Artifact] = []
    for folder in sorted(next(os.walk(root))[1]):
        folder_path = os.path.join(root, folder)
        py_files = glob.glob(os.path.join(folder_path, PY_GLOB))
        txt_files = glob.glob(os.path.join(folder_path, TXT_GLOB))
        if not py_files or not txt_files:
            # Skip incomplete folders gracefully
            continue
        # Take the first matching .py / .txt in each subfolder
        py_path = sorted(py_files)[0]
        txt_path = sorted(txt_files)[0]
        artifacts.append(
            Artifact(
                folder=folder,
                py_path=py_path,
                txt_path=txt_path,
                py_name=os.path.basename(py_path),
                txt_name=os.path.basename(txt_path),
            )
        )
    return artifacts

def slurp(path: str, max_chars: int) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        data = f.read()
    if len(data) > max_chars:
        head = data[: max_chars // 2]
        tail = data[-max_chars // 2 :]
        data = head + "\n\n...[truncated]...\n\n" + tail
    return data

def score_pair(code_text: str, paper_text: str) -> int:
    """
    Asks GPT-5 for a Likert-style association score 1..5.
    JSON schema enforcement guarantees we only get an integer in that range.
    """
    system = (
        "You are an expert research/code auditor. "
        "Given (A) a paper’s methodology and (B) a Python source file, "
        "judge how strongly the code appears to implement the specific method described."
    )

    user = f"""
Task: Produce a SINGLE integer score from 1 to 5 indicating how strongly the Python file (B) implements the specific methodology in (A).

Scoring rubric:
1 = Clearly unrelated. Different task/domain or incompatible steps/terminology.
2 = Weak relation. Some topical overlap, but key algorithmic steps do not match.
3 = Uncertain/mixed. Partial overlap (same area) but several mismatches or missing core steps.
4 = Good match. Most steps/terms align; appears to be an implementation with minor gaps.
5 = Strong match. Clear evidence this code implements the exact method (terminology, variable names, pipeline, loss/objectives, data flow, and evaluation closely align).

(A) Methodology (paper):
-----------------------
{paper_text}

(B) Python file:
----------------
{code_text}

Important:
- Return ONLY the JSON object conforming to the provided schema.
- Consider algorithmic steps, function/variable naming, preprocessing/evaluation, and domain-specific signals.
"""

    # JSON schema to force {"score": int between 1 and 5}
    response = client.responses.create(
        model=MODEL,
        reasoning={"effort": "medium"},  # optional; works on GPT-5 reasoning models
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        text={
            "format":{
                "type": "json_schema",
                "name": "likert_match_score",
                "schema": {
                        "type": "object",
                        "properties": {"score": {"type": "integer", "minimum": 1, "maximum": 5}},
                        "required": ["score"],
                        "additionalProperties": False,
                    },
            }
        }
    )

    # Parse guaranteed JSON:
    #print(response)
    data = json.loads(response.output[1].content[0].text)
    #print(data)
    #print(data['score'])
    return data['score']

def build_matrix(artifacts: List[Artifact]) -> Tuple[List[str], List[str], List[List[int]]]:
    """
    Rows = python files (by folder label), Columns = papers (by folder label).
    """
    row_labels = [a.folder for a in artifacts]
    col_labels = [a.folder for a in artifacts]
    matrix = [[0 for _ in col_labels] for _ in row_labels]

    # Preload contents to avoid re-reading in loop
    py_texts = [slurp(a.py_path, MAX_CHARS_CODE) for a in artifacts]
    paper_texts = [slurp(a.txt_path, MAX_CHARS_TEXT) for a in artifacts]

    # Pairwise scoring
    for i, (row_art, code_text) in enumerate(zip(artifacts, py_texts)):
        for j, (col_art, paper_text) in enumerate(zip(artifacts, paper_texts)):
            # Call model
            try:
                score = score_pair(code_text, paper_text)
            except Exception as e:
                # If any call fails, write 0 so you can re-run just those later if needed
                print(f"\nError on row={row_art.folder}, col={col_art.folder}: {e}")
                score = 0
            matrix[i][j] = score
            time.sleep(RATE_LIMIT_SLEEP)  # gentle pacing for rate limits
    return row_labels, col_labels, matrix

def save_csv(row_labels: List[str], col_labels: List[str], matrix: List[List[int]], out_path: str = "matrix.csv"):
    import csv
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["python\\paper"] + col_labels)
        for r_name, row in zip(row_labels, matrix):
            writer.writerow([r_name] + row)
    print(f"Saved {out_path}")

def main():
    artifacts = collect_artifacts(ROOT)
    if len(artifacts) == 0:
        raise SystemExit(f"No subfolders with .py and .txt found under: {ROOT}")
    if len(artifacts) != 20:
        print(f"⚠️ Found {len(artifacts)} folders. The script still works, but you expected 20.")

    print(f"Scoring {len(artifacts)}×{len(artifacts)} pairs with {MODEL} ...")
    rows, cols, M = build_matrix(artifacts)
    save_csv(rows, cols, M, out_path="matrix.csv")

if __name__ == "__main__":
    main()
