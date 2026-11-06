# file: build_matrix_with_attachments.py
import os, glob, time, csv, json
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI

load_dotenv()
client = OpenAI()

ROOT = "./"          # parent folder with 20 subfolders
MODEL = "gpt-5.1-mini"           # try "gpt-5.1-mini" if you want cheaper
RATE_LIMIT_SLEEP = 0.6
PY_GLOB = "*.py"
TXT_GLOB = "*.txt"

@dataclass
class Artifact:
    folder: str
    py_path: str
    txt_path: str
    py_file_id: str = ""
    txt_file_id: str = ""

def collect_artifacts(root: str) -> List[Artifact]:
    subs = sorted(next(os.walk(root))[1])
    arts: List[Artifact] = []
    for sub in subs:
        base = os.path.join(root, sub)
        py = sorted(glob.glob(os.path.join(base, PY_GLOB)))
        tx = sorted(glob.glob(os.path.join(base, TXT_GLOB)))
        if not py or not tx:
            continue
        arts.append(Artifact(folder=sub, py_path=py[0], txt_path=tx[0]))
    return arts

def upload_file(path: str, purpose: str = "assistants") -> str:
    with open(path, "rb") as f:
        up = client.files.create(file=f, purpose=purpose)
    return up.id
def _upload_for_file_search(path: str, purpose: str = "assistants") -> str:
    with open(path, "rb") as f:
        return client.files.create(file=f, purpose=purpose).id
def _parse_score(text: str) -> int:
    # accept {"score":5}, "5", or prose like "Score: 4"
    m = re.search(r'"score"\s*:\s*(\d)', text)
    if not m:
        m = re.search(r'\b([1-5])\b', text)
    if not m:
        raise RuntimeError(f"Could not parse score from: {text[:200]}")
    score = int(m.group(1))
    if not (1 <= score <= 5):
        raise ValueError(f"Score out of range: {score}")
    return score
def score_pair(code_path: str, paper_path: str, model: str = "gpt-5.1-mini") -> int:
    """
    Responses API using 'input_file' parts inside the user message.
    NO 'attachments' kwarg anywhere.
    """
    system_msg = (
        "You are an expert research/code auditor. Use only the provided files to judge "
        "how strongly the Python file implements the specific methodology. "
        "Return ONLY JSON per schema."
    )

    user_instructions = """
Task: Output a SINGLE integer score from 1 to 5 indicating how strongly the Python file implements the specific methodology.

Scoring:
1 = Clearly unrelated.
2 = Weak relation.
3 = Uncertain/mixed (partial overlap).
4 = Good match (most steps/terms align).
5 = Strong match (clear evidence of implementing the exact method).

Return ONLY JSON per schema.
"""

    paper_id = _upload_for_file_search(paper_path)
    code_id  = _upload_for_file_search(code_path)
    try:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_msg},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_instructions},
                        {"type": "input_file", "file_id": paper_id},  # paper (.txt)
                        {"type": "input_file", "file_id": code_id},   # code (.py)
                    ],
                },
            ],
            tools=[{"type": "file_search"}],
            text={
                "format":{
                    "type": "json_schema",
                    "json_schema": {
                        "name": "likert_match_score",
                        "schema": {
                            "type": "object",
                            "properties": {"score": {"type": "integer", "minimum": 1, "maximum": 5}},
                            "required": ["score"],
                            "additionalProperties": False,
                        },
                        "strict": True,
                    }
                }
            },
            temperature=0.1,
            max_output_tokens=50,
        )
        text = resp.output_text if hasattr(resp, "output_text") else resp.output[0].content[0].text
        return _parse_score(text)
    except TypeError:
            # super-old signature mismatch → fall through to chat fallback
            pass
    

    data = json.loads(resp.output[0].content[0].text)
    return int(data["score"])

def save_csv(row_labels: List[str], col_labels: List[str], matrix: List[List[int]], out_path: str):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["python\\paper"] + col_labels)
        for rname, row in zip(row_labels, matrix):
            w.writerow([rname] + row)

def main():
    arts = collect_artifacts(ROOT)
    if not arts:
        raise SystemExit(f"No subfolders with .py and .txt under {ROOT}")

    print(f"Found {len(arts)} folders.")

    # Upload all files once
    print("Uploading files ...")
    for a in tqdm(arts):
        a.py_file_id = upload_file(a.py_path)
        a.txt_file_id = upload_file(a.txt_path)

    # Build labels: rows = python (by folder), cols = papers (by folder)
    row_labels = [a.folder for a in arts]
    col_labels = [a.folder for a in arts]
    M = [[0 for _ in col_labels] for _ in row_labels]

    print("Scoring pairs ...")
    for i, a_code in enumerate(arts):
        for j, a_paper in enumerate(arts):
            try:
                score = score_pair(
                    code_path=a_code.py_path,
                    paper_path=a_paper.txt_path,
                )
            except Exception as e:
                print(f"Error (row={a_code.folder}, col={a_paper.folder}): {e}")
                score = 0
            M[i][j] = score
            time.sleep(RATE_LIMIT_SLEEP)

    save_csv(row_labels, col_labels, M, "matrix.csv")
    print("Saved matrix.csv")

if __name__ == "__main__":
    main()
