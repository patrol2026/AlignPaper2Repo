# AlignPaper2Repo

**AlignPaper2Repo** provides the replication package and Visual Studio Code extension for our paper *"An Empirical Study of Large Language Models for Aligning Research Papers with Replication Packages"* (under review) — enabling alignment between research papers and their corresponding source code repositories.

The study evaluates seven LLMs — Claude Sonnet 4.5, Gemini 2.5 Pro, and GPT-5 (proprietary); Qwen3-8B, Llama3-8B, Qwen3-30B, and Llama 3.3 70B (open-weight) — under four prompting strategies (zero-shot, one-shot, two-shot, chain-of-thought) on 117 queries drawn from 20 paper–repository pairs.

## Dataset
Please download the dataset from this link [data](https://zenodo.org/records/23991049) and put it in the root folder.

## Installation

### Option 1: Install from Source

Clone this repository and build the VS Code extension:

```bash
npm install
npm run compile
npx @vscode/vsce package
code --install-extension AlignPaper2RepoExt-*.vsix
```

### Option 2: Install from Marketplace
You can also install directly from the Visual Studio Marketplace:
[AlignPaper2RepoExt on VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=AlignPaper2Repository.AlignPaper2RepoExt)

## Model Setup

To reproduce experiments, you can use proprietary or open-source models. Set `paperRepo.apiBase` to the endpoint for your chosen model:

### Proprietary models
#### For GPT-5
```
https://api.openai.com/v1
```
#### For Gemini 2.5 Pro
```
https://generativelanguage.googleapis.com/v1beta/openai/
```
#### For Claude Sonnet 4.5
```
https://openrouter.ai/api/v1
```
### For Llama and Qwen open-weight models (via Ollama)
```
http://localhost:11434/v1
```
### TIPS:
You can adjust model endpoints and API keys from
VS Code Settings → Extensions → AlignPaper2Repo Configuration (Cmd + , on macOS).

## Running Experiments
### RQ1 – RQ2 – RQ4
To reproduce RQ1, RQ2, and RQ4 results, run:
```bash
python eval_from_dataset_csv.py --dataset_csv dataset.csv --out_csv predictions_gpt5.csv --summarizer_provider openai --summarizer_model gpt-5 --resolver_provider openai --resolver_model gpt-5 --eval_modes cot_style --k 5
```
and then summarize the results with:
```bash
python evaluate_results.py --gt "dataset.csv" --models_folder="gpt_results/" --outdir="gpt_results_summary/"
```

### RQ3 – Attribution and Misattribution

The folder `rq3` contains the paper contents and corresponding representative implementation files (20 papers × 20 files). Each model scores every paper–file pair on a 1–5 Likert scale, producing the 20×20 matrices reported in the paper.

1. Run one of the following scripts:
```bash
python gemini.py
python gpt.py
python claude.py
python open_source.py
```
2. The results are saved under `results_for_rq3/`.

3. Generate plots using:
```bash
python plot.py --folder ./results_for_rq3
```

## VS Code Extension Usage

1. Open VS Code and install AlignPaper2RepoExt.

2. Press Cmd + Shift + P, run **Paper–Repo: Set LLM API Key** (and optionally **Paper–Repo: Set Embedding API Key**) for your chosen model.

3. Press Cmd + Shift + P again → **Paper–Repo: Open App (Right Panel)**.

4. Select the paper (PDF) and its implementation repository, then click **Build Index** and wait for the completion notification.

You'll see:

- An alignment score for each prediction

- Five suggested prompts based on paper context

- A result view showing the relevant code span and a concise explanation

- The extension highlights how confident the LLM is in locating the correct code segment based on your query.

## Reproducibility Notes
RQ1 and RQ2 run on `dataset.csv` (117 queries over 20 paper–repository pairs; BM25 retrieval with K=5 candidates).

RQ3 relies on manually validated paper–repository pairs.

RQ4 measures per-query latency for API-hosted and locally hosted models; absolute values depend on network and hardware conditions, so they are environment-specific.

## Extension

[AlignPaper2RepoExt on VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=AlignPaper2Repository.AlignPaper2RepoExt)

## License

[MIT](https://choosealicense.com/licenses/mit/)
