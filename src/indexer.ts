import * as vscode from "vscode";
import fg from 'fast-glob';
import * as path from 'path';
import * as fs from 'fs/promises';
import OpenAI from 'openai';
import Parser from "tree-sitter";
const Python = require("tree-sitter-python");
import { convertLanguage } from 'tree-sitter-compat';

const parser = new Parser();
parser.setLanguage(convertLanguage(Python));


export type ParsedFunction = {
  qualname: string;     // best-effort: function name (you can extend to Class.method)
  signature: string;    // best-effort: name + parameters text
  startLine: number;
  endLine: number;
  docstring: string;
  code: string;
};


export function parseFile(source: string, maxCodeChars: number): ParsedFunction[] {
  const tree = parser.parse(source);
  const lines = source.split("\n");
  const out: ParsedFunction[] = [];

  function getText(node: any) {
    if (!node) return "";
    return source.slice(node.startIndex, node.endIndex);
  }

  function getDocstringFromBody(bodyNode: any): string {
    // Python docstring convention: first statement in body is a string literal
    if (!bodyNode || bodyNode.namedChildCount === 0) return "";
    const first = bodyNode.namedChild(0);

    // Often: expression_statement -> string
    if (first?.type === "expression_statement") {
      const inner = first.firstNamedChild;
      if (inner?.type === "string") return getText(inner);
    }
    return "";
  }

  function walk(node: any, currentClass?: string) {
    // class_definition: track class name for qualname
    if (node.type === "class_definition") {
      const nameNode = node.childForFieldName("name");
      const className = nameNode ? getText(nameNode) : currentClass;
      const body = node.childForFieldName("body");
      if (body) {
        for (let i = 0; i < body.namedChildCount; i++) {
          walk(body.namedChild(i), className);
        }
      }
      return;
    }

    if (node.type === "function_definition") {
      const nameNode = node.childForFieldName("name");
      const paramsNode = node.childForFieldName("parameters");
      const bodyNode = node.childForFieldName("body");

      const name = nameNode ? getText(nameNode) : "unknown";
      const qualname = currentClass ? `${currentClass}.${name}` : name;
      const signature = `${name}${paramsNode ? getText(paramsNode) : "()"}`;

      const startLine = node.startPosition.row + 1;
      const endLine = node.endPosition.row + 1;

      const docstring = getDocstringFromBody(bodyNode);

      // code slice from lines to keep line alignment
      const code = lines.slice(startLine - 1, endLine).join("\n").slice(0, maxCodeChars);

      out.push({ qualname, signature, startLine, endLine, docstring, code });
    }

    // continue walking
    for (let i = 0; i < node.namedChildCount; i++) {
      walk(node.namedChild(i), currentClass);
    }
  }

  walk(tree.rootNode, undefined);
  return out;
}


// Lazy import pdf-parse to keep activation light
async function parsePdfText(pdfPath?: string): Promise<string | undefined> {
if (!pdfPath) return undefined;
try {
// eslint-disable-next-line @typescript-eslint/no-var-requires
const pdf = require('pdf-parse');
const data = await fs.readFile(pdfPath);
const out = await pdf(data);
return typeof out?.text === 'string' ? out.text : undefined;
} catch {
return undefined;
}
}
export type IndexChunk = {
path: string;
startLine: number;
endLine: number;
text: string;
embedding?: number[];
};


// export type IndexBlob = {
// repoPath: string;
// chunks: IndexChunk[];
// meta: {
// madeAt: number;
// embeddingModel?: string;
// maxChunkLines: number;
// paperPresent: boolean;
// };
// };
function tokenizeLines(text: string, maxLines: number): { start: number; end: number; text: string }[] {
const lines = text.split(/\r?\n/);
const chunks: { start: number; end: number; text: string }[] = [];
for (let i = 0; i < lines.length; i += maxLines) {
const start = i;
const end = Math.min(lines.length - 1, i + maxLines - 1);
const slice = lines.slice(start, end + 1).join('');
chunks.push({ start, end, text: slice });
}
return chunks;
}


async function embedTexts(client: OpenAI, model: string, inputs: string[]): Promise<number[][]> {
const resp = await client.embeddings.create({ model, input: inputs });
return resp.data.map(d => d.embedding as number[]);
}
function cosineSim(a: number[], b: number[]): number {
let dot = 0, na = 0, nb = 0;
const n = Math.min(a.length, b.length);
for (let i = 0; i < n; i++) {
const x = a[i], y = b[i];
dot += x * y; na += x * x; nb += y * y;
}
return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-12);
}

function sanitize(s: string): string {
  if (!s) return '';
  // drop control chars and collapse whitespace
  return s
    .replace(/[\u0000-\u001F\u007F]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}
// export async function buildIndex(opts: {
// repoPath: string;
// paperPath?: string;
// model: string;
// maxChunkLines: number;
// apiKey?: string;
// embeddingApiKey?: string;
// apiBase?: string;
// embeddingModel?: string;
// }): Promise<IndexBlob> {
// const { repoPath, paperPath,apiKey,embeddingApiKey, model,maxChunkLines, apiBase, embeddingModel } = opts;
// const patterns = [
//   '**/*.{ts,tsx,js,jsx,py,java,kt,go,rs,cpp,c,hpp,h,cs,scala,rb,php,swift,sh,txt}', // everything else
//   '**/[Rr][Ee][Aa][Dd][Mm][Ee].[Mm][Dd]'                                           // README.md (any case)
// ];
// const ignore = ['**/node_modules/**', '**/.git/**', '**/dist/**', '**/build/**', '**/.venv/**'];
// const abs = await fg(patterns, { cwd: repoPath, ignore, dot: false, absolute: true });
// //vscode.window.showInformationMessage(`Found ${repoPath} files in paperpath ${paperPath}.`);

// const chunks: IndexChunk[] = [];
// for (const p of abs) {
//     try {
// const buf = await fs.readFile(p);
// const text = new TextDecoder().decode(buf);
// for (const c of tokenizeLines(text, Math.max(1, maxChunkLines))) {
// chunks.push({ path: p, startLine: c.start, endLine: c.end, text: c.text });
// }
// } catch { 
//   vscode.window.showInformationMessage(`Found ${repoPath} files in paperpath ${paperPath}.`);
//   /* ignore file read errors */ }
// }
// //vscode.window.showInformationMessage(`Indexed ${chunks.length} chunks from ${abs.length} files.`);
// //const apiKey='ollama';
//  //baseURL:model.includes('claude')?'https://api.openai.com/v1':
// const client = apiKey ? new OpenAI({ apiKey: embeddingApiKey,baseURL:model.includes('claude')?'https://api.openai.com/v1':apiBase }) : undefined;
// //vscode.window.showInformationMessage(`Using embedding model: ${client}`);
// const embModel = embeddingModel;
// //vscode.window.showInformationMessage(`apibase and embedding model: ${apiBase}, ${embModel}`);
// if ( embModel) {
//   try {
//     // Build a cleaned list of non-empty texts with their original indices
//     const cleaned: { idx: number; text: string }[] = [];
//     for (let i = 0; i < chunks.length; i++) {
//       const s = (chunks[i].text ?? '').toString().trim();
//       if (s.length) cleaned.push({ idx: i, text: s });
//     }
    
//     if (cleaned.length === 0) {
//       console.warn('[index] no non-empty texts; skipping embeddings');
//     } else {
//          // vscode.window.showInformationMessage(`Computing embeddings for  non-empty chunks...`);

//      const B = 64; // modest batch size for most providers
//       for (let i = 0; i < cleaned.length; i += B) {
//         const slice = cleaned.slice(i, i + B);
//         const res = await client!.embeddings.create({
//           model: embModel,
//           input: slice.map(x => x.text) // never empty
//         });
//        // vscode.window.showInformationMessage(`Computing and stroing to the correct index for  non-empty chunks...`);

//         // write back to the correct chunk indices
//         res.data.forEach((d, j) => {
//           chunks[slice[j].idx].embedding = d.embedding as number[];
//               //    vscode.window.showInformationMessage(`Computing and stroing to the correct index for  non-empty chunks...${d.embedding}`);

//         });
//       }
     
// }
    
//   } catch (e: any) {
//      vscode.window.showInformationMessage(`embeddings disabled during indexing:${e?.message || e}`);
//     console.warn('[index] embeddings disabled during indexing:', e?.message || e);
//     // (Optional) vscode.window.showWarningMessage('Embeddings unavailable; using text-only ranking.');
//     // Continue without embeddings; askWithLLM should handle this case.
//   }
// }

// else{
//   vscode.window.showInformationMessage(`Skipping embeddings; no API key or model.`);
// }

// return {
// repoPath,
// chunks,
// meta: { madeAt: Date.now(), embeddingModel: embModel, maxChunkLines, paperPresent: !!paperPath }
// };
// }
type PromptStrategy = "zero_shot" | "one_shot" | "two_shot" | "cot";

export type FunctionEntry = {
  file_path: string;     // relative path
  qualname: string;      // e.g., Class.method or function name
  signature: string;     // best-effort signature
  start_line: number;
  end_line: number;
  docstring: string;     // for TS/JS you can store JSDoc or empty
  code: string;
  descriptions: Partial<Record<PromptStrategy, string>>; // ND per strategy
};

export type BM25Index = {
  docTokens: string[][];
  df: Record<string, number>;
  avgdl: number;
};

export type IndexBlob = {
  repoPath: string;
  entries: FunctionEntry[];
  bm25: BM25Index;
  meta: {
    madeAt: number;
    paperPresent: boolean;
    llmModel: string;
    promptStrategy: PromptStrategy;
  };
};

function tokenize(text: string): string[] {
  return (text || "")
    .toLowerCase()
    .replace(/[^a-z0-9_]+/g, " ")   // keep snake/camel parts as tokens
    .split(/\s+/)
    .filter(t => t.length >= 2);
}

function buildBM25(entries: FunctionEntry[]): BM25Index {
  const docs: string[][] = [];
  const df: Record<string, number> = {};

  for (const e of entries) {
    // Include whatever descriptions exist (all modes) for retrieval
    const descBlob = Object.entries(e.descriptions)
      .map(([k, v]) => `${k}: ${v}`)
      .join("\n");

    const text = [
      e.file_path,
      e.qualname,
      e.signature,
      e.docstring,
      descBlob,
    ].join("\n");

    const toks = tokenize(text);
    docs.push(toks);

    const seen = new Set(toks);
    for (const t of seen) df[t] = (df[t] || 0) + 1;
  }

  const avgdl = docs.length ? docs.reduce((s, d) => s + d.length, 0) / docs.length : 0;
  return { docTokens: docs, df, avgdl };
}
async function generateNaturalDescription(opts: {
  client: OpenAI;
  llmModel: string;
  strategy: PromptStrategy;
  filePath: string;
  qualname: string;
  signature: string;
  docstring: string;
  code: string;
}): Promise<string> {
  const { client, llmModel, strategy, filePath, qualname, signature, docstring, code } = opts;

  const system =
    "You write short, accurate natural-language descriptions of code. Do not guess.";

  const user = [
    `Prompting mode: ${strategy}`,
    `File: ${filePath}`,
    `Qualified name: ${qualname}`,
    `Signature: ${signature}`,
    docstring ? `Docstring/JSDoc:\n${docstring}` : "",
    "",
    "Task: Describe what this function/method does in 1–3 sentences.",
    "Constraints: Mention the main goal, key inputs/outputs, and side effects if clear. Do not include code.",
    "",
    "Code:",
    "```",
    code,
    "```",
  ].filter(Boolean).join("\n");

  const resp = await client.chat.completions.create({
    model: llmModel,
    messages: [
      { role: "system", content: system },
      { role: "user", content: user },
    ],
    temperature: 1,
  });

  return (resp.choices?.[0]?.message?.content ?? "").trim();
}

async function mapWithConcurrency<T>(
  items: T[],
  limit: number,
  fn: (item: T, index: number) => Promise<void>
): Promise<void> {
  let i = 0;

  const workers = new Array(limit).fill(0).map(async () => {
    while (true) {
      const idx = i++;
      if (idx >= items.length) break;
      await fn(items[idx], idx);
    }
  });

  await Promise.all(workers);
}

export async function buildIndex(opts: {
  repoPath: string;
  paperPath?: string;
  model: string;
  apiKey?: string;
  apiBase?: string;
  promptStrategy: PromptStrategy;
  maxCodeChars: number; // matches Python max_code_chars
}): Promise<IndexBlob> {
  const { repoPath, paperPath, model, apiKey, apiBase, promptStrategy, maxCodeChars } = opts;

  const patterns = ["**/*.{ts,tsx,js,jsx,py,java,kt,go,rs,cpp,c,hpp,h,cs,scala,rb,php,swift}"];
  const ignore = ["**/node_modules/**", "**/.git/**", "**/dist/**", "**/build/**", "**/.venv/**"];
  const abs = await fg(patterns, { cwd: repoPath, ignore, dot: false, absolute: true });
  //vscode.window.showInformationMessage(`Indexing ${abs.length} files from ${repoPath} using model ${model}.`);
  const client = new OpenAI({
    apiKey,
    baseURL: model.includes("claude") ? "https://api.openai.com/v1" : apiBase,
  });

  const entries: FunctionEntry[] = [];

  // --- AST parse each file and collect function entries (no fallback) ---
  for (const full_path of abs) {
    try {
      const rel = path.relative(repoPath, full_path);
      const buf = await fs.readFile(full_path);
      const text = new TextDecoder().decode(buf);

      // parseFile should return: qual, sig, start, end, doc, code
      // You already have this in Python. In TS you’ll implement per-language parsing.
      const funcs = parseFile( text, maxCodeChars );

      for (const f of funcs) {
        entries.push({
          file_path: rel,
          qualname: f.qualname,
          signature: f.signature,
          start_line: f.startLine,
          end_line: f.endLine,
          docstring: f.docstring ?? "",
          code: f.code,
          descriptions: {},
        });
      }
      //vscode.window.showInformationMessage(`Parsed ${funcs.length} functions from ${rel}.`);
    } catch {
      // skip unreadable files / parse errors
    }
  }

  // --- Generate ND for current strategy ---
  // (matches your Python: e.descriptions = {} initially, then filled per mode)
  vscode.window.showInformationMessage(`Generating natural descriptions for functions using strategy`);
  await mapWithConcurrency(entries, 4, async (e) => {

    const nd = await generateNaturalDescription({
      client,
      llmModel: model,
      strategy: promptStrategy,
      filePath: e.file_path,
      qualname: e.qualname,
      signature: e.signature,
      docstring: e.docstring,
      code: e.code,
    });
    //vscode.window.showInformationMessage(`Generated ND for ${e.file_path}::${e.qualname}`);
    e.descriptions[promptStrategy] = nd;
  });

  // --- Build BM25 on file_path + qualname + signature + docstring + descriptions blob ---
  const bm25 = buildBM25(entries);
  //vscode.window.showInformationMessage(`Built BM25 index with ${entries.length} entries.`);
  return {
    repoPath,
    entries,
    bm25,
    meta: {
      madeAt: Date.now(),
      paperPresent: !!paperPath,
      llmModel: model,
      promptStrategy,
    },
  };
}

/**
 * BM25 ranker for NL→code without prebuilding state.
 * Returns [{ i, score }] where i is chunk index in index.chunks.
 */
 function bm25RankOnce(
  index: { chunks: { text: string }[] },
  question: string,
  opts: { k1?: number; b?: number; minWordLen?: number } = {}
): { i: number; score: number }[] {
  const chunks = index?.chunks || [];
  if (!chunks.length) return [];

  const k1 = opts.k1 ?? 1.6;
  const b  = opts.b  ?? 0.75;
  const MIN = opts.minWordLen ?? 2;

  // --- tiny code-aware tokenizers (local) ---
  const splitId = (s: string) =>
    s.replace(/([a-z])([A-Z])/g, '$1 $2')
     .toLowerCase()
     .split(/[^a-z0-9]+/i)
     .filter(w => w.length > MIN);

  const tokenizeCode = (text: string) => {
    const t = text || '';
    const toks: string[] = [];
    // comments & docstrings
    const comments = t.match(/\/\/.*|#.*|\/\*[\s\S]*?\*\/|"""[\s\S]*?"""|'''[\s\S]*?'''/g) || [];
    comments.forEach(c => toks.push(...splitId(c)));
    // strings
    const strings = t.match(/(["'`])(?:\\.|(?!\1).)*\1/g) || [];
    strings.forEach(s => toks.push(...splitId(s)));
    // identifiers
    const ids = t.match(/[A-Za-z_][A-Za-z0-9_]{2,}/g) || [];
    ids.forEach(id => toks.push(...splitId(id)));
    return toks;
  };

  // --- build per-doc bags + DF, avgdl ---
  const df = new Map<string, number>();
  const bags: Array<Record<string, number>> = new Array(chunks.length);
  let totalLen = 0;

  for (let i = 0; i < chunks.length; i++) {
    const toks = tokenizeCode(chunks[i].text);
    totalLen += toks.length;
    const bag: Record<string, number> = {};
    for (const t of toks) bag[t] = (bag[t] || 0) + 1;
    bags[i] = bag;

    const uniq = new Set(Object.keys(bag));
    for (const t of uniq) df.set(t, (df.get(t) || 0) + 1);
  }

  const N = Math.max(1, chunks.length);
  const avgdl = Math.max(1, totalLen / N);

  // --- query terms ---
  const qTerms = splitId(question);

  // --- score each chunk with BM25 ---
  const scored = chunks.map((_, i) => {
    const bag = bags[i];
    const dl  = Object.values(bag).reduce((a, n) => a + n, 0) || 1;
    let score = 0;

    for (const q of qTerms) {
      const f = bag[q] || 0;
      if (!f) continue;
      const n = df.get(q) || 0;
      const idf = Math.log((N - n + 0.5) / (n + 0.5) + 1e-6);
      const tf  = (f * (k1 + 1)) / (f + k1 * (1 - b + b * (dl / avgdl)));
      score += idf * tf;
    }
    return { i, score };
  });

  // higher first
  scored.sort((a, b) => b.score - a.score);
  return scored;
}
export function extractJsonObject(s: string): any | null {
  if (!s) return null;
  // strip ```json ... ```
  s = s.trim().replace(/^```(?:json)?\s*/i, "").replace(/```$/i, "");
  return s;
  vscode.window.showInformationMessage(`Extracting JSON from LLM response.. ${s}`);
  // fast path
  try { return JSON.parse(s); } catch {}

  // fallback: grab first balanced {...}
  const start = s.indexOf("{");
  if (start < 0) return null;
  let depth = 0;
  for (let i = start; i < s.length; i++) {
    const ch = s[i];
    if (ch === "{") depth++;
    else if (ch === "}") {
      depth--;
      if (depth === 0) {
        const cand = s.slice(start, i + 1);
        try { return JSON.parse(cand); } catch {}
      }
    }
  }
  return null;
}
// BM25 search that mirrors the Python implementation exactly:
// - tokenizes query
// - k1 = 1.5, b = 0.75
// - idf = log(1 + (N - df + 0.5) / (df + 0.5))
// - per-doc term frequencies computed on the fly
// - returns top-k (docIndex, score) with score > 0



// export function tokenize(text: string): string[] {
//   return (text || "")
//     .toLowerCase()
//     .replace(/[^a-z0-9_]+/g, " ")
//     .split(/\s+/)
//     .filter((t) => t.length >= 2);
// }

export function bm25Search(
  index: BM25Index,
  query: string,
  k: number = 25
): Array<[number, number]> {
  const { docTokens, df, avgdl } = index;

  if (!docTokens || docTokens.length === 0) return [];

  const q = tokenize(query);
  if (q.length === 0) return [];

  const N = docTokens.length;
  const k1 = 1.5;
  const b = 0.75;

  const scores: number[] = new Array(N).fill(0);

  for (let i = 0; i < N; i++) {
    const doc = docTokens[i];
    const dl = doc.length || 1;

    // term frequencies for this doc
    const freqs: Record<string, number> = {};
    for (const t of doc) freqs[t] = (freqs[t] || 0) + 1;

    let s = 0;
    for (const term of q) {
      const dfi = df[term] || 0;
      if (dfi === 0) continue;

      const idf = Math.log(1 + (N - dfi + 0.5) / (dfi + 0.5));

      const tf = freqs[term] || 0;
      if (tf === 0) continue;

      const denom = tf + k1 * (1 - b + b * (dl / (avgdl || 1.0)));
      s += idf * (tf * (k1 + 1)) / denom;
    }

    scores[i] = s;
  }

  // rank descending by score
  const ranked = scores
    .map((sc, i) => [i, sc] as [number, number])
    .sort((a, b) => b[1] - a[1]);

  // return top-k with sc > 0
  const out: Array<[number, number]> = [];
  for (const [i, sc] of ranked) {
    if (out.length >= k) break;
    if (sc > 0) out.push([i, sc]);
  }
  return out;
}
function makeResolverSystemPrompt(): string {
  return `
Task: Choose the best candidate that answers the user query.

Return ONLY JSON with EXACTLY these keys:
- chosen_index (int or null)   // entry_index from candidates
- llm_score (number 0..1)      // self-reported score of your choice
- reason (string < 200 words)
`;
}

type ResolverCandidate = {
  entry_index: number;
  file_path: string;
  qualname: string;
  start_line: number;
  end_line: number;
  signature: string;
  natural_description: string;
  retrieval_score: number;
};

function makeResolverUserPrompt(userQuery: string, candidates: ResolverCandidate[]) {
  const formatted = candidates.map((c, i) => {
    return [
      `#${i} entry_index=${c.entry_index} bm25=${c.retrieval_score.toFixed(4)}`,
      `file=${c.file_path} lines=[${c.start_line}-${c.end_line}]`,
      `qualname=${c.qualname}`,
      `signature=${c.signature}`,
      `natural_description=${c.natural_description}`,
    ].join("\n");
  }).join("\n---\n");

  return [
    { type: "text", text: `User query: ${userQuery}` },
    { type: "text", text: `Candidates (top-${candidates.length}):` },
    { type: "text", text: formatted },
    { type: "text", text: `Pick the best candidate and return the JSON.` },
  ];
}
function safeJsonParse(s: string): any | null {
  try {
    return JSON.parse(extractJsonObject(s));
  } catch {
    return null;
  }
}
function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

export function retrievalGapConfidence(
  scores: number[],
  temperature: number = 2.0
): number {
  // scores: descending (top first)
  if (!scores || scores.length === 0) {
    return 0.0;
  }

  if (scores.length === 1) {
    return 0.55;
  }

  const s1 = scores[0];
  const s2 = scores[1];

  const denom = Math.max(temperature, 1e-6);
  return sigmoid((s1 - s2) / denom);
}

export async function askWithLLM(index: IndexBlob, opts: {
question: string;
topK: number;
model: string;
apiKey: string;
embeddingApiKey?: string;
apiBase?: string;
temperature?: number;
}): Promise<{ path: string; startLine?: number; endLine?: number;reason?:string; bestLine?: number; score?: number } | undefined> {
const { question, topK, model, apiKey,embeddingApiKey, apiBase, temperature } = opts;
//const client = new OpenAI({ apiKey, baseURL: apiBase });
//vscode.window.showInformationMessage(`LLM asked: ${question} `);


// If we have embeddings, rank by cosine; else fallback to simple tf (length) heuristic
// let scored = index.chunks.map((c, i) => ({ i, score: 0 }));
// try {
// const embModel = index.meta.embeddingModel;
// //vscode.window.showInformationMessage(`embmodel for ${embModel}, ${index.chunks[0]?.embedding} `);

// if (embModel && index.chunks[0]?.embedding) {
//   //const apiKey='ollama';
//   const client2 = new OpenAI({ apiKey:embeddingApiKey, baseURL:model.includes('claude')?'https://api.openai.com/v1': apiBase}) ;
// const qv = (await client2.embeddings.create({ model: embModel, input: [question] })).data[0].embedding as number[];
// //  const response = await fetch('http://localhost:11434/api/embeddings', {
// //     method: 'POST',
// //     headers: { 'Content-Type': 'application/json' },
// //     body: JSON.stringify({
// //       model: 'mxbai-embed-large',
// //       input: sanitize(question), // Make sure this is an array of strings
// //     }),
// //   });
// // const raw:any = await response.json();
// // const emb: number[] = raw.embedding; 
// scored = index.chunks.map((c, i) => ({ i, score: c.embedding ? cosineSim(qv, c.embedding) : 0 }));
// //vscode.window.showInformationMessage(`Fell back to cosine similarity for ranking.${scored}`);
// }
// else{
//   scored = bm25RankOnce(index, question);

// vscode.window.showInformationMessage(`Fell back to lexical similarity for ranking.`);
// }
// } 
// catch (e: any) {
// console.warn('[ask] ranking fallback:', e?.message || e);
// // (Optional) vscode.window.showWarningMessage('Embeddings unavailable; using text-only ranking.');
// // scored stays as initialized (all zero)
// }

// scored.sort((a,b) => b.score - a.score);
// const pick = scored.slice(0, Math.max(1,topK));
// //vscode.window.showInformationMessage(`Top chunk score: ${pick.length}`);
// const ctx = pick.map(({ i, score }, rank) => ({
// rank, score, path: index.chunks[i].path, start: index.chunks[i].startLine, end: index.chunks[i].endLine, text: index.chunks[i].text
// }));
//vscode.window.showInformationMessage(`Top chunk path: ${ctx} `);
const hits=bm25Search(index.bm25, question, topK);
//vscode.window.showInformationMessage(`hits founds: ${hits.length} `);
  const candidates: ResolverCandidate[] = [];
  const hitScores: number[] = [];

  for (const [entryIdx, score] of hits) {
    const e = index.entries[entryIdx];

    const desc =
      e.descriptions?.[index.meta.promptStrategy] ??
      (e.descriptions ? (Object.values(e.descriptions)[0] ?? "") : "");

    candidates.push({
      entry_index: entryIdx,
      file_path: e.file_path,
      qualname: e.qualname,
      start_line: e.start_line,
      end_line: e.end_line,
      signature: e.signature,
      natural_description: desc,
      retrieval_score: score,
    });

    hitScores.push(score);
  }
  //vscode.window.showInformationMessage(`candidates prepared: ${candidates.length} `);

  // ---- 2) If no candidates ----
  if (candidates.length === 0) {
    return {
      path: "",
      startLine: 0,
      endLine: 0,
      reason: "No candidates found in index.",
      bestLine: 0,
      score: 0,
    };
  }

  // ---- 3) Retrieval-based score (model-agnostic) ----
  const rScore = retrievalGapConfidence(hitScores, 2.0);

  // ---- 4) Call resolver LLM ----
  const openai = new OpenAI({
    baseURL: apiBase,
    apiKey: apiKey,
  });

  const sys = makeResolverSystemPrompt();
  const user = makeResolverUserPrompt(question, candidates);
  //vscode.window.showInformationMessage(`Calling LLM resolver with ${sys}  and ${user} candidates.`);
  const completion = await openai.chat.completions.create({
    model: model,
    messages: [
      { role: "system", content: sys },
      { role: "user", content: user as any },
    ],
    temperature: 1,
  });

  const content = completion.choices[0]?.message?.content ?? "{}";
  //vscode.window.showInformationMessage(`LLM resolver response: ${content} `);
  const obj = safeJsonParse(content);

  // ---- 5) Fallback to top retrieval if invalid JSON ----
  if (!obj || typeof obj.chosen_index === "undefined") {
    const c0 = candidates[0];
    // const llmScore = 0.5;
    // const finalScore = 0.7 * rScore + 0.3 * llmScore;

    return {
      path: c0.file_path,
      startLine:  c0.start_line,
      endLine: c0.end_line,
      bestLine: 0,
      score: rScore,
      reason: c0.natural_description
      // llm_score: llmScore,
      // retrieval_score: rScore,
      // final_score: finalScore,
      // reason: "Resolver output invalid JSON; returned top retrieval match.",
      // answer: {
      //   file_path: c0.file_path,
      //   start_line: c0.start_line,
      //   end_line: c0.end_line,
      //   natural_description: c0.natural_description,
      // },
    };
  }

  // ---- 6) Clamp llm_score & combine ----
  let llmScore = Number(obj.llm_score ?? 0.5);
  if (!Number.isFinite(llmScore)) llmScore = 0.5;
  llmScore = Math.max(0.0, Math.min(1.0, llmScore));

  const finalScore = 0.7 * rScore + 0.3 * llmScore;

  // ---- 7) Attach scores + timing ----
  obj.retrieval_score = rScore;
  obj.final_score = finalScore;
  const result={path: "", startLine: 0, endLine: 0, bestLine: 0, score: finalScore, reason: obj.reason || ""};
  // Optional: add answer block if chosen_index points to a candidate
  if (obj.chosen_index !== null && Number.isInteger(obj.chosen_index)) {
    const chosen = index.entries[obj.chosen_index];
    if (chosen) {
      const desc =
          chosen.descriptions?.[index.meta.promptStrategy] ??
          (chosen.descriptions ? (Object.values(chosen.descriptions)[0] ?? "") : "");
          result.path=chosen.file_path;
          result.startLine=chosen.start_line;
          result.endLine=chosen.end_line;
          result.reason=desc;
          result.score=finalScore;
          //result.score=finalScore;
      // obj.answer = {
      //   file_path: chosen.file_path,
      //   start_line: chosen.start_line,
      //   end_line: chosen.end_line,
      //   natural_description: desc,
      // };
    }
  }

  return result;
}
export async function saveIndex(ctx: vscode.ExtensionContext, idx: IndexBlob) {
const target = vscode.Uri.joinPath(ctx.globalStorageUri, 'index.json');
const payload = JSON.stringify(idx);
await vscode.workspace.fs.writeFile(target, new TextEncoder().encode(payload));
}


export async function loadIndex(ctx: vscode.ExtensionContext): Promise<IndexBlob | undefined> {
try {
const target = vscode.Uri.joinPath(ctx.globalStorageUri, 'index.json');
const data = await vscode.workspace.fs.readFile(target);
const json = JSON.parse(new TextDecoder().decode(data)) as IndexBlob;
return json;
} catch {
return undefined;
}
}
export async function deleteIndex(ctx: vscode.ExtensionContext): Promise<boolean> {
  const target = vscode.Uri.joinPath(ctx.globalStorageUri, 'index.json');

  // If it doesn't exist, treat as "nothing to delete"
  try {
    await vscode.workspace.fs.stat(target);
  } catch {
    return false;
  }

  // Try to delete; don't send to OS trash
  try {
    await vscode.workspace.fs.delete(target, { useTrash: false });
    return true;
  } catch {
    return false;
  }
}
export async function suggestQuestions(
  index: IndexBlob,
  opts: { apiKey: string; apiBase?: string; model: string; paperPath?: string }
): Promise<string[]> {
  const client = new OpenAI({ apiKey: opts.apiKey, baseURL: opts.apiBase });

  // Try reading a small excerpt of the paper (optional)
  let paperText = '';
  if (opts.paperPath) {
    try {
      paperText = (await parsePdfText(opts.paperPath))?.slice(0, 2000) || '';
    } catch { /* ignore */ }
  }

  // Give the LLM a sense of the repo contents (limit for token safety)
  const files = Array.from(new Set(index.entries.map(c => c.file_path))).slice(0, 40);
  const system = 'You are helping a newcomer read a research paper and its codebase. Propose five short, beginner-friendly questions that help them explore the implementation based on the paper and files. Return strictly a JSON object: { "questions": string[5] }.';
  const user = `Paper excerpt (may be empty):${paperText}

Repo files (sample):
${files.join('\n')}
`;

  const chat = await client.chat.completions.create({
    model: opts.model,
    temperature: 1,
    response_format: { type: 'json_object' },
    messages: [
      { role: 'system', content: system },
      { role: 'user', content: user }
    ]
  });

  try {
    const content = chat.choices[0]?.message?.content || '{}';
    const parsed = JSON.parse(content);
    const arr = Array.isArray(parsed?.questions) ? parsed.questions : [];
    return arr.filter((s: any) => typeof s === 'string').slice(0, 5);
  } catch {
    return [];
  }
}
// indexer.ts

async function embedAll(
  client: OpenAI,
  model: string,
  texts: string[]
): Promise<number[][]> {
  // Clean input and keep index mapping
  const cleaned: { idx: number; text: string }[] = [];
  texts.forEach((t, i) => {
    const s = (t ?? "").toString().trim();
    if (s.length > 0) cleaned.push({ idx: i, text: s });
  });

  if (cleaned.length === 0) return []; // nothing to embed (avoid 400)

  const BATCH = 128; // keep it modest for most providers
  const out: number[][] = new Array(texts.length); // aligned to original length (sparse)
  for (let i = 0; i < cleaned.length; i += BATCH) {
    const slice = cleaned.slice(i, i + BATCH);
    const res = await client.embeddings.create({
      model,
      input: slice.map(x => x.text)
    });
    // write back embeddings aligned to original positions
    res.data.forEach((d, j) => { out[slice[j].idx] = d.embedding as number[]; });
  }
  return out;
}

export async function suggestFollowupsFromPrompt(
  prompt: string,
  opts: { apiKey: string; apiBase?: string; model: string ,paperPath?:string}
): Promise<string[]> {
  const client = new OpenAI({ apiKey: opts.apiKey, baseURL: opts.apiBase });
  let paperText = '';
  if (opts.paperPath) {
    try {
      paperText = (await parsePdfText(opts.paperPath))?.slice(0, 2000) || '';
    } catch { /* ignore */ }
  }

  // Give the LLM a sense of the repo contents (limit for token safety)
  //const files = Array.from(new Set(index.chunks.map(c => c.path))).slice(0, 40);

  const sys = 'You are helping a newcomer read a research paper and its codebase. Propose five short, beginner-friendly questions that help them explore the implementation based on the paper and files. Return strictly a JSON object: { "questions": string[5] }.';
  const user = `Paper excerpt (may be empty):${paperText}
and the user prompt is: ${prompt}
}`;
  // const sys =
  //   'You generate 5 short follow-up questions based ONLY on the user prompt. Return JSON: {"questions": string[5]}. No preamble.';
  // const user = `User prompt:\n${prompt}`;

  const chat = await client.chat.completions.create({
    model: opts.model,
    temperature: opts.model==='gpt-5'?1:0,
    response_format: { type: "json_object" },
    messages: [
      { role: "system", content: sys },
      { role: "user", content: user }
    ]
  });

  try {
    const content = chat.choices[0]?.message?.content || "{}";
    const parsed = JSON.parse(extractJsonObject(content));
    const arr = Array.isArray(parsed?.questions) ? parsed.questions : [];
    return arr.filter((s: any) => typeof s === "string").slice(0, 5);
  } catch {
    return [];
  }
}


