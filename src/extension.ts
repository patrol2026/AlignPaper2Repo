import * as vscode from 'vscode';
import { buildIndex, extractJsonObject,askWithLLM, loadIndex, saveIndex,suggestQuestions,deleteIndex, type IndexBlob } from './indexer';
import {  suggestFollowupsFromPrompt } from "./indexer";
import { parse } from 'csv-parse/sync';

import * as path from 'path';


const KEY_PAPER = 'paperRepo.paperPath';
const KEY_REPO = 'paperRepo.repoPath';
const SECRET_APIKEY = 'paperRepo.apiKey';
const KEY_PROMPTS = 'paperRepo.prompts';

const SECRET_EMBEDDING_APIKEY = 'paperRepo.embeddingApiKey';
let currentAppPanel: vscode.WebviewPanel | undefined; // ADD


export async function activate(context: vscode.ExtensionContext) {
  console.log("Extension host ABI:", process.versions.modules);
  console.log(
    "Node:", process.versions.node,
    "Electron:", process.versions.electron
  );
  
const { subscriptions, workspaceState, secrets, globalStorageUri } = context;
await vscode.workspace.fs.createDirectory(globalStorageUri);


const getCfg = () => vscode.workspace.getConfiguration('paperRepo');


class ControlsViewProvider implements vscode.WebviewViewProvider {
public static readonly viewId = 'paperRepo.controls';
constructor(private readonly ctx: vscode.ExtensionContext) {}
resolveWebviewView(webviewView: vscode.WebviewView): void | Thenable<void> {
webviewView.webview.options = { enableScripts: true };
const api = `const vscode = acquireVsCodeApi();`;
// <h3>Paper–Repo Q&A</h3>
// <button id='pickRepo'>Select Repo Folder</button>
// <div class='path' id='repoPath'>(none)</div>
// <button id='pickPaper'>Select Paper (PDF)</button>
// <div class='path' id='paperPath'>(optional)</div>
// <hr/>
// <button id='buildIndex'>Build Index</button>
// <button id='ask'>Ask Question</button>

//   </div>
const css = `body{font-family: var(--vscode-font-family);padding:12px;} button{margin-right:8px;margin-bottom:8px;} .path{opacity:.8;margin-top:6px}`;
const html = `<!DOCTYPE html><html><head><meta charset='UTF-8'><style>${css}</style></head><body>

<script>${api}
const basename = (p) => {
  if (!p) return '';
  const s = String(p).replace(/[/\\]+$/,'');        // trim trailing slashes
  const parts = s.split(/[/\\]/);                   // split on / or \
  return parts[parts.length - 1] || s;              // last segment
};

// In your message handler:
if (m.repoPath !== undefined) {
  const name = basename(m.repoPath);
  const node = el('repoPath');
  node.textContent = name;     // show only folder/file name
  node.title = m.repoPath;     // hover shows full path
}

if (m.paperPath !== undefined) {
  const name = basename(m.paperPath);
  const node = el('paperPath');
  node.textContent = name;
  node.title = m.paperPath;
}
const el = id=>document.getElementById(id);
el('pickRepo').onclick=()=>vscode.postMessage({type:'selectRepo'});
el('pickPaper').onclick=()=>vscode.postMessage({type:'selectPaper'});
el('buildIndex').onclick=()=>vscode.postMessage({type:'buildIndex'});
el('ask').onclick=()=>vscode.postMessage({type:'ask'});
window.addEventListener('message', e=>{
const m=e.data||{}; if(m.repoPath) {const basenameNoExt = (p) => basename(p).replace(/\.[^./\\]+$/, '');
el('repoPath').textContent=basenameNoExt(m.repoPath)+"this is file";
} if(m.paperPath) el('paperPath').textContent=m.paperPath;
});
</script></body></html>`;
webviewView.webview.html = html;
const postState = async ()=>{
webviewView.webview.postMessage({ repoPath: workspaceState.get<string>(KEY_REPO)||'(none)', paperPath: workspaceState.get<string>(KEY_PAPER)||'(optional)'});
};
postState();
webviewView.webview.onDidReceiveMessage(async (msg:any)=>{
switch(msg?.type){
case 'selectRepo':{
const pick = await vscode.window.showOpenDialog({ canSelectFiles:false, canSelectFolders:true, canSelectMany:false, title:'Select Repo Folder' });
if(pick && pick[0]){ await workspaceState.update(KEY_REPO, pick[0].fsPath); webviewView.webview.postMessage({ repoPath: pick[0].fsPath }); }
break;}
case 'selectPaper':{
const pick = await vscode.window.showOpenDialog({ canSelectFiles:true, canSelectFolders:false, canSelectMany:false, title:'Select Paper (PDF)', filters:{ PDF:['pdf'] }});
if(pick && pick[0]){ await workspaceState.update(KEY_PAPER, pick[0].fsPath); webviewView.webview.postMessage({ paperPath: pick[0].fsPath }); }
break;}
case 'buildIndex':{
await vscode.commands.executeCommand('paperRepo.buildIndex');
break;}
case 'ask':{
await vscode.commands.executeCommand('paperRepo.askQuestion');
break;}
}
});
}
}
const provider = new ControlsViewProvider(context);
subscriptions.push(vscode.window.registerWebviewViewProvider(ControlsViewProvider.viewId, provider));


// Command: Open App (Webview Panel on right)
subscriptions.push(vscode.commands.registerCommand('paperRepo.openApp', async () => {
const panel = vscode.window.createWebviewPanel(
'paperRepo.app',
'AlignPaper2Repo',
{ viewColumn: vscode.ViewColumn.Two, preserveFocus: false },
  { enableScripts: true, retainContextWhenHidden: true } // ← add this

);
currentAppPanel = panel;
panel.onDidDispose(() => { currentAppPanel = undefined; });
// window.addEventListener('message', e=>{
// const m=e.data||{}; if(m.repoPath) el('repoPath').textContent=m.repoPath; if(m.paperPath) el('paperPath').textContent=m.paperPath;
// });
const api = `const vscode = acquireVsCodeApi();`;
// const css = `
//   :root{--gap:10px}
//   body{font-family: var(--vscode-font-family);padding:12px;}
//   button{margin-right:8px;margin-bottom:8px;}
//   .path{opacity:.8;margin-top:6px}
//   .field{display:flex;gap:var(--gap);margin-top:10px}
//   input[type=text]{flex:1;padding:8px;border-radius:6px;border:1px solid var(--vscode-input-border,transparent);
//     background: var(--vscode-input-background);color: var(--vscode-input-foreground)}
//   .chip{margin:4px 6px 0 0;padding:4px 8px;border-radius:999px;display:inline-block;cursor:pointer;
//     background: var(--vscode-editorWidget-background);border:1px solid var(--vscode-editorWidget-border)}
//   .muted{opacity:.8}
// `;
const css = `
  :root{--gap:12px; --radius:10px;}
  body{font-family: var(--vscode-font-family); padding:16px; color: var(--vscode-foreground);}

  /* layout */
  .row{display:flex; gap:var(--gap); align-items:center; flex-wrap:wrap;}
  .stack{display:flex; flex-direction:column; gap:8px;}
  .grow{flex:1}

  /* cards / surfaces */
  .card{background: var(--vscode-editorWidget-background);
        border:1px solid var(--vscode-editorWidget-border);
        border-radius: var(--radius); padding:12px;}

  /* labels under buttons */
  .sub{opacity:.8; font-size:12px; margin-top:4px}

  /* inputs */
  .field{display:flex; gap:var(--gap); margin-top:12px}
  input[type=text]{
    flex:1; padding:10px 12px; border-radius:8px;
    border:1px solid var(--vscode-input-border, transparent);
    background: var(--vscode-input-background); color: var(--vscode-input-foreground);
    outline: none;
  }

  /* buttons */
  .btn{padding:10px 14px; border-radius:999px; border:1px solid transparent;
       background: var(--vscode-button-secondaryBackground);
       color: var(--vscode-button-foreground); cursor:pointer; transition: transform .06s ease;}
  .btn:hover{transform: translateY(-1px)}
  .btn:active{transform: translateY(0)}
  .btn-primary{
    background: linear-gradient(135deg, #6ea8fe 0%, #a78bfa 100%);
    color: #0b1221; border: none;
  }
  .btn-accent{
    background: linear-gradient(135deg, #22d3ee 0%, #34d399 100%);
    color: #0b1221; border: none;
  }

  /* chips */
  .chip{margin:6px 6px 0 0; padding:6px 10px; border-radius:999px; display:inline-block; cursor:pointer;
        background: var(--vscode-editorWidget-background); border:1px solid var(--vscode-editorWidget-border)}

  .muted{opacity:.8}
  .conf-row{display:flex; justify-content:flex-end; align-items:center; gap:8px; margin-top:6px}
.badge{padding:2px 10px; border-radius:999px; font-size:12px;
  border:1px solid var(--vscode-editorWidget-border); user-select:none}
.badge.c1{background:#ffe4e6; color:#7f1d1d}   /* Low  */
.badge.c2{background:#fff1db; color:#7c2d12}   /* Fair */
.badge.c3{background:#fef9c3; color:#713f12}   /* Medium */
.badge.c4{background:#dcfce7; color:#14532d}   /* Good */
.badge.c5{background:#dbeafe; color:#1e3a8a}   /* High */

  /* output bubble */
  #output{ margin-top:12px }
  .bubble{
    white-space:pre-wrap; line-height:1.5;
    background: var(--vscode-editor-inactiveSelectionBackground, rgba(128,128,128,.15));
    border:1px dashed var(--vscode-editorWidget-border);
    border-radius:12px; padding:12px;
    min-height:40px;
  }
  /* typing dots */
  .typing{display:inline-flex; gap:6px; margin-left:6px; vertical-align:baseline}
  .typing span{width:6px; height:6px; border-radius:50%; background: currentColor; opacity:.4; animation: blink 1.2s infinite}
  .typing span:nth-child(2){animation-delay:.2s}
  .typing span:nth-child(3){animation-delay:.4s}
  @keyframes blink{0%,80%,100%{opacity:.2}40%{opacity:1}}

  /* divider */
  .divider{height:1px; background: var(--vscode-editorWidget-border); margin:12px 0}
`;

// panel.webview.html = `<!DOCTYPE html>
// <html>
// <head><meta charset="UTF-8"><style>${css}</style></head>
// <body>
//   <h2>Paper–Repo App</h2>

//   <button id="pickRepo">Select Repo Folder</button>
//   <div class="path" id="repoPath">(none)</div>

//   <button id="pickPaper">Select Paper (PDF)</button>
//   <div class="path" id="paperPath">(optional)</div>

//   <hr/>

//   <button id="buildIndex">Build Index</button>

  

//   <div class="field">
//     <input id="q" type="text" placeholder="Ask a question about the implementation..." />
//     <button id="ask">Ask</button>
//   </div>


// <!-- ▼ ADD: follow-ups container (appears after you ask) -->
// <div id="followups" class="suggest"></div>

//   <div id="suggest" style="margin-top:10px"></div>

//   <script>${api}
//     const el  = id => document.getElementById(id);
//     const set = (id,val) => el(id).textContent = val;

// el('ask').onclick = () => {
//   const q = (document.getElementById('q')?.value || '').trim();
//   if (q) vscode.postMessage({ type: 'ask', q });  // send the question to the extension
// };

//     // Render up to 5 prompt chips
//     const clearPrompts = () => {
//   const host = document.getElementById('suggest');
//   if (host) host.innerHTML = '';
//   // also clear persisted webview state if you use it
//   const st = (vscode.getState && vscode.getState()) || {};
//   if (vscode.setState) vscode.setState({ ...st, prompts: [] });
// };
//     const renderPrompts = (list=[]) => {
//       const host = el('suggest');
//       host.innerHTML = '';
//       if (!list.length) return;

//       const title = document.createElement('div');
//       title.className = 'muted';
//       title.textContent = 'Suggested:';
//       host.appendChild(title);

//       const wrap = document.createElement('div');
//       host.appendChild(wrap);

//       list.slice(0,5).forEach(p => {
//         const b = document.createElement('span');
//         b.className = 'chip';
//         b.textContent = p;
        
//         b.onclick = () => {
//           el('q').value = p;
//           clearPrompts();
//           vscode.postMessage({ type: 'ask', q: p });
//         };
//         wrap.appendChild(b);
//       });
//     };

//     // Button wiring
//     el('pickRepo').onclick   = () => vscode.postMessage({ type:'selectRepo' });
//     el('pickPaper').onclick  = () => vscode.postMessage({ type:'selectPaper' });
//     el('buildIndex').onclick = () => vscode.postMessage({ type:'buildIndex' });
//     el('ask').onclick        = () => {
//       const q = (el('q')?.value || '').trim();
//       clearPrompts(); // clear prompts when asking
//       if (q) vscode.postMessage({ type:'ask', q });
//     };

//     // Receive state
//     window.addEventListener('message', e => {
//       const m = e.data || {};
//       console.log('webview received', m); // debug
//       if (m.repoPath !== undefined)  set('repoPath',  m.repoPath);
//       if (m.paperPath !== undefined) set('paperPath', m.paperPath);
//       if (Array.isArray(m.prompts))  renderPrompts(m.prompts);  // REQUIRED
//     });
//     vscode.postMessage({ type: 'ready' });
//   // ...existing script...
//   const renderFollowups = (list = []) => {
//     const host = document.getElementById('followups');
//     const host2=document.getElementById('suggest');
//     host2.innerHTML=''; // clear suggested prompts when follow-ups arrive
//     host.innerHTML = '';
//     if (!list.length) return;
//     const title = document.createElement('div');
//     title.className = 'muted';
//     title.textContent = 'More like this:';
//     host.appendChild(title);
//     const wrap = document.createElement('div');
//     host.appendChild(wrap);
//     list.slice(0,5).forEach(p => {
//       const chip = document.createElement('span');
//       chip.className = 'chip';
//       chip.textContent = p;
//       chip.onclick = () => {
//         document.getElementById('q').value = p;
//         vscode.postMessage({ type: 'ask', q: p }); // ask again immediately
//       };
//       wrap.appendChild(chip);
//     });
//   };

//   // Receive messages from the extension
//   window.addEventListener('message', e => {
//     const m = e.data || {};
//     if (Array.isArray(m.followups)) renderFollowups(m.followups); // ▼ add this
//     // keep your other handlers (repoPath, paperPath, prompts, etc.)
//   });

//   </script>
// </body>
// </html>`;
panel.webview.html = `<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><style>${css}</style></head>
<body>
  <h2>AlignPaper2Repo</h2>

  <!-- Row: Select Repo + Select Paper side-by-side -->
  <div class="card">
    <div class="row">
      <div class="stack grow">
        <button id="pickRepo" class="btn">Select Repo Folder</button>
        <div class="sub" id="repoPath">(none)</div>
      </div>
      <div class="stack grow">
        <button id="pickPaper" class="btn">Select Paper (PDF)</button>
        <div class="sub" id="paperPath">(optional)</div>
      </div>
    </div>
  </div>

  <!-- Build Index button (full width) -->
  <div class="card" style="margin-top:12px">
    <button id="buildIndex" class="btn btn-accent">Build Index</button>
  </div>

  <!-- Ask row -->
  <div class="card" style="margin-top:12px">
   <div class="conf-row">
    <span class="muted">Alignment</span>
    <span id="confBadge" class="badge">–</span>
  </div>

    <div class="field">
      <input id="q" type="text" placeholder="Ask a question about the implementation..." />
      <button id="ask" class="btn btn-primary">Ask</button>
    </div>

    <!-- Streaming answer container -->
    <div id="output" class="bubble" style="margin-top:12px"></div>
  </div>

  <!-- Suggested prompts -->
  <div class="card" style="margin-top:12px">
    <div class="muted" style="margin-bottom:6px">Suggested:</div>
    <div id="suggest"></div>
  </div>

<script>${api}
function renderConfidenceBucket(n){
  const b = document.getElementById('confBadge');
  if (!b) return;
  const bucket = Math.max(1, Math.min(5, Math.round(Number(n)||1)));
  const label = ['','Low','Fair','Medium','Good','High'][bucket];
  b.className = 'badge c' + bucket;
  b.textContent = label;
}

// If you receive a raw 0..1 score, bucketize it
function scoreToBucket(s){
  if (s == null || isNaN(s)) return 1;
  // clamp 0..1 then map to 1..5
  s = Math.max(0, Math.min(1, Number(s)));
  return Math.max(1, Math.min(5, Math.ceil(s * 5)));
}

// Show "…" while thinking
function setConfidenceThinking(){
  const b = document.getElementById('confBadge');
  if (!b) return;
  b.className = 'badge';
  b.textContent = '…';
}
  const el  = id => document.getElementById(id);
  const set = (id,val) => el(id).textContent = val;

  // Buttons wired to extension
  el('pickRepo').onclick = () => vscode.postMessage({ type:'selectRepo' });
  el('pickPaper').onclick = () => vscode.postMessage({ type:'selectPaper' });
  el('buildIndex').onclick = () => vscode.postMessage({ type:'buildIndex' });

  // Ask + progressive output
  el('ask').onclick = () => {
    const q = (el('q')?.value || '').trim();
    if (!q) return;
    setConfidenceThinking();
    startTyping();
    vscode.postMessage({ type:'ask', q });
  };

  // Progressive text helpers (call stopTyping() then appendText(...))
  let typingTimer;
  function startTyping(){
    const out = el('output');
    out.innerHTML = 'Thinking<span class="typing"><span></span><span></span><span></span></span>';
  }
  function stopTyping(){ clearTimeout(typingTimer); }
  function setAnswer(text){ el('output').textContent = text || ''; }
  // If you stream partials from the extension, call appendText(...)
  function appendText(chunk){
    const out = el('output');
    out.textContent = (out.textContent || '') + chunk;
  }

  // Suggested prompts renderers
  const clearPrompts = () => {
    const host = el('suggest');
    if (host) host.innerHTML = '';
    const st = (vscode.getState && vscode.getState()) || {};
    if (vscode.setState) vscode.setState({ ...st, prompts: [] });
  };
  const renderPrompts = (list=[]) => {
    const host = el('suggest');
    host.innerHTML = '';
    if (!list.length) {
      host.innerHTML = '<span class="muted">None yet.</span>';
      return;
    }
    list.slice(0,5).forEach(p => {
      const b = document.createElement('span');
      b.className = 'chip';
      b.textContent = p;
      b.onclick = () => {
        el('q').value = p;
        clearPrompts();
        startTyping();
        vscode.postMessage({ type:'ask', q: p });
      };
      host.appendChild(b);
    });
  };

  // Receive updates from extension
  window.addEventListener('message', e => {
    const m = e.data || {};
    if (m.repoPath !== undefined) set('repoPath', m.repoPath);
    if (m.paperPath !== undefined) set('paperPath', m.paperPath);

    // final answer text (non-streaming)
    if (typeof m.reason_text === 'string') { stopTyping(); setAnswer(m.reason_text); }
    if (typeof m.score === 'number') renderConfidenceBucket(scoreToBucket(m.score));
    // streaming chunk (optional)
    //if (typeof m.answerChunk === 'string') { appendText(m.answerChunk); }

    // suggested prompts list
    if (Array.isArray(m.prompts)) renderPrompts(m.prompts);
  });
    window.addEventListener('message', e => {
    const m = e.data || {};
    if (Array.isArray(m.followups)) renderPrompts(m.followups); // ▼ add this
    // keep your other handlers (repoPath, paperPath, prompts, etc.)
  });
</script>
</body></html>`;

panel.webview.onDidReceiveMessage(async (msg) => {
  switch (msg?.type) {
    case 'ask': {
      const q = (msg?.q ?? '').toString().trim();
      if (!q) { vscode.window.showWarningMessage('Please enter a question.'); break; }
      await vscode.commands.executeCommand('paperRepo.askQuestionDirect', q); // ← DIRECT, no palette
      break;
    }
    // ... keep your other cases (selectRepo/selectPaper/buildIndex/ready) ...
  }
});
// const postState = async () => {
// panel.webview.postMessage({ repoPath: workspaceState.get<string>(KEY_REPO)||'(none)', paperPath: workspaceState.get<string>(KEY_PAPER)||'(optional)' });
// };
currentAppPanel = panel;
panel.onDidDispose(() => { currentAppPanel = undefined; });

const postState = async () => {
  const prompts = (await workspaceState.get<string[]>(KEY_PROMPTS)) || [];
  panel.webview.postMessage({
    repoPath: workspaceState.get<string>(KEY_REPO) || '(none)',
    paperPath: workspaceState.get<string>(KEY_PAPER) || '(optional)',
    prompts
  });
};
postState();


panel.webview.onDidReceiveMessage(async (msg) => {
switch (msg?.type) {
case 'selectRepo': {
const pick = await vscode.window.showOpenDialog({ canSelectFiles:false, canSelectFolders:true, canSelectMany:false, title:'Select Repo Folder' });
if (pick && pick[0]) { await workspaceState.update(KEY_REPO, pick[0].fsPath); panel.webview.postMessage({ repoPath: pick[0].fsPath }); }
break;
}
case 'selectPaper': {
const pick = await vscode.window.showOpenDialog({ canSelectFiles:true, canSelectFolders:false, canSelectMany:false, title:'Select Paper (PDF)', filters:{ PDF:['pdf'] } });
if (pick && pick[0]) { await workspaceState.update(KEY_PAPER, pick[0].fsPath); panel.webview.postMessage({ paperPath: pick[0].fsPath }); }
break;
}
case 'buildIndex': {
await vscode.commands.executeCommand('paperRepo.buildIndex');
break;
}
case 'ask': {
await vscode.commands.executeCommand('paperRepo.askQuestion');
break;
}
}
});
}));
subscriptions.push(
vscode.commands.registerCommand('paperRepo.setApiKey', async () => {
const key = await vscode.window.showInputBox({ prompt: 'Enter your OpenAI-compatible API key', password: true });
if (!key) return; await secrets.store('paperRepo.apiKey', key);
vscode.window.showInformationMessage('API key saved.');
})
);
subscriptions.push(
vscode.commands.registerCommand('paperRepo.setEmbeddingApiKey', async () => {
const key = await vscode.window.showInputBox({ prompt: 'Enter your  Embedding API key', password: true });
if (!key) return; await secrets.store('paperRepo.embeddingApiKey', key);
vscode.window.showInformationMessage('API key saved.');
})
);




subscriptions.push(
vscode.commands.registerCommand('paperRepo.clearEmbeddingApiKey', async () => { await secrets.delete('paperRepo.embeddingApiKey'); vscode.window.showInformationMessage('API key cleared.'); })
);

subscriptions.push(
vscode.commands.registerCommand('paperRepo.clearApiKey', async () => { await secrets.delete('paperRepo.apiKey'); vscode.window.showInformationMessage('API key cleared.'); })
);


subscriptions.push(
vscode.commands.registerCommand('paperRepo.selectPaper', async () => {
const pick = await vscode.window.showOpenDialog({ canSelectFiles: true, canSelectFolders: false, canSelectMany: false, title: 'Select Paper (PDF)', filters: { PDF: ['pdf'] } });
if (!pick || !pick[0]) return; await workspaceState.update(KEY_PAPER, pick[0].fsPath);
vscode.window.showInformationMessage(`Paper selected: ${pick[0].fsPath}`);
})
);
subscriptions.push(
vscode.commands.registerCommand('paperRepo.selectRepo', async () => {
const pick = await vscode.window.showOpenDialog({ canSelectFiles: false, canSelectFolders: true, canSelectMany: false, title: 'Select Repo Folder' });
if (!pick || !pick[0]) return; await workspaceState.update(KEY_REPO, pick[0].fsPath);
vscode.window.showInformationMessage(`Repo selected: ${pick[0].fsPath}`);
})
);

subscriptions.push(
  vscode.commands.registerCommand('paperRepo.runCsvAskMultiIndex', () => runCsvAsk_MultiIndex_buildDirect(context))
);

subscriptions.push(
vscode.commands.registerCommand('paperRepo.buildIndex', async () => {
try {
const repoPath = workspaceState.get<string>(KEY_REPO);
const paperPath = workspaceState.get<string>(KEY_PAPER);
if (!repoPath) { vscode.window.showErrorMessage('Pick a repo first.'); return; }
const cfg = getCfg();
const maxChunkLines = cfg.get<number>('maxChunkLines', 120);
const embeddingModel = cfg.get<string>('embeddingModel', 'text-embedding-3-small');
const apiBase = cfg.get<string>('apiBase', 'https://api.openai.com/v1');
const apiKey = await secrets.get('paperRepo.apiKey');
const embeddingApiKey = await secrets.get('paperRepo.embeddingApiKey') || apiKey;
const model          = cfg.get<string>('model', 'gpt-4o-mini');  // <-- define model here
vscode.window.showInformationMessage(`Building AST for repo: ${repoPath}`);
const index = await buildIndex({ repoPath, paperPath,  model, apiKey: apiKey,apiBase,promptStrategy:"zero_shot", maxCodeChars: 2000, });
await saveIndex(context, index);
// LLM-suggested prompts
let prompts: string[] = [];
if (apiKey) {
  try {
    prompts = await suggestQuestions(index, { apiKey, apiBase, model, paperPath });
  } catch (e: any) {
    vscode.window.showWarningMessage(`Prompt suggestion failed: ${e?.message || e}`);
  }
}

// If LLM returned nothing, provide a soft fallback so UI shows something
if (!prompts.length) {
  prompts = [
    "What problem does the paper solve?",
    "Where is the main model architecture implemented?",
    "Where is the loss/objective defined?",
    "How is data loaded and preprocessed?",
    "How do I run the main experiment?"
  ];
}

await workspaceState.update(KEY_PROMPTS, prompts);

// PUSH updates to the right-panel app if it’s open
if (currentAppPanel) {
  currentAppPanel.webview.postMessage({
    repoPath: workspaceState.get<string>(KEY_REPO) || '(none)',
    paperPath: workspaceState.get<string>(KEY_PAPER) || '(optional)',
    prompts
  });
}

vscode.window.showInformationMessage(
  `AST AND Natural Descriptions built. Suggested prompts: ${prompts.length}`
);


vscode.window.showInformationMessage('Index built successfully.');
} catch (err: any) { vscode.window.showErrorMessage(`Index error: ${err?.message || err}`); }
})
);

 async function parseCsv(buffer: any): Promise<any[]> {
  return new Promise((resolve, reject) => {
    try {
      const text = buffer.toString('utf-8');
      const records = parse(text, {
        columns: true,      // use first row as header
        skip_empty_lines: true,
        trim: true,         // trim whitespace
      });
      resolve(records);
    } catch (err) {
      reject(err);
    }
  });
}
function csvEscape(s: string){ if(s==null) return ''; const needs=/[",\n\r]/.test(String(s)); const out=String(s).replace(/"/g,'""'); return needs?`"${out}"`:out; }
function summarizeTo200Tokens(s: string){ if(!s) return ''; const words=s.replace(/\s+/g,' ').trim().split(' '); return words.slice(0,200).join(' '); }

// simple in-memory cache so the same (repo,paper) pair isn't rebuilt multiple times
const indexCache = new Map<string, any>();
async function runCsvAsk_MultiIndex_buildDirect(context: vscode.ExtensionContext) {
  //const time1=performance.now();
  // Pick CSV (must have: data, repopath, filepath)
  const picked = await vscode.window.showOpenDialog({
    canSelectMany: false,
    filters: { 'CSV': ['csv'], 'All files': ['*'] },
    title: 'Select CSV with columns: data, repopath, filepath',
    openLabel: 'Run (Build → Ask)',
  });
  if (!picked?.length) return;

  const inUri = picked[0];
  //vscode.window.showInformationMessage(`CSV selected: ${inUri.path}`);

  const rows = await parseCsv(await vscode.workspace.fs.readFile(inUri));
  //vscode.window.showInformationMessage(`CSV loaded: ${rows} rows.`);
  
  if (!rows.length) { vscode.window.showWarningMessage('CSV is empty.'); return; }
  const first = rows[0];
  if (!('data' in first) || !('repopath' in first) || !('filepath' in first)) {
    vscode.window.showErrorMessage('CSV must have headers: data, repopath, filepath'); return;
  }

  // Settings
  const cfg           = vscode.workspace.getConfiguration('paperRepo');
  const topK          = cfg.get<number>('topK', 8);
  const model         = cfg.get<string>('model', 'gpt-4o-mini');
  const apiBase       = cfg.get<string>('apiBase', 'https://api.openai.com/v1');
  const temperature   = cfg.get<number>('temperature', 1);
  const maxChunkLines = cfg.get<number>('maxChunkLines', 120);
  //vscode.window.showInformationMessage(`Config: model=${model}, topK=${apiBase}, temp=${temperature}, maxChunkLines=${maxChunkLines}`);
  
  // Embedding creds for buildIndex; fallback to same key as LLM if not separate
  const embedApiBase   = cfg.get<string>('embeddingApiBase', apiBase);
  const embeddingModel = cfg.get<string>('embeddingModel', 'text-embedding-3-small');
  //const embeddingModel = undefined;
  const apiKeyLLM      = await context.secrets.get('paperRepo.apiKey');
  const apiKeyEmbed    = await context.secrets.get('paperRepo.embeddingApiKey') || apiKeyLLM;
  //vscode.window.showInformationMessage(`Using API base: LLM ${apiBase}, Embedding model ${embeddingModel}`);
  
  if (!apiKeyLLM)   { vscode.window.showErrorMessage('No API key set (paperRepo.apiKey).'); return; }
  if (!apiKeyEmbed) { vscode.window.showErrorMessage('No embedding key set (paperRepo.embeddingApiKey or paperRepo.apiKey).'); return; }
  //vscode.window.showInformationMessage(`API keys: LLM ${apiKeyLLM?.slice(0,4)}..., Embedding ${apiKeyEmbed?.slice(0,4)}...`);
  // Clean rows & group by pair
  type Row = { data: string; repopath: string; filepath: string };
  
  const cleaned: Row[] = rows.map(r => ({
    data: String(r.data ?? '').trim(),
    repopath: String(r.repopath ?? '').trim(),
    filepath: String(r.filepath ?? '').trim(),
  })).filter(r => r.data && r.repopath);

   //vscode.window.showInformationMessage(`CSV cleaned: ${cleaned.length} valid rows (have data+repopath).`);
   //return;
  if (!cleaned.length) { vscode.window.showWarningMessage('No valid rows with non-empty data/repopath.'); return; }

  const groups = new Map<string, Row[]>();
  const makeKey = (repoPath: string, paperPath: string) => `${repoPath}||${paperPath}`;
  for (const r of cleaned) {
    const key = makeKey(r.repopath, r.filepath);
    (groups.get(key) ?? groups.set(key, []).get(key)!).push(r);
  }

  // Phase outputs
  const indexCache = new Map<string, any>();
  const outRows: Array<{ prompt:string; repopath:string; filepath:string; hitPath:string; file:string; startLine:number; endLine:number; summary:string; }> = [];

  const totalPairs = groups.size;
  const totalPrompts = cleaned.length;
  let built = 0, done = 0;

  await vscode.window.withProgress({
    title: 'CSV Pipeline: (1) Build Indexes → (2) Run Prompts',
    location: vscode.ProgressLocation.Notification,
    cancellable: false,
  }, async (progress) => {
    const time1 = performance.now();
    // ---------- Phase 1: Build all indexes (sequential) ----------
    for (const [key, items] of groups) {
      const [repoPath, paperPath] = key.split('||');
     // vscode.window.showInformationMessage(`Processing ${repoPath} | ${paperPath || '(no paper)'} with ${items.length} prompts...`);
      //progress.report({ message: `Build ${++built}/${totalPairs}: ${nodePath.basename(repoPath)} | ${nodePath.basename(paperPath || '')}` });
      try {
        vscode.window.showInformationMessage(`Building index for ${repoPath} `);
        const index = await buildIndex({
          repoPath,
          paperPath: paperPath || undefined,
          model,
          apiKey: apiKeyLLM,
          apiBase: embedApiBase,
          promptStrategy:"zero_shot",
          maxCodeChars: 2000,
        });
       await deleteIndex(context);
      await saveIndex(context,index);
      // vscode.window.showInformationMessage(`Index built for ${repoPath} | ${paperPath || '(no paper)'}`);
      } catch (e:any) {
              vscode.window.showInformationMessage(`Index failed built for ${repoPath} | ${paperPath || '(no paper)'}`);

        // console.error('buildIndex failed:', repoPath, paperPath, e);
        // // mark all rows for this pair as failed
        
        for (const r of items) {
          outRows.push({
            prompt: r.data, repopath: repoPath, filepath: paperPath,
            hitPath: '', file: '', startLine: 0, endLine: 0,
            summary: `Index error: ${e?.message ?? e}`,
          });
        }
      }
    

    // ---------- Phase 2: Run prompts (sequential) ----------
    
      const index = await loadIndex(context);
      if (!index) {
        // already recorded failures for this group
        done += items.length;
        progress.report({ message: `Skipped ${done}/${totalPrompts} (no index)` });
        continue;
      }

      for (const r of items) {
        progress.report({ message: `Ask ${done + 1}/${totalPrompts}: ${r.data.slice(0, 48)}...` });
        try {
          //const index=await loadIndex(context);
          if (!index) { 
            vscode.window.showWarningMessage('No index found. Run "Build Index" first.'); return; }

          const hit = await askWithLLM(index, {
            question: r.data.trim(), topK, model, apiKey: apiKeyLLM, apiBase, temperature
          });
         // return;
          const rawPath   = hit?.path ?? '';
          const startLine = (hit?.startLine ?? hit?.bestLine ?? 0) | 0;
          const endLine   = Number.isFinite(hit?.endLine) ? (hit!.endLine as number) : Math.max(startLine, startLine + 15);
          const summary   = hit?.reason || '';

          outRows.push({ prompt: r.data, repopath: repoPath, filepath: paperPath, hitPath: rawPath, file: rawPath, startLine, endLine, summary });
       
        } catch (e:any) {
          outRows.push({ prompt: r.data, repopath: repoPath, filepath: paperPath, hitPath: '', file: '', startLine: 0, endLine: 0, summary: `askWithLLM error: ${e?.message ?? e}` });
        } finally {
          done++; progress.report({ message: `Progress: ${done}/${totalPrompts}` });
        }
      }
     
  }
   const time2 = performance.now();
      vscode.window.showInformationMessage(`Time taken  ${(time2 - time1) / 1000} seconds`);
    
  });
  

  // Write results CSV
  const outUri = inUri.with({ path: inUri.path.replace(/\.csv$/i, '') + `.results.${Date.now()}.csv` });
  const header = 'prompt,repopath,filepath,hitPath,file,startLine,endLine,summary\n';
  const lines  = outRows.map(r => [
    csvEscape(r.prompt), csvEscape(r.repopath), csvEscape(r.filepath),
    csvEscape(r.hitPath), csvEscape(r.file), String(r.startLine), String(r.endLine), csvEscape(r.summary)
  ].join(','));
  await vscode.workspace.fs.writeFile(outUri, Buffer.from(header + lines.join('\n'), 'utf8'));
  //vscode.window.showInformationMessage(`Done: ${outRows.length} rows → ${outUri.path}`);
}

async function runAskDirect(ctx: vscode.ExtensionContext, q: string) {
  if (!q?.trim()) { vscode.window.showWarningMessage('Please enter a question.'); return; }
  const index = await loadIndex(ctx);
  if (!index) { vscode.window.showWarningMessage('No index found. Run "Build Index" first.'); return; }
 const paperPath = workspaceState.get<string>(KEY_PAPER);
  const cfg = vscode.workspace.getConfiguration('paperRepo');
  const topK        = cfg.get<number>('topK', 5);
  const model       = cfg.get<string>('model', 'gpt-4o-mini');
  const apiBase     = cfg.get<string>('apiBase', 'https://api.openai.com/v1');
  const temperature = cfg.get<number>('temperature', 1);
  const apiKey      = await ctx.secrets.get('paperRepo.apiKey');
  if (!apiKey) { vscode.window.showErrorMessage('No API key set.'); return; }
  vscode.window.showInformationMessage(`Asking: ${q.trim().slice(0, 64)}...`);
  const hit = await askWithLLM(index, { question: q.trim(), topK, model, apiKey, apiBase, temperature });
  if (!hit) { vscode.window.showWarningMessage('No relevant file found.'); return; }
  const repoPath = workspaceState.get<string>(KEY_REPO);
  if (!repoPath) {
  vscode.window.showErrorMessage("Repository path not set.");
  return;
}
  const uri = vscode.Uri.file(
    path.join(repoPath, hit.path)
  );
  vscode.window.showInformationMessage(`Opening ${uri.fsPath}`);
  const doc = await vscode.workspace.openTextDocument(uri);
  const editor = await vscode.window.showTextDocument(doc, {
    viewColumn: vscode.ViewColumn.Beside,  // don't replace the right panel
    preview: false,
    preserveFocus: false
  });
  // After opening the file successfully:
  if(currentAppPanel){
    const reason_text = hit.reason || '';
    currentAppPanel.webview.postMessage({ reason_text  });
    const score=hit.score || 0;
    currentAppPanel.webview.postMessage({ score });
  }
try {
  const cfg = vscode.workspace.getConfiguration('paperRepo');
  const model   = cfg.get<string>('model', 'gpt-4o-mini');
  const apiBase = cfg.get<string>('apiBase', 'https://api.openai.com/v1');
  const apiKey  = await context.secrets.get('paperRepo.apiKey') || '';

  if (apiKey) {
    const followups = await suggestFollowupsFromPrompt(q, { apiKey, apiBase, model ,paperPath});
    if (followups?.length && currentAppPanel) {
      currentAppPanel.webview.postMessage({ followups });
    }
  }
} catch (e) {
  console.warn('followups failed:', (e as any)?.message || e);
}

  const startLine = Math.max(0, (hit.startLine ?? hit.bestLine ?? 0));
  const endLine   = Math.min(doc.lineCount - 1, Math.max(startLine, (hit.endLine ?? startLine + 15)));
  const deco = vscode.window.createTextEditorDecorationType({ backgroundColor: new vscode.ThemeColor('editor.findMatchBackground') });
  editor.setDecorations(deco, [new vscode.Range(new vscode.Position(startLine,0), new vscode.Position(endLine,0))]);
  setTimeout(() => deco.dispose(), 15000);
}
subscriptions.push(
  vscode.commands.registerCommand('paperRepo.askQuestionDirect', async (qArg?: string) => {
    if (typeof qArg !== 'string' || !qArg.trim()) { vscode.window.showWarningMessage('No question from UI.'); return; }
    await runAskDirect(context, qArg);
  })
);

subscriptions.push(vscode.commands.registerCommand('paperRepo.debugShowPrompts', async () => {
  const prompts = (await workspaceState.get<string[]>(KEY_PROMPTS)) || [];
  //vscode.window.showInformationMessage(`Prompts: ${JSON.stringify(prompts)}`);
}));
}

export function deactivate() {}() => {}

