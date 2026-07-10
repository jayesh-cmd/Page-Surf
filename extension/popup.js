const API = "http://127.0.0.1:8000";
const DEFAULT_MODEL_A = "openai/gpt-4o";
const DEFAULT_MODEL_B = "anthropic/claude-sonnet-4.5";

// ── DOM refs ───────────────────────────────────────────────────────────────
const compareToggle  = document.getElementById("compareToggle");
const singlePanel    = document.getElementById("singlePanel");
const compareInputRow= document.getElementById("compareInputRow");
const comparePanel   = document.getElementById("comparePanel");

// Single mode
const userQuery  = document.getElementById("userQuery");
const modelSelect= document.getElementById("modelSelect");
const askBtn     = document.getElementById("askBtn");
const responseBox= document.getElementById("responseBox");

// Compare mode
const compareQuery   = document.getElementById("compareQuery");
const modelSelectA   = document.getElementById("modelSelectA");
const modelSelectB   = document.getElementById("modelSelectB");
const compareAskBtn  = document.getElementById("compareAskBtn");
const responseA      = document.getElementById("responseA");
const responseB      = document.getElementById("responseB");

let compareMode = false;

// ── Load models from backend ────────────────────────────────────────────────
async function loadModels() {
  try {
    const res  = await fetch(`${API}/models`);
    const data = await res.json();
    const models = data.models;

    [modelSelect, modelSelectA, modelSelectB].forEach((sel, i) => {
      sel.innerHTML = "";
      models.forEach(m => {
        const opt = document.createElement("option");
        opt.value = m.id;
        opt.textContent = m.name;
        sel.appendChild(opt);
      });
    });

    // Set defaults
    modelSelect.value  = DEFAULT_MODEL_A;
    modelSelectA.value = DEFAULT_MODEL_A;
    modelSelectB.value = DEFAULT_MODEL_B;

  } catch (err) {
    [modelSelect, modelSelectA, modelSelectB].forEach(sel => {
      sel.innerHTML = `<option value="gpt-4o">GPT-4o (offline)</option>`;
    });
    console.warn("Could not load models:", err.message);
  }
}

// ── Compare mode toggle ────────────────────────────────────────────────────
compareToggle.addEventListener("click", () => {
  compareMode = !compareMode;
  document.body.classList.toggle("compare-mode", compareMode);
  compareToggle.classList.toggle("active", compareMode);
  compareToggle.innerHTML = compareMode
    ? `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" width="13" height="13"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg> Exit`
    : `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" width="13" height="13"><rect x="2" y="3" width="9" height="18" rx="2"/><rect x="13" y="3" width="9" height="18" rx="2"/></svg> Compare`;
});

// ── Grab current page content ──────────────────────────────────────────────
function getPageContent() {
  return document.body.innerText;
}

async function extractPageContent() {
  return new Promise((resolve, reject) => {
    chrome.tabs.query({ active: true, currentWindow: true }, tabs => {
      chrome.scripting.executeScript(
        { target: { tabId: tabs[0].id }, function: getPageContent },
        results => {
          if (!results || !results[0]) {
            reject(new Error("Could not extract page content. Try reloading the tab."));
          } else {
            resolve({ text: results[0].result, tabId: tabs[0].id.toString() });
          }
        }
      );
    });
  });
}

// Strip markdown symbols but keep structure (bullets, line breaks)
function stripMarkdown(text) {
  return text
    .replace(/\*\*(.+?)\*\*/gs, '$1')   // **bold** → bold
    .replace(/\*(.+?)\*/gs, '$1')        // *italic* → italic
    .replace(/^#{1,6}\s+/gm, '')         // ## Heading → Heading
    .replace(/```[\s\S]*?```/g, '')       // remove code blocks
    .replace(/`([^`]+)`/g, '$1')         // `inline code` → inline code
    .replace(/^\s*>\s?/gm, '')           // remove blockquotes >
    .trim();
}

// ── Word-by-word stream animation ──────────────────────────────────────────
function streamText(text, container, delayMs = 35) {
  container.innerHTML = '<div class="response-text"></div>';
  const holder = container.querySelector(".response-text");
  const tokens = text.split(/(\s+)/);
  let i = 0;

  function next() {
    if (i >= tokens.length) return;
    const token = tokens[i++];
    if (token.trim() === "") {
      holder.appendChild(document.createTextNode(token));
    } else {
      const span = document.createElement("span");
      span.className = "word";
      span.style.animationDelay = `0ms`;
      span.innerText = token;
      holder.appendChild(span);
    }
    container.scrollTop = container.scrollHeight;
    setTimeout(next, delayMs);
  }
  next();
}

function setLoading(container, msg = "Thinking…") {
  container.innerHTML = `<div style="color:var(--text-dim);text-align:center;margin-top:60px;display:flex;flex-direction:column;align-items:center;gap:12px;font-size:12.5px;">
    <span class="spinner"></span>${msg}
  </div>`;
}

function showError(container, msg) {
  container.innerHTML = `<div class="error-message">⚠️ ${msg}</div>`;
}

// ── Single mode – Ask ───────────────────────────────────────────────────────
askBtn.addEventListener("click", handleSingleAsk);
userQuery.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); askBtn.click(); }
});

async function handleSingleAsk() {
  const query = userQuery.value.trim();
  if (!query) return;

  askBtn.disabled = true;
  askBtn.innerHTML = '<span class="spinner"></span>Thinking…';
  setLoading(responseBox);

  try {
    const { text, tabId } = await extractPageContent();
    const res = await fetch(`${API}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        query,
        session_id: tabId,
        model: modelSelect.value || DEFAULT_MODEL_A
      })
    });

    if (!res.ok) throw new Error(`Server error ${res.status}`);
    const data = await res.json();
    streamText(stripMarkdown(data.answer), responseBox);

  } catch (err) {
    showError(responseBox, `${err.message}. Is the Python backend running on port 8000?`);
  } finally {
    askBtn.disabled = false;
    askBtn.textContent = "Ask";
  }
}

// ── Compare mode – Ask ──────────────────────────────────────────────────────
compareAskBtn.addEventListener("click", handleCompareAsk);
compareQuery.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); compareAskBtn.click(); }
});

async function handleCompareAsk() {
  const query = compareQuery.value.trim();
  if (!query) return;

  compareAskBtn.disabled = true;
  compareAskBtn.innerHTML = '<span class="spinner"></span>Comparing…';

  const mA = modelSelectA.value || DEFAULT_MODEL_A;
  const mB = modelSelectB.value || DEFAULT_MODEL_B;

  setLoading(responseA, `Asking ${getModelName(mA)}…`);
  setLoading(responseB, `Asking ${getModelName(mB)}…`);

  try {
    const { text, tabId } = await extractPageContent();
    const res = await fetch(`${API}/compare`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        query,
        session_id: tabId,
        model_a: mA,
        model_b: mB
      })
    });

    if (!res.ok) throw new Error(`Server error ${res.status}`);
    const data = await res.json();

    // Stream both responses simultaneously
    streamText(stripMarkdown(data.answer_a), responseA, 30);
    streamText(stripMarkdown(data.answer_b), responseB, 30);

  } catch (err) {
    showError(responseA, err.message);
    showError(responseB, `${err.message}. Is the Python backend running on port 8000?`);
  } finally {
    compareAskBtn.disabled = false;
    compareAskBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="3" width="9" height="18" rx="2"/><rect x="13" y="3" width="9" height="18" rx="2"/></svg> Compare Models`;
  }
}

// ── Helper: get model display name from id ─────────────────────────────────
function getModelName(id) {
  const opt = [...modelSelectA.options, ...modelSelectB.options].find(o => o.value === id);
  return opt ? opt.textContent : id;
}

// ── Init ───────────────────────────────────────────────────────────────────
loadModels();
