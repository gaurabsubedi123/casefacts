// Model Portal — the page. Plain JS, no libraries, nothing loaded from the
// internet: it has to work on a machine that is offline apart from Ollama.
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const GB = 1024 ** 3;
const fmtSize = (n) => (!n ? "" : n >= GB ? `${(n / GB).toFixed(1)} GB` : `${Math.round(n / 1024 ** 2)} MB`);
const fmtSiteSize = (n) => (!n ? "?" : n >= 1e9 ? `${(n / 1e9).toFixed(1)} GB` : `${Math.round(n / 1e6)} MB`);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const store = {
  get(key, fallback) { try { const v = localStorage.getItem(key); return v === null ? fallback : JSON.parse(v); } catch { return fallback; } },
  set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* private window */ } },
};

async function api(path, options = {}) {
  const init = { ...options };
  if (init.json !== undefined) {
    init.method = init.method || "POST";
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(init.json);
    delete init.json;
  }
  let response;
  try {
    response = await fetch(path, init);
  } catch {
    throw new Error("The portal is not responding. Is its window still open?");
  }
  let data = {};
  try { data = await response.json(); } catch { /* empty body */ }
  if (!response.ok) {
    const error = new Error(data.error || `${response.status} ${response.statusText}`);
    error.data = data;
    throw error;
  }
  return data;
}

async function pollJob(id, onUpdate, every = 700) {
  for (;;) {
    const job = await api(`/api/jobs/${id}`);
    onUpdate && onUpdate(job);
    if (job.status !== "running") return job;
    await sleep(every);
  }
}

const state = {
  status: null,
  models: [],
  docs: [],
  selected: new Set(store.get("selectedDocs", [])),
  source: "ollama",
  askJob: null,
  watchingDownloads: false,
};

// ------------------------------------------------------------------ tabs

function showTab(name) {
  $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  $$(".panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${name}`));
  store.set("tab", name);
  if (location.hash !== `#${name}`) history.replaceState(null, "", `#${name}`);
  if (name === "models") {
    loadModels();
    loadRecommended();
    if (!$("#results").innerHTML.trim()) search();
  }
  if (name === "documents") loadDocs();
}
$$(".tab").forEach((t) => t.addEventListener("click", () => showTab(t.dataset.tab)));
document.addEventListener("click", (e) => {
  const link = e.target.closest("[data-goto]");
  if (link) { e.preventDefault(); showTab(link.dataset.goto); }
});

// ---------------------------------------------------------------- status

async function refreshStatus() {
  let s;
  try { s = await api("/api/status"); } catch (e) { setBanner(esc(e.message)); return; }
  state.status = s;
  const pill = $("#ollama-pill");
  if (s.ollama.up) {
    pill.textContent = `Ollama ${s.ollama.version || ""} running`;
    pill.className = "pill good";
    setBanner("");
  } else {
    pill.textContent = "Ollama not running";
    pill.className = "pill bad";
    if (s.ollama.installed) {
      setBanner('Ollama is installed but not running. <button class="primary small-btn" id="start-ollama">Start Ollama</button>');
      $("#start-ollama").onclick = startOllama;
    } else {
      const install = s.ollama.installing?.[0];
      if (install && install.status === "running") {
        const p = install.progress || {};
        setBanner(`<span class="spin"></span>Downloading the Ollama installer… ${p.total ? Math.floor(100 * p.completed / p.total) + "%" : ""}`);
      } else if (install && install.status === "done") {
        setBanner("The Ollama installer is open — click through it. This page notices when Ollama is running.");
      } else if (s.ollama.platform === "Windows") {
        setBanner(`Ollama is not installed on this computer. It is the free program that downloads and runs the models.
          <button class="primary small-btn" id="install-ollama">Install Ollama</button>
          ${install?.status === "error" ? `<br><small>${esc(install.error)} — or <a href="${esc(s.ollama.download)}" target="_blank" rel="noopener">download it yourself</a>.</small>` : ""}`);
        $("#install-ollama").onclick = installOllama;
      } else {
        const linux = s.ollama.platform === "Linux"
          ? ' or run <code>curl -fsSL https://ollama.com/install.sh | sh</code>' : "";
        setBanner(`Ollama is not installed on this computer. It is the free program that downloads and runs the models.
          <a href="${esc(s.ollama.download)}" target="_blank" rel="noopener">Download Ollama</a>${linux}, then reload this page.`);
      }
    }
  }
  const m = s.machine;
  const machinePill = $("#machine-pill");
  machinePill.textContent = m.summary;
  machinePill.title = `${m.cpu} · ${m.os}`;
  renderMachine(m);
  if (s.ollama.up && !state.models.length) await loadModels(true);
  renderModelPicker();
  renderSetup();
}

function renderSetup() {
  const s = state.status;
  if (!s) return;
  const up = s.ollama.up;
  const hasModel = chatModels().length > 0;
  const plugged = !!s.active_model && chatModels().some((m) => m.name === s.active_model);
  const hasDocs = state.docs.length > 0;
  const steps = [
    [up, up ? "Ollama is running." : "Install or start Ollama (see the message at the top)."],
    [hasModel, 'Download a model on the <a href="#" data-goto="models">Models</a> tab — the <b>Recommended</b> list shows what runs well on this computer.'],
    [plugged, "Plug a model in: pick it in the <b>Model</b> box at the top right."],
    [hasDocs, 'Add OCR\'d PDFs or text files on the <a href="#" data-goto="documents">Documents</a> tab.'],
  ];
  $("#steps").innerHTML = steps.map(([done, text]) => `<li class="${done ? "done" : ""}">${text}</li>`).join("");
  $("#setup").hidden = steps.every(([done]) => done);
}

function setBanner(html) {
  const banner = $("#banner");
  banner.innerHTML = html;
  banner.hidden = !html;
}

async function installOllama() {
  try {
    await api("/api/ollama/install", { json: {} });
  } catch (e) { setBanner(esc(e.message)); return; }
  refreshStatus();
  const timer = setInterval(async () => {
    await refreshStatus();
    if (state.status?.ollama.up) clearInterval(timer);
  }, 2000);
}

async function startOllama() {
  setBanner('<span class="spin"></span>Starting Ollama…');
  try {
    const r = await api("/api/ollama/start", { json: {} });
    setBanner("");
    toastOk(r.message);
  } catch (e) {
    setBanner(esc(e.message));
  }
  refreshStatus();
}

function toastOk(message) {
  const box = document.createElement("div");
  box.className = "okay";
  box.style.cssText = "position:fixed;right:16px;bottom:16px;z-index:9;max-width:420px";
  box.textContent = message;
  document.body.append(box);
  setTimeout(() => box.remove(), 4000);
}

function renderMachine(m) {
  const gpus = m.gpus.length
    ? m.gpus.map((g) => `${esc(g.name)} — ${fmtSize(g.vram_total)}${g.unified ? " (shared with RAM)" : ""}${g.usable ? "" : ' <span class="muted">(not used by Ollama)</span>'}`).join("<br>")
    : '<span class="muted">none found — models will run on the CPU</span>';
  $("#machine").innerHTML = `
    <dl class="kv">
      <dt>GPU</dt><dd>${gpus}</dd>
      <dt>GPU memory for models</dt><dd>${m.gpu_memory ? fmtSize(m.gpu_memory) : "—"}</dd>
      <dt>RAM</dt><dd>${fmtSize(m.ram_total)}${m.ram_free ? ` (${fmtSize(m.ram_free)} free)` : ""}</dd>
      <dt>CPU</dt><dd>${esc(m.cpu)} · ${m.cpu_cores} threads</dd>
      <dt>Disk free for models</dt><dd>${m.disk_free ? fmtSize(m.disk_free) : "?"} <span class="muted small">${esc(m.models_dir)}</span></dd>
    </dl>
    <p class="muted small">Detected automatically each time the portal starts. Every download is checked against these numbers.</p>`;
}

// ---------------------------------------------------------------- models

function chatModels() {
  return state.models.filter((m) => !m.embedding && !m.cloud);
}

function renderModelPicker() {
  const select = $("#active-model");
  const active = state.status?.active_model || "";
  const options = chatModels().map((m) => `<option value="${esc(m.name)}" ${m.name === active ? "selected" : ""}>${esc(m.name)}</option>`);
  select.innerHTML = `<option value="">${options.length ? "choose a model" : "none downloaded"}</option>${options.join("")}`;
  if (active && !chatModels().some((m) => m.name === active)) select.value = "";
}

$("#active-model").addEventListener("change", (e) => { if (e.target.value) useModel(e.target.value); });

async function loadModels(quiet) {
  try {
    const data = await api("/api/models");
    state.models = data.models;
    $("#meaning-toggle").checked = data.meaning_search;
    renderInstalled(data);
    renderModelPicker();
    renderSetup();
  } catch (e) {
    if (!quiet) $("#installed").innerHTML = `<div class="error">${esc(e.message)}</div>`;
  }
  watchDownloads();
}

function fitBadge(fit) {
  if (!fit) return '<span class="badge plain">size unknown</span>';
  return `<span class="badge ${esc(fit.level)}" title="${esc(fit.message)}">${esc(fit.label)}</span>`;
}

function renderInstalled(data) {
  const box = $("#installed");
  if (!state.models.length) {
    box.innerHTML = '<p class="muted">No models downloaded yet. Find one on the right — <b>qwen2.5:7b</b> is a good first choice for questions about documents.</p>';
    return;
  }
  box.innerHTML = state.models.map((m) => {
    const loaded = m.loaded ? ` · loaded, ${m.loaded.gpu_share}% on GPU` : "";
    let actions = "";
    if (m.cloud) actions = '<span class="badge no" title="Runs on Ollama\'s servers — documents would leave this computer">cloud — not used</span>';
    else if (m.embedding) actions = m.embed_in_use ? '<span class="badge verified">used for search</span>' : '<span class="badge plain">embedding</span>';
    else if (m.active) actions = '<span class="badge verified">in use</span>';
    else actions = `<button class="primary small-btn" data-use="${esc(m.name)}">Use</button>`;
    return `<div class="model-row">
      <div>
        <div class="model-name">${esc(m.name)}</div>
        <div class="model-sub">${fmtSize(m.size)} · ${esc(m.parameters)} ${esc(m.quantization)}${m.embedding ? "" : ` · ${Number(m.context).toLocaleString()}-token context`}${loaded}</div>
        ${m.embedding || m.cloud ? "" : `<div class="model-sub">${fitBadge(m.fit)} ${esc(m.fit?.level === "gpu" ? "" : m.fit?.message || "")}</div>`}
      </div>
      <div class="model-actions">${actions}<button class="ghost small-btn danger" data-delete="${esc(m.name)}">Delete</button></div>
    </div>`;
  }).join("");
  $$("[data-use]", box).forEach((b) => (b.onclick = () => useModel(b.dataset.use)));
  $$("[data-delete]", box).forEach((b) => (b.onclick = () => deleteModel(b.dataset.delete)));
}

$("#refresh-models").onclick = () => loadModels();
$("#meaning-toggle").onchange = async (e) => { await api("/api/models/meaning", { json: { on: e.target.checked } }); loadModels(); };

async function useModel(name) {
  const pill = $("#ollama-pill");
  const select = $("#active-model");
  select.disabled = true;
  pill.innerHTML = `<span class="spin"></span>Plugging in ${esc(name)}…`;
  pill.className = "pill";
  try {
    const { job } = await api("/api/models/use", { json: { name } });
    const done = await pollJob(job, (j) => { if (j.progress.phase) pill.innerHTML = `<span class="spin"></span>${esc(j.progress.phase)}`; });
    if (done.status === "error") throw new Error(done.error);
    const r = done.result;
    toastOk(`${r.model} is ready — ${r.gpu_share}% on GPU, ${r.context.toLocaleString()}-token context.`);
    if (r.note) setBanner(esc(r.note));
  } catch (e) {
    setBanner(esc(e.message));
  } finally {
    select.disabled = false;
    await refreshStatus();
    await loadModels(true);
  }
}

async function deleteModel(name) {
  if (!confirm(`Delete ${name} from this computer? You can download it again later.`)) return;
  try {
    await api("/api/models/delete", { json: { name } });
  } catch (e) { alert(e.message); }
  await loadModels();
  refreshStatus();
}

// ----------------------------------------------------------- recommended

let recommendedLoaded = false;
async function loadRecommended(force) {
  if (recommendedLoaded && !force) return;
  recommendedLoaded = true;
  const box = $("#recommended");
  try {
    const data = await api("/api/recommended");
    const gpu = state.status?.machine?.gpu_memory;
    $("#rec-note").textContent = gpu
      ? `The largest size of each that runs entirely on this computer's GPU (${data.machine}).`
      : `No usable GPU was found, so these are small sizes that stay usable on the CPU (${data.machine}).`;
    box.innerHTML = data.picks.map((p) => `
      <div class="model-row">
        <div>
          <div class="model-name">${esc(p.name)}</div>
          <div class="model-sub">${esc(p.description)}</div>
          <div class="model-sub">${fmtSiteSize(p.size)} · ${fitBadge(p.fit)}</div>
        </div>
        <div class="model-actions">${p.installed ? '<span class="badge verified">downloaded</span>'
          : `<button class="primary small-btn" data-rec="${esc(p.name)}">Download</button>`}</div>
      </div>`).join("");
    $$("[data-rec]", box).forEach((b) => (b.onclick = () => {
      const p = data.picks.find((x) => x.name === b.dataset.rec);
      download(p.name, p.size, p.fit);
    }));
  } catch (e) {
    recommendedLoaded = false;
    box.innerHTML = `<div class="error">${esc(e.message)}</div>`;
  }
}

// --------------------------------------------------------------- catalog

$$("#source button").forEach((b) => b.addEventListener("click", () => {
  state.source = b.dataset.source;
  $$("#source button").forEach((x) => x.classList.toggle("on", x === b));
  $("#search-q").placeholder = state.source === "hf" ? "e.g. qwen2.5 7b instruct" : "qwen, deepseek, llama…";
  $("#results").innerHTML = "";
  $("#search-note").textContent = "";
  if (state.source === "ollama") search();
}));

$("#search-form").addEventListener("submit", (e) => { e.preventDefault(); search(); });

async function search() {
  const q = $("#search-q").value.trim();
  const results = $("#results");
  results.innerHTML = '<p class="muted"><span class="spin"></span>Searching…</p>';
  try {
    const data = await api(`/api/catalog/search?source=${state.source}&q=${encodeURIComponent(q)}`);
    $("#search-note").textContent = data.warning || "";
    if (!data.results.length) { results.innerHTML = '<p class="muted">Nothing found.</p>'; return; }
    results.innerHTML = data.results.map((r) => `
      <div class="result" data-name="${esc(r.name)}">
        <div class="result-head">
          <span class="model-name">${esc(r.name)}</span>
          <span class="muted small">${r.cloud ? '<span class="badge no">cloud only</span> ' : ""}${esc(r.pulls ? `${r.pulls} pulls` : r.downloads ? `${r.downloads.toLocaleString()} downloads` : "")}</span>
        </div>
        <div class="model-sub">${esc(r.description || "")}</div>
        ${r.sizes?.length ? `<div class="model-sub">Sizes: ${r.sizes.map(esc).join(", ")}${r.capabilities?.length ? ` · ${r.capabilities.map(esc).join(", ")}` : ""}</div>` : ""}
        <div class="tags" hidden></div>
      </div>`).join("");
    $$(".result-head", results).forEach((h) => h.addEventListener("click", () => toggleTags(h.parentElement)));
  } catch (e) {
    results.innerHTML = `<div class="error">${esc(e.message)}</div>`;
  }
}

async function toggleTags(result) {
  const box = $(".tags", result);
  if (!box.hidden) { box.hidden = true; return; }
  box.hidden = false;
  box.innerHTML = '<p class="muted small"><span class="spin"></span>Reading the available versions…</p>';
  const name = result.dataset.name;
  try {
    const data = state.source === "hf"
      ? await api(`/api/catalog/hf?repo=${encodeURIComponent(name)}`)
      : await api(`/api/catalog/tags?name=${encodeURIComponent(name)}`);
    renderTags(box, data.tags, false);
  } catch (e) {
    box.innerHTML = `<div class="error">${esc(e.message)}</div>`;
  }
}

function renderTags(box, tags, showAll) {
  const main = tags.filter((t) => !t.variant);
  const shown = showAll || !main.length ? tags : main;
  const hidden = tags.length - shown.length;
  box.innerHTML = shown.map((t) => `
    <div class="tag-row">
      <span class="model-name" title="${esc(t.name)}">${esc(t.tag)}${t.latest ? ' <span class="badge plain">latest</span>' : ""}</span>
      <span>${fmtSiteSize(t.size)}</span>
      <span>${fitBadge(t.fit)}</span>
      <span>${t.installed ? '<span class="badge verified">downloaded</span>' : `<button class="primary small-btn" data-pull="${esc(t.name)}" data-size="${t.size || 0}">Download</button>`}</span>
      ${t.fit && t.fit.level !== "gpu" ? `<span class="fitmsg">${esc(t.fit.message)}</span>` : ""}
      ${t.fit?.disk_warning ? `<span class="fitmsg" style="color:var(--bad)">${esc(t.fit.disk_warning)}</span>` : ""}
    </div>`).join("") + (hidden ? `<button class="ghost small-btn" data-all>Show ${hidden} other quantizations</button>` : "");
  $$("[data-pull]", box).forEach((b) => (b.onclick = () => {
    const tag = tags.find((t) => t.name === b.dataset.pull);
    download(tag.name, tag.size, tag.fit);
  }));
  const all = $("[data-all]", box);
  if (all) all.onclick = () => renderTags(box, tags, true);
}

$("#pull-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const name = $("#pull-name").value.trim();
  if (name) download(name, 0, null);
});

async function download(name, size, fit) {
  if (fit?.disk_warning) { alert(fit.disk_warning); return; }
  if (fit && fit.level === "no" && !confirm(`${name} is too big for this computer.\n\n${fit.message}\n\nDownload it anyway?`)) return;
  if (fit && (fit.level === "partial" || fit.level === "cpu") && !confirm(`${name}: ${fit.label}\n\n${fit.message}\n\nDownload anyway?`)) return;
  if (!fit && !size && !confirm(`The size of ${name} is not known in advance, so it cannot be checked against this computer first. Download it?`)) return;
  try {
    await api("/api/pull", { json: { name, size, force: fit?.level === "no" } });
    watchDownloads();
  } catch (e) { alert(e.message); }
}

async function watchDownloads() {
  if (state.watchingDownloads) return;
  state.watchingDownloads = true;
  try {
    for (;;) {
      const { jobs } = await api("/api/jobs?kind=pull");
      renderDownloads(jobs);
      if (!jobs.some((j) => j.status === "running")) break;
      await sleep(1200);
    }
  } catch { /* shown next time */ }
  state.watchingDownloads = false;
  const models = await api("/api/models").catch(() => null);
  // A first model, just downloaded, with nothing plugged in: plug it in, so
  // the person can go straight to asking.
  if (models && !state.status?.active_model) {
    const first = models.models.find((m) => !m.embedding && !m.cloud && m.fit?.level !== "no");
    if (first) { state.models = models.models; useModel(first.name); }
  }
  if (models) { state.models = models.models; renderInstalled(models); renderModelPicker(); renderSetup(); }
  if (recommendedLoaded) loadRecommended(true);
}

function renderDownloads(jobs) {
  $("#downloads-card").hidden = !jobs.length;
  $("#downloads").innerHTML = jobs.slice().reverse().map((j) => {
    const p = j.progress || {};
    const pct = p.total ? Math.floor((100 * p.completed) / p.total) : 0;
    let line;
    if (j.status === "running") line = `${esc(p.status || "starting")} ${p.total ? `— ${fmtSize(p.completed)} of ${fmtSize(p.total)} (${pct}%)` : ""}`;
    else if (j.status === "done") line = '<span style="color:var(--good)">Downloaded. Press <b>Use</b> on it above to plug it in.</span>';
    else if (j.status === "stopped") line = "Stopped. Download again to resume where it left off.";
    else line = `<span style="color:var(--bad)">${esc(j.error)}</span>`;
    return `<div class="result">
      <div class="result-head"><span class="model-name">${esc(j.label)}</span>
        ${j.status === "running" ? `<button class="ghost small-btn" data-stop="${j.id}">Stop</button>` : ""}</div>
      ${j.status === "running" ? `<div class="bar"><span style="width:${pct}%"></span></div>` : ""}
      <div class="model-sub">${line}</div></div>`;
  }).join("");
  $$("[data-stop]").forEach((b) => (b.onclick = () => api(`/api/jobs/${b.dataset.stop}/stop`, { json: {} })));
}

// ------------------------------------------------------------- documents

async function loadDocs() {
  try {
    const data = await api("/api/documents");
    state.docs = data.documents;
  } catch (e) {
    $("#library").innerHTML = `<div class="error">${esc(e.message)}</div>`;
    return;
  }
  for (const id of [...state.selected]) if (!state.docs.some((d) => d.id === id)) state.selected.delete(id);
  store.set("selectedDocs", [...state.selected]);
  renderLibrary();
  renderAskDocs();
  renderSetup();
}

function renderLibrary() {
  const box = $("#library");
  if (!state.docs.length) { box.innerHTML = '<p class="muted">Nothing added yet.</p>'; return; }
  box.innerHTML = state.docs.map((d) => `
    <div class="doc-row">
      <div>
        <div class="model-name">${esc(d.name)}</div>
        <div class="model-sub">${d.pages} page${d.pages === 1 ? "" : "s"} · ${d.chars.toLocaleString()} characters of text · ${esc(d.kind.toUpperCase())}
          ${d.empty_pages ? ` · <span style="color:var(--warn)">${d.empty_pages} page(s) without text</span>` : ""}</div>
      </div>
      <div class="model-actions">
        <a class="button ghost small-btn" href="/api/documents/${d.id}/file" target="_blank" rel="noopener">Open</a>
        <button class="ghost small-btn danger" data-remove="${d.id}">Remove</button>
      </div>
    </div>`).join("");
  $$("[data-remove]", box).forEach((b) => (b.onclick = async () => {
    if (!confirm("Remove this document from the portal? The original file is not touched.")) return;
    await api(`/api/documents/${b.dataset.remove}`, { method: "DELETE" });
    loadDocs();
  }));
}

function renderAskDocs() {
  const list = $("#ask-docs");
  $("#ask-docs-empty").hidden = state.docs.length > 0;
  list.innerHTML = state.docs.map((d) => `
    <li><input type="checkbox" id="d-${d.id}" data-doc="${d.id}" ${state.selected.has(d.id) ? "checked" : ""}>
      <label for="d-${d.id}">${esc(d.name)} <span class="muted small">(${d.pages} p.)</span></label></li>`).join("");
  $$("[data-doc]", list).forEach((c) => (c.onchange = () => {
    c.checked ? state.selected.add(c.dataset.doc) : state.selected.delete(c.dataset.doc);
    store.set("selectedDocs", [...state.selected]);
    $("#select-all").checked = state.selected.size === state.docs.length;
  }));
  $("#select-all").checked = state.docs.length > 0 && state.selected.size === state.docs.length;
}

$("#select-all").onchange = (e) => {
  state.selected = new Set(e.target.checked ? state.docs.map((d) => d.id) : []);
  store.set("selectedDocs", [...state.selected]);
  renderAskDocs();
};

const drop = $("#drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => upload([...e.dataTransfer.files]));
$("#file-input").onchange = (e) => { upload([...e.target.files]); e.target.value = ""; };
$("#folder-input").onchange = (e) => {
  const files = [...e.target.files].filter((f) => /\.(pdf|txt|md)$/i.test(f.name));
  e.target.value = "";
  if (!files.length) { showAddErrors([{ name: "That folder", error: "has no PDF or text files in it." }]); return; }
  upload(files);
};

function showAddErrors(errors) {
  $("#add-errors").innerHTML = errors.map((x) => `<div class="error"><b>${esc(x.name)}</b>: ${esc(x.error.replace(`${x.name}: `, ""))}</div>`).join("");
}

async function upload(files) {
  showAddErrors([]);
  const progress = $("#add-progress");
  if (!files.length) { showAddErrors([{ name: "No file", error: "was given. Choose one or more OCR'd PDF or text files." }]); return; }
  const form = new FormData();
  files.forEach((f) => form.append("files", f, f.name));
  progress.innerHTML = `<span class="spin"></span>Copying ${files.length} file(s) in…`;
  try {
    const { job, rejected } = await api("/api/documents", { method: "POST", body: form });
    if (rejected?.length) showAddErrors(rejected);
    const done = await pollJob(job, (j) => {
      const p = j.progress;
      if (p.phase) progress.innerHTML = `<span class="spin"></span>${esc(p.phase)} (file ${p.file} of ${p.files})${p.pages ? ` — page ${p.page} of ${p.pages}` : ""}`;
    });
    if (done.status === "error") throw new Error(done.error);
    const r = done.result;
    const added = r.added.filter((d) => d.new).length;
    const again = r.added.length - added;
    progress.innerHTML = r.added.length
      ? `<div class="okay">Added ${added} document(s)${again ? `; ${again} already in the library` : ""}.</div>` +
        r.added.filter((d) => d.warning).map((d) => `<div class="note"><b>${esc(d.name)}</b>: ${esc(d.warning)}</div>`).join("")
      : "";
    showAddErrors(r.skipped);
    r.added.forEach((d) => state.selected.add(d.id));
    store.set("selectedDocs", [...state.selected]);
  } catch (e) {
    progress.textContent = "";
    showAddErrors(e.data?.rejected?.length ? e.data.rejected : [{ name: "Could not add", error: e.message }]);
  }
  loadDocs();
}

// ------------------------------------------------------------------- ask

$("#ask-form").addEventListener("submit", (e) => { e.preventDefault(); askQuestion(); });
$("#question").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) askQuestion(); });
$("#stop-btn").onclick = () => state.askJob && api(`/api/jobs/${state.askJob}/stop`, { json: {} });

function askError(message) {
  const box = $("#ask-error");
  box.textContent = message || "";
  box.hidden = !message;
}

async function askQuestion() {
  askError("");
  const question = $("#question").value.trim();
  const model = $("#active-model").value;
  // The same checks the server makes, here so the message is instant.
  if (!state.docs.length) return askError("No document is attached. Add an OCR'd PDF or text file on the Documents tab first.");
  if (!state.selected.size) return askError("No document is selected. Tick at least one document above.");
  if (!model) return askError("No model is plugged in. Choose one at the top right, or download one on the Models tab.");
  if (!question) return askError("Type a question first.");

  const button = $("#ask-btn");
  const progress = $("#ask-progress");
  button.disabled = true;
  $("#stop-btn").hidden = false;
  progress.innerHTML = '<span class="spin"></span>Starting…';
  try {
    const { job } = await api("/api/ask", { json: { question, model, documents: [...state.selected] } });
    state.askJob = job;
    const done = await pollJob(job, (j) => {
      const p = j.progress;
      let text = p.phase || "working";
      if (p.phase === "reading") text = `Reading ${p.excerpts} excerpt(s)`;
      if (p.phase === "thinking") text = "Thinking";
      if (p.phase === "writing") text = "Writing the answer";
      progress.innerHTML = `<span class="spin"></span>${esc(text)} · ${j.elapsed}s`;
    }, 600);
    if (done.status === "stopped") { progress.textContent = "Stopped."; return; }
    if (done.status === "error") throw new Error(done.error);
    progress.textContent = "";
    renderAnswer(done.result);
  } catch (e) {
    progress.textContent = "";
    askError(e.message);
  } finally {
    button.disabled = false;
    $("#stop-btn").hidden = true;
    state.askJob = null;
  }
}

const VERDICT_HELP = {
  verified: "The quote is on the cited page, word for word.",
  close: "On the cited page, allowing for OCR errors.",
  joined: "Every word is on the page in order, but not as one run of text (read across columns). True to the page; not a clean quotation.",
  "wrong page": "The quote is real but the model cited the wrong page. The citation has been corrected.",
  unverified: "This quote was not found in the pages the model was given. Treat it as unsupported.",
};

function renderAnswer(r) {
  const box = $("#answer");
  box.hidden = false;
  const findings = r.findings.map((f, i) => `
    <div class="finding">
      <span class="badge ${esc(f.verdict.replace(" ", "-"))}" title="${esc(VERDICT_HELP[f.verdict] || "")}">${esc(f.verdict)}</span>
      <p class="statement">${esc(f.statement)}</p>
      ${f.quote ? `<blockquote>“${esc(f.quote)}”</blockquote>` : ""}
      ${f.doc_id ? `<button class="cite" data-f="${i}">${esc(f.title)} — page ${f.page}</button>` : ""}
    </div>`).join("");
  const meta = [r.model, r.mode, r.seconds !== undefined ? `${r.seconds}s` : "",
    r.findings.length ? `${r.verified} of ${r.findings.length} findings checked against the page` : ""].filter(Boolean).map(esc).join(" · ");
  box.innerHTML = `
    <h2>Answer</h2>
    ${r.answer ? `<p class="answer-text">${esc(r.answer)}</p>` : '<p class="muted">No answer found in the selected documents.</p>'}
    <div class="meta">${meta}</div>
    ${r.missing ? `<div class="note"><b>Not in the documents:</b> ${esc(r.missing)}</div>` : ""}
    ${r.warnings.map((w) => `<div class="note">${esc(w)}</div>`).join("")}
    ${findings}
    ${r.thinking ? `<details class="thinking"><summary>The model's reasoning</summary><pre>${esc(r.thinking)}</pre></details>` : ""}`;
  $$("[data-f]", box).forEach((b) => (b.onclick = () => openPage(r.findings[Number(b.dataset.f)])));
  if (r.findings[0]?.doc_id) openPage(r.findings[0]);
}

// ---------------------------------------------------------------- viewer

let viewerMode = store.get("viewerMode", "text");
let viewing = null;
$$("#viewer-mode button").forEach((b) => b.addEventListener("click", () => {
  viewerMode = b.dataset.mode;
  store.set("viewerMode", viewerMode);
  if (viewing) openPage(viewing);
}));

function highlight(text, quote) {
  const safe = esc(text);
  if (!quote) return safe;
  const words = quote.toLowerCase().match(/[a-z0-9]+/g) || [];
  if (!words.length) return safe;
  const pattern = new RegExp(words.map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("[^a-z0-9]+"), "i");
  const match = pattern.exec(text);
  if (!match) return safe;
  return esc(text.slice(0, match.index)) + `<mark id="hit">${esc(match[0])}</mark>` + esc(text.slice(match.index + match[0].length));
}

async function openPage(f) {
  viewing = f;
  const doc = state.docs.find((d) => d.id === f.doc_id);
  const isPdf = doc?.kind === "pdf";
  $("#viewer-title").textContent = `${f.title} — page ${f.page}`;
  $("#viewer-empty").hidden = true;
  $("#viewer-mode").hidden = !isPdf;
  const mode = isPdf ? viewerMode : "text";
  $$("#viewer-mode button").forEach((b) => b.classList.toggle("on", b.dataset.mode === mode));
  const text = $("#viewer-text");
  const frame = $("#viewer-file");
  if (mode === "file") {
    text.hidden = true;
    frame.hidden = false;
    frame.src = `/api/documents/${f.doc_id}/file#page=${f.page}`;
    return;
  }
  frame.hidden = true;
  text.hidden = false;
  text.textContent = "Loading…";
  try {
    const page = await api(`/api/documents/${f.doc_id}/page/${f.page}`);
    text.innerHTML = highlight(page.text, f.quote);
    const hit = $("#hit", text);
    if (hit) hit.scrollIntoView({ block: "center" });
  } catch (e) {
    text.textContent = e.message;
  }
}

// ------------------------------------------------------------------ start

(async function start() {
  const fromHash = location.hash.slice(1);
  showTab(["ask", "models", "documents"].includes(fromHash) ? fromHash : store.get("tab", "ask"));
  await refreshStatus();
  await loadDocs();
  setInterval(refreshStatus, 8000);
})();
