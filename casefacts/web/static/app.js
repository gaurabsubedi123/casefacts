/* The browser half. Plain JavaScript, no framework, no build step.

   Three things happen here that are worth knowing about:

   1. Long jobs (reading a folder, sweeping for a chronology) report over
      Server-Sent Events, and every connection opens with a full snapshot, so
      a tab opened halfway through shows the true state at once.
   2. Clicking a citation loads the page into the right-hand column and
      highlights the words of the quote in the page text. The highlight is
      per-word rather than per-phrase because OCR'd layout puts arbitrary
      whitespace between words, and a phrase match would silently find
      nothing on exactly the pages you most want to check.
   3. Nothing is fetched from anywhere but this server. */

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const state = {
  documents: [],
  sources: [],
  models: [],
  chosen: new Set(),
  stream: null,
};

function esc(text) {
  return String(text ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

async function api(path, options) {
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({ error: "the server said something that was not JSON" }));
  if (!response.ok && !data.error) data.error = `request failed (${response.status})`;
  return data;
}

/* ---------------------------------------------------------------- state */

async function refresh() {
  const data = await api("/api/state");
  state.documents = data.documents || [];
  state.sources = data.sources || [];
  $("#topstats").textContent = `${data.stats.documents} documents · ${data.stats.pages} pages`;
  fillScopes();
  renderDocuments();
  renderSources();
  if (data.job && data.job.status === "running") watchJob();
}

function fillScopes() {
  for (const id of ["#scope", "#chron-scope"]) {
    const select = $(id);
    const previous = select.value;
    select.innerHTML = '<option value="">everything indexed</option>';
    if (state.sources.length) {
      const group = document.createElement("optgroup");
      group.label = "a whole folder";
      for (const source of state.sources) {
        const option = document.createElement("option");
        option.value = "folder:" + source.root;
        option.textContent = source.root;
        group.appendChild(option);
      }
      select.appendChild(group);
    }
    if (state.documents.length) {
      const group = document.createElement("optgroup");
      group.label = "one document";
      for (const doc of state.documents) {
        const option = document.createElement("option");
        option.value = "doc:" + doc.doc_id;
        option.textContent = `${doc.title}  (${doc.page_count} pages)`;
        group.appendChild(option);
      }
      select.appendChild(group);
    }
    select.value = previous;
  }
}

function scopeOf(select) {
  const value = $(select).value;
  if (value.startsWith("doc:")) return { doc: value.slice(4) };
  if (value.startsWith("folder:")) return { folder: value.slice(7) };
  return {};
}

function renderSources() {
  const box = $("#sources");
  if (!state.sources.length) { box.innerHTML = '<p class="empty">Nothing plugged in yet.</p>'; return; }
  box.innerHTML = state.sources.map((source) => `
    <div class="source">
      <span class="count">${source.documents} docs · ${source.pages || 0} pages</span>
      <span class="path">${esc(source.root)}</span>
      <button class="link forget" data-root="${esc(source.root)}">forget</button>
    </div>`).join("");
  $$(".forget", box).forEach((button) => {
    button.onclick = async () => {
      if (!confirm(`Remove ${button.dataset.root} from the index?\n\nYour files are not touched.`)) return;
      await api("/api/forget", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ root: button.dataset.root }),
      });
      refresh();
    };
  });
}

function renderDocuments() {
  const box = $("#documents");
  if (!state.documents.length) { box.innerHTML = '<p class="empty">Nothing indexed yet.</p>'; return; }
  box.innerHTML = state.documents.map((doc) => `
    <div class="doc">
      <div class="title">${esc(doc.title)}</div>
      <div class="about">
        ${doc.page_count} pages
        ${doc.mean_confidence != null ? `· OCR ${Math.round(doc.mean_confidence)}%` : ""}
        · ${esc(doc.origin || "")}
        ${doc.original_path ? `· ${esc(doc.original_path)}` : ""}
      </div>
      ${(doc.aliases || []).map((a) => `<div class="alias">also filed as ${esc(a.rel_path)}</div>`).join("")}
    </div>`).join("");
}

/* ------------------------------------------------------------- plugging in */

$("#add-form").onsubmit = async (event) => {
  event.preventDefault();
  const path = $("#add-path").value.trim();
  if (!path) return;
  const data = await api("/api/add", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path, ocr: !$("#add-noocr").checked, rebuild: $("#add-rebuild").checked }),
  });
  if (data.error) { alert(data.error); return; }
  watchJob();
};

$("#job-stop").onclick = () => api("/api/job/stop", { method: "POST" });

function watchJob() {
  if (state.stream) state.stream.close();
  const job = $("#job");
  job.hidden = false;
  const log = $("#job-log");
  const bar = $("#job-bar");
  const stream = new EventSource("/api/job/stream");
  state.stream = stream;
  // A previous job hid it on finishing; this one can be stopped.
  $("#job-stop").hidden = false;

  stream.onmessage = (message) => {
    const event = JSON.parse(message.data);
    if (event.event === "ping") return;

    if (event.event === "snapshot" || event.event === "closed" || event.event === "finished") {
      $("#job-label").textContent = `${event.kind || ""} ${event.label || ""} — ${event.status}`;
      (event.lines || []).forEach((line) => log.append(line + "\n"));
      if (event.status && event.status !== "running") {
        stream.close();
        state.stream = null;
        bar.style.width = "100%";
        $("#job-stop").hidden = true;
        if (event.error) log.append("error: " + event.error + "\n");
        refresh();
        if (event.kind === "chronology") loadChronology();
      }
      return;
    }

    $("#job-stop").hidden = false;
    if (event.event === "ocr") log.append(event.line + "\n");
    if (event.event === "error") log.append("! " + (event.error || "") + "\n");
    if (event.event === "document") {
      $("#job-label").textContent = `${event.file} — ${event.pages} pages  (${event.position}/${event.total})`;
      bar.style.width = `${(event.position / Math.max(1, event.total)) * 100}%`;
    }
    if (event.event === "embedding") {
      $("#job-label").textContent = `indexing ${event.file} — ${event.done}/${event.total}`;
    }
    if (event.event === "page") {
      $("#job-label").textContent = `page ${event.position}/${event.total} — ${event.title} p.${event.page}`;
      bar.style.width = `${(event.position / Math.max(1, event.total)) * 100}%`;
    }
    if (event.event === "done") {
      $("#job-label").textContent = "finished";
    }
    log.scrollTop = log.scrollHeight;
  };
  stream.onerror = () => { stream.close(); state.stream = null; };
}

/* ------------------------------------------------------------------ models */

async function loadModels() {
  const data = await api("/api/models");
  const box = $("#model-list");
  if (data.error) { box.innerHTML = `<div class="warning">${esc(data.error)}</div>`; return; }

  state.models = (data.models || []).filter((m) => m.kind !== "embed");
  const preferred = state.models.find((m) => m.default) || state.models[0];
  if (preferred && !state.chosen.size) state.chosen.add(preferred.name);

  box.innerHTML = state.models.map((model) => `
    <label class="check" style="display:flex;align-items:flex-start;gap:8px;margin:8px 0">
      <input type="checkbox" value="${esc(model.name)}" ${state.chosen.has(model.name) ? "checked" : ""}>
      <span>
        <b>${esc(model.label)}</b> <span class="hint">${model.size_gb} GB</span><br>
        <span class="hint">${esc(model.notes)}</span>
      </span>
    </label>`).join("")
    + (data.missing || []).map((model) => `
    <div class="hint" style="margin:8px 0">
      <b>${esc(model.label)}</b> — not installed. <code>ollama pull ${esc(model.name)}</code><br>${esc(model.notes)}
    </div>`).join("");

  $$("input[type=checkbox]", box).forEach((input) => {
    input.onchange = () => {
      input.checked ? state.chosen.add(input.value) : state.chosen.delete(input.value);
      updateModelSummary();
    };
  });

  const chronModel = $("#chron-model");
  chronModel.innerHTML = state.models.map((m) =>
    `<option value="${esc(m.name)}" ${m.default ? "selected" : ""}>${esc(m.label)}</option>`).join("");
  updateModelSummary();
}

function updateModelSummary() {
  const chosen = Array.from(state.chosen);
  $("#model-summary").textContent = chosen.length > 1
    ? `— comparing ${chosen.length}` : `— ${chosen[0] || "none chosen"}`;
}

/* -------------------------------------------------------------- asking */

$("#ask-form").onsubmit = async (event) => {
  event.preventDefault();
  const question = $("#question").value.trim();
  if (!question) return;

  const button = $("#ask-form button.primary");
  button.disabled = true;
  const started = Date.now();
  const box = $("#answer");
  const timer = setInterval(() => {
    box.innerHTML = `<div class="panel"><span class="spinner"></span> reading the records — ${
      Math.round((Date.now() - started) / 1000)}s</div>`;
  }, 200);

  const body = { question, models: Array.from(state.chosen), whole: $("#whole").checked, ...scopeOf("#scope") };
  const data = await api("/api/ask", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });

  clearInterval(timer);
  button.disabled = false;
  if (data.error) { box.innerHTML = `<div class="warning failed">${esc(data.error)}</div>`; return; }
  box.innerHTML = data.answers ? renderComparison(data) : renderAnswer(data);
  wireCitations(box);
};

function renderAnswer(answer) {
  const warnings = (answer.warnings || []).map((w) => `<div class="warning">${esc(w)}</div>`).join("");
  const head = `<div class="answer-meta">${esc(answer.model)} · ${esc(answer.mode)} · ${answer.seconds}s
    · ${answer.verified}/${answer.claimed} findings checked against their page</div>`;
  const summary = answer.answer ? `<p class="summary">${esc(answer.answer)}</p>` : "";
  const missing = answer.missing
    ? `<div class="panel"><b>Not in the records:</b> ${esc(answer.missing)}</div>` : "";
  const findings = (answer.findings || []).map(renderFinding).join("");
  const nothing = !answer.findings.length && !answer.answer
    ? '<div class="panel">Nothing in the indexed pages answers this.</div>' : "";
  return `<div class="panel">${head}${summary}${warnings}</div>${findings}${missing}${nothing}`;
}

const VERDICTS = {
  verified: ["verified", "This quote is on the page it cites, word for word."],
  close: ["close", "On the cited page, allowing for OCR errors in the text."],
  joined: ["joined", "Every word is on the page in this order, but read across the layout — true to the page, not a verbatim quotation."],
  "wrong page": ["moved", "The quote is real but the model named the wrong page; the citation has been corrected."],
  unverified: ["unverified", "This quote is not on any page that was retrieved. Treat it as unsupported."],
};

function renderFinding(finding) {
  const [cls, explanation] = VERDICTS[finding.verdict] || ["unverified", ""];
  const lowConfidence = finding.confidence != null && finding.confidence < 70
    ? `<span class="lowconf">OCR ${Math.round(finding.confidence)}% — read the page image</span>` : "";
  return `
    <div class="finding ${cls === "verified" || cls === "close" ? "" : cls}">
      <div class="statement">${esc(finding.statement)}</div>
      <blockquote>${esc(finding.quote)}</blockquote>
      <div class="foot">
        <span class="badge ${cls}">${cls}</span>
        ${finding.doc_id
          ? `<button class="cite" data-doc="${esc(finding.doc_id)}" data-page="${finding.page}"
                data-quote="${esc(finding.quote)}">${esc(finding.citation)}</button>`
          : '<span class="note">no page</span>'}
        ${lowConfidence}
      </div>
      <div class="note">${esc(explanation)}</div>
    </div>`;
}

function renderComparison(result) {
  const columns = result.summary.map((row) => {
    const answer = result.answers.find((a) => a.model === row.model);
    return `
      <div class="panel">
        <div class="compare-head">
          <h3>${esc(row.model)}</h3>
          <span class="hint">${row.seconds}s · ${row.verified} checked · ${row.unverified} unsupported</span>
        </div>
        <p class="summary">${esc(row.answer || "(no answer)")}</p>
        ${(answer.findings || []).map(renderFinding).join("")}
        ${(answer.warnings || []).map((w) => `<div class="warning">${esc(w)}</div>`).join("")}
      </div>`;
  }).join("");

  const agreement = result.agreement || {};
  const shared = (agreement.pages_all_models_cited || []);
  const only = Object.entries(agreement.only_one_model || {})
    .filter(([, pages]) => pages.length)
    .map(([model, pages]) => `<li>only <b>${esc(model)}</b>: ${esc(pages.join(", "))}</li>`).join("");
  const head = `
    <div class="agreement">
      <b>They agree on ${shared.length} page${shared.length === 1 ? "" : "s"}</b>
      ${shared.length ? `— ${esc(shared.join(", "))}` : ""}
      <div class="hint">Pages every model landed on independently are the ones to read first.</div>
      ${only ? `<ul class="hint">${only}</ul>` : ""}
    </div>`;
  return `<div class="compare">${head}${columns}</div>`;
}

function wireCitations(root) {
  $$(".cite", root).forEach((button) => {
    button.onclick = () => showPage(button.dataset.doc, button.dataset.page, button.dataset.quote);
  });
}

/* -------------------------------------------------------- the page viewer */

async function showPage(docId, page, quote) {
  const viewer = $("#viewer");
  viewer.innerHTML = '<div class="viewer-empty"><span class="spinner"></span> loading the page…</div>';
  const data = await api(`/api/page?doc=${encodeURIComponent(docId)}&page=${page}`);
  if (data.error) { viewer.innerHTML = `<div class="viewer-empty">${esc(data.error)}</div>`; return; }

  const image = data.has_image
    ? `<img src="/api/preview?doc=${encodeURIComponent(docId)}&page=${page}" alt="page ${page}">`
    : '<p class="hint">No page image for this document — it was indexed from text alone.</p>';
  const confidence = data.confidence != null
    ? `<span class="about">OCR ${Math.round(data.confidence)}%${data.needs_review ? " · flagged for review" : ""}</span>`
    : "";

  viewer.innerHTML = `
    <div class="viewer-head">
      <span class="title">${esc(data.title)}</span>
      <span class="about">page ${data.page_no} of ${data.page_count}</span>
      ${data.bates ? `<span class="about">${esc(data.bates)}</span>` : ""}
      ${confidence}
    </div>
    <div class="viewer-body">
      ${image}
      <pre>${highlight(data.text || "", quote)}</pre>
    </div>`;
  viewer.scrollTop = 0;
}

function highlight(text, quote) {
  const escaped = esc(text);
  if (!quote) return escaped;
  // Word by word, not phrase by phrase: a layout-preserved OCR page puts runs
  // of spaces inside what the model quoted as one phrase, so a phrase match
  // would highlight nothing on exactly the pages worth checking by eye.
  const words = Array.from(new Set(
    quote.toLowerCase().split(/[^a-z0-9]+/i).filter((w) => w.length >= 4)
  )).sort((a, b) => b.length - a.length).slice(0, 40);
  if (!words.length) return escaped;
  const pattern = new RegExp(`(${words.map(escapeRegex).join("|")})`, "gi");
  return escaped.replace(pattern, "<mark>$1</mark>");
}

function escapeRegex(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/* -------------------------------------------------------------- chronology */

$("#chron-build").onclick = async () => {
  const body = { model: $("#chron-model").value, ...scopeOf("#chron-scope") };
  const data = await api("/api/chronology", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (data.error) { alert(data.error); return; }
  watchJob();
};

$("#chron-load").onclick = () => loadChronology();

async function loadChronology() {
  const scope = scopeOf("#chron-scope");
  const params = new URLSearchParams({
    model: $("#chron-model").value,
    unverified: $("#chron-unverified").checked ? "1" : "0",
    ...(scope.doc ? { doc: scope.doc } : {}),
    ...(scope.folder ? { folder: scope.folder } : {}),
  });
  $("#chron-csv").href = "/api/chronology.csv?" + params.toString();

  const box = $("#chronology");
  box.innerHTML = '<div class="panel"><span class="spinner"></span> loading…</div>';
  const data = await api("/api/chronology?" + params.toString());
  if (!data.events || !data.events.length) {
    box.innerHTML = '<div class="panel">Nothing extracted yet. Press <b>Build it</b>.</div>';
    return;
  }

  const gapAfter = new Map((data.gaps || []).map((gap) => [gap.after, gap]));
  const rows = [];
  for (const event of data.events) {
    rows.push(`
      <tr>
        <td class="date">${esc(event.date || "—")}</td>
        <td>
          ${esc(event.event)}
          ${event.verdict !== "verified" ? `<span class="badge unverified">${esc(event.verdict)}</span>` : ""}
          <div class="hint">${esc([event.provider, event.facility].filter(Boolean).join(" · "))}</div>
        </td>
        <td><span class="cat">${esc(event.category)}</span></td>
        <td><button class="cite" data-doc="${esc(event.doc_id)}" data-page="${event.page}"
              data-quote="${esc(event.quote)}">${esc(event.citation)}</button></td>
      </tr>`);
    const gap = gapAfter.get(event.date);
    if (gap) {
      rows.push(`<tr class="gap"><td colspan="4">${gap.days} days with no record — ${esc(gap.after)} to ${esc(gap.before)}</td></tr>`);
    }
  }

  box.innerHTML = `
    <div class="panel" style="padding:0;overflow:auto;max-height:70vh">
      <table class="chron">
        <thead><tr><th>Date</th><th>Event</th><th>Kind</th><th>Where it says so</th></tr></thead>
        <tbody>${rows.join("")}</tbody>
      </table>
    </div>
    ${(data.gaps || []).length
      ? `<div class="panel"><b>${data.gaps.length} gap${data.gaps.length === 1 ? "" : "s"} of 30+ days.</b>
         <span class="hint">These are what a defence examiner calls a break in treatment.</span></div>`
      : ""}`;
  wireCitations(box);
}

/* -------------------------------------------------------------------- tabs */

$$(".tab").forEach((tab) => {
  tab.onclick = () => {
    $$(".tab").forEach((other) => other.classList.toggle("active", other === tab));
    $$(".tabpane").forEach((pane) => { pane.hidden = pane.dataset.pane !== tab.dataset.tab; });
    if (tab.dataset.tab === "chronology") loadChronology();
  };
});

refresh();
loadModels();
