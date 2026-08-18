/* Agentic RAG - frontend logic */
"use strict";

const messagesEl = document.getElementById("messages");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("sendBtn");
const newChatBtn = document.getElementById("newChatBtn");
const docListEl = document.getElementById("docList");
const statsEl = document.getElementById("stats");
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");

let conversationId = null;
let currentConvId = null;
let currentUser = null;
let sending = false;
let searchCollection = "";
let uploadCollection = "default";

/* ---------------- helpers ---------------- */

function scrollBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function addMessage(role, text, sources) {
  const wrap = document.createElement("div");
  wrap.className = "msg " + role;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  if (text !== "") bubble.textContent = text;
  wrap.appendChild(bubble);
  messagesEl.appendChild(wrap);
  // Assistant messages loaded from history: rebuild the clickable [n] citations
  // and the source cards from the persisted sources, so they survive switching
  // between chats and page reloads.
  if (role === "assistant" && text && sources && sources.length) {
    renderAnswer(bubble, text);
    renderSources(wrap, sources);
  }
  scrollBottom();
  return { wrap, bubble };
}

function renderAnswer(bubble, answer) {
  bubble.innerHTML = markdownHtml(answer);
  bubble.classList.add("markdown");
}

// Render an assistant answer as HTML: Markdown (via `marked` when available,
// otherwise a small built-in renderer) plus clickable [n] citation links.
function markdownHtml(text) {
  let html;
  if (window.marked && window.marked.parse) {
    try {
      html = window.marked.parse(text, { gfm: true, breaks: true, async: false });
    } catch (e) {
      html = null;
    }
  }
  if (!html) html = fallbackMarkdown(text);
  // Wrap [n] citation markers as clickable links (e.g. [1], [2, 3]).
  return html.replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, (m, nums) => {
    const first = nums.trim().split(",")[0].trim();
    return '<span class="cite-link" data-cite="' + first + '">' + m + "</span>";
  });
}

function renderSources(container, sources) {
  if (!sources || !sources.length) return;
  const panel = document.createElement("div");
  panel.className = "sources-panel";
  const title = document.createElement("div");
  title.className = "sources-title";
  title.textContent = "Sources";
  panel.appendChild(title);

  const seenImages = new Set();
  for (const s of sources) {
    const card = document.createElement("div");
    card.className = "source-card";
    card.dataset.citation = s.citation;

    const head = document.createElement("div");
    head.className = "source-card-head";
    const badge = document.createElement("span");
    badge.className = "source-badge";
    badge.textContent = s.citation;
    head.appendChild(badge);
    const t = document.createElement("span");
    t.className = "source-card-title";
    t.textContent = s.title + (s.page ? " • p." + s.page : "");
    head.appendChild(t);
    card.appendChild(head);

    const body = document.createElement("div");
    body.className = "source-card-body";
    if (s.snippet) {
      const p = document.createElement("p");
      p.className = "source-snippet";
      p.textContent = s.snippet;
      body.appendChild(p);
    }
    if (s.image_id && !seenImages.has(s.image_id)) {
      seenImages.add(s.image_id);
      const a = document.createElement("a");
      a.href = "/images/" + s.image_id;
      a.target = "_blank";
      a.title = "Open full image";
      const img = document.createElement("img");
      img.className = "source-img";
      img.src = "/images/" + s.image_id;
      a.appendChild(img);
      body.appendChild(a);
    }
    card.appendChild(body);
    panel.appendChild(card);
  }
  container.appendChild(panel);
  scrollBottom();
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* ---- Lightweight built-in Markdown renderer (used when `marked` is unavailable) ---- */

function fallbackInline(s) {
  s = s.replace(/`([^`\n]+)`/g, "<code>$1</code>");
  s = s.replace(/!\[([^\]]*)\]\(([^)\s]+)\)/g, '<img src="$2" alt="$1" />');
  s = s.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  s = s.replace(/\*\*\*([^*]+)\*\*\*/g, "<strong><em>$1</em></strong>");
  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/__([^_]+)__/g, "<strong>$1</strong>");
  s = s.replace(/(^|[^*])\*([^*\s][^*]*)\*/g, "$1<em>$2</em>");
  s = s.replace(/(^|[^_])_([^_\s][^_]*)_/g, "$1<em>$2</em>");
  s = s.replace(/~~([^~]+)~~/g, "<del>$1</del>");
  return s;
}

function fallbackMarkdown(text) {
  const lines = escapeHtml(text).split(/\r?\n/);
  let html = "";
  let i = 0;

  const isHr = (l) => /^\s*(?:-{3,}|\*{3,}|_{3,})\s*$/.test(l);
  const isTableSep = (l) => /^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$/.test(l);
  const splitRow = (r) => {
    const parts = r.trim().split("|").map((c) => c.trim());
    if (parts.length > 1 && parts[0] === "") parts.shift();
    if (parts.length > 1 && parts[parts.length - 1] === "") parts.pop();
    return parts;
  };
  const itemMatch = (l) => {
    const t = l.match(/^\s*[-*+]\s+\[([ xX])\]\s+(.*)$/);
    if (t) return { ordered: false, checked: t[1].toLowerCase() === "x", content: t[2] };
    const u = l.match(/^\s*[-*+]\s+(.*)$/);
    if (u) return { ordered: false, checked: null, content: u[1] };
    const o = l.match(/^\s*\d+[.)]\s+(.*)$/);
    if (o) return { ordered: true, checked: null, content: o[1] };
    return null;
  };

  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i++; continue; }

    // Fenced code block
    if (/^```/.test(line)) {
      const buf = [];
      i++;
      while (i < lines.length && !/^```/.test(lines[i])) { buf.push(lines[i]); i++; }
      i++; // skip closing fence
      html += "<pre><code>" + buf.join("\n") + "</code></pre>";
      continue;
    }

    // Heading
    const h = line.match(/^(#{1,6})\s+(.*)$/);
    if (h) {
      const lvl = h[1].length;
      html += "<h" + lvl + ">" + fallbackInline(h[2]) + "</h" + lvl + ">";
      i++;
      continue;
    }

    // Horizontal rule
    if (isHr(line)) { html += "<hr>"; i++; continue; }

    // Blockquote
    if (/^>\s?/.test(line)) {
      const buf = [];
      while (i < lines.length && /^>\s?/.test(lines[i])) { buf.push(lines[i].replace(/^>\s?/, "")); i++; }
      html += "<blockquote>" + fallbackInline(buf.join("<br>")) + "</blockquote>";
      continue;
    }

    // Table
    if (line.includes("|") && i + 1 < lines.length && isTableSep(lines[i + 1])) {
      const header = splitRow(line);
      const rows = [];
      i += 2;
      while (i < lines.length && lines[i].includes("|") && lines[i].trim()) { rows.push(splitRow(lines[i])); i++; }
      html += "<table><thead><tr>" + header.map((c) => "<th>" + fallbackInline(c) + "</th>").join("") + "</tr></thead><tbody>"
        + rows.map((r) => "<tr>" + r.map((c) => "<td>" + fallbackInline(c) + "</td>").join("") + "</tr>").join("")
        + "</tbody></table>";
      continue;
    }

    // Lists (task / bullet / numbered)
    const first = itemMatch(line);
    if (first) {
      const items = [];
      while (i < lines.length) {
        const m = itemMatch(lines[i]);
        if (!m || m.ordered !== first.ordered) break;
        items.push(m);
        i++;
      }
      const tag = first.ordered ? "ol" : "ul";
      html += "<" + tag + ">"
        + items.map((it) => "<li>"
          + (it.checked !== null ? '<input type="checkbox" disabled' + (it.checked ? " checked" : "") + "> " : "")
          + fallbackInline(it.content) + "</li>").join("")
        + "</" + tag + ">";
      continue;
    }

    // Paragraph
    const buf = [line];
    i++;
    while (i < lines.length && lines[i].trim()
      && !/^(#{1,6})\s/.test(lines[i]) && !/^```/.test(lines[i]) && !/^>\s?/.test(lines[i])
      && !isHr(lines[i]) && !itemMatch(lines[i])
      && !(lines[i].includes("|") && i + 1 < lines.length && isTableSep(lines[i + 1]))) {
      buf.push(lines[i]);
      i++;
    }
    html += "<p>" + fallbackInline(buf.join("<br>")) + "</p>";
  }
  return html;
}

function toast(msg) {
  let t = document.querySelector(".toast");
  if (!t) { t = document.createElement("div"); t.className = "toast"; document.body.appendChild(t); }
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(t._h);
  t._h = setTimeout(() => t.classList.remove("show"), 3000);
}

/* ---------------- live time-based greeting ---------------- */
// Placeholder until login lands. A future login can set window.AGENTIC_RAG_USER
// (or localStorage "rag_user_name") so the greeting uses the real name.
function displayName() {
  try {
    if (currentUser && currentUser.user && currentUser.user.display_name) return currentUser.user.display_name;
  } catch (e) {}
  if (typeof window.AGENTIC_RAG_USER === "string" && window.AGENTIC_RAG_USER.trim()) {
    return window.AGENTIC_RAG_USER.trim();
  }
  try {
    const n = localStorage.getItem("rag_user_name");
    if (n) return n;
  } catch (e) {}
  return "User";
}

/* ---------------- users & persisted chat history ------------------------- */

let loginMode = "login"; // 'login' | 'register'

function loadUser() {
  try {
    const raw = localStorage.getItem("rag_user");
    return raw ? JSON.parse(raw) : null; // { token, user: {id, username, display_name} }
  } catch (e) { return null; }
}

function saveUser(u) { try { localStorage.setItem("rag_user", JSON.stringify(u)); } catch (e) {} }

function authHeaders() {
  const h = { "Content-Type": "application/json" };
  if (currentUser && currentUser.token) h["Authorization"] = "Bearer " + currentUser.token;
  return h;
}

function showLogin() { document.getElementById("loginModal").classList.remove("hidden"); }
function hideLogin() { document.getElementById("loginModal").classList.add("hidden"); }

function setLoginMode(mode) {
  loginMode = mode;
  const register = mode === "register";
  document.getElementById("loginTitle").textContent = register ? "Create your account 👋" : "Welcome back 👋";
  document.getElementById("loginSub").textContent = register
    ? "Pick a username and password — your chats are saved here."
    : "Sign in to continue — your chats are saved here.";
  document.getElementById("loginBtn").textContent = register ? "Create account" : "Sign in";
  document.getElementById("loginToggle").textContent = register
    ? "Already have an account? Sign in" : "New here? Create an account";
  document.getElementById("loginNote").textContent = "";
}

async function doLogin() {
  const name = document.getElementById("loginName").value.trim();
  const pass = document.getElementById("loginPass").value;
  if (!name || !pass) { toast("⚠️ Enter a username and password"); return; }
  const path = loginMode === "register" ? "/api/register" : "/api/login";
  const body = loginMode === "register"
    ? JSON.stringify({ username: name.toLowerCase(), password: pass, display_name: name })
    : JSON.stringify({ username: name.toLowerCase(), password: pass });
  const resp = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body });
  if (!resp.ok) {
    const d = await resp.json().catch(() => ({}));
    document.getElementById("loginNote").textContent = "⚠️ " + (d.detail || "login failed");
    return;
  }
  let data = await resp.json();
  if (!data.token) {
    // just registered -> log in to get a token
    const lg = await fetch("/api/login", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: name.toLowerCase(), password: pass }),
    });
    data = await lg.json();
  }
  currentUser = { token: data.token, user: data.user };
  saveUser(currentUser);
  hideLogin();
  updateUserChip();
  loadHistory();
  showGreeting(); // re-greet with the real name
}

function updateUserChip() {
  const name = displayName();
  const u = (currentUser && currentUser.user) || {};
  document.getElementById("userName").textContent = name;
  document.getElementById("userAvatar").textContent = (name[0] || "U").toUpperCase();
  document.getElementById("userSub").textContent = u.username ? "@" + u.username : "Account";
  document.getElementById("profileName").textContent = name;
  document.getElementById("profileUsername").textContent = u.username ? "@" + u.username : "";
}

/* ---------------- profile menu (reset password + logout) ---------------- */

function toggleProfileMenu() {
  if (!currentUser) return;
  const menu = document.getElementById("profileMenu");
  if (menu.classList.contains("hidden")) {
    document.getElementById("profileMenu").classList.remove("hidden");
    resetPasswordForm();
  } else {
    closeProfileMenu();
  }
}

function closeProfileMenu() {
  document.getElementById("profileMenu").classList.add("hidden");
  // Discard any in-progress password edits — nothing is saved unless Update is clicked.
  resetPasswordForm();
}

function resetPasswordForm() {
  document.getElementById("profilePassword").classList.add("hidden");
  document.getElementById("pwCurrent").value = "";
  document.getElementById("pwNew").value = "";
  document.getElementById("pwNote").textContent = "";
}

function showPasswordForm() {
  document.getElementById("profilePassword").classList.remove("hidden");
  document.getElementById("pwNote").textContent = "";
  document.getElementById("pwCurrent").focus();
}

async function updatePassword() {
  const cur = document.getElementById("pwCurrent").value;
  const nw = document.getElementById("pwNew").value;
  const note = document.getElementById("pwNote");
  if (!cur || !nw) { note.textContent = "⚠️ Fill in both fields"; return; }
  note.textContent = "";
  let resp;
  try {
    resp = await fetch("/api/password", {
      method: "POST", headers: authHeaders(),
      body: JSON.stringify({ current_password: cur, new_password: nw }),
    });
  } catch (e) {
    note.textContent = "⚠️ Could not reach server"; return;
  }
  if (resp.status === 401) { closeProfileMenu(); showLogin(); return; }
  const d = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    note.textContent = "⚠️ " + (d.detail || "could not change password");
    return;
  }
  note.textContent = "✅ Password updated";
  toast("✅ Password updated");
  setTimeout(closeProfileMenu, 700);
}

async function logout() {
  closeProfileMenu();
  if (currentUser && currentUser.token) {
    try { await fetch("/api/logout", { method: "POST", headers: authHeaders() }); } catch (e) {}
  }
  currentUser = null;
  currentConvId = null;
  conversationId = null;
  try { localStorage.removeItem("rag_user"); } catch (e) {}
  updateUserChip();
  document.getElementById("historyList").innerHTML = "";
  messagesEl.innerHTML = "";
  showLogin();
  showGreeting();
}

function timeAgo(iso) {
  const s = Math.floor((Date.now() - new Date(iso).getTime()) / 1000);
  if (s < 60) return "just now";
  if (s < 3600) return Math.floor(s / 60) + "m ago";
  if (s < 86400) return Math.floor(s / 3600) + "h ago";
  return Math.floor(s / 86400) + "d ago";
}

async function loadHistory() {
  const list = document.getElementById("historyList");
  if (!currentUser || !currentUser.token) { list.innerHTML = ""; return; }
  try {
    const resp = await fetch("/conversations", { headers: authHeaders() });
    if (resp.status === 401) { showLogin(); return; }
    const convs = await resp.json();
    list.innerHTML = "";
    if (!convs.length) {
      list.innerHTML = '<div class="empty">No chats yet — ask something!</div>';
      return;
    }
    for (const c of convs) {
      const item = document.createElement("div");
      item.className = "history-item" + (c.id === currentConvId ? " active" : "");
      item.dataset.id = c.id;
      const title = document.createElement("div");
      title.className = "history-title";
      title.textContent = c.title || "New chat";
      const meta = document.createElement("div");
      meta.className = "history-meta";
      meta.textContent = `${c.messages || 0} msgs · ${timeAgo(c.created_at)}`;
      const del = document.createElement("button");
      del.className = "history-del";
      del.textContent = "🗑";
      del.title = "Delete chat";
      item.appendChild(title); item.appendChild(meta); item.appendChild(del);
      item.addEventListener("click", (e) => {
        if (e.target.closest(".history-del")) return;
        openConversation(c.id);
      });
      del.addEventListener("click", async (e) => {
        e.stopPropagation();
        const resp = await fetch(`/conversations/${c.id}`, { method: "DELETE", headers: authHeaders() });
        if (resp.ok) {
          if (c.id === currentConvId) {
            conversationId = null; currentConvId = null;
            messagesEl.innerHTML = "";
            showGreeting();
          }
          loadHistory();
        } else {
          toast("⚠️ Could not delete chat");
        }
      });
      list.appendChild(item);
    }
  } catch (e) {
    list.innerHTML = '<div class="empty">Could not load history</div>';
  }
}

async function openConversation(id) {
  try {
    const resp = await fetch(`/conversations/${id}`, { headers: authHeaders() });
    if (resp.status === 401) { showLogin(); return; }
    const data = await resp.json();
    if (data.detail) { toast("⚠️ Could not open chat"); return; }
    conversationId = id;
    currentConvId = id;
    removeGreeting();
    messagesEl.innerHTML = "";
    for (const m of data.messages || []) {
      addMessage(m.role === "user" ? "user" : "assistant", m.content, m.sources || []);
    }
    scrollBottom();
    loadHistory();
  } catch (e) {
    toast("⚠️ Could not open chat");
  }
}

const GREETINGS = [
  { from: 5, to: 12, lines: [
      "Good morning, {name}! ☀️ Ready when you are.",
      "Morning, {name}! 🌤️ What shall we look at?",
  ]},
  { from: 12, to: 17, lines: [
      "Good afternoon, {name}! 😊 Ask me anything.",
      "Afternoon, {name}! ☀️ What do you need?",
  ]},
  { from: 17, to: 21, lines: [
      "Coffee o'clock, {name}! ☕ Ask away.",
      "Evening, {name}! ☕ I'm here when you are.",
  ]},
  { from: 21, to: 5, lines: [
      "Up late, {name}? 🌙 I'm here for you.",
      "Late night, {name}! 🌙 Ready when you are.",
  ]},
];

let greetingTimer = null;

function makeGreetingText() {
  const h = new Date().getHours();
  let bucket = GREETINGS[3];
  for (const g of GREETINGS) {
    if (h >= g.from && h < g.to) { bucket = g; break; }
  }
  const line = bucket.lines[Math.floor(Math.random() * bucket.lines.length)];
  return line.replace(/\{name\}/g, displayName());
}

function showGreeting() {
  removeGreeting();
  const text = makeGreetingText();
  const wrap = document.createElement("div");
  wrap.className = "greeting";
  const textEl = document.createElement("div");
  textEl.className = "greeting-text";
  wrap.appendChild(textEl);
  messagesEl.appendChild(wrap);
  scrollBottom();

  // Type it out live with an elapsed-time rAF loop, so it can never fall
  // behind even if the tab is throttled — it always catches up to the correct
  // character position.
  const cursor = document.createElement("span");
  cursor.className = "cursor";
  const start = performance.now();
  const speed = 16; // ms per character
  textEl.appendChild(cursor);
  const tick = (now) => {
    const i = Math.min(text.length, Math.max(1, Math.floor((now - start) / speed)));
    textEl.textContent = text.slice(0, i);
    textEl.appendChild(cursor);
    scrollBottom();
    if (i >= text.length) {
      cursor.remove();
      greetingTimer = null;
      return;
    }
    greetingTimer = requestAnimationFrame(tick);
  };
  greetingTimer = requestAnimationFrame(tick);
}

function removeGreeting() {
  if (greetingTimer) { cancelAnimationFrame(greetingTimer); greetingTimer = null; }
  const g = messagesEl.querySelector(".greeting");
  if (g) g.remove();
}

async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text || sending) return;
  removeGreeting();

  addMessage("user", text);
  inputEl.value = "";
  autosize();
  sending = true;
  sendBtn.disabled = true;

  const { bubble } = addMessage("assistant", "");
  const cursor = document.createElement("span");
  cursor.className = "cursor";
  bubble.appendChild(cursor);

  let answer = "";
  let sources = [];

  try {
    const resp = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({
        messages: [{ role: "user", content: text }],
        stream: true,
        conversation_id: conversationId,
        collection: searchCollection || null,
      }),
    });
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, idx).trim();
        buffer = buffer.slice(idx + 1);
        if (!line.startsWith("data: ")) continue;
        const data = line.slice(6);
        if (data === "[DONE]") continue;
        try {
          const ev = JSON.parse(data);
          const delta = ev.choices && ev.choices[0] && ev.choices[0].delta && ev.choices[0].delta.content;
          if (delta) {
            answer += delta;
            cursor.remove();
            renderAnswer(bubble, answer);
            scrollBottom();
          }
          if (ev.sources) sources = ev.sources;
          if (ev.conversation_id) { conversationId = ev.conversation_id; currentConvId = conversationId; }
        } catch (e) { /* ignore malformed lines */ }
      }
    }
  } catch (err) {
    cursor.remove();
    bubble.textContent = "⚠️ " + err.message;
  } finally {
    cursor.remove();
    if (answer) renderAnswer(bubble, answer);
    renderSources(bubble.parentElement, sources);
    sending = false;
    sendBtn.disabled = false;
    inputEl.focus();
    loadHistory();
  }
}

/* ---------------- sidebar / uploads ---------------- */

async function loadSidebar() {
  try {
    const url = searchCollection
      ? "/documents?collection=" + encodeURIComponent(searchCollection)
      : "/documents";
    const docs = await (await fetch(url)).json();
    docListEl.innerHTML = "";
    const totalChunks = docs.reduce((a, d) => a + (d.chunks || 0), 0);
    if (!docs.length) {
      const li = document.createElement("li");
      li.className = "empty";
      li.textContent = searchCollection
        ? "No docs in this table yet — upload some!"
        : "No documents yet — upload one!";
      docListEl.appendChild(li);
    } else {
      // Render every doc in the selected table; the list has a ~5-row viewport
      // with a vertical scrollbar, so long tables stay compact and scrollable.
      docs.forEach((d) => {
        const li = document.createElement("li");
        li.innerHTML = '<span class="doc-title">' + escapeHtml(d.title) + "</span>" +
          '<span class="badge">' + d.chunks + " chunks</span>";
        li.title = (d.collection || "") + " • " + d.source_type + " • " +
          new Date(d.created_at).toLocaleString();
        docListEl.appendChild(li);
      });
    }
    statsEl.innerHTML =
      '<div class="stat"><div class="n">' + docs.length + '</div><div class="l">docs</div></div>' +
      '<div class="stat"><div class="n">' + totalChunks + '</div><div class="l">chunks</div></div>';
  } catch (e) { /* ignore */ }
}

async function loadCollections() {
  try {
    const cols = await (await fetch("/collections")).json();
    const sel = document.getElementById("collectionSelect");
    const prev = sel.value;
    sel.innerHTML = '<option value="">All collections</option>';
    for (const c of cols) {
      const opt = document.createElement("option");
      opt.value = c.name;
      opt.textContent = c.name + " (" + c.docs + " docs)";
      sel.appendChild(opt);
    }
    if (prev) sel.value = prev;
  } catch (e) { /* ignore */ }
}

/* ---------------- live upload progress ---------------- */

// Phase stepper: guarantees every stage (Uploading → Extracting → Chunking →
// Embedding → Done) is visible on screen for at least UPHASE_DWELL ms, even
// when the backend races through a stage between two polls (e.g. instant
// extraction of small text files). While a held phase lags the real progress,
// it shows a representative anchor percent instead of a misleading real one.
const UPHASE_ORDER = ["Uploading", "Extracting", "Chunking", "Embedding", "Done"];
const UPHASE_ANCHOR = { Uploading: 5, Extracting: 20, Chunking: 30, Embedding: null, Done: 100 };
const UPHASE_DWELL = 500;
let upDisp = "Uploading"; // phase currently displayed
let upDispAt = 0;         // when the displayed phase was set

function uploadProgressPhaseFor(p) {
  if (p >= 100) return "Done";
  if (p >= 35) return "Embedding";
  if (p >= 30) return "Chunking";
  if (p >= 10) return "Extracting";
  return "Uploading";
}

function uploadProgressShow(file, idx, total) {
  const el = document.getElementById("uploadProgress");
  el.classList.remove("hidden", "success");
  const sp = el.querySelector(".upload-spinner");
  sp.classList.remove("done");
  sp.textContent = "";
  document.querySelector(".upload-progress-bar").style.display = "";
  document.getElementById("upFile").textContent =
    (total > 1 ? (idx + 1) + "/" + total + " · " : "") + file.name;
  upDisp = "Uploading";
  upDispAt = 0;
  uploadProgressUpdate(0);
}

function uploadProgressUpdate(percent) {
  const p = Math.max(0, Math.min(100, Math.round(percent || 0)));
  const target = uploadProgressPhaseFor(p);
  const now = Date.now();
  const ti = UPHASE_ORDER.indexOf(target);
  const di = UPHASE_ORDER.indexOf(upDisp);
  if (ti > di && now - upDispAt >= UPHASE_DWELL) {
    // Backend is ahead of what we've shown — advance one stage at a time so
    // every phase stays visible for its dwell period.
    upDisp = UPHASE_ORDER[di + 1];
    upDispAt = now;
  } else if (ti <= di) {
    // Already at (or past) the target — follow the real progress.
    upDisp = target;
    upDispAt = now;
  }
  const shown = ti === di ? p : (UPHASE_ANCHOR[upDisp] ?? p);
  document.getElementById("upPhase").textContent = upDisp + " — " + shown + "%";
  document.getElementById("upFill").style.width = shown + "%";
}

function uploadProgressHide() {
  document.getElementById("uploadProgress").classList.add("hidden");
}

// Completion notification shown at the top of the chat window once a file
// finishes ("Uploaded ...") — auto-dismisses after a few seconds.
function uploadProgressDone(message, ok = true) {
  const el = document.getElementById("uploadProgress");
  el.classList.add("success");
  const sp = el.querySelector(".upload-spinner");
  sp.classList.add("done");
  sp.textContent = ok ? "✅" : "⚠️";
  document.querySelector(".upload-progress-bar").style.display = "none";
  document.getElementById("upFile").textContent = "";
  document.getElementById("upPhase").textContent = (ok ? "Uploaded " : "⚠️ ") + message;
  clearTimeout(uploadProgressDone._t);
  uploadProgressDone._t = setTimeout(uploadProgressHide, 4200);
}

function uploadProgressError(msg) {
  uploadProgressDone(msg || "Upload failed", false);
}

let UPLOAD_CONFIG = { max_upload_mb: 0, max_upload_files: 0 };

async function loadUploadConfig() {
  try {
    const resp = await fetch("/api/config");
    if (resp.ok) UPLOAD_CONFIG = { ...UPLOAD_CONFIG, ...(await resp.json()) };
  } catch (e) { /* keep defaults */ }
}

async function pollUploadProgress(uploadId, stop, file, idx, total) {
  while (!stop.done) {
    try {
      const resp = await fetch("/ingest/progress/" + encodeURIComponent(uploadId), { cache: "no-store" });
      if (resp.ok) {
        const p = await resp.json();
        if (p.status === "error") { stop.done = true; return; }
        if (p.status === "done") { uploadProgressUpdate(100); stop.done = true; return; }
        document.getElementById("upFile").textContent =
          (total > 1 ? (idx + 1) + "/" + total + " · " : "") + file.name;
        uploadProgressUpdate(p.percent);
      }
    } catch (e) { /* keep polling */ }
    await new Promise((res) => setTimeout(res, 350));
  }
}

async function uploadFiles(files) {
  // Apply the per-batch file-count limit (extra files are dropped with a notice).
  const maxFiles = UPLOAD_CONFIG.max_upload_files > 0 ? UPLOAD_CONFIG.max_upload_files : files.length;
  const batch = files.slice(0, maxFiles);
  if (batch.length < files.length) {
    toast("⚠️ Only " + maxFiles + " files can be uploaded at once — " + (files.length - batch.length) + " skipped");
  }
  const total = batch.length;
  if (!total) return;

  // Upload one file at a time (sequential), showing each file's own progress.
  for (let idx = 0; idx < total; idx++) {
    const f = batch[idx];
    // Client-side size pre-check (server enforces it too).
    if (UPLOAD_CONFIG.max_upload_mb > 0 && f.size > UPLOAD_CONFIG.max_upload_mb * 1024 * 1024) {
      uploadProgressDone("\u201C" + f.name + "\u201D exceeds the " + UPLOAD_CONFIG.max_upload_mb + " MB limit", false);
      continue;
    }
    uploadProgressShow(f, idx, total);
    const uploadId = (crypto.randomUUID && crypto.randomUUID()) ||
      ("u" + Date.now().toString(36) + Math.random().toString(36).slice(2));
    const fd = new FormData();
    fd.append("file", f);
    fd.append("collection", uploadCollection || "default");
    const stop = { done: false };
    const poller = pollUploadProgress(uploadId, stop, f, idx, total);
    let resp;
    try {
      resp = await fetch("/ingest?upload_id=" + encodeURIComponent(uploadId), { method: "POST", body: fd });
    } catch (err) {
      stop.done = true;
      uploadProgressError(err.message);
      continue;
    }
    stop.done = true;
    await poller.catch(() => {});
    const j = await resp.json().catch(() => ({}));
    if (j.skipped) {
      uploadProgressDone("\u201C" + f.name + "\u201D already exists in table \u201C" + j.collection + "\u201D", false);
    } else if (j.note) {
      uploadProgressDone(f.name + " — " + j.note, false);
    } else if (!resp.ok) {
      uploadProgressDone(f.name + ": " + (j.detail || "HTTP " + resp.status), false);
    } else {
      uploadProgressDone(f.name + " → " + j.chunks + " chunks (table: " + j.collection + ")");
    }
  }
  loadCollections();
  loadSidebar();
}

/* ---------------- events ---------------- */

sendBtn.addEventListener("click", sendMessage);
document.getElementById("loginBtn").addEventListener("click", doLogin);
document.getElementById("loginToggle").addEventListener("click", () => setLoginMode(loginMode === "login" ? "register" : "login"));
document.getElementById("loginName").addEventListener("keydown", (e) => { if (e.key === "Enter") doLogin(); });
document.getElementById("loginPass").addEventListener("keydown", (e) => { if (e.key === "Enter") doLogin(); });
document.getElementById("userChip").addEventListener("click", toggleProfileMenu);
document.getElementById("profileResetBtn").addEventListener("click", showPasswordForm);
document.getElementById("pwCancelBtn").addEventListener("click", resetPasswordForm);
document.getElementById("pwUpdateBtn").addEventListener("click", updatePassword);
document.getElementById("pwNew").addEventListener("keydown", (e) => { if (e.key === "Enter") updatePassword(); });
document.getElementById("profileLogoutBtn").addEventListener("click", logout);
// Clicking outside the profile menu closes it (unsaved password edits are discarded).
document.addEventListener("click", (e) => {
  const menu = document.getElementById("profileMenu");
  if (menu.classList.contains("hidden")) return;
  if (menu.contains(e.target) || document.getElementById("userChip").contains(e.target)) return;
  closeProfileMenu();
});
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeProfileMenu(); });
newChatBtn.addEventListener("click", () => {
  conversationId = null;
  currentConvId = null;
  messagesEl.innerHTML = "";
  showGreeting();
  loadHistory();
  inputEl.focus();
});

document.getElementById("collectionSelect").addEventListener("change", (e) => {
  searchCollection = e.target.value;
  uploadCollection = searchCollection || "default";
  document.getElementById("uploadCollection").value = uploadCollection;
  loadSidebar();
});
document.getElementById("uploadCollection").addEventListener("change", (e) => {
  uploadCollection = e.target.value.trim() || "default";
});

// Clicking a [n] citation in the answer scrolls to and highlights its source card.
messagesEl.addEventListener("click", (e) => {
  const link = e.target.closest(".cite-link");
  if (!link) return;
  const card = messagesEl.querySelector('.source-card[data-citation="' + link.dataset.cite + '"]');
  if (card) {
    card.scrollIntoView({ behavior: "smooth", block: "center" });
    card.classList.remove("flash");
    void card.offsetWidth; // restart the highlight animation
    card.classList.add("flash");
  }
});

inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

function autosize() {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + "px";
}
inputEl.addEventListener("input", autosize);

["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("dragover"); }));
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("dragover"); }));
dropzone.addEventListener("drop", (e) => uploadFiles(Array.from(e.dataTransfer.files)));
fileInput.addEventListener("change", () => { uploadFiles(Array.from(fileInput.files)); fileInput.value = ""; });

/* ---------------- boot ---------------- */

currentUser = loadUser();
updateUserChip();
if (currentUser && currentUser.token) {
  // Validate the stored token; on success load history, else re-login.
  fetch("/api/me", { headers: authHeaders() })
    .then((resp) => {
      if (resp.ok) return resp.json();
      throw new Error("unauthorized");
    })
    .then((user) => {
      currentUser.user = user;
      saveUser(currentUser);
      updateUserChip();
      loadHistory();
    })
    .catch(() => {
      currentUser = null;
      try { localStorage.removeItem("rag_user"); } catch (e) {}
      updateUserChip();
      showLogin();
    });
} else {
  showLogin();
}
loadUploadConfig();
loadCollections();
loadSidebar();
showGreeting();
inputEl.focus();
