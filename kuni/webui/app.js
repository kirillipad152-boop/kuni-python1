"use strict";
// Every value coming from the server is inserted as text, never as markup.
const $ = (id) => document.getElementById(id);
let state = null, tab = "overview", logKind = "flow", logSeq = 0, lastSettings = "", usageHours = 24, selectedTurnChat = null;

function el(tag, props, ...kids) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (key === "class") node.className = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value);
  }
  for (const kid of kids.flat()) if (kid != null) node.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  return node;
}

async function api(path, body) {
  const options = body === undefined ? { credentials: "same-origin" }
    : { method: "POST", credentials: "same-origin", headers: { "Content-Type": "application/json", "X-Kuni-Panel": "1" },
        body: JSON.stringify(body) };
  const response = await fetch(path, options);
  let data = {};
  try { data = await response.json(); } catch (_) { /* not JSON */ }
  if (response.status === 401 && path !== "/api/login") showLogin();
  if (!response.ok) throw new Error(data.error || "HTTP " + response.status);
  return data;
}

function toast(text, bad) { const t = $("toast"); t.textContent = text; t.className = bad ? "bad" : ""; }
async function act(body, confirmText) {
  if (confirmText && !confirm(confirmText)) return false;
  let ok = true;
  try { toast((await api("/api/action", body)).message || "Готово."); } catch (e) { toast(e.message, true); ok = false; }
  refresh(); loadTab();
  return ok;
}

function dur(seconds) {
  if (seconds == null) return "—";
  const d = Math.floor(seconds / 86400), h = Math.floor(seconds % 86400 / 3600), m = Math.floor(seconds % 3600 / 60);
  return d ? `${d} д ${h} ч` : h ? `${h} ч ${m} мин` : m ? `${m} мин ${seconds % 60} с` : `${seconds} с`;
}
const who = (id, title) => (title ? `${title} (${id})` : String(id));
const empty = (text) => el("p", { class: "muted" }, text);
const clock = (ts) => new Date(ts * 1000).toLocaleTimeString("ru-RU");

function showLogin() { $("panel").hidden = true; $("login").hidden = false; }
function showPanel() { $("login").hidden = true; $("panel").hidden = false; }

async function login(token) {
  try { await api("/api/login", { token }); $("login-error").textContent = ""; showPanel(); await refresh(); await loadLogs(true); }
  catch (e) { $("login-error").textContent = e.message; showLogin(); }
}

function card(title, value, hint) { return el("div", { class: "card" }, el("span", {}, title), el("b", {}, value), hint ? el("span", {}, hint) : null); }

// ---------------------------------------------------------------------------------------------------------------
// overview + state-driven parts (also feeds the access and API tabs)
// ---------------------------------------------------------------------------------------------------------------
function render(s) {
  const badge = $("badge");
  badge.textContent = s.stopping ? "останавливается" : s.paused ? "на паузе" : "работает";
  badge.className = "badge " + (s.paused || s.stopping ? "warn" : "ok");
  $("version").textContent = s.version;
  $("btn-pause").textContent = s.paused ? "Продолжить" : "Поставить на паузу";
  $("btn-restart").disabled = !s.can_restart;
  $("btn-restart").title = s.can_restart ? "" : "Доступно только под systemd";
  const l = s.llm;
  $("cards").replaceChildren(
    card("Аптайм", dur(s.uptime)), card("Модель", s.model), card("Доступ", s.lockdown),
    card("Очередь", s.queue.length), card("Узлов памяти", s.memory.nodes, `${s.memory.edges} связей`),
    card("Напоминаний", s.reminders.length), card("429 за 10 мин", l.rate_limits_10min,
      l.last_success_ago == null ? "успехов не было" : `успех ${dur(l.last_success_ago)} назад`),
    card("Ключей API", l.keys, l.keys_cooling ? `${l.keys_cooling} остывают` : "все свободны"),
    card("Диск свободно", s.disk_free_percent + "%"), card("Память процесса", s.rss_mb == null ? "—" : s.rss_mb + " МБ"),
    card("Контекстов", s.contexts), card("Предупреждения", s.alerts.length ? s.alerts.join(", ") : "нет"));

  const snapshot = JSON.stringify(s.settings);
  if (snapshot !== lastSettings && !$("settings").contains(document.activeElement)) {
    lastSettings = snapshot;
    $("settings").replaceChildren(...s.settings.map(settingRow));
  }
  $("workers").replaceChildren(...(s.workers.length ? s.workers.map((w) => el("div", { class: "item" },
    el("span", { class: "grow" }, `Воркер ${w.id}: ` + (w.sleeping ? "спит" : w.busy ? "занят" : "свободен") +
      (w.chat_id ? ` — ${w.title || w.chat_id} (${dur(w.age)})` : "")))) : [empty("нет")]));
  $("queue").replaceChildren(...(s.queue.length ? s.queue.map((q) => el("div", { class: "item" },
    el("span", { class: "grow" }, `${q.kind}: ${q.title || q.chat_id || "—"}`), el("span", { class: "muted" },
      `приоритет ${q.priority}, ждёт ${dur(q.age)}`))) : [empty("пусто")]));
  $("reminders").replaceChildren(...(s.reminders.length ? s.reminders.map((r) => el("div", { class: "item" },
    el("span", { class: "grow" }, `${r.due} · ${r.title || r.chat_id}: ${r.note}` + (r.attempts ? ` (повторов доставки: ${r.attempts})` : "")),
    el("button", { class: "warn", onclick: () => act({ action: "cancel_reminder", id: r.id }, "Отменить напоминание?") }, "Отменить")))
    : [empty("нет")]));
  renderAccess(s); renderKeys(s); renderJobs(s);
}

function settingRow(item) {
  let control;
  if (item.type === "bool") {
    control = el("input", { type: "checkbox", id: "s-" + item.key });
    control.checked = item.value;
    control.addEventListener("change", () => act({ action: "set", key: item.key, value: control.checked }));
  } else if (item.type === "enum") {
    control = el("select", { id: "s-" + item.key }, item.options.map((o) => el("option", { value: o }, o)));
    control.value = item.value;
    control.addEventListener("change", () => act({ action: "set", key: item.key, value: control.value },
      item.key === "lockdown" ? "Изменить режим доступа?" : null));
  } else {
    control = el("input", { type: "number", id: "s-" + item.key, min: item.min, max: item.max, step: item.type === "int" ? 1 : 0.01 });
    control.value = item.value;
    control.addEventListener("change", () => act({ action: "set", key: item.key, value: Number(control.value) }));
  }
  return el("div", { class: "setting" }, el("label", { for: "s-" + item.key }, item.label), control);
}

function renderAccess(s) {
  const row = (c, buttons) => el("div", { class: "item" }, el("span", { class: "grow" }, who(c.chat_id, c.title),
    c.count ? el("small", {}, `сообщений отброшено: ${c.count}, последнее ${dur(c.ago)} назад`) : null), buttons);
  const listAct = (action, list, id) => () => act({ action, list, chat_id: id });
  $("denied").replaceChildren(...(s.denied.length ? s.denied.map((c) => row(c, [
    el("button", { onclick: listAct("list_add", "allowed", c.chat_id) }, "Разрешить"),
    el("button", { class: "warn", onclick: listAct("list_add", "blocked", c.chat_id) }, "Заблокировать")])) : [empty("никого")]));
  for (const which of ["allowed", "blocked"]) {
    $("list-" + which).replaceChildren(...(s.lists[which].length ? s.lists[which].map((c) => row(c,
      el("button", { class: "warn", onclick: listAct("list_remove", which, c.chat_id) }, "Убрать"))) : [empty("пусто")]));
  }
}

function renderKeys(s) {
  $("env-warn").hidden = !s.env_key_override;
  const head = el("tr", {}, ...["Ключ", "Роль", "Запросов", "Успешных", "429", "Ошибок", "Был использован", "Состояние", ""].map((t) => el("th", {}, t)));
  const rows = s.keys.map((k) => el("tr", {}, el("td", {}, k.hint + " · " + k.id.slice(0, 6)), el("td", {}, k.primary ? "основной" : "дополнительный"),
    el("td", {}, k.requests), el("td", {}, k.ok), el("td", {}, k.rate_limited), el("td", {}, k.errors),
    el("td", {}, k.last_used_ago == null ? "—" : dur(k.last_used_ago) + " назад"),
    el("td", {}, k.cooling ? `остывает ${k.cooling} с` : "свободен"),
    el("td", {}, el("button", { onclick: () => act({ action: "key_test", id: k.id }) }, "Проверить"),
      k.primary ? null : el("button", { class: "warn", onclick: () => act({ action: "key_remove", id: k.id }, "Убрать ключ из пула?") }, "Убрать"))));
  $("keys").replaceChildren(s.keys.length ? el("table", {}, head, ...rows) : empty("ключей нет"));
}

function renderJobs(s) {
  const parts = s.jobs.running.map((n) => (n === "memory_dump" ? "сохранение памяти идёт…" : "«сон» идёт…"));
  for (const [name, text] of Object.entries(s.jobs.last)) if (!s.jobs.running.includes(name)) parts.push((name === "memory_dump" ? "сохранение: " : "сон: ") + text);
  $("mem-jobs").textContent = parts.join(" · ");
  $("mem-dump").disabled = s.jobs.running.includes("memory_dump");
  $("mem-sleep").disabled = s.jobs.running.includes("memory_sleep");
}

async function refresh() {
  if (document.hidden || $("panel").hidden) return;
  try { state = await api("/api/state"); render(state); } catch (e) { if (!$("panel").hidden) toast(e.message, true); }
}

// ---------------------------------------------------------------------------------------------------------------
// tabs
// ---------------------------------------------------------------------------------------------------------------
function loadTab() {
  if (document.hidden || $("panel").hidden) return;
  ({ turns: loadTurns, channels: loadChannels, memory: loadMemory, api: loadUsage, backups: loadBackups, logs: () => loadLogs(true) }[tab] || (() => {}))();
}

async function loadTurns() {
  try {
    const { chats } = await api("/api/traces");
    $("turn-chats").replaceChildren(...(chats.length ? chats.map((c) => el("div", { class: "item click" + (c.chat_id === selectedTurnChat ? " sel" : ""),
      onclick: () => { selectedTurnChat = c.chat_id; loadTurns(); } },
      el("span", { class: "grow" }, c.chat_id === 0 ? "Самостоятельные размышления" : who(c.chat_id, c.title),
        el("small", {}, `${clock(c.started)} · ${c.outcome}`)))) : [empty("ходов ещё не было")]));
    if (selectedTurnChat != null) renderTurns(await api("/api/traces?chat_id=" + selectedTurnChat));
  } catch (e) { toast(e.message, true); }
}

function renderTurns(data) {
  const view = $("turn-view");
  if (!data.turns.length) { view.replaceChildren(empty("для этого чата ходов нет")); return; }
  view.replaceChildren(...data.turns.map((t) => el("div", { class: "turn" },
    el("h3", {}, `${clock(t.started)} · ${t.kind}` + (t.message_id ? ` · сообщение ${t.message_id}` : "") + " → " + t.outcome),
    ...t.steps.map((st) => el("div", { class: "step" },
      el("span", { class: "muted" }, st.t),
      st.note ? el("pre", { class: "note" }, st.note) : null,
      st.thought ? el("details", {}, el("summary", { class: "muted" }, "мысли"), el("pre", { class: "thought" }, st.thought)) : null,
      st.said ? el("pre", {}, "говорит: " + st.said) : null,
      ...st.calls.map((c) => el("pre", { class: "call" }, `▶ ${c.name}(${c.args})`)),
      ...st.results.map((r) => el("pre", { class: "res" }, `◀ ${r.name}: ${r.text}`)))))));
}

async function loadChannels() {
  try {
    const { channels } = await api("/api/channels");
    $("channels").replaceChildren(...(channels.length ? channels.map((c) => el("div", { class: "item" },
      el("span", { class: "grow" }, c.title || c.chat_id, el("small", {}, `${c.chat_id} · непрочитанных: ${c.unread}` + (c.blocked ? " · в чёрном списке" : ""))),
      el("button", { onclick: () => act({ action: "channel_read", chat_id: c.chat_id }) }, "Прочитать"),
      el("button", { class: "warn", onclick: () => act({ action: "list_" + (c.blocked ? "remove" : "add"), list: "blocked", chat_id: c.chat_id }) },
        c.blocked ? "Из чёрного списка" : "В чёрный список"),
      el("button", { class: "danger", onclick: () => act({ action: "channel_leave", chat_id: c.chat_id }, "Выйти из канала?") }, "Выйти"))) : [empty("каналов нет")]));
  } catch (e) { $("channels").replaceChildren(empty(e.message)); }
}

async function loadMemory() {
  try {
    const q = encodeURIComponent($("mem-q").value.trim());
    const data = await api("/api/memory?q=" + q);
    $("mem-count").textContent = `найдено ${data.found} из ${data.total}, показано ${data.nodes.length}`;
    $("mem-list").replaceChildren(...(data.nodes.length ? data.nodes.map((n) => el("div", { class: "item click", onclick: () => showNode(n.id) },
      el("span", { class: "grow" }, `${n.name} [${n.type}]`, el("small", {}, n.summary)))) : [empty("ничего")]));
    const { chats } = await api("/api/working_memory");
    const select = $("wm-chat"), current = select.value;
    select.replaceChildren(...chats.map((c) => el("option", { value: c.chat_id }, `${who(c.chat_id, c.title)} · ${c.size} Б`)));
    if (current && [...select.options].some((o) => o.value === current)) select.value = current;
    await loadWorkingMemory();
  } catch (e) { toast(e.message, true); }
}

async function showNode(id) {
  try {
    const n = await api("/api/memory?id=" + encodeURIComponent(id));
    $("mem-node").replaceChildren(el("h3", {}, `${n.name} [${n.type}]`),
      el("p", { class: "muted" }, `уверенность ${n.confidence} · ${n.visibility}` + (n.source_chat_id ? ` · чат ${n.source_chat_id}` : "") +
        (n.aliases.length ? ` · алиасы: ${n.aliases.join(", ")}` : "")),
      el("pre", { class: "thought" }, n.summary),
      ...n.edges.map((e) => el("div", { class: "item" }, el("span", { class: "grow" },
        `${e.outgoing ? "→" : "←"} ${e.relation} ${e.other} (${e.confidence})`, el("small", {}, e.fact)))),
      el("button", { class: "danger", onclick: async () => { if (await act({ action: "memory_forget", id: n.id }, "Забыть узел вместе со связями? Это необратимо.")) $("mem-node").replaceChildren(empty("забыто")); } }, "Забыть узел"));
  } catch (e) { toast(e.message, true); }
}

async function loadWorkingMemory() {
  const id = $("wm-chat").value;
  if (!id) return;
  try { $("wm-text").value = (await api("/api/working_memory?chat_id=" + encodeURIComponent(id))).text; } catch (e) { toast(e.message, true); }
}

async function loadUsage() {
  try {
    const { series, totals } = await api("/api/usage?hours=" + usageHours);
    $("usage-totals").textContent = `за период: запросов ${totals.req}, успешных ${totals.ok}, 429 — ${totals.r429}, ошибок ${totals.err}; токенов: ${totals.prompt} → ${totals.completion}`;
    drawChart(series);
  } catch (e) { toast(e.message, true); }
}

function drawChart(series) {
  const canvas = $("usage-chart"), ctx = canvas.getContext("2d"), w = canvas.width, h = canvas.height, pad = 28;
  ctx.clearRect(0, 0, w, h);
  const max = Math.max(5, ...series.map((p) => p.req)), bar = (w - pad * 2) / series.length;
  ctx.font = "11px sans-serif"; ctx.fillStyle = "#8b93a1"; ctx.fillText(String(max), 2, pad - 12); ctx.fillText("0", 12, h - pad + 12);
  series.forEach((p, i) => {
    const x = pad + i * bar, scale = (h - pad * 2) / max, okH = p.ok * scale, limH = p.r429 * scale, errH = p.err * scale;
    let y = h - pad;
    for (const [height, color] of [[okH, "#4ec27b"], [limH, "#e5b04a"], [errH, "#ef6b6b"]]) { ctx.fillStyle = color; ctx.fillRect(x + 1, y - height, Math.max(1, bar - 2), height); y -= height; }
    if (i % Math.ceil(series.length / 8) === 0) { ctx.fillStyle = "#8b93a1"; ctx.fillText(p.hour.slice(5, 10) + " " + p.hour.slice(11) + "ч", x, h - 8); }
  });
  ctx.fillStyle = "#4ec27b"; ctx.fillText("успешные", w - 190, 14); ctx.fillStyle = "#e5b04a"; ctx.fillText("429", w - 120, 14); ctx.fillStyle = "#ef6b6b"; ctx.fillText("ошибки", w - 85, 14);
}

async function loadBackups() {
  try {
    const data = await api("/api/backups");
    $("backup-note").textContent = data.configured ? "Архивы на сервере. На компьютер их забирает scripts/kuni-backup-pull." : "misc.backup_dir не задан.";
    $("backups").replaceChildren(...(data.archives.length ? data.archives.map((a) => el("div", { class: "item" },
      el("span", { class: "grow" }, a.name), el("span", { class: "muted" }, `${Math.round(a.size / 1024)} КБ · ${dur(a.age)} назад`))) : [empty("архивов нет")]));
  } catch (e) { toast(e.message, true); }
}

async function loadLogs(reset) {
  if (document.hidden || $("panel").hidden || tab !== "logs") return;
  if (reset) { logSeq = 0; $("log").replaceChildren(); }
  try {
    const { entries } = await api(`/api/logs?kind=${logKind}&after=${logSeq}`);
    const box = $("log"), stick = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
    for (const e of entries) {
      logSeq = Math.max(logSeq, e.seq);
      box.append(el("div", { class: e.level }, el("span", { class: "t" }, e.t + " "), `${e.name}: ${e.text}`));
    }
    while (box.childElementCount > 500) box.firstChild.remove();
    if (stick) box.scrollTop = box.scrollHeight;
  } catch (_) { /* shown by refresh */ }
}

// ---------------------------------------------------------------------------------------------------------------
// wiring
// ---------------------------------------------------------------------------------------------------------------
$("login-form").addEventListener("submit", (e) => { e.preventDefault(); login($("token").value.trim()); $("token").value = ""; });
$("logout").addEventListener("click", async () => { try { await api("/api/logout", {}); } catch (_) { /* ignore */ } showLogin(); });
$("btn-pause").addEventListener("click", () => act({ action: state && state.paused ? "resume" : "pause" }));
$("btn-wake").addEventListener("click", () => act({ action: "wake" }));
$("btn-proactive").addEventListener("click", () => act({ action: "proactive" }));
$("btn-clear").addEventListener("click", () => act({ action: "clear_queue" }, "Очистить всю очередь? Пропущенные сообщения останутся непрочитанными."));
$("btn-restart").addEventListener("click", () => act({ action: "restart" }, "Перезапустить Куни? Она сохранит память и вернётся через пару минут."));
$("send-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const chat = Number($("send-chat").value), text = $("send-text").value;
  if (!Number.isInteger(chat) || !confirm(`Отправить от имени Куни в чат ${chat}?`)) return;
  if (await act({ action: "send_message", chat_id: chat, text })) $("send-text").value = "";
});
$("list-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const chat = Number($("list-chat").value);
  if (Number.isInteger(chat) && await act({ action: "list_add", list: $("list-which").value, chat_id: chat })) $("list-chat").value = "";
});
$("key-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = $("key-new"), key = input.value.trim();
  input.value = "";  // the key is never kept in the page
  if (key) await act({ action: "key_add", key });
});
$("channels-refresh").addEventListener("click", loadChannels);
$("mem-dump").addEventListener("click", () => act({ action: "memory_dump" }));
$("mem-sleep").addEventListener("click", () => act({ action: "memory_sleep" }, "Запустить «сон»? Он делает много запросов к API и может идти долго."));
$("mem-form").addEventListener("submit", (e) => { e.preventDefault(); loadMemory(); });
$("wm-chat").addEventListener("change", loadWorkingMemory);
$("wm-save").addEventListener("click", () => act({ action: "save_working_memory", chat_id: Number($("wm-chat").value), text: $("wm-text").value },
  "Перезаписать рабочую память этого чата?"));
$("backup-now").addEventListener("click", () => act({ action: "backup_now" }));
for (const button of document.querySelectorAll("#tabs button")) button.addEventListener("click", () => {
  tab = button.dataset.tab;
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("active", b === button));
  document.querySelectorAll(".tab").forEach((t) => { t.hidden = t.id !== "tab-" + tab; });
  loadTab();
});
for (const button of document.querySelectorAll(".logtab")) button.addEventListener("click", () => {
  document.querySelectorAll(".logtab").forEach((b) => b.classList.toggle("active", b === button));
  logKind = button.dataset.kind; loadLogs(true);
});
for (const button of document.querySelectorAll(".range")) button.addEventListener("click", () => {
  document.querySelectorAll(".range").forEach((b) => b.classList.toggle("active", b === button));
  usageHours = Number(button.dataset.hours); loadUsage();
});

setInterval(refresh, 3000);
setInterval(() => { if (tab === "logs") loadLogs(false); else if (tab === "turns") loadTurns(); }, 3000);
setInterval(() => { if (tab === "channels" || tab === "api" || tab === "backups") loadTab(); }, 30000);
(async () => {
  const token = decodeURIComponent(location.hash.slice(1));
  if (token) {
    history.replaceState(null, "", location.pathname);  // the token never stays in the address bar or history
    await login(token);
    return;
  }
  try { await api("/api/state"); showPanel(); await refresh(); } catch (_) { showLogin(); }
})();
