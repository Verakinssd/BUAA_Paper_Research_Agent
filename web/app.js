/* Local, dependency-free research workspace. All paper/model text is untrusted. */
"use strict";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));
const activeStatuses = new Set(["queued", "running"]);
const statusLabels = { queued: "等待开始", running: "研究中", completed: "已完成", partial: "部分完成", failed: "未完成", cancelled: "已停止" };
const phases = [
  { id: "search", title: "多源检索", description: "连接公开学术文献" },
  { id: "select", title: "智能筛选", description: "寻找相关的研究证据" },
  { id: "read", title: "全文阅读", description: "提取方法、发现与局限" },
  { id: "report", title: "形成报告", description: "梳理线索与研究建议" },
];
const account = { user: null, mode: "login", pending: false, epoch: 0, users: [], total: 0, page: 1, pageSize: 20, search: "", searchTimer: null, adminVersion: 0, editing: null };
const state = { jobs: [], job: null, currentId: null, activeJobId: null, resourceBusy: false, submitting: false, model: null, models: null, switching: false, modelPoll: null, statusRefreshing: false, view: "home", tab: "papers", poll: null, requestVersion: 0 };
const modelBusy = (catalog = state.models) => Boolean(state.model?.mode !== "remote" && catalog?.managed && ["unloading", "loading", "recovering"].includes(catalog.status));

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
}
function icon(name) { return `<svg class="icon" aria-hidden="true"><use href="#i-${name}"/></svg>`; }
function safeUrl(value) {
  try { const parsed = new URL(String(value)); return ["https:", "http:"].includes(parsed.protocol) ? parsed.href : null; }
  catch { return null; }
}
function badge(status) { return `<span class="status-pill ${Object.hasOwn(statusLabels, status) ? status : "failed"}">${status === "running" ? '<span class="status-dot connected"></span>' : ""}${escapeHtml(statusLabels[status] || "未知状态")}</span>`; }
function parseDate(value) { return new Date(typeof value === "number" ? value * 1000 : value); }
function dateLabel(value) {
  const date = parseDate(value);
  return Number.isNaN(date.getTime()) ? "时间未知" : new Intl.DateTimeFormat("zh-CN", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }).format(date);
}
function duration(value) {
  const seconds = Math.max(0, Math.floor(Number(value) || 0));
  return seconds < 60 ? `${seconds} 秒` : seconds < 3600 ? `${Math.floor(seconds / 60)} 分 ${seconds % 60} 秒` : `${Math.floor(seconds / 3600)} 时 ${Math.floor(seconds % 3600 / 60)} 分`;
}
function toast(message, type = "info") {
  const item = document.createElement("div");
  item.className = `toast ${type}`;
  item.textContent = message;
  $("#toasts").append(item);
  setTimeout(() => item.remove(), 5500);
}
async function api(path, options = {}) {
  const epoch = account.epoch;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 20000);
  try {
    const response = await fetch(path, { credentials: "same-origin", ...options, signal: controller.signal, headers: { ...(options.body && !(options.body instanceof Blob) ? { "Content-Type": "application/json" } : {}), ...options.headers } });
    const body = await response.json().catch(() => ({}));
    if (epoch !== account.epoch) throw new Error("账号已切换，请重新操作。");
    if (response.status === 401 && !["/api/auth/login", "/api/auth/register"].includes(path)) requireLogin("登录状态已失效，请重新登录。");
    if (!response.ok) throw new Error(typeof body.error === "string" ? body.error : `请求未完成（HTTP ${response.status}）`);
    return body;
  } catch (error) {
    if (error.name === "AbortError") throw new Error("请求超时，请检查网页服务是否仍在运行。");
    if (error instanceof TypeError) throw new Error("无法连接网页服务，请检查连接后重试。");
    throw error;
  } finally { clearTimeout(timeout); }
}

// Only generated tags are allowed. Raw HTML is always escaped; links accept HTTP(S).
function inlineMarkdown(value, depth = 0) {
  const text = String(value ?? "");
  if (depth > 4) return escapeHtml(text);
  const pattern = /(`[^`\n]+`|\[[^\]\n]+\]\([^\s)]+\)|\*\*[^*\n]+\*\*|\*[^*\n]+\*)/g;
  let result = "", position = 0;
  for (const match of text.matchAll(pattern)) {
    result += escapeHtml(text.slice(position, match.index));
    const token = match[0];
    if (token.startsWith("`")) result += `<code>${escapeHtml(token.slice(1, -1))}</code>`;
    else if (token.startsWith("[")) {
      const split = token.indexOf("](");
      const href = safeUrl(token.slice(split + 2, -1));
      const label = inlineMarkdown(token.slice(1, split), depth + 1);
      result += href ? `<a href="${escapeHtml(href)}" target="_blank" rel="noopener noreferrer">${label}</a>` : label;
    } else if (token.startsWith("**")) result += `<strong>${inlineMarkdown(token.slice(2, -2), depth + 1)}</strong>`;
    else result += `<em>${inlineMarkdown(token.slice(1, -1), depth + 1)}</em>`;
    position = match.index + token.length;
  }
  return result + escapeHtml(text.slice(position));
}
function markdown(value) {
  const lines = String(value || "").replace(/\r\n?/g, "\n").split("\n");
  let html = "", i = 0;
  const tableSeparator = (line) => /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(line || "");
  const cells = (line) => line.trim().replace(/^\||\|$/g, "").split("|").map((cell) => cell.trim());
  const blockStart = (line) => /^\s*(#{1,6}\s|```|[-*+]\s|\d+[.)]\s|>\s?|---+$)/.test(line);
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i++; continue; }
    if (/^\s*```/.test(line)) {
      const code = []; i++;
      while (i < lines.length && !/^\s*```/.test(lines[i])) code.push(lines[i++]);
      html += `<pre><code>${escapeHtml(code.join("\n"))}</code></pre>`; i++; continue;
    }
    const heading = line.match(/^\s*(#{1,6})\s+(.+)$/);
    if (heading) { html += `<h${heading[1].length}>${inlineMarkdown(heading[2])}</h${heading[1].length}>`; i++; continue; }
    if (/^\s*([-*_])(?:\s*\1){2,}\s*$/.test(line)) { html += "<hr>"; i++; continue; }
    if (line.includes("|") && tableSeparator(lines[i + 1])) {
      html += `<div class="table-scroll"><table><thead><tr>${cells(line).map((cell) => `<th>${inlineMarkdown(cell)}</th>`).join("")}</tr></thead><tbody>`;
      i += 2;
      while (i < lines.length && lines[i].includes("|") && lines[i].trim()) html += `<tr>${cells(lines[i++]).map((cell) => `<td>${inlineMarkdown(cell)}</td>`).join("")}</tr>`;
      html += "</tbody></table></div>"; continue;
    }
    if (/^\s*>/.test(line)) {
      const quote = [];
      while (i < lines.length && /^\s*>/.test(lines[i])) quote.push(lines[i++].replace(/^\s*>\s?/, ""));
      html += `<blockquote>${quote.map((item) => inlineMarkdown(item)).join("<br>")}</blockquote>`; continue;
    }
    const unordered = /^\s*[-*+]\s+/.test(line), ordered = /^\s*\d+[.)]\s+/.test(line);
    if (unordered || ordered) {
      const tag = ordered ? "ol" : "ul", expression = ordered ? /^\s*\d+[.)]\s+/ : /^\s*[-*+]\s+/;
      html += `<${tag}>`;
      while (i < lines.length && expression.test(lines[i])) html += `<li>${inlineMarkdown(lines[i++].replace(expression, ""))}</li>`;
      html += `</${tag}>`; continue;
    }
    const paragraph = [line]; i++;
    while (i < lines.length && lines[i].trim() && !blockStart(lines[i]) && !tableSeparator(lines[i + 1])) paragraph.push(lines[i++]);
    html += `<p>${paragraph.map((item) => inlineMarkdown(item)).join("<br>")}</p>`;
  }
  return html;
}

function closeSidebar() {
  $("#sidebar").classList.remove("is-open");
  $("#mobile-overlay").hidden = true;
  $("#mobile-menu").setAttribute("aria-expanded", "false");
}
function showView(view) {
  if (!account.user) return;
  if (view === "admin" && account.user.role !== "admin") view = "home";
  state.view = view;
  for (const name of ["home", "history", "run", "profile", "admin"]) $(`#${name}-view`).hidden = name !== view;
  $("#breadcrumb-current").textContent = { home: "工作台", history: "研究记录", run: "研究详情", profile: "个人空间", admin: "用户管理" }[view];
  for (const name of ["home", "history", "profile", "admin"]) {
    const button = $(`#nav-${name}`);
    button.classList.toggle("active", name === view);
    if (name === view) button.setAttribute("aria-current", "page"); else button.removeAttribute("aria-current");
  }
  if (view !== "run") { clearTimeout(state.poll); state.requestVersion++; }
  $("#account-dropdown").hidden = true;
  $("#account-trigger").setAttribute("aria-expanded", "false");
  if (view === "profile") { populateProfile(); loadAPISettings(); }
  if (view === "admin") loadAdminUsers();
  if (view !== "run") document.title = `${$("#breadcrumb-current").textContent} · Paper`;
  closeSidebar();
  window.scrollTo({ top: 0, behavior: "instant" });
}
function updateComposer() {
  const characters = $("#idea").value.length;
  $("#idea-count").textContent = `${characters} / 2000`;
  const busy = Boolean(state.activeJobId || state.resourceBusy);
  const unavailable = !state.model?.connected || state.switching || (state.model?.mode !== "remote" && state.models?.managed && state.models.status !== "ready");
  $("#start-button").disabled = state.submitting || busy || unavailable;
  $("#start-button span").textContent = state.submitting ? "正在开始…" : busy ? "已有研究运行中" : "开始研究";
  $("#research-hint").textContent = busy ? state.activeJobId ? "有一项研究正在运行，可从研究记录查看进度。" : "共享研究资源正在使用中，请稍后开始。" : modelBusy() || state.switching ? "正在加载模型，完成后即可开始研究。" : unavailable ? "模型尚未连接，可在个人空间配置网络 API，或启动本地服务。" : state.model?.mode === "remote" ? "检索真实论文 · 使用你的网络 API 分析（可能产生费用）" : "检索真实论文 · 由本地模型完成分析";
}
function newResearch() {
  showView("home");
  $("#concurrent").checked = false;
  $("#idea").value = "";
  state.currentId = null;
  updateComposer();
  renderHistory();
  $("#idea").focus({ preventScroll: true });
}
async function refreshStatus() {
  if (!account.user || state.statusRefreshing) return;
  const epoch = account.epoch;
  state.statusRefreshing = true;
  clearTimeout(state.modelPoll);
  try {
    const [health, catalog] = await Promise.allSettled([api("/api/status"), api("/api/models")]);
    if (epoch !== account.epoch || !account.user) return;
    state.models = catalog.status === "fulfilled" ? catalog.value : null;
    if (health.status !== "fulfilled") throw health.reason;
    const data = health.value;
    state.model = data.model || { connected: false };
    state.activeJobId = data.active_job_id || null;
    state.resourceBusy = Boolean(data.resource_busy);
    const current = state.model.mode === "remote" ? null : state.models?.models?.find((m) => m.id === state.models.current_id);
    $("#model-name").textContent = current ? `${current.id.toUpperCase()} · Qwen3.8-27B` : state.model.name || "本地模型";
    const connected = state.model.connected && !modelBusy();
    $("#model-state").innerHTML = `<span class="status-dot ${connected ? "connected" : "offline"}"></span>${modelBusy() ? "模型切换中…" : connected ? state.model.mode === "remote" ? "网络 API · 已配置" : "已连接 · 就绪" : "尚未连接"}`;
    $("#refresh-status").title = state.model.detail || "重新检查模型连接";
  } catch {
    if (!account.user || epoch !== account.epoch) return;
    state.model = { connected: false };
    $("#model-state").innerHTML = '<span class="status-dot offline"></span>网页服务未连接';
  } finally {
    state.statusRefreshing = false;
  }
  renderModelControls();
  updateComposer();
  if (state.model?.mode !== "remote" && (modelBusy() || (state.models?.managed && state.models.status === "ready" && !state.model?.connected))) state.modelPoll = setTimeout(refreshStatus, 1200);
}
function renderModelControls() {
  const isAdmin = account.user?.role === "admin";
  const catalog = state.models, select = $("#model-select");
  if (state.model?.mode === "remote") {
    select.innerHTML = `<option>${escapeHtml(state.model.name)} · 我的网络 API</option>`;
    select.disabled = true;
    $("#switch-model").hidden = true;
    $("#switch-model").disabled = true;
    $("#model-loading-label").textContent = "个人配置";
    $("#model-switch-hint").textContent = "使用你保存的 API Key；可在个人空间测试连接或切回本地模型。";
    $("#model-switch-hint").classList.remove("model-hint-error");
    return;
  }
  const entries = Array.isArray(catalog?.models) ? catalog.models : [];
  const chosen = select.value || catalog?.target_id || catalog?.current_id || "q4_k_m";
  select.innerHTML = entries.length ? entries.map((m) => {
    const percent = Math.min(99, Math.floor(100 * Number(m.downloaded_bytes || 0) / Number(m.expected_bytes || 1)));
    const suffix = m.id === catalog.current_id ? " · 当前" : m.file_status === "downloading" ? ` · 下载中 ${percent}%` : !m.available ? " · 文件未就绪" : "";
    return `<option value="${escapeHtml(m.id)}" ${!m.available ? "disabled" : ""}>${escapeHtml(m.label + suffix)}</option>`;
  }).join("") : '<option value="">模型切换尚未连接</option>';
  if (entries.some((m) => m.id === chosen)) select.value = chosen;
  const pending = state.switching || modelBusy(catalog), occupied = Boolean(state.activeJobId || state.resourceBusy || catalog?.blocked_by_job);
  select.disabled = !isAdmin || !catalog?.managed || pending || occupied;
  const selected = entries.find((m) => m.id === select.value);
  $("#switch-model").hidden = !isAdmin;
  $("#switch-model").disabled = !isAdmin || !catalog?.managed || pending || occupied || !selected?.available || (catalog.status === "ready" && selected.id === catalog.current_id);
  $("#switch-model").textContent = pending ? "正在加载…" : "加载模型";
  const target = entries.find((m) => m.id === catalog?.target_id);
  $("#model-loading-label").textContent = pending ? `${catalog?.status === "recovering" ? "正在恢复原模型" : "正在切换模型"}${target ? ` · ${target.id.toUpperCase()}` : ""}` : catalog?.current_id ? `当前 ${catalog.current_id.toUpperCase()}` : "";
  $("#model-switch-hint").textContent = !isAdmin ? "使用当前共享模型进行研究；模型切换由管理员负责。" : catalog?.error || (!catalog?.managed ? "网页切换需启用模型服务：python deploy/serve_gguf.py --managed" : occupied ? "研究运行期间锁定模型；完成或停止任务后可切换。" : pending ? "正在卸载和加载模型，请稍候。此过程不会同时加载两份权重。" : "Q5 使用单槽位优先预留显存。未下载完成的模型无法选择；点击左下角刷新可更新文件状态。");
  $("#model-switch-hint").classList.toggle("model-hint-error", Boolean(catalog?.error));
}
async function switchModel() {
  if ($("#switch-model").disabled) return;
  state.switching = true;
  renderModelControls(); updateComposer();
  try {
    state.models = await api("/api/models/switch", { method: "POST", body: JSON.stringify({ model_id: $("#model-select").value }) });
    toast("已开始切换模型，加载状态会自动更新。");
  } catch (error) { toast(error.message, "error"); }
  finally { state.switching = false; renderModelControls(); updateComposer(); await refreshStatus(); }
}
function jobCard(job) {
  const count = Number(job.completed_count || 0);
  return `<button class="research-card" data-job-id="${escapeHtml(job.id)}"><span class="research-card-top"><span class="tag">${icon("paper")}文献研究</span>${badge(job.status)}</span><span class="research-card-title">${escapeHtml(job.idea || "未命名研究")}</span><span class="research-card-meta"><span>${icon("clock")}${escapeHtml(dateLabel(job.created_at))}</span><span>${count} 篇已读 ${icon("arrow")}</span></span></button>`;
}
function emptyState(title, description) { return `<div class="empty-state"><span class="empty-icon">${icon("book")}</span><h3>${escapeHtml(title)}</h3><p>${escapeHtml(description)}</p></div>`; }
function renderHistory() {
  $("#history-count").textContent = state.jobs.length;
  $("#stat-papers").textContent = state.jobs.reduce((total, job) => total + Number(job.completed_count || 0), 0);
  $("#stat-research").textContent = state.jobs.filter((job) => job.status === "completed").length;
  $("#sidebar-history").innerHTML = state.jobs.length ? state.jobs.slice(0, 7).map((job) => `<button class="history-button ${job.id === state.currentId ? "selected" : ""}" data-job-id="${escapeHtml(job.id)}"><span class="history-title">${icon("paper")}${escapeHtml(job.idea || "未命名研究")}</span><span class="history-meta">${escapeHtml(dateLabel(job.created_at))}<span class="history-status ${activeStatuses.has(job.status) ? "running" : ""}">${escapeHtml(statusLabels[job.status] || "未完成")}</span></span></button>`).join("") : '<p class="sidebar-empty">每一个好问题，都值得留下。</p>';
  $("#recent-jobs").innerHTML = state.jobs.length ? state.jobs.slice(0, 4).map(jobCard).join("") : emptyState("你的研究旅程，从这里开始", "输入一个研究想法，开始检索和阅读论文。");
  const query = $("#history-search").value.trim().toLocaleLowerCase();
  const filtered = state.jobs.filter((job) => String(job.idea || "").toLocaleLowerCase().includes(query));
  $("#all-jobs").innerHTML = filtered.length ? filtered.map(jobCard).join("") : emptyState(query ? "没有找到匹配的研究" : "还没有研究记录", query ? "换一个关键词试试。" : "完成的研究将自动保存在这里。");
  updateProfileStats();
}
async function refreshHistory(announce = false) {
  if (!account.user) return;
  try {
    const data = await api("/api/jobs");
    state.jobs = Array.isArray(data.jobs) ? data.jobs.filter((job) => !job.mock) : [];
    renderHistory();
    if (announce) toast("研究记录已更新");
  } catch (error) { if (announce) toast(error.message, "error"); }
}

function switchTab(tab) {
  state.tab = tab;
  for (const name of ["papers", "report", "logs"]) {
    const button = $(`#tab-${name}`);
    button.classList.toggle("active", name === tab);
    button.setAttribute("aria-selected", String(name === tab));
    button.tabIndex = name === tab ? 0 : -1;
    $(`#${name}-panel`).hidden = name !== tab;
  }
}
function renderProgress(job) {
  const terminal = !activeStatuses.has(job.status);
  let current = phases.findIndex((phase) => phase.id === job.stage);
  if (job.stage === "done" || job.report) current = 4;
  if (current < 0) current = 0;
  $("#progress-label").textContent = terminal ? ({ completed: "研究已完成，发现已经就绪", partial: "研究部分完成，请查看下方说明", failed: "研究未完成，请查看错误与日志", cancelled: "研究已停止，已生成的结果仍会保留" }[job.status] || "研究已结束") : job.status === "queued" ? "正在准备研究任务" : `${phases[Math.min(current, 3)].title}中…`;
  $("#progress-count").textContent = `${Number(job.completed_count || 0)} / ${Number(job.top_k || 3)} 篇已完成 · ${duration(job.elapsed_seconds)}`;
  $("#progress-fill").style.width = `${Math.min(100, current * 25 + (terminal ? 0 : 8))}%`;
  $("#pipeline-steps").innerHTML = phases.map((phase, index) => {
    const complete = index < current, active = index === current;
    return `<div class="pipeline-step ${complete ? "complete" : active ? (terminal ? "failed" : "active") : ""}"><span class="step-number">${complete ? icon("check") : index + 1}</span><div><span class="step-label">${phase.title}</span><span class="step-caption">${phase.description}</span></div></div>`;
  }).join("");
}
function renderPapers(job) {
  const opened = new Set($$(".paper-summary[open]").map((item) => item.dataset.paperId));
  const papers = Array.isArray(job.papers) ? job.papers : [];
  $("#paper-count").textContent = papers.length;
  if (!papers.length) {
    $("#paper-list").innerHTML = emptyState(activeStatuses.has(job.status) ? "正在寻找与你的问题相关的论文" : "本次尚未生成精选论文", activeStatuses.has(job.status) ? "检索和筛选完成后，论文会出现在这里。进展可在右侧动态中查看。" : "查看运行日志了解原因，或用更简短的主题重新研究。");
    return;
  }
  $("#paper-list").innerHTML = papers.map((paper, index) => {
    const url = safeUrl(paper.url), year = paper.year || "";
    const authors = (Array.isArray(paper.authors) ? paper.authors : []).map((author) => typeof author === "string" ? author : author.name || "").filter(Boolean);
    const sources = Array.isArray(paper.sources) ? paper.sources : [];
    const title = escapeHtml(paper.title || "未命名论文");
    const pdf = paper.pdf_available === true || (paper.pdf_available === undefined && paper.status === "ok");
    const paperStatus = paper.status === "ok" ? "已完成阅读" : paper.status === "failed" ? "处理未完成" : paper.status === "cancelled" ? "已停止处理" : "等待 / 正在分析";
    return `<article class="paper-card"><div class="paper-rank">${String(index + 1).padStart(2, "0")}</div><div class="paper-content"><div class="paper-meta">${sources.map((source) => `<span class="source-tag">${escapeHtml(source)}</span>`).join("")}${year ? `<span>${escapeHtml(year)}</span>` : ""}<span class="paper-status ${paper.status === "ok" ? "ok" : paper.status === "failed" ? "failed" : "pending"}">${paper.status === "ok" ? icon("check") : icon("clock")}${paperStatus}</span></div><h3 class="paper-title">${title}</h3>${authors.length ? `<p class="paper-authors">${escapeHtml(authors.slice(0, 3).join(" · "))}${authors.length > 3 ? " 等" : ""}</p>` : ""}${paper.selection_reason ? `<div class="paper-reason">${icon("spark")}<p>${escapeHtml(paper.selection_reason)}</p></div>` : ""}${paper.error ? `<div class="notice error">${escapeHtml(paper.error)}</div>` : ""}<div class="paper-actions">${url ? `<a class="text-button" href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">论文来源${icon("external")}</a>` : ""}${pdf ? `<a class="text-button" href="/api/jobs/${encodeURIComponent(job.id)}/papers/${encodeURIComponent(paper.id)}/pdf" target="_blank" rel="noopener noreferrer">${icon("paper")}查看 PDF</a>` : ""}${paper.markdown_available ? `<a class="text-button" href="/api/jobs/${encodeURIComponent(job.id)}/papers/${encodeURIComponent(paper.id)}/markdown" download>${icon("paper")}下载全文 Markdown</a>` : ""}<span class="paper-id">${escapeHtml(paper.id)}</span></div>${paper.summary ? `<details class="paper-summary" data-paper-id="${escapeHtml(paper.id)}" ${opened.has(String(paper.id)) ? "open" : ""}><summary>${icon("book")}阅读全文总结${icon("chevron")}</summary><div class="markdown">${markdown(paper.summary)}</div></details>` : ""}</div></article>`;
  }).join("");
}
function eventDescription(item) {
  const id = item.id ? `${item.id} · ` : "";
  const descriptions = {
    search_started: ["正在检索学术文献", item.query || "查询 arXiv 与 OpenAlex"],
    sources_searched: ["多源检索完成", `合并去重后找到 ${Number(item.deduplicated_count || 0)} 篇候选论文`],
    selection_started: ["正在筛选候选论文", `任务 Agent 正在选择 Top ${Number(item.requested_top_k || 3)}`],
    task_agent_selected: ["确定精选论文", `任务 Agent 已完成 ${Array.isArray(item.selection) ? item.selection.length : ""} 篇论文的筛选`],
    paper_started: [`${id}开始阅读`, item.title || "准备论文全文"],
    pdf_download_started: [`${id}正在下载全文`, "获取公开可访问的 PDF"],
    pdf_download_progress: [`${id}全文下载中`, `已下载 ${(Number(item.downloaded_bytes || 0) / 1024 / 1024).toFixed(1)} MB${item.total_bytes ? ` / ${(Number(item.total_bytes) / 1024 / 1024).toFixed(1)} MB` : ""}`],
    pdf_download_completed: [`${id}全文下载完成`, `${(Number(item.pdf_bytes || 0) / 1024 / 1024).toFixed(1)} MB`],
    paper_cache_hit: [`${id}复用已缓存全文`, "继续进行本次模型分析"],
    pdf_conversion_started: [`${id}正在解析 PDF`, "提取全文文字，准备阅读材料"],
    pdf_conversion_completed: [`${id}全文解析完成`, `${Number(item.markdown_chars || 0).toLocaleString("zh-CN")} 个字符`],
    summary_started: [`${id}开始深度阅读`, "分段整理论文中的证据"],
    summary_chunk_started: [`${id}正在阅读第 ${Number(item.chunk || 1)} 段`, "等待本地模型生成该片段的事实笔记"],
    summary_chunk_completed: [`${id}完成第 ${Number(item.chunk || 1)} 段阅读`, "已提取该片段的事实笔记"],
    summary_completed: [`${id}论文总结完成`, "已整合方法、实验与局限"],
    paper_processed: [`${id}${item.status === "ok" ? "处理完成" : "处理未完成"}`, item.error || "论文总结已保存"],
    report_started: ["正在撰写研究报告", "综合论文证据，梳理研究方向"],
    report_completed: ["报告阶段结束", item.metrics?.task_success ? "所有请求的环节已完成" : "请结合论文数量与失败情况查看结果"],
  };
  return descriptions[item.event] || null;
}
function renderActivity(job) {
  const events = (Array.isArray(job.events) ? job.events : []).map((item) => ({ item, description: eventDescription(item) })).filter((entry) => entry.description).slice(-18).reverse();
  $("#activity-live").hidden = !activeStatuses.has(job.status);
  $("#activity-list").innerHTML = events.length ? events.map(({ item, description }) => `<div class="activity-item"><span class="activity-dot ${item.status === "failed" ? "failed" : ""}"></span><div class="activity-content"><strong>${escapeHtml(description[0])}</strong><p>${escapeHtml(description[1])}</p><time class="activity-time">${escapeHtml(dateLabel(item.time))}</time></div></div>`).join("") : '<p class="activity-empty">任务启动后，实际执行进展会显示在这里。</p>';
  const sources = job.metrics?.sources || {};
  $("#source-health").innerHTML = Object.keys(sources).length ? `<div class="sidebar-label">来源状态</div>${Object.entries(sources).map(([name, source]) => `<div class="source-health-row"><span>${escapeHtml(name)}</span><span class="source-result ${source.status === "failed" ? "failed" : ""}">${source.status === "failed" ? "请求失败" : `${Number(source.count || 0)} 篇`}</span></div>`).join("")}` : "";
}
function renderJob(job) {
  state.job = job;
  $("#run-title").textContent = job.idea || "文献研究";
  $("#run-eyebrow").textContent = activeStatuses.has(job.status) ? "FOLLOW THE EVIDENCE" : "YOUR RESEARCH, COLLECTED";
  $("#run-meta").innerHTML = `${badge(job.status)}<span>文献研究</span>${job.model_id ? `<span>${escapeHtml(job.model_id.toUpperCase())}</span>` : ""}<span>Top ${Number(job.top_k || 3)}</span><span>${escapeHtml(dateLabel(job.created_at))}</span>`;
  $("#cancel-job").hidden = !activeStatuses.has(job.status);
  $("#cancel-job").disabled = false;
  $("#download-report").disabled = !job.report;
  const messages = [];
  if (job.status === "partial") messages.push("本次部分完成。请核对已完成论文数量、来源状态和报告内容。");
  if (job.status === "failed") messages.push(job.error || "研究未完成，请查看运行日志与模型连接状态。");
  if (job.status === "cancelled") messages.push("研究已停止，已生成的论文与分析结果仍可查看。");
  $("#run-notice").hidden = !messages.length;
  $("#run-notice").className = `notice ${job.status === "failed" ? "error" : job.status === "partial" ? "warning" : ""}`;
  $("#run-notice").textContent = messages.join(" ");
  renderProgress(job);
  renderPapers(job);
  renderActivity(job);
  $("#report-content").innerHTML = job.report ? markdown(job.report) : emptyState("让发现慢慢汇成报告", activeStatuses.has(job.status) ? "全文阅读完成后，Agent 会整合论文证据和研究建议。" : "本次没有生成研究报告，可查看日志了解原因。");
  const log = $("#log-output"), nearBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 50;
  log.textContent = (Array.isArray(job.logs) ? job.logs : []).join("\n") || "等待运行日志…";
  if (nearBottom) log.scrollTop = log.scrollHeight;
  document.title = `${activeStatuses.has(job.status) ? "研究中" : statusLabels[job.status] || "研究详情"} · Paper`;
}
async function loadJob(id) {
  clearTimeout(state.poll);
  const version = ++state.requestVersion;
  state.currentId = id;
  try {
    const job = await api(`/api/jobs/${encodeURIComponent(id)}`);
    if (version !== state.requestVersion) return;
    showView("run");
    renderJob(job);
    renderHistory();
    switchTab("papers");
    if (activeStatuses.has(job.status)) schedulePoll(id, version);
  } catch (error) { toast(error.message, "error"); }
}
function schedulePoll(id, version) {
  clearTimeout(state.poll);
  state.poll = setTimeout(async () => {
    if (state.currentId !== id || state.requestVersion !== version || state.view !== "run") return;
    try {
      const job = await api(`/api/jobs/${encodeURIComponent(id)}`);
      if (state.currentId !== id || state.requestVersion !== version) return;
      renderJob(job);
      if (activeStatuses.has(job.status)) schedulePoll(id, version);
      else { state.activeJobId = null; await Promise.allSettled([refreshHistory(), refreshStatus()]); }
    } catch (error) { toast(error.message, "error"); schedulePoll(id, version); }
  }, 1200);
}
async function startResearch(event) {
  event.preventDefault();
  const idea = $("#idea").value.trim();
  if (!idea || state.submitting) { $("#idea").focus(); return; }
  state.submitting = true; updateComposer();
  try {
    const job = await api("/api/jobs", { method: "POST", body: JSON.stringify({ idea, top_k: Number($("#top-k").value), concurrent: $("#concurrent").checked }) });
    state.activeJobId = job.id;
    renderModelControls();
    await refreshHistory();
    await loadJob(job.id);
  } catch (error) { toast(error.message, "error"); await refreshStatus(); }
  finally { state.submitting = false; updateComposer(); }
}

$("#research-form").addEventListener("submit", startResearch);
$("#model-select").addEventListener("change", renderModelControls);
$("#switch-model").addEventListener("click", switchModel);
$("#idea").addEventListener("input", updateComposer);
$("#new-research").addEventListener("click", newResearch);
$("#brand-home").addEventListener("click", (event) => { event.preventDefault(); showView("home"); });
$("#nav-home").addEventListener("click", () => showView("home"));
$("#back-home").addEventListener("click", () => showView("home"));
for (const element of [$("#nav-history"), $("#view-all")]) element.addEventListener("click", () => { showView("history"); refreshHistory(); });
$("#history-search").addEventListener("input", renderHistory);
$("#history-refresh").addEventListener("click", () => refreshHistory(true));
$("#refresh-status").addEventListener("click", async () => { await refreshStatus(); toast(state.model?.detail || "尚未连接模型，请检查个人配置或本地服务。"); });
$$("[data-idea]").forEach((button) => button.addEventListener("click", () => { $("#idea").value = button.dataset.idea; updateComposer(); $("#idea").focus(); }));
document.addEventListener("click", (event) => { const button = event.target.closest("[data-job-id]"); if (button) loadJob(button.dataset.jobId); });
$$("[data-tab]").forEach((button) => button.addEventListener("click", () => switchTab(button.dataset.tab)));
$(".tabs").addEventListener("keydown", (event) => {
  if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
  event.preventDefault();
  const tabs = ["papers", "report", "logs"], current = tabs.indexOf(state.tab);
  const next = event.key === "Home" ? 0 : event.key === "End" ? 2 : (current + (event.key === "ArrowRight" ? 1 : -1) + 3) % 3;
  switchTab(tabs[next]); $(`#tab-${tabs[next]}`).focus();
});
$("#reuse-idea").addEventListener("click", () => {
  if (!state.job) return;
  const job = state.job;
  newResearch();
  $("#idea").value = job.idea || "";
  $("#top-k").value = String(job.top_k || 3);
  $("#concurrent").checked = false;
});
$("#cancel-job").addEventListener("click", async () => {
  if (!state.currentId) return;
  $("#cancel-job").disabled = true;
  try {
    const job = await api(`/api/jobs/${encodeURIComponent(state.currentId)}/cancel`, { method: "POST", body: "{}" });
    clearTimeout(state.poll);
    renderJob(job);
    await Promise.allSettled([refreshHistory(), refreshStatus()]);
  } catch (error) { toast(error.message, "error"); $("#cancel-job").disabled = false; }
});
$("#download-report").addEventListener("click", () => { if (state.job?.report) window.location.assign(`/api/jobs/${encodeURIComponent(state.job.id)}/report`); });
$("#copy-logs").addEventListener("click", async () => {
  try { await navigator.clipboard.writeText($("#log-output").textContent); toast("运行日志已复制"); }
  catch { toast("浏览器未允许访问剪贴板，可直接选中日志复制。", "error"); }
});
$("#help-button").addEventListener("click", () => $("#help-dialog").showModal());
$("#close-help").addEventListener("click", () => $("#help-dialog").close());
$("#help-dialog").addEventListener("click", (event) => { if (event.target === $("#help-dialog")) { const rect = event.target.getBoundingClientRect(); if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) event.target.close(); } });
$("#mobile-menu").addEventListener("click", () => { const open = $("#sidebar").classList.toggle("is-open"); $("#mobile-overlay").hidden = !open; $("#mobile-menu").setAttribute("aria-expanded", String(open)); });
$("#mobile-overlay").addEventListener("click", closeSidebar);
document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") { closeSidebar(); $("#account-dropdown").hidden = true; $("#account-trigger").setAttribute("aria-expanded", "false"); }
  if (!account.user) return;
  if ($(":modal")) return;
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter" && state.view === "home") { event.preventDefault(); if (!$("#start-button").disabled) $("#research-form").requestSubmit(); }
  const editing = event.target.matches("input, textarea, select, [contenteditable]");
  if (!editing && !event.ctrlKey && !event.metaKey && !event.altKey && event.key.toLowerCase() === "n") { event.preventDefault(); newResearch(); }
});
window.addEventListener("beforeunload", () => { clearTimeout(state.poll); clearTimeout(state.modelPoll); });

// Cookie-backed accounts. Clearing a session also clears every private view.
function formError(id, message = "") { const target = $(id); target.textContent = message; target.hidden = !message; }
function roleBadge(user) { return `<span class="role-badge ${user.role === "admin" ? "admin" : ""}">${user.role === "admin" ? icon("shield") : ""}${user.role === "admin" ? "管理员" : "普通用户"}</span>`; }
function avatarMarkup(user) {
  if (user?.avatar_url) {
    try {
      const url = new URL(user.avatar_url, location.origin);
      if (url.origin === location.origin && /^\/api\/users\/[a-zA-Z0-9_-]+\/avatar$/.test(url.pathname)) return `<img src="${escapeHtml(url.pathname + url.search)}" alt="" loading="lazy">`;
    } catch { /* A malformed avatar URL falls back to a name initial. */ }
  }
  return escapeHtml(Array.from(user?.display_name || user?.username || "研")[0].toLocaleUpperCase());
}
function setAuthMode(mode) {
  account.mode = mode;
  const register = mode === "register";
  for (const name of ["login", "register"]) { const tab = $(`#auth-${name}-tab`); tab.classList.toggle("active", mode === name); tab.setAttribute("aria-selected", String(mode === name)); }
  $("#auth-heading").textContent = register ? "为你的好奇心，留个位置" : "欢迎回到你的研究空间";
  $("#auth-description").textContent = register ? "创建账号，开启属于你的研究旅程。" : "登录，让上一次的灵感继续生长。";
  $("#auth-eyebrow").textContent = register ? "A NEW CHAPTER BEGINS" : "WELCOME BACK";
  $("#auth-display-field").hidden = !register;
  $("#auth-confirm-field").hidden = !register;
  $("#auth-confirm-password").required = register;
  $("#auth-password").minLength = register ? 8 : 1;
  $("#auth-password").autocomplete = register ? "new-password" : "current-password";
  $("#auth-password").placeholder = register ? "设置密码，至少 8 位" : "输入密码";
  $("#auth-submit span").textContent = register ? "创建我的研究空间" : "进入研究空间";
  $("#auth-form-hint").textContent = register ? "注册后即为普通用户，拥有独立的研究记录。" : "专注于问题，让每次探索都留下痕迹。";
  formError("#auth-error");
}
function requireLogin(message = "") {
  account.epoch++;
  account.user = null;
  account.editing = null;
  account.users = [];
  account.total = 0;
  account.adminVersion++;
  clearTimeout(state.poll); clearTimeout(state.modelPoll); clearTimeout(account.searchTimer);
  state.requestVersion++;
  state.jobs = []; state.job = null; state.currentId = null; state.activeJobId = null; state.resourceBusy = false; state.model = null; state.models = null; state.submitting = false; state.statusRefreshing = false;
  closeSidebar();
  $$("dialog[open]").forEach((dialog) => dialog.close());
  $("#account-dropdown").hidden = true;
  $("#auth-screen").hidden = false;
  $("#toasts").replaceChildren();
  $("#session-loading").hidden = true;
  document.body.classList.remove("auth-pending");
  document.body.classList.add("is-signed-out");
  $("#idea").value = ""; $("#history-search").value = ""; $("#admin-search").value = "";
  $("#api-settings-form").reset(); personalAPI.config = null; personalAPI.version++; $("#api-key-status").textContent = ""; $("#api-settings-note").textContent = ""; $("#profile-form").reset(); $("#password-form").reset(); $("#admin-edit-form").reset(); $("#admin-reset-form").reset();
  $$("[data-password]").forEach((toggle) => { document.getElementById(toggle.dataset.password).type = "password"; toggle.setAttribute("aria-label", "显示密码"); toggle.setAttribute("aria-pressed", "false"); });
  for (const id of ["paper-list", "report-content", "activity-list", "log-output", "run-title", "run-meta", "run-notice", "source-health", "progress-count", "admin-users", "admin-edit-name", "admin-edit-joined", "profile-display-name", "profile-handle", "profile-bio-preview", "sidebar-user-name", "topbar-user-name", "dropdown-name", "dropdown-username"]) $(`#${id}`).textContent = "";
  for (const id of ["sidebar-avatar", "topbar-avatar", "profile-avatar", "profile-edit-avatar", "admin-edit-avatar"]) $(`#${id}`).textContent = "";
  renderHistory(); updateComposer();
  setAuthMode("login");
  $("#auth-form").reset();
  formError("#auth-error", message);
  document.title = "登录 · Paper 研究空间";
}
function renderIdentity() {
  const user = account.user;
  if (!user) return;
  const name = user.display_name || user.username, admin = user.role === "admin";
  $("#sidebar-user-name").textContent = name;
  $("#topbar-user-name").textContent = name;
  $("#dropdown-name").textContent = name;
  $("#dropdown-username").textContent = `@${user.username}`;
  $("#sidebar-user-role").textContent = admin ? "管理员 · 个人研究空间" : "普通用户 · 个人研究空间";
  $("#topbar-user-role").textContent = admin ? "管理员" : "个人工作区";
  $("#nav-admin").hidden = !admin;
  for (const id of ["sidebar-avatar", "topbar-avatar", "profile-avatar", "profile-edit-avatar"]) $(`#${id}`).innerHTML = avatarMarkup(user);
  $("#profile-display-name").textContent = name;
  $("#profile-handle").textContent = `@${user.username}`;
  $("#profile-bio-preview").textContent = user.bio || "让好奇心，成为你的个性签名。";
  $("#profile-role-badge").className = `role-badge ${admin ? "admin" : ""}`;
  $("#profile-role-badge").textContent = admin ? "管理员" : "普通用户";
  $("#remove-avatar").hidden = !user.avatar_url;
  updateProfileStats();
  renderModelControls();
  if (!admin && state.view === "admin") { $("#admin-user-dialog").close(); showView("home"); }
}
function updateProfileStats() {
  $("#profile-research-count").textContent = state.jobs.filter((job) => job.status === "completed").length;
  $("#profile-paper-count").textContent = state.jobs.reduce((sum, job) => sum + Number(job.completed_count || 0), 0);
  const date = account.user ? parseDate(account.user.created_at) : null;
  $("#profile-joined").textContent = date && !Number.isNaN(date.getTime()) ? new Intl.DateTimeFormat("zh-CN", { year: "numeric", month: "2-digit" }).format(date) : "—";
}
function populateProfile() {
  if (!account.user) return;
  renderIdentity();
  $("#profile-username").value = account.user.username;
  $("#profile-display").value = account.user.display_name || "";
  $("#profile-bio").value = account.user.bio || "";
  $("#bio-count").textContent = `${$("#profile-bio").value.length} / 200`;
  $("#profile-save-note").textContent = "资料仅用于你的 Paper 账号。";
  $("#password-form").reset(); formError("#profile-error"); formError("#password-error");
}
async function enterWorkspace(user) {
  const epoch = ++account.epoch;
  account.user = user;
  account.page = 1; account.search = "";
  $("#auth-screen").hidden = true;
  $("#session-loading").hidden = true;
  document.body.classList.remove("auth-pending", "is-signed-out");
  $("#auth-password").value = ""; $("#auth-confirm-password").value = "";
  renderIdentity(); showView("home");
  await Promise.allSettled([refreshStatus(), refreshHistory()]);
  return epoch;
}
async function bootstrapAccounts() {
  try {
    const data = await api("/api/auth/me");
    if (data.user) await enterWorkspace(data.user); else requireLogin();
  } catch (error) { requireLogin(error.message); }
}
async function refreshSession() {
  try {
    const data = await api("/api/auth/me");
    if (!data.user) { requireLogin("登录状态已失效，请重新登录。"); return; }
    if (!account.user || String(data.user.id) !== String(account.user.id)) { requireLogin("账号已变更，请重新登录。"); return; }
    account.user = data.user;
    renderIdentity();
  } catch { /* Other API requests provide a connection error without discarding a valid session. */ }
}
$("#auth-login-tab").addEventListener("click", () => { if (!account.pending) setAuthMode("login"); });
$("#auth-register-tab").addEventListener("click", () => { if (!account.pending) setAuthMode("register"); });
$(".auth-tabs").addEventListener("keydown", (event) => { if (["ArrowLeft", "ArrowRight"].includes(event.key) && !account.pending) { event.preventDefault(); setAuthMode(account.mode === "login" ? "register" : "login"); $(`#auth-${account.mode}-tab`).focus(); } });
document.addEventListener("click", (event) => {
  const toggle = event.target.closest("[data-password]");
  if (toggle) { const input = document.getElementById(toggle.dataset.password); const visible = input.type === "password"; input.type = visible ? "text" : "password"; toggle.setAttribute("aria-label", visible ? "隐藏密码" : "显示密码"); toggle.setAttribute("aria-pressed", String(visible)); }
  if (!event.target.closest(".account-menu")) { $("#account-dropdown").hidden = true; $("#account-trigger").setAttribute("aria-expanded", "false"); }
  const manage = event.target.closest("[data-manage-user]");
  if (manage && account.user?.role === "admin") openUserEditor(manage.dataset.manageUser);
});
$("#auth-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (account.pending) return;
  const register = account.mode === "register", password = $("#auth-password").value;
  if (register && password !== $("#auth-confirm-password").value) { formError("#auth-error", "两次输入的密码不一致。"); $("#auth-confirm-password").focus(); return; }
  account.pending = true;
  $("#auth-submit").disabled = true; $("#auth-login-tab").disabled = true; $("#auth-register-tab").disabled = true;
  $("#auth-submit span").textContent = register ? "正在创建空间…" : "正在登录…";
  formError("#auth-error");
  try {
    const payload = { username: $("#auth-username").value.trim(), password };
    if (register && $("#auth-display-name").value.trim()) payload.display_name = $("#auth-display-name").value.trim();
    const data = await api(`/api/auth/${register ? "register" : "login"}`, { method: "POST", body: JSON.stringify(payload) });
    const epoch = await enterWorkspace(data.user);
    if (epoch === account.epoch && account.user) toast(register ? "你的研究空间已就绪。欢迎加入 Paper！" : `欢迎回来，${data.user.display_name || data.user.username}。`);
  } catch (error) { formError("#auth-error", error.message); }
  finally { account.pending = false; $("#auth-submit").disabled = false; $("#auth-login-tab").disabled = false; $("#auth-register-tab").disabled = false; $("#auth-submit span").textContent = account.mode === "register" ? "创建我的研究空间" : "进入研究空间"; }
});
$("#account-trigger").addEventListener("click", () => { const dropdown = $("#account-dropdown"); dropdown.hidden = !dropdown.hidden; $("#account-trigger").setAttribute("aria-expanded", String(!dropdown.hidden)); });
for (const selector of ["#nav-profile", "#sidebar-account", "#dropdown-profile"]) $(selector).addEventListener("click", () => showView("profile"));
$("#nav-admin").addEventListener("click", () => showView("admin"));
$("#logout-button").addEventListener("click", async () => {
  $("#logout-button").disabled = true;
  try { await api("/api/auth/logout", { method: "POST", body: "{}" }); requireLogin(); }
  catch (error) { toast(error.message, "error"); }
  finally { $("#logout-button").disabled = false; }
});
$("#profile-bio").addEventListener("input", () => { $("#bio-count").textContent = `${$("#profile-bio").value.length} / 200`; });
$("#profile-form").addEventListener("submit", async (event) => {
  event.preventDefault(); $("#save-profile").disabled = true; formError("#profile-error");
  try {
    const username = $("#profile-username").value.trim();
    const data = await api("/api/auth/profile", { method: "PATCH", body: JSON.stringify({ username, display_name: $("#profile-display").value.trim() || username, bio: $("#profile-bio").value }) });
    account.user = data.user; renderIdentity();
    $("#profile-username").value = data.user.username;
    $("#profile-display").value = data.user.display_name;
    $("#profile-bio").value = data.user.bio || "";
    $("#bio-count").textContent = `${$("#profile-bio").value.length} / 200`;
    $("#profile-save-note").textContent = "已保存，新的资料已同步。";
    toast("个人资料已更新。");
  } catch (error) { formError("#profile-error", error.message); }
  finally { $("#save-profile").disabled = false; }
});
$("#password-form").addEventListener("submit", async (event) => {
  event.preventDefault(); formError("#password-error");
  if ($("#new-password").value !== $("#confirm-new-password").value) { formError("#password-error", "两次输入的新密码不一致。"); return; }
  $("#save-password").disabled = true;
  try {
    const data = await api("/api/auth/password", { method: "POST", body: JSON.stringify({ current_password: $("#current-password").value, new_password: $("#new-password").value }) });
    account.user = data.user; renderIdentity(); $("#password-form").reset(); toast("密码已更新，其他登录会话已退出。");
  } catch (error) { formError("#password-error", error.message); }
  finally { $("#save-password").disabled = false; }
});
$("#upload-avatar").addEventListener("click", () => $("#avatar-file").click());
$("#avatar-file").addEventListener("change", async () => {
  const file = $("#avatar-file").files[0];
  if (!file) return;
  if (!["image/png", "image/jpeg", "image/webp"].includes(file.type) || file.size > 3 * 1024 * 1024) { toast("请选择不超过 3 MB 的 JPG、PNG 或 WebP 图片。", "error"); $("#avatar-file").value = ""; return; }
  $("#upload-avatar").disabled = true; $("#remove-avatar").disabled = true;
  try { const data = await api("/api/auth/avatar", { method: "POST", headers: { "Content-Type": file.type }, body: file }); account.user = data.user; renderIdentity(); toast("新头像已保存。"); }
  catch (error) { toast(error.message, "error"); }
  finally { $("#upload-avatar").disabled = false; $("#remove-avatar").disabled = false; $("#avatar-file").value = ""; }
});
$("#remove-avatar").addEventListener("click", async () => {
  $("#remove-avatar").disabled = true;
  try { const data = await api("/api/auth/avatar", { method: "DELETE" }); account.user = data.user; renderIdentity(); toast("已恢复默认头像。"); }
  catch (error) { toast(error.message, "error"); }
  finally { $("#remove-avatar").disabled = false; }
});
async function loadAdminUsers() {
  if (account.user?.role !== "admin") return;
  const version = ++account.adminVersion;
  $("#admin-users").classList.add("is-loading");
  if (!account.users.length) $("#admin-users").innerHTML = emptyState("正在打开成员列表", "读取空间成员与账号信息。");
  try {
    const query = new URLSearchParams({ search: account.search, page: String(account.page), page_size: String(account.pageSize) });
    const data = await api(`/api/admin/users?${query}`);
    if (version !== account.adminVersion || account.user?.role !== "admin") return;
    account.users = data.users || []; account.total = Number(data.total || 0); account.page = Number(data.page || 1);
    if (!account.search) $("#admin-user-total").textContent = account.total;
    $("#admin-result-count").textContent = account.search ? `找到 ${account.total} 位成员` : `共 ${account.total} 位成员`;
    const pages = Math.max(1, Math.ceil(account.total / account.pageSize));
    $("#admin-page").textContent = `${account.page} / ${pages}`;
    $("#admin-pagination-label").textContent = account.total ? `显示 ${(account.page - 1) * account.pageSize + 1}–${Math.min(account.page * account.pageSize, account.total)} 位成员` : "暂无匹配成员";
    $("#admin-prev").disabled = account.page <= 1;
    $("#admin-next").disabled = account.page >= pages;
    renderAdminUsers();
  } catch (error) { if (version === account.adminVersion && account.user?.role === "admin") { $("#admin-users").innerHTML = emptyState("暂时无法读取成员", error.message); toast(error.message, "error"); } }
  finally { if (version === account.adminVersion) $("#admin-users").classList.remove("is-loading"); }
}
function renderAdminUsers() {
  if (!account.users.length) { $("#admin-users").innerHTML = emptyState("没有找到匹配的成员", "尝试其他用户名或显示名称。"); return; }
  $("#admin-users").innerHTML = `<table class="users-table"><thead><tr><th>成员</th><th>身份</th><th>账号状态</th><th>加入时间</th><th>操作</th></tr></thead><tbody>${account.users.map((user) => `<tr><td><div class="table-user"><span class="user-avatar">${avatarMarkup(user)}</span><div><strong>${escapeHtml(user.display_name || user.username)}${String(user.id) === String(account.user.id) ? '<span class="you-tag">你</span>' : ""}</strong><span>@${escapeHtml(user.username)}</span></div></div></td><td>${roleBadge(user)}</td><td><span class="user-status ${user.is_active ? "" : "inactive"}"><span></span>${user.is_active ? "正常使用" : "已停用"}</span></td><td>${escapeHtml(dateLabel(user.created_at))}</td><td><button type="button" class="secondary-button manage-user" data-manage-user="${escapeHtml(user.id)}">${icon("edit")}管理</button></td></tr>`).join("")}</tbody></table>`;
}
function openUserEditor(id) {
  const user = account.users.find((item) => String(item.id) === String(id));
  if (!user) return;
  account.editing = user;
  const self = String(user.id) === String(account.user.id);
  $("#admin-edit-name").textContent = user.display_name || user.username;
  $("#admin-edit-joined").textContent = `@${user.username} · ${dateLabel(user.created_at)} 加入`;
  $("#admin-edit-avatar").innerHTML = avatarMarkup(user);
  $("#admin-edit-username").value = user.username;
  $("#admin-edit-display").value = user.display_name || "";
  $("#admin-edit-bio").value = user.bio || "";
  $("#admin-edit-role").value = user.role;
  $("#admin-edit-active").value = String(user.is_active);
  $("#admin-edit-active").disabled = self;
  $("#admin-self-hint").textContent = self ? "当前登录账号不可停用。" : "权限变更立即生效。";
  $("#admin-reset-form").hidden = self;
  $(".admin-reset-section > p").textContent = self ? "请在个人空间修改自己的密码。" : "将使此账号的已有登录会话失效。请将新密码私下告知用户。";
  $("#admin-reset-form").reset();
  formError("#admin-edit-error"); formError("#admin-reset-error");
  $("#admin-user-dialog").showModal();
}
$("#admin-search").addEventListener("input", () => { clearTimeout(account.searchTimer); account.searchTimer = setTimeout(() => { account.search = $("#admin-search").value.trim(); account.page = 1; loadAdminUsers(); }, 300); });
$("#admin-refresh").addEventListener("click", loadAdminUsers);
$("#admin-prev").addEventListener("click", () => { if (account.page > 1) { account.page--; loadAdminUsers(); } });
$("#admin-next").addEventListener("click", () => { if (account.page * account.pageSize < account.total) { account.page++; loadAdminUsers(); } });
$("#close-admin-user").addEventListener("click", () => $("#admin-user-dialog").close());
$("#admin-user-dialog").addEventListener("click", (event) => { if (event.target === $("#admin-user-dialog")) { const rect = event.target.getBoundingClientRect(); if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) event.target.close(); } });
$("#admin-edit-form").addEventListener("submit", async (event) => {
  event.preventDefault(); if (!account.editing) return;
  $("#admin-save-user").disabled = true; formError("#admin-edit-error");
  try {
    const username = $("#admin-edit-username").value.trim();
    const data = await api(`/api/admin/users/${encodeURIComponent(account.editing.id)}`, { method: "PATCH", body: JSON.stringify({ username, display_name: $("#admin-edit-display").value.trim() || username, bio: $("#admin-edit-bio").value, role: $("#admin-edit-role").value, is_active: $("#admin-edit-active").value === "true" }) });
    if (String(data.user.id) === String(account.user.id)) { account.user = data.user; renderIdentity(); }
    $("#admin-user-dialog").close(); toast("成员资料与权限已更新。");
    if (account.user.role === "admin") await loadAdminUsers();
  } catch (error) { formError("#admin-edit-error", error.message); }
  finally { $("#admin-save-user").disabled = false; }
});
$("#admin-reset-form").addEventListener("submit", async (event) => {
  event.preventDefault(); if (!account.editing || !$("#admin-reset-confirm").checked) return;
  $("#admin-reset-submit").disabled = true; formError("#admin-reset-error");
  const self = String(account.editing.id) === String(account.user.id);
  try {
    await api(`/api/admin/users/${encodeURIComponent(account.editing.id)}/password`, { method: "POST", body: JSON.stringify({ new_password: $("#admin-reset-password").value }) });
    $("#admin-reset-form").reset();
    if (self) requireLogin("密码已重置，请使用新密码登录。"); else toast("密码已重置，该用户原有登录会话已失效。");
  } catch (error) { formError("#admin-reset-error", error.message); }
  finally { $("#admin-reset-submit").disabled = false; }
});
const personalAPI = { config: null, version: 0, pending: false };
function toggleAPIFields() {
  const remote = $("#api-mode").value === "remote";
  $("#api-remote-fields").hidden = !remote;
  $("#api-remote-fields").disabled = !remote || personalAPI.pending;
  $("#api-test").hidden = !remote;
}
function setAPIBusy(busy) {
  personalAPI.pending = busy;
  for (const id of ["api-mode", "api-save", "api-test", "api-delete"]) $(`#${id}`).disabled = busy;
  toggleAPIFields();
}
function fillAPISettings(config) {
  personalAPI.config = config;
  $("#api-mode").value = config.mode;
  $("#api-base").value = config.base_url;
  $("#api-model").value = config.model;
  $("#api-context").value = config.context_tokens;
  $("#api-key").value = "";
  $("#api-key").type = "password";
  const toggle = $('[data-password="api-key"]');
  toggle.setAttribute("aria-pressed", "false"); toggle.setAttribute("aria-label", "显示密码");
  $("#api-key-status").textContent = config.has_api_key ? "已保存 · 留空保留" : "尚未配置";
  $("#api-key").placeholder = config.has_api_key ? "已保存，留空保留原 Key" : "输入你的 API Key";
  toggleAPIFields();
}
async function loadAPISettings() {
  const version = ++personalAPI.version;
  setAPIBusy(true); formError("#api-settings-error");
  $("#api-settings-note").textContent = "正在读取个人配置…";
  try {
    const config = await api("/api/settings/model");
    if (version !== personalAPI.version || !account.user) return;
    fillAPISettings(config);
    $("#api-settings-note").textContent = config.mode === "remote" ? "当前使用个人网络 API。" : "当前使用本地模型服务。";
  } catch (error) { if (version === personalAPI.version && account.user) { formError("#api-settings-error", error.message); $("#api-settings-note").textContent = "配置读取失败，请重新打开个人空间。"; } }
  finally { if (version === personalAPI.version) setAPIBusy(false); }
}
function apiSettingsPayload() {
  const payload = { mode: $("#api-mode").value, base_url: $("#api-base").value.trim(), model: $("#api-model").value.trim(), context_tokens: Number($("#api-context").value) };
  if ($("#api-key").value.trim()) payload.api_key = $("#api-key").value.trim();
  return payload;
}
$("#open-api-settings").addEventListener("click", () => { showView("profile"); $("#api-settings-form").scrollIntoView({ behavior: "smooth", block: "center" }); });
$("#api-mode").addEventListener("change", toggleAPIFields);
$("#api-settings-form").addEventListener("input", () => { $("#api-settings-note").textContent = "配置已修改，保存后对新任务生效。"; formError("#api-settings-error"); });
$("#api-settings-form").addEventListener("submit", async (event) => {
  event.preventDefault(); if (personalAPI.pending) return;
  const version = personalAPI.version;
  setAPIBusy(true); formError("#api-settings-error");
  try {
    const config = await api("/api/settings/model", { method: "PUT", body: JSON.stringify(apiSettingsPayload()) });
    if (version !== personalAPI.version || !account.user) return;
    fillAPISettings(config); $("#api-settings-note").textContent = "已保存，对下一次研究生效。";
    toast("模型配置已保存。"); await refreshStatus();
  } catch (error) { if (version === personalAPI.version && account.user) formError("#api-settings-error", error.message); }
  finally { if (version === personalAPI.version) setAPIBusy(false); }
});
$("#api-test").addEventListener("click", async () => {
  if (personalAPI.pending || !$("#api-settings-form").reportValidity()) return;
  const version = personalAPI.version;
  setAPIBusy(true); formError("#api-settings-error"); $("#api-settings-note").textContent = "正在测试 API 连接和工具调用…";
  try {
    const result = await api("/api/settings/model/test", { method: "POST", body: JSON.stringify(apiSettingsPayload()) });
    if (version === personalAPI.version && account.user) $("#api-settings-note").textContent = result.message;
  } catch (error) { if (version === personalAPI.version && account.user) { $("#api-settings-note").textContent = "测试未通过，配置尚未保存。"; formError("#api-settings-error", error.message); } }
  finally { if (version === personalAPI.version) setAPIBusy(false); }
});
$("#api-delete").addEventListener("click", async () => {
  if (personalAPI.pending) return;
  const version = personalAPI.version;
  setAPIBusy(true); formError("#api-settings-error");
  try {
    const config = await api("/api/settings/model", { method: "DELETE" });
    if (version !== personalAPI.version || !account.user) return;
    fillAPISettings(config); $("#api-settings-note").textContent = "网络配置已清除，已切回本地模式。"; await refreshStatus();
  } catch (error) { if (version === personalAPI.version && account.user) formError("#api-settings-error", error.message); }
  finally { if (version === personalAPI.version) setAPIBusy(false); }
});
updateComposer();
bootstrapAccounts();
setInterval(async () => { if (!document.hidden && account.user) { await refreshSession(); if (account.user) { refreshStatus(); refreshHistory(); } } }, 15000);
