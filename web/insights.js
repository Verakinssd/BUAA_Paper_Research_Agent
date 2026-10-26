/* Demo panel for a research run: stage runtimes and the first-stage candidate ranking.
 * Self-contained (no dependency on app.js). The data comes from the job snapshot the
 * server already returns: metrics.json, trace events and candidates.json (rank_score).
 * Pure helpers come first so they can be unit-tested without a DOM. */
const RANKING_LABELS = {
  basic: "基础排序",
  bm25: "BM25",
  dense_linear: "稠密向量线性融合",
  hybrid_rrf: "混合 RRF",
};

function insightEscape(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
}

function insightSeconds(value) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  if (value < 10) return `${value.toFixed(1)} 秒`;
  const seconds = Math.round(value);
  if (seconds < 60) return `${seconds} 秒`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)} 分 ${seconds % 60} 秒`;
  return `${Math.floor(seconds / 3600)} 时 ${Math.floor(seconds % 3600 / 60)} 分`;
}

function insightNumber(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

/* Stage durations. Prefer metrics.json (exact); fall back to trace event timestamps. */
function insightStages(job) {
  const metrics = job?.metrics || {};
  const events = Array.isArray(job?.events) ? job.events : [];
  const at = (name) => {
    const hits = events.filter((item) => item?.event === name);
    return hits.length ? insightNumber(hits[hits.length - 1].time) : null;
  };
  const between = (from, to) => {
    const start = at(from), end = at(to);
    return start !== null && end !== null && end >= start ? end - start : null;
  };
  const search = insightNumber(metrics.search_seconds) ?? between("search_started", "sources_searched");
  const process = insightNumber(metrics.processing_seconds) ?? between("task_agent_selected", "report_started");
  const select = between("selection_started", "task_agent_selected");
  const report = between("report_started", "report_completed");
  const total = insightNumber(metrics.server_total_seconds);
  const searchRow = { key: "search", label: "多源检索", hint: "检索、去重与第一阶段排序", seconds: search };
  const processRow = { key: "process", label: "全文处理", hint: "下载 PDF、转换 Markdown、逐篇总结", seconds: process };
  let rows;
  if (select === null && report === null && total !== null && search !== null && process !== null) {
    // The trace tail was truncated, so select and report cannot be split; show the remainder.
    rows = [searchRow, processRow, { key: "select_report", label: "筛选 + 报告", hint: "任务 Agent 选出 Top-k，并生成综合报告", seconds: Math.max(0, total - search - process) }];
  } else {
    rows = [searchRow,
      { key: "select", label: "任务 Agent 筛选", hint: "从候选中选出 Top-k", seconds: select },
      processRow,
      { key: "report", label: "综合报告", hint: "整合各篇总结生成研究报告", seconds: report }];
  }
  const known = rows.filter((row) => row.seconds !== null);
  const sum = known.reduce((acc, row) => acc + row.seconds, 0);
  for (const row of rows) row.share = row.seconds !== null && sum > 0 ? row.seconds / sum : null;
  return { rows, sum, hasData: known.length > 0, total: total ?? (known.length ? sum : null) };
}

/* First-stage candidate ranking, plus where the task Agent's Top-k came from. */
function insightRanking(job) {
  const candidates = Array.isArray(job?.candidates) ? job.candidates : [];
  const picked = new Set((Array.isArray(job?.papers) ? job.papers : []).map((paper) => String(paper.id)));
  const rows = candidates.map((item, index) => ({
    order: index,
    id: String(item.id ?? ""),
    title: String(item.title || "未命名论文"),
    year: item.year ? String(item.year) : "",
    sources: Array.isArray(item.sources) ? item.sources.map(String) : [],
    score: insightNumber(item.rank_score),
  }));
  rows.sort((a, b) => (b.score ?? -Infinity) - (a.score ?? -Infinity) || a.order - b.order);
  const best = rows.reduce((acc, row) => Math.max(acc, row.score ?? 0), 0);
  rows.forEach((row, index) => {
    row.rank = index + 1;
    row.picked = picked.has(row.id);
    row.width = row.score !== null && best > 0 ? Math.max(2, Math.round(row.score / best * 100)) : 0;
  });
  const method = job?.metrics?.sources?.ranking_method;
  return {
    rows,
    method: typeof method === "string" ? method : "",
    methodLabel: typeof method === "string" ? (RANKING_LABELS[method] || method) : "",
    pickedRanks: rows.filter((row) => row.picked).map((row) => row.rank),
  };
}

function insightChips(job, ranking) {
  const chips = [];
  if (job?.model_label || job?.model_id) chips.push(`模型 ${job.model_label || String(job.model_id).toUpperCase()}`);
  chips.push(`Top ${Number(job?.top_k || job?.metrics?.requested_top_k || 3)}`);
  chips.push(job?.concurrent ? "论文并发 2" : "论文串行");
  if (ranking.methodLabel) chips.push(`排序 ${ranking.methodLabel}`);
  return chips;
}

function insightsHtml(job) {
  const stages = insightStages(job);
  const ranking = insightRanking(job);
  const chips = insightChips(job, ranking);
  const stageList = stages.rows.map((row) => {
    const width = row.share === null ? 0 : Math.max(2, Math.round(row.share * 100));
    const percent = row.share === null ? "" : ` · ${Math.round(row.share * 100)}%`;
    return `<li class="stage-row"><div class="stage-head"><span class="stage-name">${insightEscape(row.label)}</span><span class="stage-value">${insightSeconds(row.seconds)}${percent}</span></div><div class="stage-bar" aria-hidden="true"><i style="width:${width}%"></i></div><span class="stage-hint">${insightEscape(row.hint)}</span></li>`;
  }).join("");
  const totalLine = stages.total !== null ? `<p class="insight-total">服务端合计 <strong>${insightSeconds(stages.total)}</strong>${job?.elapsed_seconds ? `（含排队与启动，页面计时 ${insightSeconds(Number(job.elapsed_seconds))}）` : ""}</p>` : "";
  const picked = ranking.pickedRanks.length
    ? `<p class="insight-note">任务 Agent 入选的论文在初排中的名次：${ranking.pickedRanks.map((rank) => `<b>#${rank}</b>`).join("、")}。</p>` : "";
  const table = ranking.rows.map((row) => `<tr class="${row.picked ? "picked" : ""}"><td class="rank-no">${row.rank}</td><td class="rank-id">${insightEscape(row.id)}</td><td class="rank-title" title="${insightEscape(row.title)}">${insightEscape(row.title)}${row.year ? `<span class="rank-year">${insightEscape(row.year)}</span>` : ""}</td><td class="rank-score"><div class="rank-bar" aria-hidden="true"><i style="width:${row.width}%"></i></div><span>${row.score === null ? "—" : row.score.toFixed(3)}</span></td><td class="rank-flag">${row.picked ? '<span class="picked-tag">入选</span>' : ""}</td></tr>`).join("");
  return `<div class="insight-chips">${chips.map((chip) => `<span class="insight-chip">${insightEscape(chip)}</span>`).join("")}</div>
<div class="insight-grid">
<section class="insight-block" aria-label="运行时间"><h3 class="insight-title">运行时间</h3>${stages.hasData ? `<ul class="stage-list">${stageList}</ul>${totalLine}` : '<p class="insight-empty">各阶段完成后显示耗时。</p>'}</section>
<section class="insight-block" aria-label="候选论文排序"><h3 class="insight-title">候选论文排序${ranking.methodLabel ? `<span class="method-pill">${insightEscape(ranking.methodLabel)}</span>` : ""}<span class="insight-count">${ranking.rows.length} 篇</span></h3>${ranking.rows.length ? `${picked}<div class="rank-table-wrap"><table class="rank-table"><thead><tr><th>名次</th><th>ID</th><th>标题</th><th>排序分数</th><th></th></tr></thead><tbody>${table}</tbody></table></div>` : '<p class="insight-empty">检索完成后显示候选论文的排序结果。</p>'}</section>
</div>`;
}

function renderInsights(job) {
  const card = document.getElementById("insight-card");
  if (!card) return;
  const hasCandidates = Array.isArray(job?.candidates) && job.candidates.length > 0;
  const hasStages = insightStages(job).hasData;
  card.hidden = !(hasCandidates || hasStages);
  if (!card.hidden) card.querySelector("#insight-body").innerHTML = insightsHtml(job);
}
