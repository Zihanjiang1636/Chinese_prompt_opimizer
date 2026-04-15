const defaultUserId = "local-user";
let activeSessionId = null;
let runtimeConfig = null;

const sourcePromptInput = document.getElementById("sourcePrompt");
const taskGoalInput = document.getElementById("taskGoal");
const platformTagInput = document.getElementById("platformTag");
const styleHintInput = document.getElementById("styleHint");
const mustKeepTermsInput = document.getElementById("mustKeepTerms");
const saveSessionInput = document.getElementById("saveSession");
const strategySelect = document.getElementById("strategySelect");
const noticeNode = document.getElementById("notice");
const bestPromptNode = document.getElementById("bestPrompt");
const scoreSummaryNode = document.getElementById("scoreSummary");
const briefExplanationNode = document.getElementById("briefExplanation");
const analysisSummaryNode = document.getElementById("analysisSummary");
const historyListNode = document.getElementById("historyList");
const runtimeConfigNode = document.getElementById("runtimeConfig");
const reportMetaNode = document.getElementById("reportMeta");
const reportSummaryNode = document.getElementById("reportSummary");
const providerPresetListNode = document.getElementById("providerPresetList");

function setNotice(text) {
  noticeNode.textContent = text || "";
}

function splitTerms(value) {
  return value
    .split(/[,，\n]/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function renderCards(target, items) {
  target.innerHTML = items
    .map(
      (item) => `
        <div class="score-card">
          <span>${item.label}</span>
          <span class="value">${item.value}</span>
        </div>
      `,
    )
    .join("");
}

function renderStrategyOptions(config) {
  const strategies = config.strategies || [];
  strategySelect.innerHTML = strategies
    .map(
      (item) => `<option value="${item.id}" ${item.id === (config.default_strategy || "balanced") ? "selected" : ""}>${item.label}</option>`,
    )
    .join("");
}

function renderRuntimeConfig(config) {
  runtimeConfig = config;
  renderCards(runtimeConfigNode, [
    { label: "模板版本", value: config.template_version || "unknown" },
    { label: "LLM 模式", value: config.llm_mode || "unknown" },
    { label: "当前模型", value: config.llm_model || "unknown" },
    { label: "评测样本", value: String(config.dataset_case_count || 0) },
  ]);
  renderStrategyOptions(config);
  providerPresetListNode.innerHTML = (config.provider_presets || [])
    .map(
      (item) => `
        <div class="history-item">
          <strong>${item.label}</strong>
          <div><code>${item.provider}</code> · <code>${item.base_url}</code></div>
          <small>${item.note}</small>
        </div>
      `,
    )
    .join("");
}

function renderResult(result) {
  bestPromptNode.textContent = result.best_prompt || "暂无结果";
  const summary = result.score_summary || {};
  renderCards(scoreSummaryNode, [
    { label: "胜出总分", value: summary.winner_total ?? "-" },
    { label: "原始基线", value: summary.original_baseline_total ?? "-" },
    { label: "直改基线", value: summary.direct_baseline_total ?? "-" },
    { label: "策略", value: result.strategy_label || result.strategy || "-" },
  ]);

  briefExplanationNode.innerHTML = (result.brief_explanation || [])
    .map((item) => `<div class="explanation-item">${item}</div>`)
    .join("");

  const analysis = result.analysis || {};
  const strengths = (analysis.strengths || []).map((item) => `<li>${item}</li>`).join("");
  const risks = (analysis.risks || []).map((item) => `<li>${item}</li>`).join("");
  const culturalSignals = (analysis.cultural_signals || []).map((item) => `<li>${item}</li>`).join("");

  analysisSummaryNode.innerHTML = `
    <p><strong>摘要：</strong>${result.analysis_summary || ""}</p>
    <p><strong>模板版本：</strong>${result.template_version || "unknown"}</p>
    <h3>优势</h3>
    <ul>${strengths || "<li>暂无</li>"}</ul>
    <h3>风险</h3>
    <ul>${risks || "<li>暂无</li>"}</ul>
    <h3>中文语感线索</h3>
    <ul>${culturalSignals || "<li>暂无</li>"}</ul>
  `;
}

function renderHistory(sessions) {
  if (!sessions.length) {
    historyListNode.innerHTML = '<div class="history-item">还没有历史记录。</div>';
    return;
  }

  historyListNode.innerHTML = sessions
    .map(
      (item) => `
        <button class="history-item" data-session-id="${item.session_id}">
          <strong>${item.task_goal}</strong>
          <div>${item.source_prompt}</div>
          <small>${item.strategy_label || item.strategy} · ${item.confidence_band} · 胜出分 ${item.score_summary?.winner_total ?? "-"}</small>
        </button>
      `,
    )
    .join("");

  historyListNode.querySelectorAll("[data-session-id]").forEach((button) => {
    button.addEventListener("click", async () => {
      const sessionId = button.getAttribute("data-session-id");
      if (!sessionId) return;
      const response = await fetch(`/api/prompt-copilot/history/${sessionId}?user_id=${defaultUserId}`);
      const payload = await response.json();
      activeSessionId = sessionId;
      renderResult(payload.data.session);
      setNotice("已载入历史记录。");
    });
  });
}

function renderReport(report, reportName) {
  if (!report) {
    reportMetaNode.textContent = "还没有回归报告。运行 scripts/run_prompt_copilot_regression.py 后这里会出现摘要。";
    reportSummaryNode.innerHTML = "";
    return;
  }

  const summary = report.summary || {};
  reportMetaNode.textContent = `最新报告：${reportName || "unknown"} · 策略 ${summary.strategy || "balanced"} · 样本数 ${summary.case_count || 0}`;
  renderCards(reportSummaryNode, [
    { label: "平均胜出分", value: summary.avg_winner_total ?? "-" },
    { label: "平均原始基线", value: summary.avg_original_baseline_total ?? "-" },
    { label: "平均直改基线", value: summary.avg_direct_baseline_total ?? "-" },
    { label: "双基线胜率", value: summary.both_baselines_beaten_ratio ?? "-" },
    { label: "相对原句提升", value: summary.avg_margin_vs_original ?? "-" },
    { label: "相对直改提升", value: summary.avg_margin_vs_direct ?? "-" },
  ]);
}

async function refreshHistory() {
  const response = await fetch(`/api/prompt-copilot/history?user_id=${defaultUserId}`);
  const payload = await response.json();
  renderHistory(payload.data.sessions || []);
}

async function refreshRuntimeConfig() {
  const response = await fetch("/api/prompt-copilot/config");
  const payload = await response.json();
  renderRuntimeConfig(payload.data || {});
}

async function refreshReport() {
  const response = await fetch("/api/prompt-copilot/report/latest");
  const payload = await response.json();
  renderReport(payload.data.report, payload.data.report_name);
}

document.getElementById("optimizeButton").addEventListener("click", async () => {
  setNotice("正在解析、改写并进行内部筛选…");
  const response = await fetch("/api/prompt-copilot/optimize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      user_id: defaultUserId,
      source_prompt: sourcePromptInput.value,
      task_goal: taskGoalInput.value,
      platform_tag: platformTagInput.value,
      style_hint: styleHintInput.value,
      must_keep_terms: splitTerms(mustKeepTermsInput.value),
      save_session: saveSessionInput.checked,
      strategy: strategySelect.value || (runtimeConfig?.default_strategy ?? "balanced"),
    }),
  });

  const payload = await response.json();
  if (payload.status !== "success") {
    setNotice(payload.message || "优化失败。");
    return;
  }

  activeSessionId = payload.data.session_id;
  renderResult(payload.data);
  setNotice("优化完成，已挑出胜出版本。");
  await refreshHistory();
  await refreshRuntimeConfig();
});

document.getElementById("copyButton").addEventListener("click", async () => {
  await navigator.clipboard.writeText(bestPromptNode.textContent || "");
  setNotice("最优提示词已复制。");
  if (activeSessionId) {
    await fetch(`/api/prompt-copilot/feedback?session_id=${activeSessionId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: defaultUserId, copied: true }),
    });
  }
});

document.getElementById("likeButton").addEventListener("click", async () => {
  if (!activeSessionId) return;
  await fetch(`/api/prompt-copilot/feedback?session_id=${activeSessionId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: defaultUserId, adopted: true, closer_to_goal: true }),
  });
  setNotice("已记录为“更像我要的”。");
});

document.getElementById("dislikeButton").addEventListener("click", async () => {
  if (!activeSessionId) return;
  await fetch(`/api/prompt-copilot/feedback?session_id=${activeSessionId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: defaultUserId, closer_to_goal: false }),
  });
  setNotice("已记录为“还不够像”。");
});

document.getElementById("refreshHistoryButton").addEventListener("click", refreshHistory);
document.getElementById("refreshReportButton").addEventListener("click", refreshReport);

document.getElementById("exampleButton").addEventListener("click", () => {
  sourcePromptInput.value = "帮我把这句开场写得更引人注目，但不要太像广告。";
  taskGoalInput.value = "短视频开场前三秒更抓人";
  platformTagInput.value = "短视频";
  styleHintInput.value = "有文采但别虚";
  mustKeepTermsInput.value = "引人注目, 第一眼";
  strategySelect.value = runtimeConfig?.default_strategy ?? "balanced";
});

Promise.all([refreshHistory(), refreshRuntimeConfig(), refreshReport()]).catch(() => {
  setNotice("初始化数据加载失败，请稍后刷新。");
});
