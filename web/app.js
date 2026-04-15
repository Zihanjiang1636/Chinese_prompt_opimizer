const defaultUserId = "local-user";
let activeSessionId = null;

const sourcePromptInput = document.getElementById("sourcePrompt");
const taskGoalInput = document.getElementById("taskGoal");
const platformTagInput = document.getElementById("platformTag");
const styleHintInput = document.getElementById("styleHint");
const mustKeepTermsInput = document.getElementById("mustKeepTerms");
const saveSessionInput = document.getElementById("saveSession");
const noticeNode = document.getElementById("notice");
const bestPromptNode = document.getElementById("bestPrompt");
const scoreSummaryNode = document.getElementById("scoreSummary");
const briefExplanationNode = document.getElementById("briefExplanation");
const analysisSummaryNode = document.getElementById("analysisSummary");
const historyListNode = document.getElementById("historyList");

function setNotice(text) {
  noticeNode.textContent = text || "";
}

function splitTerms(value) {
  return value
    .split(/[，,\n]/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function renderResult(result) {
  bestPromptNode.textContent = result.best_prompt || "暂无结果";
  const summary = result.score_summary || {};
  scoreSummaryNode.innerHTML = [
    { label: "胜出总分", value: summary.winner_total },
    { label: "原始基线", value: summary.original_baseline_total },
    { label: "直改基线", value: summary.direct_baseline_total },
  ]
    .map(
      (item) => `
        <div class="score-card">
          <span>${item.label}</span>
          <span class="value">${item.value ?? "-"}</span>
        </div>
      `,
    )
    .join("");

  briefExplanationNode.innerHTML = (result.brief_explanation || [])
    .map((item) => `<div class="explanation-item">${item}</div>`)
    .join("");

  const analysis = result.analysis || {};
  const strengths = (analysis.strengths || []).map((item) => `<li>${item}</li>`).join("");
  const risks = (analysis.risks || []).map((item) => `<li>${item}</li>`).join("");
  const culturalSignals = (analysis.cultural_signals || []).map((item) => `<li>${item}</li>`).join("");
  analysisSummaryNode.innerHTML = `
    <p>${result.analysis_summary || ""}</p>
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
          <small>${item.confidence_band} · 胜出分 ${item.score_summary?.winner_total ?? "-"}</small>
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

async function refreshHistory() {
  const response = await fetch(`/api/prompt-copilot/history?user_id=${defaultUserId}`);
  const payload = await response.json();
  renderHistory(payload.data.sessions || []);
}

document.getElementById("optimizeButton").addEventListener("click", async () => {
  setNotice("正在解析、改写并进行内部筛选……");
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

document.getElementById("exampleButton").addEventListener("click", () => {
  sourcePromptInput.value = "帮我把这句开场写得更引人注目，但不要太像广告。";
  taskGoalInput.value = "短视频开场前三秒更抓人";
  platformTagInput.value = "短视频";
  styleHintInput.value = "有文采但别虚";
  mustKeepTermsInput.value = "引人注目, 第一眼";
});

refreshHistory().catch(() => {
  setNotice("历史记录暂时加载失败。");
});
