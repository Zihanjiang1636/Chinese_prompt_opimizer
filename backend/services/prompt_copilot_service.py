"""Prompt Copilot optimization workflow."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from uuid import uuid4

from backend.core.utils import ensure_text, load_json, now_iso
from backend.services.database_service import DatabaseService, database_service
from backend.services.llm_service import LLMService, llm_service

DIMENSION_KEYS = (
    "task_completion",
    "clarity",
    "naturalness",
    "literary_gain",
    "imagery",
    "ambiguity_control",
    "executability",
)

STRATEGY_PROFILES: dict[str, dict[str, Any]] = {
    "balanced": {
        "label": "平衡",
        "description": "默认策略，兼顾任务完成度、自然度和文采。",
        "weight_boosts": {},
        "focus": "先稳住任务边界，再适度增强文采与画面感。",
    },
    "literary": {
        "label": "文采优先",
        "description": "更强调语感、节奏和画面唤起。",
        "weight_boosts": {"literary_gain": 0.18, "imagery": 0.18, "naturalness": 0.08, "clarity": -0.04},
        "focus": "优先保留中文语感、意象和节奏，但不能丢清晰度。",
    },
    "conversion": {
        "label": "转化优先",
        "description": "更强调抓人、清晰和可执行。",
        "weight_boosts": {"task_completion": 0.22, "clarity": 0.16, "executability": 0.18, "ambiguity_control": 0.14, "literary_gain": -0.08},
        "focus": "优先强化抓手、结构和行动性，避免空泛修辞削弱转化。",
    },
}

PROVIDER_PRESETS = [
    {"id": "openai", "label": "OpenAI 官方接口", "provider": "openai", "base_url": "https://api.openai.com/v1", "note": "适合直接调用 OpenAI 模型。"},
    {"id": "openai-compatible", "label": "OpenAI Compatible", "provider": "openai-compatible", "base_url": "https://your-provider.example/v1", "note": "适合兼容 OpenAI API 的云服务或网关。"},
    {"id": "ollama", "label": "Ollama 本地接口", "provider": "openai-compatible", "base_url": "http://127.0.0.1:11434/v1", "note": "适合本地开源模型联调。"},
    {"id": "lm-studio", "label": "LM Studio 本地接口", "provider": "openai-compatible", "base_url": "http://127.0.0.1:1234/v1", "note": "适合桌面本地模型调试。"},
    {"id": "qwen-dashscope", "label": "通义千问 / DashScope", "provider": "dashscope", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "note": "适合中文能力较强的兼容接口。"},
    {"id": "deepseek", "label": "DeepSeek 兼容接口", "provider": "deepseek", "base_url": "https://api.deepseek.com/v1", "note": "适合做效果与成本的平衡。"},
]

PROMPT_HISTORY_FILENAME = "prompt_copilot_history.json"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = PROJECT_ROOT / "prompt-copilot"


@dataclass
class CandidateResult:
    label: str
    prompt: str
    stage: str
    scores: dict[str, float]
    total: float
    notes: list[str]
    generated_output: str = ""
    judge_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "prompt": self.prompt,
            "stage": self.stage,
            "scores": self.scores,
            "total": round(self.total, 3),
            "notes": self.notes,
            "generated_output": self.generated_output,
            "judge_summary": self.judge_summary,
        }


class PromptCopilotService:
    def __init__(self, database: DatabaseService | None = None, llm: LLMService | None = None, asset_root: Path | None = None) -> None:
        self.database = database or database_service
        self.llm = llm or llm_service
        self.asset_root = asset_root or ASSET_ROOT
        self.eval_dataset = self._load_asset(self.asset_root / "datasets" / "creative-eval-set.json", {"cases": []})
        self.rubric = self._load_asset(self.asset_root / "rubric" / "scoring-dimensions.json", {"scenario_weights": {}})
        self.template_bank = self._load_asset(self.asset_root / "prompt-templates" / "default.json", {"version": "fallback"})

    def analyze_and_optimize(self, *, user_id: str, source_prompt: str, task_goal: str, platform_tag: str | None = None, style_hint: str | None = None, must_keep_terms: list[str] | None = None, save_session: bool = True, strategy: str | None = None) -> dict[str, Any]:
        source_prompt = ensure_text(source_prompt)
        task_goal = ensure_text(task_goal)
        if not source_prompt or not task_goal:
            return {"status": "invalid_input", "message": "source_prompt and task_goal are required."}

        keep_terms = self._normalize_terms(must_keep_terms or [])
        scenario = self._classify_task_goal(task_goal)
        strategy_id = self._normalize_strategy(strategy)
        analysis = self._analyze_prompt(source_prompt, task_goal, platform_tag, style_hint, keep_terms, scenario, strategy_id)
        candidates = self._generate_candidates(source_prompt, task_goal, style_hint, keep_terms, strategy_id)
        prescreened = self._prescreen_candidates(candidates, source_prompt, task_goal, keep_terms, scenario, strategy_id)
        replayed = self._run_replay(prescreened, task_goal, analysis, scenario, keep_terms, strategy_id)
        winner, direct_baseline, original_baseline = self._pick_winner(replayed)
        explanation = self._build_explanation(winner, direct_baseline, original_baseline, analysis, strategy_id)
        confidence_band = self._confidence_band(winner.total, max(original_baseline.total, direct_baseline.total))
        session_id = f"pc-{uuid4().hex[:10]}"
        history_item = self._build_history_item(session_id, source_prompt, task_goal, winner, direct_baseline, original_baseline, explanation, analysis, confidence_band, save_session, platform_tag, style_hint, keep_terms, replayed, strategy_id)
        if save_session:
            self._save_history_item(user_id, history_item)

        return {
            "status": "success",
            "message": "Prompt Copilot optimization complete.",
            "session_id": session_id,
            "best_prompt": winner.prompt,
            "brief_explanation": explanation,
            "analysis_summary": analysis["summary"],
            "confidence_band": confidence_band,
            "save_enabled": bool(save_session),
            "selected_label": winner.label,
            "template_version": self.template_bank.get("version", "unknown"),
            "strategy": strategy_id,
            "strategy_label": STRATEGY_PROFILES[strategy_id]["label"],
            "score_summary": history_item["score_summary"],
            "analysis": analysis,
            "history_item": history_item,
        }

    def get_runtime_config(self) -> dict[str, Any]:
        return {
            "template_version": self.template_bank.get("version", "unknown"),
            "llm_provider": self.llm.provider,
            "llm_model": self.llm.model,
            "llm_mode": "stub" if self.llm.stub_mode else "live",
            "dataset_case_count": len(self.eval_dataset.get("cases", [])),
            "default_strategy": "balanced",
            "strategies": [{"id": key, "label": value["label"], "description": value["description"]} for key, value in STRATEGY_PROFILES.items()],
            "provider_presets": PROVIDER_PRESETS,
        }

    def list_history(self, user_id: str) -> list[dict[str, Any]]:
        items = self._load_history(user_id).get("sessions", [])
        ordered = sorted(items, key=lambda item: item.get("updated_at", ""), reverse=True)
        return [{
            "session_id": item.get("session_id", ""),
            "created_at": item.get("created_at", ""),
            "updated_at": item.get("updated_at", ""),
            "source_prompt": item.get("source_prompt", ""),
            "task_goal": item.get("task_goal", ""),
            "best_prompt": item.get("best_prompt", ""),
            "confidence_band": item.get("confidence_band", "low"),
            "selected_label": item.get("selected_label", ""),
            "strategy": item.get("strategy", "balanced"),
            "strategy_label": item.get("strategy_label", STRATEGY_PROFILES["balanced"]["label"]),
            "score_summary": item.get("score_summary", {}),
            "feedback": item.get("feedback", {}),
            "feedback_summary": item.get("feedback_summary", {}),
        } for item in ordered]

    def get_history_item(self, user_id: str, session_id: str) -> dict[str, Any] | None:
        return next((deepcopy(item) for item in self._load_history(user_id).get("sessions", []) if item.get("session_id") == session_id), None)

    def record_feedback(self, *, user_id: str, session_id: str, copied: bool | None, adopted: bool | None, closer_to_goal: bool | None, edited_prompt: str | None, note: str | None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = self._load_history(user_id)
        target = next((item for item in payload.get("sessions", []) if item.get("session_id") == session_id), None)
        if target is None:
            return {"status": "session_not_found", "message": "Prompt copilot session not found."}
        feedback = target.setdefault("feedback", {})
        if copied is not None:
            feedback["copied"] = bool(copied)
        if adopted is not None:
            feedback["adopted"] = bool(adopted)
        if closer_to_goal is not None:
            feedback["closer_to_goal"] = bool(closer_to_goal)
        if edited_prompt is not None:
            feedback["edited_prompt"] = ensure_text(edited_prompt)
        if note is not None:
            feedback["note"] = ensure_text(note)
        if metadata:
            feedback["metadata"] = {**feedback.get("metadata", {}), **metadata}
        feedback["updated_at"] = now_iso()
        target["feedback_summary"] = self._build_feedback_summary(feedback, target.get("best_prompt", ""))
        target["updated_at"] = feedback["updated_at"]
        self.database.write_user_file(user_id, PROMPT_HISTORY_FILENAME, payload)
        return {"status": "success", "message": "Feedback saved.", "session": deepcopy(target)}

    def _load_asset(self, path: Path, default: dict[str, Any]) -> dict[str, Any]:
        return load_json(path, default)

    def _load_history(self, user_id: str) -> dict[str, Any]:
        return self.database.read_user_file(user_id, PROMPT_HISTORY_FILENAME, {"sessions": []})

    def _save_history_item(self, user_id: str, item: dict[str, Any]) -> None:
        payload = self._load_history(user_id)
        payload["sessions"] = [entry for entry in payload.get("sessions", []) if entry.get("session_id") != item["session_id"]] + [item]
        self.database.write_user_file(user_id, PROMPT_HISTORY_FILENAME, payload)

    def _normalize_terms(self, terms: list[str]) -> list[str]:
        normalized: list[str] = []
        for item in terms:
            value = ensure_text(item)
            if value and value not in normalized:
                normalized.append(value)
        return normalized

    def _normalize_strategy(self, strategy: str | None) -> str:
        value = ensure_text(strategy).lower().replace("_", "-")
        return value if value in STRATEGY_PROFILES else "balanced"

    def _compose_weights(self, scenario: str, strategy_id: str) -> dict[str, float]:
        base_weights = self.rubric.get("scenario_weights", {}).get(scenario) or {}
        boosts = STRATEGY_PROFILES[strategy_id].get("weight_boosts", {})
        return {key: round(max(0.2, float(base_weights.get(key, 1.0)) * (1 + float(boosts.get(key, 0.0)))), 3) for key in DIMENSION_KEYS}
    def _classify_task_goal(self, task_goal: str) -> str:
        mapping = {
            "hook": ("开场", "前三秒", "短视频", "抓人", "吸睛"),
            "seeding": ("种草", "安利", "推荐", "测评"),
            "ad_copy": ("广告", "口播", "投放", "品牌"),
            "headline": ("标题", "封面", "点击"),
            "value_prop": ("卖点", "亮点", "优势", "利益点"),
            "emotion": ("情绪", "共鸣", "氛围", "画面感", "余味"),
        }
        for label, keywords in mapping.items():
            if any(keyword in task_goal for keyword in keywords):
                return label
        return "general_copy"

    def _analyze_prompt(self, source_prompt: str, task_goal: str, platform_tag: str | None, style_hint: str | None, must_keep_terms: list[str], scenario: str, strategy_id: str) -> dict[str, Any]:
        fallback = self._heuristic_analysis(source_prompt, task_goal, platform_tag, style_hint, must_keep_terms, scenario, strategy_id)
        system_prompt = self._template("analyze_system_prompt", "你是中文提示词分析助手。请只输出 JSON 对象。")
        user_prompt = json.dumps({
            "source_prompt": source_prompt,
            "task_goal": task_goal,
            "platform_tag": platform_tag or "",
            "style_hint": style_hint or "",
            "must_keep_terms": must_keep_terms,
            "scenario": scenario,
            "strategy": strategy_id,
            "strategy_focus": STRATEGY_PROFILES[strategy_id]["focus"],
        }, ensure_ascii=False)
        response = self.llm.generate_text(system_prompt, user_prompt, json.dumps(fallback, ensure_ascii=False))
        parsed = self._extract_json_object(response.get("text", ""))
        if not isinstance(parsed, dict):
            return fallback
        merged = deepcopy(fallback)
        for key in ("core_intent", "tone", "strengths", "risks", "cultural_signals", "summary"):
            if parsed.get(key):
                merged[key] = parsed[key]
        return merged

    def _heuristic_analysis(self, source_prompt: str, task_goal: str, platform_tag: str | None, style_hint: str | None, must_keep_terms: list[str], scenario: str, strategy_id: str) -> dict[str, Any]:
        strengths: list[str] = []
        risks: list[str] = []
        cultural_signals: list[str] = []
        if len(source_prompt) <= 18:
            risks.append("原提示词比较短，任务边界和输出期待还不够完整。")
        else:
            strengths.append("原提示词已经给出了明确方向，适合在此基础上做结构化增强。")
        if any(word in source_prompt for word in ("画面", "节奏", "共鸣", "情绪", "故事", "留白")):
            strengths.append("原句自带中文表达质感，适合保留语感而不是简单改写成说明句。")
        if any(word in source_prompt for word in ("高级", "炸", "惊艳", "有感觉", "震撼")):
            risks.append("存在偏抽象的质量词，模型可能会把它理解成空泛润色。")
        if any(word in source_prompt for word in ("成语", "典故", "意境", "余味", "留白")):
            cultural_signals.append("提示词里带有明显的中文文化表达线索，优化时要保留语感，但不能放大歧义。")
        if must_keep_terms:
            strengths.append("已经给出保留关键词，便于做定向增强而不是重写用户意图。")
        tone = ensure_text(style_hint) or ("偏传播感" if scenario in {"hook", "headline", "ad_copy"} else "偏真实表达")
        summary = f"当前提示词的核心目标是“{task_goal}”。本轮会按“{STRATEGY_PROFILES[strategy_id]['label']}”策略优化，先补足任务边界与执行约束，再决定是否增强文采、画面感和中文语感。"
        return {
            "core_intent": task_goal,
            "tone": tone,
            "strengths": strengths or ["原提示词有明确方向，适合做结构化增强。"],
            "risks": risks or ["如果只追求更华丽的表达，结果可能变得好看但不稳定。"],
            "cultural_signals": cultural_signals,
            "summary": summary,
            "scenario": scenario,
            "platform_tag": ensure_text(platform_tag),
            "style_hint": ensure_text(style_hint),
            "must_keep_terms": must_keep_terms,
            "strategy": strategy_id,
            "strategy_label": STRATEGY_PROFILES[strategy_id]["label"],
        }

    def _generate_candidates(self, source_prompt: str, task_goal: str, style_hint: str | None, keep_terms: list[str], strategy_id: str) -> dict[str, str]:
        keep_line = f"必须保留这些关键词：{', '.join(keep_terms)}。" if keep_terms else ""
        style_line = f"风格偏好：{style_hint}。" if ensure_text(style_hint) else ""
        strategy_line = f"优化策略：{STRATEGY_PROFILES[strategy_id]['label']}。{STRATEGY_PROFILES[strategy_id]['focus']}"
        fallback = {
            "original": source_prompt,
            "direct_rewrite": f"请围绕“{task_goal}”完成内容创作，在保留原始意图“{source_prompt}”的前提下，给出更完整、更稳妥的表达。{keep_line}{style_line}{strategy_line}",
            "clearer": f"你是一名中文内容创作者，请完成任务目标：{task_goal}。基础要求：{source_prompt}。请先明确核心受众、传播重点和输出边界，再给出适合直接使用的提示词。语言要清楚、重点前置、避免空泛形容词。{keep_line}{style_line}{strategy_line}",
            "literary": f"请围绕“{task_goal}”创作中文提示词，保留原始表达意图：{source_prompt}。语言可以更有节奏、语感和记忆点，但不能牺牲清晰度和可执行性。{keep_line}{style_line}{strategy_line}",
            "vivid": f"请围绕“{task_goal}”生成一版更有画面感的中文提示词，以“{source_prompt}”为基础，优先增强场景感、动作感和情绪触发点，让输出更容易在第一眼抓住读者，但不要喊口号。{keep_line}{style_line}{strategy_line}",
            "executable": f"请完成任务：{task_goal}。原始要求：{source_prompt}。输出时请遵循以下规则：1. 先给一句核心钩子。2. 再补关键卖点或情绪推动。3. 结尾形成明确记忆点。整体中文表达自然、可直接发布。{keep_line}{style_line}{strategy_line}",
            "conservative": f"请基于“{source_prompt}”做一版保守增强，目标任务是：{task_goal}。只补足结构、语义边界和稳定性，不为了追求华丽而改变原意。{keep_line}{style_line}{strategy_line}",
        }
        system_prompt = self._template("generate_system_prompt", "你是中文提示词优化器。请只输出 JSON 对象。")
        user_prompt = json.dumps({"source_prompt": source_prompt, "task_goal": task_goal, "style_hint": style_hint or "", "keep_terms": keep_terms, "strategy": strategy_id, "strategy_focus": STRATEGY_PROFILES[strategy_id]["focus"]}, ensure_ascii=False)
        response = self.llm.generate_text(system_prompt, user_prompt, json.dumps(fallback, ensure_ascii=False))
        parsed = self._extract_json_object(response.get("text", ""))
        if not isinstance(parsed, dict):
            return fallback
        merged = deepcopy(fallback)
        for key in fallback:
            value = ensure_text(str(parsed.get(key, "")))
            if value:
                merged[key] = value
        return merged

    def _prescreen_candidates(self, candidates: dict[str, str], source_prompt: str, task_goal: str, keep_terms: list[str], scenario: str, strategy_id: str) -> list[CandidateResult]:
        weights = self._compose_weights(scenario, strategy_id)
        case = self._pick_eval_case(scenario)
        results: list[CandidateResult] = []
        for label, prompt in candidates.items():
            scores = self._score_prompt_text(prompt, source_prompt, task_goal, keep_terms, case, strategy_id)
            results.append(CandidateResult(label=label, prompt=prompt, stage="prescreen", scores=scores, total=self._weighted_total(scores, weights), notes=self._prompt_notes(label, prompt, keep_terms, strategy_id)))
        baselines = [item for item in results if item.label in {"original", "direct_rewrite"}]
        challengers = sorted([item for item in results if item.label not in {"original", "direct_rewrite"}], key=lambda item: item.total, reverse=True)[:3]
        return baselines + challengers

    def _pick_eval_case(self, scenario: str) -> dict[str, Any]:
        cases = self.eval_dataset.get("cases", [])
        for item in cases:
            if item.get("scenario") == scenario:
                return item
        return cases[0] if cases else {}

    def _score_prompt_text(self, prompt: str, source_prompt: str, task_goal: str, keep_terms: list[str], case: dict[str, Any], strategy_id: str) -> dict[str, float]:
        clarity = 5.0 + min(self._keyword_hits(prompt, ("要求", "输出", "任务", "目标", "规则", "步骤", "先", "再", "最后")) * 0.45, 3.0)
        clarity += 0.5 if task_goal in prompt else 0.0
        executability = 4.6 + min(self._keyword_hits(prompt, ("1.", "2.", "3.", "规则", "步骤", "直接使用", "可直接发布")) * 0.55, 3.2)
        literary_gain = 4.0 + min(self._keyword_hits(prompt, ("节奏", "语感", "意象", "留白", "余味", "记忆点", "耐读")) * 0.55, 3.2)
        imagery = 4.0 + min(self._keyword_hits(prompt, ("画面", "场景", "动作", "细节", "第一眼", "情绪")) * 0.6, 3.3)
        naturalness = 5.8 + (0.7 if 30 <= len(prompt) <= 220 else 0.0)
        ambiguity_control = 7.4 - min(self._keyword_hits(prompt, ("高级", "炸", "惊艳", "震撼", "绝绝子", "有感觉")) * 0.65, 3.0)
        ambiguity_control += 0.4 if "不要" in prompt else 0.0
        keep_hits = sum(term in prompt for term in keep_terms)
        task_completion = 5.4 + min(self._keyword_hits(prompt, ("受众", "重点", "卖点", "钩子", "情绪", "记忆点", "结构")) * 0.35, 2.2)
        task_completion += 1.0 if keep_terms and keep_hits == len(keep_terms) else 0.0
        task_completion += 0.4 if source_prompt in prompt else 0.0
        if strategy_id == "literary":
            literary_gain += 0.25
            imagery += 0.2
        elif strategy_id == "conversion":
            clarity += 0.2
            executability += 0.3
            ambiguity_control += 0.2
        preferred = case.get("preferred_dimensions", [])
        if "clarity" in preferred:
            clarity += 0.2
        if "executability" in preferred:
            executability += 0.2
        if "literary_gain" in preferred:
            literary_gain += 0.2
        if "imagery" in preferred:
            imagery += 0.2
        raw_scores = {"task_completion": task_completion, "clarity": clarity, "naturalness": naturalness, "literary_gain": literary_gain, "imagery": imagery, "ambiguity_control": ambiguity_control, "executability": executability}
        return {key: max(1.0, min(round(value, 3), 10.0)) for key, value in raw_scores.items()}
    def _weighted_total(self, scores: dict[str, float], weights: dict[str, float]) -> float:
        total = 0.0
        weight_sum = 0.0
        for key in DIMENSION_KEYS:
            weight = float(weights.get(key, 1.0))
            total += scores.get(key, 0.0) * weight
            weight_sum += weight
        return total / weight_sum if weight_sum else 0.0

    def _prompt_notes(self, label: str, prompt: str, keep_terms: list[str], strategy_id: str) -> list[str]:
        notes = {
            "original": ["保留用户原始表达，作为零优化基线。"],
            "direct_rewrite": ["单轮直接改写基线，用来防止“改了但不一定更好”的情况。"],
            "clearer": ["优先补足任务边界和输出结构。"],
            "literary": ["增强中文语感和节奏，但要防止过度抽象。"],
            "vivid": ["加强画面和情绪触发，更适合抓眼场景。"],
            "executable": ["强调步骤感和可执行性。"],
            "conservative": ["偏保守增强，用于没有明显赢家时兜底。"],
        }.get(label, [])
        notes.append(f"当前策略：{STRATEGY_PROFILES[strategy_id]['label']}。")
        if keep_terms:
            notes.append("已纳入保留关键词约束。")
        if len(prompt) > 200:
            notes.append("结构较完整，但要留意输入成本。")
        return notes

    def _run_replay(self, candidates: list[CandidateResult], task_goal: str, analysis: dict[str, Any], scenario: str, keep_terms: list[str], strategy_id: str) -> list[CandidateResult]:
        weights = self._compose_weights(scenario, strategy_id)
        replayed: list[CandidateResult] = []
        for candidate in candidates:
            generated_output = self._simulate_output(candidate.prompt, task_goal)
            scores, summary = self._judge_output(candidate.prompt, generated_output, task_goal, analysis, keep_terms, strategy_id)
            prompt_signal = candidate.total * 0.35
            replay_signal = self._weighted_total(scores, weights) * 0.65
            replayed.append(CandidateResult(label=candidate.label, prompt=candidate.prompt, stage="replay", scores=scores, total=prompt_signal + replay_signal, notes=candidate.notes, generated_output=generated_output, judge_summary=summary))
        return replayed

    def _simulate_output(self, candidate_prompt: str, task_goal: str) -> str:
        fallback = self._heuristic_generated_output(task_goal)
        system_prompt = self._template("simulate_system_prompt", "你是一名中文内容创作者。请根据给定提示词直接产出结果，只输出成稿。")
        user_prompt = json.dumps({"task_goal": task_goal, "prompt": candidate_prompt}, ensure_ascii=False)
        response = self.llm.generate_text(system_prompt, user_prompt, fallback)
        return ensure_text(response.get("text", "")) or fallback

    def _heuristic_generated_output(self, task_goal: str) -> str:
        if "标题" in task_goal:
            return "一句话先把最能让人停下来的价值点抛出来，再用更具体的利益感把读者拉住。"
        if "口播" in task_goal or "短视频" in task_goal:
            return "别让表达停在说明层面，先把最抓人的信息前置，再用一个生活化细节把情绪托住。"
        return f"围绕“{task_goal}”，先把核心重点前置，再用一两个具体细节把画面感和记忆点托起来。"

    def _judge_output(self, candidate_prompt: str, generated_output: str, task_goal: str, analysis: dict[str, Any], keep_terms: list[str], strategy_id: str) -> tuple[dict[str, float], str]:
        fallback_scores = self._heuristic_output_scores(candidate_prompt, generated_output, keep_terms, strategy_id)
        fallback_summary = "基于固定维度完成输出评估，优先检查任务完成度、自然度、歧义控制和策略匹配。"
        system_prompt = self._template("judge_system_prompt", "你是中文创作评测器。请只输出 JSON 对象。")
        user_prompt = json.dumps({"task_goal": task_goal, "analysis": analysis, "candidate_prompt": candidate_prompt, "generated_output": generated_output, "keep_terms": keep_terms, "strategy": strategy_id}, ensure_ascii=False)
        response = self.llm.generate_text(system_prompt, user_prompt, json.dumps({**fallback_scores, "summary": fallback_summary}, ensure_ascii=False))
        parsed = self._extract_json_object(response.get("text", ""))
        if not isinstance(parsed, dict):
            return fallback_scores, fallback_summary
        scores: dict[str, float] = {}
        for key in DIMENSION_KEYS:
            try:
                value = float(parsed.get(key, fallback_scores[key]))
            except (TypeError, ValueError):
                value = fallback_scores[key]
            scores[key] = max(1.0, min(round(value, 3), 10.0))
        summary = ensure_text(str(parsed.get("summary", fallback_summary))) or fallback_summary
        return scores, summary

    def _heuristic_output_scores(self, candidate_prompt: str, generated_output: str, keep_terms: list[str], strategy_id: str) -> dict[str, float]:
        output = ensure_text(generated_output)
        task_completion = 5.2 + min(self._keyword_hits(output, ("重点", "细节", "情绪", "价值", "画面", "记忆点")) * 0.45, 2.6)
        clarity = 5.6 + (0.6 if 18 <= len(output) <= 160 else 0.0)
        naturalness = 6.1 + (0.5 if "，" in output or "。" in output else 0.0)
        literary_gain = 4.2 + min(self._keyword_hits(output, ("画面", "余味", "节奏", "第一眼", "托住")) * 0.55, 2.6)
        imagery = 4.0 + min(self._keyword_hits(output, ("看见", "细节", "场景", "动作", "第一眼")) * 0.6, 2.8)
        executability = 5.0 + min(self._keyword_hits(candidate_prompt, ("输出", "规则", "步骤", "直接发布", "钩子")) * 0.4, 2.0)
        ambiguity_control = 7.0 - min(self._keyword_hits(output, ("高级", "震撼", "绝绝子", "有感觉")) * 0.6, 2.0)
        if keep_terms:
            ambiguity_control += 0.4 if all(term in candidate_prompt for term in keep_terms) else -0.4
        if strategy_id == "literary":
            literary_gain += 0.25
            imagery += 0.2
        elif strategy_id == "conversion":
            task_completion += 0.2
            clarity += 0.2
            executability += 0.25
        raw_scores = {"task_completion": task_completion, "clarity": clarity, "naturalness": naturalness, "literary_gain": literary_gain, "imagery": imagery, "ambiguity_control": ambiguity_control, "executability": executability}
        return {key: max(1.0, min(round(value, 3), 10.0)) for key, value in raw_scores.items()}

    def _pick_winner(self, replayed: list[CandidateResult]) -> tuple[CandidateResult, CandidateResult, CandidateResult]:
        indexed = {item.label: item for item in replayed}
        original_baseline = indexed["original"]
        direct_baseline = indexed["direct_rewrite"]
        baseline_floor = max(original_baseline.total, direct_baseline.total)
        challengers = [item for item in replayed if item.label not in {"original", "direct_rewrite"}]
        eligible = [item for item in challengers if item.total >= baseline_floor]
        if eligible:
            winner = max(eligible, key=lambda item: item.total)
        else:
            conservative = indexed.get("conservative")
            winner = conservative if conservative and conservative.total >= direct_baseline.total else direct_baseline
        return winner, direct_baseline, original_baseline

    def _build_explanation(self, winner: CandidateResult, direct_baseline: CandidateResult, original_baseline: CandidateResult, analysis: dict[str, Any], strategy_id: str) -> list[str]:
        reasons: list[str] = []
        if winner.scores["clarity"] >= max(direct_baseline.scores["clarity"], original_baseline.scores["clarity"]):
            reasons.append("这版把任务边界和输出重点说得更清楚，模型更容易稳定抓住你的真实意图。")
        if winner.scores["literary_gain"] >= direct_baseline.scores["literary_gain"]:
            reasons.append("它保留了中文表达的语感和节奏，但没有把修辞放到信息前面。")
        if winner.scores["ambiguity_control"] >= direct_baseline.scores["ambiguity_control"]:
            reasons.append("相比更冒进的写法，这版对歧义的控制更稳，不容易把“有文采”写成“空泛”。")
        if strategy_id == "conversion" and winner.scores["executability"] >= direct_baseline.scores["executability"]:
            reasons.append("这次还额外强化了可执行性，更适合标题、口播和卖点这类要快速落地的场景。")
        if not reasons:
            reasons.append("当前胜出版本主要赢在更稳：它不一定最花哨，但更容易得到可直接使用的结果。")
        if analysis.get("cultural_signals"):
            reasons.append("系统也保留了你原句里的中文语感线索，没有把它压扁成干巴巴的说明句。")
        return reasons[:4]

    def _confidence_band(self, winner_total: float, baseline_total: float) -> str:
        margin = winner_total - baseline_total
        if margin >= 0.8:
            return "high"
        if margin >= 0.35:
            return "medium"
        return "low"

    def _build_history_item(self, session_id: str, source_prompt: str, task_goal: str, best_candidate: CandidateResult, direct_baseline: CandidateResult, original_baseline: CandidateResult, explanation: list[str], analysis: dict[str, Any], confidence_band: str, save_enabled: bool, platform_tag: str | None, style_hint: str | None, keep_terms: list[str], replayed: list[CandidateResult], strategy_id: str) -> dict[str, Any]:
        timestamp = now_iso()
        return {
            "session_id": session_id,
            "created_at": timestamp,
            "updated_at": timestamp,
            "source_prompt": source_prompt,
            "task_goal": task_goal,
            "best_prompt": best_candidate.prompt,
            "brief_explanation": explanation,
            "analysis_summary": analysis.get("summary", ""),
            "analysis": analysis,
            "confidence_band": confidence_band,
            "save_enabled": save_enabled,
            "platform_tag": ensure_text(platform_tag),
            "style_hint": ensure_text(style_hint),
            "must_keep_terms": keep_terms,
            "selected_label": best_candidate.label,
            "template_version": self.template_bank.get("version", "unknown"),
            "strategy": strategy_id,
            "strategy_label": STRATEGY_PROFILES[strategy_id]["label"],
            "score_summary": {"winner_total": round(best_candidate.total, 3), "original_baseline_total": round(original_baseline.total, 3), "direct_baseline_total": round(direct_baseline.total, 3)},
            "feedback": {"copied": False, "adopted": False, "closer_to_goal": None, "edited_prompt": "", "note": "", "metadata": {}, "updated_at": ""},
            "feedback_summary": self._build_feedback_summary({"copied": False, "adopted": False, "closer_to_goal": None, "edited_prompt": "", "note": "", "metadata": {}, "updated_at": ""}, best_candidate.prompt),
            "internal_trace": {
                "winner": best_candidate.to_dict(),
                "baselines": {"original": original_baseline.to_dict(), "direct_rewrite": direct_baseline.to_dict()},
                "replayed": [item.to_dict() for item in replayed],
            },
        }

    def _build_feedback_summary(self, feedback: dict[str, Any], best_prompt: str) -> dict[str, Any]:
        edited_prompt = ensure_text(feedback.get("edited_prompt"))
        similarity = self._text_similarity(best_prompt, edited_prompt) if edited_prompt else 0.0
        copied = bool(feedback.get("copied"))
        adopted = bool(feedback.get("adopted"))
        closer = feedback.get("closer_to_goal")

        score = 0.0
        if copied:
            score += 0.2
        if adopted:
            score += 0.35
        if closer is True:
            score += 0.25
        elif closer is False:
            score -= 0.15
        if edited_prompt:
            score += 0.1 if similarity >= 0.6 else 0.05
        if adopted and edited_prompt and similarity >= 0.9:
            adoption_state = "direct_adopt"
        elif adopted and edited_prompt:
            adoption_state = "edited_adopt"
        elif adopted:
            adoption_state = "adopted"
        elif copied:
            adoption_state = "copied_only"
        elif closer is False:
            adoption_state = "not_close"
        else:
            adoption_state = "unrated"

        return {
            "signal_score": round(max(0.0, min(score, 1.0)), 3),
            "adoption_state": adoption_state,
            "edit_similarity": round(similarity, 3),
            "has_edited_prompt": bool(edited_prompt),
        }

    def _keyword_hits(self, text: str, keywords: tuple[str, ...]) -> int:
        return sum(keyword in text for keyword in keywords)

    def _text_similarity(self, left: str, right: str) -> float:
        if not ensure_text(left) or not ensure_text(right):
            return 0.0
        return SequenceMatcher(a=left, b=right).ratio()

    def _extract_json_object(self, text: str) -> dict[str, Any] | None:
        cleaned = ensure_text(text)
        if not cleaned:
            return None
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _template(self, key: str, fallback: str) -> str:
        value = ensure_text(str(self.template_bank.get(key, "")))
        return value or fallback


prompt_copilot_service = PromptCopilotService()
