"""Prompt copilot optimization workflow."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass
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
    """Optimizes Chinese creative prompts with internal baselines and replay scoring."""

    def __init__(
        self,
        database: DatabaseService | None = None,
        llm: LLMService | None = None,
        asset_root: Path | None = None,
    ) -> None:
        self.database = database or database_service
        self.llm = llm or llm_service
        self.asset_root = asset_root or ASSET_ROOT
        self.eval_dataset = self._load_asset(
            self.asset_root / "datasets" / "creative-eval-set.json",
            {"cases": []},
        )
        self.rubric = self._load_asset(
            self.asset_root / "rubric" / "scoring-dimensions.json",
            {"dimensions": [], "scenario_weights": {}},
        )
        self.template_bank = self._load_asset(
            self.asset_root / "prompt-templates" / "default.json",
            {
                "version": "fallback",
                "analyze_system_prompt": "",
                "generate_system_prompt": "",
                "simulate_system_prompt": "",
                "judge_system_prompt": "",
            },
        )

    def analyze_and_optimize(
        self,
        *,
        user_id: str,
        source_prompt: str,
        task_goal: str,
        platform_tag: str | None = None,
        style_hint: str | None = None,
        must_keep_terms: list[str] | None = None,
        save_session: bool = True,
    ) -> dict[str, Any]:
        normalized_source = ensure_text(source_prompt)
        normalized_goal = ensure_text(task_goal)
        if not normalized_source or not normalized_goal:
            return {
                "status": "invalid_input",
                "message": "source_prompt and task_goal are required.",
            }

        keep_terms = self._normalize_terms(must_keep_terms or [])
        scenario = self._classify_task_goal(normalized_goal)
        analysis = self._analyze_prompt(
            source_prompt=normalized_source,
            task_goal=normalized_goal,
            platform_tag=platform_tag,
            style_hint=style_hint,
            must_keep_terms=keep_terms,
            scenario=scenario,
        )
        candidates = self._generate_candidates(
            source_prompt=normalized_source,
            task_goal=normalized_goal,
            style_hint=style_hint,
            keep_terms=keep_terms,
            analysis=analysis,
        )

        prescreened = self._prescreen_candidates(
            candidates=candidates,
            source_prompt=normalized_source,
            task_goal=normalized_goal,
            keep_terms=keep_terms,
            scenario=scenario,
        )
        replayed = self._run_replay(
            candidates=prescreened,
            task_goal=normalized_goal,
            analysis=analysis,
            scenario=scenario,
            keep_terms=keep_terms,
        )
        winner, direct_baseline, original_baseline = self._pick_winner(replayed)
        confidence_band = self._confidence_band(
            winner.total,
            max(original_baseline.total, direct_baseline.total),
        )
        explanation = self._build_brief_explanation(
            winner=winner,
            direct_baseline=direct_baseline,
            original_baseline=original_baseline,
            analysis=analysis,
        )
        session_id = f"pc-{uuid4().hex[:10]}"
        history_item = self._build_history_item(
            session_id=session_id,
            source_prompt=normalized_source,
            task_goal=normalized_goal,
            best_candidate=winner,
            direct_baseline=direct_baseline,
            original_baseline=original_baseline,
            explanation=explanation,
            analysis=analysis,
            confidence_band=confidence_band,
            save_enabled=save_session,
            platform_tag=platform_tag,
            style_hint=style_hint,
            keep_terms=keep_terms,
            replayed=replayed,
        )
        if save_session:
            self._save_history_item(user_id, history_item)

        return {
            "status": "success",
            "message": "Prompt copilot optimization complete.",
            "session_id": session_id,
            "best_prompt": winner.prompt,
            "brief_explanation": explanation,
            "analysis_summary": analysis["summary"],
            "confidence_band": confidence_band,
            "save_enabled": bool(save_session),
            "selected_label": winner.label,
            "template_version": self.template_bank.get("version", "unknown"),
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
        }

    def list_history(self, user_id: str) -> list[dict[str, Any]]:
        payload = self._load_history(user_id)
        items = payload.get("sessions", [])
        ordered = sorted(items, key=lambda item: item.get("updated_at", ""), reverse=True)
        return [
            {
                "session_id": item.get("session_id", ""),
                "created_at": item.get("created_at", ""),
                "updated_at": item.get("updated_at", ""),
                "source_prompt": item.get("source_prompt", ""),
                "task_goal": item.get("task_goal", ""),
                "best_prompt": item.get("best_prompt", ""),
                "confidence_band": item.get("confidence_band", "low"),
                "selected_label": item.get("selected_label", ""),
                "score_summary": item.get("score_summary", {}),
                "feedback": item.get("feedback", {}),
            }
            for item in ordered
        ]

    def get_history_item(self, user_id: str, session_id: str) -> dict[str, Any] | None:
        payload = self._load_history(user_id)
        return next(
            (deepcopy(item) for item in payload.get("sessions", []) if item.get("session_id") == session_id),
            None,
        )

    def record_feedback(
        self,
        *,
        user_id: str,
        session_id: str,
        copied: bool | None,
        adopted: bool | None,
        closer_to_goal: bool | None,
        note: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = self._load_history(user_id)
        items = payload.get("sessions", [])
        target = next((item for item in items if item.get("session_id") == session_id), None)
        if target is None:
            return {"status": "session_not_found", "message": "Prompt copilot session not found."}

        feedback = target.setdefault("feedback", {})
        if copied is not None:
            feedback["copied"] = bool(copied)
        if adopted is not None:
            feedback["adopted"] = bool(adopted)
        if closer_to_goal is not None:
            feedback["closer_to_goal"] = bool(closer_to_goal)
        if note is not None:
            feedback["note"] = ensure_text(note)
        if metadata:
            feedback["metadata"] = {**feedback.get("metadata", {}), **metadata}
        feedback["updated_at"] = now_iso()
        target["updated_at"] = feedback["updated_at"]
        self.database.write_user_file(user_id, PROMPT_HISTORY_FILENAME, payload)
        return {"status": "success", "message": "Feedback saved.", "session": deepcopy(target)}

    def _load_history(self, user_id: str) -> dict[str, Any]:
        return self.database.read_user_file(user_id, PROMPT_HISTORY_FILENAME, {"sessions": []})

    def _save_history_item(self, user_id: str, item: dict[str, Any]) -> None:
        payload = self._load_history(user_id)
        items = [entry for entry in payload.get("sessions", []) if entry.get("session_id") != item["session_id"]]
        items.append(item)
        payload["sessions"] = items
        self.database.write_user_file(user_id, PROMPT_HISTORY_FILENAME, payload)

    def _load_asset(self, path: Path, default: dict[str, Any]) -> dict[str, Any]:
        return load_json(path, default)

    def _normalize_terms(self, terms: list[str]) -> list[str]:
        normalized: list[str] = []
        for item in terms:
            value = ensure_text(item)
            if value and value not in normalized:
                normalized.append(value)
        return normalized

    def _classify_task_goal(self, task_goal: str) -> str:
        lowered = task_goal.lower()
        mapping = {
            "hook": ("开场", "前3秒", "吸睛", "抓人", "短视频"),
            "seeding": ("种草", "安利", "推荐", "测评"),
            "ad_copy": ("广告", "口播", "投放", "品牌"),
            "headline": ("标题", "封面", "标题党"),
            "value_prop": ("卖点", "亮点", "优势", "利益点"),
            "emotion": ("情绪", "共鸣", "治愈", "热血"),
        }
        for label, keywords in mapping.items():
            if any(keyword in lowered for keyword in keywords):
                return label
        return "general_copy"

    def _analyze_prompt(
        self,
        *,
        source_prompt: str,
        task_goal: str,
        platform_tag: str | None,
        style_hint: str | None,
        must_keep_terms: list[str],
        scenario: str,
    ) -> dict[str, Any]:
        fallback = self._heuristic_analysis(
            source_prompt=source_prompt,
            task_goal=task_goal,
            platform_tag=platform_tag,
            style_hint=style_hint,
            must_keep_terms=must_keep_terms,
            scenario=scenario,
        )
        system_prompt = self._template(
            "analyze_system_prompt",
            (
                "你是中文提示词分析助手。"
                "请只输出 JSON 对象，不要加解释。"
                "字段包括 core_intent、tone、strengths、risks、cultural_signals、summary。"
            ),
        )
        user_prompt = json.dumps(
            {
                "source_prompt": source_prompt,
                "task_goal": task_goal,
                "platform_tag": platform_tag or "",
                "style_hint": style_hint or "",
                "must_keep_terms": must_keep_terms,
                "scenario": scenario,
            },
            ensure_ascii=False,
        )
        response = self.llm.generate_text(system_prompt, user_prompt, json.dumps(fallback, ensure_ascii=False))
        parsed = self._extract_json_object(response.get("text", ""))
        if not isinstance(parsed, dict):
            return fallback

        merged = deepcopy(fallback)
        for key in ("core_intent", "tone", "strengths", "risks", "cultural_signals", "summary"):
            value = parsed.get(key)
            if value:
                merged[key] = value
        return merged

    def _heuristic_analysis(
        self,
        *,
        source_prompt: str,
        task_goal: str,
        platform_tag: str | None,
        style_hint: str | None,
        must_keep_terms: list[str],
        scenario: str,
    ) -> dict[str, Any]:
        strength_notes: list[str] = []
        risk_notes: list[str] = []
        cultural_signals: list[str] = []
        if len(source_prompt) <= 18:
            risk_notes.append("原提示词较短，任务边界和输出要求可能不够完整。")
        else:
            strength_notes.append("原提示词已经提供了基本语感，适合在此基础上增强结构。")
        if any(word in source_prompt for word in ("画面", "意象", "情绪", "节奏", "共鸣", "故事")):
            strength_notes.append("原提示词自带一定文采导向，适合保留表达感染力。")
        if any(word in source_prompt for word in ("高级", "好一点", "有感觉", "震撼")):
            risk_notes.append("存在偏抽象的质量词，模型可能理解成笼统润色而不是具体执行。")
        if any(word in source_prompt for word in ("成语", "典故", "意境", "留白")):
            cultural_signals.append("提示词带有明显中文文化表达倾向，可强化语感但要控制歧义。")
        if must_keep_terms:
            strength_notes.append("已提供保留关键词，适合做定向增强而不是重写意图。")
        tone = style_hint or ("偏传播感" if scenario in {"hook", "headline", "ad_copy"} else "偏表达质感")
        summary = (
            f"当前提示词的核心目标是“{task_goal}”。"
            "优化重点会放在补齐任务边界、稳住中文自然度，同时保留必要的文采和画面感。"
        )
        return {
            "core_intent": task_goal,
            "tone": tone,
            "strengths": strength_notes or ["原提示词有明确方向，适合做结构化增强。"],
            "risks": risk_notes or ["如果直接放大文采而不补约束，容易让结果变得好看但不稳定。"],
            "cultural_signals": cultural_signals,
            "summary": summary,
            "scenario": scenario,
            "platform_tag": ensure_text(platform_tag),
            "style_hint": ensure_text(style_hint),
            "must_keep_terms": must_keep_terms,
        }

    def _generate_candidates(
        self,
        *,
        source_prompt: str,
        task_goal: str,
        style_hint: str | None,
        keep_terms: list[str],
        analysis: dict[str, Any],
    ) -> dict[str, str]:
        fallback = self._heuristic_candidates(source_prompt, task_goal, style_hint, keep_terms)
        system_prompt = self._template(
            "generate_system_prompt",
            (
                "你是中文提示词优化器。"
                "请只输出 JSON 对象，包含 direct_rewrite、clearer、literary、vivid、executable、conservative 六个字段。"
                "每个字段是一条可直接给大模型使用的中文提示词。"
            ),
        )
        user_prompt = json.dumps(
            {
                "source_prompt": source_prompt,
                "task_goal": task_goal,
                "style_hint": style_hint or "",
                "keep_terms": keep_terms,
                "analysis": analysis,
            },
            ensure_ascii=False,
        )
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

    def _heuristic_candidates(
        self,
        source_prompt: str,
        task_goal: str,
        style_hint: str | None,
        keep_terms: list[str],
    ) -> dict[str, str]:
        keep_line = f"必须保留这些关键词：{', '.join(keep_terms)}。" if keep_terms else ""
        style_line = f"风格偏好：{style_hint}。" if ensure_text(style_hint) else ""
        return {
            "original": source_prompt,
            "direct_rewrite": (
                f"请围绕“{task_goal}”完成内容创作。"
                f"在保留原始意图“{source_prompt}”的前提下，给出更完整、更稳定的表达。"
                f"{keep_line}{style_line}"
            ),
            "clearer": (
                f"你是一名中文内容创作者。请完成任务目标：{task_goal}。"
                f"基础要求：{source_prompt}。"
                "请先明确核心受众与传播重点，再输出成稿。"
                "要求语言清晰、重点前置、避免空泛形容词。"
                f"{keep_line}{style_line}"
            ),
            "literary": (
                f"请围绕“{task_goal}”创作中文内容。"
                f"保留原始表达意图：{source_prompt}。"
                "语言要有节奏感和中文韵味，允许适度使用意象、留白和有记忆点的句子，"
                "但不能牺牲信息清晰度。"
                f"{keep_line}{style_line}"
            ),
            "vivid": (
                f"请围绕“{task_goal}”生成一版更有画面感的中文表达。"
                f"以“{source_prompt}”为基础，"
                "优先增强场景感、动作感和情绪触发点，让读者更容易被第一眼抓住。"
                "避免空喊口号，细节必须能被想象出来。"
                f"{keep_line}{style_line}"
            ),
            "executable": (
                f"请完成任务：{task_goal}。"
                f"原始要求：{source_prompt}。"
                "输出时请遵循以下规则：1. 先给一句核心钩子。2. 再展开关键卖点或情绪推动。"
                "3. 结尾要形成明确记忆点。整体中文表达自然、可直接发布。"
                f"{keep_line}{style_line}"
            ),
            "conservative": (
                f"请基于“{source_prompt}”做一版保守增强。目标任务是：{task_goal}。"
                "只补足结构、语义边界和输出稳定性，不要为了追求华丽而改变原意。"
                f"{keep_line}{style_line}"
            ),
        }

    def _prescreen_candidates(
        self,
        *,
        candidates: dict[str, str],
        source_prompt: str,
        task_goal: str,
        keep_terms: list[str],
        scenario: str,
    ) -> list[CandidateResult]:
        scenario_weights = self.rubric.get("scenario_weights", {}).get(scenario) or {}
        case = self._pick_eval_case(scenario)
        results: list[CandidateResult] = []
        for label, prompt in candidates.items():
            scores = self._score_prompt_text(
                prompt_text=prompt,
                source_prompt=source_prompt,
                task_goal=task_goal,
                keep_terms=keep_terms,
                case=case,
            )
            total = self._weighted_total(scores, scenario_weights)
            notes = self._prompt_notes(label, prompt, keep_terms)
            results.append(
                CandidateResult(
                    label=label,
                    prompt=prompt,
                    stage="prescreen",
                    scores=scores,
                    total=total,
                    notes=notes,
                )
            )

        base_labels = {"original", "direct_rewrite"}
        baselines = [item for item in results if item.label in base_labels]
        challengers = sorted(
            [item for item in results if item.label not in base_labels],
            key=lambda item: item.total,
            reverse=True,
        )[:3]
        return baselines + challengers

    def _pick_eval_case(self, scenario: str) -> dict[str, Any]:
        cases = self.eval_dataset.get("cases", [])
        for item in cases:
            if item.get("scenario") == scenario:
                return item
        return cases[0] if cases else {}

    def _score_prompt_text(
        self,
        *,
        prompt_text: str,
        source_prompt: str,
        task_goal: str,
        keep_terms: list[str],
        case: dict[str, Any],
    ) -> dict[str, float]:
        prompt = ensure_text(prompt_text)
        instructions = ("要求", "输出", "任务", "目标", "请", "1.", "2.", "3.")
        clarity = 5.0 + min(sum(token in prompt for token in instructions) * 0.65, 3.0)
        clarity += 0.4 if task_goal in prompt else 0.0
        executability = 4.5 + min(sum(token in prompt for token in ("1.", "2.", "3.", "规则", "步骤", "结尾")) * 0.7, 3.4)
        literary_gain = 4.0 + min(sum(token in prompt for token in ("节奏", "韵味", "意象", "留白", "余味", "画面")) * 0.75, 3.2)
        imagery = 4.0 + min(sum(token in prompt for token in ("画面", "场景", "动作", "细节", "情绪", "第一眼")) * 0.8, 3.5)
        naturalness = 6.0 if len(prompt) < 220 else 5.0
        naturalness += 0.7 if "，" in prompt else 0.0
        ambiguity_control = 7.5
        ambiguity_control -= min(sum(token in prompt for token in ("高级", "炸裂", "绝绝子", "有感觉", "震撼")) * 0.8, 3.0)
        ambiguity_control += 0.3 if "不要" in prompt else 0.0
        keep_hits = sum(term in prompt for term in keep_terms)
        task_completion = 5.3
        task_completion += min(sum(token in prompt for token in ("核心", "重点", "受众", "卖点", "情绪", "记忆点")) * 0.35, 2.0)
        task_completion += 1.1 if keep_terms and keep_hits == len(keep_terms) else 0.0
        task_completion += 0.6 if source_prompt in prompt else 0.0

        preferred = case.get("preferred_dimensions", [])
        if "literary_gain" in preferred:
            literary_gain += 0.4
        if "imagery" in preferred:
            imagery += 0.4
        if "executability" in preferred:
            executability += 0.4
        if "clarity" in preferred:
            clarity += 0.3

        raw_scores = {
            "task_completion": task_completion,
            "clarity": clarity,
            "naturalness": naturalness,
            "literary_gain": literary_gain,
            "imagery": imagery,
            "ambiguity_control": ambiguity_control,
            "executability": executability,
        }
        return {key: max(1.0, min(round(value, 3), 10.0)) for key, value in raw_scores.items()}

    def _weighted_total(self, scores: dict[str, float], scenario_weights: dict[str, float]) -> float:
        total = 0.0
        weight_sum = 0.0
        for key in DIMENSION_KEYS:
            weight = float(scenario_weights.get(key, 1.0))
            total += scores.get(key, 0.0) * weight
            weight_sum += weight
        return total / weight_sum if weight_sum else 0.0

    def _prompt_notes(self, label: str, prompt: str, keep_terms: list[str]) -> list[str]:
        notes = {
            "original": ["保留原始用户表达，作为零优化基线。"],
            "direct_rewrite": ["单轮直接改写基线，用来拦住“改了但不一定更好”的情况。"],
            "clearer": ["优先补任务边界和输出结构。"],
            "literary": ["增强中文韵味，但需要防止过度抽象。"],
            "vivid": ["加强画面和情绪触发，更适合抓眼场景。"],
            "executable": ["更强调步骤和可执行性。"],
            "conservative": ["偏保守增强，用于无明显赢家时兜底。"],
        }.get(label, [])
        if keep_terms:
            notes.append("已纳入保留关键词约束。")
        if len(prompt) > 200:
            notes.append("结构较完整，但要留意输入成本。")
        return notes

    def _run_replay(
        self,
        *,
        candidates: list[CandidateResult],
        task_goal: str,
        analysis: dict[str, Any],
        scenario: str,
        keep_terms: list[str],
    ) -> list[CandidateResult]:
        scenario_weights = self.rubric.get("scenario_weights", {}).get(scenario) or {}
        replayed: list[CandidateResult] = []
        for candidate in candidates:
            generated_output = self._simulate_output(candidate.prompt, task_goal)
            scores, summary = self._judge_output(
                candidate_prompt=candidate.prompt,
                generated_output=generated_output,
                task_goal=task_goal,
                analysis=analysis,
                scenario_weights=scenario_weights,
                keep_terms=keep_terms,
            )
            prompt_signal = candidate.total * 0.35
            replay_signal = self._weighted_total(scores, scenario_weights) * 0.65
            replayed.append(
                CandidateResult(
                    label=candidate.label,
                    prompt=candidate.prompt,
                    stage="replay",
                    scores=scores,
                    total=prompt_signal + replay_signal,
                    notes=candidate.notes,
                    generated_output=generated_output,
                    judge_summary=summary,
                )
            )
        return replayed

    def _simulate_output(self, candidate_prompt: str, task_goal: str) -> str:
        fallback = self._heuristic_generated_output(task_goal)
        system_prompt = self._template(
            "simulate_system_prompt",
            "你是一名中文内容创作者。请根据给定提示词直接产出结果，只输出成稿。",
        )
        user_prompt = json.dumps(
            {"task_goal": task_goal, "prompt": candidate_prompt},
            ensure_ascii=False,
        )
        response = self.llm.generate_text(system_prompt, user_prompt, fallback)
        return ensure_text(response.get("text", "")) or fallback

    def _heuristic_generated_output(self, task_goal: str) -> str:
        if "标题" in task_goal:
            return "一眼记住你的亮点：把核心价值说得更抓人，也更具体。"
        if "口播" in task_goal or "短视频" in task_goal:
            return "别让表达只停留在说明，而要一开口就让人看见重点。先把最核心的信息前置，再用细节托住情绪。"
        return f"围绕“{task_goal}”，先把最核心的信息前置，再用一两个具体细节把情绪和画面托起来。"

    def _judge_output(
        self,
        *,
        candidate_prompt: str,
        generated_output: str,
        task_goal: str,
        analysis: dict[str, Any],
        scenario_weights: dict[str, float],
        keep_terms: list[str],
    ) -> tuple[dict[str, float], str]:
        fallback_scores = self._heuristic_output_scores(
            candidate_prompt=candidate_prompt,
            generated_output=generated_output,
            keep_terms=keep_terms,
        )
        fallback_summary = "基于固定维度完成输出评估，重点检查完成度、自然度和是否因文采增加歧义。"
        system_prompt = self._template(
            "judge_system_prompt",
            (
                "你是中文创作评测器。"
                "请只输出 JSON 对象，字段为 task_completion、clarity、naturalness、literary_gain、imagery、ambiguity_control、executability、summary。"
                "分数范围 1 到 10。"
            ),
        )
        user_prompt = json.dumps(
            {
                "task_goal": task_goal,
                "analysis": analysis,
                "candidate_prompt": candidate_prompt,
                "generated_output": generated_output,
                "keep_terms": keep_terms,
            },
            ensure_ascii=False,
        )
        fallback_text = json.dumps({**fallback_scores, "summary": fallback_summary}, ensure_ascii=False)
        response = self.llm.generate_text(system_prompt, user_prompt, fallback_text)
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

    def _heuristic_output_scores(
        self,
        *,
        candidate_prompt: str,
        generated_output: str,
        keep_terms: list[str],
    ) -> dict[str, float]:
        output = ensure_text(generated_output)
        task_completion = 5.2 + min(sum(token in output for token in ("核心", "重点", "细节", "情绪", "价值")) * 0.55, 3.0)
        clarity = 5.4 + min(output.count("，") * 0.25, 2.2)
        clarity += 0.6 if len(output) > 28 else 0.0
        naturalness = 6.2 if 18 <= len(output) <= 180 else 5.0
        literary_gain = 4.2 + min(sum(token in output for token in ("画面", "情绪", "余味", "节奏", "抓住")) * 0.7, 3.0)
        imagery = 4.0 + min(sum(token in output for token in ("看见", "第一眼", "细节", "场景", "动作")) * 0.75, 3.0)
        executability = 5.0 + min(sum(token in candidate_prompt for token in ("输出", "步骤", "结尾", "核心钩子")) * 0.45, 2.0)
        ambiguity_control = 7.0 - min(sum(token in output for token in ("感觉", "高级", "震撼")) * 0.65, 2.0)
        if keep_terms:
            ambiguity_control += 0.4 if all(term in candidate_prompt for term in keep_terms) else -0.5

        raw_scores = {
            "task_completion": task_completion,
            "clarity": clarity,
            "naturalness": naturalness,
            "literary_gain": literary_gain,
            "imagery": imagery,
            "ambiguity_control": ambiguity_control,
            "executability": executability,
        }
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

    def _confidence_band(self, winner_total: float, baseline_total: float) -> str:
        margin = winner_total - baseline_total
        if margin >= 0.8:
            return "high"
        if margin >= 0.35:
            return "medium"
        return "low"

    def _build_brief_explanation(
        self,
        *,
        winner: CandidateResult,
        direct_baseline: CandidateResult,
        original_baseline: CandidateResult,
        analysis: dict[str, Any],
    ) -> list[str]:
        reasons = []
        if winner.scores["clarity"] >= max(direct_baseline.scores["clarity"], original_baseline.scores["clarity"]):
            reasons.append("这版把任务边界和输出重点说得更清楚，模型更容易稳定抓住你的真实意图。")
        if winner.scores["literary_gain"] >= direct_baseline.scores["literary_gain"]:
            reasons.append("它保留了中文表达的文采和节奏感，但没有把修辞放到信息之前。")
        if winner.scores["ambiguity_control"] >= direct_baseline.scores["ambiguity_control"]:
            reasons.append("相比更激进的写法，这版对歧义的控制更稳，不容易把“有文采”变成“空泛”。")
        if not reasons:
            reasons.append("当前胜出版本主要赢在更稳，它没有明显冒进，但比原提示词更容易得到可用结果。")
        if analysis.get("cultural_signals"):
            reasons.append("系统也保留了你原句里的中文语感线索，没有把它简单压扁成干巴巴的说明句。")
        return reasons[:4]

    def _build_history_item(
        self,
        *,
        session_id: str,
        source_prompt: str,
        task_goal: str,
        best_candidate: CandidateResult,
        direct_baseline: CandidateResult,
        original_baseline: CandidateResult,
        explanation: list[str],
        analysis: dict[str, Any],
        confidence_band: str,
        save_enabled: bool,
        platform_tag: str | None,
        style_hint: str | None,
        keep_terms: list[str],
        replayed: list[CandidateResult],
    ) -> dict[str, Any]:
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
            "score_summary": {
                "winner_total": round(best_candidate.total, 3),
                "original_baseline_total": round(original_baseline.total, 3),
                "direct_baseline_total": round(direct_baseline.total, 3),
            },
            "feedback": {
                "copied": False,
                "adopted": False,
                "closer_to_goal": None,
                "note": "",
                "metadata": {},
                "updated_at": "",
            },
            "internal_trace": {
                "winner": best_candidate.to_dict(),
                "baselines": {
                    "original": original_baseline.to_dict(),
                    "direct_rewrite": direct_baseline.to_dict(),
                },
                "replayed": [item.to_dict() for item in replayed],
            },
        }

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
