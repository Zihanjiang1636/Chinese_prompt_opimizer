"""Run a regression pack against Prompt Copilot and save a report."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from backend.services.prompt_copilot_service import ASSET_ROOT, PromptCopilotService


class DeterministicLLM:
    def generate_text(self, system_prompt: str, user_prompt: str, fallback_text: str, **_: object):
        return {"text": fallback_text}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Prompt Copilot regression cases.")
    parser.add_argument("--dataset", default=str(ASSET_ROOT / "datasets" / "real-world-prompt-regression.json"), help="Dataset JSON file to run.")
    parser.add_argument("--output", default="", help="Optional output JSON path.")
    parser.add_argument("--limit", type=int, default=0, help="Optional number of cases to run from the top of the dataset.")
    parser.add_argument("--user-id", default="prompt-copilot-regression", help="Synthetic user id used for running the service.")
    parser.add_argument("--strategy", default="balanced", help="Optimization strategy to apply to the full regression run.")
    parser.add_argument("--stub", action="store_true", help="Use deterministic fallback outputs instead of live LLM calls.")
    return parser.parse_args()


def load_dataset(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def build_report_filename(dataset_path: Path) -> Path:
    reports_dir = ASSET_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir / f"{dataset_path.stem}-report.json"


def run_cases(service: PromptCopilotService, cases: list[dict], user_id: str, strategy: str = "balanced") -> dict:
    records: list[dict] = []
    for case in cases:
        result = service.analyze_and_optimize(
            user_id=user_id,
            source_prompt=case["source_prompt"],
            task_goal=case["task_goal"],
            platform_tag=case.get("platform_tag"),
            style_hint=case.get("style_hint"),
            must_keep_terms=case.get("must_keep_terms", []),
            save_session=False,
            strategy=strategy,
        )
        score_summary = result["score_summary"]
        records.append(
            {
                "id": case["id"],
                "scenario": case.get("scenario", ""),
                "task_goal": case["task_goal"],
                "selected_label": result["selected_label"],
                "confidence_band": result["confidence_band"],
                "strategy": result["strategy"],
                "strategy_label": result["strategy_label"],
                "template_version": result["template_version"],
                "score_summary": score_summary,
                "winner_margin_vs_original": round(score_summary["winner_total"] - score_summary["original_baseline_total"], 3),
                "winner_margin_vs_direct": round(score_summary["winner_total"] - score_summary["direct_baseline_total"], 3),
                "best_prompt": result["best_prompt"],
                "brief_explanation": result["brief_explanation"],
            }
        )

    both_better = [item for item in records if item["winner_margin_vs_original"] >= 0 and item["winner_margin_vs_direct"] >= 0]
    winner_totals = [item["score_summary"]["winner_total"] for item in records]
    original_totals = [item["score_summary"]["original_baseline_total"] for item in records]
    direct_totals = [item["score_summary"]["direct_baseline_total"] for item in records]
    label_breakdown = dict(Counter(item["selected_label"] for item in records))

    return {
        "summary": {
            "case_count": len(records),
            "strategy": strategy,
            "avg_winner_total": round(mean(winner_totals), 3) if winner_totals else 0.0,
            "avg_original_baseline_total": round(mean(original_totals), 3) if original_totals else 0.0,
            "avg_direct_baseline_total": round(mean(direct_totals), 3) if direct_totals else 0.0,
            "avg_margin_vs_original": round(mean(item["winner_margin_vs_original"] for item in records), 3) if records else 0.0,
            "avg_margin_vs_direct": round(mean(item["winner_margin_vs_direct"] for item in records), 3) if records else 0.0,
            "both_baselines_beaten": len(both_better),
            "both_baselines_beaten_ratio": round(len(both_better) / len(records), 3) if records else 0.0,
            "selected_label_breakdown": label_breakdown,
        },
        "records": records,
    }


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset)
    payload = load_dataset(dataset_path)
    cases = payload.get("cases", [])
    if args.limit > 0:
        cases = cases[: args.limit]

    service = PromptCopilotService(asset_root=ASSET_ROOT, llm=DeterministicLLM() if args.stub else None)
    report = run_cases(service, cases, args.user_id, strategy=args.strategy)
    output_path = Path(args.output) if args.output else build_report_filename(dataset_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig") as handle:
        json.dump({"dataset": {"path": str(dataset_path), "version": payload.get("version", ""), "description": payload.get("description", "")}, **report}, handle, ensure_ascii=False, indent=2)

    print(f"Regression report written to: {output_path}")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

