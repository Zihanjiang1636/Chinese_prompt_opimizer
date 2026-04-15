from __future__ import annotations

import unittest
from pathlib import Path

from backend.services.prompt_copilot_service import PromptCopilotService
from scripts.run_prompt_copilot_regression import run_cases


class LLMStub:
    def generate_text(self, system_prompt: str, user_prompt: str, fallback_text: str, **_: object):
        return {"text": fallback_text}


class DatabaseStub:
    def read_user_file(self, user_id: str, filename: str, default):
        return default

    def write_user_file(self, user_id: str, filename: str, payload) -> None:
        return None


class PromptCopilotRegressionScriptTests(unittest.TestCase):
    def test_run_cases_builds_summary(self) -> None:
        asset_root = Path(__file__).resolve().parents[1] / "prompt-copilot"
        service = PromptCopilotService(database=DatabaseStub(), llm=LLMStub(), asset_root=asset_root)
        cases = [
            {
                "id": "case-1",
                "scenario": "hook",
                "source_prompt": "帮我把这句开场写得更引人注目。",
                "task_goal": "短视频开场前三秒更抓人",
                "platform_tag": "短视频",
                "style_hint": "有文采但别虚",
                "must_keep_terms": ["引人注目"],
            },
            {
                "id": "case-2",
                "scenario": "headline",
                "source_prompt": "帮我把这个标题写得更吸引人。",
                "task_goal": "内容标题更抓人但不能标题党",
                "platform_tag": "内容标题",
                "style_hint": "抓眼但不夸张",
                "must_keep_terms": ["吸引人"],
            },
        ]

        report = run_cases(service, cases, "regression-runner", strategy="literary")

        self.assertEqual(report["summary"]["case_count"], 2)
        self.assertEqual(len(report["records"]), 2)
        self.assertEqual(report["summary"]["strategy"], "literary")
        self.assertIn("avg_winner_total", report["summary"])
        self.assertIn("winner_margin_vs_direct", report["records"][0])


if __name__ == "__main__":
    unittest.main()
