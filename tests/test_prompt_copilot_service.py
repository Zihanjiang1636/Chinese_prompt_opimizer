from __future__ import annotations

import unittest
from copy import deepcopy
from pathlib import Path

from backend.services.prompt_copilot_service import PromptCopilotService


class DatabaseStub:
    def __init__(self) -> None:
        self.user_state: dict[str, dict[str, object]] = {}

    def read_user_file(self, user_id: str, filename: str, default):
        return deepcopy(self.user_state.get(user_id, {}).get(filename, default))

    def write_user_file(self, user_id: str, filename: str, payload) -> None:
        scoped = self.user_state.setdefault(user_id, {})
        scoped[filename] = deepcopy(payload)


class LLMStub:
    def generate_text(self, system_prompt: str, user_prompt: str, fallback_text: str, **_: object):
        return {"text": fallback_text}


class PromptCopilotServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        asset_root = Path(__file__).resolve().parents[1] / "prompt-copilot"
        self.database = DatabaseStub()
        self.service = PromptCopilotService(database=self.database, llm=LLMStub(), asset_root=asset_root)

    def test_analyze_and_optimize_saves_history_and_beats_baseline(self) -> None:
        result = self.service.analyze_and_optimize(
            user_id="demo-user",
            source_prompt="帮我把这句开场写得更引人注目。",
            task_goal="短视频开场前三秒更抓人",
            must_keep_terms=["引人注目"],
            strategy="conversion",
            save_session=True,
        )

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["session_id"].startswith("pc-"))
        self.assertTrue(result["best_prompt"])
        self.assertIn(result["confidence_band"], {"low", "medium", "high"})
        self.assertEqual(result["strategy"], "conversion")

        stored = self.database.read_user_file("demo-user", "prompt_copilot_history.json", {"sessions": []})
        self.assertEqual(len(stored["sessions"]), 1)
        session = stored["sessions"][0]
        self.assertEqual(session["session_id"], result["session_id"])
        self.assertEqual(session["best_prompt"], result["best_prompt"])
        self.assertIn("winner_total", session["score_summary"])

    def test_save_session_false_does_not_write_history(self) -> None:
        result = self.service.analyze_and_optimize(
            user_id="demo-user",
            source_prompt="帮我把这句种草文案说得更有感觉。",
            task_goal="种草文案增强真实感和想象空间",
            save_session=False,
        )

        self.assertEqual(result["status"], "success")
        stored = self.database.read_user_file("demo-user", "prompt_copilot_history.json", {"sessions": []})
        self.assertEqual(stored["sessions"], [])

    def test_record_feedback_updates_session(self) -> None:
        result = self.service.analyze_and_optimize(
            user_id="demo-user",
            source_prompt="帮我写一句更吸引注意的短视频开场。",
            task_goal="短视频开场前三秒更抓人",
            save_session=True,
        )

        feedback = self.service.record_feedback(
            user_id="demo-user",
            session_id=result["session_id"],
            copied=True,
            adopted=True,
            closer_to_goal=True,
            edited_prompt="这一版我又手动改了一点，但整体保留了你的结构。",
            note="这一版更像我要的。",
            metadata={"source": "unit-test"},
        )

        self.assertEqual(feedback["status"], "success")
        saved = feedback["session"]["feedback"]
        self.assertTrue(saved["copied"])
        self.assertTrue(saved["adopted"])
        self.assertTrue(saved["closer_to_goal"])
        self.assertTrue(saved["edited_prompt"])
        self.assertEqual(saved["note"], "这一版更像我要的。")
        self.assertIn("signal_score", feedback["session"]["feedback_summary"])


if __name__ == "__main__":
    unittest.main()
