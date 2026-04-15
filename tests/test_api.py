from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


class ApiTests(unittest.TestCase):
    def test_health_endpoint(self) -> None:
        response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "healthy")
        self.assertIn(payload["llm_mode"], {"stub", "live"})

    def test_runtime_config_endpoint(self) -> None:
        response = client.get("/api/prompt-copilot/config")

        self.assertEqual(response.status_code, 200)
        payload = response.json()["data"]
        self.assertIn("template_version", payload)
        self.assertIn("llm_mode", payload)
        self.assertTrue(payload["strategies"])
        self.assertTrue(payload["provider_presets"])

    def test_optimize_history_feedback_flow(self) -> None:
        optimize = client.post(
            "/api/prompt-copilot/optimize",
            json={
                "user_id": "api-user",
                "source_prompt": "帮我把这句开场写得更引人注目。",
                "task_goal": "短视频开场前三秒更抓人",
                "platform_tag": "短视频",
                "style_hint": "有文采但别虚",
                "must_keep_terms": ["引人注目"],
                "save_session": True,
                "strategy": "literary",
            },
        )

        self.assertEqual(optimize.status_code, 200)
        payload = optimize.json()["data"]
        session_id = payload["session_id"]
        self.assertEqual(payload["strategy"], "literary")

        history = client.get("/api/prompt-copilot/history", params={"user_id": "api-user"})
        self.assertEqual(history.status_code, 200)
        self.assertTrue(history.json()["data"]["sessions"])

        detail = client.get(f"/api/prompt-copilot/history/{session_id}", params={"user_id": "api-user"})
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["data"]["session"]["session_id"], session_id)

        feedback = client.post(
            "/api/prompt-copilot/feedback",
            params={"session_id": session_id},
            json={
                "user_id": "api-user",
                "copied": True,
                "closer_to_goal": True,
                "adopted": True,
                "edited_prompt": "这是我最后真的会发出去的版本。",
            },
        )
        self.assertEqual(feedback.status_code, 200)
        self.assertTrue(feedback.json()["data"]["session"]["feedback"]["copied"])
        self.assertIn("signal_score", feedback.json()["data"]["session"]["feedback_summary"])

    def test_latest_report_endpoint(self) -> None:
        response = client.get("/api/prompt-copilot/report/latest")

        self.assertEqual(response.status_code, 200)
        self.assertIn("report", response.json()["data"])


if __name__ == "__main__":
    unittest.main()
