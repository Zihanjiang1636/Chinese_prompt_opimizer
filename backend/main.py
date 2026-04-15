"""FastAPI app for the standalone Chinese Prompt Optimizer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.config import ALLOWED_CORS_ORIGINS, DEFAULT_USER_ID, HOST, PORT
from backend.services import llm_service, prompt_copilot_service

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class ApiEnvelope(BaseModel):
    status: str
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class OptimizeRequest(BaseModel):
    user_id: str = DEFAULT_USER_ID
    source_prompt: str
    task_goal: str
    platform_tag: str | None = None
    style_hint: str | None = None
    must_keep_terms: list[str] = Field(default_factory=list)
    save_session: bool = True
    strategy: str = "balanced"


class FeedbackRequest(BaseModel):
    user_id: str = DEFAULT_USER_ID
    copied: bool | None = None
    adopted: bool | None = None
    closer_to_goal: bool | None = None
    edited_prompt: str | None = None
    note: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


app = FastAPI(
    title="Chinese Prompt Optimizer",
    description="Standalone Chinese prompt understanding and optimization tool.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(ALLOWED_CORS_ORIGINS),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "llm_mode": "stub" if llm_service.stub_mode else "live",
        "provider": llm_service.provider,
        "model": llm_service.model,
    }


@app.get("/api/prompt-copilot/config", response_model=ApiEnvelope)
async def runtime_config() -> ApiEnvelope:
    return ApiEnvelope(status="success", data=prompt_copilot_service.get_runtime_config())


@app.get("/api/prompt-copilot/report/latest", response_model=ApiEnvelope)
async def latest_report() -> ApiEnvelope:
    reports_dir = Path(__file__).resolve().parent.parent / "prompt-copilot" / "reports"
    report_files = sorted(reports_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not report_files:
        return ApiEnvelope(status="success", data={"report": None, "report_name": None})

    with report_files[0].open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    return ApiEnvelope(status="success", data={"report": payload, "report_name": report_files[0].name})


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.post("/api/prompt-copilot/optimize", response_model=ApiEnvelope)
async def optimize(payload: OptimizeRequest) -> ApiEnvelope:
    result = prompt_copilot_service.analyze_and_optimize(
        user_id=payload.user_id,
        source_prompt=payload.source_prompt,
        task_goal=payload.task_goal,
        platform_tag=payload.platform_tag,
        style_hint=payload.style_hint,
        must_keep_terms=payload.must_keep_terms,
        save_session=payload.save_session,
        strategy=payload.strategy,
    )
    return ApiEnvelope(status="success" if result["status"] == "success" else "error", message=result.get("message", ""), data=result)


@app.get("/api/prompt-copilot/history", response_model=ApiEnvelope)
async def history(user_id: str = Query(DEFAULT_USER_ID)) -> ApiEnvelope:
    return ApiEnvelope(status="success", data={"sessions": prompt_copilot_service.list_history(user_id)})


@app.get("/api/prompt-copilot/history/{session_id}", response_model=ApiEnvelope)
async def history_detail(session_id: str, user_id: str = Query(DEFAULT_USER_ID)) -> ApiEnvelope:
    item = prompt_copilot_service.get_history_item(user_id, session_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return ApiEnvelope(status="success", data={"session": item})


@app.post("/api/prompt-copilot/feedback", response_model=ApiEnvelope)
async def feedback(session_id: str = Query(...), payload: FeedbackRequest | None = None) -> ApiEnvelope:
    request = payload or FeedbackRequest()
    result = prompt_copilot_service.record_feedback(
        user_id=request.user_id,
        session_id=session_id,
        copied=request.copied,
        adopted=request.adopted,
        closer_to_goal=request.closer_to_goal,
        edited_prompt=request.edited_prompt,
        note=request.note,
        metadata=request.metadata,
    )
    if result["status"] != "success":
        raise HTTPException(status_code=404, detail=result["message"])
    return ApiEnvelope(status="success", message=result["message"], data=result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host=HOST, port=PORT, reload=False, log_level="info")

