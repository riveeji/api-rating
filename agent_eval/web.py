from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from agent_eval.assistant import AssistantService
from agent_eval.config import Settings, get_settings
from agent_eval.experiments import ExperimentService, build_services
from agent_eval.models import AssistantAskRequest, ExperimentRunRequest


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    services = build_services(settings)
    experiment_service = ExperimentService(services)
    assistant_service = AssistantService(services)
    templates = Jinja2Templates(directory=str(settings.templates_dir))

    app = FastAPI(title=settings.app_name, debug=settings.debug)
    app.mount("/static", StaticFiles(directory=str(settings.static_dir)), name="static")
    app.state.experiments = experiment_service
    app.state.assistant = assistant_service
    app.state.templates = templates

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request) -> HTMLResponse:
        payload = experiment_service.dashboard()
        return templates.TemplateResponse(
            request,
            "dashboard.html",
            {
                "request": request,
                "title": settings.app_name,
                "payload": payload,
                "leaderboard_json": json.dumps(payload["leaderboard"], ensure_ascii=False),
                "failures_json": json.dumps(payload["failures"], ensure_ascii=False),
            },
        )

    @app.get("/assistant")
    def assistant_home(request: Request) -> Any:
        payload = assistant_service.home()
        if wants_json(request):
            return JSONResponse(payload)
        return templates.TemplateResponse(
            request,
            "assistant.html",
            {"request": request, "payload": payload, "title": "在线知识库助手"},
        )

    @app.get("/assistant/{session_id}")
    def assistant_session(session_id: str, request: Request) -> Any:
        try:
            payload = assistant_service.session_detail(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if wants_json(request):
            return JSONResponse(payload)
        return templates.TemplateResponse(
            request,
            "assistant.html",
            {"request": request, "payload": payload, "title": "在线知识库助手"},
        )

    @app.post("/assistant/ask")
    def assistant_ask(request_body: AssistantAskRequest) -> dict[str, Any]:
        try:
            return assistant_service.ask(request_body)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/experiments")
    def experiments(request: Request) -> Any:
        items = experiment_service.experiments()
        if wants_json(request):
            return JSONResponse(items)
        return templates.TemplateResponse(
            request,
            "experiments.html",
            {"request": request, "items": items, "title": "实验列表"},
        )

    @app.get("/experiments/{experiment_id}")
    def experiment_detail(experiment_id: str, request: Request) -> Any:
        try:
            payload = experiment_service.experiment_detail(experiment_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if wants_json(request):
            return JSONResponse(payload)
        return templates.TemplateResponse(
            request,
            "experiment_detail.html",
            {
                "request": request,
                "payload": payload,
                "title": f"实验详情 {experiment_id}",
                "leaderboard_json": json.dumps(payload["leaderboard"], ensure_ascii=False),
                "failures_json": json.dumps(payload["failures"], ensure_ascii=False),
                "category_json": json.dumps(payload["category_breakdown"], ensure_ascii=False),
            },
        )

    @app.get("/leaderboard")
    def leaderboard(request: Request) -> Any:
        entries = experiment_service.leaderboard()
        if wants_json(request):
            return JSONResponse(entries)
        return templates.TemplateResponse(
            request,
            "leaderboard.html",
            {"request": request, "entries": entries, "title": "排行榜"},
        )

    @app.get("/tasks")
    def tasks() -> list[dict[str, Any]]:
        return experiment_service.tasks()

    @app.get("/runs/{run_id}")
    def run_detail(run_id: str, request: Request) -> Any:
        try:
            payload = experiment_service.run_detail(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if wants_json(request):
            return JSONResponse(payload)
        return templates.TemplateResponse(
            request,
            "run_detail.html",
            {"request": request, "payload": payload, "title": f"运行详情 {run_id}"},
        )

    @app.get("/failures")
    def failures(request: Request) -> Any:
        items = experiment_service.failures()
        if wants_json(request):
            return JSONResponse(items)
        return templates.TemplateResponse(
            request,
            "failures.html",
            {
                "request": request,
                "items": items,
                "title": "失败分析",
                "failures_json": json.dumps(items, ensure_ascii=False),
            },
        )

    @app.post("/experiments/run")
    def run_experiment(request_body: ExperimentRunRequest | None = None) -> dict[str, Any]:
        try:
            return experiment_service.run_experiment(request_body)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def wants_json(request: Request) -> bool:
    accept = request.headers.get("accept", "")
    return request.query_params.get("format") == "json" or "application/json" in accept.lower()


app = create_app()
