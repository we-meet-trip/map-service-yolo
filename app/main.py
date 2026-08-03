"""yolo-vision-agent FastAPI 엔트리포인트.

lifespan:
  - YoloDetector 를 생성해 app.state 에 주입.
  - build_graph() 로 LangGraph 인스턴스 준비.

엔드포인트:
  GET  /health          헬스 체크.
  GET  /metrics         Prometheus 스크레이프용 지표.
  WS   /ws/vision       음성 트리거 프레임 수신 + 실시간 객체 인식.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, WebSocket
from prometheus_fastapi_instrumentator import Instrumentator

from app.agent_settings import get_settings
from app.api.ws import vision_ws
from app.graph.vision_graph import build_graph
from app.vision.detector import YoloDetector

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """앱 기동·종료 훅.

    기동:
      - GEMINI_API_KEY 검증.
      - YoloDetector 생성 → app.state 주입.
      - build_graph() 로 LangGraph 인스턴스 → app.state.graph.
      - app.state.job_timeout_seconds 설정.

    종료:
      - 별도 정리 리소스 없음(YOLO/Gemini 는 stateless).
    """
    settings = get_settings()
    if not settings.GEMINI_API_KEY.get_secret_value():
        raise RuntimeError("GEMINI_API_KEY is required")

    app.state.detector = YoloDetector(
        model_path=settings.YOLO_MODEL_PATH,
        confidence=settings.YOLO_CONFIDENCE,
    )
    app.state.graph = build_graph()
    app.state.job_timeout_seconds = settings.JOB_TIMEOUT_SECONDS

    logger.info(
        "yolo-vision-agent: initialized model=%s",
        settings.YOLO_MODEL_PATH,
    )
    yield
    logger.info("yolo-vision-agent: shutdown")


app = FastAPI(
    title="yolo-vision-agent",
    version="0.1.0",
    lifespan=lifespan,
)

# /metrics 노출 — 스택의 다른 FastAPI 서비스와 동일한 스크레이프 규약.
Instrumentator().instrument(app).expose(app, include_in_schema=False)


@app.get("/health")
async def health() -> dict[str, str]:
    """헬스 체크."""
    return {"status": "ok", "service": "yolo-vision-agent"}


@app.websocket("/ws/vision")
async def websocket_vision(websocket: WebSocket) -> None:
    """폰 카메라 스트림 수신 WebSocket 엔드포인트."""
    await vision_ws(websocket)
