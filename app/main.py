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
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, WebSocket
from prometheus_fastapi_instrumentator import Instrumentator

from app.agent_settings import get_settings
from app.api.ws import vision_ws
from app.api.permit import PermitClient
from app.vision.runtime import VisionRuntime

# import 만으로 uvicorn.access·httpx 로거에 좌표 가림 필터가 걸린다.
from app.log_redaction import CoordinateRedactingFilter
from app.graph.vision_graph import build_graph
from app.vision.detector import YoloDetector

logger = logging.getLogger(__name__)


def _configure_logging(level: str) -> None:
    """루트 로거에 stdout 핸들러를 붙인다(핸들러가 없을 때만).

    uvicorn 의 기본 로깅 설정은 `uvicorn*` 로거만 구성하고 루트 로거는 건드리지
    않는다. 그래서 이 함수 없이는 `app.*` 로거로 남긴 기록이 출력 대상을 못 찾아
    전량 유실된다 — 어떤 연결이 무엇을 인식했고 무엇이 실패했는지가 보이지 않는다.
    경고 이상은 파이썬의 최후 수단 핸들러로 새어 나가지만 시각도 로거명도 없다.

    좌표 가림 필터를 이 핸들러에도 건다. 외부 요청 로거에만 걸어 두면
    애플리케이션 로그로 나가는 좌표는 그대로 남아 가림이 반쪽이 된다.

    이미 핸들러가 있으면(uvicorn `--log-config`, 테스트 하니스) 그 설정을 존중하고
    아무것도 하지 않는다 — 핸들러를 덧붙이면 같은 로그가 두 줄씩 출력된다.
    """
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    handler.addFilter(CoordinateRedactingFilter())
    root.addHandler(handler)
    root.setLevel(level.upper())


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
    # 루트 로거를 가장 먼저 세운다. 아래 부팅 실패 사유와 모델 적재 기록이
    # 전부 app.* 로거를 쓴다.
    _configure_logging(settings.LOG_LEVEL)
    if not settings.GEMINI_API_KEY.get_secret_value():
        raise RuntimeError("GEMINI_API_KEY is required")

    app.state.vision_permit = PermitClient(
        settings.USER_SERVICE_BASE_URL, settings.VISION_INTERNAL_TOKEN.get_secret_value(),
        settings.VISION_PERMIT_TIMEOUT_SECONDS,
    )
    app.state.vision_runtime = VisionRuntime(settings.VISION_MAX_CONCURRENT_JOBS)

    try:
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
    finally:
        app.state.vision_runtime.close()
        await app.state.vision_permit.aclose()
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
