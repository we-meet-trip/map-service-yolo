"""WebSocket 엔드포인트 — 폰 카메라 스트림 수신.

클라이언트(모바일 앱)는 아래 프로토콜로 통신한다.

  연결: ws://<host>/ws/vision
  송신: JSON { frame_b64, session_id, voice_triggered }
  수신: JSON VisionResponse

음성 트리거 방식:
  1) VisionRequest 파싱.
  2) voice_triggered=false 이면 무시.
  3) voice_triggered=true 이면 YOLO 탐지 즉시 실행.
  4) 탐지 객체가 있으면 LangGraph 파이프라인 실행.
  5) 결과를 VisionResponse JSON 으로 반환.
"""
from __future__ import annotations

import asyncio
import json
import logging

from fastapi import WebSocket, WebSocketDisconnect
from google import genai

from app.agent_settings import get_settings
from app.nodes.kakao_local import extract_search_keyword, is_local_query, kakao_local_search
from app.schemas.vision_schemas import (
    GraphState,
    IdentifyResult,
    VisionRequest,
    VisionResponse,
)

logger = logging.getLogger(__name__)


async def vision_ws(websocket: WebSocket) -> None:
    """WebSocket 연결 핸들러.

    `app.state.detector`, `app.state.graph` 를 lifespan 이 주입해야 한다.
    voice_triggered=true 인 프레임만 처리한다.
    """
    await websocket.accept()
    app = websocket.app
    detector = app.state.detector
    graph = app.state.graph
    timeout = app.state.job_timeout_seconds

    logger.info("vision_ws: client connected")
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                req = VisionRequest(**json.loads(raw))
            except Exception as e:
                await websocket.send_text(
                    VisionResponse(
                        session_id="unknown", status="failed", error=f"parse error: {e}"
                    ).model_dump_json()
                )
                continue

            # 텍스트 전용 채팅 모드 (이미지 없이 voice_text만 있는 경우)
            if not req.voice_triggered and req.voice_text:
                settings = get_settings()
                client = genai.Client(api_key=settings.GEMINI_API_KEY.get_secret_value())

                # 장소 검색 의도 감지 → Kakao 로컬 API
                if is_local_query(req.voice_text) and settings.KAKAO_REST_API_KEY:
                    lat = req.location.lat if req.location else None
                    lng = req.location.lng if req.location else None
                    keyword = extract_search_keyword(req.voice_text)
                    logger.info("kakao_local_search keyword=%s lat=%s lng=%s", keyword, lat, lng)
                    try:
                        places = await kakao_local_search(
                            query=keyword,
                            api_key=settings.KAKAO_REST_API_KEY,
                            lat=lat,
                            lng=lng,
                        )
                    except Exception as e:
                        places = []
                        logger.warning("kakao_local_search error: %s", e)

                    if places:
                        lines = []
                        for i, p in enumerate(places, 1):
                            dist = f" ({p['distance']}m)" if p["distance"] else ""
                            phone = f"  ☎ {p['phone']}" if p["phone"] else ""
                            lines.append(f"{i}. {p['name']}{dist}\n   {p['address']}{phone}")
                        answer = "\n".join(lines)
                    else:
                        answer = f"'{keyword}' 검색 결과가 없어요. 다른 키워드로 말씀해 주세요."
                else:
                    # 일반 대화 → Gemini
                    prompt_parts = []
                    if req.prior_context:
                        prompt_parts.append(f"[이전 인식 정보]\n{req.prior_context}")
                    if req.conversation_history:
                        history_lines = "\n".join(
                            f"{'사용자' if m.get('role') == 'user' else 'AI'}: {m.get('content', '')}"
                            for m in req.conversation_history[-8:]
                        )
                        prompt_parts.append(f"[대화 기록]\n{history_lines}")
                    prompt_parts.append(f"사용자 질문에 위 맥락을 참고해 친절하게 한국어로 두 세 문장으로 답해줘: {req.voice_text}")

                    try:
                        gemini_resp = await asyncio.wait_for(
                            client.aio.models.generate_content(
                                model=settings.GEMINI_MODEL,
                                contents=["\n\n".join(prompt_parts)],
                            ),
                            timeout=settings.GEMINI_TIMEOUT_SECONDS,
                        )
                        answer = gemini_resp.text.strip()
                    except Exception as e:
                        answer = f"답변 생성 실패: {e}"

                await websocket.send_text(
                    VisionResponse(
                        session_id=req.session_id,
                        identify_result=IdentifyResult(
                            name="AI 답변",
                            category="other",
                            description=answer,
                            search_query=req.voice_text,
                        ),
                        status="done",
                    ).model_dump_json()
                )
                continue

            # 음성 트리거가 아니면 무시
            if not req.voice_triggered:
                continue

            logger.info("vision_ws: voice triggered session=%s", req.session_id)

            # YOLO 탐지 (동기 → run_in_executor 로 블로킹 방지)
            loop = asyncio.get_event_loop()
            detected = await loop.run_in_executor(
                None, detector.detect, req.frame_b64
            )

            # 신뢰도 가장 높은 객체 선택 (없으면 None — Gemini 가 직접 식별)
            best = max(detected, key=lambda d: d.confidence) if detected else None

            # LangGraph 파이프라인 실행
            state = GraphState(
                session_id=req.session_id,
                location=req.location,
                frame_b64=req.frame_b64,
                voice_text=req.voice_text,
                detected_object=best,
            )
            try:
                raw_result = await asyncio.wait_for(
                    graph.ainvoke(state),
                    timeout=timeout,
                )
                result = GraphState(**raw_result) if isinstance(raw_result, dict) else raw_result
            except asyncio.TimeoutError:
                result = state
                result.error = f"pipeline timeout after {timeout:.0f}s"

            response = VisionResponse(
                session_id=req.session_id,
                detected_object=result.detected_object,
                identify_result=result.identify_result,
                search_results=result.search_results,
                status="failed" if result.error else "done",
                error=result.error,
            )
            await websocket.send_text(response.model_dump_json())

    except WebSocketDisconnect:
        logger.info("vision_ws: client disconnected")
    except Exception:
        logger.exception("vision_ws: unexpected error")
