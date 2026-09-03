"""WebSocket 엔드포인트 — 폰 카메라 스트림 수신.

클라이언트(모바일 앱)는 아래 프로토콜로 통신한다.

  연결: ws://<host>/ws/vision
  송신: JSON { frame_b64, session_id, voice_triggered }
  수신: JSON VisionResponse

음성 트리거 방식:
  1) VisionRequest 파싱.
  2) voice_triggered=false 이고 voice_text 만 있으면 텍스트 대화로 처리.
  3) voice_triggered=false 이고 voice_text 도 없으면 무시.
  4) voice_triggered=true 이면 YOLO 탐지를 즉시 실행한다.
  5) LangGraph 파이프라인(identify → search)을 실행한다.
     탐지 객체가 없어도 Gemini 가 이미지를 직접 보고 식별하므로 진행한다.
  6) 결과를 VisionResponse JSON 으로 반환.

연결당 처리 상한은 2)와 4)에서만, 외부를 부르기 직전에 한 칸씩 쓴다.
1)에서 형식이 깨져 되돌려 보내는 요청과 3)의 무시되는 프레임은 상한을 쓰지 않는다.
"""
from __future__ import annotations

import asyncio
import json
import logging

from fastapi import WebSocket, WebSocketDisconnect

from app.agent_settings import get_settings
from app.gemini_client import get_gemini_client
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

    # 이 엔드포인트는 자체 인증이 없다. 관문이 막는 것은 접속 시도 횟수뿐이라,
    # 한 번 맺은 연결로 계속 밀어 넣으면 외부 모델 호출 한도를 혼자 소진한다.
    # 그 한도는 일정 생성과 공유하므로 여기서 새면 추천까지 멈춘다.
    max_jobs = get_settings().WS_MAX_JOBS_PER_CONNECTION
    jobs = 0

    async def consume_quota(session_id: str) -> bool:
        """외부를 부르기 직전에 상한 한 칸을 쓴다. 남아 있지 않으면 알리고 닫는다.

        세는 대상은 실제로 외부를 부르는 처리다. 읽어 들인 프레임 수를 세면,
        형식이 깨진 요청이나 아무 일도 하지 않고 넘기는 프레임까지 상한을 깎아
        외부를 한 번도 부르지 않은 연결이 끊긴다. 남용을 막자는 상한이 정상
        사용자를 먼저 끊는 셈이라, 세는 자리를 호출 직전으로 옮겼다.
        """
        nonlocal jobs
        if jobs >= max_jobs:
            logger.warning(
                "vision_ws: 연결당 처리 상한 도달 max=%d session=%s",
                max_jobs, session_id,
            )
            await websocket.send_text(
                VisionResponse(
                    session_id=session_id,
                    status="failed",
                    error="이 연결에서 처리할 수 있는 횟수를 넘었어요. 다시 연결해주세요.",
                ).model_dump_json()
            )
            await websocket.close()
            return False
        jobs += 1
        return True

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
                if not await consume_quota(req.session_id):
                    return
                settings = get_settings()
                client = get_gemini_client()
                answer: str | None = None

                # 장소 검색 의도 감지 → Kakao 로컬 API 우선 시도
                if is_local_query(req.voice_text) and settings.KAKAO_REST_API_KEY:
                    lat = req.location.lat if req.location else None
                    lng = req.location.lng if req.location else None
                    keyword = extract_search_keyword(req.voice_text)
                    # 좌표 값은 기기 위치라 로그에 남기지 않는다. 있었는지만 남긴다.
                    logger.info("kakao_local_search keyword=%s has_loc=%s", keyword, lat is not None)
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

                # Kakao 결과 없거나 장소 검색이 아닌 경우 → Gemini
                if answer is None:
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

            if not await consume_quota(req.session_id):
                return

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
                # 보여줄 것이 있으면 성공이다. 식별을 못 했을 때만 실패로 알린다.
                status="done" if result.identify_result else "failed",
                error=result.error,
            )
            await websocket.send_text(response.model_dump_json())

    except WebSocketDisconnect:
        logger.info("vision_ws: client disconnected")
    except Exception:
        logger.exception("vision_ws: unexpected error")
