"""Gemini Vision 객체 식별 노드.

YOLO 레이블에 의존하지 않고 Gemini Vision 이 이미지를 직접 보고
정확한 이름·카테고리·설명·검색 쿼리를 추출한다.

위치 컨텍스트를 함께 전달해 랜드마크·음식·제품 등의 정확도를 높인다.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging

from google import genai
from google.genai import types as genai_types

from app.agent_settings import get_settings
from app.nodes.kakao_local import kakao_reverse_geocode
from app.schemas.vision_schemas import GraphState, IdentifyResult

logger = logging.getLogger(__name__)


_IDENTIFY_PROMPT = """\
이 사진을 보고 가장 주목할 만한 객체 또는 장소를 정확히 식별해줘.
{location_line}
{voice_line}

반드시 JSON 으로만 응답해. 다른 텍스트 없이:
{{
  "name": "정확한 이름 (예: 경복궁, 비빔밥, 아이폰15)",
  "category": "landmark | food | product | animal | plant | other 중 하나",
  "description": "사용자 질문에 답하거나, 질문이 없으면 이 객체에 대한 흥미로운 두 세 문장 한국어 설명",
  "search_query": "검색에 최적화된 한국어 쿼리"
}}
"""


async def identify_node(state: GraphState) -> GraphState:
    """Gemini Vision 이 이미지를 직접 분석해 객체를 식별하는 LangGraph 노드.

    YOLO detected_object 는 참고용 로그에만 사용하며, 핵심 식별은 Gemini Vision 이 담당.
    location 은 선택값이며 없으면 프롬프트에서 생략한다.
    식별 실패 시 state.error 에 사유를 기록하고 반환.
    """
    if not state.frame_b64:
        state.error = "identify_node: frame_b64 is empty"
        return state

    settings = get_settings()
    client = genai.Client(api_key=settings.GEMINI_API_KEY.get_secret_value())

    if state.location:
        address = ""
        if settings.KAKAO_REST_API_KEY:
            address = await kakao_reverse_geocode(
                state.location.lat, state.location.lng, settings.KAKAO_REST_API_KEY
            )
        if address:
            location_line = f"사용자 현재 위치: {address}"
        else:
            location_line = f"사용자 현재 위치: 위도 {state.location.lat}, 경도 {state.location.lng}"
    else:
        location_line = ""
    voice_line = (
        f"사용자 질문: {state.voice_text}"
        if state.voice_text
        else ""
    )

    prompt = _IDENTIFY_PROMPT.format(
        location_line=location_line,
        voice_line=voice_line,
    )

    image_part = genai_types.Part.from_bytes(
        data=base64.b64decode(state.frame_b64),
        mime_type="image/jpeg",
    )

    try:
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=[prompt, image_part],
            ),
            timeout=settings.GEMINI_TIMEOUT_SECONDS,
        )
        raw = response.text.strip()
        # 마크다운 코드블록 제거
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        state.identify_result = IdentifyResult(**data)
        logger.info(
            "identify_node: name=%s category=%s yolo_hint=%s",
            state.identify_result.name,
            state.identify_result.category,
            state.detected_object.label if state.detected_object else "none",
        )
    except asyncio.TimeoutError:
        state.error = "identify_node: Gemini timeout"
        logger.warning("identify_node: timeout session=%s", state.session_id)
    except Exception as e:
        state.error = f"identify_node: {e}"
        logger.exception("identify_node: error session=%s", state.session_id)

    return state
