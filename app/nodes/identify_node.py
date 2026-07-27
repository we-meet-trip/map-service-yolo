"""Gemini Vision 객체 식별 노드.

YOLO 가 탐지한 객체 + 위치 정보를 Gemini Vision 에 보내
정확한 이름·카테고리·설명·검색 쿼리를 추출한다.

위치 컨텍스트를 함께 전달해 "건물 + 경복궁 근처" → "경복궁" 처럼
정확도를 높인다.
"""
from __future__ import annotations

import asyncio
import json
import logging

from google import genai
from google.genai import types as genai_types

from app.agent_settings import get_settings
from app.schemas.vision_schemas import GraphState, IdentifyResult

logger = logging.getLogger(__name__)

_IDENTIFY_PROMPT = """\
아래 정보를 바탕으로 사진 속 객체를 정확히 식별하고, 사용자의 질문이 있으면 그에 맞춰 답해줘.

탐지된 객체 레이블: {label} (신뢰도: {confidence:.0%})
{location_line}
{voice_line}

반드시 JSON 으로만 응답해. 다른 텍스트 없이:
{{
  "name": "정확한 이름 (예: 경복궁, 비빔밥, 아이폰15)",
  "category": "landmark | food | product | animal | plant | other 중 하나",
  "description": "사용자 질문에 답하는 두 세 문장 한국어 설명",
  "search_query": "검색에 최적화된 한국어 쿼리"
}}
"""


async def identify_node(state: GraphState) -> GraphState:
    """Gemini Vision 으로 탐지 객체를 식별하는 LangGraph 노드.

    state.detected_object 를 읽어 state.identify_result 를 채운다.
    location 은 선택값이며 없으면 프롬프트에서 생략한다.
    식별 실패 시 state.error 에 사유를 기록하고 반환.
    """
    if state.detected_object is None:
        state.error = "identify_node: detected_object is None"
        return state

    settings = get_settings()
    client = genai.Client(api_key=settings.GEMINI_API_KEY.get_secret_value())

    location_line = (
        f"사용자 현재 위치: 위도 {state.location.lat}, 경도 {state.location.lng}"
        if state.location
        else "사용자 현재 위치: 정보 없음"
    )
    voice_line = (
        f"사용자 질문: {state.voice_text}"
        if state.voice_text
        else ""
    )

    prompt = _IDENTIFY_PROMPT.format(
        label=state.detected_object.label,
        confidence=state.detected_object.confidence,
        location_line=location_line,
        voice_line=voice_line,
    )

    # 프레임 이미지를 인라인으로 첨부
    image_part = genai_types.Part.from_bytes(
        data=__import__("base64").b64decode(state.frame_b64),
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
            "identify_node: name=%s category=%s",
            state.identify_result.name,
            state.identify_result.category,
        )
    except asyncio.TimeoutError:
        state.error = "identify_node: Gemini timeout"
        logger.warning("identify_node: timeout session=%s", state.session_id)
    except Exception as e:
        state.error = f"identify_node: {e}"
        logger.exception("identify_node: error session=%s", state.session_id)

    return state
