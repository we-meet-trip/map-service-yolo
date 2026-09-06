"""검색 노드.

IdentifyResult.search_query 를 받아 외부 API 를 호출하고
SearchResult 리스트를 반환한다.

현재 구현: Wikipedia API (무료, 키 불필요).
확장 포인트: 카카오 로컬 API, Google Places API 등 추가 가능.
"""
from __future__ import annotations

import logging
import re
import urllib.parse

import httpx

from app.schemas.vision_schemas import GraphState, SearchResult

logger = logging.getLogger(__name__)

_WIKIPEDIA_API = "https://ko.wikipedia.org/w/api.php"
_SEARCH_LIMIT = 3


async def search_node(state: GraphState) -> GraphState:
    """검색 쿼리로 Wikipedia 를 조회해 state.search_results 를 채우는 노드.

    state.identify_result 가 없거나 state.error 가 있으면 스킵.
    검색 실패는 빈 목록으로 넘긴다 — 곁들이는 설명이라 식별 결과까지 버리지 않는다.
    """
    if state.error or state.identify_result is None:
        return state

    query = state.identify_result.search_query
    results: list[SearchResult] = []

    try:
        headers = {"User-Agent": "yolo-vision-agent/1.0 (https://github.com/we-meet-trip; dmlwjds2@gmail.com)"}
        async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
            # 1단계: 검색
            params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": _SEARCH_LIMIT,
                "format": "json",
                "uselang": "ko",
            }
            resp = await client.get(_WIKIPEDIA_API, params=params)
            resp.raise_for_status()
            data = resp.json()
            hits = data.get("query", {}).get("search", [])

            for hit in hits:
                title = hit.get("title", "")
                snippet = hit.get("snippet", "")
                # HTML 태그 제거
                snippet = re.sub(r"<[^>]+>", "", snippet)
                encoded = urllib.parse.quote(title.replace(" ", "_"))
                url = f"https://ko.wikipedia.org/wiki/{encoded}"
                results.append(
                    SearchResult(
                        title=title,
                        summary=snippet,
                        source="wikipedia",
                        url=url,
                    )
                )

        state.search_results = results
        logger.info(
            "search_node: results=%d", len(results),
        )
    except Exception as e:
        # 검색이 실패해도 식별 결과는 살린다. 여기서 사유를 남기면
        # 응답 전체가 실패로 나가 사진 속 대상을 알아내고도 아무것도 못 보여준다.
        state.search_results = []
        logger.warning(
            "search_node: 검색 실패 session=%s reason=%s",
            state.session_id, e,
        )

    return state
