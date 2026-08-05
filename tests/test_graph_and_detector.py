"""그래프 배선과 프레임 디코딩 테스트.

그래프는 노드 두 개를 순서대로 잇는 게 전부라, 순서가 바뀌면 검색이 식별 결과 없이
먼저 돌아 아무것도 못 찾는다. 디코딩은 사용자가 보낸 값이 그대로 들어오는 자리라
깨진 입력에도 죽지 않아야 한다.
"""
from __future__ import annotations

import asyncio
import base64

import pytest


def run(coro):
    return asyncio.run(coro)


# ───────────────────────── 그래프 ─────────────────────────


def test_graph_runs_identify_then_search(monkeypatch):
    """identify 가 먼저, search 가 나중이다. 순서가 뒤집히면 검색이 빈손으로 돈다."""
    from app.graph import vision_graph as module
    from app.schemas.vision_schemas import GraphState, IdentifyResult

    order: list[str] = []

    async def identify(state):
        order.append("identify")
        state.identify_result = IdentifyResult(
            name="n", category="other", description="d", search_query="q"
        )
        return state

    async def search(state):
        order.append("search")
        assert state.identify_result is not None, "검색은 식별 결과를 받아야 한다"
        return state

    monkeypatch.setattr(module, "identify_node", identify)
    monkeypatch.setattr(module, "search_node", search)

    graph = module.build_graph()
    result = run(graph.ainvoke(
        GraphState(session_id="s", frame_b64=base64.b64encode(b"x").decode())
    ))

    assert order == ["identify", "search"]
    assert result is not None


def test_graph_reaches_search_even_when_identify_failed(monkeypatch):
    """분기 엣지가 없으므로 search 는 항상 불린다 — 스스로 건너뛰는 책임은 노드에 있다."""
    from app.graph import vision_graph as module
    from app.schemas.vision_schemas import GraphState

    visited: list[str] = []

    async def identify(state):
        visited.append("identify")
        state.error = "boom"
        return state

    async def search(state):
        visited.append("search")
        return state

    monkeypatch.setattr(module, "identify_node", identify)
    monkeypatch.setattr(module, "search_node", search)

    graph = module.build_graph()
    run(graph.ainvoke(
        GraphState(session_id="s", frame_b64=base64.b64encode(b"x").decode())
    ))

    assert visited == ["identify", "search"]


# ───────────────────────── 디코딩 ─────────────────────────


def test_detector_returns_empty_on_broken_frame():
    """디코딩이 실패해도 예외를 올리지 않는다 — 연결 하나가 통째로 끊긴다."""
    pytest.importorskip("ultralytics", reason="런타임 이미지에만 설치된다")
    from app.vision.detector import YoloDetector

    detector = YoloDetector.__new__(YoloDetector)
    detector._model = None
    detector._confidence = 0.5

    assert detector.detect("!!! not base64 !!!") == []
    assert detector.detect(base64.b64encode(b"not an image").decode()) == []
