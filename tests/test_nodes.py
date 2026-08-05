"""식별·검색 노드의 저하 계약 테스트.

두 노드는 실패를 서로 다르게 다뤄야 한다. 식별이 실패하면 보여 줄 것이 없으니 사유를
남기고, 검색이 실패하면 곁들이는 설명이 없을 뿐이니 식별 결과는 살려서 넘긴다. 이
비대칭이 무너지면 사진 속 대상을 알아내고도 화면에 아무것도 뜨지 않는다.
"""
from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace

import pytest

from app.schemas.vision_schemas import GraphState, IdentifyResult


def run(coro):
    return asyncio.run(coro)


def state(**kwargs) -> GraphState:
    body = {"session_id": "s1", "frame_b64": base64.b64encode(b"jpeg").decode()}
    body.update(kwargs)
    return GraphState(**body)


def identified(**kwargs) -> IdentifyResult:
    body = {
        "name": "경복궁",
        "category": "landmark",
        "description": "설명",
        "search_query": "경복궁",
    }
    body.update(kwargs)
    return IdentifyResult(**body)


# ───────────────────────── search_node ─────────────────────────


def test_search_skips_when_identify_missing():
    from app.nodes.search_node import search_node

    result = run(search_node(state()))

    assert result.search_results == []
    assert result.error is None


def test_search_skips_when_error_present():
    from app.nodes.search_node import search_node

    result = run(search_node(state(error="식별 실패", identify_result=identified())))

    assert result.search_results == []


def test_search_failure_keeps_identify_result(monkeypatch):
    """검색이 죽어도 식별 결과는 남는다. 여기서 error 를 세우면 응답 전체가 실패로 나간다."""
    from app.nodes import search_node as module

    class Boom:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            raise RuntimeError("wikipedia down")

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(module.httpx, "AsyncClient", Boom)

    result = run(module.search_node(state(identify_result=identified())))

    assert result.search_results == []
    assert result.error is None, "검색 실패가 식별 결과를 버리게 하면 안 된다"
    assert result.identify_result is not None


def test_search_strips_html_and_builds_url(monkeypatch):
    """검색 요약의 태그를 걷어 내고, 제목으로 원문 링크를 만든다."""
    from app.nodes import search_node as module

    payload = {"query": {"search": [
        {"title": "경복궁", "snippet": "조선의 <span class=\"x\">법궁</span>이다"},
    ]}}

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return payload

    class Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, params=None):
            return Resp()

    monkeypatch.setattr(module.httpx, "AsyncClient", Client)

    result = run(module.search_node(state(identify_result=identified())))

    assert len(result.search_results) == 1
    hit = result.search_results[0]
    assert "<" not in hit.summary and ">" not in hit.summary
    assert hit.summary == "조선의 법궁이다"
    assert hit.source == "wikipedia"
    assert hit.url.startswith("https://ko.wikipedia.org/wiki/")


# ───────────────────────── identify_node ─────────────────────────


def test_identify_rejects_empty_frame():
    from app.nodes.identify_node import identify_node

    result = run(identify_node(state(frame_b64="")))

    assert result.error is not None
    assert result.identify_result is None


def _stub_gemini(monkeypatch, text: str):
    from app.nodes import identify_node as module

    async def generate(**kwargs):
        return SimpleNamespace(text=text)

    client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    )
    monkeypatch.setattr(module, "get_gemini_client", lambda: client)
    return module


JSON_BODY = (
    '{"name":"경복궁","category":"landmark",'
    '"description":"조선의 법궁","search_query":"경복궁"}'
)


@pytest.mark.parametrize(
    "raw",
    [
        JSON_BODY,
        f"```json\n{JSON_BODY}\n```",
        f"```\n{JSON_BODY}\n```",
    ],
)
def test_identify_parses_with_and_without_code_fence(monkeypatch, raw):
    """모델이 코드블록으로 감싸 주든 아니든 같은 결과가 나와야 한다."""
    module = _stub_gemini(monkeypatch, raw)

    result = run(module.identify_node(state()))

    assert result.error is None, f"파싱 실패: {result.error}"
    assert result.identify_result.name == "경복궁"
    assert result.identify_result.category == "landmark"


def test_identify_records_reason_on_bad_json(monkeypatch):
    """형식이 어긋나면 사유를 남긴다. 조용히 넘어가면 빈 화면만 뜬다."""
    module = _stub_gemini(monkeypatch, "죄송합니다, 잘 모르겠어요")

    result = run(module.identify_node(state()))

    assert result.identify_result is None
    assert result.error is not None
    assert result.error.startswith("identify_node:")


def test_identify_records_reason_on_timeout(monkeypatch):
    from app.nodes import identify_node as module

    async def slow(**kwargs):
        await asyncio.sleep(10)

    client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=slow))
    )
    monkeypatch.setattr(module, "get_gemini_client", lambda: client)
    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "0.01")

    result = run(module.identify_node(state()))

    assert result.identify_result is None
    assert "timeout" in (result.error or "")
