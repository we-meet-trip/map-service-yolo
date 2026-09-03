"""좌표가 로그에 남지 않는 것을 고정하는 테스트.

기기 위치는 요청 처리에만 쓰고 로그에는 남기지 않는다는 규칙을 세 겹으로 확인한다.

  - 위치가 든 장소 검색을 실제 핸들러로 돌려도 로그 전체에 좌표 값이 없다.
  - 대조군: 고치기 전 형태의 문장을 만들면 같은 탐지 패턴에 걸린다.
    즉 가림을 빼면 이 테스트가 빨갛게 되는 구조임을 증명한다.
  - 가림 필터 자체: lat/lng 계열과 카카오 쿼리 x/y 는 지워지고,
    좌표가 아닌 숫자(max, index 등)는 건드리지 않는다.
"""
from __future__ import annotations

import logging
import re

from app.log_redaction import CoordinateRedactingFilter
from tests.conftest import run
from tests.fakes import FakeGeminiClient, FakeWebSocket, fake_app

# 좌표 누출 탐지 패턴: lat/lng 계열 이름 또는 쿼리 구분자 뒤 단독 x/y 에
# 숫자 값이 붙어 있으면 누출로 본다.
_LEAK = re.compile(r"(?:lat|lng|[?&][xy])=-?\d")

_LAT, _LNG = 37.123456, 127.654321


def _search_message() -> str:
    """위치가 든 장소 검색 프레임(텍스트 대화 분기, 카카오 경로 진입)."""
    import json

    return json.dumps({
        "session_id": "s1",
        "frame_b64": "",
        "voice_triggered": False,
        "voice_text": "근처 카페 찾아줘",
        "location": {"lat": _LAT, "lng": _LNG},
    })


async def _fake_kakao_search(query, api_key, lat=None, lng=None, **kwargs):
    """카카오 검색 대역. 실제 HTTP 호출 없이 결과 한 건을 돌려준다."""
    return [{
        "name": "테스트카페", "address": "테스트로 1", "phone": "",
        "category": "카페", "distance": "120",
    }]


def test_search_with_location_leaves_no_coordinates_in_logs(monkeypatch, caplog):
    """위치가 든 검색을 처리해도 로그 어디에도 좌표 값이 남지 않는다."""
    from app.api import ws as ws_module
    from app.api.ws import vision_ws

    monkeypatch.setattr(ws_module, "get_gemini_client", lambda: FakeGeminiClient())
    monkeypatch.setattr(ws_module, "kakao_local_search", _fake_kakao_search)

    app = fake_app()
    socket = FakeWebSocket(app, [_search_message()])

    with caplog.at_level(logging.INFO):
        run(vision_ws(socket))

    # 검색 로그 자체는 남아야 한다 — 아무것도 안 찍혀 통과하는 헛시험 방지.
    assert "kakao_local_search" in caplog.text
    assert socket.responses()[-1]["status"] == "done"
    assert _LEAK.search(caplog.text) is None, "로그에 좌표 값이 남았다"


def test_prefilter_render_is_caught_by_leak_pattern():
    """대조군: 고치기 전 형태의 문장은 탐지 패턴에 걸린다.

    위 테스트의 '패턴 부재' 단언이 실제로 누출을 잡을 수 있는 패턴임을,
    두 누출 형태(직접 로그·httpx 요청 URL)로 증명한다.
    """
    old_direct = "kakao_local_search keyword=%s lat=%s lng=%s" % ("카페", _LAT, _LNG)
    httpx_style = (
        f'HTTP Request: GET https://dapi.kakao.com/v2/local/search/keyword.json'
        f'?query=카페&y={_LAT}&x={_LNG} "HTTP/1.1 200 OK"'
    )
    assert _LEAK.search(old_direct)
    assert _LEAK.search(httpx_style)


def test_filter_scrubs_coordinate_values():
    """lat/lng 계열과 쿼리 x/y 값은 자리표시자로 바뀐다."""
    scrub = CoordinateRedactingFilter._scrub

    assert scrub("lat=37.123 lng=127.456") == "lat=*** lng=***"
    assert scrub("start_lat=37.1 end_longitude=127.2") == "start_lat=*** end_longitude=***"
    assert (
        scrub("GET https://dapi.kakao.com/v2/local/geo/coord2address.json?x=127.1&y=37.5")
        == "GET https://dapi.kakao.com/v2/local/geo/coord2address.json?x=***&y=***"
    )


def test_filter_keeps_non_coordinate_numbers():
    """좌표가 아닌 숫자 값은 건드리지 않는다."""
    scrub = CoordinateRedactingFilter._scrub

    for text in ("max=300", "index=2", "?query=카페&size=5&radius=2000", "delay=5"):
        assert scrub(text) == text


def test_httpx_logger_records_are_scrubbed(caplog):
    """httpx 로거로 나가는 요청 로그는 길목에서 가려진다.

    실제 httpx 는 주소를 문자열이 아니라 URL 객체 인자로 넘기므로,
    문자열 아닌 인자로 흉내 내어 문장 조립 경로까지 함께 확인한다.
    """

    class _UrlLike:
        def __str__(self) -> str:
            return "https://dapi.kakao.com/v2/local/search/keyword.json?query=카페&y=37.123456&x=127.654321"

    with caplog.at_level(logging.INFO, logger="httpx"):
        logging.getLogger("httpx").info(
            'HTTP Request: %s %s "%s"', "GET", _UrlLike(), "HTTP/1.1 200 OK"
        )

    assert "***" in caplog.text, "가림이 아예 동작하지 않았다"
    assert _LEAK.search(caplog.text) is None, "httpx 로그에 좌표 값이 남았다"
