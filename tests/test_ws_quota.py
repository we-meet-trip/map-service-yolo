"""연결당 처리 상한이 무엇을 세는지 고정하는 테스트.

이 상한은 자체 인증이 없는 엔드포인트에서 외부 모델 호출이 새어 나가는 것을 막으려고
둔 것이다. 그런데 읽어 들인 프레임 수를 세면, 형식이 깨진 요청이나 아무 일도 하지 않고
넘기는 프레임까지 상한을 깎아 **외부를 한 번도 부르지 않은 연결이 끊긴다.** 남용을
막자는 장치가 정상 사용자를 먼저 끊는 셈이라, 세는 자리를 호출 직전으로 옮겼다.

여기서 못 박는 것:
  - 무시되는 프레임과 형식 오류는 상한을 쓰지 않는다.
  - 외부를 부르는 처리(텍스트 대화 · 이미지 인식)만 상한을 쓴다.
  - 상한을 다 쓰면 사유를 알리고 연결을 닫는다.
"""
from __future__ import annotations

import asyncio
import json
import os

import pytest

from tests.fakes import (
    FakeDetector,
    FakeGeminiClient,
    FakeGraph,
    FakeWebSocket,
    fake_app,
    jpeg_frame,
)


def run(coro):
    return asyncio.run(coro)


def frame(**kwargs) -> str:
    body = {"session_id": "s1", "frame_b64": "", "voice_triggered": False}
    body.update(kwargs)
    return json.dumps(body)


@pytest.fixture
def limit_two(monkeypatch):
    """상한을 2 로 낮춰 경계를 짧게 만든다."""
    monkeypatch.setenv("WS_MAX_JOBS_PER_CONNECTION", "2")


@pytest.fixture
def stub_gemini(monkeypatch):
    """모델 호출을 대역으로 바꾼다. 실제 외부 호출은 일어나지 않는다."""
    from app.api import ws as ws_module

    client = FakeGeminiClient()
    monkeypatch.setattr(ws_module, "get_gemini_client", lambda: client)
    # 장소 질의로 새지 않도록 카카오 경로를 끈다(모델 경로만 보려는 것이다).
    monkeypatch.setattr(ws_module, "is_local_query", lambda _text: False)
    return client


def test_ignored_frames_do_not_consume_quota(limit_two, stub_gemini):
    """음성 트리거도 텍스트도 없는 프레임은 아무 일도 하지 않으므로 상한을 쓰지 않는다."""
    from app.api.ws import vision_ws

    app = fake_app()
    socket = FakeWebSocket(app, [frame() for _ in range(20)])

    run(vision_ws(socket))

    assert socket.closed is False, "외부를 한 번도 부르지 않았는데 연결이 끊겼다"
    assert socket.sent == [], "무시할 프레임에 응답을 보내면 안 된다"
    assert stub_gemini.calls == 0


def test_malformed_frames_do_not_consume_quota(limit_two, stub_gemini):
    """형식이 깨진 요청은 사유만 돌려주고 상한은 쓰지 않는다."""
    from app.api.ws import vision_ws

    app = fake_app()
    socket = FakeWebSocket(app, ["{ not json" for _ in range(10)])

    run(vision_ws(socket))

    assert socket.closed is False
    assert len(socket.errors()) == 10
    assert all("parse error" in e for e in socket.errors())


def test_text_conversation_consumes_quota(limit_two, stub_gemini):
    """텍스트 대화는 모델을 부르므로 상한을 쓴다. 다 쓰면 사유를 알리고 닫는다."""
    from app.api.ws import vision_ws

    app = fake_app()
    socket = FakeWebSocket(app, [frame(voice_text=f"질문 {i}") for i in range(5)])

    run(vision_ws(socket))

    assert stub_gemini.calls == 2, "상한 2 를 넘겨 모델을 부르면 안 된다"
    assert socket.closed is True
    last = socket.responses()[-1]
    assert last["status"] == "failed"
    assert "횟수" in last["error"]
    assert last["session_id"] == "s1", "어느 세션이 막혔는지 알려 줘야 한다"


def test_image_recognition_consumes_quota(limit_two, stub_gemini):
    """이미지 인식도 외부를 부르므로 같은 상한을 쓴다."""
    from app.api.ws import vision_ws

    detector = FakeDetector()
    graph = FakeGraph()
    app = fake_app(detector=detector, graph=graph)
    socket = FakeWebSocket(
        app,
        [frame(voice_triggered=True, frame_b64=jpeg_frame()) for _ in range(5)],
    )

    run(vision_ws(socket))

    assert detector.calls == 2
    assert graph.calls == 2
    assert socket.closed is True


def test_mixed_traffic_only_counts_external_calls(limit_two, stub_gemini):
    """무시·오류 프레임이 섞여 있어도 상한은 외부를 부른 횟수만 따라간다."""
    from app.api.ws import vision_ws

    app = fake_app()
    incoming = [
        frame(),                       # 무시
        "{ broken",                    # 오류
        frame(voice_text="첫 질문"),     # 1회차
        frame(),                       # 무시
        "{ broken",                    # 오류
        frame(voice_text="둘째 질문"),   # 2회차
        frame(),                       # 무시
    ]
    socket = FakeWebSocket(app, incoming)

    run(vision_ws(socket))

    assert stub_gemini.calls == 2
    assert socket.closed is False, "상한을 정확히 2 회 썼으므로 아직 닫히지 않는다"


def test_quota_default_matches_setting(monkeypatch, stub_gemini):
    """설정값이 그대로 상한이 된다(하드코딩된 숫자가 끼어들지 않는다)."""
    from app.api.ws import vision_ws

    monkeypatch.setenv("WS_MAX_JOBS_PER_CONNECTION", "3")
    app = fake_app()
    socket = FakeWebSocket(app, [frame(voice_text="q") for _ in range(10)])

    run(vision_ws(socket))

    assert stub_gemini.calls == 3
