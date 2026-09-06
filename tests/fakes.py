"""테스트용 대역 객체.

실제 소켓·모델·외부 API 없이 WebSocket 핸들러의 흐름만 돌리기 위한 최소 구현이다.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from fastapi import WebSocketDisconnect
from app.vision.runtime import VisionRuntime


class FakeWebSocket:
    """미리 준비한 프레임을 순서대로 내주고, 보낸 것을 모아 두는 소켓 대역.

    준비한 프레임이 떨어지면 WebSocketDisconnect 를 던져 실제 종료를 흉내 낸다.
    """

    def __init__(self, app, incoming: list[str], protocols=None):
        self.app = app
        self._incoming = list(incoming)
        self.sent: list[str] = []
        self.closed = False
        self.accepted = False
        self.scope = {"subprotocols": protocols if protocols is not None else ["map.vision.v1", "bearer.test.jwt.signature"]}
        self.close_code = None

    async def accept(self, subprotocol=None) -> None:
        self.accepted = True
        self.subprotocol = subprotocol

    async def receive_text(self) -> str:
        if self.closed or not self._incoming:
            raise WebSocketDisconnect(code=1000)
        return self._incoming.pop(0)

    async def send_text(self, text: str) -> None:
        self.sent.append(text)

    async def close(self, code=1000) -> None:
        self.closed = True
        self.close_code = code

    # ---- 검증 도우미 ----

    def responses(self) -> list[dict]:
        return [json.loads(t) for t in self.sent]

    def errors(self) -> list[str]:
        return [r.get("error") or "" for r in self.responses()
                if r.get("status") == "failed"]


class FakeDetector:
    """호출 횟수만 세는 YOLO 대역. 탐지 결과는 항상 비어 있다."""

    def __init__(self):
        self.calls = 0

    def detect(self, frame_b64: str):
        self.calls += 1
        return []


class FakeGraph:
    """상태를 그대로 돌려주는 그래프 대역. 호출 횟수를 센다."""

    def __init__(self, result=None):
        self.calls = 0
        self._result = result

    async def ainvoke(self, state):
        self.calls += 1
        return self._result if self._result is not None else state


def fake_app(detector=None, graph=None, timeout: float = 5.0):
    """lifespan 이 주입하는 app.state 만 흉내 낸 객체."""
    return SimpleNamespace(
        state=SimpleNamespace(
            detector=detector or FakeDetector(),
            graph=graph or FakeGraph(),
            job_timeout_seconds=timeout,
            vision_permit=FakePermit(),
            vision_runtime=VisionRuntime(2),
        )
    )


class FakePermit:
    def __init__(self, limit=60):
        self.used = 0
        self.limit = limit
        self.calls = []

    async def check(self, token, *, consume):
        from app.api.permit import PermitError
        self.calls.append(consume)
        if consume:
            if self.used >= self.limit:
                raise PermitError(429)
            self.used += 1
        return SimpleNamespace(user_id=1, remaining=self.limit-self.used)


def jpeg_frame():
    import base64
    from io import BytesIO
    from PIL import Image
    buffer = BytesIO()
    Image.new("RGB", (2, 2)).save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()


class FakeGeminiClient:
    """generate_content 호출 횟수를 세고 고정 텍스트를 돌려주는 모델 대역."""

    def __init__(self, text: str = "테스트 답변"):
        self.calls = 0
        self._text = text
        self.aio = SimpleNamespace(models=SimpleNamespace(
            generate_content=self._generate
        ))

    async def _generate(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(text=self._text)
