"""Policy failures stop at the BFF permit; no real model or provider is called."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from app.api.permit import PermitClient, PermitError
from app.api.ws import vision_ws
from tests.fakes import FakeWebSocket, fake_app, jpeg_frame


@pytest.mark.parametrize("body,code", [
    ({"code": "AGE_RESTRICTED", "message": "private upstream detail"}, "AGE_RESTRICTED"),
    ({"code": "SERVICE_POLICY_REQUIRED"}, "SERVICE_POLICY_REQUIRED"),
    ({"code": "untrusted detail", "message": "private upstream detail"}, None),
    ({"code": []}, None),
    ([], None),
])
def test_only_known_policy_codes_survive_403(body, code):
    async def scenario():
        client = PermitClient("http://user.invalid", "internal-test", 1,
                              transport=httpx.MockTransport(lambda _: httpx.Response(403, json=body)))
        try:
            with pytest.raises(PermitError) as error:
                await client.check("test.jwt.signature", consume=True)
            assert error.value.status == 403 and error.value.code == code
            assert str(error.value) == "vision permit denied"
        finally:
            await client.aclose()
    asyncio.run(scenario())


@pytest.mark.parametrize("status", [403, 500, 502])
def test_malformed_provider_error_body_is_not_forwarded(status):
    async def scenario():
        client = PermitClient("http://user.invalid", "internal-test", 1, transport=httpx.MockTransport(
            lambda _: httpx.Response(status, text="private upstream detail")))
        try:
            with pytest.raises(PermitError) as error:
                await client.check("test.jwt.signature", consume=True)
            assert error.value.code is None
            assert error.value.status == (403 if status == 403 else 503)
            assert "private" not in str(error.value)
        finally:
            await client.aclose()
    asyncio.run(scenario())


def test_transport_failure_has_fixed_safe_status_and_message():
    def unavailable(request):
        raise httpx.ConnectError("private upstream detail", request=request)
    async def scenario():
        client = PermitClient("http://user.invalid", "internal-test", 1,
                              transport=httpx.MockTransport(unavailable))
        try:
            with pytest.raises(PermitError) as error:
                await client.check("test.jwt.signature", consume=True)
            assert error.value.status == 503 and error.value.code is None
            assert str(error.value) == "vision permit denied"
        finally:
            await client.aclose()
    asyncio.run(scenario())


@pytest.mark.parametrize("code", ["AGE_RESTRICTED", "SERVICE_POLICY_REQUIRED"])
@pytest.mark.parametrize("image", [False, True])
def test_existing_socket_policy_revocation_blocks_inference(monkeypatch, code, image):
    app = fake_app()
    app.state.vision_permit.check = AsyncMock(side_effect=[SimpleNamespace(user_id=1), PermitError(403, code)])
    process = AsyncMock(side_effect=AssertionError("must not invoke inference"))
    monkeypatch.setattr("app.api.ws._process", process)
    socket = FakeWebSocket(app, [json.dumps({"session_id": "synthetic-request", "request_id": "r1",
                                           "frame_b64": jpeg_frame() if image else "", "voice_triggered": image,
                                           "voice_text": "synthetic question"})])
    asyncio.run(vision_ws(socket))
    assert socket.accepted and socket.close_code == 4403
    reply = socket.responses()[-1]
    assert reply["code"] == code and reply["status"] == "failed"
    assert reply["request_id"] == "r1"
    process.assert_not_awaited()
    assert app.state.detector.calls == 0 and app.state.graph.calls == 0
    app.state.vision_runtime.close()


@pytest.mark.parametrize("code", ["AGE_RESTRICTED", "SERVICE_POLICY_REQUIRED"])
def test_policy_denied_handshake_never_accepts_or_reads_a_frame(monkeypatch, code):
    app = fake_app()
    app.state.vision_permit.check = AsyncMock(side_effect=PermitError(403, code))
    process = AsyncMock()
    monkeypatch.setattr("app.api.ws._process", process)
    socket = FakeWebSocket(app, ["must remain unread"])
    asyncio.run(vision_ws(socket))
    assert not socket.accepted and socket.close_code == 4403 and not socket.sent
    assert socket._incoming == ["must remain unread"]
    process.assert_not_awaited()
    app.state.vision_runtime.close()
