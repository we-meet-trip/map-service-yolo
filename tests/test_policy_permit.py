"""Policy failures stop at the BFF permit; no real model or provider is called."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from app.api.permit import PermitClient, PermitError
from app.api.ws import vision_ws
from tests.fakes import FakePermit, FakeWebSocket, fake_app, jpeg_frame
from app.schemas.vision_schemas import IdentifyResult, VisionResponse


@pytest.mark.parametrize("body,code", [
    ({"code": "AI_CONSENT_REQUIRED"}, "AI_CONSENT_REQUIRED"),
    ({"code": "AI_CONSENT_CHANGED"}, "AI_CONSENT_CHANGED"),
    ({"code": "AGE_INFORMATION_REQUIRED"}, "AGE_INFORMATION_REQUIRED"),
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


@pytest.mark.parametrize("code", ["AGE_INFORMATION_REQUIRED", "AGE_RESTRICTED", "SERVICE_POLICY_REQUIRED"])
@pytest.mark.parametrize("image", [False, True])
def test_existing_socket_policy_revocation_blocks_inference(monkeypatch, code, image):
    app = fake_app()
    app.state.vision_permit.check = AsyncMock(side_effect=[SimpleNamespace(user_id=1, consent_revision=1, include_location=False), PermitError(403, code)])
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


@pytest.mark.parametrize("code", ["AGE_INFORMATION_REQUIRED", "AGE_RESTRICTED", "SERVICE_POLICY_REQUIRED"])
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


def _protected_result(req):
    return VisionResponse(session_id=req.session_id, request_id=req.request_id, status="done",
                          identify_result=IdentifyResult(name="SYNTHETIC_PROTECTED_RESULT", category="other",
                              description="SYNTHETIC_PROTECTED_RESULT", search_query="synthetic"))


@pytest.mark.parametrize("status,code,close", [
    (403, "AGE_INFORMATION_REQUIRED", 4403),
    (403, "AGE_RESTRICTED", 4403),
    (403, "SERVICE_POLICY_REQUIRED", 4403),
    (401, None, 4401),
    (503, None, 1013),
])
@pytest.mark.parametrize("image", [False, True])
def test_policy_change_while_inference_is_pending_withholds_result(monkeypatch, status, code, close, image):
    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()
        app = fake_app()
        permit = FakePermit(limit=1)
        app.state.vision_permit = permit
        original_check = permit.check
        changed = False

        async def check(token, *, consume, expected_revision=None):
            result = await original_check(token, consume=consume, expected_revision=expected_revision)
            if changed:
                raise PermitError(status, code)
            return result

        async def process(req, _):
            started.set()
            await release.wait()
            return _protected_result(req)

        permit.check = check
        process_mock = AsyncMock(side_effect=process)
        monkeypatch.setattr("app.api.ws._process", process_mock)
        socket = FakeWebSocket(app, [json.dumps({"session_id": "synthetic-pending", "request_id": "r1",
            "frame_b64": jpeg_frame() if image else "", "voice_triggered": image, "voice_text": "synthetic"})])
        running = asyncio.create_task(vision_ws(socket))
        try:
            await asyncio.wait_for(started.wait(), 1)
            changed = True
            release.set()
            await asyncio.wait_for(running, 1)
            assert socket.accepted and socket.close_code == close
            assert len(socket.sent) == 1 and "SYNTHETIC_PROTECTED_RESULT" not in socket.sent[0]
            reply = socket.responses()[0]
            assert reply["status"] == "failed" and reply["code"] == code
            assert reply["request_id"] == "r1" and reply["identify_result"] is None
            assert reply["search_results"] == []
            assert permit.calls == [False, True, False] and permit.used == 1
            process_mock.assert_awaited_once()
            # A denied response must release the runtime slot too.
            app.state.vision_runtime.reserve(image=image)
            app.state.vision_runtime.release(image=image)
        finally:
            release.set()
            if not running.done():
                running.cancel()
                await asyncio.gather(running, return_exceptions=True)
            app.state.vision_runtime.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("last_owner", [1, "1", 2])
def test_result_owner_must_match_original_authenticated_socket(monkeypatch, last_owner):
    app = fake_app()
    app.state.vision_permit.check = AsyncMock(side_effect=[SimpleNamespace(user_id=1, consent_revision=1, include_location=False),
        SimpleNamespace(user_id=1, consent_revision=1, include_location=False), SimpleNamespace(user_id=last_owner, consent_revision=1, include_location=False)])
    monkeypatch.setattr("app.api.ws._process", AsyncMock(side_effect=lambda req, _: _protected_result(req)))
    socket = FakeWebSocket(app, [json.dumps({"session_id": "synthetic-owner", "voice_text": "synthetic"})])
    asyncio.run(vision_ws(socket))
    assert [call.kwargs["consume"] for call in app.state.vision_permit.check.await_args_list] == [False, True, False]
    if str(last_owner) == "1":
        assert socket.responses()[0]["status"] == "done" and not socket.closed
        assert "SYNTHETIC_PROTECTED_RESULT" in socket.sent[0]
    else:
        assert socket.responses()[0]["status"] == "failed" and socket.close_code == 4401
        assert "SYNTHETIC_PROTECTED_RESULT" not in socket.sent[0]
    app.state.vision_runtime.close()


@pytest.mark.parametrize("post_status,code,close", [
    (200, None, None), (403, "AGE_INFORMATION_REQUIRED", 4403),
    (503, "SERVICE_POLICY_UNAVAILABLE", 1013),
])
def test_post_result_permit_uses_real_client_consume_false_wire_contract(monkeypatch, post_status, code, close):
    async def scenario():
        calls = []
        def transport(request):
            calls.append(json.loads(request.content)["consume"])
            if len(calls) == 3 and post_status != 200:
                return httpx.Response(post_status, json={"code": code, "message": "private upstream detail"})
            return httpx.Response(200, json={"user_id": 1, "remaining": 0,
                "reset_at": "2026-09-08T00:00:00Z", "consent_revision": 1, "include_location": False})
        app = fake_app()
        client = PermitClient("http://user.invalid", "internal-test", 1, transport=httpx.MockTransport(transport))
        app.state.vision_permit = client
        monkeypatch.setattr("app.api.ws._process", AsyncMock(side_effect=lambda req, _: _protected_result(req)))
        socket = FakeWebSocket(app, [json.dumps({"session_id": "synthetic-wire", "voice_text": "synthetic"})])
        try:
            await vision_ws(socket)
            assert calls == [False, True, False]
            assert socket.close_code == close
            assert "private upstream detail" not in socket.sent[0]
            if post_status == 200:
                assert socket.responses()[0]["status"] == "done"
                assert "SYNTHETIC_PROTECTED_RESULT" in socket.sent[0]
            else:
                assert socket.responses()[0]["status"] == "failed"
                assert socket.responses()[0]["code"] == (code if post_status == 403 else None)
                assert "SYNTHETIC_PROTECTED_RESULT" not in socket.sent[0]
        finally:
            await client.aclose()
            app.state.vision_runtime.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["before", "after"])
def test_revoke_then_regrant_never_resumes_original_socket_epoch(monkeypatch, phase):
    app = fake_app()
    def permission(revision):
        return SimpleNamespace(user_id=1, consent_revision=revision, include_location=False)
    app.state.vision_permit.check = AsyncMock(side_effect=[permission(1), permission(3 if phase == "before" else 1), permission(3)])
    process = AsyncMock(side_effect=lambda req, _: _protected_result(req))
    monkeypatch.setattr("app.api.ws._process", process)
    socket = FakeWebSocket(app, [json.dumps({"session_id": "synthetic-epoch", "voice_text": "synthetic"})])
    asyncio.run(vision_ws(socket))
    assert socket.close_code == 4403
    assert socket.responses()[0]["code"] == "AI_CONSENT_CHANGED"
    assert "SYNTHETIC_PROTECTED_RESULT" not in socket.sent[0]
    assert process.await_count == (0 if phase == "before" else 1)
    app.state.vision_runtime.close()


@pytest.mark.parametrize("include_location", [False, True])
def test_server_location_opt_in_controls_structured_location_before_provider(monkeypatch, include_location):
    app = fake_app()
    permission = SimpleNamespace(user_id=1, consent_revision=1, include_location=include_location)
    app.state.vision_permit.check = AsyncMock(return_value=permission)
    process = AsyncMock(side_effect=lambda req, _: _protected_result(req))
    monkeypatch.setattr("app.api.ws._process", process)
    socket = FakeWebSocket(app, [json.dumps({"session_id": "synthetic-location", "voice_text": "synthetic",
        "location": {"lat": 37.5, "lng": 127.0}})])
    asyncio.run(vision_ws(socket))
    assert (process.await_args.args[0].location is not None) == include_location
    assert [call.kwargs.get("expected_revision") for call in app.state.vision_permit.check.await_args_list] == [None, 1, 1]
    app.state.vision_runtime.close()


@pytest.mark.parametrize("fields", [{}, {"consent_revision": 0, "include_location": False},
    {"consent_revision": 1}, {"consent_revision": "1", "include_location": False}])
def test_legacy_or_malformed_permission_contract_fails_closed(fields):
    async def scenario():
        body = {"user_id": 1, "remaining": 0, "reset_at": "2026-09-08T00:00:00Z", **fields}
        client = PermitClient("http://user.invalid", "internal-test", 1, transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json=body)))
        try:
            with pytest.raises(PermitError) as denied:
                await client.check("a.b.c", consume=False)
            assert denied.value.status == 503
        finally:
            await client.aclose()
    asyncio.run(scenario())
