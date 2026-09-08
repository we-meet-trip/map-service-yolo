"""JWT 검증 및 계정별 quota의 소유자인 User BFF에 매 처리 허가를 요청한다."""
from datetime import datetime

import httpx
from pydantic import BaseModel, Field, ValidationError


class PermitError(Exception):
    def __init__(self, status: int, code: str | None = None):
        self.status = status
        self.code = code if status == 403 and code in ("AGE_INFORMATION_REQUIRED", "AGE_RESTRICTED", "SERVICE_POLICY_REQUIRED", "AI_CONSENT_REQUIRED", "AI_CONSENT_CHANGED") else None
        super().__init__("vision permit denied")


class Permit(BaseModel):
    user_id: str | int
    remaining: int = Field(ge=0)
    reset_at: datetime
    consent_revision: int = Field(gt=0, strict=True)
    include_location: bool = Field(strict=True)


class PermitClient:
    def __init__(self, base_url: str, internal_token: str, timeout: float,
                 *, transport=None):
        if not internal_token:
            raise ValueError("VISION_INTERNAL_TOKEN is required")
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=timeout, follow_redirects=False,
            headers={"X-Internal-Token": internal_token}, transport=transport,
        )

    async def check(self, bearer: str, *, consume: bool, expected_revision: int | None = None) -> Permit:
        try:
            payload = {"consume": consume}
            if expected_revision is not None:
                payload["expected_revision"] = expected_revision
            response = await self._client.post(
                "/api/v1/vision/permit", headers={"Authorization": f"Bearer {bearer}"},
                json=payload,
            )
            if response.status_code != 200:
                code = None
                if response.status_code == 403:
                    try:
                        body = response.json()
                        candidate = body.get("code") if isinstance(body, dict) else None
                        if isinstance(candidate, str):
                            code = candidate
                    except ValueError:
                        pass
                raise PermitError(response.status_code if response.status_code in {401, 403, 429} else 503, code)
            permit = Permit.model_validate(response.json())
            if not str(permit.user_id).strip() or permit.reset_at.tzinfo is None:
                raise ValueError("invalid permit")
            return permit
        except (httpx.HTTPError, ValidationError, ValueError) as exc:
            raise PermitError(503) from exc

    async def aclose(self):
        await self._client.aclose()
