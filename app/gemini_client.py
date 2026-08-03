"""Gemini 클라이언트 싱글톤 접근자.

`genai.Client` 는 내부에 HTTP 커넥션 풀을 들고 있어 요청마다 새로 만들면
연결 재사용이 되지 않는다. 프로세스 단위로 하나만 만들어 공유한다.
"""
from __future__ import annotations

from functools import lru_cache

from google import genai

from app.agent_settings import get_settings


@lru_cache(maxsize=1)
def get_gemini_client() -> genai.Client:
    """프로세스 단위로 캐시되는 Gemini 클라이언트."""
    settings = get_settings()
    return genai.Client(api_key=settings.GEMINI_API_KEY.get_secret_value())
