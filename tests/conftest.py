"""pytest 공용 fixture / 환경 부트스트랩.

`AgentSettings` 는 프로세스 1회 캐시되는 싱글턴이라, 테스트마다 다른 상한을
쓰려면 캐시 슬롯을 비워 줘야 한다. 여기서 그 정리를 맡는다.

호출 관계:
  - 각 테스트가 `from app...` 임포트를 만나기 전에 본 파일이 먼저 평가된다.
"""
from __future__ import annotations

import asyncio
import os

import pytest

# 실제 호출은 일어나지 않지만, 키가 비어 있으면 텍스트 대화가 카카오 경로로
# 가지 않고 곧장 모델 경로로 빠진다. 두 갈래를 모두 테스트하려면 값이 필요하다.
os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("KAKAO_REST_API_KEY", "test-kakao-key")


@pytest.fixture(autouse=True)
def reset_settings_cache():
    """설정 싱글턴을 테스트마다 비운다. 환경변수 변경이 실제로 반영되게 한다."""
    import app.agent_settings as agent_settings

    agent_settings._settings = None
    yield
    agent_settings._settings = None


def run(coro):
    """코루틴을 동기 테스트에서 실행한다(hub 레포와 같은 방식)."""
    return asyncio.run(coro)
