"""yolo-vision-agent 환경설정.

`AgentSettings(BaseSettings)` 는 환경 변수(또는 `.env`) 로부터 값을
읽어 들이는 설정 객체이고, `get_settings()` 는 프로세스 1회 캐시되는
싱글톤 접근자다.
"""
from __future__ import annotations

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    """yolo-vision-agent 의 모든 환경 변수 묶음.

    model_config:
      - env_file=".env": 작업 디렉토리의 .env 파일 자동 로드.
      - extra="ignore": 정의되지 않은 환경 변수는 무시.

    Gemini 관련:
      GEMINI_API_KEY: Gemini Vision 호출용 API 키. 비밀값(SecretStr).
      GEMINI_MODEL: 사용할 Gemini 모델명.
      GEMINI_TIMEOUT_SECONDS: Gemini 호출 1건의 타임아웃(초).

    YOLO 관련:
      YOLO_MODEL_PATH: .pt 모델 파일 경로. models/ 디렉토리 기준.
      YOLO_CONFIDENCE: 탐지 신뢰도 임계값(0.0~1.0).
      YOLO_STABLE_FRAMES: 동일 객체가 연속 N프레임 탐지될 때 에이전트 호출.

    서비스 관련:
      JOB_TIMEOUT_SECONDS: 에이전트 1회 실행 한도(초).
      SHUTDOWN_GRACE_SECONDS: lifespan 종료 시 대기 한도(초).
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    GEMINI_API_KEY: SecretStr = SecretStr("")
    GEMINI_MODEL: str = "gemini-2.0-flash"
    GEMINI_TIMEOUT_SECONDS: float = 30.0
    KAKAO_REST_API_KEY: str = ""

    YOLO_MODEL_PATH: str = "models/yolo11n.pt"
    YOLO_CONFIDENCE: float = 0.5
    YOLO_STABLE_FRAMES: int = 5

    JOB_TIMEOUT_SECONDS: float = 30.0
    SHUTDOWN_GRACE_SECONDS: float = 40.0


_settings: AgentSettings | None = None


def get_settings() -> AgentSettings:
    """프로세스 단위 싱글톤 AgentSettings 접근자."""
    global _settings
    if _settings is None:
        _settings = AgentSettings()
    return _settings
