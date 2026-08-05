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
      YOLO_MODEL_PATH: .pt 모델 파일 경로. 작업 디렉토리 기준 상대 경로.
      YOLO_CONFIDENCE: 탐지 신뢰도 임계값(0.0~1.0).

    서비스 관련:
      JOB_TIMEOUT_SECONDS: 에이전트 1회 실행 한도(초).
        식별 + 검색을 모두 담아야 하므로 GEMINI_TIMEOUT_SECONDS 보다 커야 한다.
        작으면 식별을 마치고도 검색 도중 잘려 결과가 통째로 버려진다.
      WS_MAX_JOBS_PER_CONNECTION: 연결 하나가 돌릴 수 있는 인식 횟수 상한.
        인식 1건마다 외부 모델을 부르므로, 상한이 없으면 연결 하나로 호출
        한도를 소진시켜 같은 키를 쓰는 다른 기능까지 멈춘다.

    관측(로깅):
      LOG_LEVEL: 루트 로거 레벨. uvicorn 은 자기 로거만 구성하고 루트 로거에는
        핸들러를 붙이지 않는다. app/main.py 의 _configure_logging 이 부팅 시
        루트 핸들러가 비어 있을 때만 stdout 핸들러를 붙여 app.* 로그를 살린다.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    GEMINI_API_KEY: SecretStr = SecretStr("")
    GEMINI_MODEL: str = "gemini-2.5-flash"
    GEMINI_TIMEOUT_SECONDS: float = 25.0
    KAKAO_REST_API_KEY: str = ""

    YOLO_MODEL_PATH: str = "models/yolo11n.pt"
    YOLO_CONFIDENCE: float = 0.5

    JOB_TIMEOUT_SECONDS: float = 45.0
    WS_MAX_JOBS_PER_CONNECTION: int = 60

    LOG_LEVEL: str = "INFO"


_settings: AgentSettings | None = None


def get_settings() -> AgentSettings:
    """프로세스 단위 싱글톤 AgentSettings 접근자."""
    global _settings
    if _settings is None:
        _settings = AgentSettings()
    return _settings
