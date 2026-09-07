"""yolo-vision-agent 요청·응답·상태 Pydantic 스키마."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

FRAME_MAX_BYTES = 2 * 1024 * 1024
FRAME_MAX_PIXELS = 2048 * 2048
FRAME_MAX_BASE64 = 4 * ((FRAME_MAX_BYTES + 2) // 3)
WS_MAX_BYTES = 4 * 1024 * 1024


class Location(BaseModel):
    """사용자 GPS 위치.

    lat: 위도.
    lng: 경도.
    accuracy_meters: 위치 정확도(미터). 없을 수 있음.
    """
    model_config = ConfigDict(allow_inf_nan=False)
    lat: float = Field(ge=33, le=43)
    lng: float = Field(ge=124, le=132)
    accuracy_meters: Optional[float] = Field(default=None, ge=0, le=100000)


class DetectedObject(BaseModel):
    """YOLO 가 탐지한 객체 1건.

    label: YOLO 클래스 레이블(예: "building", "food").
    confidence: 신뢰도(0.0~1.0).
    bbox: 바운딩 박스 [x1, y1, x2, y2] (픽셀 좌표).
    """
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: List[float] = Field(min_length=4, max_length=4)


class VisionRequest(BaseModel):
    """WebSocket 으로 수신하는 프레임 + 위치 페이로드.

    frame_b64: Base64 인코딩된 JPEG 프레임.
    location: 사용자 현재 위치. 선택값.
    session_id: 클라이언트 세션 식별자.
    voice_triggered: 음성 발화 시작 트리거 여부.
    """
    frame_b64: str = Field(default="", max_length=FRAME_MAX_BASE64)
    location: Optional[Location] = None
    session_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    request_id: Optional[str] = Field(default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    voice_triggered: bool = False
    voice_text: Optional[str] = Field(default=None, max_length=2000)
    prior_context: Optional[str] = Field(default=None, max_length=2000)
    conversation_history: list[dict] = Field(default_factory=list, max_length=8)

    @field_validator("conversation_history")
    @classmethod
    def validate_history(cls, messages):
        for message in messages:
            if (set(message) != {"role", "content"} or not isinstance(message["role"], str)
                    or message["role"] not in {"user", "assistant"}):
                raise ValueError("invalid conversation role")
            if not isinstance(message["content"], str) or len(message["content"]) > 2000:
                raise ValueError("invalid conversation content")
        return messages


class IdentifyResult(BaseModel):
    """Gemini Vision 이 반환하는 객체 식별 결과.

    name: 식별된 정확한 이름(예: "경복궁", "비빔밥").
    category: 대분류(예: "landmark", "food", "product", "animal", "plant", "other").
    description: 간단한 설명.
    search_query: 검색 노드에서 쓸 최적화된 쿼리.
    """
    name: str
    category: Literal["landmark", "food", "product", "animal", "plant", "other"]
    description: str
    search_query: str


class SearchResult(BaseModel):
    """검색 노드가 반환하는 정보 1건.

    title: 검색 결과 제목.
    summary: 요약 정보.
    source: 정보 출처(예: "wikipedia", "kakao").
    url: 원문 링크. 없을 수 있음.
    """
    title: str
    summary: str
    source: str
    url: Optional[str] = None


class VisionResponse(BaseModel):
    """WebSocket 으로 클라이언트에 반환하는 최종 응답.

    session_id: 요청 세션 식별자.
    detected_object: YOLO 가 탐지한 대표 객체.
    identify_result: Gemini 가 식별한 상세 정보.
    search_results: 검색 결과 리스트.
    status: 처리 결과 상태.
    error: 실패 시 사유.
    """
    session_id: str
    request_id: Optional[str] = None
    detected_object: Optional[DetectedObject] = None
    identify_result: Optional[IdentifyResult] = None
    search_results: List[SearchResult] = Field(default_factory=list)
    status: Literal["done", "failed"] = "done"
    code: Optional[Literal["AGE_RESTRICTED", "SERVICE_POLICY_REQUIRED"]] = None
    error: Optional[str] = None


class GraphState(BaseModel):
    """LangGraph 파이프라인 상태.

    session_id: 클라이언트 세션 식별자.
    location: 사용자 GPS 위치.
    frame_b64: Base64 인코딩된 JPEG 프레임.
    detected_object: YOLO 탐지 결과(안정화된 대표 객체).
    identify_result: Gemini Vision 식별 결과.
    search_results: 검색 노드 결과.
    error: 오류 발생 시 사유.
    """
    session_id: str
    location: Optional[Location] = None
    frame_b64: str
    voice_text: Optional[str] = None
    detected_object: Optional[DetectedObject] = None
    identify_result: Optional[IdentifyResult] = None
    search_results: List[SearchResult] = Field(default_factory=list)
    error: Optional[str] = None
