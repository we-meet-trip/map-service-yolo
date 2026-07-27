"""YOLO 추론 + 프레임 안정화 모듈.

역할:
  1) `YoloDetector`: YOLO 모델 로드 및 단일 프레임 추론.
  2) `FrameStabilizer`: 동일 객체가 YOLO_STABLE_FRAMES 연속 탐지될 때만
     에이전트 호출을 허용하는 안정화 필터.
     매 프레임 LLM 을 호출하면 비용·지연이 폭발하므로 이 단계에서 제어.
"""
from __future__ import annotations

import base64
import logging
from collections import Counter
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from app.schemas.vision_schemas import DetectedObject

logger = logging.getLogger(__name__)


class YoloDetector:
    """YOLO 모델 래퍼.

    model_path: .pt 파일 경로.
    confidence: 탐지 신뢰도 임계값.

    사용처: lifespan 에서 1회 생성 후 `app.state.detector` 에 저장.
    """

    def __init__(self, model_path: str, confidence: float) -> None:
        self._model = YOLO(model_path)
        self._confidence = confidence
        logger.info("YoloDetector loaded model=%s conf=%.2f", model_path, confidence)

    def detect(self, frame_b64: str) -> list[DetectedObject]:
        """Base64 JPEG 프레임에서 객체를 탐지해 `DetectedObject` 리스트를 반환.

        신뢰도 임계값 이하 탐지는 필터링한다.
        프레임 디코딩 실패 시 빈 리스트 반환.
        """
        try:
            img_bytes = base64.b64decode(frame_b64)
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                logger.warning("YoloDetector: frame decode failed")
                return []
        except Exception:
            logger.exception("YoloDetector: frame decode error")
            return []

        results = self._model(frame, conf=self._confidence, verbose=False)
        detected: list[DetectedObject] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                label = self._model.names[int(box.cls[0])]
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].tolist()
                detected.append(
                    DetectedObject(label=label, confidence=confidence, bbox=bbox)
                )
        return detected


class FrameStabilizer:
    """연속 프레임 안정화 필터.

    동일 레이블이 `stable_frames` 번 연속으로 탐지되어야
    `get_stable_object()` 가 해당 객체를 반환한다.
    조건 미달이면 None 반환 → 에이전트 미호출.

    _counter: 최근 탐지 레이블 카운터.
    _stable_frames: 안정화 임계 프레임 수.
    """

    def __init__(self, stable_frames: int) -> None:
        self._stable_frames = stable_frames
        self._history: list[str] = []

    def update(self, detected: list[DetectedObject]) -> Optional[DetectedObject]:
        """탐지 결과로 히스토리를 갱신하고, 안정화된 객체가 있으면 반환.

        탐지된 객체 중 가장 신뢰도 높은 것을 대표로 사용.
        탐지 결과가 없으면 히스토리를 초기화한다.
        """
        if not detected:
            self._history.clear()
            return None

        best = max(detected, key=lambda d: d.confidence)
        self._history.append(best.label)

        # 윈도우를 stable_frames 로 제한
        if len(self._history) > self._stable_frames:
            self._history.pop(0)

        # 윈도우가 가득 찼고 모두 같은 레이블이면 안정화 판정
        if len(self._history) == self._stable_frames:
            counter = Counter(self._history)
            top_label, count = counter.most_common(1)[0]
            if count == self._stable_frames:
                self._history.clear()  # 안정화 후 초기화 → 중복 호출 방지
                # best 중 top_label 과 일치하는 것 반환
                for d in detected:
                    if d.label == top_label:
                        return d
        return None
