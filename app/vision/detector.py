"""YOLO 추론 모듈.

`YoloDetector` 가 YOLO 모델을 로드하고 단일 프레임을 추론한다.

클라이언트는 음성 트리거 시점에 프레임 한 장만 보내므로(연속 스트림이 아니다)
여러 프레임을 모아 판정하는 안정화 필터는 두지 않는다. 트리거당 1회 추론이
곧 1회 LLM 호출이라 호출량은 사용자의 발화 횟수로 제한된다.
"""
from __future__ import annotations

import base64
import logging

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
