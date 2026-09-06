# MAP Vision 서비스

`/ws/vision`은 `map.vision.v1`, `bearer.<사용자 JWT>` WebSocket subprotocol을
함께 받아 인증한다. 서버는 `map.vision.v1`만 협상 응답에 싣는다.
User BFF의 `POST /api/v1/vision/permit`을 연결 시 `{consume:false}`,
매 처리 직전에 `{consume:true}`로 호출한다. JWT와 `VISION_INTERNAL_TOKEN`을
함께 보내며 만료·탈퇴·계정 quota 초과·BFF 장애는 모두 외부 추론 전에 막는다.
계정 quota는 User Redis가 소유하므로 재접속으로 초기화되지 않는다.

요청은 `session_id`(상관 ID), 선택적 `request_id`, `voice_triggered`, `voice_text`,
`frame_b64`, 선택적 `location`, `prior_context`, `conversation_history`를 사용한다.
응답은 두 ID를 그대로 돌려준다. 현재 앱은 매 요청에 고유 `session_id`를 만든다.

- 이미지: 유효한 JPEG만 허용, 디코딩 전 최대 2MiB/4MP 검사.
- 위치: 한국 범위와 유한한 숫자만 허용한다.
- 텍스트/이전 맥락: 각 2,000자. 대화 기록: user/assistant 최대 8개, 각 2,000자.
- WebSocket 메시지: 4MiB, 전송 대기열 4개. 연결당 메시지/처리/유휴시간 상한 적용.
- 프로세스 전체 동시 작업은 기본 2개, 공유 YOLO 모델 추론은 전용 executor 1개다.
  추론·식별·검색 전체에 작업 제한시간을 적용한다. native 추론이 응답 제한시간
  이후에도 실행 중이면 완료까지 추가 이미지 추론을 거절하며 큐에 쌓지 않는다.
  native 호출 자체를 강제 종료하는 기능은 없으므로 실제 모델이 멈춘 경우에는
  운영자가 프로세스 상태를 확인해야 한다.

`VISION_INTERNAL_TOKEN`은 User와 같은 값을 주입해야 한다. 기본 계정 일일 상한은
User가 관리하며 운영 확정값을 User 설정으로 적용한다. 학습 수집은 하지 않는다.
텍스트 장소 검색은 Kakao 실측 결과가 없으면 실패하며 모델로 장소를 지어내지 않는다.

검증: `python -m pytest -q`. 테스트는 HTTP MockTransport, WebSocket ASGI 및
모델 대역을 사용하며 실제 Gemini/Kakao 호출이나 모델 가중치 다운로드를 하지 않는다.
