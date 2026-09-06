# syntax=docker/dockerfile:1.7
# map-service-yolo — FastAPI + YOLO + Gemini Vision 실시간 인식 에이전트
#
# 멀티 스테이지 이미지 구성:
#   1) builder stage : 컴파일 도구를 설치하고 requirements.txt 를 휠로 빌드해 /wheels 에 모은다.
#   2) runtime stage : 슬림 베이스 위에 builder 가 만든 휠만 사용해 의존성을 설치한다.
# 비루트 사용자(app, uid=10001) 로 컨테이너를 띄우며, uvicorn 으로 app.main:app 을 8000 에 노출한다.
#
# CPU 인덱스에서 torch/torchvision을 먼저 선택하고 그 판을 제약으로 고정한다.
# extra-index만 쓰면 PyPI의 더 높은 CUDA 판을 선택할 수 있다.

ARG PYTHON_VERSION=3.12

# ─── builder stage ───────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /build
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
 && pip freeze | grep -E '^(torch|torchvision)==' > /build/torch-constraints.txt \
 && pip wheel --wheel-dir=/wheels --constraint=/build/torch-constraints.txt \
      --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt


# ─── runtime stage ───────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# ultralytics 가 opencv-python(비 headless)을 전이 의존으로 끌고 올 수 있어
# 런타임에도 OpenCV 의 네이티브 의존 라이브러리를 둔다.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*
# 권한 최소화를 위해 비루트 사용자 app(uid 10001) 생성.
# ultralytics 는 설정 파일을 홈의 .config 아래 두는데, 그 디렉터리가 없으면
# 홈 전체를 쓰기 불가로 보고 /tmp 로 물러난다(기동 때마다 경고 + 설정 재생성).
# 미리 만들어 두어 위치를 고정한다.
RUN useradd -m -u 10001 app \
 && mkdir -p /home/app/.config \
 && chown app:app /home/app/.config
WORKDIR /app
COPY --from=builder /wheels /wheels
COPY requirements.txt .
# --no-index --find-links=/wheels: PyPI 접근 없이 builder 의 휠만으로 설치.
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
 && pip check \
 && rm -rf /wheels

# YOLO 가중치를 빌드 시점에 이미지로 내려받는다(bake).
#   - 기동 시 네트워크와 쓰기 권한이 필요 없어 오프라인·비루트에서도 결정적으로 뜬다.
#   - ultralytics 의 이름 기반 자동 다운로드는 경로 접두사가 붙으면 동작이 달라지므로
#     릴리스 태그로 고정한 URL 을 표준 라이브러리로 직접 받는다(이미지에 curl 불요).
#   - 크기 검증으로 잘린 다운로드를 빌드 단계에서 걸러낸다.
#   - 소스 복사보다 먼저 실행해 코드 수정 시에도 이 레이어 캐시가 유지되게 한다.
# 무결성을 더 강하게 고정하려면 최초 빌드 후 sha256 을 구해
#   ADD --checksum=sha256:<hash> ${YOLO_MODEL_URL} /app/models/yolo11n.pt
# 로 교체할 수 있다.
ARG YOLO_MODEL_URL=https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
RUN python -c "import os, urllib.request; os.makedirs('/app/models', exist_ok=True); urllib.request.urlretrieve('${YOLO_MODEL_URL}', '/app/models/yolo11n.pt')" \
 && test "$(stat -c%s /app/models/yolo11n.pt)" -gt 1000000

# 애플리케이션 소스를 app 사용자 소유로 복사.
COPY --chown=app:app app ./app
USER app
RUN python -c "import app.main, cv2, torch; assert torch.version.cuda is None; from ultralytics import YOLO; YOLO('/app/models/yolo11n.pt')"
EXPOSE 8000
# HEALTHCHECK
#   - /health 가 200 을 돌려주면 healthy.
#   - torch import 와 YOLO 모델 로드가 느려 start-period 를 넉넉히 둔다.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status==200 else 1)"
# ENTRYPOINT
#   - uvicorn 으로 app.main 모듈의 `app` (FastAPI 인스턴스) 를 0.0.0.0:8000 에 바인딩.
ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--ws-max-size", "4194304", "--ws-max-queue", "4"]
