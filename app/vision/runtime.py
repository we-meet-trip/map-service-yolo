"""프로세스 전체 작업 상한과 공유 모델의 단일 추론 실행기."""
import asyncio
from concurrent.futures import ThreadPoolExecutor


class VisionBusy(Exception):
    pass


class VisionRuntime:
    def __init__(self, max_jobs: int):
        self._max_jobs = max_jobs
        self._active = 0
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vision-inference")
        self._inference = None
        self._image_reserved = False

    def reserve(self, *, image: bool):
        if self._active >= self._max_jobs or (image and (
            self._image_reserved or self._inference is not None and not self._inference.done()
        )):
            raise VisionBusy()
        self._active += 1
        if image:
            self._image_reserved = True

    def release(self, *, image: bool = False):
        self._active -= 1
        if image:
            self._image_reserved = False

    async def detect(self, detector, frame_b64):
        if self._inference is not None and not self._inference.done():
            raise VisionBusy()
        self._inference = asyncio.get_running_loop().run_in_executor(
            self._executor, detector.detect, frame_b64
        )
        self._inference.add_done_callback(lambda future: None if future.cancelled() else future.exception())
        # 응답 제한시간이 지나도 native 추론은 실행 중일 수 있다. 완료까지
        # 같은 모델을 다시 호출하거나 executor에 작업을 쌓지 않는다.
        return await asyncio.shield(self._inference)

    def close(self):
        self._executor.shutdown(wait=False, cancel_futures=True)
