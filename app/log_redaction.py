"""로그 좌표 가림 필터.

무거운 모델 의존이 없는 자리에 따로 둔다. app/main.py 는 YOLO 모듈을
끌어오므로, 필터가 거기 있으면 가림만 확인하고 싶은 곳까지 모델 설치를
요구하게 된다. 이 모듈을 import 하면 외부 요청 로거에 필터가 걸린다.
"""
from __future__ import annotations

import logging
import re


class CoordinateRedactingFilter(logging.Filter):
    """로그에 실린 기기 좌표를 가린다.

    장소 검색과 역지오코딩은 기기 위치를 카카오 API 쿼리(x/y)로 싣는데,
    나가는 요청을 남기는 로거(httpx)가 요청 URL 을 통째로 찍는다. 아무
    조치를 하지 않으면 기기 위치가 정상 요청 한 줄마다 로그에 쌓인다.
    URL 을 만드는 자리가 아니라 로그로 나가는 길목에서 한 번에 막는다.

    좌표를 담는 이름은 두 갈래다. lat/lng 계열은 앞에 말이 붙어도
    (start_lat 처럼) 이름의 일부로 보아 함께 잡는다. 카카오 쿼리의 단독
    x/y 는 흔한 글자라, 쿼리 구분자(? &) 바로 뒤에 올 때만 잡아
    index=2 같은 좌표 아닌 값이 지워지는 것을 막는다.
    """

    _PATTERNS = (
        re.compile(r"\b(\w*(?:latitude|longitude|lat|lng))=-?\d+(?:\.\d+)?"),
        re.compile(r"([?&][xy])=-?\d+(?:\.\d+)?"),
    )

    @classmethod
    def _scrub(cls, text: str) -> str:
        """한 문자열에서 좌표 값을 자리표시자로 바꾼다."""
        for pattern in cls._PATTERNS:
            text = pattern.sub(r"\1=***", text)
        return text

    def filter(self, record: logging.LogRecord) -> bool:
        """레코드의 본문과 인자에 섞인 좌표를 가린다.

        인자가 문자열이 아닐 수 있다. 나가는 요청 로그는 주소를 문자열이
        아니라 URL 객체로 넘기는데, 문자열만 훑으면 그 인자가 그대로 통과해
        완성된 문장에는 값이 남는다. 그래서 문자열 인자를 가린 뒤 문장을
        만들어 보고, 그래도 가릴 것이 남아 있으면 그때는 문장을 미리 조립해
        통째로 가린다.

        미리 조립하는 것은 가릴 것이 남은 레코드에만 한다. 모든 레코드를
        조립하면 지연 서식의 이점이 사라지고, 인자를 따로 보는 처리기가
        있으면 그 값도 함께 잃는다.
        """
        if isinstance(record.msg, str):
            record.msg = self._scrub(record.msg)
        if record.args:
            record.args = tuple(
                self._scrub(a) if isinstance(a, str) else a
                for a in record.args
            )
            try:
                rendered = record.getMessage()
            except (TypeError, ValueError):
                # 서식과 인자가 맞지 않는 레코드다. 여기서 막을 것은 없고,
                # 원래대로 두면 로깅 쪽이 자기 방식으로 알린다.
                return True
            cleaned = self._scrub(rendered)
            if cleaned != rendered:
                record.msg = cleaned
                record.args = ()
        return True


logging.getLogger("uvicorn.access").addFilter(CoordinateRedactingFilter())
# 나가는 요청 로그도 같은 길목을 지나게 한다. 이 로거는 자기 핸들러를 두지
# 않고 루트로 올려 보내지만, 필터는 레코드를 만든 로거에서 먼저 도므로
# 여기에 걸어야 args 가 문자열로 남아 있는 동안 가릴 수 있다.
logging.getLogger("httpx").addFilter(CoordinateRedactingFilter())
