"""외부 호출 전에 base64/JPEG와 디코딩 크기를 제한한다."""
import base64
import binascii
from io import BytesIO

from PIL import Image, UnidentifiedImageError

from app.schemas.vision_schemas import FRAME_MAX_BYTES, FRAME_MAX_PIXELS


def validate_frame(frame_b64: str) -> None:
    try:
        raw = base64.b64decode(frame_b64, validate=True)
        if not raw or len(raw) > FRAME_MAX_BYTES or not raw.endswith(b"\xff\xd9"):
            raise ValueError("invalid JPEG frame")
        with Image.open(BytesIO(raw)) as image:
            if image.format != "JPEG" or image.width * image.height > FRAME_MAX_PIXELS:
                raise ValueError("invalid JPEG dimensions")
            image.load()
    except (binascii.Error, OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValueError("invalid JPEG frame") from exc
