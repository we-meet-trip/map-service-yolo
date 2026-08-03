"""Kakao 로컬 API — 키워드 장소 검색 + 역지오코딩."""
from __future__ import annotations

import re

import httpx

_KAKAO_KEYWORD_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"
_KAKAO_COORD2ADDRESS_URL = "https://dapi.kakao.com/v2/local/geo/coord2address.json"
_KAKAO_COORD2REGIONCODE_URL = "https://dapi.kakao.com/v2/local/geo/coord2regioncode.json"


async def kakao_reverse_geocode(lat: float, lng: float, api_key: str) -> str:
    """위도/경도 → 실제 주소 문자열 반환.

    도로명 주소 우선, 없으면 지번 주소 반환.
    실패 시 빈 문자열 반환.
    """
    headers = {"Authorization": f"KakaoAK {api_key}"}
    params = {"x": lng, "y": lat}

    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            resp = await client.get(_KAKAO_COORD2ADDRESS_URL, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            docs = data.get("documents", [])
            if docs:
                doc = docs[0]
                road = doc.get("road_address")
                if road:
                    building = road.get("building_name", "")
                    address = road.get("address_name", "")
                    return f"{address} {building}".strip()
                jibun = doc.get("address", {})
                return jibun.get("address_name", "")
        except Exception:
            pass
    return ""

# 장소 유형 키워드 (우선순위 높은 순)
_PLACE_TYPES = [
    "빵집", "베이커리", "카페", "커피", "식당", "음식점", "맛집",
    "편의점", "마트", "슈퍼", "병원", "약국", "은행", "주유소",
    "헬스장", "헬스", "영화관", "영화", "노래방", "pc방", "피씨방",
    "치킨", "피자", "분식", "한식", "중식", "일식", "양식", "버거",
    "술집", "바", "이자카야", "고깃집", "삼겹살",
]

# 불용어 (제거할 단어)
_STOPWORDS = {
    "나", "저", "저는", "나는", "인데", "이고", "이야", "근처", "주변",
    "가까운", "알려", "알려줘", "알려주세요", "찾아", "찾아줘", "찾아주세요",
    "어디", "어디야", "어디예요", "있어", "있나", "있나요", "있어요",
    "좀", "혹시", "그냥", "그리고", "가르쳐", "가르쳐줘", "추천",
    "추천해", "추천해줘", "추천해주세요", "거", "것",
}


def is_local_query(text: str) -> bool:
    """텍스트가 장소 검색 의도인지 판단.

    위치 트리거("근처", "주변" 등) AND 장소 유형("카페", "식당" 등)이
    동시에 있어야 장소 검색으로 판단. 어느 하나만 있으면 일반 대화로 처리.
    """
    location_triggers = {"근처", "주변", "가까운", "맛집", "찾아줘", "찾아주세요", "어디있어", "어디 있어"}
    has_trigger = any(kw in text for kw in location_triggers)
    has_place_type = any(pt in text for pt in _PLACE_TYPES)
    return has_trigger and has_place_type


def extract_search_keyword(text: str) -> str:
    """음성 문장에서 Kakao 검색에 적합한 키워드 추출.

    우선순위:
    1. 장소 유형 키워드가 있으면 그 단어를 중심으로
    2. 없으면 불용어 제거 후 남은 단어 조합
    """
    # 장소 유형 키워드 찾기
    found_type = next((pt for pt in _PLACE_TYPES if pt in text), None)

    # 지역명 추출 (한글 0~5글자 + 지역 접미사 or 고유 캠퍼스명)
    area_match = re.search(
        r"([가-힣]{0,5}(?:대학교?|캠퍼스|에리카|역|동|구|시|읍|면))",
        text,
    )
    area = area_match.group(1).strip() if area_match else ""

    if found_type:
        # "에리카 빵집" 또는 그냥 "빵집"
        return f"{area} {found_type}".strip() if area else found_type

    # 장소 유형 없으면 불용어 제거
    words = text.split()
    cleaned = [w for w in words if w not in _STOPWORDS and len(w) > 1]
    return " ".join(cleaned) if cleaned else text


async def kakao_local_search(
    query: str,
    api_key: str,
    lat: float | None = None,
    lng: float | None = None,
    radius: int = 2000,
    size: int = 5,
) -> list[dict]:
    """Kakao 키워드 장소 검색.

    Args:
        query: 검색어 (예: "에리카 빵집")
        api_key: Kakao REST API 키
        lat: 중심 위도 (없으면 전국 검색)
        lng: 중심 경도
        radius: 반경(m), 최대 20000
        size: 결과 개수

    Returns:
        장소 정보 딕셔너리 리스트
        [{"name": ..., "address": ..., "phone": ..., "category": ..., "distance": ...}]
    """
    params: dict = {"query": query, "size": size}
    # 좌표 0.0 은 유효한 값이므로 truthiness 가 아니라 None 여부로 판단한다.
    if lat is not None and lng is not None:
        params["y"] = lat
        params["x"] = lng
        params["radius"] = radius
        params["sort"] = "distance"

    headers = {"Authorization": f"KakaoAK {api_key}"}

    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.get(_KAKAO_KEYWORD_URL, params=params, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    places = []
    for doc in data.get("documents", []):
        places.append({
            "name": doc.get("place_name", ""),
            "address": doc.get("road_address_name") or doc.get("address_name", ""),
            "phone": doc.get("phone", ""),
            "category": doc.get("category_name", ""),
            "distance": doc.get("distance", ""),
        })
    return places
