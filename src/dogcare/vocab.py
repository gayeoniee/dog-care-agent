"""피부 모델의 어휘 — **원본은 저쪽입니다** (`dog-skin-screening/src/config.py`).

여기 적어두는 이유는 하나입니다. 게이트가 "모델이 이 단어를 말했나" 를 보려면
단어 목록이 필요한데, 그걸 보려고 torch 를 끌고 오는 서브레포를 import 하면
오케스트레이터가 **피부 모델 없이는 못 뜨게** 됩니다. 게이트는 모델이 안 붙은
상태에서도 돌아야 합니다 (그게 테스트의 절반입니다).

⚠️ 저쪽이 바뀌면 여기도 바뀌어야 합니다. `tests/test_vocab_sync.py` 가
   서브레포가 곁에 있을 때만 대조합니다 — 없으면 건너뜁니다. 한 번 어긋난
   적이 있습니다: 계열 이름이 2026-09-10 에 보호자 말로 바뀌었는데
   (`미란·궤양` → `벗겨지거나 패인 상처`) 여기는 옛 이름을 들고 있었습니다.
"""

from __future__ import annotations

from typing import Any

#: 2단계 병변 6종. **앱에도 답변에도 이름이 나가면 안 되는 것들입니다.**
#: holdout 에서 1등 이름이 56.6% 틀렸습니다 — 그래서 저쪽 응답 계약에는
#: "1등" 필드가 아예 없습니다. 이 목록은 그 계약을 **문장에서도** 지키려고 있습니다.
LESION_KO: dict[str, str] = {
    "A1": "구진·플라크",
    "A2": "비듬·각질·상피성잔고리",
    "A3": "태선화·과다색소침착",
    "A4": "농포·여드름",
    "A5": "미란·궤양",
    "A6": "결절·종괴",
}

LESION_EN: dict[str, str] = {
    "A1": "Papule / Plaque",
    "A2": "Scale / Crust / Epidermal collarette",
    "A3": "Lichenification / Hyperpigmentation",
    "A4": "Pustule / Acne",
    "A5": "Erosion / Ulcer",
    "A6": "Nodule / Mass",
}

#: 계열 4군 — **이건 말해도 됩니다** (2026-09-08 사용자 결정). 이름이 아니라
#: 묶음이고, 그 확률은 안에 든 것을 **더한 값**이라 아무것도 안 숨깁니다.
#:
#: ★ 이름이 6종과 **글자가 하나도 안 겹칩니다.** 2026-09-10 에 보호자 말로
#:   바꾸면서 그렇게 됐습니다. 덕분에 게이트가 단순해집니다 — 계열 문장은
#:   금지 어휘를 아예 안 건드리므로 예외를 둘 일이 없습니다.
GROUPS: dict[str, tuple[str, ...]] = {
    "솟아오른 변화": ("A1", "A4"),
    "피부 표면·색·두께 변화": ("A2", "A3"),
    "벗겨지거나 패인 상처": ("A5",),
    "깊거나 단단한 혹": ("A6",),
}

#: 계열마다 붙는 **용어 풀이** — 6종 이름 그대로입니다. 저쪽은 이걸 네 줄
#: 전부에 같은 방식으로 실어서 앱 "자세히 보기" 에 둡니다(단정이 아니라 풀이).
#: ⚠️ **모델에게는 주지 않습니다** — 프롬프트로 참으라고 하는 것보다 안 주는
#:    편이 싸고 확실합니다 (`mcp_servers/skin_screening_server.py:_for_model`).
GROUP_LABELS: dict[str, str] = {
    g: "·".join(LESION_KO[c] for c in codes) for g, codes in GROUPS.items()
}


def _tokens(name: str) -> list[str]:
    """`"비듬·각질·상피성잔고리"` → `["비듬", "각질", "상피성잔고리"]`.

    ★ 통짜 문자열만 막으면 게이트가 헛돕니다. 모델은 계약에 적힌 그대로
      "미란·궤양" 이라고 쓰지 않고 **"궤양이 의심됩니다"** 라고 씁니다.
      가운뎃점과 슬래시로 쪼개서 조각마다 봅니다.
    """
    out: list[str] = []
    for part in name.replace("/", "·").split("·"):
        part = part.strip()
        if len(part) >= 2:          # "A" 같은 한 글자는 아무 문장에나 걸린다
            out.append(part)
    return out


#: 금지 어휘 전체 — 6종 이름을 조각낸 것. 한국어와 영어 둘 다.
LESION_TERMS: frozenset[str] = frozenset(
    t for name in (*LESION_KO.values(), *LESION_EN.values()) for t in _tokens(name)
)


def group_name(group: Any) -> str | None:
    """계약의 `stage2.group` 에서 계열 이름만 꺼냅니다.

    ⚠️ **dict 입니다.** `{"name", "prob", "percent", "confidence", "text",
       "labels", "feature"}`. 문자열인 줄 알고 `.replace()` 를 불렀다가
       실기기에서 `AttributeError` 로 죽은 적이 있습니다 — 게이트가 죽으면
       답이 안 나가는 게 아니라 **요청이 통째로 죽습니다.**
       옛 계약이나 테스트가 문자열을 줄 수도 있어 둘 다 받습니다.
    """
    if not group:
        return None
    if isinstance(group, dict):
        name = group.get("name")
        return str(name) if name else None
    return str(group)


def terms_allowed_by(group: Any) -> frozenset[str]:
    """그 계열 **이름이** 승인해 주는 단어들.

    지금 계열 이름 넷은 6종과 글자가 안 겹쳐서 **늘 빈 집합**입니다. 그래도
    남겨 둡니다 — 옛 이름(`미란·궤양`)은 A5 와 글자가 같았고, 이름이 다시
    바뀌면 여기가 자동으로 맞습니다. 여는 열쇠는 프롬프트가 아니라 **판정
    결과 자신**이라는 규칙도 이 함수가 들고 있습니다.
    """
    name = group_name(group)
    return frozenset(_tokens(name)) if name else frozenset()
