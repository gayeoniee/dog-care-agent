"""피부 모델의 어휘 — **원본은 저쪽입니다** (`dog-skin-screening/src/config.py`).

여기 적어두는 이유는 하나입니다. 게이트가 "모델이 이 단어를 말했나"를 보려면
단어 목록이 필요한데, 그걸 보려고 torch 를 끌고 오는 서브레포를 import 하면
오케스트레이터가 **피부 모델 없이는 못 뜨게** 됩니다. 게이트는 모델이 안 붙은
상태에서도 돌아야 합니다 (그게 테스트의 절반입니다).

⚠️ 저쪽이 바뀌면 여기도 바뀌어야 합니다. `tests/test_vocab_sync.py` 가
   서브레포가 곁에 있을 때만 대조합니다 — 없으면 조용히 건너뜁니다.
"""

from __future__ import annotations

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

#: 계열 4군 — **이건 말해도 됩니다** (2026-09-08 사용자 결정, 저쪽
#: `docs/결정_계열_이름을_켤_것인가.md`). 이름이 아니라 묶음이라서입니다.
GROUPS: dict[str, tuple[str, ...]] = {
    "융기·발진": ("A1", "A4"),
    "표면 변화": ("A2", "A3"),
    "미란·궤양": ("A5",),
    "결절·종괴": ("A6",),
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


def terms_allowed_by(group: str | None) -> frozenset[str]:
    """그 계열이 **승인해 주는** 단어들.

    계열 이름 두 개(`미란·궤양`·`결절·종괴`)는 6종 이름과 글자가 같습니다.
    그래서 "궤양" 을 무조건 막으면 **승인된 계열 문장까지 막힙니다.**
    모델이 실제로 `group` 을 내놓았을 때만 그 조각들을 열어 줍니다.
    """
    if not group:
        return frozenset()
    return frozenset(_tokens(group))
