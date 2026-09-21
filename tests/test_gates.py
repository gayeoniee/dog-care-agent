"""게이트 테스트 — **이 파일의 절반은 "말하면 안 되는 것" 목록입니다.**

저쪽(dog-skin-screening)의 `tests/test_agent.py` 가 계약에 "1등" 필드가 생기는
것을 감시하듯, 여기는 **문장에** 그게 생기는 것을 감시합니다. 필드를 막아도
문장으로 새면 같은 일입니다.
"""

from __future__ import annotations

import pytest

from dogcare.gates import DISCLAIMER, TurnFacts, check
from dogcare.vocab import LESION_KO, LESION_TERMS

ABNORMAL = {
    "verdict": "abnormal",
    "stage2": {"shown": True, "group": None, "alert": None},
    "meta": {"stage2_arms": 3},
}


def _with_group(group):
    return {**ABNORMAL, "stage2": {**ABNORMAL["stage2"], "group": group}}


def facts(**kw):
    kw.setdefault("expected_stage2_arms", 3)
    return TurnFacts(**kw)


# ── G1. 병변 6종 이름은 문장에도 못 나온다 ────────────────────
@pytest.mark.parametrize("term", sorted(LESION_TERMS))
def test_g1_모든_병변_어휘가_막힌다(term):
    """26개 어휘 **전부** 확인합니다. 하나씩 빠뜨리는 게 이런 목록의 실패 방식입니다."""
    answer = f"사진을 보면 {term} 소견이 있습니다. {DISCLAIMER}"
    r = check(answer, facts(had_image=True, screening=ABNORMAL))
    assert not r.ok and any(v.gate == "G1" for v in r.violations), term


def test_g1_계열은_승인되면_통과한다():
    answer = f"모양만 보면 표면 변화 계열에 가깝습니다. {DISCLAIMER}"
    assert check(answer, facts(had_image=True, screening=_with_group("표면 변화"))).ok


def test_g1_계열이_null이면_같은_문장도_막힌다():
    """★ 승인의 출처는 프롬프트가 아니라 판정 결과입니다."""
    answer = f"모양만 보면 궤양 계열에 가깝습니다. {DISCLAIMER}"
    assert not check(answer, facts(had_image=True, screening=ABNORMAL)).ok


def test_g1_계열이_6종과_글자가_같을_때_승인되면_열린다():
    """미란·궤양은 계열 이름이자 A5 이름입니다. group 이 열쇠입니다."""
    answer = f"모양만 보면 미란·궤양 계열에 가깝습니다. {DISCLAIMER}"
    assert check(answer, facts(had_image=True, screening=_with_group("미란·궤양"))).ok


def test_g1_승인된_계열이어도_다른_계열_이름은_막힌다():
    answer = f"미란·궤양 계열이지만 농포일 수도 있습니다. {DISCLAIMER}"
    r = check(answer, facts(had_image=True, screening=_with_group("미란·궤양")))
    assert not r.ok and "농포" in str(r)


def test_g1_영어_이름도_막힌다():
    answer = f"This looks like a Pustule. {DISCLAIMER}"
    assert not check(answer, facts(had_image=True, screening=ABNORMAL)).ok


def test_g1_피부와_무관한_턴은_건드리지_않는다():
    """산책 상담에 '결절' 이 나올 일은 없지만, 게이트 범위는 분명해야 합니다."""
    assert check("산책 줄을 짧게 잡아 주세요. [자료 1]",
                 facts(question="산책할 때 줄을 당겨요",
                       rag_coverage="ok", rag_source_count=3)).ok


# ── G2. 면책은 코드가 붙이고 코드가 확인한다 ──────────────────
def test_g2_면책이_빠지면_막힌다():
    r = check("표면 변화 계열에 가깝습니다.",
              facts(had_image=True, screening=_with_group("표면 변화")))
    assert not r.ok and any(v.gate == "G2" for v in r.violations)


def test_g2_판정이_없는_턴에는_면책을_요구하지_않는다():
    assert check("산책 줄을 짧게 잡아 주세요.", facts(question="산책 상담")).ok


# ── G3. 인용은 실제 근거 개수를 못 넘는다 ─────────────────────
def test_g3_자료없음인데_인용하면_막힌다():
    r = check("[자료 2] 에 따르면 괜찮습니다.", facts(question="고양이 모래", rag_coverage="none"))
    assert not r.ok and any(v.gate == "G3" for v in r.violations)


def test_g3_범위를_넘는_인용은_막힌다():
    r = check("[자료 1] 과 [자료 7] 을 보세요.",
              facts(question="짖음", rag_coverage="ok", rag_source_count=3))
    assert not r.ok and "7" in str(r)


def test_g3_범위_안_인용은_통과한다():
    assert check("[자료 1] 과 [자료 3] 을 보세요.",
                 facts(question="짖음", rag_coverage="ok", rag_source_count=3)).ok


# ── G4. 앙상블이 조용히 줄면 판정을 못 쓴다 ────────────────────
def test_g4_팔이_줄면_막힌다():
    """배포에서 실제로 났던 사고입니다 (2026-09-07)."""
    one_arm = {**_with_group("표면 변화"), "meta": {"stage2_arms": 1}}
    r = check(f"표면 변화 계열입니다. {DISCLAIMER}", facts(had_image=True, screening=one_arm))
    assert not r.ok and any(v.gate == "G4" for v in r.violations)


def test_g4_정상_판정은_2단계를_안_도므로_검사하지_않는다():
    normal = {"verdict": "normal", "stage2": {"group": None}, "meta": {}}
    assert check(f"이상 소견은 보이지 않습니다. {DISCLAIMER}",
                 facts(had_image=True, screening=normal)).ok


# ── G5. 사진 없이 피부를 판정하지 않는다 ──────────────────────
def test_g5_사진_없는_피부질문에_그냥_답하면_막힌다():
    r = check("보습제를 발라 주세요.", facts(question="우리 개 피부가 자꾸 붉어져요",
                                       rag_coverage="ok", rag_source_count=2))
    assert not r.ok and any(v.gate == "G5" for v in r.violations)


def test_g5_사진을_요청하면_통과한다():
    assert check("그 부위가 잘 보이게 사진을 한 장 찍어 주시겠어요?",
                 facts(question="우리 개 피부가 자꾸 붉어져요",
                       rag_coverage="ok", rag_source_count=2)).ok


# ── 게이트는 첫 위반에서 멈추지 않는다 ────────────────────────
def test_위반이_여럿이면_전부_보고한다():
    r = check("농포가 보입니다. [자료 9] 참고.",
              facts(had_image=True, screening=ABNORMAL, rag_coverage="ok", rag_source_count=2))
    assert {v.gate for v in r.violations} >= {"G1", "G2", "G3"}


def test_어휘_목록이_6종을_다_덮는다():
    """LESION_KO 가 늘면 여기서 걸립니다 — 목록이 조용히 새지 않게."""
    for code, name in LESION_KO.items():
        assert any(t in LESION_TERMS for t in name.split("·")), code


# ── 방아쇠 어휘 회귀 ──────────────────────────────────────────
@pytest.mark.parametrize("question", [
    "우리 개 배가 자꾸 빨개져요",        # ← "빨갛" 만 넣었다가 놓쳤던 것
    "등이 빨간데 괜찮을까요",
    "자꾸 긁어요",
    "털 빠짐이 심해요",
    "발을 계속 간지러워해요",
    "딱지 같은 게 생겼어요",
    "피부가 부어올랐어요",
])
def test_g5_활용형이_달라도_피부질문으로_잡힌다(question):
    """한국어는 어간이 바뀝니다. 목록에 없는 활용형 하나가 통과 구멍이 됩니다."""
    r = check("보습제를 발라 주세요.", facts(question=question, rag_coverage="full",
                                        rag_source_count=2))
    assert not r.ok and any(v.gate == "G5" for v in r.violations), question
