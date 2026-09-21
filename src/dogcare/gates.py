"""★ 게이트 — **모델이 말해도 되는 것을 코드가 정합니다.**

이 파일이 이 저장소의 요점입니다. 나머지는 배선입니다.

왜 있나
-------
서브에이전트 둘은 각자 자기 자리에서 이미 조심하고 있습니다.

* 피부 모델은 응답 계약에 **"1등 병변" 필드가 없습니다.** holdout 에서 그 이름이
  56.6% 틀려서, 앱이 고를 수 없게 계약에서 아예 뺐습니다.
* RAG 는 근거가 없으면 **모른다고 말합니다.** 범위 밖 질문 거절률 0/4 → 7/7.

그런데 그 둘 사이에 LLM 을 하나 세우는 순간, **두 안전장치가 같이 무력해집니다.**
계약에 1등 필드가 없어도 분포는 있으니, 6종 확률을 그대로 읽은 모델은
"농포일 가능성이 높아 보입니다" 라고 씁니다. 필드를 뺀 의미가 문장에서 사라집니다.
RAG 가 "자료가 없다" 고 한 자리도, 모델이 앞뒤를 매끄럽게 이으려고
[자료 3] 을 붙이면 없던 근거가 생깁니다.

그래서 **오케스트레이터의 안전은 프롬프트가 아니라 여기서 정합니다.**
프롬프트로 부탁한 것은 재 볼 수 없지만, 여기 있는 것은 테스트가 잡습니다.

원칙
----
1. 게이트는 **LLM 의 말이 아니라 툴이 실제로 돌려준 것만** 봅니다 (TurnFacts).
2. 트리거 조건은 **게이트를 더 엄하게만** 만들 수 있습니다. 키워드로 게이트를
   *켜는* 건 안전하지만 (헛걸려도 검사가 한 번 더 도는 것뿐), 키워드로 *끄는*
   건 안전하지 않습니다 — 그 목록에 없는 단어 하나가 통과 구멍이 됩니다.
3. 게이트가 걸리면 **답을 고쳐 쓰지 않고 막습니다.** 고쳐 쓰면 무엇이 걸렸는지가
   기록에서 사라집니다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from dogcare.vocab import LESION_TERMS, terms_allowed_by

#: 피부 모델이 붙이는 면책. **모델이 쓰는 문장이 아니라 코드가 붙이는 문장입니다.**
#: 그래서 원문 대조가 성립합니다 — 모델이 바꿔 쓸 수 있는 것이면 못 잽니다.
DISCLAIMER = (
    "이 결과는 수의학적 진단이 아니며, 수의사의 진료를 대체하지 않습니다. "
    "참고용 스크리닝 정보로만 사용해 주세요."
)

#: G1/G5 를 **켜는** 방아쇠. 끄는 데는 안 씁니다 (원칙 2).
#: 사진이 붙었거나 피부 툴을 불렀으면 이 목록과 무관하게 이미 켜집니다.
#   ⚠️ 한국어 활용형에 걸립니다. "빨갛" 만 넣었다가 **"빨개져요" 를 놓쳤습니다**
#      (evals/run.py gates, G5). 어간이 바뀌는 말은 형태를 나열해야 합니다.
_SKIN_HINTS = (
    "피부", "병변", "발진", "긁", "가려", "간지", "털 빠", "탈모", "각질", "비듬",
    "딱지", "진물", "고름", "부었", "부어올", "붓",
    "붉", "빨갛", "빨개", "빨간", "빨감",
    "뾰루지", "혹", "멍울", "종기", "습진", "두드러기", "각질",
)

#: RAG 가 답변에 심는 인용 표시. 예) [자료 3] · [자료 12]
_CITATION = re.compile(r"\[자료\s*(\d+)\]")


@dataclass(frozen=True)
class GateViolation:
    gate: str
    reason: str
    detail: str = ""

    def __str__(self) -> str:
        return f"{self.gate} {self.reason}" + (f" — {self.detail}" if self.detail else "")


@dataclass(frozen=True)
class TurnFacts:
    """이 턴에 **서브에이전트가 실제로 돌려준 것.** LLM 의 주장은 안 들어옵니다."""

    question: str = ""
    had_image: bool = False
    #: 피부 MCP 가 돌려준 계약 JSON 그대로. 안 불렀으면 None.
    screening: dict[str, Any] | None = None
    #: RAG MCP 가 돌려준 coverage — ok | needs_detail | none. 안 불렀으면 None.
    rag_coverage: str | None = None
    #: RAG 가 돌려준 근거 개수. 인용 번호의 상한이 됩니다.
    rag_source_count: int = 0
    #: 앙상블 팔 수 기대값. None 이면 검사하지 않습니다.
    expected_stage2_arms: int | None = None

    @property
    def skin_related(self) -> bool:
        """이 턴을 피부 게이트에 걸 것인가. **넓게 잡습니다** (원칙 2)."""
        if self.had_image or self.screening is not None:
            return True
        return any(h in self.question for h in _SKIN_HINTS)


# ──────────────────────────────────────────────────────────────
# G1 — 병변 이름 누설
# ──────────────────────────────────────────────────────────────
def g1_lesion_name(answer: str, facts: TurnFacts) -> list[GateViolation]:
    """6종 이름을 문장에 쓰면 막습니다. **계열(group)은 승인된 말입니다.**

    승인의 출처는 프롬프트가 아니라 **판정 결과 자신**입니다. 확신이 낮아
    stage2.group 이 null 이면 그 턴에는 아무 이름도 못 씁니다. 확신이 있어
    "표면 변화" 를 내놓았으면 그 조각만 열립니다.

    계열 이름 둘(미란·궤양 · 결절·종괴)은 6종 이름과 글자가 같습니다. 그래서
    "궤양" 을 무조건 막으면 **승인된 문장까지 막힙니다.** 여는 열쇠는 group 입니다.
    """
    if not facts.skin_related:
        return []
    group = None
    if facts.screening:
        group = (facts.screening.get("stage2") or {}).get("group")
    allowed = terms_allowed_by(group)
    low = answer.lower()
    hits = sorted({t for t in LESION_TERMS - allowed if t.lower() in low})
    if not hits:
        return []
    why = f"계열 승인 없음(group={group!r})" if facts.screening else "판정 결과가 없는 턴"
    return [GateViolation("G1", "병변 이름이 답변에 나왔습니다", f"{', '.join(hits)} — {why}")]


# ──────────────────────────────────────────────────────────────
# G2 — 면책이 떨어져 나감
# ──────────────────────────────────────────────────────────────
def g2_disclaimer(answer: str, facts: TurnFacts) -> list[GateViolation]:
    """판정을 말했으면 면책이 **원문 그대로** 붙어 있어야 합니다.

    모델에게 "면책을 붙여 주세요" 라고 부탁하지 않습니다 — 코드가 붙이고, 여기서
    붙었는지 봅니다. 부탁한 것은 잴 수 없고 붙인 것은 잴 수 있습니다.
    """
    if not facts.screening:
        return []
    if DISCLAIMER in answer:
        return []
    return [GateViolation("G2", "면책 문구가 답변에 없습니다",
                          "코드가 붙이는 문장입니다 — 조합 과정에서 떨어졌습니다")]


# ──────────────────────────────────────────────────────────────
# G3 — 인용 무결성
# ──────────────────────────────────────────────────────────────
def g3_citations(answer: str, facts: TurnFacts) -> list[GateViolation]:
    """없는 근거를 가리키는 인용을 막습니다.

    RAG 가 "자료 없음" 으로 답한 턴에 인용이 하나라도 있으면, 그건 **모델이
    앞뒤를 매끄럽게 이으려고 만들어 낸 번호**입니다. 개수만 세면 잡힙니다.
    """
    cited = [int(n) for n in _CITATION.findall(answer)]
    if not cited:
        return []
    if facts.rag_coverage == "none":
        nums = ", ".join(str(n) for n in sorted(set(cited)))
        return [GateViolation("G3", "자료가 없다고 한 턴에 인용이 있습니다", f"[자료 {nums}]")]
    bad = sorted({n for n in cited if n < 1 or n > facts.rag_source_count})
    if bad:
        return [GateViolation("G3", "근거 범위를 벗어난 인용입니다",
                              f"근거 {facts.rag_source_count}건인데 {bad} 을 가리킵니다")]
    return []


# ──────────────────────────────────────────────────────────────
# G4 — 앙상블이 조용히 줄어듦
# ──────────────────────────────────────────────────────────────
def g4_ensemble_arms(answer: str, facts: TurnFacts) -> list[GateViolation]:
    """2단계 팔 수가 기대와 다르면 **판정을 못 쓰게 막습니다.**

    저쪽에서 실제로 났던 사고입니다 — 배포에서 앙상블이 조용히 1팔로 줄었는데,
    응답은 멀쩡해 보여서 아무도 몰랐습니다(2026-09-07). 성능만 떨어집니다.
    meta.stage2_arms 를 계약에 넣은 게 그래서고, 여기서 한 번 더 봅니다.
    """
    if facts.expected_stage2_arms is None or not facts.screening:
        return []
    if facts.screening.get("verdict") != "abnormal":
        return []                      # 정상·재촬영은 2단계를 안 돕니다
    got = (facts.screening.get("meta") or {}).get("stage2_arms")
    if got == facts.expected_stage2_arms:
        return []
    return [GateViolation("G4", "앙상블 팔 수가 기대와 다릅니다",
                          f"{got}팔 — {facts.expected_stage2_arms}팔을 기대했습니다")]


# ──────────────────────────────────────────────────────────────
# G5 — 사진 없이 피부를 판정함
# ──────────────────────────────────────────────────────────────
def g5_needs_photo(answer: str, facts: TurnFacts) -> list[GateViolation]:
    """사진 없이 피부 질문을 받으면 **사진을 달라고 해야** 합니다.

    RAG 코퍼스에도 피부를 스치는 문서가 있어서, 사진 없이도 그럴듯한 답이 나옵니다.
    그런데 그 답에는 이 보호자의 개가 안 들어 있습니다. 판정은 사진이 있을 때만
    합니다. (G1 이 이름을 막지만, 이름 없이도 판정조로 쓸 수 있습니다)
    """
    if facts.screening is not None or facts.had_image or not facts.skin_related:
        return []
    if any(w in answer for w in ("사진", "찍어", "촬영")):
        return []
    return [GateViolation("G5", "사진 없이 피부 질문에 답했습니다",
                          "판정은 사진이 있을 때만 합니다 — 사진을 요청해야 합니다")]


GATES = (g1_lesion_name, g2_disclaimer, g3_citations, g4_ensemble_arms, g5_needs_photo)


@dataclass
class GateReport:
    violations: list[GateViolation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def __str__(self) -> str:
        return "통과" if self.ok else "\n".join(str(v) for v in self.violations)


def check(answer: str, facts: TurnFacts) -> GateReport:
    """게이트를 **전부** 돌립니다 — 첫 위반에서 멈추지 않습니다.

    멈추면 기록에 하나만 남고, 고친 뒤 다시 돌려야 나머지가 보입니다.
    """
    return GateReport([v for gate in GATES for v in gate(answer, facts)])
