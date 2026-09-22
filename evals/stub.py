"""가짜 서브에이전트 — **라우팅만** 재려고 씁니다.

라우팅 평가에서 진짜 모델을 부르면 두 가지가 섞입니다. "툴을 맞게 골랐나" 와
"그 툴이 좋은 답을 했나" 는 다른 질문인데, 한 번에 재면 RAG 코퍼스가 약한 날
라우팅 점수가 같이 떨어집니다. 여기서는 **툴이 무엇을 돌려줄지 고정**해 두고
LLM 이 무엇을 골랐는지만 봅니다.

툴 스키마는 진짜 서버에서 뜬 것과 **같은 모양**이어야 의미가 있습니다.
`--record` 로 실제 서버에서 한 번 떠서 `evals/tools.json` 에 받아 둡니다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
TOOLS_SNAPSHOT = HERE / "tools.json"

#: 툴이 돌려줄 고정 응답. 라우팅을 재는 동안 내용은 안 바뀝니다.
CANNED: dict[str, Any] = {
    "ask_behavior_question": {
        "answer": "산책 중에는 줄이 느슨할 때만 앞으로 가게 해주세요. [자료 1]",
        "coverage": "full",
        "coverage_note": None,
        "source_count": 2,
        "sources": [{"n": 1, "title": "Leash walking", "source": "pmc:1", "score": 0.72,
                     "excerpt": "..."},
                    {"n": 2, "title": "보듬TV 산책", "source": "yt:2", "score": 0.68,
                     "excerpt": "..."}],
        "latency_ms": 900,
    },
    "factcheck_claim": {
        "claims": [{"claim": "서열을 잡아야 한다", "verdict": "contradicted",
                    "note": "AVSAB 가 공식 반박한 지배이론입니다."}],
    },
    "screen_skin_photo": {
        "contract_version": "1.0", "verdict": "abnormal",
        "headline": "피부에 이상 소견이 보입니다.",
        "body": "이 사진만으로 정확하게 알 수 없습니다.",
        "action": "수의사 진료를 받아보시기를 권합니다.",
        "stage1": {"abnormal_prob": 0.71, "abnormal_percent": 71.0,
                   "threshold": 0.1466, "calibrated": True},
        # ★ 실제 계약 모양 — group 은 dict, 이름은 2026-09-10 의 보호자 말.
        #   옛 이름("표면 변화")·문자열 group 으로 돌린 적이 있다. 게이트는 양쪽을 받지만
        #   평가는 진짜와 같은 모양으로 재야 숫자가 진짜에 대해서도 참이다.
        "stage2": {"shown": True,
                   "groups": [{"name": "피부 표면·색·두께 변화", "prob": 0.54, "percent": 54.0},
                              {"name": "솟아오른 변화", "prob": 0.21, "percent": 21.0},
                              {"name": "벗겨지거나 패인 상처", "prob": 0.15, "percent": 15.0},
                              {"name": "깊거나 단단한 혹", "prob": 0.10, "percent": 10.0}],
                   "group": {"name": "피부 표면·색·두께 변화", "prob": 0.54, "percent": 54.0,
                             "confidence": 0.38,
                             "text": "모양만 보면 피부 표면·색·두께 변화에 가깝습니다.",
                             "feature": "딱지, 둥근 비늘, 검어진 피부, 두꺼워진 피부"},
                   "alert": None},
        "text": "피부에 이상 소견이 보입니다.",
        "disclaimer": ("이 결과는 수의학적 진단이 아니며, 수의사의 진료를 대체하지 않습니다. "
                       "참고용 스크리닝 정보로만 사용해 주세요."),
        "meta": {"stage2_arms": 3, "stage2_crops": ["m2.5", "f320", "m2.5"]},
        "_note": "6종 분포는 의도적으로 빠져 있습니다. 계열(group)까지만 말하세요.",
    },
    "screening_health": {"loaded": True, "expected_stage2_arms": 3},
    "behavior_rag_health": {"ok": True, "documents": 622, "chunks": 9911},
}


class StubSubagents:
    """`Subagents` 와 같은 자리에 꽂히는 가짜. 부른 툴 이름을 기록합니다."""

    def __init__(self, tools: list[dict[str, Any]] | None = None) -> None:
        self.tools = tools if tools is not None else load_tools()
        self.owner = {t["function"]["name"]: "stub" for t in self.tools}
        self.failed: dict[str, str] = {}
        self.called: list[str] = []

    async def call(self, name: str, arguments: dict[str, Any]) -> Any:
        self.called.append(name)
        return CANNED.get(name, {"error": f"스텁에 없는 툴: {name}"})

    def note_failures(self) -> str:
        return ""


def load_tools() -> list[dict[str, Any]]:
    if not TOOLS_SNAPSHOT.exists():
        raise FileNotFoundError(
            f"{TOOLS_SNAPSHOT} 가 없습니다. 먼저 실제 서버에서 스키마를 받아 두세요:\n"
            "    uv run python evals/run.py record")
    return json.loads(TOOLS_SNAPSHOT.read_text(encoding="utf-8"))


def save_tools(tools: list[dict[str, Any]]) -> Path:
    TOOLS_SNAPSHOT.write_text(json.dumps(tools, ensure_ascii=False, indent=2), encoding="utf-8")
    return TOOLS_SNAPSHOT
