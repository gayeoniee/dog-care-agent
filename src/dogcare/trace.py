"""실행 기록 — **무엇을 보고 그렇게 답했는지**가 남아야 합니다.

에이전트 루프의 곤란한 점은 같은 질문에 매번 다르게 도는 것입니다. 답만 남기면
왜 그랬는지 못 봅니다. 여기는 **툴을 무엇으로 불렀고 무엇이 돌아왔는지**를
그대로 적습니다 — 게이트가 본 것과 같은 것을 사람도 보게.

⚠️ 보호자의 질문과 사진 경로가 들어갑니다. `traces/` 는 .gitignore 에 있습니다.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    result: Any = None
    error: str | None = None
    elapsed_ms: float = 0.0


@dataclass
class Trace:
    question: str
    image_path: str | None = None
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started_at: float = field(default_factory=time.time)
    rounds: int = 0
    calls: list[ToolCall] = field(default_factory=list)
    answer: str = ""
    #: 최종 답의 게이트 위반. **비어 있는 게 통과입니다.**
    violations: list[str] = field(default_factory=list)
    #: **첫 초안**이 걸린 위반. 고쳐 쓰거나 조립하면 `violations` 는 비지만 이건 남습니다 —
    #: "게이트가 얼마나 자주 걸리나" 는 이 칸에서 셉니다. 최종만 보면 게이트가 늘 한가해 보입니다.
    first_pass_violations: list[str] = field(default_factory=list)
    #: 고쳐 쓰기로 통과했나.
    repaired: bool = False
    #: LLM 을 빼고 코드가 조립했나.
    composed: bool = False
    blocked: bool = False
    #: 이 턴에 **붙지 않은** 서브에이전트와 그 이유. 하나가 죽어도 나머지로 답하는데,
    #: 무엇이 빠진 채 답했는지는 기록과 화면에 적혀야 한다.
    subagent_failures: dict[str, str] = field(default_factory=dict)
    #: 이 턴이 LLM 에 쓴 토큰. 비용은 모델 단가 × 이것 — 단가는 여기 안 둔다(바뀐다).
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_ms: float = 0.0

    def add(self, call: ToolCall) -> ToolCall:
        self.calls.append(call)
        return call

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        p = directory / f"{time.strftime('%Y%m%d-%H%M%S')}-{self.run_id}.json"
        p.write_text(json.dumps(self.scrubbed(), ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    def scrubbed(self) -> dict[str, Any]:
        """기록용 dict — **사진 경로는 파일 이름만.**

        절대경로는 누가 어디서 돌렸는지 말해주는 정보고, 기록은 공개 뷰어에 올라갈 수
        있다. 사이트 빌더가 지우던 걸 처음부터 안 남기게 했다.
        """
        d = asdict(self)
        if d.get("image_path"):
            d["image_path"] = Path(d["image_path"]).name
        for c in d.get("calls") or []:
            a = c.get("arguments") or {}
            if isinstance(a, dict) and a.get("image_path"):
                a["image_path"] = Path(str(a["image_path"])).name
        return d
