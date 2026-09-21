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
    #: 게이트 위반. **비어 있는 게 통과입니다.**
    violations: list[str] = field(default_factory=list)
    blocked: bool = False
    elapsed_ms: float = 0.0

    def add(self, call: ToolCall) -> ToolCall:
        self.calls.append(call)
        return call

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        p = directory / f"{time.strftime('%Y%m%d-%H%M%S')}-{self.run_id}.json"
        p.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        return p
