"""에이전트 루프 — 툴을 고르고, 부르고, **게이트를 통과한 것만** 내보냅니다.

    질문(+사진) → [LLM 이 툴 고름] → MCP 서브에이전트 → 답 초안
                                                        |
                                              게이트 ─ 통과 → 그대로
                                                     └ 위반 → 한 번 고쳐 쓰게
                                                              → 또 위반 → 코드가 조립

마지막 갈래가 요점입니다. 게이트가 두 번 걸리면 **LLM 을 빼고** 서브에이전트가
돌려준 문장들로 답을 맞춥니다. 그 문장들은 저쪽 저장소가 이미 사람에게 보여도
된다고 정한 것이라, 조립한 결과는 정의상 안전합니다. 답이 조금 뻣뻣해지지만
"막혔습니다" 보다 낫고, 무엇보다 **틀린 병변 이름보다 낫습니다.**
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from dogcare.config import Settings, get_settings
from dogcare.gates import DISCLAIMER, GateReport, TurnFacts, check
from dogcare.llm import ToolCallingLLM
from dogcare.prompt import SYSTEM
from dogcare.subagents import Subagents
from dogcare.trace import ToolCall, Trace

#: 게이트가 걸렸을 때 모델에게 돌려주는 말. **무엇이 걸렸는지 그대로** 알려줍니다.
_REPAIR = """방금 답은 내보낼 수 없습니다. 걸린 것:

{violations}

같은 툴 결과로 다시 쓰세요. 툴을 새로 부르지 말고, 걸린 부분만 고치세요.
병변 이름을 쓸 수 없으면 어떤 병변인지는 이 사진만으로 판단하기 어렵다고
두세요 — 지어내지 말고 모른다고 하는 게 맞습니다."""


class Turn:
    """한 번의 상담. 결과와 기록을 같이 들고 있습니다."""

    def __init__(self, answer: str, trace: Trace, gates: GateReport, composed: bool) -> None:
        self.answer = answer
        self.trace = trace
        self.gates = gates
        #: True 면 LLM 이 아니라 코드가 조립한 답입니다.
        self.composed = composed


def _facts(question: str, image_path: str | None, calls: list[ToolCall],
           settings: Settings) -> TurnFacts:
    """게이트가 볼 사실. **툴이 실제로 돌려준 것만** 담습니다."""
    screening = None
    coverage: str | None = None
    n_sources = 0
    for c in calls:
        if c.error or not isinstance(c.result, dict):
            continue
        if c.name == "screen_skin_photo" and "verdict" in c.result:
            screening = c.result
        elif c.name == "ask_behavior_question":
            coverage = c.result.get("coverage")
            n_sources = int(c.result.get("source_count") or 0)
    return TurnFacts(
        question=question,
        had_image=bool(image_path),
        screening=screening,
        rag_coverage=coverage,
        rag_source_count=n_sources,
        # mock 은 팔이 하나뿐입니다 — 배선을 보는 모드에서 G4 가 걸릴 이유가 없습니다.
        expected_stage2_arms=None if settings.skin_mock else (settings.skin_expected_arms or None),
    )


def _attach_disclaimer(text: str, facts: TurnFacts) -> str:
    """면책은 **코드가 붙입니다.** 그래서 G2 가 원문 대조로 잴 수 있습니다."""
    if not facts.screening or DISCLAIMER in text:
        return text
    return f"{text.rstrip()}\n\n{DISCLAIMER}"


def _compose(calls: list[ToolCall], facts: TurnFacts) -> str:
    """LLM 을 빼고 **서브에이전트의 문장만으로** 답을 맞춥니다.

    피부 계약의 headline/body/action 과 RAG 의 answer 는 저쪽이 사람에게 보여도
    된다고 정한 문장 그대로입니다. 여기서 새로 쓰는 말은 잇는 말뿐입니다.
    """
    parts: list[str] = []
    s = facts.screening
    if s:
        parts += [s.get("headline", ""), s.get("body", "")]
        stage2 = s.get("stage2") or {}
        if group := stage2.get("group"):
            parts.append(f"모양만 보면 {group} 계열에 가깝습니다. 진단이 아닙니다.")
        if alert := stage2.get("alert"):
            parts.append(str(alert))
        parts.append(s.get("action", ""))
    for c in calls:
        if c.name != "ask_behavior_question" or not isinstance(c.result, dict):
            continue
        if ans := c.result.get("answer"):
            parts.append(str(ans))
    if not parts:
        parts.append("지금은 답을 드리기 어렵습니다. 잠시 뒤에 다시 물어봐 주세요.")
    return _attach_disclaimer("\n\n".join(p for p in parts if p), facts)


async def run_turn(question: str, image_path: str | None = None,
                   history: list[dict[str, str]] | None = None,
                   settings: Settings | None = None,
                   sub: Subagents | None = None) -> Turn:
    settings = settings or get_settings()
    t0 = time.perf_counter()
    trace = Trace(question=question, image_path=image_path)

    async def _go(agents: Subagents) -> Turn:
        llm = ToolCallingLLM(settings)
        user = question if not image_path else (
            f"{question}\n\n[사진이 있습니다. 경로: {image_path}]")
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM}]
        messages += list(history or [])
        messages.append({"role": "user", "content": user})

        draft = ""
        for _ in range(settings.max_tool_rounds):
            trace.rounds += 1
            msg = await llm.chat(messages, agents.tools)
            messages.append(msg)
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                draft = msg.get("content") or ""
                break
            for tc in tool_calls:
                fn = tc["function"]
                args = json.loads(fn.get("arguments") or "{}")
                call = trace.add(ToolCall(name=fn["name"], arguments=args))
                c0 = time.perf_counter()
                try:
                    call.result = await agents.call(fn["name"], args)
                except Exception as exc:
                    call.error = f"{type(exc).__name__}: {exc}"
                    call.result = {"error": call.error}
                call.elapsed_ms = round((time.perf_counter() - c0) * 1000, 1)
                messages.append({"role": "tool", "tool_call_id": tc["id"],
                                 "content": json.dumps(call.result, ensure_ascii=False)})
        else:
            # 왕복 한도를 다 썼는데 모델이 답을 안 냈습니다. 더 돌리지 않습니다 —
            # 루프가 도는 걸 모르고 토큰을 태우는 게 제일 흔한 사고입니다.
            trace.blocked = True

        facts = _facts(question, image_path, trace.calls, settings)
        answer = _attach_disclaimer(draft, facts)
        report = check(answer, facts)

        if not report.ok and draft:
            # ★ 한 번은 고쳐 쓸 기회를 줍니다. 무엇이 걸렸는지 그대로 알려주고,
            #   툴은 다시 못 부르게 합니다 (같은 사실 위에서 문장만 고치는 일입니다).
            messages.append({"role": "user",
                             "content": _REPAIR.format(violations=str(report))})
            trace.rounds += 1
            msg = await llm.chat(messages)
            answer = _attach_disclaimer(msg.get("content") or "", facts)
            report = check(answer, facts)

        composed = False
        if not report.ok or not answer.strip():
            answer = _compose(trace.calls, facts)
            composed = True
            # 조립한 답도 **똑같이 잽니다.** 안전하다고 믿고 안 재면, 조립 규칙이
            # 틀린 날 아무도 모릅니다.
            report = check(answer, facts)

        trace.answer = answer
        trace.violations = [str(v) for v in report.violations]
        trace.blocked = trace.blocked or not report.ok
        trace.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return Turn(answer, trace, report, composed)

    if sub is not None:
        return await _go(sub)
    async with Subagents(settings) as agents:
        return await _go(agents)


def save_trace(turn: Turn, settings: Settings | None = None) -> Path:
    return turn.trace.save((settings or get_settings()).trace_dir)
