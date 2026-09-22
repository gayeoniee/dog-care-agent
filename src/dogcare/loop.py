"""에이전트 루프 — 툴을 고르고, 부르고, **게이트를 통과한 것만** 내보냅니다.

    질문(+사진) → [LLM 이 툴 고름] → MCP 서브에이전트 → 답 초안
                                                        |
                                              게이트 ─ 통과 → 그대로
                                                     └ 위반 → 한 번 고쳐 쓰게
                                                              → 또 위반 → 코드가 조립

고쳐 쓰는 방식이 두 가지입니다.

* 보통은 **문장만** 고치게 합니다. 툴은 다시 못 부릅니다 — 같은 사실 위에서
  말을 고르는 일이라 새 사실이 필요 없습니다.
* G6(찾아보지 않고 자료가 없다고 말함)이면 **툴을 다시 열어 줍니다.** 말을
  고쳐서 될 일이 아니라 가서 찾아봐야 하는 일입니다. 찾아본 뒤에는 사실이
  달라졌으므로 _facts 를 다시 모으고 게이트를 다시 겁니다.

마지막 갈래가 요점입니다. 게이트가 두 번 걸리면 **LLM 을 빼고** 서브에이전트가
돌려준 문장들로 답을 맞춥니다. 그 문장들은 저쪽 저장소가 이미 사람에게 보여도
된다고 정한 것이라, 조립한 결과는 정의상 안전합니다. 답이 조금 뻣뻣해지지만
"막혔습니다" 보다 낫고, 무엇보다 **틀린 병변 이름보다 낫습니다.**
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dogcare.config import Settings, get_settings
from dogcare.gates import DISCLAIMER, GateReport, TurnFacts, check
from dogcare.llm import ToolCallingLLM
from dogcare.prompt import SYSTEM
from dogcare.subagents import Subagents
from dogcare.trace import ToolCall, Trace

#: 진행 이벤트 — 웹 UI 가 "지금 뭘 하는 중" 을 보여주려고 받는다.
#: (단계 이름, 부가 정보). 게이트 구조상 스트리밍은 안 되므로(다 받아야 검사한다)
#: 단계 표시가 그 자리를 대신한다.
OnEvent = Callable[[str, dict[str, Any]], None]

#: 게이트가 걸렸을 때 모델에게 돌려주는 말. **무엇이 걸렸는지 그대로** 알려줍니다.
_REPAIR = """방금 답은 내보낼 수 없습니다. 걸린 것:

{violations}

같은 툴 결과로 다시 쓰세요. 툴을 새로 부르지 말고, 걸린 부분만 고치세요.
병변 이름을 쓸 수 없으면 어떤 병변인지는 이 사진만으로 판단하기 어렵다고
두세요 — 지어내지 말고 모른다고 하는 게 맞습니다."""

#: G6 전용. 말을 고치라는 게 아니라 **찾아보라**는 말입니다.
_REPAIR_SEARCH = """방금 답은 내보낼 수 없습니다. 걸린 것:

{violations}

자료에 무엇이 있는지는 검색해 봐야 압니다. 지금 ask_behavior_question 을
불러서 실제로 찾아보세요. 그 결과 coverage 가 none 이면 그때 자료가 없다고
말하면 됩니다 — 그건 정당합니다."""


class Turn:
    """한 번의 상담. 결과와 기록을 같이 들고 있습니다."""

    def __init__(self, answer: str, trace: Trace, gates: GateReport, composed: bool) -> None:
        self.answer = answer
        self.trace = trace
        self.gates = gates
        #: True 면 LLM 이 아니라 코드가 조립한 답입니다.
        self.composed = composed


def _facts(question: str, image_path: str | None, calls: list[ToolCall],
           settings: Settings, prior_screening: dict[str, Any] | None = None) -> TurnFacts:
    """게이트가 볼 사실. **툴이 실제로 돌려준 것만** 담습니다.

    `prior_screening` 은 **앞 턴**의 판정입니다. 이번 턴에 사진이 없어도 "그거
    궤양이야?" 같은 이어 묻기는 그 판정 위에서 답하므로, 게이트도 그 판정을
    봐야 합니다 — 없으면 G1·G7 이 이어 묻기에서 눈을 감습니다. 팀 버전의
    `ScreeningHistory` 가 같은 자리입니다. 이번 턴에 새 판정이 있으면 그게 이깁니다.
    """
    screening = prior_screening
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


async def _execute(tool_calls: list[dict[str, Any]], agents: Subagents, trace: Trace,
                   messages: list[dict[str, Any]], pinned: dict[str, dict[str, Any]],
                   emit: OnEvent) -> None:
    """툴을 부르고 결과를 대화와 기록 양쪽에 남깁니다.

    `pinned` 는 **모델이 정하면 안 되는 인자**입니다. 사진 경로와 가이드 프레임은
    요청에 들어 있는 사실이지 모델이 고를 것이 아닙니다. 그대로 두면 모델이
    경로를 조금 다르게 쓰거나(그러면 파일을 못 찾습니다) 네모를 지어냅니다 —
    2단계는 **네모의 크기**를 쓰기 때문에 지어낸 네모는 조용히 판정을 바꿉니다.
    """
    for tc in tool_calls:
        fn = tc["function"]
        args = json.loads(fn.get("arguments") or "{}")
        args.update(pinned.get(fn["name"], {}))
        call = trace.add(ToolCall(name=fn["name"], arguments=args))
        emit("tool", {"name": fn["name"]})
        t0 = time.perf_counter()
        try:
            call.result = await agents.call(fn["name"], args)
        except Exception as exc:
            call.error = f"{type(exc).__name__}: {exc}"
            call.result = {"error": call.error}
        call.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        messages.append({"role": "tool", "tool_call_id": tc["id"],
                         "content": json.dumps(call.result, ensure_ascii=False)})


async def run_turn(question: str, image_path: str | None = None,
                   guide_box: list[float] | None = None,
                   history: list[dict[str, str]] | None = None,
                   prior_screening: dict[str, Any] | None = None,
                   settings: Settings | None = None,
                   sub: Subagents | None = None,
                   on_event: OnEvent | None = None) -> Turn:
    settings = settings or get_settings()
    emit: OnEvent = on_event or (lambda kind, info: None)
    t0 = time.perf_counter()
    trace = Trace(question=question, image_path=image_path)

    async def _go(agents: Subagents) -> Turn:
        llm = ToolCallingLLM(settings)
        pinned: dict[str, dict[str, Any]] = {}
        user = question
        if image_path:
            pinned["screen_skin_photo"] = {"image_path": image_path}
            note = "[사진이 있습니다. screen_skin_photo 를 부르세요"
            if guide_box:
                pinned["screen_skin_photo"]["guide_box"] = guide_box
                note += " — 가이드 프레임도 함께 주어졌습니다"
            user = f"{question}\n\n{note}. 경로와 프레임은 코드가 채웁니다.]"
        elif prior_screening:
            # 사진은 없지만 앞 턴에 판정이 있다. 모델에게는 **판정이 말한 것만** 준다 —
            # verdict 와 계열 이름. 6종 분포·labels 는 애초에 뷰에 없다.
            s2 = prior_screening.get("stage2") or {}
            g = s2.get("group")
            gname = g.get("name") if isinstance(g, dict) else g
            note = (f"[앞 턴의 피부 판정 기록: verdict={prior_screening.get('verdict')}"
                    + (f", 계열={gname}" if gname else ", 계열 없음(확신 낮음)")
                    + ". 새 사진은 없습니다 — 이 기록 위에서 답하고, 판정을 새로 지어내지 마세요.]")
            user = f"{question}\n\n{note}"
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM}]
        messages += list(history or [])
        messages.append({"role": "user", "content": user})

        draft = ""
        for _ in range(settings.max_tool_rounds):
            trace.rounds += 1
            emit("llm", {"round": trace.rounds})
            msg = await llm.chat(messages, agents.tools)
            messages.append(msg)
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                draft = msg.get("content") or ""
                break
            await _execute(tool_calls, agents, trace, messages, pinned, emit)
        else:
            # 왕복 한도를 다 썼는데 모델이 답을 안 냈습니다. 더 돌리지 않습니다 —
            # 루프가 도는 걸 모르고 토큰을 태우는 게 제일 흔한 사고입니다.
            trace.blocked = True

        facts = _facts(question, image_path, trace.calls, settings, prior_screening)
        answer = _attach_disclaimer(draft, facts)
        emit("gate", {"pass": 1})
        report = check(answer, facts)
        trace.first_pass_violations = [str(v) for v in report.violations]

        if not report.ok and draft:
            emit("repair", {"violations": [str(v) for v in report.violations]})
            # ★ 한 번은 고쳐 쓸 기회를 줍니다. 무엇이 걸렸는지 그대로 알려줍니다.
            #   보통은 툴을 못 부르게 합니다 — 같은 사실 위에서 문장만 고르는 일입니다.
            #   G6 만 다릅니다. "찾아보지 않고 자료가 없다고 말했다" 는 말이 아니라
            #   **행동**이 틀린 것이라, 말을 고쳐서는 안 되고 가서 찾아봐야 합니다.
            search = any(v.gate == "G6" for v in report.violations)
            template = _REPAIR_SEARCH if search else _REPAIR
            messages.append({"role": "user", "content": template.format(violations=str(report))})
            trace.rounds += 1
            msg = await llm.chat(messages, agents.tools if search else None)
            messages.append(msg)
            if search and (tool_calls := msg.get("tool_calls") or []):
                await _execute(tool_calls, agents, trace, messages, pinned, emit)
                trace.rounds += 1
                msg = await llm.chat(messages)
                messages.append(msg)
                # 툴을 더 불렀으니 **사실이 달라졌습니다.** 다시 모읍니다.
                facts = _facts(question, image_path, trace.calls, settings, prior_screening)
            answer = _attach_disclaimer(msg.get("content") or "", facts)
            emit("gate", {"pass": 2})
            report = check(answer, facts)
            trace.repaired = report.ok

        composed = False
        if not report.ok or not answer.strip():
            emit("compose", {"violations": [str(v) for v in report.violations]})
            answer = _compose(trace.calls, facts)
            composed = True
            # 조립한 답도 **똑같이 잽니다.** 안전하다고 믿고 안 재면, 조립 규칙이
            # 틀린 날 아무도 모릅니다.
            report = check(answer, facts)

        trace.answer = answer
        trace.subagent_failures = dict(getattr(agents, "failed", {}) or {})
        trace.composed = composed
        trace.violations = [str(v) for v in report.violations]
        trace.blocked = trace.blocked or not report.ok
        trace.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        emit("done", {"blocked": trace.blocked, "composed": composed})
        return Turn(answer, trace, report, composed)

    if sub is not None:
        return await _go(sub)
    async with Subagents(settings) as agents:
        return await _go(agents)


def save_trace(turn: Turn, settings: Settings | None = None) -> Path:
    return turn.trace.save((settings or get_settings()).trace_dir)
