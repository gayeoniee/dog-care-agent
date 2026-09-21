"""루프 테스트 — **게이트가 눈을 감는 일**을 막는 쪽에 무게를 둡니다.

한 번 실제로 그랬습니다. MCP 서버가 판정 JSON 을 structured_content 가 아니라
**글자로** 보냈는데, 그러면 `_facts` 가 dict 가 아니라고 흘려보내서
`facts.screening` 이 None 이 됩니다. G1·G2·G4 가 전부 꺼진 채로 답이 나갔고,
답은 멀쩡해 보여서 아무도 몰랐습니다.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from dogcare.config import get_settings
from dogcare.loop import _facts, run_turn
from dogcare.trace import ToolCall

ABNORMAL_JSON = {
    "contract_version": "1.0",
    "verdict": "abnormal",
    "headline": "피부에 이상 소견이 보입니다.",
    "body": "이 사진만으로 정확하게 알 수 없습니다.",
    "action": "수의사 진료를 받아보시기를 권합니다.",
    "stage1": {"abnormal_prob": 0.71, "threshold": 0.1466, "calibrated": True},
    "stage2": {"shown": True, "groups": {}, "group": None, "alert": None},
    "disclaimer": "이 결과는 수의학적 진단이 아니며, 수의사의 진료를 대체하지 않습니다. "
                  "참고용 스크리닝 정보로만 사용해 주세요.",
    "meta": {"stage2_arms": 3},
}


def test_판정이_dict면_게이트가_본다():
    f = _facts("사진 봐주세요", "a.jpg",
               [ToolCall(name="screen_skin_photo", arguments={}, result=ABNORMAL_JSON)],
               get_settings())
    assert f.screening is not None and f.expected_stage2_arms == 3


def test_판정이_글자로_오면_게이트가_눈을_감는다():
    """★ 이게 실제로 났던 고장입니다. 아래 test_서버가_글자를_보내도 가 짝입니다."""
    f = _facts("사진 봐주세요", "a.jpg",
               [ToolCall(name="screen_skin_photo", arguments={},
                         result=json.dumps(ABNORMAL_JSON, ensure_ascii=False))],
               get_settings())
    assert f.screening is None          # 이 상태를 만들지 않는 게 subagents.call 의 일


class _FakeSession:
    """글자만 돌려주는 서버 — structured_content 를 안 채우는 구현을 흉내냅니다."""

    def __init__(self, payload: Any) -> None:
        self._payload = payload

    async def call_tool(self, name: str, arguments: dict) -> Any:
        text = self._payload if isinstance(self._payload, str) else json.dumps(self._payload)

        class _C:
            def __init__(self, t): self.text = t

        class _R:
            def __init__(self, t):
                self.is_error = False
                self.structured_content = None
                self.content = [_C(t)]

        return _R(text)


async def test_서버가_글자를_보내도_dict로_돌려준다():
    from dogcare.subagents import Subagents

    sub = Subagents(get_settings())
    sub.owner["screen_skin_photo"] = "skin"
    sub.sessions["skin"] = _FakeSession(ABNORMAL_JSON)         # type: ignore[assignment]
    got = await sub.call("screen_skin_photo", {})
    assert isinstance(got, dict) and got["verdict"] == "abnormal"


async def test_JSON이_아닌_글자는_그대로_둔다():
    from dogcare.subagents import Subagents

    sub = Subagents(get_settings())
    sub.owner["t"] = "x"
    sub.sessions["x"] = _FakeSession("그냥 문장입니다")            # type: ignore[assignment]
    assert await sub.call("t", {}) == "그냥 문장입니다"


# ── 사진 경로와 가이드 프레임은 모델이 정하지 않는다 ──────────
class _FakeAgents:
    """툴 인자를 기록만 하는 가짜. 모델이 무엇을 보냈든 코드가 덮어썼는지 본다."""

    def __init__(self) -> None:
        self.tools: list[dict] = [{"type": "function", "function": {
            "name": "screen_skin_photo", "description": "", "parameters": {}}}]
        self.owner = {"screen_skin_photo": "skin"}
        self.failed: dict[str, str] = {}
        self.seen: list[dict] = []

    async def call(self, name: str, arguments: dict) -> Any:
        self.seen.append(arguments)
        return ABNORMAL_JSON

    def note_failures(self) -> str:
        return ""


class _FakeLLM:
    """첫 턴에 **틀린 경로와 지어낸 네모**로 툴을 부르고, 다음 턴에 답합니다."""

    def __init__(self) -> None:
        self.n = 0

    async def chat(self, messages, tools=None):
        self.n += 1
        if self.n == 1:
            return {"role": "assistant", "content": None, "tool_calls": [{
                "id": "1", "type": "function",
                "function": {"name": "screen_skin_photo",
                             "arguments": json.dumps({"image_path": "엉뚱한/경로.jpg",
                                                      "guide_box": [0.9, 0.9, 0.05, 0.05]})}}]}
        return {"role": "assistant",
                "content": "피부에 이상 소견이 보입니다. 수의사 진료를 받아보시기를 권합니다."}


async def test_모델이_경로와_네모를_지어내도_코드가_덮어쓴다(monkeypatch):
    """2단계는 **네모의 크기**를 쓴다 — 지어낸 네모는 조용히 판정을 바꾼다."""
    import dogcare.loop as loop

    monkeypatch.setattr(loop, "ToolCallingLLM", lambda settings: _FakeLLM())
    agents = _FakeAgents()
    turn = await run_turn("봐주세요", image_path="진짜/사진.jpg", guide_box=[0.1, 0.5, 0.5, 0.3],
                          settings=get_settings(), sub=agents)          # type: ignore[arg-type]
    assert agents.seen == [{"image_path": "진짜/사진.jpg", "guide_box": [0.1, 0.5, 0.5, 0.3]}]
    # 면책은 **코드가** 붙인다 — 모델이 안 썼는데도 붙어 있어야 한다.
    assert ABNORMAL_JSON["disclaimer"] in turn.answer
    assert turn.gates.ok, str(turn.gates)


@pytest.mark.parametrize("bad", ["0.1,0.2", "1.5,0,0.2,0.2", "a,b,c,d"])
def test_잘못된_box는_거부한다(bad):
    from dogcare.cli import _box

    with pytest.raises(SystemExit):
        _box(bad)
