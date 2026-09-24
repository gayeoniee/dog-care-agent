"""서브에이전트 호출 — 타임아웃과 죽은 프로세스 재기동.

진짜 MCP 프로세스 대신 가짜 세션을 꽂는다. 여기서 재는 건 **호출 한 번이 서버를 영영 매달지
않는가**와 **도중에 죽은 서브에이전트가 다음 호출에서 살아나는가**다.
"""

from __future__ import annotations

import asyncio
from typing import Any

import anyio

from dogcare.config import get_settings
from dogcare.subagents import Subagents


class _Result:
    def __init__(self, text: str) -> None:
        self.is_error = False
        self.structured_content = {"result": {"ok": text}}
        self.content: list[Any] = []


class _Session:
    def __init__(self, *, hang: float = 0.0, dead: bool = False, ping_ok: bool = True) -> None:
        self.hang, self.dead, self.ping_ok = hang, dead, ping_ok
        self.calls = 0

    async def call_tool(self, name: str, arguments: dict) -> Any:
        self.calls += 1
        if self.dead:
            raise anyio.ClosedResourceError
        if self.hang:
            await asyncio.sleep(self.hang)
        return _Result(name)

    async def send_ping(self) -> None:
        if not self.ping_ok:
            await asyncio.sleep(60)


def _agents(first: _Session, *later: _Session, timeout: float = 0.05) -> Subagents:
    s = get_settings()
    object.__setattr__(s, "tool_timeout", timeout)
    a = Subagents(s)
    a.owner["t"] = "k"
    a.sessions["k"] = first
    a._specs["k"] = {"key": "k"}
    queue = list(later)

    async def _connect(spec):
        if not queue:
            raise RuntimeError("no more")
        sess = queue.pop(0)
        a.sessions["k"] = sess
        return sess

    a._connect = _connect                      # type: ignore[method-assign]
    return a


async def test_툴이_느리면_그_툴만_오류로_돌아온다():
    a = _agents(_Session(hang=1.0))
    out = await a.call("t", {})
    assert "초 안에 답하지" in out["error"] and a.restarts["k"] == 0


async def test_타임아웃_뒤_ping_도_없으면_다시_띄운다(monkeypatch):
    import dogcare.subagents as sa

    monkeypatch.setattr(sa, "_PING_TIMEOUT", 0.05)
    fresh = _Session()
    a = _agents(_Session(hang=1.0, ping_ok=False), fresh)
    out = await a.call("t", {})
    assert "다시 띄웠습니다" in out["error"] and a.restarts["k"] == 1
    assert (await a.call("t", {})) == {"ok": "t"} and fresh.calls == 1


async def test_죽은_서브에이전트는_한_번_다시_띄우고_같은_호출을_이어_간다():
    fresh = _Session()
    a = _agents(_Session(dead=True), fresh)
    assert (await a.call("t", {})) == {"ok": "t"}
    assert a.restarts["k"] == 1 and "k" not in a.failed


async def test_다시_못_띄우면_failed_에_남고_오류로_답한다():
    a = _agents(_Session(dead=True))            # 대기열이 비어 재기동이 실패한다
    out = await a.call("t", {})
    assert "죽었습니다" in out["error"] or "붙어 있지" in out["error"]
    assert "재기동 실패" in a.failed["k"]
