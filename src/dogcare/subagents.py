"""서브에이전트 연결 — MCP stdio 서버 둘을 띄우고 툴 목록을 모읍니다.

왜 stdio 인가
-------------
서버가 **남의 프로세스**라서입니다. 피부 쪽은 torch·timm 을, RAG 쪽은
bge-m3·asyncpg 를 답니다. 한 venv 에 몰아넣으면 torch 버전 하나 때문에
상담이 안 뜨는 날이 옵니다. 여기서는 각자 `uv run --project <레포>` 로
자기 venv 안에서 뜨고, 오케스트레이터는 mcp 하나만 압니다.

값은 곁들여 옵니다 — 피부 모델이 죽어도 상담은 돕니다. 한 프로세스면
import 하나로 둘 다 안 뜹니다.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections import Counter
from contextlib import AsyncExitStack, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from dogcare.config import ROOT, Settings

#: 서브에이전트 프로세스가 **죽었다**는 신호. stdio 파이프가 닫히면 이 중 하나로 옵니다.
#: 툴이 "실패했다"(is_error) 는 것과 다릅니다 — 그건 살아 있는 서버의 정상 응답입니다.
_DEAD = (anyio.ClosedResourceError, anyio.BrokenResourceError, anyio.EndOfStream,
         BrokenPipeError, ConnectionError, EOFError)
#: 타임아웃 뒤 "살아는 있나" 를 묻는 시간. 이 안에 ping 도 못 받으면 죽은 것으로 봅니다.
_PING_TIMEOUT = 5.0


def _spec(settings: Settings) -> list[dict[str, Any]]:
    """띄울 서버 둘. `--with mcp` 로 **저쪽 레포를 건드리지 않고** 의존성만 얹습니다.

    데모 모드면 **이 저장소의 venv 로 스텁 둘**을 띄웁니다. 서브레포도 가중치도
    DB 도 필요 없습니다 — 계약 모양만 같은 가짜라, 오케스트레이터와 게이트는
    진짜와 똑같이 돕니다. 답 내용만 몇 문장짜리 고정값입니다.
    """
    if settings.demo:
        # ★ 스텁은 **지금 이 인터프리터**로 띄웁니다. 스텁이 필요한 건 mcp 하나뿐이고
        #   그건 이미 여기에 있습니다. uv 를 거치지 않아야 uv 가 없는 곳(HF Spaces 의
        #   Gradio 런타임)에서도 뜹니다.
        return [
            {"key": "behavior_rag", "project": ROOT, "interpreter": sys.executable,
             "script": ROOT / "mcp_servers" / "behavior_rag_stub_server.py", "env": {}},
            {"key": "skin", "project": ROOT, "interpreter": sys.executable,
             "script": ROOT / "mcp_servers" / "skin_screening_stub_server.py", "env": {}},
        ]
    return [
        {
            "key": "behavior_rag",
            "project": settings.behavior_rag_repo,
            "script": ROOT / "mcp_servers" / "behavior_rag_server.py",
            # bge-m3 를 돌리려면 sentence-transformers 가 필요합니다. 저쪽이
            # 일부러 extra 로 빼 뒀습니다 — torch 없이도 앱이 뜨게 하려고.
            "extras": ["hf"],
            "env": settings.env_for_rag(),
        },
        {
            "key": "skin",
            "project": settings.skin_repo,
            "script": ROOT / "mcp_servers" / "skin_screening_server.py",
            # mock 은 torch 가 필요 없습니다. 배선만 볼 때 수 GB 를 받지 않습니다.
            "extras": [] if settings.skin_mock else ["train"],
            "env": settings.env_for_skin(),
        },
    ]


def _sanitize(schema: dict[str, Any]) -> dict[str, Any]:
    """툴 스키마를 **OpenAI 호환 서버가 받아 주는 모양**으로 깎습니다.

    파이썬 타입힌트에서 나온 `list[float] | None` 은 JSON Schema 로 가면
    `anyOf: [{type: array}, {type: null}]` 이 됩니다. Gemini 의 OpenAI 호환
    엔드포인트는 이걸 400 으로 돌려보냅니다 — **에러 메시지에 anyOf 라는 말이
    안 나와서** 모델 이름이나 키를 의심하게 됩니다. 널 갈래를 떼고 나머지를
    올려 붙이면 의미가 같습니다 (안 주면 되는 인자라는 건 required 가 말합니다).
    """
    if not isinstance(schema, dict):
        return schema
    out = {k: v for k, v in schema.items() if k not in ("$schema", "additionalProperties")}
    if "anyOf" in out:
        alts = [a for a in out.pop("anyOf") if a.get("type") != "null"]
        if len(alts) == 1:
            out = {**_sanitize(alts[0]), **out}
        elif alts:
            out["anyOf"] = [_sanitize(a) for a in alts]
    for key in ("properties", "$defs"):
        if isinstance(out.get(key), dict):
            out[key] = {k: _sanitize(v) for k, v in out[key].items()}
    if isinstance(out.get("items"), dict):
        out["items"] = _sanitize(out["items"])
    return out


def _attr(obj: Any, *names: str, default: Any = None) -> Any:
    """mcp 1.x(camelCase) 와 2.x(snake_case) 를 같이 봅니다.

    2.x 에서 `inputSchema` → `input_schema`, `isError` → `is_error` 로 바뀌었습니다.
    서브레포의 venv 가 각자 해석하므로 버전이 갈릴 수 있어 양쪽을 봅니다.
    """
    for n in names:
        if (v := getattr(obj, n, None)) is not None:
            return v
    return default


@dataclass
class Subagents:
    """열린 MCP 세션들과, 툴 이름 → 세션 지도."""

    settings: Settings
    sessions: dict[str, ClientSession] = field(default_factory=dict)
    tools: list[dict[str, Any]] = field(default_factory=list)
    owner: dict[str, str] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)
    #: 돌다가 죽어서 다시 띄운 횟수. health 와 trace 에 적힙니다.
    restarts: Counter[str] = field(default_factory=Counter)
    _specs: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: 서브에이전트마다 자기 스택 — 하나만 닫고 다시 열 수 있어야 재기동이 됩니다.
    _stacks: dict[str, AsyncExitStack] = field(default_factory=dict)

    async def __aenter__(self) -> Subagents:
        for spec in _spec(self.settings):
            self._specs[spec["key"]] = spec
            try:
                await self._open(spec)
            except Exception as exc:
                # ★ 하나가 안 떠도 나머지로 계속합니다. 피부 모델이 없다고
                #   행동 상담까지 멈출 이유가 없습니다. 무엇이 빠졌는지는
                #   `failed` 에 남고, 답변 아래에 그대로 적힙니다.
                self.failed[spec["key"]] = f"{type(exc).__name__}: {exc}"
        return self

    async def __aexit__(self, *exc: Any) -> None:
        for key in list(self._stacks):
            await self._close(key)

    async def _close(self, key: str) -> None:
        """한 서브에이전트만 내립니다. 이미 죽은 프로세스면 닫다가 또 예외가 나는데, 삼킵니다."""
        self.sessions.pop(key, None)
        if stack := self._stacks.pop(key, None):
            with suppress(Exception):
                await stack.aclose()

    async def _connect(self, spec: dict[str, Any]) -> ClientSession:
        """프로세스를 띄우고 세션을 엽니다. 툴 등록은 안 합니다 — 재기동 때는 이것만 합니다."""
        project = Path(spec["project"])
        if not project.exists():
            raise FileNotFoundError(f"서브레포가 없습니다: {project} — .env 의 경로를 확인하세요")
        if interp := spec.get("interpreter"):
            command, args = interp, [str(spec["script"])]
        else:
            command = "uv"
            args = ["run", "--project", str(project),
                    *[a for e in spec.get("extras", []) for a in ("--extra", e)],
                    "--with", "mcp>=2,<3",
                    "python", str(spec["script"])]
        params = StdioServerParameters(
            command=command,
            args=args,
            # ★ **빈 값은 빼고 넘깁니다.** 우리 .env 의 DATABASE_URL= (빈 줄) 이
            #   자식에게 그대로 가면, 저쪽 레포가 자기 .env 를 읽기도 전에 빈
            #   문자열로 덮여서 "Could not parse SQLAlchemy URL" 로 죽습니다.
            #   빈 값은 "안 정했다" 는 뜻이지 "빈 문자열로 정했다" 가 아닙니다.
            #   VIRTUAL_ENV 도 뺍니다 — 남기면 자식 uv 가 매번 경고합니다.
            env={k: v for k, v in os.environ.items() if v and k != "VIRTUAL_ENV"}
                | {k: v for k, v in spec["env"].items() if v},
        )
        stack = AsyncExitStack()
        await stack.__aenter__()
        try:
            read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
        except BaseException:
            await stack.aclose()
            raise
        self._stacks[spec["key"]] = stack
        self.sessions[spec["key"]] = session
        return session

    async def _open(self, spec: dict[str, Any]) -> None:
        session = await self._connect(spec)
        for tool in (await session.list_tools()).tools:
            if tool.name in self.owner:
                raise RuntimeError(f"툴 이름이 겹칩니다: {tool.name}")
            self.owner[tool.name] = spec["key"]
            self.tools.append({
                "type": "function",
                "function": {"name": tool.name,
                             "description": (tool.description or "").strip(),
                             "parameters": _sanitize(
                                 _attr(tool, "input_schema", "inputSchema", default={}))},
            })

    async def _reopen(self, key: str) -> bool:
        """죽은 서브에이전트를 **한 번** 다시 띄웁니다. 못 띄우면 `failed` 에 적고 False."""
        await self._close(key)
        try:
            await self._connect(self._specs[key])
        except Exception as exc:
            self.failed[key] = f"재기동 실패 — {type(exc).__name__}: {exc}"
            return False
        self.restarts[key] += 1
        self.failed.pop(key, None)
        return True

    async def _alive(self, key: str) -> bool:
        """타임아웃 뒤에 묻습니다 — 느린 건가, 죽은 건가. ping 도 못 받으면 죽은 것입니다."""
        session = self.sessions.get(key)
        if session is None:
            return False
        try:
            await asyncio.wait_for(session.send_ping(), timeout=_PING_TIMEOUT)
            return True
        except Exception:
            return False

    async def call(self, name: str, arguments: dict[str, Any]) -> Any:
        """툴 하나를 부릅니다. **타임아웃이 있고, 죽은 서브에이전트는 한 번 다시 띄웁니다.**

        전에는 둘 다 없었다 — 진짜 모드에서 피부 모델이 멈추면 그 요청은 LLM 타임아웃과 무관하게
        영영 매달렸고(동시 상한이 2 라 두 번이면 서버 전체가 답을 못 한다), 돌다가 프로세스가 죽으면
        그 뒤 모든 호출이 오류였다. 기동 실패만 `failed` 에 남고 도중 죽음은 아무도 안 봤다.

        타임아웃은 재기동 사유가 아니다 — 느린 추론을 죽이면 안 된다. 대신 ping 으로 살아 있나
        묻고, 그것마저 안 오면 그때 다시 띄운다.
        """
        key = self.owner.get(name)
        if key is None:
            return {"error": f"그런 툴이 없습니다: {name}"}
        limit = self.settings.tool_timeout
        for attempt in (1, 2):
            session = self.sessions.get(key)
            if session is None:
                if attempt == 2 or not await self._reopen(key):
                    return {"error": f"서브에이전트 {key} 가 붙어 있지 않습니다 — "
                                     f"{self.failed.get(key, '이유 없음')}"}
                continue
            try:
                res = await asyncio.wait_for(session.call_tool(name, arguments), timeout=limit)
            except TimeoutError:
                if await self._alive(key):
                    return {"error": f"툴이 {limit:.0f}초 안에 답하지 않았습니다: {name}"}
                await self._reopen(key)          # 죽은 것 — 다음 호출을 위해 다시 띄운다
                return {"error": f"서브에이전트 {key} 가 응답하지 않아 다시 띄웠습니다: {name}"}
            except _DEAD as exc:
                if attempt == 2 or not await self._reopen(key):
                    return {"error": f"서브에이전트 {key} 가 죽었습니다 — {type(exc).__name__}"}
                continue                          # 다시 띄웠으니 한 번 더
            break
        if _attr(res, "is_error", "isError"):
            return {"error": "".join(getattr(c, "text", "") for c in res.content)}
        # dict 를 돌려주는 툴은 structured_content 로 옵니다 (mcp 1.x 는 structuredContent).
        if sc := _attr(res, "structured_content", "structuredContent"):
            return sc.get("result", sc) if isinstance(sc, dict) else sc
        text = "".join(getattr(c, "text", "") for c in res.content)
        # ★ 서버가 structured_content 를 안 채우고 **JSON 을 글자로** 보내는 일이
        #   있습니다. 그때 문자열을 그대로 위로 올리면 `_facts` 가 dict 가 아니라고
        #   흘려보내고, **게이트가 피부 턴에서 통째로 눈을 감습니다.** 답은 멀쩡해
        #   보여서 아무도 모릅니다 — 실제로 그랬습니다. 여기서 되돌려 놓습니다.
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            return text
        return parsed if isinstance(parsed, dict | list) else text

    def note_failures(self) -> str:
        if not self.failed:
            return ""
        lines = [f"  - {k}: {v}" for k, v in self.failed.items()]
        return "붙지 않은 서브에이전트가 있습니다:\n" + "\n".join(lines)


def print_failures(sub: Subagents) -> None:
    if sub.failed:
        print(sub.note_failures(), file=sys.stderr)
