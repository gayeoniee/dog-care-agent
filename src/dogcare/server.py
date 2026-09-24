"""웹 UI 서버 — `dogcare serve`.

    uv run dogcare serve            # http://127.0.0.1:8765
    uv run dogcare --demo serve     # 스텁 서브에이전트로

서브에이전트는 **기동 때 한 번** 열어 두고 요청마다 재사용합니다. 요청마다 열면
MCP 프로세스 둘이 매번 뜨고, 진짜 모드에서는 bge-m3 2.3GB 와 가중치 1.2GB 를
매번 올립니다.

진행 표시는 SSE 입니다. 게이트 구조상 **토큰 스트리밍은 안 됩니다** — 답을 다
받아야 검사할 수 있고, 검사에 걸리면 그 답은 안 나갑니다. 그래서 글자 대신
단계("툴 부르는 중 → 게이트 → 고쳐 쓰는 중")를 흘립니다.

위생 (공개 데모를 열려면 있어야 하는 것)
- 사진은 **판정이 끝나면 지웁니다.** 보호자 사진을 동의 없이 모으지 않는다 —
  저쪽 serve.py 의 원칙("메모리에서 처리한 뒤 버린다")과 같습니다. 디버그용 KEEP_UPLOADS=1.
- 질문 1,000자 · 사진 12MB · JPEG/PNG/WebP 만 (매직 바이트로 봅니다. 확장자는 이름일 뿐).
- 세션은 메모리라 30분 만료 · 200개 상한. 없으면 켜 둔 동안 계속 찹니다.
- IP 당 분당 RATE_PER_MIN 회 · 동시 MAX_CONCURRENT 개. 공개 데모에 LLM 키를 물리면
  누가 반복 호출해 쿼터를 태웁니다.
- 앞 대화는 마지막 HISTORY_MAX_MESSAGES 개만 LLM 에 넘깁니다. 안 자르면 긴 대화가
  턴마다 프롬프트를 키웁니다. 레이트 리밋 표와 작업 표도 요청마다 치웁니다.
- 인증은 없습니다. 인터넷에 열 때는 앞에 인증 프록시를 둡니다.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from dogcare.config import ROOT, Settings, get_settings, require_llm_key
from dogcare.loop import run_turn, save_trace
from dogcare.subagents import Subagents

STATIC = Path(__file__).resolve().parent / "static"
UPLOADS = ROOT / "uploads"
MAX_UPLOAD = 12 * 1024 * 1024        # 휴대폰 사진 한 장이면 충분합니다
MAX_QUESTION = 1_000                 # 저쪽 SkinPayload.question 과 같은 상한
KEEP_UPLOADS = os.environ.get("KEEP_UPLOADS", "0") == "1"
SESSION_TTL_S = 30 * 60
SESSION_MAX = 200
#: LLM 에 넘기는 앞 대화의 상한(메시지 수 = 턴 × 2). 세션은 30분 살아서, 그 안에 스무 턴을
#: 이어 가면 프롬프트가 턴마다 자란다 — 프롬프트 토큰이 p50 1,815 인 건 **첫 턴** 숫자다.
#: 게이트가 보는 판정(`state["screening"]`)은 따로 들고 다니므로 이 상한과 무관하다.
HISTORY_MAX_MESSAGES = int(os.environ.get("HISTORY_MAX_MESSAGES", "10"))
#: 결과를 아무도 안 받아 간 작업을 치우는 나이. SSE 를 안 열거나 도중에 끊으면 `jobs` 항목이
#: 영영 남는다 — 공개 데모에서 탭을 닫는 사람이 곧 그 경우다.
JOB_TTL_S = 5 * 60
RATE_PER_MIN = int(os.environ.get("RATE_PER_MIN", "10"))
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "2"))

_JPEG = bytes([0xFF, 0xD8, 0xFF])
_PNG = bytes([0x89]) + b"PNG"


def client_ip(request: Request) -> str:
    """레이트 리밋의 열쇠. **프록시 뒤에서는 `request.client.host` 가 전부 같은 값**이다.

    HF Spaces 는 모든 요청이 프록시 IP 로 들어온다. 그걸 그대로 쓰면 분당 6회 제한이
    "방문자 한 명당" 이 아니라 "전 세계 합쳐서" 가 된다 — 한 사람이 쓰면 다른 사람이
    429 를 본다. 프록시가 붙이는 `X-Forwarded-For` 의 첫 값(원래 클라이언트)을 쓴다.
    로컬에서는 그 헤더가 없으니 client.host 로 돌아간다.

    ⚠️ 첫 값은 **클라이언트가 지어낼 수 있다** — 프록시가 헤더를 덮어쓰지 않고 뒤에 붙이는
    구성이면, 요청마다 다른 값을 실어 분당 제한을 피할 수 있다. HF 프록시가 어느 쪽인지는
    바깥에서 확인할 길이 없어 그대로 둔다. 이 제한은 예의 수준이고, 진짜 상한은 동시 실행
    `MAX_CONCURRENT` 와 무료 티어의 하루 500회다 — 우회해서 얻는 것이 남의 쿼터뿐이다.
    """
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip() or "?"
    return request.client.host if request.client else "?"


def user_facing_error(exc: BaseException) -> str:
    """보호자 화면에 보일 오류 문장.

    라이브 데모에서 무료 티어 하루 한도(모델당 500회)가 끝나자 화면에
    `LLMError: HTTP 429 — [{ "error": { "code": 429, ...` 가 그대로 찍혔다.
    보호자가 읽을 문장이 아니다. 한도·과부하·시간초과는 사람 말로 바꾸고,
    나머지는 종류만 남긴다 (자세한 건 서버 로그와 트레이스에 있다).
    """
    text = f"{type(exc).__name__}: {exc}"
    if "HTTP 429" in text:
        return ("지금은 무료 이용 한도에 걸려 답을 만들 수 없습니다. "
                "잠시 뒤(한도가 하루치면 다음 날) 다시 시도해 주세요.")
    if "HTTP 503" in text or "HTTP 502" in text:
        return "모델 서버가 붐빕니다. 잠시 뒤 다시 시도해 주세요."
    if "Timeout" in type(exc).__name__ or "timed out" in text.lower():
        return "답을 만드는 데 너무 오래 걸려 멈췄습니다. 다시 시도해 주세요."
    return text.splitlines()[0][:200]


def sniff_image(data: bytes) -> str | None:
    """매직 바이트로 사진인지 봅니다. 확장자는 이름일 뿐입니다."""
    if data.startswith(_JPEG):
        return "jpg"
    if data.startswith(_PNG):
        return "png"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "webp"
    return None


class _Job:
    """한 요청의 진행 큐. SSE 가 여기서 꺼내 흘립니다."""

    def __init__(self, now: float) -> None:
        self.q: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self.result: dict[str, Any] | None = None
        self.task: asyncio.Task[None] | None = None
        self.created = now


def build_app(settings: Settings) -> FastAPI:
    agents_box: dict[str, Subagents] = {}
    jobs: dict[str, _Job] = {}
    #: 세션별 상태 — 대화와 **마지막 피부 판정**. 판정을 들고 다니는 이유: "그거 궤양이야?"
    #: 같은 이어 묻기가 앞 턴 판정 위에서 답하므로, 게이트도 그 판정을 봐야 한다.
    sessions: dict[str, dict[str, Any]] = {}
    hits: dict[str, deque[float]] = defaultdict(deque)
    gate = asyncio.Semaphore(MAX_CONCURRENT)

    def _evict_sessions(now: float) -> None:
        for k in [k for k, s in sessions.items() if now - s["seen"] > SESSION_TTL_S]:
            sessions.pop(k, None)
        while len(sessions) > SESSION_MAX:          # 제일 오래 안 쓴 것부터
            sessions.pop(min(sessions, key=lambda k: sessions[k]["seen"]), None)
        # ★ 요청마다 같이 치운다 — 셋 다 "켜 둔 동안 계속 차는" 표다.
        #   hits: IP 마다 deque 가 생기고 비어도 안 없어졌다. 방문자 수만큼 자란다.
        #   jobs: 결과가 났는데 SSE 로 안 가져간 작업. 탭을 닫으면 끝까지 남는다.
        for ip in [ip for ip, q in hits.items() if not q or now - q[-1] > 60]:
            hits.pop(ip, None)
        for jid in [jid for jid, j in jobs.items() if now - j.created > JOB_TTL_S]:
            jobs.pop(jid, None)

    def _rate_ok(ip: str, now: float) -> bool:
        q = hits[ip]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE_PER_MIN:
            return False
        q.append(now)
        return True

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        async with Subagents(settings) as agents:
            agents_box["agents"] = agents
            yield
        agents_box.clear()

    app = FastAPI(title="dog-care-agent", lifespan=lifespan)
    app.state.tables = {"sessions": sessions, "hits": hits, "jobs": jobs}   # 테스트가 본다

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        agents = agents_box.get("agents")
        return {"ok": agents is not None, "demo": settings.demo, "model": settings.llm_model,
                "subagents": sorted(agents.sessions) if agents else [],
                "failed": agents.failed if agents else {},
                "tools": sorted(agents.owner) if agents else [],
                "limits": {"question_chars": MAX_QUESTION, "upload_bytes": MAX_UPLOAD,
                           "rate_per_min": RATE_PER_MIN, "max_concurrent": MAX_CONCURRENT,
                           "history_messages": HISTORY_MAX_MESSAGES}}

    @app.post("/api/ask")
    async def ask(request: Request,
                  question: str = Form(...),
                  session_id: str = Form(""),
                  box: str = Form(""),
                  image: UploadFile | None = File(None)) -> JSONResponse:  # noqa: B008 — FastAPI 관용구
        agents = agents_box.get("agents")
        if agents is None:
            raise HTTPException(503, "서브에이전트가 아직 안 떴습니다")
        now = time.time()
        ip = client_ip(request)
        if not _rate_ok(ip, now):
            raise HTTPException(429, f"분당 {RATE_PER_MIN}회까지입니다. 잠시 뒤에 다시 보내주세요")
        question = question.strip()
        if not question:
            raise HTTPException(400, "질문이 비어 있습니다")
        if len(question) > MAX_QUESTION:
            raise HTTPException(413, f"질문은 {MAX_QUESTION}자까지입니다")

        image_path: str | None = None
        if image is not None and image.filename:
            data = await image.read()
            if len(data) > MAX_UPLOAD:
                raise HTTPException(413, "사진이 너무 큽니다 (12MB 까지)")
            ext = sniff_image(data)
            if ext is None:
                raise HTTPException(415, "JPEG · PNG · WebP 사진만 받습니다")
            UPLOADS.mkdir(exist_ok=True)
            p = UPLOADS / f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}.{ext}"
            p.write_bytes(data)
            image_path = str(p)
        guide = None
        if box:
            try:
                guide = [float(x) for x in box.split(",")]
                assert len(guide) == 4 and all(0 <= v <= 1 for v in guide)
            except (ValueError, AssertionError):
                raise HTTPException(400, "box 는 0~1 사이 네 숫자입니다") from None

        _evict_sessions(now)
        sid = session_id or uuid.uuid4().hex[:8]
        state = sessions.setdefault(sid, {"history": [], "screening": None, "seen": now})
        state["seen"] = now
        history = state["history"]
        job = _Job(now)
        job_id = uuid.uuid4().hex[:8]
        jobs[job_id] = job

        def emit(kind: str, info: dict[str, Any]) -> None:
            job.q.put_nowait({"kind": kind, **info})

        async def run() -> None:
            try:
                async with gate:                    # 동시 실행 상한
                    turn = await run_turn(question, image_path=image_path, guide_box=guide,
                                          history=history[-HISTORY_MAX_MESSAGES:],
                                          prior_screening=state["screening"],
                                          settings=settings, sub=agents, on_event=emit)
                trace_path = save_trace(turn, settings)
                history.append({"role": "user", "content": question})
                history.append({"role": "assistant", "content": turn.answer})
                del history[:-HISTORY_MAX_MESSAGES]      # 저장도 같은 상한 — 세션이 자라지 않게
                state["screening"] = _last_screening(turn) or state["screening"]
                job.result = _summarize(turn, trace_path, sid)
            except Exception as exc:
                job.result = {"error": user_facing_error(exc)}
            finally:
                if image_path and not KEEP_UPLOADS:
                    Path(image_path).unlink(missing_ok=True)   # 판정 끝나면 사진은 지운다
                job.q.put_nowait(None)

        job.task = asyncio.create_task(run())      # 참조를 잡아 두지 않으면 GC 가 태스크를 지운다
        return JSONResponse({"job_id": job_id, "session_id": sid})

    @app.get("/api/events/{job_id}")
    async def events(job_id: str) -> StreamingResponse:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404)

        async def gen() -> AsyncIterator[str]:
            while True:
                item = await job.q.get()
                if item is None:
                    yield f"event: result\ndata: {json.dumps(job.result, ensure_ascii=False)}\n\n"
                    jobs.pop(job_id, None)
                    return
                yield f"event: progress\ndata: {json.dumps(item, ensure_ascii=False)}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    @app.get("/api/stats")
    async def stats() -> dict[str, Any]:
        """traces/ 집계 — 화면이 "이 서버가 지금까지 뭘 막았나" 를 보여주려고 부른다."""
        from dogcare.stats import collect

        s = collect(settings.trace_dir)
        return {"turns": s.turns, "first_pass_hit": s.first_pass_hit, "repaired": s.repaired,
                "composed": s.composed, "blocked": s.blocked,
                "gate_hits": dict(s.gate_hits), "tool_calls": dict(s.tool_calls),
                "tool_errors": dict(s.tool_errors)}

    @app.get("/api/trace/{name}")
    async def trace(name: str) -> FileResponse:
        p = settings.trace_dir / f"{Path(name).name}"
        if not p.exists() or p.suffix != ".json":
            raise HTTPException(404)
        return FileResponse(p, media_type="application/json")

    return app


def _last_screening(turn: Any) -> dict[str, Any] | None:
    for c in turn.trace.calls:
        if c.name == "screen_skin_photo" and isinstance(c.result, dict) and "verdict" in c.result:
            return c.result
    return None


def _summarize(turn: Any, trace_path: Path, sid: str) -> dict[str, Any]:
    """화면이 그릴 만큼만. 전체는 trace 파일에 있습니다."""
    screening = None
    s = _last_screening(turn)
    if s:
        s2 = s.get("stage2") or {}
        g = s2.get("group") or {}
        screening = {"verdict": s.get("verdict"),
                     "abnormal_percent": (s.get("stage1") or {}).get("abnormal_percent"),
                     "group": g.get("name") if isinstance(g, dict) else g,
                     "group_percent": g.get("percent") if isinstance(g, dict) else None,
                     "groups": s2.get("groups") or [],
                     "arms": (s.get("meta") or {}).get("stage2_arms"),
                     "demo": bool(s.get("_demo"))}
    return {
        "session_id": sid,
        "answer": turn.answer,
        "composed": turn.composed,
        "violations": turn.trace.violations,
        "blocked": turn.trace.blocked,
        "rounds": turn.trace.rounds,
        "elapsed_ms": turn.trace.elapsed_ms,
        "calls": [{"name": c.name, "elapsed_ms": c.elapsed_ms, "error": c.error}
                  for c in turn.trace.calls],
        "screening": screening,
        "subagent_failures": turn.trace.subagent_failures,
        "trace": trace_path.name,
    }


def serve(host: str = "127.0.0.1", port: int = 8765, demo: bool = False) -> None:
    import uvicorn

    settings = get_settings().with_demo(demo)
    require_llm_key(settings)
    uvicorn.run(build_app(settings), host=host, port=port, log_level="warning")
