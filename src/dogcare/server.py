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

from dogcare.config import ROOT, Settings, get_settings
from dogcare.loop import run_turn, save_trace
from dogcare.subagents import Subagents

STATIC = Path(__file__).resolve().parent / "static"
UPLOADS = ROOT / "uploads"
MAX_UPLOAD = 12 * 1024 * 1024        # 휴대폰 사진 한 장이면 충분합니다
MAX_QUESTION = 1_000                 # 저쪽 SkinPayload.question 과 같은 상한
KEEP_UPLOADS = os.environ.get("KEEP_UPLOADS", "0") == "1"
SESSION_TTL_S = 30 * 60
SESSION_MAX = 200
RATE_PER_MIN = int(os.environ.get("RATE_PER_MIN", "10"))
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "2"))

_JPEG = bytes([0xFF, 0xD8, 0xFF])
_PNG = bytes([0x89]) + b"PNG"


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

    def __init__(self) -> None:
        self.q: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self.result: dict[str, Any] | None = None
        self.task: asyncio.Task[None] | None = None


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
                           "rate_per_min": RATE_PER_MIN, "max_concurrent": MAX_CONCURRENT}}

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
        ip = request.client.host if request.client else "?"
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
        job = _Job()
        job_id = uuid.uuid4().hex[:8]
        jobs[job_id] = job

        def emit(kind: str, info: dict[str, Any]) -> None:
            job.q.put_nowait({"kind": kind, **info})

        async def run() -> None:
            try:
                async with gate:                    # 동시 실행 상한
                    turn = await run_turn(question, image_path=image_path, guide_box=guide,
                                          history=list(history),
                                          prior_screening=state["screening"],
                                          settings=settings, sub=agents, on_event=emit)
                trace_path = save_trace(turn, settings)
                history.append({"role": "user", "content": question})
                history.append({"role": "assistant", "content": turn.answer})
                state["screening"] = _last_screening(turn) or state["screening"]
                job.result = _summarize(turn, trace_path, sid)
            except Exception as exc:
                job.result = {"error": f"{type(exc).__name__}: {exc}"}
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
    uvicorn.run(build_app(settings), host=host, port=port, log_level="warning")
