"""웹 UI 서버 — `dogcare serve`.

    uv run dogcare serve            # http://127.0.0.1:8765
    uv run dogcare --demo serve     # 스텁 서브에이전트로

서브에이전트는 **기동 때 한 번** 열어 두고 요청마다 재사용합니다. 요청마다 열면
MCP 프로세스 둘이 매번 뜨고, 진짜 모드에서는 bge-m3 2.3GB 와 가중치 1.2GB 를
매번 올립니다.

진행 표시는 SSE 입니다. 게이트 구조상 **토큰 스트리밍은 안 됩니다** — 답을 다
받아야 검사할 수 있고, 검사에 걸리면 그 답은 안 나갑니다. 그래서 글자 대신
단계("툴 부르는 중 → 게이트 → 고쳐 쓰는 중")를 흘립니다.

⚠️ 로컬 데모 서버입니다. 인증·업로드 크기 제한 외의 보호가 없습니다.
   사진은 `uploads/` 에 남습니다 (gitignore).
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
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


class _Job:
    """한 요청의 진행 큐. SSE 가 여기서 꺼내 흘립니다."""

    def __init__(self) -> None:
        self.q: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self.result: dict[str, Any] | None = None
        self.task: asyncio.Task[None] | None = None


def build_app(settings: Settings) -> FastAPI:
    agents_box: dict[str, Subagents] = {}
    jobs: dict[str, _Job] = {}
    #: 세션별 대화 — 멀티턴용. 메모리라 서버가 내려가면 사라집니다.
    sessions: dict[str, list[dict[str, str]]] = {}

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
                "tools": sorted(agents.owner) if agents else []}

    @app.post("/api/ask")
    async def ask(request: Request,
                  question: str = Form(...),
                  session_id: str = Form(""),
                  box: str = Form(""),
                  image: UploadFile | None = File(None)) -> JSONResponse:  # noqa: B008 — FastAPI 관용구
        agents = agents_box.get("agents")
        if agents is None:
            raise HTTPException(503, "서브에이전트가 아직 안 떴습니다")
        image_path: str | None = None
        if image is not None and image.filename:
            data = await image.read()
            if len(data) > MAX_UPLOAD:
                raise HTTPException(413, "사진이 너무 큽니다 (12MB 까지)")
            UPLOADS.mkdir(exist_ok=True)
            suffix = Path(image.filename).suffix.lower() or ".jpg"
            p = UPLOADS / f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}{suffix}"
            p.write_bytes(data)
            image_path = str(p)
        guide = None
        if box:
            try:
                guide = [float(x) for x in box.split(",")]
                assert len(guide) == 4 and all(0 <= v <= 1 for v in guide)
            except (ValueError, AssertionError):
                raise HTTPException(400, "box 는 0~1 사이 네 숫자입니다") from None

        sid = session_id or uuid.uuid4().hex[:8]
        history = sessions.setdefault(sid, [])
        job = _Job()
        job_id = uuid.uuid4().hex[:8]
        jobs[job_id] = job

        def emit(kind: str, info: dict[str, Any]) -> None:
            job.q.put_nowait({"kind": kind, **info})

        async def run() -> None:
            try:
                turn = await run_turn(question, image_path=image_path, guide_box=guide,
                                      history=list(history), settings=settings,
                                      sub=agents, on_event=emit)
                trace_path = save_trace(turn, settings)
                history.append({"role": "user", "content": question})
                history.append({"role": "assistant", "content": turn.answer})
                job.result = _summarize(turn, trace_path, sid)
            except Exception as exc:
                job.result = {"error": f"{type(exc).__name__}: {exc}"}
            finally:
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


def _summarize(turn: Any, trace_path: Path, sid: str) -> dict[str, Any]:
    """화면이 그릴 만큼만. 전체는 trace 파일에 있습니다."""
    screening = None
    for c in turn.trace.calls:
        if c.name == "screen_skin_photo" and isinstance(c.result, dict) and "verdict" in c.result:
            s2 = c.result.get("stage2") or {}
            g = s2.get("group") or {}
            screening = {"verdict": c.result.get("verdict"),
                         "abnormal_percent": (c.result.get("stage1") or {}).get("abnormal_percent"),
                         "group": g.get("name") if isinstance(g, dict) else g,
                         "group_percent": g.get("percent") if isinstance(g, dict) else None,
                         "groups": s2.get("groups") or [],
                         "arms": (c.result.get("meta") or {}).get("stage2_arms"),
                         "demo": bool(c.result.get("_demo"))}
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
        "trace": trace_path.name,
    }


def serve(host: str = "127.0.0.1", port: int = 8765, demo: bool = False) -> None:
    import uvicorn

    settings = get_settings().with_demo(demo)
    uvicorn.run(build_app(settings), host=host, port=port, log_level="warning")
