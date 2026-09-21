"""행동 상담 RAG 서브에이전트 (MCP · stdio).

    dog-behavior-rag 레포를 **그 레포의 venv 안에서** 감쌉니다.

    uv run --project ../dog-behavior-rag --with mcp \
        python mcp_servers/behavior_rag_server.py

여기서 답을 **다시 만들지 않습니다.**
-------------------------------------
저쪽 RagService 가 검색·근거선별·생성·폼 정리까지 이미 합니다. 그 파이프라인의
값어치는 대부분 **포기할 줄 아는 것**에 있습니다 — 근거가 없으면 없다고 하고,
원인이 여러 갈래면 되묻습니다 (범위 밖 질문 거절률 0/4 → 7/7).

오케스트레이터가 청크만 받아다 자기 프롬프트로 다시 답을 쓰면 **그 거절이
사라집니다.** 그래서 이 서버는 답과 함께 `coverage` 를 그대로 넘기고, 위층은
그걸 지우지 않고 얹기만 합니다. 게이트 G3 가 그 약속을 감시합니다.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# mcp 2.x 에서 FastMCP 가 MCPServer 로 이름이 바뀌었습니다. 쓰는 API 는 같아서
# 이름만 맞춰 둡니다 — 서브레포의 venv 가 각자 해석하므로 버전이 갈릴 수 있습니다.
try:
    from mcp.server.mcpserver import MCPServer  # mcp >= 2
except ModuleNotFoundError:                             # mcp 1.x
    from mcp.server.fastmcp import FastMCP as MCPServer

REPO = Path(os.environ.get("BEHAVIOR_RAG_REPO",
                           Path(__file__).resolve().parents[2] / "dog-behavior-rag"))
sys.path.insert(0, str(REPO))

# 저쪽 설정은 **자기 레포의 .env** 를 읽습니다. 커서를 그쪽으로 옮겨야
# DATABASE_URL·LLM_API_KEY 가 잡힙니다 (pydantic-settings 가 상대경로로 찾습니다).
os.chdir(REPO)

mcp = MCPServer("behavior-rag")
_state: dict[str, Any] = {}


async def _ready() -> dict[str, Any]:
    """엔진·임베더는 **한 번만** 세웁니다. bge-m3 는 2.3GB 입니다."""
    if _state:
        return _state
    from app.core.config import get_settings
    from app.db.session import create_engine, create_session_factory
    from app.services.embeddings.registry import get_embedder

    settings = get_settings()
    engine = create_engine(settings.database_url)
    embedder = get_embedder(settings)
    await embedder.warmup()
    _state.update(settings=settings, engine=engine,
                  factory=create_session_factory(engine), embedder=embedder)
    return _state


def _build(kind: str, session):
    """FastAPI 의 Depends 배선을 프로세스 밖에서 그대로 재현합니다.

    저쪽 `app/api/deps.py` 와 **같은 인자**로 세웁니다. 여기서 한 줄이라도
    달라지면 API 로 부를 때와 MCP 로 부를 때 답이 갈라집니다 — 그러면 저쪽
    평가 숫자(수작업 34문항·자동 581문항)가 이쪽에 대해서는 거짓이 됩니다.
    """
    from app.api.deps import factcheck_service, rag_service

    fn = {"rag": rag_service, "factcheck": factcheck_service}[kind]
    return fn(_state["settings"], session, _state["embedder"])


@mcp.tool()
async def ask_behavior_question(question: str,
                                history: list[dict] | None = None,
                                top_k: int | None = None) -> dict:
    """반려견 문제행동·훈련 상담. **출처가 확인된 자료 안에서만** 답합니다.

    자료에 없는 주제면 답을 지어내지 않고 coverage="none" 으로 돌아옵니다.
    원인이 여러 갈래인 질문이면 coverage="needs_detail" 로 되묻습니다.

    Args:
        question: 보호자의 질문 (한국어 구어 그대로 주세요 — 재작성은 저쪽이 합니다).
        history: 이전 대화. [{"role": "user"|"assistant", "content": "..."}]
        top_k: 근거 개수. 기본 5.

    Returns:
        answer 는 [자료 N] 으로 인용합니다. **그 N 은 sources 의 순번입니다 —
        sources 개수를 넘는 번호를 새로 붙이지 마세요.**
    """
    from app.schemas.chat import Turn

    st = await _ready()
    turns = [Turn(**t) for t in (history or [])]
    async with st["factory"]() as session:
        res = await _build("rag", session).answer(question, top_k=top_k, history=turns)
    return {
        "answer": res.answer,
        # full | partial | none | needs_detail — 위층이 이 값을 지우지 않습니다.
        "coverage": res.coverage,
        "coverage_note": res.coverage_note,
        "source_count": len(res.sources),
        "sources": [{"n": i, "title": s.document_title, "source": s.source,
                     "score": round(s.score, 4), "excerpt": s.content[:400]}
                    for i, s in enumerate(res.sources, 1)],
        "latency_ms": res.latency_ms,
    }


@mcp.tool()
async def factcheck_claim(text: str, top_k: int | None = None) -> dict:
    """인터넷에서 본 훈련 조언을 코퍼스에 대고 검증합니다.

    주장별로 근거 있음 / 자료와 배치 / 자료 없음을 판정합니다.
    지배이론("서열을 잡아라")처럼 AVSAB 가 공식 반박한 조언이 여기서 걸립니다.
    """
    st = await _ready()
    async with st["factory"]() as session:
        res = await _build("factcheck", session).check(text, top_k=top_k)
    return res.model_dump(mode="json")


@mcp.tool()
async def behavior_rag_health() -> dict:
    """코퍼스가 적재돼 있나. 청크 수와 임베딩 모델을 돌려줍니다."""
    from sqlalchemy import text as sql

    try:
        st = await _ready()
        async with st["factory"]() as session:
            n_doc = (await session.execute(sql("select count(*) from documents"))).scalar()
            n_chunk = (await session.execute(sql("select count(*) from chunks"))).scalar()
        return {"ok": True, "documents": n_doc, "chunks": n_chunk,
                "embedder": st["embedder"].name, "dim": st["embedder"].dimension,
                "llm": st["settings"].llm_model, "repo": str(REPO)}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "repo": str(REPO),
                "hint": "코퍼스가 적재돼 있나요? 저쪽 docs/guide.md 의 적재 절차를 보세요."}


if __name__ == "__main__":
    mcp.run()
