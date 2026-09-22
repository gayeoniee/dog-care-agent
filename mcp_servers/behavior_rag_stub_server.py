"""행동 상담 RAG **스텁** (MCP · stdio) — 데모 모드 전용.

진짜 서브에이전트(`behavior_rag_server.py`)는 pgvector · bge-m3 · 코퍼스 적재가
필요하고, 코퍼스는 재배포할 수 없다(보듬TV 자막). 그래서 남이 클론해서 3분 안에
뜨게 하려면 **툴 모양만 같은 가짜**가 필요하다.

이 스텁이 지키는 것은 **응답 계약**이다 — `answer` · `coverage` · `source_count`
· `sources[].n`. 오케스트레이터와 게이트는 이 계약만 본다. 내용은 몇 문장짜리
고정 답이고, 모든 출처에 "데모" 라고 적혀 있다. **실제 코퍼스가 아니다.**

    uv run python mcp_servers/behavior_rag_stub_server.py
"""

from __future__ import annotations

try:
    from mcp.server.mcpserver import MCPServer  # mcp >= 2
except ModuleNotFoundError:                             # mcp 1.x
    from mcp.server.fastmcp import FastMCP as MCPServer

mcp = MCPServer("behavior-rag-stub")

_DEMO = "데모 자료 — 실제 코퍼스가 아닙니다"

#: 키워드 → (답, 출처 제목들). 답은 저쪽 폼(진단 → 단계 → 주의점)을 흉내 낸다.
_CANNED: list[tuple[tuple[str, ...], str, list[str]]] = [
    (("줄", "당겨", "당기"),
     "진단: 줄을 당기면 앞으로 갈 수 있다는 것을 학습한 행동입니다. [자료 1]\n\n"
     "이렇게 해보세요\n1. 줄이 팽팽해지는 순간 멈춰 서세요.\n"
     "2. 줄이 느슨해질 때만 다시 걷습니다. [자료 2]\n"
     "3. 짧은 구간부터 매일 반복하세요.\n\n"
     "주의점: 끌려가면 당기는 것이 통한다는 학습이 강화됩니다.",
     ["Loose-leash walking guide", "산책 줄 당김 교정"]),
    (("짖", "초인종"),
     "진단: 소리 자극에 대한 경계 반응입니다. [자료 1]\n\n"
     "이렇게 해보세요\n1. 초인종 소리를 작게 틀어 두고 간식을 줍니다.\n"
     "2. 반응이 없을 때만 소리를 조금씩 키웁니다. [자료 2]\n\n"
     "주의점: 짖는 중에 간식을 주면 짖음이 강화됩니다.",
     ["Desensitization and counterconditioning", "초인종 짖음"]),
    (("혼자", "분리", "울어"),
     "원인을 좁히려면 알려주세요.\n1. 나가자마자 우나요, 한참 뒤에 우나요?\n"
     "2. 집에 있을 때도 보호자를 계속 따라다니나요?",
     []),
    (("깨물", "물어", "이빨"),
     "진단: 어린 개의 입질은 놀이·탐색 행동입니다. [자료 1]\n\n"
     "이렇게 해보세요\n1. 이빨이 닿는 순간 놀이를 멈추고 등을 돌립니다.\n"
     "2. 씹어도 되는 장난감으로 바꿔 줍니다.\n\n"
     "주의점: 손으로 밀치거나 소리치면 놀이로 받아들입니다.",
     ["Puppy mouthing and play biting"]),
    (("똥", "변을 먹", "식분"),
     "진단: 식분증은 원인이 여러 갈래입니다. [자료 1]\n\n"
     "이렇게 해보세요\n1. 배변 직후 바로 치웁니다.\n"
     "2. 사료 양과 급여 횟수를 확인합니다. [자료 2]\n\n"
     "주의점: 혼내면 배변을 숨기는 쪽으로 갑니다.",
     ["Coprophagia in dogs", "식분증 관리"]),
]


def _lookup(question: str) -> tuple[str, str, list[str]]:
    for keys, answer, titles in _CANNED:
        if any(k in question for k in keys):
            coverage = "needs_detail" if not titles else "full"
            return answer, coverage, titles
    return "참고할 자료가 없습니다. 이 주제는 다루고 있지 않습니다.", "none", []


@mcp.tool()
async def ask_behavior_question(question: str,
                                history: list[dict] | None = None,
                                top_k: int | None = None) -> dict:
    """반려견 문제행동·훈련 상담 (데모 스텁). 출처가 확인된 자료 안에서만 답합니다.

    자료에 없는 주제면 coverage="none", 원인이 여러 갈래면 "needs_detail".
    answer 는 [자료 N] 으로 인용하며 N 은 sources 의 순번입니다.
    """
    answer, coverage, titles = _lookup(question)
    return {
        "answer": answer,
        "coverage": coverage,
        "coverage_note": _DEMO,
        "source_count": len(titles),
        "sources": [{"n": i, "title": t, "source": f"demo://{i}", "score": 0.7,
                     "excerpt": _DEMO} for i, t in enumerate(titles, 1)],
        "latency_ms": 5,
        "_demo": True,
    }


@mcp.tool()
async def factcheck_claim(text: str, top_k: int | None = None) -> dict:
    """인터넷에서 본 훈련 조언 검증 (데모 스텁)."""
    contradicted = any(k in text for k in ("서열", "알파", "복종", "눌러", "목덜미"))
    return {
        "claims": [{"claim": text[:120],
                    "verdict": "contradicted" if contradicted else "unsupported",
                    "note": ("지배이론에 기반한 조언입니다. AVSAB 가 공식 반박했습니다."
                             if contradicted else "판단할 자료가 없습니다."),
                    "sources": []}],
        "_demo": True,
    }


@mcp.tool()
async def behavior_rag_health() -> dict:
    """스텁 상태."""
    return {"ok": True, "demo": True, "documents": len(_CANNED), "chunks": len(_CANNED),
            "note": _DEMO}


if __name__ == "__main__":
    mcp.run()
