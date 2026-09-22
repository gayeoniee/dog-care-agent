"""`evals/tools.json` 이 서버 코드와 어긋나지 않았나 (C2).

라우팅 평가는 실제 서버를 안 띄우고 **스냅샷 스키마**를 LLM 에 줍니다. 서버에
툴을 더하거나 이름을 바꿨는데 스냅샷을 안 갱신하면, 평가는 옛 툴 목록으로
조용히 돌고 숫자는 거짓이 됩니다. 서버 파일을 정적으로 읽어 `@mcp.tool()` 아래
함수 이름을 모으고 스냅샷과 대조합니다 — 서버를 띄울 필요가 없습니다.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVERS = ["behavior_rag_server.py", "skin_screening_server.py"]
STUBS = ["behavior_rag_stub_server.py", "skin_screening_stub_server.py"]


def _tool_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            for d in node.decorator_list:
                f = d.func if isinstance(d, ast.Call) else d
                if isinstance(f, ast.Attribute) and f.attr == "tool":
                    out.add(node.name)
    return out


def test_스냅샷과_서버의_툴_이름이_같다():
    snap = json.loads((ROOT / "evals" / "tools.json").read_text(encoding="utf-8"))
    in_snapshot = {t["function"]["name"] for t in snap}
    in_code = set().union(*(_tool_names(ROOT / "mcp_servers" / s) for s in SERVERS))
    assert in_snapshot == in_code, (
        f"어긋남 — 스냅샷에만: {sorted(in_snapshot - in_code)}, "
        f"코드에만: {sorted(in_code - in_snapshot)}. "
        "`uv run python evals/run.py record` 로 다시 받으세요.")


def test_스텁은_진짜와_툴_이름이_같다():
    """데모에서 도는 것과 진짜에서 도는 것이 같은 툴이어야 라우팅 평가가 둘 다에 대해 참이다."""
    real = set().union(*(_tool_names(ROOT / "mcp_servers" / s) for s in SERVERS))
    stub = set().union(*(_tool_names(ROOT / "mcp_servers" / s) for s in STUBS))
    assert real == stub, f"진짜에만: {sorted(real - stub)}, 스텁에만: {sorted(stub - real)}"


def test_스냅샷_스키마에_anyOf가_없다():
    """Gemini 가 anyOf 를 400 으로 돌려보낸다 — `_sanitize` 가 걷어냈는지 스냅샷에서 본다."""
    blob = (ROOT / "evals" / "tools.json").read_text(encoding="utf-8")
    assert '"anyOf"' not in blob
