"""피부 스크리닝 서브에이전트 (MCP · stdio).

    dog-skin-screening 레포를 **그 레포의 venv 안에서** 감쌉니다.

왜 별도 프로세스인가
--------------------
피부 쪽은 torch·timm 을, RAG 쪽은 bge-m3·asyncpg 를 답니다. 한 venv 에 둘을
같이 넣으면 torch 버전 하나 때문에 상담이 안 뜨는 날이 옵니다. MCP 의 stdio
전송은 **서버가 남의 프로세스**라서, 각자 자기 레포의 venv 로 띄우면 됩니다.
오케스트레이터는 mcp 하나만 알면 됩니다.

    uv run --project ../dog-skin-screening --with mcp \
        python mcp_servers/skin_screening_server.py

가중치
------
레포에 없습니다 (best.pt 가 100MB 리밋을 넘습니다). 허깅페이스에서 받습니다 —
**DAENGS 개발 서버가 쓰던 바로 그 릴리스**(gayoniee/daengs-skin-screening-release@v1)
입니다. 서버 PC 가 아니라 HF 에 있어서, 서버가 없어져도 그대로 돕니다.

⚠️ 모델에게 6종 분포를 그대로 주지 않습니다 — 아래 `_for_model` 을 보세요.
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

REPO = Path(os.environ.get("SKIN_REPO", Path(__file__).resolve().parents[2] / "deeplearning_test"))
RELEASE_REPO = os.environ.get("SKIN_RELEASE_REPO", "gayoniee/daengs-skin-screening-release")
RELEASE_REVISION = os.environ.get("SKIN_RELEASE_REVISION", "v1")
RELEASE_DIR = os.environ.get("SKIN_RELEASE_DIR") or ""
EXPECTED_ARMS = int(os.environ.get("SKIN_EXPECTED_ARMS", "3"))
MOCK = os.environ.get("SKIN_MOCK", "0") == "1"
#: 1 이면 6종 분포까지 모델에게 보냅니다. 기본은 **안 보냅니다** (아래 참조).
SHOW_DISTRIBUTION = os.environ.get("SKIN_SHOW_DISTRIBUTION", "0") == "1"

sys.path.insert(0, str(REPO))       # 저쪽이 `from src.agent import ...` 로 자기를 부릅니다

mcp = MCPServer("skin-screening")
_agent: Any = None


def _resolve_release() -> str:
    if RELEASE_DIR:
        return RELEASE_DIR
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=RELEASE_REPO, revision=RELEASE_REVISION,
                             token=os.environ.get("HF_TOKEN") or None)


def _load() -> Any:
    """가중치는 **첫 요청 때** 올립니다. import 만으로 1.6GB 를 물면 안 됩니다."""
    global _agent
    if _agent is not None:
        return _agent
    if MOCK:
        from src.agent import MockAgent

        _agent = MockAgent()
        return _agent

    from src.agent import ScreeningAgent

    root = _resolve_release()
    agent = ScreeningAgent.from_release(root)

    # ★ 팔 수를 **여기서** 셉니다. 저쪽 배포에서 앙상블이 조용히 1팔로 줄었던
    #   사고가 있어서(2026-09-07), 줄어든 채로 뜨는 걸 기동 때 막습니다.
    #   게이트 G4 는 응답마다 한 번 더 보는 두 번째 그물입니다.
    got = len(getattr(agent, "arms2", []) or [])
    if EXPECTED_ARMS and got != EXPECTED_ARMS:
        raise RuntimeError(
            f"2단계 팔이 {got}개입니다 — {EXPECTED_ARMS}개를 기대했습니다. "
            f"릴리스: {root}\n"
            "릴리스가 덜 받아졌거나 revision 이 다릅니다. "
            "SKIN_EXPECTED_ARMS 를 낮춰서 넘기지 마세요 — 성능만 조용히 떨어집니다.")
    _agent = agent
    return _agent


#: 모델에게 주지 않는 자리. **6종 이름이 실려 오는 곳 전부**입니다.
#:   distribution  6종 확률 분포 (앱은 안 그리고 콘솔이 씁니다)
#:   labels        계열마다 붙는 용어 풀이 — 내용이 곧 6종 한글 이름입니다
#:                 ("비듬·각질·상피성잔고리·태선화·과다색소침착")
_HIDDEN_FROM_MODEL = ("distribution", "labels")


def _strip(obj):
    """`labels` 같은 키를 **중첩 구조 어디서든** 떼어냅니다."""
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in _HIDDEN_FROM_MODEL}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


def _for_model(contract: dict) -> dict:
    """모델에게 건넬 판정 뷰. **6종 이름이 실린 자리를 전부 뺍니다.**

    게이트 G1 이 6종 이름을 문장에서 막지만, 그건 마지막 그물입니다. 첫 줄은
    **애초에 안 보여주는 것**입니다 — 이름을 쥐여주고 쓰지 말라고 부탁하는 건
    안 주는 것보다 비싸고 덜 확실합니다.

    처음에는 `distribution` 만 뺐는데, **`labels` 로 고스란히 새고 있었습니다.**
    저쪽은 계열 네 줄 **전부에** 용어 풀이를 같은 방식으로 실어서 앱 "자세히
    보기" 에 둡니다 — 사람에게는 단정이 아니라 풀이입니다. 그런데 산문을 쓰는
    LLM 은 그 뉘앙스를 못 지키고 하나를 고릅니다.

    모델이 볼 것은 계열 **이름과 확률**까지입니다 (`솟아오른 변화` 같은
    보호자 말). 전체 계약은 오케스트레이터의 trace 에 그대로 남습니다 —
    숨기는 게 아니라 **모델에게만 안 주는** 것입니다.
    """
    if SHOW_DISTRIBUTION:
        return contract
    view = _strip(contract)
    view["_note"] = ("6종 병변 이름은 의도적으로 빠져 있습니다. 이름을 말하면 안 됩니다 — "
                     "holdout 에서 1등 이름이 56.6% 틀렸습니다. "
                     "stage2.group.text 를 그대로 쓰세요.")
    return view


@mcp.tool()
def screen_skin_photo(image_path: str, guide_box: list[float] | None = None) -> dict:
    """반려견 피부 사진 한 장을 스크리닝합니다. 진단이 아니라 "병원에 가볼 만한가" 까지입니다.

    Args:
        image_path: 사진 파일 경로.
        guide_box: 촬영 가이드 프레임 [x, y, w, h] (0~1 정규화, 원본 기준).
            병변에 맞춰 준 네모입니다. 없으면 화면 중앙으로 물러섭니다 —
            2단계는 네모 **크기**를 쓰기 때문에 없으면 학습과 어긋납니다.

    Returns:
        판정 계약. verdict 는 normal / abnormal / retake.
        **병변 6종 이름은 들어 있지 않습니다.** 계열(group)까지만 말하세요.
    """
    p = Path(image_path).expanduser()
    if not p.exists():
        return {"error": f"사진을 찾을 수 없습니다: {p}"}
    return _for_model(_load().screen(p, box=guide_box))


@mcp.tool()
def screening_health() -> dict:
    """어떤 가중치를 물고 있나. 앙상블 팔 수와 임계값을 돌려줍니다."""
    try:
        agent = _load()
    except Exception as exc:
        return {"loaded": False, "error": f"{type(exc).__name__}: {exc}",
                "release_repo": RELEASE_REPO, "release_revision": RELEASE_REVISION}
    info = agent.describe() if hasattr(agent, "describe") else {}
    return {"loaded": True, "mock": MOCK, "expected_stage2_arms": EXPECTED_ARMS,
            "release_repo": None if RELEASE_DIR else RELEASE_REPO,
            "release_revision": None if RELEASE_DIR else RELEASE_REVISION, **info}


if __name__ == "__main__":
    mcp.run()
