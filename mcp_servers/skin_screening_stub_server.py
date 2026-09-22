"""피부 스크리닝 **스텁** (MCP · stdio) — 데모 모드 전용.

진짜 서버는 허깅페이스에서 1.2GB 를 받고 torch 를 올린다. 남이 3분 안에 화면을
보려면 **계약 모양만 같은 가짜**가 필요하다. 이 스텁은 진짜 계약의 필드를 그대로
낸다 — `verdict` · `headline` · `body` · `action` · `stage1` · `stage2.groups` ·
`stage2.group` · `disclaimer` · `meta.stage2_arms`. 게이트가 보는 것 전부다.

판정은 **사진 내용과 무관하다.** 파일 이름 해시로 정상/이상을 고른다 — 데모에서
두 갈래를 다 보여주기 위해서다. 응답에 `_demo: true` 가 박힌다.

    uv run python mcp_servers/skin_screening_stub_server.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path

try:
    from mcp.server.mcpserver import MCPServer  # mcp >= 2
except ModuleNotFoundError:                             # mcp 1.x
    from mcp.server.fastmcp import FastMCP as MCPServer

mcp = MCPServer("skin-screening-stub")

DISCLAIMER = ("이 결과는 수의학적 진단이 아니며, 수의사의 진료를 대체하지 않습니다. "
              "참고용 스크리닝 정보로만 사용해 주세요.")
_NOTE = ("6종 병변 이름은 의도적으로 빠져 있습니다. 이름을 말하면 안 됩니다 — "
         "holdout 에서 1등 이름이 56.6% 틀렸습니다. stage2.group.text 를 그대로 쓰세요.")

#: 계열 4군 — 실제 계약과 같은 이름 (2026-09-10 보호자 말).
_GROUPS = ("솟아오른 변화", "피부 표면·색·두께 변화", "벗겨지거나 패인 상처", "깊거나 단단한 혹")
_FEATURE = {
    "솟아오른 변화": "작게 솟은 돌기, 고름이 찬 돌기",
    "피부 표면·색·두께 변화": "비듬, 딱지, 두꺼워지거나 검게 변한 피부",
    "벗겨지거나 패인 상처": "까짐, 진물, 출혈, 깊게 패인 부위",
    "깊거나 단단한 혹": "만져지는 덩어리",
}


def _contract(verdict: str, abnormal: float, group: str | None, arms: int) -> dict:
    head = {"normal": "뚜렷한 이상 소견은 보이지 않습니다.",
            "abnormal": "피부에 이상 소견이 보입니다.",
            "retake": "판단이 어려운 사진입니다."}[verdict]
    body = {"normal": ("사진으로 확인할 수 있는 범위에는 한계가 있습니다. 가려워하거나, "
                       "냄새가 나거나, 계속 핥는 등 평소와 다른 행동이 있다면 결과와 무관하게 "
                       "병원에 가보시는 것을 권합니다."),
            "abnormal": "이 사진만으로 정확하게 알 수 없습니다.",
            "retake": ("이상한 부위가 잘 보이도록, 밝은 곳에서 초점을 맞춰 "
                       "다시 찍어주세요.")}[verdict]
    action = {"normal": "평소와 다른 점이 있으면 진료를 받아보세요.",
              "abnormal": "수의사 진료를 받아보시기를 권합니다.",
              "retake": "사진을 다시 찍어주세요."}[verdict]
    groups = []
    g_obj = None
    if verdict == "abnormal":
        base = {g: 0.05 for g in _GROUPS}
        base[group or _GROUPS[1]] = 0.85
        tot = sum(base.values())
        groups = [{"name": g, "prob": round(p / tot, 4), "percent": round(p / tot * 100, 1)}
                  for g, p in sorted(base.items(), key=lambda kv: -kv[1])]
        if group:
            g_obj = {"name": group, "prob": groups[0]["prob"], "percent": groups[0]["percent"],
                     "confidence": round(groups[0]["prob"] * abnormal, 4),
                     "text": f"모양만 보면 {group}에 가깝습니다.",
                     "feature": _FEATURE[group],
                     "caveat": ("진단이 아닙니다. 여러 원인에서 나타날 수 있어 "
                                "모양만으로는 원인을 알 수 없어요.")}
    return {
        "contract_version": "1.0",
        "verdict": verdict, "headline": head, "body": body, "action": action,
        "stage1": {"abnormal_prob": round(abnormal, 4),
                   "abnormal_percent": round(abnormal * 100, 1),
                   "threshold": 0.1466, "calibrated": True},
        "stage2": {"shown": verdict == "abnormal", "groups": groups, "group": g_obj, "alert": None},
        "text": head, "disclaimer": DISCLAIMER,
        "meta": {"stage2_arms": arms, "stage2_crops": ["m2.5", "f320", "m2.5"][:arms],
                 "stage1_tag": "f320", "stage2_tag": "m2.5", "elapsed_ms": 3.0},
        "_demo": True, "_note": _NOTE,
    }


@mcp.tool()
def screen_skin_photo(image_path: str, guide_box: list[float] | None = None) -> dict:
    """반려견 피부 사진 스크리닝 (데모 스텁). 판정은 사진 내용과 무관합니다.

    파일 이름에 normal / retake 가 있으면 그 판정, 아니면 이름 해시로 고릅니다.
    병변 6종 이름은 들어 있지 않습니다. 계열(group)까지만 말하세요.
    """
    p = Path(image_path)
    name = p.name.lower()
    if "retake" in name:
        return _contract("retake", 0.0, None, 3)
    if "normal" in name:
        return _contract("normal", 0.06, None, 3)
    h = int(hashlib.md5(name.encode()).hexdigest(), 16)
    if h % 4 == 0:
        return _contract("normal", 0.08, None, 3)
    group = _GROUPS[h % len(_GROUPS)]
    # 확신이 낮은 갈래도 하나 둔다 — group 이 null 이면 계열도 못 말한다.
    if h % 5 == 0:
        return _contract("abnormal", 0.62, None, 3)
    return _contract("abnormal", 0.86, group, 3)


@mcp.tool()
def screening_health() -> dict:
    """스텁 상태."""
    return {"loaded": True, "demo": True, "mock": True, "stage2_arms": 3,
            "expected_stage2_arms": 3, "note": "데모 스텁 — 진짜 가중치가 아닙니다"}


if __name__ == "__main__":
    mcp.run()
