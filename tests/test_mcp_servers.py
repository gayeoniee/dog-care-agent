"""MCP 서버의 **순수 함수**를 잽니다 — 서버를 띄우지 않고.

제일 중요한 방어선(`_for_model` 의 6종 이름 제거)이 실기기 trace 로만 확인돼
있었습니다. 저쪽 계약이 바뀌어 이름이 새 자리로 실려 오면 조용히 샙니다.
여기서 잡습니다. 스텁 서버의 계약 모양도 같이 봅니다 — 진짜와 필드가 같아야
게이트가 데모에서도 진짜에서도 똑같이 돕니다.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dogcare.vocab import GROUP_LABELS, GROUPS, LESION_TERMS  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "mcp_servers" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def skin():
    return _load("skin_screening_server")


@pytest.fixture(scope="module")
def skin_stub():
    return _load("skin_screening_stub_server")


@pytest.fixture(scope="module")
def rag_stub():
    return _load("behavior_rag_stub_server")


#: 저쪽 계약을 흉내 낸 응답 — labels 가 **네 줄 전부**와 group 에 실려 온다.
def _real_shaped():
    groups = [{"name": g, "labels": GROUP_LABELS[g], "prob": 0.25, "percent": 25.0} for g in GROUPS]
    return {
        "verdict": "abnormal",
        "stage1": {"abnormal_prob": 0.8},
        "stage2": {"shown": True,
                   "distribution": [{"code": "A5", "label": "미란·궤양", "prob": 0.6}],
                   "groups": groups,
                   "group": {"name": "벗겨지거나 패인 상처", "labels": "미란·궤양",
                             "prob": 0.6, "text": "모양만 보면 벗겨지거나 패인 상처에 가깝습니다."},
                   "alert": None},
        "meta": {"stage2_arms": 3},
    }


def test_모델_뷰에_6종_이름이_한_글자도_없다(skin):
    view = skin._for_model(_real_shaped())
    blob = json.dumps(view, ensure_ascii=False)
    leaked = sorted(t for t in LESION_TERMS if t in blob)
    assert not leaked, f"모델 뷰에 6종 어휘가 샜다: {leaked}"
    assert "distribution" not in blob and "labels" not in blob


def test_모델_뷰는_계열_이름과_확률은_남긴다(skin):
    view = skin._for_model(_real_shaped())
    assert view["stage2"]["group"]["name"] == "벗겨지거나 패인 상처"
    assert [g["name"] for g in view["stage2"]["groups"]] == list(GROUPS)
    assert "_note" in view


def test_SHOW_DISTRIBUTION이면_그대로_준다(skin, monkeypatch):
    monkeypatch.setattr(skin, "SHOW_DISTRIBUTION", True)
    assert skin._for_model(_real_shaped()) == _real_shaped()


def test_strip은_중첩_어디서든_뗀다(skin):
    nested = {"a": [{"labels": "x", "keep": 1}], "b": {"c": {"distribution": [], "d": 2}}}
    assert skin._strip(nested) == {"a": [{"keep": 1}], "b": {"c": {"d": 2}}}


# ── 스텁 계약이 진짜와 같은 모양인가 ─────────────────────────
REQUIRED = {"contract_version", "verdict", "headline", "body", "action", "stage1", "stage2",
            "disclaimer", "meta"}


@pytest.mark.parametrize("name", ["a.jpg", "normal_dog.jpg", "retake_blur.jpg", "b.jpg", "c.jpg",
                                  "d.jpg", "e.jpg", "f.jpg"])
def test_스텁_피부_계약_모양(skin_stub, name):
    r = skin_stub.screen_skin_photo(name)
    assert REQUIRED <= set(r), REQUIRED - set(r)
    assert r["verdict"] in {"normal", "abnormal", "retake"}
    assert r["meta"]["stage2_arms"] == 3
    assert r["_demo"] is True
    blob = json.dumps(r, ensure_ascii=False)
    assert not any(t in blob for t in LESION_TERMS), "스텁도 6종 이름을 내면 안 된다"
    if r["stage2"]["group"] is None and r["verdict"] == "abnormal":
        # 확신 낮은 갈래는 분포도 평평해야 앞뒤가 맞는다 — 85% 막대 옆 "계열 없음" 은
        # 모델이 막대를 읽고 단정하게 만들었다 (실제로 그랬다).
        assert r["stage2"]["groups"][0]["percent"] < 50


def test_스텁_피부는_이름으로_갈래를_고른다(skin_stub):
    assert skin_stub.screen_skin_photo("x_normal.jpg")["verdict"] == "normal"
    assert skin_stub.screen_skin_photo("x_retake.jpg")["verdict"] == "retake"


async def test_스텁_RAG_계약_모양(rag_stub):
    hit = await rag_stub.ask_behavior_question("산책할 때 줄을 당겨요")
    assert hit["coverage"] == "full" and hit["source_count"] == 2
    assert [s["n"] for s in hit["sources"]] == [1, 2]
    miss = await rag_stub.ask_behavior_question("고양이 모래 추천")
    assert miss["coverage"] == "none" and miss["source_count"] == 0
    ask = await rag_stub.ask_behavior_question("혼자 두면 울어요")
    assert ask["coverage"] == "needs_detail"
    for r in (hit, miss, ask):
        assert r["_demo"] is True
