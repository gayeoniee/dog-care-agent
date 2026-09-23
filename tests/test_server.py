"""웹 서버 (S6) — `/api/ask` → SSE → 결과, 그리고 위생 규칙.

LLM 은 가짜로 바꾼다 (툴 하나 부르고 답한다). 서브에이전트도 가짜다. 여기서 재는 건
**서버가 지키기로 한 것**이다 — 질문 길이 · 사진 검증 · 판정 뒤 사진 삭제 · 레이트
리밋 · 멀티턴 판정 유지 · 트레이스에 절대경로 없음.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

import dogcare.server as server
from dogcare.config import get_settings

JPEG = bytes([0xFF, 0xD8, 0xFF, 0xE0]) + b"\x00" * 64
ABNORMAL = {
    "verdict": "abnormal", "headline": "피부에 이상 소견이 보입니다.", "body": "b", "action": "a",
    "stage1": {"abnormal_percent": 86.0},
    "stage2": {"group": {"name": "벗겨지거나 패인 상처", "percent": 85.0}, "groups": []},
    "disclaimer": "이 결과는 수의학적 진단이 아니며, 수의사의 진료를 대체하지 않습니다. "
                  "참고용 스크리닝 정보로만 사용해 주세요.",
    "meta": {"stage2_arms": 3},
}


class _FakeAgents:
    def __init__(self) -> None:
        self.tools = [{"type": "function", "function": {"name": "screen_skin_photo",
                                                        "description": "", "parameters": {}}}]
        self.owner = {"screen_skin_photo": "skin"}
        self.sessions = {"skin": object()}
        self.failed: dict[str, str] = {}
        self.seen: list[dict[str, Any]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return None

    async def call(self, name: str, arguments: dict[str, Any]) -> Any:
        self.seen.append(arguments)
        return ABNORMAL

    def note_failures(self) -> str:
        return ""


class _FakeLLM:
    """사진이 있으면 툴을 부르고, 없으면 바로 답한다."""

    def __init__(self, settings) -> None:
        self.n = 0

    async def chat(self, messages, tools=None):
        self.n += 1
        has_img = "screen_skin_photo 를 부르세요" in messages[-1].get("content", "") \
            if messages[-1].get("role") == "user" else False
        if self.n == 1 and has_img:
            return {"role": "assistant", "content": None, "tool_calls": [{
                "id": "1", "type": "function",
                "function": {"name": "screen_skin_photo", "arguments": "{}"}}]}
        return {"role": "assistant", "content": "모양만 보면 벗겨지거나 패인 상처에 가깝습니다."}


@pytest.fixture
def client(monkeypatch, tmp_path):
    import dogcare.loop as loop

    agents = _FakeAgents()
    monkeypatch.setattr(server, "Subagents", lambda settings: agents)
    monkeypatch.setattr(loop, "ToolCallingLLM", _FakeLLM)
    monkeypatch.setattr(server, "UPLOADS", tmp_path / "uploads")
    monkeypatch.setattr(server, "RATE_PER_MIN", 3)
    settings = get_settings()
    object.__setattr__(settings, "trace_dir", tmp_path / "traces")
    app = server.build_app(settings)
    with TestClient(app) as c:
        c.agents = agents                                 # type: ignore[attr-defined]
        yield c


def _result(c: TestClient, job_id: str) -> dict[str, Any]:
    with c.stream("GET", f"/api/events/{job_id}") as s:
        ev = None
        for line in s.iter_lines():
            if line.startswith("event:"):
                ev = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and ev == "result":
                return json.loads(line[5:])
    raise AssertionError("result 이벤트가 안 왔다")


def test_health(client):
    h = client.get("/api/health").json()
    assert h["ok"] and h["subagents"] == ["skin"] and "limits" in h


def test_질문이_너무_길면_413(client):
    r = client.post("/api/ask", data={"question": "가" * 1001})
    assert r.status_code == 413


def test_빈_질문은_400(client):
    assert client.post("/api/ask", data={"question": "   "}).status_code == 400


def test_사진이_아니면_415(client):
    r = client.post("/api/ask", data={"question": "봐줘"},
                    files={"image": ("x.jpg", b"not an image at all", "image/jpeg")})
    assert r.status_code == 415


def test_box가_이상하면_400(client):
    r = client.post("/api/ask", data={"question": "봐줘", "box": "1.5,0,0,0"},
                    files={"image": ("x.jpg", JPEG, "image/jpeg")})
    assert r.status_code == 400


def test_사진_턴은_판정이_돌아오고_사진은_지워진다(client, tmp_path):
    r = client.post("/api/ask", data={"question": "봐줘", "box": "0.1,0.1,0.5,0.5"},
                    files={"image": ("x.jpg", JPEG, "image/jpeg")})
    assert r.status_code == 200
    res = _result(client, r.json()["job_id"])
    assert res["screening"]["verdict"] == "abnormal"
    assert res["screening"]["group"] == "벗겨지거나 패인 상처"
    assert ABNORMAL["disclaimer"] in res["answer"]           # 면책은 코드가 붙인다
    assert res["violations"] == []
    # S1: 판정이 끝나면 사진은 없다
    assert not list((tmp_path / "uploads").glob("*")), "사진이 남아 있다"
    # 코드가 경로와 프레임을 덮어썼다
    seen = client.agents.seen[0]
    assert seen["guide_box"] == [0.1, 0.1, 0.5, 0.5] and seen["image_path"].endswith(".jpg")
    # S7: 트레이스에 절대경로가 없다
    tr = json.loads((tmp_path / "traces" / res["trace"]).read_text(encoding="utf-8"))
    assert "/" not in tr["image_path"] and "\\" not in tr["image_path"]


def test_이어_묻기는_앞_턴_판정을_들고_간다(client):
    r = client.post("/api/ask", data={"question": "봐줘"},
                    files={"image": ("x.jpg", JPEG, "image/jpeg")})
    sid = r.json()["session_id"]
    _result(client, r.json()["job_id"])
    r2 = client.post("/api/ask", data={"question": "그거 어떻게 해요", "session_id": sid})
    res2 = _result(client, r2.json()["job_id"])
    # 사진이 없어도 판정이 실려 있으므로 면책이 붙는다 = 게이트가 판정을 봤다
    assert ABNORMAL["disclaimer"] in res2["answer"]
    assert res2["session_id"] == sid


def test_레이트_리밋_429(client):
    for _ in range(3):
        assert client.post("/api/ask", data={"question": "안녕"}).status_code == 200
    assert client.post("/api/ask", data={"question": "안녕"}).status_code == 429


def test_트레이스_경로_탈출_불가(client):
    assert client.get("/api/trace/..%2F..%2Fpyproject.toml").status_code == 404
    assert client.get("/api/trace/nope.json").status_code == 404


def test_sniff_image():
    assert server.sniff_image(JPEG) == "jpg"
    assert server.sniff_image(bytes([0x89]) + b"PNG\r\n") == "png"
    assert server.sniff_image(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "webp"
    assert server.sniff_image(b"GIF89a") is None
    assert server.sniff_image(b"RIFF\x00\x00\x00\x00WAVE") is None


def test_stats_endpoint(client):
    r = client.post("/api/ask", data={"question": "봐줘"},
                    files={"image": ("x.jpg", JPEG, "image/jpeg")})
    _result(client, r.json()["job_id"])
    s = client.get("/api/stats").json()
    assert s["turns"] >= 1 and "gate_hits" in s and s["tool_calls"].get("screen_skin_photo") == 1


def test_LLM_키_없으면_서브에이전트_전에_죽는다():
    from dogcare.config import Settings, require_llm_key

    s = Settings()
    object.__setattr__(s, "llm_api_key", "")
    with pytest.raises(SystemExit):
        require_llm_key(s)


def test_user_facing_error_hides_provider_json() -> None:
    """무료 한도 429 가 화면에 JSON 덩어리로 찍히던 것 — 보호자 문장으로 바꾼다."""
    from dogcare.llm import LLMError

    raw = LLMError('HTTP 429 — [{ "error": { "code": 429, '
                   '"message": "You exceeded your current quota" } }]')
    msg = server.user_facing_error(raw)
    assert "한도" in msg
    assert "429" not in msg and "{" not in msg

    busy = server.user_facing_error(LLMError("4회 시도 후 실패 — HTTP 503 — high demand"))
    assert "붐빕니다" in busy
    # 모르는 오류는 첫 줄만, 종류를 남긴다
    other = server.user_facing_error(RuntimeError("툴 이름이 겹칩니다\n두 번째 줄"))
    assert other == "RuntimeError: 툴 이름이 겹칩니다"


def test_client_ip_prefers_forwarded_header() -> None:
    """HF Spaces 프록시 뒤에서 client.host 가 전부 같아 한 사람이 쓰면 모두가 429 를 보던 것."""
    from starlette.requests import Request

    def req(headers: dict[str, str], host: str = "10.0.0.1") -> Request:
        raw = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
        return Request({"type": "http", "headers": raw, "client": (host, 1234),
                        "method": "GET", "path": "/", "query_string": b""})

    assert server.client_ip(req({"X-Forwarded-For": "203.0.113.9, 10.0.0.1"})) == "203.0.113.9"
    assert server.client_ip(req({})) == "10.0.0.1"
