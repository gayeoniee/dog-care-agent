"""설정 — .env 하나에서 읽습니다.

서브레포 경로가 **설정인 이유**: MCP 서버를 각자 그 레포의 venv 로 띄우기
때문입니다. 경로를 코드에 박으면 다른 사람이 클론했을 때 안 돕니다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def _p(key: str, default: str) -> Path:
    v = os.environ.get(key) or default
    p = Path(v).expanduser()
    return p if p.is_absolute() else (ROOT / p).resolve()


@dataclass(frozen=True)
class Settings:
    behavior_rag_repo: Path = field(
        default_factory=lambda: _p("BEHAVIOR_RAG_REPO", "../dog-behavior-rag"))
    skin_repo: Path = field(
        default_factory=lambda: _p("SKIN_REPO", "../dog-skin-screening"))

    skin_release_repo: str = os.environ.get(
        "SKIN_RELEASE_REPO", "gayoniee/daengs-skin-screening-release")
    skin_release_revision: str = os.environ.get("SKIN_RELEASE_REVISION", "v1")
    skin_release_dir: str = os.environ.get("SKIN_RELEASE_DIR", "")
    skin_expected_arms: int = int(os.environ.get("SKIN_EXPECTED_ARMS", "3") or 0)
    skin_mock: bool = os.environ.get("SKIN_MOCK", "0") == "1"
    #: 데모 모드 — 서브레포·가중치·DB 없이 **스텁 MCP 서버 둘**로 뜬다. LLM 키만 필요.
    #: 남이 클론해서 3분 안에 화면을 보게 하려고 있다. 응답에 `_demo: true` 가 박힌다.
    demo: bool = os.environ.get("DOGCARE_DEMO", "0") == "1"

    llm_base_url: str = os.environ.get("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
    #: 기본은 3.1-flash-lite — 라우팅 27/30 · 적대적 47턴 · judge 98% 가 전부 이 모델 숫자다.
    #: 3.6-flash 는 503(high demand)이 잦아 기본에서 뺐다 (.env.example 참고).
    llm_model: str = os.environ.get("LLM_MODEL", "gemini-3.1-flash-lite")
    llm_api_key: str = os.environ.get("LLM_API_KEY", "")
    llm_temperature: float = float(os.environ.get("LLM_TEMPERATURE", "0.2"))
    #: 한 응답의 상한. **안 보내면 로컬 모델이 끝없이 생성한다** — granite4.1 이 한 요청에서
    #: 1,000토큰 넘게 이어 가며 라우팅 평가를 멈춰 세웠다. Gemini 는 알아서 끊어서 안 보였다.
    llm_max_tokens: int = int(os.environ.get("LLM_MAX_TOKENS", "1024"))
    llm_timeout: float = float(os.environ.get("LLM_TIMEOUT_SECONDS", "120"))
    #: 429·503 재시도 횟수. 무료 티어는 "high demand" 503 이 자주 난다.
    llm_retries: int = int(os.environ.get("LLM_RETRIES", "4"))

    #: 한 턴에 허용할 툴 호출 왕복. 넘으면 **답을 만들지 않고 멈춥니다** —
    #: 루프가 도는 걸 모르고 토큰을 태우는 게 제일 흔한 사고입니다.
    max_tool_rounds: int = int(os.environ.get("MAX_TOOL_ROUNDS", "6"))

    trace_dir: Path = field(default_factory=lambda: _p("TRACE_DIR", "traces"))

    def with_demo(self, on: bool) -> Settings:
        """CLI `--demo` 가 부른다. 스텁은 팔 셋을 흉내 내므로 mock(1팔) 과 다르다."""
        import dataclasses
        return dataclasses.replace(self, demo=on) if on else self

    def env_for_skin(self) -> dict[str, str]:
        """피부 MCP 서버 프로세스에 넘길 환경. **HF 토큰도 여기로 갑니다.**"""
        e = {
            "SKIN_REPO": str(self.skin_repo),
            "SKIN_RELEASE_REPO": self.skin_release_repo,
            "SKIN_RELEASE_REVISION": self.skin_release_revision,
            "SKIN_RELEASE_DIR": self.skin_release_dir,
            "SKIN_EXPECTED_ARMS": str(self.skin_expected_arms),
            "SKIN_MOCK": "1" if self.skin_mock else "0",
        }
        if os.environ.get("HF_TOKEN"):
            e["HF_TOKEN"] = os.environ["HF_TOKEN"]
        return e

    def env_for_rag(self) -> dict[str, str]:
        return {"BEHAVIOR_RAG_REPO": str(self.behavior_rag_repo)}


def get_settings() -> Settings:
    return Settings()


def require_llm_key(settings: Settings) -> None:
    """LLM 키가 없으면 **서브에이전트를 띄우기 전에** 멈춥니다.

    전에는 MCP 서버 둘을 다 올린 뒤(진짜 모드면 bge-m3 와 가중치까지) 첫 LLM 호출에서
    "LLM_API_KEY 가 비어 있습니다" 로 죽었다. 30초~수 분을 버리고 나서야 알려 주는 셈이다.
    """
    if not settings.llm_api_key:
        raise SystemExit(
            "LLM_API_KEY 가 비어 있습니다 — .env 를 확인하세요." + "\n"
            "  Gemini 무료 티어: https://aistudio.google.com/apikey" + "\n"
            "  로컬 Ollama 면:   LLM_BASE_URL=http://localhost:11434/v1  LLM_API_KEY=ollama")
