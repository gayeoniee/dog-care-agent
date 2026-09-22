# dog-care-agent — 데모 모드 컨테이너 (D1).
#
# 스텁 서브에이전트 둘로 뜬다. 서브레포 · 가중치 · DB 가 없고 LLM 키만 있으면 된다.
# 진짜 모드는 이 이미지로 안 된다 — 세 저장소를 나란히 두고 로컬에서 돌린다.
#
#   docker build -t dog-care-agent .
#   docker run -p 8765:8765 -e LLM_API_KEY=... dog-care-agent
#
# HF Spaces 에는 이 이미지를 쓰지 않는다 — Docker SDK 가 유료로 잠겨 있다(2026-09).
# 라이브 데모는 Gradio SDK 위에서 `space/app.py` 로 뜬다: `uv run tools/push_space.py`.
# PORT 를 읽는 건 그때의 흔적이자, 다른 PaaS 에 올릴 때를 위해 남겨 둔다.
#   docker run -p 7860:7860 -e PORT=7860 -e LLM_API_KEY=... dog-care-agent

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUTF8=1 \
    DOGCARE_DEMO=1 RATE_PER_MIN=6 MAX_CONCURRENT=2 TRACE_DIR=/tmp/traces

# 의존성 먼저 (캐시 레이어). 소스는 다음 단계에서.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY mcp_servers ./mcp_servers
RUN uv sync --frozen --no-dev

# 컨테이너 안에서는 uv 가 스텁 서버를 --project . 로 띄운다 (subagents._spec 의 ROOT).
EXPOSE 8765
CMD ["sh", "-c", "uv run dogcare --demo serve --host 0.0.0.0 --port ${PORT:-8765}"]
