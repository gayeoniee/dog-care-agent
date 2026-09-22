"""HF Spaces 진입점 — Gradio SDK Space 위에서 **FastAPI 를 그대로** 띄운다.

Spaces 의 Docker SDK 가 유료로 잠겨서(2026-09) Dockerfile 을 못 쓴다. Gradio SDK
런타임은 `python app.py` 를 돌리고 7860 포트가 열리기만 기다리므로, gradio 를
import 하지 않고 uvicorn 을 직접 올려도 뜬다. 이 파일은 저장소 루트로 복사되어
올라간다 (`tools/push_space.py` 가 한다) — `src/` 가 옆에 있는 것을 전제한다.

데모 모드다. 스텁 서브에이전트 둘로 뜨고, 필요한 건 Space Secret `LLM_API_KEY` 뿐.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

# Dockerfile 의 ENV 와 같은 값. Space 의 Variables 로 덮어쓸 수 있다.
os.environ.setdefault("DOGCARE_DEMO", "1")
os.environ.setdefault("RATE_PER_MIN", "6")
os.environ.setdefault("MAX_CONCURRENT", "2")
os.environ.setdefault("TRACE_DIR", "/tmp/traces")
os.environ.setdefault("PYTHONUTF8", "1")

from dogcare.server import serve  # noqa: E402

# ★ 무료 하드웨어가 ZeroGPU 뿐이다(CPU Basic 은 PRO). ZeroGPU 는 시작할 때
#   `@spaces.GPU` 함수 목록을 **startup report** 로 받지 못하면
#   "No @spaces.GPU function detected during startup" 으로 죽는다. 그 보고는
#   `spaces` 가 `gr.Blocks.launch` 에 끼워 넣은 훅이 보내는데, 우리는 gradio 를
#   띄우지 않으므로 같은 함수를 직접 부른다. 자리표시 함수는 **부르지 않는다** —
#   이 데모는 GPU 를 안 쓴다. (spaces/zero/__init__.py 의 startup() 과 같은 순서)
try:
    import spaces  # 런타임이 깔아 준다. 로컬엔 없다.

    @spaces.GPU
    def _zerogpu_placeholder() -> None:
        return None

    from spaces.zero.client import startup_report

    startup_report()
except Exception as exc:  # 로컬(패키지 없음)·비-ZeroGPU 에선 그냥 지나간다
    print("zerogpu startup report skipped:", type(exc).__name__, exc)

if __name__ == "__main__":
    serve(host="0.0.0.0", port=int(os.environ.get("PORT", "7860")), demo=True)
