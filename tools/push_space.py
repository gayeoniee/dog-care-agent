# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface_hub>=0.30", "python-dotenv>=1.0"]
# ///
"""HF Spaces 에 데모를 올린다 — Space 생성 · Secret · 업로드까지 한 번에.

    uv run tools/push_space.py            # .env 의 HF_TOKEN · LLM_API_KEY 를 쓴다
    uv run tools/push_space.py --dry-run  # 올릴 파일 목록만

왜 git push 가 아니라 upload 인가
--------------------------------
Space 는 README.md 맨 위에 YAML 머리말(sdk · app_file …)을 요구한다. 그걸 GitHub
README 에 넣으면 GitHub 이 표로 그려서 지저분해진다. 그래서 **Space 용 트리를 따로
조립**한다 — `space/README.md` 와 `space/app.py` 를 루트로 올리고, `src/` 와 스텁
서버 둘, pyproject 직접 의존성으로 만든 requirements.txt 만 싣는다. 진짜 서버 · 평가 · 테스트 ·
문서는 안 올라간다 (필요 없고, 저쪽 레포 경로가 박혀 있다).

SDK 는 **Gradio** 다. Docker SDK 가 유료로 잠겨서(2026-09) Dockerfile 은 로컬·CI 용.
Gradio 런타임은 `python app.py` 를 돌리고 7860 이 열리길 기다릴 뿐이라 FastAPI 가 그대로 뜬다.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

SPACE_ID = os.environ.get("HF_SPACE_ID", "gayoniee/dog-care-agent")

#: 올릴 것. (상대경로, 목적지)
FILES = [
    ("space/app.py", "app.py"),
    ("space/README.md", "README.md"),
    ("LICENSE", "LICENSE"),
    ("mcp_servers/behavior_rag_stub_server.py", "mcp_servers/behavior_rag_stub_server.py"),
    ("mcp_servers/skin_screening_stub_server.py", "mcp_servers/skin_screening_stub_server.py"),
]


#: Space 런타임이 `gradio[mcp]` 를 같이 깔고, 그게 `mcp<2` 를 요구한다. 우리 lock 은
#: mcp 2.x 라 그대로 올리면 pip 이 ResolutionImpossible 로 빌드를 죽인다 (실제로 그랬다).
#: 코드에 1.x shim(`_attr`, FastMCP 폴백)이 있고 1.30 에서 119 테스트가 통과하므로
#: Space 에서만 1.x 를 쓴다.
SPACE_MCP = "mcp>=1.21,<2"


def _requirements() -> str:
    """pyproject 의 직접 의존성만, 느슨하게. lock 을 그대로 고정하면 mcp 가 충돌한다."""
    import tomllib

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    deps = [SPACE_MCP if d.startswith("mcp") else d for d in project["dependencies"]]
    return "".join(d + "\n" for d in deps)


def _stage(dst: Path) -> list[Path]:
    for src_rel, dst_rel in FILES:
        target = dst / dst_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / src_rel, target)
    shutil.copytree(ROOT / "src", dst / "src",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (dst / "requirements.txt").write_text(_requirements(), encoding="utf-8")
    return sorted(p.relative_to(dst) for p in dst.rglob("*") if p.is_file())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--space", default=SPACE_ID)
    a = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "space"
        stage.mkdir()
        files = _stage(stage)
        print(f"올릴 파일 {len(files)}개 → {a.space}")
        for f in files:
            print("  ", f.as_posix())
        if a.dry_run:
            return 0

        token = os.environ.get("HF_TOKEN")
        llm_key = os.environ.get("LLM_API_KEY")
        if not token or not llm_key:
            print("HF_TOKEN 과 LLM_API_KEY 가 .env 에 있어야 합니다.", file=sys.stderr)
            return 2

        from huggingface_hub import HfApi

        api = HfApi(token=token)
        url = api.create_repo(a.space, repo_type="space", space_sdk="gradio",
                              space_hardware=os.environ.get("HF_SPACE_HARDWARE", "zero-a10g"),
                              private=False, exist_ok=True)
        print("Space:", url)
        # ★ 키는 Secret 으로만. 모델·엔드포인트는 Variable(공개) — 값이 비밀이 아니다.
        api.add_space_secret(a.space, "LLM_API_KEY", llm_key)
        for k in ("LLM_MODEL", "LLM_BASE_URL"):
            if v := os.environ.get(k):
                api.add_space_variable(a.space, k, v)
        api.upload_folder(repo_id=a.space, repo_type="space", folder_path=str(stage),
                          delete_patterns=["*"],
                          commit_message="데모 트리 업로드 (tools/push_space.py)")
        print("올렸습니다. 빌드 로그:", f"https://huggingface.co/spaces/{a.space}?logs=build")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
