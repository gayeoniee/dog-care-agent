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
서버 둘, `uv export` 로 뽑은 requirements.txt 만 싣는다. 진짜 서버 · 평가 · 테스트 ·
문서는 안 올라간다 (필요 없고, 저쪽 레포 경로가 박혀 있다).

SDK 는 **Gradio** 다. Docker SDK 가 유료로 잠겨서(2026-09) Dockerfile 은 로컬·CI 용.
Gradio 런타임은 `python app.py` 를 돌리고 7860 이 열리길 기다릴 뿐이라 FastAPI 가 그대로 뜬다.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
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


def _requirements() -> str:
    """uv.lock 을 그대로 고정한 requirements.txt. Space 는 pip 으로 깐다."""
    out = subprocess.run(
        ["uv", "export", "--no-dev", "--no-hashes", "--no-emit-project", "--no-header"],
        cwd=ROOT, capture_output=True, text=True, check=True, encoding="utf-8").stdout
    return "".join(line for line in out.splitlines(keepends=True) if not line.startswith("#"))


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
