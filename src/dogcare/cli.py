"""명령줄.

    uv run dogcare ask "산책할 때 줄을 당겨요"
    uv run dogcare ask "여기 좀 봐주세요" --image ~/Pictures/dog.jpg
    uv run dogcare health          # 서브에이전트가 붙었나, 무엇을 물고 있나

⚠️ 콘솔 인코딩을 import 시점에 고칩니다. 안 고치면 파이프·파일로 나갈 때
   cp949 가 되고, 한글 한 줄 때문에 **프로그램이 뜨지도 못합니다.**
   (저쪽 dog-skin-screening 이 같은 데서 한 번 죽었습니다 — src/env.py 주석)
"""

from __future__ import annotations

import argparse
import asyncio
import io
import sys


def _fix_console() -> None:
    for stream in ("stdout", "stderr"):
        s = getattr(sys, stream, None)
        if isinstance(s, io.TextIOWrapper) and (s.encoding or "").lower() != "utf-8":
            s.reconfigure(encoding="utf-8", errors="replace")


_fix_console()

from dogcare.config import get_settings  # noqa: E402
from dogcare.loop import run_turn, save_trace  # noqa: E402
from dogcare.subagents import Subagents  # noqa: E402

RULE = "─" * 60


async def _ask(args: argparse.Namespace) -> int:
    settings = get_settings()
    turn = await run_turn(args.question, image_path=args.image, settings=settings)

    print(RULE)
    print(turn.answer)
    print(RULE)

    if turn.composed:
        print("※ 이 답은 LLM 이 아니라 코드가 조립했습니다 — 게이트가 두 번 걸렸습니다.")
    for v in turn.trace.violations:
        print(f"※ 게이트: {v}")

    calls = " · ".join(f"{c.name}({c.elapsed_ms:.0f}ms)" for c in turn.trace.calls) or "없음"
    print(f"툴 {len(turn.trace.calls)}회: {calls}")
    print(f"왕복 {turn.trace.rounds}회 · {turn.trace.elapsed_ms:.0f}ms")
    if args.trace:
        print(f"기록: {save_trace(turn, settings)}")
    return 1 if turn.trace.blocked else 0


async def _health(_: argparse.Namespace) -> int:
    settings = get_settings()
    async with Subagents(settings) as agents:
        if note := agents.note_failures():
            print(note)
        print(f"붙은 서브에이전트: {', '.join(agents.sessions) or '없음'}")
        print(f"툴 {len(agents.tools)}개: {', '.join(agents.owner)}")
        for probe in ("behavior_rag_health", "screening_health"):
            if probe in agents.owner:
                print(f"\n[{probe}]")
                print(await agents.call(probe, {}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="dogcare", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("ask", help="한 번 물어봅니다")
    a.add_argument("question")
    a.add_argument("--image", help="피부 사진 경로")
    a.add_argument("--trace", action="store_true", help="실행 기록을 파일로 남깁니다")
    a.set_defaults(fn=_ask)

    h = sub.add_parser("health", help="서브에이전트가 붙었나")
    h.set_defaults(fn=_health)

    args = ap.parse_args()
    return asyncio.run(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
