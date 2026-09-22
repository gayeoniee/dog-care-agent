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


def _box(raw: str | None) -> list[float] | None:
    """`"0.1,0.52,0.55,0.28"` → `[0.1, 0.52, 0.55, 0.28]`.

    ⚠️ **이걸 안 주면 판정이 조용히 달라집니다.** 저쪽 1단계는 네모의 *중심*만,
    2단계는 네모의 *크기*를 씁니다. 없으면 화면 중앙으로 물러서는데, 병변이
    가운데 있지 않으면 엉뚱한 데를 봅니다 (실제로 결절 사진이 정상으로 나왔습니다).
    """
    if not raw:
        return None
    try:
        parts = [float(x) for x in raw.replace(" ", "").split(",")]
    except ValueError:
        raise SystemExit(f"--box 를 숫자로 읽을 수 없습니다: {raw!r} (x,y,w,h)") from None
    if len(parts) != 4 or not all(0.0 <= v <= 1.0 for v in parts):
        raise SystemExit(f"--box 는 0~1 사이 네 숫자입니다: {raw!r}")
    return parts


async def _ask(args: argparse.Namespace) -> int:
    settings = get_settings().with_demo(args.demo)
    turn = await run_turn(args.question, image_path=args.image,
                          guide_box=_box(args.box), settings=settings)

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


async def _health(args: argparse.Namespace) -> int:
    settings = get_settings().with_demo(args.demo)
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


def _serve(args: argparse.Namespace) -> int:
    from dogcare.server import serve

    print(f"http://{args.host}:{args.port}  (Ctrl+C 로 종료)")
    serve(host=args.host, port=args.port, demo=args.demo)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="dogcare", description=__doc__.split("\n")[0])
    ap.add_argument("--demo", action="store_true",
                    help="스텁 서브에이전트로 뜬다. 서브레포·가중치·DB 불필요 (LLM 키만)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("ask", help="한 번 물어봅니다")
    a.add_argument("question")
    a.add_argument("--image", help="피부 사진 경로")
    a.add_argument("--box", metavar="x,y,w,h",
                   help="촬영 가이드 프레임 (0~1 정규화). 병변에 맞춘 네모입니다. "
                        "안 주면 화면 중앙으로 물러섭니다 — 2단계가 학습과 어긋납니다")
    a.add_argument("--trace", action="store_true", help="실행 기록을 파일로 남깁니다")
    a.set_defaults(fn=_ask)

    h = sub.add_parser("health", help="서브에이전트가 붙었나")
    h.set_defaults(fn=_health)

    s = sub.add_parser("serve", help="웹 UI 를 띄웁니다")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.set_defaults(fn=_serve)

    args = ap.parse_args()
    if args.cmd == "serve":
        return args.fn(args)
    return asyncio.run(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
