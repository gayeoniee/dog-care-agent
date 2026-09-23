"""정적 쇼케이스를 `site/` 에 짓는다 (B3) — GitHub Pages 가 서빙하는 것.

Pages 는 파이썬을 못 돌린다. 올라갈 수 있는 건 **트레이스 뷰어 + 평가 결과 +
스크린샷**뿐이다. 코드를 안 돌려도 "이 시스템이 뭘 보고 뭘 막았나" 가 보이게.

    uv run python tools/build_site.py --pick 20260921-235342-5929b1571203.json

무엇을 담나
- evals/out/traces-adversarial/*.json  — 적대적 평가의 실제 트레이스 (사진 경로는 지운다)
- traces/ 에서 고른 데모 트레이스      — `--pick` 으로 파일 이름을 준다
- evals/out/{routing,gates,adversarial}-*.json 최신 하나씩
- docs/assets/*.png

⚠️ 트레이스에는 질문 원문이 들어 있다. 평가셋 문항과 데모 문항만 올린다 —
   보호자의 실제 질문은 올리지 않는다. `site/data` 는 이 스크립트가 만든다.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
DATA = SITE / "data"


def _scrub(t: dict) -> dict:
    """사진 경로·업로드 경로를 지운다. 게이트가 본 것(판정 계약)은 남긴다."""
    t = json.loads(json.dumps(t, ensure_ascii=False))
    if t.get("image_path"):
        t["image_path"] = Path(t["image_path"]).name
    for c in t.get("calls") or []:
        a = c.get("arguments") or {}
        if "image_path" in a:
            a["image_path"] = Path(a["image_path"]).name
    return t


def _latest(prefix: str) -> Path | None:
    xs = sorted((ROOT / "evals" / "out").glob(f"{prefix}-*.json"))
    return xs[-1] if xs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pick", nargs="*", default=[], help="traces/ 에서 올릴 트레이스 파일 이름")
    args = ap.parse_args()

    if DATA.exists():
        shutil.rmtree(DATA)
    (DATA / "traces").mkdir(parents=True)
    index: dict = {"traces": [], "evals": {}, "shots": []}

    srcs = list(sorted((ROOT / "evals" / "out" / "traces-adversarial").glob("*.json")))
    srcs += [ROOT / "traces" / n for n in args.pick]
    for p in srcs:
        if not p.exists():
            continue
        t = _scrub(json.loads(p.read_text(encoding="utf-8")))
        (DATA / "traces" / p.name).write_text(json.dumps(t, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
        index["traces"].append({
            "file": p.name, "question": t.get("question", "")[:80],
            "tools": [c["name"] for c in t.get("calls") or []],
            "first_pass_violations": t.get("first_pass_violations") or [],
            "repaired": bool(t.get("repaired")), "composed": bool(t.get("composed")),
            "blocked": bool(t.get("blocked")), "elapsed_ms": t.get("elapsed_ms"),
            "kind": "adversarial" if "traces-adversarial" in str(p) else "demo",
        })

    for kind in ("routing", "gates", "adversarial"):
        p = _latest(kind)
        if p:
            shutil.copy(p, DATA / f"{kind}.json")
            index["evals"][kind] = p.name
    # judge 채점(judge-*.json)과 보정(judge-calibration-*.json)은 이름이 겹치므로 따로 고른다.
    # 채점 rows 에는 질문 앞 60자가 들어 있어 그대로 올려도 사진 경로는 없다.
    js = [p for p in sorted((ROOT / "evals" / "out").glob("judge-*.json"))
          if "calibration" not in p.name]
    if js:
        shutil.copy(js[-1], DATA / "judge.json")
        index["evals"]["judge"] = js[-1].name

    (DATA / "shots").mkdir(exist_ok=True)
    for p in sorted((ROOT / "docs" / "assets").glob("*.png")):
        shutil.copy(p, DATA / "shots" / p.name)
        index["shots"].append(p.name)

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    m = re.search(r"^# .*?\n\n(.*?)\n\n", readme, re.S)
    index["tagline"] = (m.group(1) if m else "").replace("**", "")
    (DATA / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    print(f"site/data: 트레이스 {len(index['traces'])} · 평가 {sorted(index['evals'])} "
          f"· 스크린샷 {len(index['shots'])}")


if __name__ == "__main__":
    main()
