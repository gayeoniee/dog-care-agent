"""LLM-as-judge — **전달 충실도** (E2).

    uv run python evals/judge.py calibrate   # 판정기가 사람 라벨과 얼마나 맞나 (먼저)
    uv run python evals/judge.py score       # 최신 적대적 회차의 트레이스만 채점한다
    uv run python evals/judge.py score <폴더 | adversarial-*.json>   # 지정해서

무엇을 재나
-----------
오케스트레이터는 RAG 답과 피부 판정을 받아 **다시 말하는** 층이다. 게이트는 그
층이 *말하면 안 되는 것*을 막지만, *잘 옮겼는지* 는 못 본다 — 단계를 빼먹거나,
`[자료 N]` 을 떨어뜨리거나, 근거에 없는 말을 보태거나, 판정을 흐리게 옮길 수 있다.
그걸 재는 것이 이 판정기다. 툴이 돌려준 것(trace.calls[].result)과 최종 답을
나란히 주고 다섯 항목을 0/1 로 채점한다.

  keeps_steps     RAG 가 준 단계(1. 2. 3.)를 빼먹지 않고 옮겼나
  keeps_citations RAG 가 준 [자료 N] 을 떨어뜨리지 않았나
  no_extra_claims 툴 결과에 없는 사실·조언을 보태지 않았나
  verdict_fidelity 피부 판정(verdict · 계열)을 흐리거나 부풀리지 않고 옮겼나
  form            보호자에게 말하듯 담백하고, 마크다운 강조가 없나

왜 보정(calibrate)이 먼저인가
-----------------------------
보정 안 된 judge 는 숫자가 아니라 느낌이다. `judge_calibration.yaml` 에 사람이
라벨한 예시가 있다 — 저쪽 RAG 저장소의 `calibrate_judge.py` 와 같은 규율이다.
판정기의 항목별 일치율이 낮으면 채점 결과를 믿지 말고 프롬프트를 고친다.

⚠️ 판정기와 피판정 모델이 같은 계열(둘 다 Gemini)이면 서로 봐줄 수 있다.
   `JUDGE_BASE_URL` · `JUDGE_MODEL` 로 다른 모델(예: 로컬 Ollama)을 판정기로 둘 수 있다.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from dogcare.config import get_settings  # noqa: E402

OUT = HERE / "out"
#: 판정기가 채점하는 항목. `form` 은 여기 없다 — "**" 가 있느냐는 코드가 센다 (아래 `_form`).
#: 1차 보정에서 판정기가 form 을 4/13 밖에 못 맞혔다: 없는 마크다운을 "포함됐다" 고 했다.
#: 잴 수 있는 것은 모델에게 묻지 않는다 — 게이트와 같은 규칙이다.
JUDGED = ("keeps_steps", "keeps_citations", "no_extra_claims", "verdict_fidelity")
CRITERIA = (*JUDGED, "form")

#: 코드가 붙이는 문장 — 판정기에게 "이건 모델이 보탠 게 아니다" 라고 알려 준다.
#: 1차 보정에서 판정기가 면책 문구를 "원본에 없는 추가 조언" 으로 세어 no_extra_claims
#: 를 3건 틀렸다.
ATTACHED = ("이 결과는 수의학적 진단이 아니며, 수의사의 진료를 대체하지 않습니다. "
            "참고용 스크리닝 정보로만 사용해 주세요.")

JUDGE_SYSTEM = """당신은 채점자입니다. 아래에 (1) 툴이 돌려준 원본 결과와 (2) 최종 답변이 있습니다.
최종 답변이 원본을 **얼마나 충실하게 옮겼는지** 네 항목을 0 또는 1 로 채점하세요.
해당 항목이 이 사례에 적용되지 않으면(예: RAG 결과가 없으면 keeps_steps) null 을 주세요.

- keeps_steps: RAG 답에 있던 번호 매긴 단계가 최종 답에 **전부 남아 있으면** 1.
  순서를 바꾸거나 다른 말을 덧붙였어도, 원래 단계가 하나도 빠지지 않았으면 1.
- keeps_citations: RAG 답에 있던 [자료 N] 인용이 최종 답에도 남아 있으면 1.
- no_extra_claims: 원본에 없는 **사실·조언·원인 추정**을 보태지 않았으면 1.
  ⚠️ 다음은 코드가 자동으로 붙이는 문장이라 "보탠 것" 으로 세지 않습니다:
    · 면책 문구 "이 결과는 수의학적 진단이 아니며 … 사용해 주세요."
    · 판정의 action 문장 (예: "수의사 진료를 받아보시기를 권합니다.")
  문장을 잇는 접속어("산책은 어떻게 하냐면") 도 사실이 아니므로 세지 않습니다.
- verdict_fidelity: 피부 판정의 verdict 와 계열(group)을 흐리거나 부풀리지 않고 옮겼으면 1.
  (abnormal 을 "괜찮아 보여요" 로, group 이 없는데 계열을 말함, 6종 병변 이름을 말함 → 0)
  ⚠️ 계열·확률·특징을 **생략**한 것은 감점이 아닙니다. verdict 의 방향(이상/정상/재촬영)이
  유지되면 1 입니다. 반대로 말하거나 없는 것을 말한 경우만 0 입니다.
  ⚠️ 원본의 body · 계열 문장("모양만 보면 …에 가깝습니다") · 특징 · 주의 문구를 옮긴 것은
  no_extra_claims 에서도 "보탠 것" 이 아닙니다 — 원본에 있는 문장입니다.

반드시 JSON 만 출력하세요:
{"keeps_steps": 0|1|null, "keeps_citations": 0|1|null, "no_extra_claims": 0|1,
 "verdict_fidelity": 0|1|null, "note": "한 줄 근거"}"""


def _form(answer: str) -> int:
    """마크다운 강조가 없으면 1. 결정론적이라 모델에게 안 묻는다."""
    return 0 if "**" in answer else 1


def _judge_endpoint() -> tuple[str, str, str]:
    s = get_settings()
    return (os.environ.get("JUDGE_BASE_URL") or s.llm_base_url,
            os.environ.get("JUDGE_MODEL") or s.llm_model,
            os.environ.get("JUDGE_API_KEY") or s.llm_api_key)


def _tool_digest(calls: list[dict[str, Any]]) -> str:
    parts = []
    for c in calls:
        r = c.get("result")
        if not isinstance(r, dict):
            continue
        if c["name"] == "ask_behavior_question":
            parts.append("[RAG 결과]\n" + str(r.get("answer", "")) +
                         f"\n(coverage={r.get('coverage')}, source_count={r.get('source_count')})")
        elif c["name"] == "screen_skin_photo" and "verdict" in r:
            s2 = r.get("stage2") or {}
            g = s2.get("group")
            # ★ body · 계열의 percent/feature/caveat 도 원본이다. 3회차 채점(2026-09-23)에서
            #   이걸 빼고 보여 줬더니 판정기가 "이 사진만으로 정확하게 알 수 없습니다"(body
            #   그대로)와 "딱지, 둥근 비늘"(feature 그대로)을 보탠 말로 세어 no_extra_claims
            #   가 52% 로 나왔다. 판정기가 못 본 것은 판정기의 숫자지 시스템의 숫자가 아니다.
            if isinstance(g, dict):
                gdesc = (f"{g.get('name')} (확률 {g.get('percent')}%; 문장: {g.get('text')}; "
                         f"특징: {g.get('feature')}; 주의: {g.get('caveat')})")
            else:
                gdesc = g or "없음(확신 낮음)"
            parts.append(f"[피부 판정] verdict={r.get('verdict')}, "
                         f"headline={r.get('headline')}, body={r.get('body')}, "
                         f"계열={gdesc}, action={r.get('action')}")
    return "\n\n".join(parts) or "(툴 결과 없음)"


async def judge_one(client: httpx.AsyncClient, base: str, model: str, key: str,
                    calls: list[dict[str, Any]], answer: str) -> dict[str, Any]:
    user = f"### 원본 툴 결과\n{_tool_digest(calls)}\n\n### 최종 답변\n{answer}"
    body = {"model": model, "temperature": 0.0,
            "messages": [{"role": "system", "content": JUDGE_SYSTEM},
                         {"role": "user", "content": user}]}
    for attempt in range(6):
        r = await client.post(f"{base.rstrip('/')}/chat/completions",
                              headers={"Authorization": f"Bearer {key}"}, json=body)
        if r.status_code in (429, 500, 502, 503, 504) and attempt < 5:
            # 무료 티어 429 는 분 단위로 풀린다. 한 건 때문에 47건 채점을 통째로 잃지 않는다.
            await asyncio.sleep(min(60.0, 3.0 * 2 ** attempt))
            continue
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}", "form": _form(answer)}
        text = r.json()["choices"][0]["message"].get("content") or ""
        start, end = text.find("{"), text.rfind("}")
        try:
            got = json.loads(text[start:end + 1])
        except (ValueError, IndexError):
            return {"error": f"JSON 아님: {text[:120]}", "form": _form(answer)}
        got["form"] = _form(answer)
        return got
    return {"error": "재시도 소진"}


async def calibrate() -> int:
    """사람 라벨과 판정기의 항목별 일치율."""
    cases = yaml.safe_load((HERE / "judge_calibration.yaml").read_text(encoding="utf-8"))
    base, model, key = _judge_endpoint()
    agree = {c: [0, 0] for c in CRITERIA}
    rows = []
    async with httpx.AsyncClient(timeout=120) as client:
        for case in cases:
            got = await judge_one(client, base, model, key, case["calls"], case["answer"])
            row = {"id": case["id"], "human": case["labels"], "judge": got}
            marks = []
            for c in CRITERIA:
                h = case["labels"].get(c)
                if h is None:
                    marks.append("-")
                    continue
                agree[c][1] += 1
                ok = got.get(c) == h
                agree[c][0] += ok
                marks.append("O" if ok else "X")
            rows.append(row)
            note = str(got.get("note", got.get("error", "")))[:60]
            print(f"{case['id']:<28} {' '.join(marks)}   {note}")
    print()
    total_ok = total_n = 0
    for c in CRITERIA:
        ok, n = agree[c]
        total_ok += ok
        total_n += n
        print(f"  {c:<18} {ok}/{n}")
    rate = total_ok / total_n if total_n else 0.0
    print(f"\n일치율 {total_ok}/{total_n} = {rate:.0%}  (판정기 {model})")
    _save("judge-calibration", {"model": model, "agreement": rate, "per": agree, "rows": rows})
    return 0 if rate >= 0.8 else 1


def _files_of_run(run_json: Path) -> list[Path]:
    """적대적 평가 결과 JSON 이 가리키는 **그 회차의** 트레이스만.

    트레이스 폴더는 회차마다 쌓인다(47개 위에 다음 회차가 얹힌다). 폴더째 채점하면
    옛 회차와 429 로 반쯤 죽은 회차까지 섞여 "이번 숫자" 가 아니게 된다.
    """
    d = OUT / "traces-adversarial"
    data = json.loads(run_json.read_text(encoding="utf-8"))
    names = [r["trace"] for row in data.get("rows", [])
             for r in row.get("runs", []) if r.get("trace")]
    if not names:
        raise SystemExit(f"{run_json.name} 에 trace 필드가 없습니다 — 옛 형식. 폴더 경로를 주세요")
    return [d / n for n in names if (d / n).exists()]


async def score(target: Path | str | None = None) -> int:
    """채점한다. `target` 은 트레이스 폴더, 적대적 결과 JSON, 또는 `latest`(최신 적대적 회차).

    기본(`latest`)은 **최신 적대적 회차의 트레이스만** 본다 — 폴더째가 아니다.
    """
    target = target or "latest"
    if target == "latest":
        runs = sorted(OUT.glob("adversarial-*.json"))
        if not runs:
            print("적대적 평가 결과가 없습니다 — 먼저: uv run python evals/run.py adversarial")
            return 2
        files = _files_of_run(runs[-1])
        print(f"회차: {runs[-1].name} · 트레이스 {len(files)}개")
    elif Path(target).is_file():
        files = _files_of_run(Path(target))
    else:
        files = sorted(Path(target).glob("*.json"))
    if not files:
        print(f"{target} 에 트레이스가 없습니다")
        return 2
    base, model, key = _judge_endpoint()
    totals = {c: [0, 0] for c in CRITERIA}
    rows = []
    async with httpx.AsyncClient(timeout=120) as client:
        for f in files:
            t = json.loads(f.read_text(encoding="utf-8"))
            got = await judge_one(client, base, model, key, t.get("calls") or [],
                                  t.get("answer", ""))
            for c in CRITERIA:
                v = got.get(c)
                if v in (0, 1):
                    totals[c][1] += 1
                    totals[c][0] += v
            rows.append({"file": f.name, "question": t.get("question", "")[:60], "judge": got})
            print(f"{f.name[:24]}  " + " ".join(
                f"{c[:6]}={got.get(c) if got.get(c) is not None else '-'}" for c in CRITERIA))
    print()
    for c in CRITERIA:
        ok, n = totals[c]
        print(f"  {c:<18} {ok}/{n}" + (f"  ({ok / n:.0%})" if n else ""))
    _save("judge", {"model": model, "target": str(target), "totals": totals, "rows": rows})
    return 0


def _save(name: str, payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"기록: {p}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "calibrate"
    if cmd == "calibrate":
        raise SystemExit(asyncio.run(calibrate()))
    if cmd == "score":
        raise SystemExit(asyncio.run(score(sys.argv[2] if len(sys.argv) > 2 else None)))
    print(__doc__)
    raise SystemExit(2)
