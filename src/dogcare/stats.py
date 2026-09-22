"""`dogcare stats` — traces/ 를 집계합니다 (B5).

트레이스는 턴마다 하나씩 쌓입니다. 하나씩 열어 보면 "이 턴에 뭐가 걸렸나" 는
보이는데 "얼마나 자주 걸리나" 는 안 보입니다. 게이트가 장식이 아니라는 건
발동률로 말해야 합니다.

    uv run dogcare stats
    uv run dogcare stats --dir evals/out/traces-adversarial
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Stats:
    turns: int = 0
    blocked: int = 0
    composed: int = 0
    with_violations: int = 0
    first_pass_hit: int = 0
    repaired: int = 0
    gate_hits: Counter[str] = field(default_factory=Counter)
    tool_calls: Counter[str] = field(default_factory=Counter)
    tool_errors: Counter[str] = field(default_factory=Counter)
    tool_ms: dict[str, list[float]] = field(default_factory=dict)
    turn_ms: list[float] = field(default_factory=list)
    rounds: list[int] = field(default_factory=list)
    prompt_tokens: list[float] = field(default_factory=list)
    completion_tokens: list[float] = field(default_factory=list)

    def add(self, t: dict[str, Any]) -> None:
        self.turns += 1
        self.blocked += bool(t.get("blocked"))
        self.turn_ms.append(float(t.get("elapsed_ms") or 0))
        self.rounds.append(int(t.get("rounds") or 0))
        if t.get("prompt_tokens") or t.get("completion_tokens"):   # 기록 전 트레이스는 0
            self.prompt_tokens.append(float(t.get("prompt_tokens") or 0))
            self.completion_tokens.append(float(t.get("completion_tokens") or 0))
        viol = t.get("violations") or []
        self.with_violations += bool(viol)
        first = t.get("first_pass_violations") or []
        self.first_pass_hit += bool(first)
        self.repaired += bool(t.get("repaired"))
        # 발동은 **첫 초안** 기준으로 센다 — 최종만 세면 고쳐 쓴 턴이 다 빠진다.
        for v in first or viol:
            self.gate_hits[v.split(" ", 1)[0]] += 1
        for c in t.get("calls") or []:
            self.tool_calls[c["name"]] += 1
            if c.get("error"):
                self.tool_errors[c["name"]] += 1
            self.tool_ms.setdefault(c["name"], []).append(float(c.get("elapsed_ms") or 0))
        if t.get("composed"):
            self.composed += 1


def _p(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = min(len(xs) - 1, max(0, round(q * (len(xs) - 1))))
    return xs[k]


def collect(directory: Path) -> Stats:
    s = Stats()
    for p in sorted(directory.glob("*.json")):
        try:
            s.add(json.loads(p.read_text(encoding="utf-8")))
        except (ValueError, KeyError):
            continue
    return s


def render(s: Stats, directory: Path) -> str:
    if not s.turns:
        return f"{directory} 에 트레이스가 없습니다."
    L = [f"트레이스 {s.turns}턴 — {directory}", ""]
    L.append(f"첫 초안이 걸림       {s.first_pass_hit}/{s.turns}   ← 게이트 발동률")
    L.append(f"  고쳐 쓰기로 통과   {s.repaired}")
    L.append(f"  코드가 조립        {s.composed}")
    L.append(f"  최종도 위반(막힘)  {s.blocked}")
    if s.gate_hits:
        hits = " · ".join(f"{g} {n}" for g, n in sorted(s.gate_hits.items()))
        L.append(f"게이트 발동          {hits}")
    L.append(f"왕복                 중앙값 {statistics.median(s.rounds):.0f} "
             f"· 최대 {max(s.rounds)}")
    L.append(f"턴 지연              p50 {_p(s.turn_ms, .5) / 1000:.1f}s "
             f"· p95 {_p(s.turn_ms, .95) / 1000:.1f}s")
    if s.prompt_tokens:
        pt, ct = s.prompt_tokens, s.completion_tokens
        L.append(f"토큰/턴 ({len(pt)}턴)     프롬프트 p50 {_p(pt, .5):.0f} · p95 {_p(pt, .95):.0f}"
                 f" · 완성 p50 {_p(ct, .5):.0f} · p95 {_p(ct, .95):.0f}"
                 f" · 합계 {int(sum(pt) + sum(ct)):,}")
    L.append("")
    L.append("툴                   호출  오류   p50      p95")
    for name, n in s.tool_calls.most_common():
        ms = s.tool_ms.get(name, [])
        L.append(f"  {name:<20} {n:>4} {s.tool_errors[name]:>5} "
                 f"{_p(ms, .5):>7.0f}ms {_p(ms, .95):>7.0f}ms")
    return "\n".join(L)
