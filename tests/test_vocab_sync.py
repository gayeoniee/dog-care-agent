"""어휘가 저쪽과 어긋나지 않았나 — **서브레포가 곁에 있을 때만** 봅니다.

`vocab.py` 는 저쪽 `src/config.py` 의 **사본**입니다. 사본을 두는 이유는
게이트가 torch 없이 돌아야 하기 때문인데, 사본은 원본이 바뀌면 조용히
틀려집니다. 실제로 한 번 그랬습니다 — 계열 이름이 2026-09-10 에 보호자 말로
바뀌었는데(`미란·궤양` → `벗겨지거나 패인 상처`) 여기는 옛 이름을 들고 있었고,
게이트는 **아무 에러 없이** 엉뚱한 것을 막고 있었습니다.

서브레포가 없으면 건너뜁니다. CI 가 없는 저장소라 이 테스트는 **사람이
로컬에서 돌릴 때** 값을 합니다.
"""

from __future__ import annotations

import sys

import pytest

from dogcare.config import get_settings
from dogcare.vocab import GROUP_LABELS, GROUPS, LESION_KO


def _their_config():
    repo = get_settings().skin_repo
    if not (repo / "src" / "config.py").exists():
        pytest.skip(f"피부 서브레포가 없습니다: {repo}")
    sys.path.insert(0, str(repo))
    try:
        from src import config
    except Exception as exc:
        pytest.skip(f"저쪽 config 를 못 읽었습니다: {type(exc).__name__}: {exc}")
    finally:
        sys.path.remove(str(repo))
    return config


def test_6종_한글이름이_저쪽과_같다():
    theirs = _their_config().CLASS_KO
    ours = LESION_KO
    for code, name in ours.items():
        assert theirs.get(code) == name, f"{code}: 저쪽 {theirs.get(code)!r} ≠ 여기 {name!r}"
    #: 저쪽이 코드를 늘리면 (A7 은 정상이라 제외) 여기도 늘어야 한다.
    extra = {c for c in theirs if c.startswith("A") and c != "A7"} - set(ours)
    assert not extra, f"저쪽에만 있는 병변 코드: {sorted(extra)}"


def test_계열_이름과_묶음이_저쪽과_같다():
    cfg = _their_config()
    theirs = {g: tuple(sorted(c for c, gg in cfg.MORPH_GROUP_KEEP_A6.items() if gg == g))
              for g in dict.fromkeys(cfg.MORPH_GROUP_KEEP_A6.values())}
    ours = {g: tuple(sorted(codes)) for g, codes in GROUPS.items()}
    assert ours == theirs, f"계열이 어긋났습니다\n 저쪽: {theirs}\n 여기: {ours}"


def test_용어풀이가_저쪽과_같다():
    theirs = _their_config().GROUP_LABELS
    assert theirs == GROUP_LABELS, f"labels 가 어긋났습니다\n 저쪽: {theirs}\n 여기: {GROUP_LABELS}"
