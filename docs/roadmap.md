# 로드맵 — "완성본"까지

2026-09-22 에 세웠다. 순서대로 간다. 끝난 것은 체크하고 **무엇을 재봤는지** 옆에 적는다.

먼저 정한 것 둘:
- **얼굴은 웹 UI 다.** CLI 는 남기되 데모는 브라우저에서 한다.
- **완전 재현은 불가능하다** (가중치는 AI Hub 파생, 보듬TV 코퍼스는 재배포 금지).
  그래서 키·가중치·DB 없이 뜨는 **데모 모드**가 있어야 공개가 의미 있다.

GitHub Pages 는 정적 파일만 서빙한다. 파이썬·토치·postgres 는 못 올라간다.
올라가는 건 트레이스 뷰어·평가 대시보드·스크린샷이다. 라이브 에이전트는 로컬 실행.

## A. 반드시

- [x] **A2 데모 모드** `dogcare --demo` — 스텁 MCP 서버 둘(`mcp_servers/*_stub_server.py`). 서브레포·가중치·DB 없이 LLM 키만으로 뜬다. 실측: health 2초, 상담 한 턴 10.7초(툴 4ms)
- [x] **A1 웹 UI** `dogcare serve` — FastAPI + 한 장짜리 페이지. 사진 업로드 · 끌고 늘리는 가이드 프레임 · SSE 단계 표시 · 게이트/조립 표시 · 판정 카드 · 기록 보기. 실측(데모): 상담 3.6초, 사진 3.1초
- [x] **A3 스크린샷** — `docs/assets/ui-*.png` (헤드리스 Chromium). 적대적 질문 장면에서 **G7 을 새로 찾았다** — group 이 null 인데 막대를 읽고 계열을 단정
- [x] **C1 MCP 서버 테스트** — `tests/test_mcp_servers.py`: 모델 뷰에 6종 어휘 0건, 스텁 계약 모양이 진짜와 같음
- [x] **C2 tools.json 드리프트 검사** — `tests/test_tools_snapshot.py`: 서버 코드의 `@mcp.tool` 이름과 스냅샷 대조, 스텁과 진짜의 툴 이름 일치
- [x] **A5 CI** — `.github/workflows/ci.yml`: ruff · pytest · 게이트 평가. ubuntu + windows 매트릭스
- [x] **A4 LICENSE + 재배포 정책 절** — MIT(코드만). 가중치·코퍼스·스크린샷 사진의 조건을 LICENSE 와 README 에 적음
- [ ] **공개 전환** ← 사용자가 직접. 순서:
  1. `gh repo edit gayeoniee/dog-care-agent --visibility public --accept-visibility-change-consequences`
  2. Settings → Pages → Source: **GitHub Actions** (private 레포는 Pages 가 유료라 공개 뒤에 켠다)
  3. `gh workflow run pages.yml` — `site/` 가 https://gayeoniee.github.io/dog-care-agent/ 로
  4. HF 릴리스 `gayoniee/daengs-skin-screening-release` 는 **private 그대로 둬도 된다** —
     README 가 데모 모드를 첫 화면으로 안내하고, 진짜 모드는 HF_TOKEN 이 있는 사람의 것이다.
     public 으로 돌리려면 모델카드에 AI Hub 파생·연구 목적 문구를 먼저 적는다

## B. 서사

- [x] **B1 e2e 게이트 발동률** — `evals/run.py adversarial`. 트레이스에 `first_pass_violations`·`repaired`·`composed` 를 남기고 센다. 실측 2회: 첫 초안 걸림 **4/24 → 1/23** (합 5/47), 전부 고쳐 쓰기로 통과, 조립 0, 최종 위반 0. 호출 오류 13 (무료 티어). 회차마다 다른 게 이 층의 성질이라 범위로 적음
- [x] **B2 적대적 평가** — `evals/adversarial.yaml` 10문항 × 3회. 걸린 게이트: G1(수의사가 농포라 했다), G5(사진 없이 판정해 달라), G6(검색하지 말고 답해 달라)
- [x] **B5 `dogcare stats`** — 첫 초안 걸림/고쳐쓰기/조립/막힘 · 게이트별 발동 · 툴 p50/p95 · 턴 p50/p95
- [x] **B4 멀티턴** — `run_turn(prior_screening=…)`. 웹 세션과 `dogcare chat` 이 마지막 판정을 들고 다닌다. 이어 묻기 "그거 궤양이야?" 에서 G1 이 잡는 테스트 추가
- [x] **B3 트레이스 뷰어 → GitHub Pages** — `site/index.html` + `tools/build_site.py`(사진 경로 지움) + `.github/workflows/pages.yml`. **Pages 켜기는 공개 전환과 함께 사용자가** (private 레포는 Pages 가 유료)

## C. 결함

- [x] **C3 지연 체감** — SSE 단계 표시(툴 부르는 중 → 게이트 → 고쳐 쓰는 중). 토큰 스트리밍을 안 하는 이유는 `server.py` 독스트링과 README 에
- [x] **C4 LLM 공급자** — `.env.example` 에 Gemini(기본) · LM Studio · Ollama 블록. 3.6-flash 503 과 3.1-flash-lite 실측을 적음
- [x] **C5 서브에이전트 다운 e2e 실측** — `SKIN_REPO` 를 없는 경로로 두고 사진 질문. 피부 툴이 `그런 툴이 없습니다`(0ms) 로 돌아오자 모델이 "사진 분석 기능이 준비되어 있지 않습니다" 라고 정직하게 답하고 RAG 로 이어감. 빠진 서브에이전트를 trace·CLI·화면에 적게 함(`subagent_failures`)
- [x] **C6 Windows 흔적** — CI ubuntu + windows 매트릭스 둘 다 초록 (run 35706344150)

## D. 하지 않는다

- 팀 저장소·서버 연결 (`CLAUDE.md` 규칙)
- 서브레포 수정 — 데모 모드는 여기서 MCP 스텁으로
- 라이브 배포에 먼저 매달리기 — Pages 정적 쇼케이스가 먼저
