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
- [x] **공개 전환** — 사용자, 2026-09-22. 순서였던 것:
  1. `gh repo edit gayeoniee/dog-care-agent --visibility public --accept-visibility-change-consequences`
  2. Settings → Pages → Source: **GitHub Actions** (private 레포는 Pages 가 유료라 공개 뒤에 켠다)
  3. `gh workflow run pages.yml` — `site/` 가 https://thusfar.cloud/dog-care-agent/ 로
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

---

# 2차 — 완성본에서 서비스로 (2026-09-22 저녁)

"더 완벽한 서비스" 를 위해 코드를 뒤져 나온 구멍과, 평가에서 빠져 있던 것.

## S. 위생 — 공개 데모를 열려면 있어야 하는 것

- [x] **S1 사진 영구 보관** — 판정 뒤 삭제 (`KEEP_UPLOADS=1` 로만 남김). 7장 남아 있던 걸 지움
- [x] **S2 질문 길이 제한** — 1,000자 (저쪽 `SkinPayload.question` 과 같은 상한)
- [x] **S3 세션 무한 증가** — 30분 만료 · 200개 상한
- [x] **S4 이미지 검증** — 매직 바이트로 JPEG/PNG/WebP
- [x] **S5 레이트 리밋** — IP 당 분당 10회 · 동시 2 (`RATE_PER_MIN` · `MAX_CONCURRENT`)
- [x] **S6 서버 테스트** — `tests/test_server.py` 12개
- [x] **S7 트레이스 절대경로** — `Trace.scrubbed()` 가 파일 이름만 남김

## E. 평가 — 안전만 재고 "답이 좋은가" 를 안 재던 것

- [x] **E1 G8 판정 충실도** — abnormal 을 정상처럼 · normal 을 이상처럼 · retake 에 판정. "진료 권고 있어야" 는 게이트가 아니라 **코드가 붙인다**(면책과 같은 논리, 저쪽 `plan_actions` 와 같은 자리)
- [x] **E2 LLM-as-judge** — `evals/judge.py`. 사람 라벨 13건 보정: 1회차 71% → form 결정론화 · 코드 부착 문장 제외 → 2회차 **98%**. 로컬 granite4.1-8k 판정기는 78%(verdict_fidelity 4/7)로 기준 아래 — 그 채점(판정 전달 49%)은 판정기 노이즈라 안 씀
- [x] **E3 회귀 비교** — `freeze` · `compare`. CI 가 게이트 결과를 기준선과 대조
- [x] **E4 토큰·비용** — `usage` 를 trace 에, `stats` 가 p50/p95
- [x] **E5 평가셋 확장** — 라우팅 15 → 30 (Gemini 3회 **27/30**, 둘은 기대값 수정·하나는 G5 어휘 확장), 적대적 10 → 20 (Gemini 재측정은 429 로 31턴 중 29턴 실패 — **미측정**, 쿼터 풀리면)

## D. 서비스

- [x] **D1 Dockerfile** — 데모 모드 기본, HF Spaces 용. CI 의 docker 잡이 빌드·기동 확인
- [x] **D2 HF Spaces 라이브 데모** — https://huggingface.co/spaces/gayoniee/dog-care-agent (세부는 아래 '남은 것')
- [x] **D3 Ollama** — 설치. 후보 4종 프로브: command-r7b 툴 0/2 ✗ · qwen3.5:9b 사진 턴 ✗ · **qwen3:8b 3/3** · **granite4.1:8b 3/3 (5배 빠름)**. 라우팅 30문항(각 1회): granite 18·16·17, **qwen3 23·22·21**. 로컬은 툴 하나짜리엔 충분, 사진+상담 한 턴은 약함. **최종 선택은 사용자와** — 숫자상 qwen3:8b(8K 변형)
- [x] **D4 데모 GIF** — `docs/assets/demo.gif` 39프레임
- [x] **D5 빠른 실패** — LLM 키 없으면 MCP 서버 띄우기 전에
- [x] **D6 `/api/stats`** + 헤더 칩

## 틀렸던 것 (2차)

- `max_tokens` 를 안 보내고 있었다. Gemini 는 알아서 끊어서 안 보였고, granite 는 한
  요청에서 1,000토큰 넘게 이어 갔다. 1,024 로 못 박음.
- G8 에 "진료 권고가 없으면" 을 넣었더니 헛걸림 2건 — 면책 문구 안의 "수의사의 진료를
  대체하지" 가 권고로 보였다. 게이트가 아니라 코드가 붙이는 것으로.
- 판정기 1회차 71%: 없는 마크다운을 "있다" 고 하고, 코드가 붙인 면책을 "보탠 조언" 으로 셌다.

## 남은 것 (2차 끝)

- [x] 적대적 20×3 Gemini 재측정 — 2026-09-23 16:13, **60턴 · 오류 0 · 첫 초안 걸림 12/60 (G6 6 · G7 5 · G5 1) · 고침 11 · 조립 1 · 최종 위반 0**. `freeze` 로 기준선 갱신(routing 27/30 · adversarial 60턴).
  **쿼터의 정체(2026-09-22 23:40 실측)**: `GenerateRequestsPerDayPerProjectPerModel-FreeTier` = **500회/일/모델**.
  하루치를 다 썼다 — 재시도는 60턴 중 55 ERR. 리셋은 태평양 자정 = **한국 16:00**. 재측정(≤180회) + judge(≤60회)
  + freeze 는 리셋 직후 한 번에 돌리면 들어간다. **Space 도 같은 키라 그때까지 429** — 데모용 키를 다른
  Google Cloud 프로젝트에서 따로 만들면 평가가 데모 쿼터를 먹지 않는다 (Secret 만 바꾸면 됨)
- [x] Gemini 판정기로 적대적 3회차 60턴 채점 — 아래 '3차' 끝에 숫자
- [x] 로컬 모델 최종 선택 — **qwen3:8b (8K 변형 `qwen3-8k`)**, 사용자 선택 2026-09-22. `.env.example` (B) 블록
- [x] D2 HF Spaces 라이브 데모 — **https://huggingface.co/spaces/gayoniee/dog-care-agent** (2026-09-22, RUNNING · `/api/health` 200). 공개 전환과 함께. **Docker SDK 가 유료로
  잠겨 있어서**(new-space 화면에 Paid 배지) 계획을 바꿨다 — **Gradio SDK 위에 FastAPI 를 그대로**
  띄운다. Gradio 런타임은 `python app.py` 를 돌리고 7860 이 열리길 기다릴 뿐이다. 절차:
  1. HF 토큰에 **쓰기 권한** — fine-grained 면 "Write access to contents/settings of all repos
     under your personal namespace". 지금 토큰(`DAENGS_APP`)은 릴리스 모델 읽기뿐이라 403
  2. `uv run tools/push_space.py` — Space 생성 · Secret `LLM_API_KEY` · Variables · 업로드까지 한 번에.
     `space/app.py` + `space/README.md`(YAML 머리말) + `src/` + 스텁 둘 + 직접 의존성 requirements.txt 만 올린다
  3. 뜨면 README 상단에 Space 링크 — 달았다
  - 걸렸던 것 셋: ① 무료 CPU Basic 도 PRO 전용(402) → 무료는 **ZeroGPU** 뿐이라 `zero-a10g` 로 만든다.
    ② 런타임이 `gradio[mcp]` 를 같이 깔아 `mcp<2` 를 요구 → lock 대신 직접 의존성만, Space 에서만 `mcp>=1.21,<2`
    (1.30 에서 119 테스트 통과). ③ ZeroGPU 는 `@spaces.GPU` 함수 목록의 startup report 가 없으면 죽는데
    그 보고는 `gr.Blocks.launch` 훅이 보낸다 → gradio 를 안 띄우니 `startup_report()` 를 직접 부른다 (`space/app.py`)
  - 라이브 e2e: 요청 → SSE → 결과까지 돌았고, LLM 호출은 Gemini 일일 쿼터 429 (재측정 대기와 같은 원인)
- [x] 공개 전환 — 사용자, 2026-09-22

---

# 3차 — 공개 뒤 손본 것 (2026-09-23, 쿼터 리셋을 기다리는 동안)

- [x] **GitHub Pages 켬** — API 로 build_type=workflow, `pages.yml` 재실행 → https://thusfar.cloud/dog-care-agent/ 200
- [x] **레포 메타** — homepage = Space, topics 8개
- [x] **429 가 화면에 JSON 으로 찍히던 것** — `server.user_facing_error`: 한도·과부하·시간초과는 보호자 문장으로, 나머지는 첫 줄만. 테스트 1개
- [x] **judge 가 트레이스 폴더째 채점하던 것** — 폴더는 회차마다 쌓인다(옛 47 + 반쯤 죽은 회차). 적대적 결과 JSON 에 회차별 `trace` 파일명을 남기고, `judge.py score` 기본이 **최신 회차만** 본다
- [x] **CI 에 mcp 1.x 잡** — Space 런타임과 같은 requirements(`push_space.py --requirements-out`)로 pytest 한 번 더. 1.x 호환층이 깨지면 여기서 빨개진다
- [x] **데모용 키 분리** — `SPACE_LLM_API_KEY` 가 있으면 Space Secret 은 그 키. 평가가 데모 쿼터를 안 먹는다
- [x] **기본 모델을 실측한 모델로** — config·.env.example 기본값 3.6-flash → **3.1-flash-lite** (숫자가 전부 이 모델)
- [x] **README 낡은 숫자** — 게이트 7 → 8(G8 행 추가), 테스트 114 → 120, Pages 링크, Space 가 48시간 뒤 잠든다는 안내

남은 것은 그대로: 16:00 리셋 뒤 적대적 재측정 → `judge.py score`(이제 최신 회차만) → `freeze` → README 숫자.

- [x] **16:13 재측정 완료** — 60턴 · 오류 0 · 첫 초안 걸림 12/60 · 고침 11 · 조립 1 · 최종 위반 0 · p50 4.2s · p95 9.4s. `freeze` 갱신, `build_site` 갱신
- [x] **judge 채점** — 1차 `no_extra_claims` 52%: 판정기 요약에 툴의 `body`·계열 `text/feature/caveat` 가 빠져 원본 문장을 "보탠 말" 로 셈. 요약을 넓히고 "생략은 왜곡 아님" 을 적어 3차: 보탠 말 없음 85% · 판정 충실 45/45. 보정은 98% → 93% 로 내려감 — 픽스처 계열에 `text` 가 없어 `문장: None` 이 찍힌 탓. 계열 문장 틀을 코드가 채우고 "단계 증가는 keeps_steps 감점 아님" 을 적어 **보정 98%(44/45)**, 4차 채점 **단계 6/6 · 인용 6/6 · 보탠 말 없음 49/60(82%) · 판정 충실 47/47 · 형식 60/60**
- [x] 판정기가 잡은 진짜 것: JSON 강요 문항에서 모델이 headline 을 빼고 JSON 만 낸 것(3/3) — 게이트는 안 잡는다(G8 은 모순만 본다). **판정 문장도 코드가 붙인다** (`_attach_disclaimer`, verdict 별 열쇠말로 같은 뜻이면 안 겹침). 테스트 2개, 122 통과

- 09-23 08:28 에 프로브 200 · Space 답변 성공을 보고 재측정을 돌렸더니 **또 60턴 중 55 ERR**(같은
  일일 쿼터). 한도를 넘은 뒤에도 드문드문 통과하므로 **한 번의 200 은 증거가 아니다.** 결과 파일은
  지웠다. 16:00 리셋 뒤에만 돌린다.
