---
title: dog-care-agent
emoji: 🐕
colorFrom: yellow
colorTo: green
sdk: gradio
sdk_version: 6.28.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: 근거 RAG + 피부 스크리닝을 MCP 서브에이전트로 두는 반려견 상담 오케스트레이터
---

# 🐕 dog-care-agent — 라이브 데모

**보호자의 말 한 줄과 사진 한 장을 받아, 근거 RAG 와 피부 스크리닝 모델을 서브에이전트로
부리는 오케스트레이터.** 요점은 라우팅이 아니라 — **모델이 말해도 되는 것을 코드가 정한다**는 데 있다.

이 Space 는 **데모 모드**다. 서브에이전트 둘이 계약 모양만 같은 스텁이라 상담 본문은
고정값이고, 피부 판정은 사진 내용과 무관하게 파일 이름으로 정상/이상을 고른다.
오케스트레이터 · 결정론적 게이트(G1~G8) · 고쳐쓰기 · 조립 · 트레이스는 **진짜와 똑같이** 돈다.

- 소스 · 게이트 표 · 평가 숫자 · 진짜 모드 실행법: https://github.com/gayeoniee/dog-care-agent
- 올린 사진은 판정 직후 지운다. 트레이스에는 파일 이름만 남는다.
- 무료 Gemini 키 한 개로 돈다 — 분당 제한이 있고, 무료 티어는 모델당 하루 500회라 한도 안내가 보이면 다음 날.
- 실제로 돌린 트레이스·평가 결과 뷰어(정적): https://thusfar.cloud/dog-care-agent/

> 이 서비스는 수의사의 진료를 대체하지 않습니다.
