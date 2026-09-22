"""LLM — OpenAI 호환 프로토콜이면 무엇이든 (Gemini · LM Studio · Ollama · vLLM).

`openai-compatible` 은 서비스가 아니라 **규격**입니다. 저쪽 RAG 가 같은 선택을
했고(`LLM_BASE_URL` 만 바꾸면 교체), 여기도 그대로 갑니다 — 두 저장소가 서로
다른 방식으로 LLM 을 물면 설정이 두 군데가 됩니다.

⚠️ **툴 호출을 지원하는 모델이어야 합니다.** 루프가 그걸로 돕니다.
   gemini-3.1-flash-lite 는 RAG 의 질의 재작성(30토큰)에는 충분하지만 여기서는
   부족합니다 — 라우팅은 분별이 필요한 일이라 기본을 flash 로 둡니다.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

import httpx

from dogcare.config import Settings

#: 기다렸다 다시 걸어야 하는 응답.
#:   429 무료 티어 호출 한도
#:   503 "This model is currently experiencing high demand" — 무료 티어에서 흔합니다
#:   500·502·504 게이트웨이 쪽 일시 장애
_RETRYABLE = frozenset({429, 500, 502, 503, 504})


class LLMError(RuntimeError):
    pass


class ToolCallingLLM:
    def __init__(self, settings: Settings) -> None:
        self._s = settings
        #: 이 인스턴스가 쓴 토큰 합. OpenAI 호환 응답의 `usage` 를 **버리고 있었다** —
        #: 턴당 비용을 모르면 "서비스" 라 부를 수 없다. 루프가 턴 끝에 trace 로 옮긴다.
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0
        if not settings.llm_api_key:
            raise LLMError("LLM_API_KEY 가 비어 있습니다 — .env 를 확인하세요")

    async def chat(self, messages: list[dict[str, Any]],
                   tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._s.llm_model,
            "messages": messages,
            "temperature": self._s.llm_temperature,
            "max_tokens": self._s.llm_max_tokens,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        last = ""
        attempts = max(1, self._s.llm_retries)
        async with httpx.AsyncClient(timeout=self._s.llm_timeout) as client:
            for i in range(attempts):
                try:
                    r = await client.post(
                        f"{self._s.llm_base_url.rstrip('/')}/chat/completions",
                        headers={"Authorization": f"Bearer {self._s.llm_api_key}"},
                        json=body,
                    )
                except httpx.RequestError as exc:
                    last = f"{type(exc).__name__}: {exc}"
                    await self._backoff(i, None)
                    continue

                if r.status_code == 200:
                    j = r.json()
                    u = j.get("usage") or {}
                    self.prompt_tokens += int(u.get("prompt_tokens") or 0)
                    self.completion_tokens += int(u.get("completion_tokens") or 0)
                    self.calls += 1
                    return j["choices"][0]["message"]

                # ★ 본문을 그대로 답니다. 400 일 때 "키가 틀렸나" 로 한참을 보낸 적이
                #   있는데 실은 툴 스키마의 anyOf 였습니다 (subagents._sanitize 참조).
                last = f"HTTP {r.status_code} — {r.text[:600]}"
                if r.status_code not in _RETRYABLE or i == attempts - 1:
                    raise LLMError(last)
                await self._backoff(i, r.headers.get("retry-after"))

        raise LLMError(f"{attempts}회 시도 후 실패 — {last}")

    async def _backoff(self, attempt: int, retry_after: str | None) -> None:
        """지수 백오프 + 지터.

        지터가 없으면 평가처럼 **연달아 거는 호출이 같은 박자로 다시 몰려서**
        같은 한도에 또 걸립니다. 서버가 `Retry-After` 를 주면 그걸 따릅니다.
        """
        if retry_after:
            try:
                await asyncio.sleep(min(float(retry_after), 60.0))
                return
            except ValueError:
                pass
        await asyncio.sleep(min(2.0 ** attempt, 30.0) * (0.5 + random.random()))
