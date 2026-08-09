"""A small wrapper around the OpenAI client.

Everything that talks to the model goes through here, which buys three things:

  - one place to configure the key, model and temperature
  - one place to add retries, so a transient 429 does not abandon a 60-question
    evaluation run halfway through
  - one place to count tokens, so you can report what the project actually cost

Temperature is 0 by default and that is not a stylistic choice. Evaluation
compares runs against each other; if the model produces a different answer each
time, you cannot tell whether a change in your score came from your change or
from the sampler.
"""

from __future__ import annotations

import json
import time
from typing import Any

from copilot.config import settings


class LLMError(RuntimeError):
    pass


class MissingAPIKey(LLMError):
    pass


class LLMClient:
    """Chat completions with JSON-schema-constrained output."""

    def __init__(self, model: str | None = None, *, api_key: str | None = None, max_retries: int = 4):
        self.model = model or settings.llm_model
        self.api_key = api_key or settings.openai_api_key
        self.max_retries = max_retries
        self._client = None

        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0

    @property
    def client(self):
        if self._client is None:
            if not self.api_key or self.api_key.startswith("sk-replace"):
                raise MissingAPIKey(
                    "OPENAI_API_KEY is not set.\n"
                    "  1. Copy-Item .env.example .env\n"
                    "  2. put your key in .env  (platform.openai.com/api-keys)"
                )
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def complete_json(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        *,
        schema_name: str = "response",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[dict[str, Any], str]:
        """Call the model and get back parsed JSON matching `schema`.

        Uses structured outputs, so the API itself enforces the schema. The
        alternative - asking nicely for JSON in the prompt and parsing whatever
        comes back - fails a few percent of the time, and a few percent across a
        60-question evaluation is several broken runs.
        """
        temperature = settings.llm_temperature if temperature is None else temperature

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=temperature,
                    max_tokens=max_tokens or settings.max_answer_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": schema_name, "schema": schema, "strict": True},
                    },
                )
            except Exception as exc:  # noqa: BLE001 - retry on anything transient
                last_error = exc
                if attempt == self.max_retries - 1:
                    break
                # Exponential backoff: 1s, 2s, 4s. Retrying immediately during a
                # rate limit just burns the next attempt too.
                time.sleep(2**attempt)
                continue

            self.calls += 1
            if response.usage:
                self.prompt_tokens += response.usage.prompt_tokens
                self.completion_tokens += response.usage.completion_tokens

            raw = response.choices[0].message.content or "{}"
            return json.loads(raw), raw

        raise LLMError(f"LLM call failed after {self.max_retries} attempts: {last_error}")

    def estimated_cost_usd(self) -> float:
        """Rough running cost, for reporting at the end of an evaluation.

        gpt-4o-mini pricing as of writing: $0.15 per 1M input tokens, $0.60 per
        1M output. Hardcoded rather than fetched - it is a report line, not a bill.
        """
        return (self.prompt_tokens / 1_000_000) * 0.15 + (self.completion_tokens / 1_000_000) * 0.60

    def usage_summary(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd(), 4),
        }


_default: LLMClient | None = None


def get_llm() -> LLMClient:
    global _default
    if _default is None:
        _default = LLMClient()
    return _default
