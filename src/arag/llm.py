"""LLM providers behind one small interface.

- OllamaLLM (default): a local model such as Qwen3-8B.
- AnthropicLLM (optional, `pip install -e ".[anthropic]"`): Claude via the API, to measure the quality gap.
"""

from __future__ import annotations

import re
from typing import Protocol

import requests

from .config import Settings


class LLMError(Exception):
    """The model could not be reached or did not return a usable response."""


class LLM(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str: ...


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    return _THINK.sub("", text or "").strip()


class OllamaLLM:
    THINKING_FAMILIES = ("qwen3", "deepseek-r1", "qwq", "magistral")

    def __init__(self, model: str, base_url: str = "http://localhost:11434", timeout_s: float = 300, num_ctx: int = 8192):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.num_ctx = num_ctx
        self.name = f"ollama:{model}"

    def complete(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "keep_alive": "30m",
            "options": {"temperature": 0, "seed": 7, "num_ctx": self.num_ctx},
        }
        if self.model.lower().startswith(self.THINKING_FAMILIES):
            payload["think"] = False
        try:
            r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout_s)
        except requests.ConnectionError as e:
            raise LLMError(f"Cannot reach Ollama at {self.base_url}. Start it, then run:  ollama pull {self.model}") from e
        except requests.Timeout as e:
            raise LLMError(f"Ollama did not answer within {self.timeout_s:g} seconds.") from e
        if r.status_code == 404:
            raise LLMError(f"Model '{self.model}' is not installed in Ollama. Run:  ollama pull {self.model}")
        if r.status_code >= 400:
            raise LLMError(f"Ollama error {r.status_code}: {r.text[:300]}")
        return strip_think(r.json().get("message", {}).get("content", ""))


class AnthropicLLM:
    def __init__(self, model: str = "claude-opus-5-5", max_tokens: int = 2000):
        try:
            import anthropic
        except ImportError as e:
            raise LLMError('The Claude provider needs the extra:  pip install -e ".[anthropic]"') from e
        self._anthropic = anthropic
        self.client = anthropic.Anthropic()  # credentials from ANTHROPIC_API_KEY
        self.model = model
        self.max_tokens = max_tokens
        self.name = f"anthropic:{model}"

    def complete(self, system: str, user: str) -> str:
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except self._anthropic.AnthropicError as e:
            raise LLMError(f"Claude API: {e}") from e
        return strip_think("".join(b.text for b in resp.content if b.type == "text"))


def make_llm(settings: Settings) -> LLM:
    if settings.llm_provider == "ollama":
        return OllamaLLM(settings.llm_model, settings.ollama_base_url)
    if settings.llm_provider == "anthropic":
        return AnthropicLLM(settings.llm_model)
    raise ValueError(f"Unknown LLM_PROVIDER '{settings.llm_provider}' (use ollama or anthropic)")
