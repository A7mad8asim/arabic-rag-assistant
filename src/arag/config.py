"""Settings, read from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv

# The repository root (eval/, data/). Override with ARAG_HOME if the package is installed elsewhere.
PROJECT_ROOT = Path(os.getenv("ARAG_HOME") or Path(__file__).resolve().parents[2])
EVAL_DIR = PROJECT_ROOT / "eval"

load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_MODELS = {"ollama": "qwen3:8b", "anthropic": "claude-opus-5-5"}


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    llm_provider: str  # ollama (default, local) | anthropic
    llm_model: str
    ollama_base_url: str
    retrieval: str  # bm25 | hybrid
    embedding_model: str
    top_k: int
    portal_url: str
    publisher: str
    max_rows_per_dataset: int
    rows_per_chunk: int

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @classmethod
    def from_env(cls) -> "Settings":
        provider = _env("LLM_PROVIDER", "ollama").lower()
        data_dir = Path(_env("DATA_DIR", str(PROJECT_ROOT / "data")))
        if not data_dir.is_absolute():
            data_dir = PROJECT_ROOT / data_dir
        return cls(
            data_dir=data_dir,
            llm_provider=provider,
            llm_model=_env("LLM_MODEL", DEFAULT_MODELS.get(provider, "qwen3:8b")),
            ollama_base_url=_env("OLLAMA_BASE_URL", "http://localhost:11434"),
            retrieval=_env("RETRIEVAL", "bm25").lower(),
            embedding_model=_env("EMBEDDING_MODEL", "bge-m3"),
            top_k=int(_env("TOP_K", "6")),
            portal_url=_env("PORTAL_URL", "https://www.data.gov.qa").rstrip("/"),
            publisher=_env("PUBLISHER", "National Planning Council"),
            max_rows_per_dataset=int(_env("MAX_ROWS_PER_DATASET", "5000")),
            rows_per_chunk=int(_env("ROWS_PER_CHUNK", "20")),
        )

    def with_(self, **changes) -> "Settings":
        return replace(self, **changes)
