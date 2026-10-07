"""Build, save and search the chunk index.

- BM25 over normalized, lightly stemmed tokens (default). Because every chunk carries both
  languages, an Arabic question and its English translation can hit the same chunk.
- Optional dense vectors (bge-m3 via Ollama), fused with BM25 by reciprocal rank fusion.
"""

from __future__ import annotations

import json
import logging
import math
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import requests

from .chunking import Chunk, dataset_chunks
from .config import Settings
from .portal import load_raw
from .text import tokenize

log = logging.getLogger(__name__)

Embedder = Callable[[Sequence[str]], np.ndarray]


class BM25:
    def __init__(self, docs: Sequence[Sequence[str]], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=float)
        self.avg = float(self.len.mean()) if len(docs) else 0.0
        df = Counter(t for d in self.tf for t in d)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.postings: dict[str, list[int]] = {}
        for i, d in enumerate(self.tf):
            for t in d:
                self.postings.setdefault(t, []).append(i)

    def scores(self, query: Sequence[str]) -> dict[int, float]:
        out: dict[int, float] = {}
        for t in set(query):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in self.postings[t]:
                f = self.tf[i][t]
                norm = self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
                out[i] = out.get(i, 0.0) + idf * f * (self.k1 + 1) / (f + norm)
        return out


def ollama_embedder(model: str, base_url: str, batch: int = 32, timeout_s: float = 300) -> Embedder:
    def embed(texts: Sequence[str]) -> np.ndarray:
        vecs = []
        for i in range(0, len(texts), batch):
            r = requests.post(f"{base_url.rstrip('/')}/api/embed", json={"model": model, "input": list(texts[i : i + batch])}, timeout=timeout_s)
            r.raise_for_status()
            vecs.extend(r.json()["embeddings"])
        v = np.asarray(vecs, dtype=np.float32)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)

    return embed


def rrf(rankings: Sequence[Sequence[int]], k: int = 60) -> list[int]:
    """Reciprocal rank fusion of several rankings (lists of document indices, best first)."""
    score: dict[int, float] = {}
    for ranking in rankings:
        for rank, i in enumerate(ranking):
            score[i] = score.get(i, 0.0) + 1 / (k + rank + 1)
    return sorted(score, key=score.get, reverse=True)


@dataclass
class Hit:
    chunk: Chunk
    score: float


class Index:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray | None = None, embed: Embedder | None = None):
        self.chunks = chunks
        self.bm25 = BM25([tokenize(c.text) for c in chunks])
        self.vectors = vectors
        self.embed = embed

    @classmethod
    def build(cls, datasets, portal_url: str, rows_per_chunk: int = 20, embed: Embedder | None = None) -> "Index":
        chunks = [c for d in datasets for c in dataset_chunks(d, portal_url, rows_per_chunk)]
        vectors = embed([c.text for c in chunks]) if embed else None
        return cls(chunks, vectors, embed)

    def search(self, query: str, k: int = 6, pool: int = 50) -> list[Hit]:
        bm = self.bm25.scores(tokenize(query))
        ranked = sorted(bm, key=bm.get, reverse=True)[:pool]
        if self.vectors is not None and self.embed is not None:
            try:
                q = self.embed([query])[0]
                dense = list(np.argsort(-(self.vectors @ q))[:pool])
                fused = rrf([ranked, dense])
                return [Hit(self.chunks[i], 0.0) for i in fused[:k]]
            except Exception as e:  # a missing embedding model must not break answering
                log.warning("dense retrieval failed, using BM25 only: %s", e)
        return [Hit(self.chunks[i], bm[i]) for i in ranked[:k]]

    # ------------------------------------------------------------------ persistence

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        with open(index_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        with open(index_dir / "bm25.pkl", "wb") as f:
            pickle.dump(self.bm25, f)
        if self.vectors is not None:
            np.save(index_dir / "vectors.npy", self.vectors)

    @classmethod
    def load(cls, index_dir: Path, embed: Embedder | None = None) -> "Index":
        if not (index_dir / "chunks.jsonl").exists():
            raise FileNotFoundError(f"No index in {index_dir}. Run:  arag fetch  then  arag index")
        self = cls.__new__(cls)
        with open(index_dir / "chunks.jsonl", encoding="utf-8") as f:
            self.chunks = [Chunk(**json.loads(line)) for line in f]
        with open(index_dir / "bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)
        vec = index_dir / "vectors.npy"
        self.vectors = np.load(vec) if vec.exists() else None
        self.embed = embed if self.vectors is not None else None
        return self


def embedder_for(settings: Settings) -> Embedder | None:
    return ollama_embedder(settings.embedding_model, settings.ollama_base_url) if settings.retrieval == "hybrid" else None


def build_index(settings: Settings) -> Index:
    index = Index.build(load_raw(settings.raw_dir), settings.portal_url, settings.rows_per_chunk, embedder_for(settings))
    index.save(settings.index_dir)
    return index
