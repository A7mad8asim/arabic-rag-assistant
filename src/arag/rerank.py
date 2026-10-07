"""Rerank a pool of retrieved chunks before they go to the model.

BM25 scores a chunk as a whole, so a 20-row chunk that mentions "Doha", "2023" and "hotel" in
*different* rows can outrank the chunk holding the one row that has all three. The rerankers
look at single rows instead:

- RowReranker ("rows", default): scores each chunk by its best single row, i.e. how many of the
  question's words and numbers (years, age groups ...) that one row contains, blended with the
  BM25 score. No model, about a millisecond per question.
- LLMReranker ("llm"): shows the model the question and the best-matching rows of each candidate
  and asks which candidates contain the answer. Slower (one extra model call) but reads meaning.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Protocol

from .index import Hit, Index, rrf_scores
from .llm import LLM
from .text import numbers_in, tokenize


class Reranker(Protocol):
    name: str

    def rerank(self, question: str, hits: list[Hit]) -> list[Hit]: ...


def _rows(text: str) -> tuple[str, list[str]]:
    header, _, body = text.partition("\n")
    return header, [r for r in body.split("\n") if r.strip()]


@dataclass
class RowMatch:
    score: float  # word coverage + number coverage of the best row, 0..2
    row: str


def _row_scores(question: str, header: str, rows: list[str]) -> list[float]:
    """For each row: share of the question's words found in the row (or the dataset title) plus share of its numbers."""
    q_words = set(tokenize(question))
    q_nums = numbers_in(question)
    title_words = set(tokenize(header))
    scores = []
    for row in rows:
        cover = len(q_words & (title_words | set(tokenize(row)))) / len(q_words) if q_words else 0.0
        nums = len(q_nums & numbers_in(row)) / len(q_nums) if q_nums else 0.0
        scores.append(cover + nums)
    return scores


def best_row(question: str, chunk_text: str, kind: str) -> RowMatch:
    header, rows = _rows(chunk_text)
    candidates = rows if kind == "rows" and rows else [chunk_text]
    scores = _row_scores(question, header, candidates)
    i = max(range(len(candidates)), key=lambda j: (scores[j], -j))
    return RowMatch(scores[i], candidates[i])


FOCUS_NOTE = "(only the rows that best match the question are shown)"


def focus_rows(question: str, chunk_text: str, kind: str, keep: int) -> str:
    """The chunk as the model should see it: its header and only the `keep` rows that best match the question.

    A 20-row chunk invites the model to quote a neighbouring row; this leaves it the few rows that can answer.
    Rows keep their original order. Card chunks, and chunks with `keep` rows or fewer, are returned unchanged.
    """
    header, rows = _rows(chunk_text)
    if keep <= 0 or kind != "rows" or len(rows) <= keep:
        return chunk_text
    scores = _row_scores(question, header, rows)
    top = sorted(range(len(rows)), key=lambda i: (-scores[i], i))[:keep]
    return "\n".join([header, FOCUS_NOTE] + [rows[i] for i in sorted(top)])


class RowReranker:
    name = "rows"

    def __init__(self, w_bm25: float = 0.5, w_row: float = 1.0):
        self.w_bm25, self.w_row = w_bm25, w_row

    def rerank(self, question: str, hits: list[Hit]) -> list[Hit]:
        if not hits:
            return hits
        top = max(h.score for h in hits) or 1.0
        scored = [(self.w_bm25 * h.score / top + self.w_row * best_row(question, h.chunk.text, h.chunk.kind).score, i, h)
                  for i, h in enumerate(hits)]
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [Hit(h.chunk, s) for s, _, h in scored]


LLM_SYSTEM = """You judge which numbered candidates contain the answer to a question about Qatar's statistics.
Each candidate shows a dataset title and its rows that best match the question.
Reply with only a JSON list of candidate numbers that contain the answer, best first, for example [3, 1].
Reply [] if none of them contains it."""


class LLMReranker:
    """Ask the model to pick the candidates that contain the answer; keep the rest in their prior order."""

    name = "llm"

    def __init__(self, llm: LLM, candidates: int = 12, rows_per_candidate: int = 3, prefilter: RowReranker | None = None):
        self.llm = llm
        self.candidates = candidates
        self.rows_per_candidate = rows_per_candidate
        self.prefilter = prefilter or RowReranker()

    def _snippet(self, question: str, hit: Hit) -> str:
        header, rows = _rows(hit.chunk.text)
        if hit.chunk.kind != "rows":
            return hit.chunk.text[:400]
        ranked = sorted(rows, key=lambda r: -best_row(question, header + "\n" + r, "rows").score)
        return header + "\n" + "\n".join(r[:300] for r in ranked[: self.rows_per_candidate])

    def rerank(self, question: str, hits: list[Hit]) -> list[Hit]:
        hits = self.prefilter.rerank(question, hits)
        pool = hits[: self.candidates]
        prompt = "\n\n".join(f"[{i}] {self._snippet(question, h)}" for i, h in enumerate(pool, 1))
        reply = self.llm.complete(LLM_SYSTEM, f"Candidates:\n\n{prompt}\n\nQuestion: {question}")
        picked: list[int] = []
        m = re.search(r"\[[\d,\s]*\]", reply or "")
        if m:
            try:
                picked = [int(n) for n in json.loads(m.group(0)) if 0 < int(n) <= len(pool)]
            except (ValueError, TypeError):
                picked = []
        order = list(dict.fromkeys(picked)) + [i for i in range(1, len(pool) + 1) if i not in picked]
        return [pool[i - 1] for i in order] + hits[self.candidates :]


def match_text(question: str, extra: list[str] | None) -> str:
    """The text rows are matched against: the question plus its translations (so dialect words that never
    appear in the data are backed up by their Modern Standard Arabic or English equivalents)."""
    return " ".join([question, *(extra or [])])


def retrieve(index: Index, question: str, k: int = 6, reranker: Reranker | None = None, pool: int = 30,
             expand: Callable[[str], list[str]] | None = None, extra: list[str] | None = None) -> list[Hit]:
    """BM25 (or hybrid) retrieval of a candidate pool, optionally reranked, cut to the top k.

    Extra queries (`extra`, or computed by `expand`, for example a QueryTranslator) widen the search:
    the pool is the reciprocal rank fusion of the searches for the question and each extra query, and
    the reranker matches rows against the question and its translations together.
    """
    if extra is None:
        extra = expand(question) if expand else []
    if not extra:
        candidates = index.search(question, pool if reranker else k)
    else:
        runs = [index.search(q, pool) for q in [question, *extra]]
        fused = rrf_scores([[h.chunk.id for h in run] for run in runs])
        chunks = {h.chunk.id: h.chunk for run in runs for h in run}
        order = sorted(fused, key=fused.get, reverse=True)[: pool if reranker else k]
        candidates = [Hit(chunks[cid], fused[cid]) for cid in order]
    return reranker.rerank(match_text(question, extra), candidates)[:k] if reranker else candidates[:k]


def make_reranker(name: str, llm: LLM | None = None) -> Reranker | None:
    if name in ("", "none"):
        return None
    if name == "rows":
        return RowReranker()
    if name == "llm":
        if llm is None:
            raise ValueError("the llm reranker needs a model")
        return LLMReranker(llm)
    raise ValueError(f"Unknown RERANK '{name}' (use none, rows or llm)")
