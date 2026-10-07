"""Score retrieval and answers against a gold set.

Gold set: JSON lines with
  id, lang (en|ar), question, kind (lookup | unanswerable),
  dataset_ids (datasets that answer it: the primary one first, then exact duplicates and other
  tables that hold the same figure; empty when unanswerable),
  answer (numbers that must appear in a correct answer), dialect (Gulf dialect question)

Retrieval metrics are dataset-level (did a chunk from an accepted dataset come back?):
hit@1, hit@k and MRR, counted over distinct datasets. Answer metrics: a lookup is correct when
the answer is grounded, contains every gold number and cites a chunk of an accepted dataset;
an unanswerable question is correct when the system says it could not find the answer.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .index import Index
from .llm import LLM
from .pipeline import ask
from .text import numbers_in


def load_gold(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def retrieval_rank(index: Index, question: str, dataset_ids: list[str], k: int) -> int | None:
    """1-based rank of the first accepted dataset among the distinct datasets retrieved, or None."""
    seen: list[str] = []
    for h in index.search(question, k):
        if h.chunk.dataset_id not in seen:
            seen.append(h.chunk.dataset_id)
    ranks = [seen.index(d) + 1 for d in dataset_ids if d in seen]
    return min(ranks) if ranks else None


def answer_correct(item: dict, ans) -> bool:
    if item["kind"] == "unanswerable":
        return ans.status == "not_found"
    if ans.status != "answered":
        return False
    gold_numbers = {n for v in item["answer"] for n in numbers_in(str(v))}
    cites_gold = any(h.chunk.dataset_id in item["dataset_ids"] for h in ans.sources)
    return gold_numbers <= numbers_in(ans.text) and cites_gold


def _pct(xs: list[bool]) -> float | None:
    return round(100 * sum(xs) / len(xs), 1) if xs else None


def evaluate(gold: list[dict], index: Index, llm: LLM | None = None, k: int = 6) -> dict:
    rows = []
    for item in gold:
        row = {"id": item["id"], "lang": item["lang"], "kind": item["kind"]}
        if item["kind"] == "lookup":
            rank = retrieval_rank(index, item["question"], item["dataset_ids"], k)
            row.update(rank=rank, hit1=rank == 1, hitk=rank is not None, rr=1 / rank if rank else 0.0)
        if llm is not None:
            ans = ask(item["question"], index, llm, k)
            row.update(status=ans.status, answer=ans.text, correct=answer_correct(item, ans), seconds=round(ans.seconds, 2))
        rows.append(row)

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups["all"].append(r)
        groups[r["lang"]].append(r)
    summary = {}
    for name, rs in groups.items():
        lookups = [r for r in rs if r["kind"] == "lookup"]
        s = {
            "questions": len(rs),
            "hit@1": _pct([r["hit1"] for r in lookups]),
            f"hit@{k}": _pct([r["hitk"] for r in lookups]),
            "mrr": round(sum(r["rr"] for r in lookups) / len(lookups), 3) if lookups else None,
        }
        if llm is not None:
            s["answer_accuracy"] = _pct([r["correct"] for r in rs])
            s["unanswerable_refused"] = _pct([r["correct"] for r in rs if r["kind"] == "unanswerable"])
        summary[name] = s
    return {"summary": summary, "rows": rows}
