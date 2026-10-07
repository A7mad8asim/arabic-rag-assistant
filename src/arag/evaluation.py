"""Score retrieval and answers against a gold set.

Gold set: JSON lines with
  id, lang (en|ar), question, kind (lookup | unanswerable),
  dataset_ids (datasets that answer it: the primary one first, then exact duplicates and other
  tables that hold the same figure; empty when unanswerable),
  answer (numbers that must appear in a correct answer), dialect (Gulf dialect question),
  split (dev: used to tune the system | test: held out)

Retrieval metrics, over the top k chunks the model sees:
- dataset hit@1 / hit@k / MRR: did a chunk from an accepted dataset come back (counted over distinct datasets)?
- evidence@1 / evidence@k: did the chunk holding the answer row (an accepted dataset's chunk containing
  every gold number) come back? This is the stricter test: the right table but the wrong rows fails it.
Answer metrics: a lookup is correct when the answer is grounded, contains every gold number and cites a
chunk of an accepted dataset; an unanswerable question is correct when the system says it could not find it.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .index import Hit, Index
from .llm import LLM
from .pipeline import ask
from .rerank import Reranker, retrieve
from .text import numbers_in


def load_gold(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _gold_numbers(item: dict) -> set[str]:
    return {n for v in item["answer"] for n in numbers_in(str(v))}


def dataset_rank(hits: list[Hit], dataset_ids: list[str]) -> int | None:
    """1-based rank of the first accepted dataset among the distinct datasets in `hits`, or None."""
    seen: list[str] = []
    for h in hits:
        if h.chunk.dataset_id not in seen:
            seen.append(h.chunk.dataset_id)
    ranks = [seen.index(d) + 1 for d in dataset_ids if d in seen]
    return min(ranks) if ranks else None


def evidence_rank(hits: list[Hit], item: dict) -> int | None:
    """1-based rank of the first chunk from an accepted dataset that contains every gold number, or None."""
    need = _gold_numbers(item)
    for i, h in enumerate(hits, 1):
        if h.chunk.dataset_id in item["dataset_ids"] and need <= numbers_in(h.chunk.text):
            return i
    return None


def retrieval_rank(index: Index, question: str, dataset_ids: list[str], k: int, reranker: Reranker | None = None) -> int | None:
    return dataset_rank(retrieve(index, question, k, reranker), dataset_ids)


def answer_correct(item: dict, ans) -> bool:
    if item["kind"] == "unanswerable":
        return ans.status == "not_found"
    if ans.status != "answered":
        return False
    cites_gold = any(h.chunk.dataset_id in item["dataset_ids"] for h in ans.sources)
    return _gold_numbers(item) <= numbers_in(ans.text) and cites_gold


def _pct(xs: list[bool]) -> float | None:
    return round(100 * sum(xs) / len(xs), 1) if xs else None


def summarize(rows: list[dict], k: int, with_answers: bool) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        for g in ("all", r["lang"], r.get("split", "dev"), f"{r.get('split', 'dev')}_{r['lang']}"):
            groups[g].append(r)
        if r.get("dialect"):
            groups["gulf_dialect"].append(r)
    summary = {}
    for name, rs in groups.items():
        lookups = [r for r in rs if r["kind"] == "lookup"]
        s = {
            "questions": len(rs),
            "hit@1": _pct([r["rank"] == 1 for r in lookups]),
            f"hit@{k}": _pct([r["rank"] is not None for r in lookups]),
            "mrr": round(sum(1 / r["rank"] for r in lookups if r["rank"]) / len(lookups), 3) if lookups else None,
            "evidence@1": _pct([r["evidence"] == 1 for r in lookups]),
            f"evidence@{k}": _pct([r["evidence"] is not None for r in lookups]),
        }
        if with_answers:
            s["answer_accuracy"] = _pct([r["correct"] for r in rs])
            s["lookup_accuracy"] = _pct([r["correct"] for r in lookups])
            s["unanswerable_refused"] = _pct([r["correct"] for r in rs if r["kind"] == "unanswerable"])
            s["wrong_but_shown"] = sum(1 for r in lookups if r["status"] == "answered" and not r["correct"])
            s["withheld"] = sum(1 for r in rs if r["status"] == "ungrounded")
        summary[name] = s
    return summary


def evaluate(gold: list[dict], index: Index, llm: LLM | None = None, k: int = 6, reranker: Reranker | None = None) -> dict:
    rows = []
    for item in gold:
        row = {"id": item["id"], "lang": item["lang"], "kind": item["kind"],
               "split": item.get("split", "dev"), "dialect": item.get("dialect", False)}
        if llm is not None:
            ans = ask(item["question"], index, llm, k, reranker)
            hits = ans.hits  # the chunks the model saw: retrieve (and an LLM rerank) only once
            row.update(status=ans.status, answer=ans.text, correct=answer_correct(item, ans), seconds=round(ans.seconds, 2),
                       cited=[h.chunk.dataset_id for h in ans.sources])
        else:
            hits = retrieve(index, item["question"], k, reranker)
        if item["kind"] == "lookup":
            row.update(rank=dataset_rank(hits, item["dataset_ids"]), evidence=evidence_rank(hits, item))
        rows.append(row)
    return {"summary": summarize(rows, k, llm is not None), "rows": rows}
