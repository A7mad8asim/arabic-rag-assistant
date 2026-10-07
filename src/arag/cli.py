"""Command line: arag fetch | index | search | ask | eval."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime

from .config import DEFAULT_MODELS, EVAL_DIR, Settings
from .index import Index, build_index, embedder_for
from .llm import LLMError, make_llm
from .rerank import make_reranker, retrieve

RERANKERS = ["none", "rows", "llm"]


def _save(result: dict, name: str | None = None) -> None:
    out_dir = EVAL_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    (out_dir / f"{name or datetime.now().strftime('%Y%m%d-%H%M%S')}.json").write_text(text, encoding="utf-8")
    (out_dir / "latest.json").write_text(text, encoding="utf-8")


def _ablation_table(results: dict[str, dict], k: int) -> str:
    """Markdown table: one row per configuration, test split first (held out), then dev."""
    def cell(s, key):
        v = s.get(key)
        return "–" if v is None else (f"{v:.0f}%" if isinstance(v, float) and key != "mrr" else str(v))

    lines = []
    for split, label in (("test", "Held-out test split (80 questions)"), ("dev", "Dev split (40 questions, used for tuning)")):
        lines += [f"**{label}**", "",
                  f"| Reranker | Evidence@1 | Evidence@{k} | Dataset hit@1 | Answer accuracy (EN / AR) | Unanswerable refused | Wrong but shown | Withheld |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for name, r in results.items():
            s, en, ar = r["summary"][split], r["summary"][f"{split}_en"], r["summary"][f"{split}_ar"]
            lines.append(f"| {name} | {cell(s, 'evidence@1')} | {cell(s, f'evidence@{k}')} | {cell(s, 'hit@1')} | "
                         f"{cell(s, 'answer_accuracy')} ({cell(en, 'answer_accuracy')} / {cell(ar, 'answer_accuracy')}) | "
                         f"{cell(s, 'unanswerable_refused')} | {s.get('wrong_but_shown', '–')} | {s.get('withheld', '–')} |")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")  # Arabic output on Windows consoles
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    p = argparse.ArgumentParser(prog="arag", description="Ask Qatar's open statistics, in Arabic or English.")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download the catalog and dataset rows from the Open Data portal")
    f.add_argument("--limit", type=int, help="only the first N datasets (for a quick try)")
    sub.add_parser("index", help="build the search index from the downloaded datasets")
    a = sub.add_parser("ask", help="ask a question")
    a.add_argument("question")
    a.add_argument("--provider", choices=["ollama", "anthropic"])
    a.add_argument("--rerank", choices=RERANKERS)
    s = sub.add_parser("search", help="show the retrieved chunks only (no answer)")
    s.add_argument("question")
    s.add_argument("--rerank", choices=RERANKERS)
    e = sub.add_parser("eval", help="score retrieval (and answers) on the gold set")
    e.add_argument("--gold", default=str(EVAL_DIR / "gold.jsonl"))
    e.add_argument("--retrieval-only", action="store_true", help="skip answering; score retrieval only")
    e.add_argument("--provider", choices=["ollama", "anthropic"])
    e.add_argument("--rerank", choices=RERANKERS)
    e.add_argument("--ablation", action="store_true", help="run every reranker and write eval/results/ablation.md")
    args = p.parse_args(argv)

    settings = Settings.from_env()
    if getattr(args, "provider", None) and args.provider != settings.llm_provider:
        settings = settings.with_(llm_provider=args.provider, llm_model=DEFAULT_MODELS[args.provider])
    if getattr(args, "rerank", None):
        settings = settings.with_(rerank=args.rerank)

    if args.cmd == "fetch":
        from .portal import fetch

        print(json.dumps(fetch(settings, limit=args.limit), indent=2))
        return 0
    if args.cmd == "index":
        index = build_index(settings)
        print(f"{len(index.chunks):,} chunks from {len({c.dataset_id for c in index.chunks}):,} datasets -> {settings.index_dir}")
        return 0

    index = Index.load(settings.index_dir, embedder_for(settings))
    needs_llm = args.cmd == "ask" or settings.rerank == "llm" or (args.cmd == "eval" and (args.ablation or not args.retrieval_only))
    try:
        llm = make_llm(settings) if needs_llm else None
        reranker = make_reranker(settings.rerank, llm)
    except (LLMError, ValueError) as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    if args.cmd == "search":
        for i, h in enumerate(retrieve(index, args.question, settings.top_k, reranker), 1):
            print(f"[{i}] {h.score:.2f}  {h.chunk.id}\n    {h.chunk.text[:300]}\n")
        return 0
    if args.cmd == "ask":
        from .pipeline import ask

        try:
            ans = ask(args.question, index, llm, settings.top_k, reranker)
        except LLMError as err:
            print(f"Error: {err}", file=sys.stderr)
            return 1
        print(ans.text, "\n")
        shown = ans.cited if ans.status == "answered" else range(1, min(3, len(ans.hits)) + 1)
        for i in shown:
            h = ans.hits[i - 1]
            print(f"  [{i}] {h.chunk.title_en} | {h.chunk.title_ar}\n      {h.chunk.url}")
        print(f"\n({ans.status}, {ans.seconds:.1f} s)")
        return 0
    if args.cmd == "eval":
        from .evaluation import evaluate, load_gold

        gold = load_gold(args.gold)
        meta = {"retrieval": settings.retrieval, "top_k": settings.top_k, "chunks": len(index.chunks),
                "model": None if llm is None else llm.name, "date": datetime.now().strftime("%Y-%m-%d")}
        if args.ablation:
            results = {}
            for name in RERANKERS:
                logging.info("ablation: reranker=%s", name)
                result = evaluate(gold, index, llm, settings.top_k, make_reranker(name, llm))
                result["settings"] = dict(meta, rerank=name)
                _save(result, f"ablation_{name}")
                results[name] = result
            table = _ablation_table(results, settings.top_k)
            (EVAL_DIR / "results" / "ablation.md").write_text(
                f"# Reranker ablation ({meta['date']}, {meta['model']}, {meta['retrieval']} retrieval, top {settings.top_k})\n\n{table}",
                encoding="utf-8")
            print(table)
            return 0
        result = evaluate(gold, index, None if args.retrieval_only else llm, settings.top_k, reranker)
        result["settings"] = dict(meta, rerank=settings.rerank)
        _save(result)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
