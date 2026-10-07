"""Command line: arag fetch | index | ask | eval."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime

from .config import DEFAULT_MODELS, EVAL_DIR, Settings
from .index import Index, build_index, embedder_for
from .llm import LLMError, make_llm


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")  # Arabic output on Windows consoles
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    p = argparse.ArgumentParser(prog="arag", description="Ask Qatar's open statistics, in Arabic or English.")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download the catalog and dataset rows from the Open Data portal")
    f.add_argument("--limit", type=int, help="only the first N datasets (for a quick try)")
    sub.add_parser("index", help="build the search index from the downloaded datasets")
    a = sub.add_parser("ask", help="ask a question")
    a.add_argument("question")
    a.add_argument("--provider", choices=["ollama", "anthropic"])
    s = sub.add_parser("search", help="show the retrieved chunks only (no model)")
    s.add_argument("question")
    e = sub.add_parser("eval", help="score retrieval (and answers) on the gold set")
    e.add_argument("--gold", default=str(EVAL_DIR / "gold.jsonl"))
    e.add_argument("--retrieval-only", action="store_true", help="skip the model; score retrieval only")
    e.add_argument("--provider", choices=["ollama", "anthropic"])
    args = p.parse_args(argv)

    settings = Settings.from_env()
    if getattr(args, "provider", None) and args.provider != settings.llm_provider:
        settings = settings.with_(llm_provider=args.provider, llm_model=DEFAULT_MODELS[args.provider])

    if args.cmd == "fetch":
        from .portal import fetch

        print(json.dumps(fetch(settings, limit=args.limit), indent=2))
        return 0
    if args.cmd == "index":
        index = build_index(settings)
        print(f"{len(index.chunks):,} chunks from {len({c.dataset_id for c in index.chunks}):,} datasets -> {settings.index_dir}")
        return 0

    index = Index.load(settings.index_dir, embedder_for(settings))
    if args.cmd == "search":
        for i, h in enumerate(index.search(args.question, settings.top_k), 1):
            print(f"[{i}] {h.score:.2f}  {h.chunk.id}\n    {h.chunk.text[:300]}\n")
        return 0
    if args.cmd == "ask":
        from .pipeline import ask

        try:
            ans = ask(args.question, index, make_llm(settings), settings.top_k)
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

        llm = None if args.retrieval_only else make_llm(settings)
        result = evaluate(load_gold(args.gold), index, llm, settings.top_k)
        result["settings"] = {"retrieval": settings.retrieval, "top_k": settings.top_k,
                              "model": None if llm is None else llm.name, "chunks": len(index.chunks)}
        out = EVAL_DIR / "results" / f"{datetime.now():%Y%m%d-%H%M%S}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        (EVAL_DIR / "results" / "latest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
