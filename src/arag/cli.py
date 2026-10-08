"""Command line: arag fetch | index | embed | search | ask | eval."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime

from .config import DEFAULT_MODELS, EVAL_DIR, Settings
from .index import Index, add_vectors, build_index, embedder_for
from .llm import LLMError, make_llm
from .query import QueryTranslator
from .rerank import make_reranker, retrieve

RERANKERS = ["none", "rows", "llm"]


def _save(result: dict, name: str | None = None) -> None:
    out_dir = EVAL_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    (out_dir / f"{name or datetime.now().strftime('%Y%m%d-%H%M%S')}.json").write_text(text, encoding="utf-8")
    (out_dir / "latest.json").write_text(text, encoding="utf-8")


def _config_key(settings: dict) -> tuple:
    return (settings.get("retrieval", "bm25") == "hybrid", bool(settings.get("translate_query")),
            settings.get("focus_rows") or 0, {"none": 0, "rows": 1, "llm": 2}.get(settings.get("rerank"), 9))


def _label(settings: dict) -> str:
    parts = [settings.get("retrieval", "bm25")]
    if settings.get("translate_query"):
        parts.append("query translation")
    parts.append(f"rerank {settings.get('rerank', 'none')}")
    if settings.get("focus_rows"):
        parts.append(f"{settings['focus_rows']} rows shown")
    return " + ".join(parts)


def _suffix(settings: Settings) -> str:
    return (f"_focus{settings.focus_rows}" if settings.focus_rows else "") + \
           ("_hybrid" if settings.retrieval == "hybrid" else "") + ("_translate" if settings.translate_query else "")


def _ablation_table(results: list[dict], k: int) -> str:
    """Markdown tables, one row per saved configuration: the held-out test split first, then dev."""
    def cell(s, key):
        v = s.get(key)
        return "–" if v is None else (f"{v:.0f}%" if isinstance(v, float) else str(v))

    results = sorted(results, key=lambda r: _config_key(r["settings"]))
    lines = []
    for split, label in (("test", "Held-out test split (80 questions)"), ("dev", "Dev split (40 questions, used for tuning)")):
        lines += [f"**{label}**", "",
                  f"| Configuration | Evidence@1 | Evidence@{k} | Answer row shown to the model | Answer accuracy (EN / AR) "
                  "| Unanswerable refused | Wrong but shown | Withheld | Time per question |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for r in results:
            if split not in r["summary"]:
                continue
            s, en, ar = r["summary"][split], r["summary"][f"{split}_en"], r["summary"][f"{split}_ar"]
            if "evidence_shown" not in s and not r["settings"].get("focus_rows"):
                s = dict(s, evidence_shown=s.get(f"evidence@{k}"))  # runs saved before this metric: no focus, so the same
            secs = [x["seconds"] for x in r["rows"] if x.get("split") == split and "seconds" in x]
            lines.append(f"| {_label(r['settings'])} | {cell(s, 'evidence@1')} | {cell(s, f'evidence@{k}')} | "
                         f"{cell(s, 'evidence_shown')} | {cell(s, 'answer_accuracy')} ({cell(en, 'answer_accuracy')} / "
                         f"{cell(ar, 'answer_accuracy')}) | {cell(s, 'unanswerable_refused')} | {s.get('wrong_but_shown', '–')} | "
                         f"{s.get('withheld', '–')} | {sum(secs) / len(secs) if secs else 0:.1f} s |")
        lines.append("")
    return "\n".join(lines)


def _add_pipeline_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rerank", choices=RERANKERS)
    parser.add_argument("--focus", type=int, help="show the model only N best-matching rows per table chunk (0 = all)")
    parser.add_argument("--retrieval", choices=["bm25", "hybrid"], help="hybrid needs `arag embed` first")
    parser.add_argument("--translate", action=argparse.BooleanOptionalAction, default=None,
                        help="also search with the question translated into the other language")
    parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=None,
                        help="check that the answer's row matches every condition of the question; withhold if not")


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
    sub.add_parser("embed", help="add embedding vectors (EMBEDDING_MODEL, via Ollama) to the index, for hybrid retrieval")
    a = sub.add_parser("ask", help="ask a question")
    a.add_argument("question")
    a.add_argument("--provider", choices=["ollama", "anthropic"])
    _add_pipeline_options(a)
    s = sub.add_parser("search", help="show the retrieved chunks only (no answer)")
    s.add_argument("question")
    _add_pipeline_options(s)
    e = sub.add_parser("eval", help="score retrieval (and answers) on the gold set")
    e.add_argument("--gold", default=str(EVAL_DIR / "gold.jsonl"))
    e.add_argument("--retrieval-only", action="store_true", help="skip answering; score retrieval only")
    e.add_argument("--provider", choices=["ollama", "anthropic"])
    _add_pipeline_options(e)
    e.add_argument("--split", choices=["dev", "test", "large", "test2"], help="only this split of the gold set")
    e.add_argument("--ablation", action="store_true",
                   help="run the rerankers in --rerankers at the current settings and rebuild eval/results/ablation.md "
                        "from all saved runs")
    e.add_argument("--rerankers", default=",".join(RERANKERS), help="comma-separated rerankers for --ablation")
    args = p.parse_args(argv)

    settings = Settings.from_env()
    if getattr(args, "provider", None) and args.provider != settings.llm_provider:
        settings = settings.with_(llm_provider=args.provider, llm_model=DEFAULT_MODELS[args.provider])
    if getattr(args, "rerank", None):
        settings = settings.with_(rerank=args.rerank)
    if getattr(args, "focus", None) is not None:
        settings = settings.with_(focus_rows=args.focus)
    if getattr(args, "retrieval", None):
        settings = settings.with_(retrieval=args.retrieval)
    if getattr(args, "translate", None) is not None:
        settings = settings.with_(translate_query=args.translate)
    if getattr(args, "verify", None) is not None:
        settings = settings.with_(verify_row=args.verify)

    if args.cmd == "fetch":
        from .portal import fetch

        print(json.dumps(fetch(settings, limit=args.limit), indent=2))
        return 0
    if args.cmd == "index":
        index = build_index(settings)
        print(f"{len(index.chunks):,} chunks from {len({c.dataset_id for c in index.chunks}):,} datasets -> {settings.index_dir}")
        return 0
    if args.cmd == "embed":
        n = add_vectors(settings)
        print(f"{n:,} chunks embedded with {settings.embedding_model} -> {settings.index_dir / 'vectors.npy'}")
        return 0

    index = Index.load(settings.index_dir, embedder_for(settings))
    if settings.retrieval == "hybrid" and index.vectors is None:
        print("Error: hybrid retrieval needs vectors. Run:  ollama pull bge-m3  then  arag embed", file=sys.stderr)
        return 1
    needs_llm = (args.cmd == "ask" or settings.rerank == "llm" or settings.translate_query
                 or (args.cmd == "eval" and (args.ablation or not args.retrieval_only)))
    try:
        llm = make_llm(settings) if needs_llm else None
        reranker = make_reranker(settings.rerank, llm)
    except (LLMError, ValueError) as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1
    expand = QueryTranslator(llm) if settings.translate_query else None

    if args.cmd == "search":
        if expand:
            print("Also searching with:", "; ".join(expand(args.question)) or "(no translation)", "\n")
        for i, h in enumerate(retrieve(index, args.question, settings.top_k, reranker, expand=expand), 1):
            print(f"[{i}] {h.score:.2f}  {h.chunk.id}\n    {h.chunk.text[:300]}\n")
        return 0
    if args.cmd == "ask":
        from .pipeline import ask

        try:
            ans = ask(args.question, index, llm, settings.top_k, reranker, settings.focus_rows, expand, settings.verify_row)
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

        gold = [g for g in load_gold(args.gold) if not args.split or g.get("split", "dev") == args.split]
        meta = {"retrieval": settings.retrieval, "translate_query": settings.translate_query, "verify_row": settings.verify_row,
                "top_k": settings.top_k,
                "chunks": len(index.chunks), "focus_rows": settings.focus_rows,
                "model": None if llm is None else llm.name, "date": datetime.now().strftime("%Y-%m-%d")}
        if args.ablation:
            for name in [r.strip() for r in args.rerankers.split(",") if r.strip()]:
                logging.info("ablation: %s", _label(dict(meta, rerank=name)))
                result = evaluate(gold, index, llm, settings.top_k, make_reranker(name, llm), settings.focus_rows, expand,
                                  settings.verify_row)
                result["settings"] = dict(meta, rerank=name)
                _save(result, f"ablation_{name}{_suffix(settings)}")
            saved = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((EVAL_DIR / "results").glob("ablation_*.json"))]
            table = _ablation_table(saved, settings.top_k)
            (EVAL_DIR / "results" / "ablation.md").write_text(
                f"# Ablation ({meta['date']}, {meta['model']}, top {settings.top_k} chunks)\n\n{table}", encoding="utf-8")
            print(table)
            return 0
        result = evaluate(gold, index, None if args.retrieval_only else llm, settings.top_k, reranker,
                          settings.focus_rows, expand, settings.verify_row)
        result["settings"] = dict(meta, rerank=settings.rerank)
        _save(result)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
