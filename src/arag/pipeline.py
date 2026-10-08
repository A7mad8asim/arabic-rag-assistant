"""Question → retrieved sources → (focused rows) → cited answer → grounding check.

The rule, as in Ask-the-Data: the model may only repeat numbers that appear in the sources it
cites. An answer with a number that is not in its cited sources is not shown; the reader gets
the sources instead. With row focus, each table chunk is cut to the rows that best match the
question before the model sees it, and the check uses exactly the text the model saw.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable

from .index import Hit, Index
from .llm import LLM
from .rerank import Reranker, focus_rows, match_text, retrieve
from .text import is_arabic, numbers_in

NOT_FOUND = "NOT_FOUND"

SYSTEM = f"""You answer questions about official statistics of the State of Qatar, using ONLY the numbered sources given.

Rules:
- Answer in the language of the question (Arabic or English), in 1 to 3 sentences.
- Copy every number exactly as it appears in a source. Never calculate, round, estimate or convert numbers.
- After each fact, cite the source number in square brackets, for example [2]. Every sentence with a number needs a citation.
- Mention the year (and the unit, if the source gives one) of each figure. QR or QAR means Qatari riyals (in Arabic: ريال قطري); KG means kilograms (كيلوغرام).
- If the sources do not contain the answer, reply with exactly: {NOT_FOUND}
"""

# Citations as the model writes them: [2], [n2], [1, 2], [١] ...
_CITE = re.compile(r"\[\s*(?:n|source\s*|مصدر\s*)?([0-9٠-٩]+(?:\s*[,،]\s*[0-9٠-٩]+)*)\s*\]", re.IGNORECASE)
_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


# The model sometimes declines in words instead of the NOT_FOUND token ("The provided data does not include ...").
_DECLINE = re.compile(
    r"\b(?:(?:does|do|did) not (?:include|contain|provide|have|specify|mention|cover|list|show|give|report)"
    r"|(?:is|are) not (?:available|included|provided|mentioned|listed|given)|no (?:data|information|figures?|records?) "
    r"(?:is|are|was|were|on|for|about)|not (?:found|available) in|cannot (?:be )?(?:found|determined|answered)|unable to)\b"
    r"|لا (?:يوجد|توجد|يتوفر|تتوفر|تحتوي|يحتوي|تتضمن|يتضمن|تشمل|يشمل|تذكر|يذكر)|لم (?:يتم|يرد|ترد|أجد|نجد|تذكر|يذكر)"
    r"|غير (?:متوفر|متوفرة|متاح|متاحة|موجود|موجودة|مذكور|مذكورة)|ليست? (?:متوفر|متاح|موجود)",
    re.IGNORECASE,
)


def declined(answer: str, question: str = "") -> bool:
    """True when the reply is not an answer: the NOT_FOUND token, a decline phrase, or no figure beyond the
    numbers already in the question.

    Every question here asks for a figure. A reply that only repeats the question's numbers ("Qatar won the
    2022 final.") is treated as "not found", never shown. (A new figure without a citation is not a decline:
    the grounding check withholds it and shows the sources instead.)
    """
    body = _CITE.sub(" ", answer or "")
    if not body.strip() or NOT_FOUND in body or _DECLINE.search(body):
        return True
    return not (numbers_in(body) - numbers_in(question))


def citations(answer: str) -> list[int]:
    return [int(n) for group in _CITE.findall(answer) for n in re.split(r"\s*[,،]\s*", group.translate(_AR_DIGITS))]


@dataclass
class Answer:
    question: str
    text: str
    status: str  # answered | not_found | ungrounded | error
    hits: list[Hit]
    cited: list[int] = field(default_factory=list)  # 1-based source numbers the answer cites
    ungrounded_numbers: list[str] = field(default_factory=list)
    raw: str = ""
    seconds: float = 0.0
    context: list[str] = field(default_factory=list)  # each source's text exactly as the model saw it
    extra_queries: list[str] = field(default_factory=list)  # e.g. the question's translation, also searched

    @property
    def sources(self) -> list[Hit]:
        return [self.hits[i - 1] for i in self.cited if 0 < i <= len(self.hits)]


def build_context(question: str, hits: list[Hit], focus: int = 0) -> list[str]:
    """Each source's text as the model will see it: whole, or cut to its `focus` best-matching rows."""
    return [focus_rows(question, h.chunk.text, h.chunk.kind, focus) for h in hits]


def build_prompt(question: str, context: list[str]) -> str:
    blocks = [f"[{i}] {text}" for i, text in enumerate(context, 1)]
    return "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"


def check_grounding(answer: str, question: str, context: list[str]) -> tuple[list[int], list[str]]:
    """The cited source numbers, and any number in the answer found neither in those sources nor in the question."""
    cited = sorted({n for n in citations(answer) if 0 < n <= len(context)})
    body = _CITE.sub(" ", answer)
    allowed = numbers_in(question)
    for i in cited:
        allowed |= numbers_in(context[i - 1])
    missing = sorted(n for n in numbers_in(body) if n not in allowed)
    if numbers_in(body) and not cited:
        missing = sorted(numbers_in(body) - numbers_in(question))
    return cited, missing


VERIFY_SYSTEM = """You check whether one row of a statistics table answers a question exactly.
Compare the question with the table title and the row: the year or period, the place, the group (gender, nationality, age),
the category, and what is measured (a count, a value in riyals, a percentage, a rate ...).
Reply with only "yes" if the row matches every condition the question states, otherwise only "no"."""


def answer_rows(answer: str, question: str, context: list[str], cited: list[int]) -> list[tuple[str, str]]:
    """(table title, row) pairs in the cited sources that contain one of the answer's own figures."""
    figures = numbers_in(_CITE.sub(" ", answer)) - numbers_in(question)
    out = []
    for i in cited:
        header, _, body = context[i - 1].partition("\n")
        for row in body.split("\n"):
            if row.strip() and numbers_in(row) & figures and (header, row) not in out:
                out.append((header, row))
    return out


def row_matches(llm: LLM, question: str, header: str, row: str) -> bool:
    reply = llm.complete(VERIFY_SYSTEM, f"Question: {question}\n\nTable: {header}\nRow: {row}")
    return (reply or "").strip().strip("\"'.").lower().startswith(("yes", "نعم"))


def _message(kind: str, arabic: bool) -> str:
    messages = {
        "not_found": ("I could not find this in Qatar's open statistics.",
                      "لم أجد إجابة لهذا السؤال في البيانات الإحصائية المفتوحة لدولة قطر."),
        "ungrounded": ("I could not verify every number in the answer against its sources, so here are the most relevant sources instead.",
                       "لم أتمكن من التحقق من جميع الأرقام في الإجابة من مصادرها، لذا إليك أقرب المصادر."),
        "unverified": ("I found a figure, but could not confirm that its row matches every part of the question, so here are the most relevant sources instead.",
                       "وجدت رقماً، لكن لم أتمكن من التأكد من أن صفّه يطابق كل شروط السؤال، لذا إليك أقرب المصادر."),
    }
    return messages[kind][1 if arabic else 0]


def ask(question: str, index: Index, llm: LLM, k: int = 6, reranker: Reranker | None = None, focus: int = 0,
        expand: Callable[[str], list[str]] | None = None, verify: bool = False) -> Answer:
    """`focus` > 0 shows the model only that many best-matching rows of each table chunk (0 = whole chunks).
    `expand` (a QueryTranslator) adds the question in the other language to the search.
    `verify` asks the model, in a second short call, whether the row behind the answer matches every condition of
    the question; if it does not, the answer is withheld ("unverified") and the sources are shown instead."""
    start = time.perf_counter()
    extra = expand(question) if expand else []
    hits = retrieve(index, question, k, reranker, extra=extra)
    arabic = is_arabic(question)
    if not hits:
        return Answer(question, _message("not_found", arabic), "not_found", hits, seconds=time.perf_counter() - start,
                      extra_queries=extra)

    context = build_context(match_text(question, extra), hits, focus)
    raw = llm.complete(SYSTEM, build_prompt(question, context)).strip()
    if declined(raw, question):
        return Answer(question, _message("not_found", arabic), "not_found", hits, raw=raw,
                      seconds=time.perf_counter() - start, context=context, extra_queries=extra)

    cited, missing = check_grounding(raw, question, context)
    status, text = ("ungrounded", _message("ungrounded", arabic)) if missing else ("answered", raw)
    if status == "answered" and verify:
        rows = answer_rows(raw, question, context, cited)[:2]
        if rows and not any(row_matches(llm, question, header, row) for header, row in rows):
            status, text = "unverified", _message("unverified", arabic)
    return Answer(question, text, status, hits, cited, missing, raw, time.perf_counter() - start, context, extra)
