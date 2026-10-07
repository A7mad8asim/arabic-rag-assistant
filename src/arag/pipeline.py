"""Question → retrieved sources → cited answer → grounding check.

The rule, as in Ask-the-Data: the model may only repeat numbers that appear in the sources it
cites. An answer with a number that is not in its cited sources is not shown; the reader gets
the sources instead.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .index import Hit, Index
from .llm import LLM
from .text import is_arabic, numbers_in

NOT_FOUND = "NOT_FOUND"

SYSTEM = f"""You answer questions about official statistics of the State of Qatar, using ONLY the numbered sources given.

Rules:
- Answer in the language of the question (Arabic or English), in 1 to 3 sentences.
- Copy every number exactly as it appears in a source. Never calculate, round, estimate or convert numbers.
- After each fact, cite the source number in square brackets, for example [2]. Every sentence with a number needs a citation.
- Mention the year (and the unit, if the source gives one) of each figure.
- If the sources do not contain the answer, reply with exactly: {NOT_FOUND}
"""

# Citations as the model writes them: [2], [n2], [1, 2], [١] ...
_CITE = re.compile(r"\[\s*(?:n|source\s*|مصدر\s*)?([0-9٠-٩]+(?:\s*[,،]\s*[0-9٠-٩]+)*)\s*\]", re.IGNORECASE)
_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


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

    @property
    def sources(self) -> list[Hit]:
        return [self.hits[i - 1] for i in self.cited if 0 < i <= len(self.hits)]


def build_prompt(question: str, hits: list[Hit]) -> str:
    blocks = [f"[{i}] {h.chunk.text}" for i, h in enumerate(hits, 1)]
    return "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"


def check_grounding(answer: str, question: str, hits: list[Hit]) -> tuple[list[int], list[str]]:
    """The cited source numbers, and any number in the answer found neither in those sources nor in the question."""
    cited = sorted({n for n in citations(answer) if 0 < n <= len(hits)})
    body = _CITE.sub(" ", answer)
    allowed = numbers_in(question)
    for i in cited:
        allowed |= numbers_in(hits[i - 1].chunk.text)
    missing = sorted(n for n in numbers_in(body) if n not in allowed)
    if numbers_in(body) and not cited:
        missing = sorted(numbers_in(body) - numbers_in(question))
    return cited, missing


def _message(kind: str, arabic: bool) -> str:
    messages = {
        "not_found": ("I could not find this in Qatar's open statistics.",
                      "لم أجد إجابة لهذا السؤال في البيانات الإحصائية المفتوحة لدولة قطر."),
        "ungrounded": ("I could not verify every number in the answer against its sources, so here are the most relevant sources instead.",
                       "لم أتمكن من التحقق من جميع الأرقام في الإجابة من مصادرها، لذا إليك أقرب المصادر."),
    }
    return messages[kind][1 if arabic else 0]


def ask(question: str, index: Index, llm: LLM, k: int = 6) -> Answer:
    start = time.perf_counter()
    hits = index.search(question, k)
    arabic = is_arabic(question)
    if not hits:
        return Answer(question, _message("not_found", arabic), "not_found", hits, seconds=time.perf_counter() - start)

    raw = llm.complete(SYSTEM, build_prompt(question, hits)).strip()
    if not raw or NOT_FOUND in raw:
        return Answer(question, _message("not_found", arabic), "not_found", hits, raw=raw, seconds=time.perf_counter() - start)

    cited, missing = check_grounding(raw, question, hits)
    if missing:
        return Answer(question, _message("ungrounded", arabic), "ungrounded", hits, cited, missing, raw,
                      time.perf_counter() - start)
    return Answer(question, raw, "answered", hits, cited, [], raw, time.perf_counter() - start)
