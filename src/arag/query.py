"""Query translation: search with the question in both languages.

Every chunk is bilingual, but not evenly: titles and descriptions are in both languages while
some column labels and values exist only in English (or only in Arabic). A question can therefore
miss a chunk whose matching words are in the other language. The translator asks the model for
the question in the other language (Arabic -> English, English -> Arabic); retrieval searches with
both and fuses the two rankings by reciprocal rank fusion.
"""

from __future__ import annotations

import logging

from .llm import LLM, LLMError
from .text import is_arabic

log = logging.getLogger(__name__)

SYSTEM = """Translate the user's question about Qatar's official statistics into {target}.
Keep every number, year, name and place. If the question is in a Gulf dialect, translate its meaning.
Reply with the translated question only, on one line."""


class QueryTranslator:
    name = "translate"

    def __init__(self, llm: LLM):
        self.llm = llm

    def __call__(self, question: str) -> list[str]:
        target = "English" if is_arabic(question) else "Modern Standard Arabic"
        try:
            reply = self.llm.complete(SYSTEM.format(target=target), question)
        except LLMError as e:  # translation is an extra; never let it stop the answer
            log.warning("query translation failed, searching with the original only: %s", e)
            return []
        lines = [line.strip() for line in (reply or "").splitlines() if line.strip()]
        return [lines[0]] if lines and lines[0] != question else []
