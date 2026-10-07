from arag.llm import LLMError
from arag.query import QueryTranslator
from arag.rerank import retrieve

from conftest import FakeLLM


class SystemSpy(FakeLLM):
    def __init__(self, reply):
        super().__init__(reply)
        self.systems = []

    def complete(self, system, user):
        self.systems.append(system)
        return super().complete(system, user)


def test_translator_targets_the_other_language():
    llm = SystemSpy("How many gyms were in Doha in 2023?\n")
    assert QueryTranslator(llm)("كم عدد الصالات في الدوحة 2023؟") == ["How many gyms were in Doha in 2023?"]
    QueryTranslator(llm)("How many gyms were in Doha in 2023?")
    assert "into English" in llm.systems[0] and "into Modern Standard Arabic" in llm.systems[1]


def test_translator_failure_falls_back_to_the_original():
    class Broken:
        name = "broken"

        def complete(self, system, user):
            raise LLMError("down")

    assert QueryTranslator(Broken())("any question") == []
    assert QueryTranslator(FakeLLM(""))("any question") == []
    assert QueryTranslator(FakeLLM("any question"))("any question") == []  # an echo adds nothing


def test_expansion_fuses_both_searches(index):
    # The extra Arabic query pulls in the electricity rows; fused scores stay usable by the row reranker.
    expanded = retrieve(index, "residential sector 2023", 3, expand=lambda q: ["استهلاك الكهرباء القطاع السكني"])
    assert expanded[0].chunk.dataset_id == "electricity-consumption"
    assert all(h.score > 0 for h in expanded)
    assert len(retrieve(index, "residential sector 2023", 3, expand=lambda q: [])) == 3  # no extra query: plain search
