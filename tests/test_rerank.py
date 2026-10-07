from arag.chunking import Chunk
from arag.evaluation import evidence_rank
from arag.index import Hit
from arag.rerank import FOCUS_NOTE, LLMReranker, RowReranker, best_row, focus_rows, make_reranker, retrieve

from conftest import FakeLLM

HEADER = "Gyms by Entity and Municipality | الصالات الرياضية حسب الجهة والبلدية"


def chunk(cid, rows, kind="rows"):
    return Chunk(cid, "gyms", kind, "Gyms", "الصالات", HEADER + "\n" + "\n".join(rows), "u")


# Every word of the question appears in A, but spread over different rows; B holds the one row that matches.
SPREAD = chunk("A", ["Year: 2023 · Type: Private Gyms · Municipality: Al Rayyan · Number: 52",
                     "Year: 2022 · Type: Hotel Gyms · Municipality: Doha · Number: 89"])
EXACT = chunk("B", ["Year: 2023 · Type: Hotel Gyms · Municipality: Doha · Number: 107"])
QUESTION = "How many hotel gyms were there in Doha in 2023?"


def test_best_row_needs_all_conditions_in_one_row():
    assert best_row(QUESTION, EXACT.text, "rows").score > best_row(QUESTION, SPREAD.text, "rows").score
    assert "107" in best_row(QUESTION, EXACT.text, "rows").row


def test_row_reranker_promotes_the_matching_row():
    hits = [Hit(SPREAD, 9.0), Hit(EXACT, 6.0)]  # BM25 prefers the spread-out chunk
    assert [h.chunk.id for h in RowReranker().rerank(QUESTION, hits)] == ["B", "A"]


def test_llm_reranker_follows_the_models_pick_and_keeps_the_rest():
    hits = [Hit(chunk(c, [f"Year: 2020 · Number: {i}"]), 1.0) for i, c in enumerate("PQRS")]
    llm = FakeLLM("The answer is in [3].")
    out = LLMReranker(llm, prefilter=RowReranker(w_row=0)).rerank("question", hits)
    assert [h.chunk.id for h in out] == ["R", "P", "Q", "S"]
    assert "Question: question" in llm.prompts[0]


def test_llm_reranker_survives_a_useless_reply():
    hits = [Hit(chunk(c, ["Year: 2020"]), 1.0) for c in "PQ"]
    out = LLMReranker(FakeLLM("I am not sure."), prefilter=RowReranker(w_row=0)).rerank("q", hits)
    assert [h.chunk.id for h in out] == ["P", "Q"]


def test_retrieve_with_reranker_on_the_fixture(index):
    hits = retrieve(index, "How many gyms were there in Doha in 2023?", 3, make_reranker("rows"))
    assert "Number: 104" in hits[0].chunk.text


def test_evidence_rank():
    item = {"dataset_ids": ["gyms"], "answer": ["107"]}
    assert evidence_rank([Hit(SPREAD, 1), Hit(EXACT, 1)], item) == 2
    assert evidence_rank([Hit(SPREAD, 1)], item) is None


MANY = chunk("M", [f"Year: 2023 · Type: Hotel Gyms · Municipality: Town{i} · Number: {i}" for i in range(10)]
                  + ["Year: 2023 · Type: Hotel Gyms · Municipality: Doha · Number: 107"])


def test_focus_keeps_only_the_best_rows_in_order():
    text = focus_rows(QUESTION, MANY.text, "rows", 2)
    lines = text.split("\n")
    assert lines[0] == HEADER and lines[1] == FOCUS_NOTE
    assert len(lines) == 4 and "Number: 107" in text and "Town9" not in text


def test_focus_leaves_cards_and_short_chunks_alone():
    assert focus_rows(QUESTION, EXACT.text, "rows", 3) == EXACT.text
    assert focus_rows(QUESTION, MANY.text, "card", 1) == MANY.text
    assert focus_rows(QUESTION, MANY.text, "rows", 0) == MANY.text
