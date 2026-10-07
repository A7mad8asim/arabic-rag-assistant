from arag.evaluation import evaluate
from arag.pipeline import NOT_FOUND, ask, citations

from conftest import FakeLLM


def test_grounded_answer_is_shown(index):
    q = "How many gyms were there in Doha in 2023?"
    src = next(i for i, h in enumerate(index.search(q, 6), 1) if "Number: 104" in h.chunk.text)
    llm = FakeLLM(f"Doha had 104 gyms in 2023 [{src}].")
    ans = ask(q, index, llm)
    assert ans.status == "answered" and "104" in ans.text
    assert ans.sources[0].chunk.dataset_id == "gyms-by-municipality"
    assert "Question: How many gyms" in llm.prompts[0]


def test_citation_variants():
    assert citations("a [2] b [n3] c [1, 4] d [١] e [مصدر 5]") == [2, 3, 1, 4, 1, 5]
    assert citations("no citation, 2023 [sic]") == []


def test_invented_number_is_blocked(index):
    ans = ask("How many gyms were there in Doha in 2023?", index, FakeLLM("Doha had 120 gyms in 2023 [1]."))
    assert ans.status == "ungrounded" and ans.ungrounded_numbers == ["120"]
    assert "120" not in ans.text


def test_number_without_citation_is_blocked(index):
    ans = ask("How many gyms were there in Doha in 2023?", index, FakeLLM("Doha had 104 gyms in 2023."))
    assert ans.status == "ungrounded"


def test_not_found_in_the_question_language(index):
    ans = ask("كم عدد المستشفيات في الخور؟", index, FakeLLM(NOT_FOUND))
    assert ans.status == "not_found" and "لم أجد" in ans.text


def test_evaluation_scores_retrieval_and_answers(index):
    gold = [
        {"id": "G1", "lang": "en", "kind": "lookup", "question": "How many gyms were in Doha in 2023?",
         "dataset_ids": ["gyms-by-municipality"], "answer": ["104"]},
        {"id": "G1", "lang": "ar", "kind": "lookup", "question": "كم عدد الصالات الرياضية في الدوحة عام 2023؟",
         "dataset_ids": ["gyms-by-municipality"], "answer": ["104"]},
        {"id": "U1", "lang": "en", "kind": "unanswerable", "question": "How many hospitals are in Al Khor?",
         "dataset_ids": [], "answer": []},
    ]
    retrieval = evaluate(gold, index)["summary"]
    assert retrieval["all"]["hit@1"] == 100.0 and retrieval["ar"]["mrr"] == 1.0

    result = evaluate(gold, index, FakeLLM(NOT_FOUND))["summary"]
    assert result["all"]["unanswerable_refused"] == 100.0
    assert result["en"]["answer_accuracy"] == 50.0  # the lookup is wrong, the refusal is right
