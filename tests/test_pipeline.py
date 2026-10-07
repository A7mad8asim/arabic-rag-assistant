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


def test_focus_shows_fewer_rows_and_grounds_on_what_was_shown(index):
    q = "How many hotel gyms were there in Doha in 2023?"
    llm = FakeLLM("Doha had 104 gyms in 2023 [1].")
    ans = ask(q, index, llm, focus=1)
    rows_shown = [t for t, h in zip(ans.context, ans.hits) if h.chunk.kind == "rows"]
    assert rows_shown and all(t.count("\n") == 2 for t in rows_shown)  # header, note, one row (fixture chunks have 2)
    assert all(t in llm.prompts[0] for t in ans.context)  # the prompt holds exactly the focused text


def test_grounding_uses_the_focused_text():
    from arag.pipeline import check_grounding

    shown = ["Gyms\nYear: 2023 · Municipality: Doha · Number: 104"]
    assert check_grounding("104 gyms [1].", "Doha 2023?", shown) == ([1], [])
    assert check_grounding("41 gyms [1].", "Doha 2023?", shown) == ([1], ["41"])  # 41 was cut from the context


def test_declines_in_prose_count_as_not_found():
    from arag.pipeline import declined

    for reply in ("NOT_FOUND", "The provided data does not include the price of karak tea.",
                  "The sources do not mention 2040.", "لم يتم ذكر قيمة تعويضات التأمين في البيانات المقدمة.",
                  "لا يوجد أي بيانات تشير إلى عدد الطلاب في 2021/2022.", "I am not sure [1].", "", "NOT_FOUND [2]"):
        assert declined(reply), reply
    for reply in ("In 2023, there were 107 hotel gyms in Doha [1].", "في سنة 2023، كانت هناك 107 صالة رياضية في فنادق الدوحة [2].",
                  "Qatari guests spent 868,940 hotel nights in 2019 [3]."):
        assert not declined(reply), reply


def test_prose_refusal_is_reported_as_not_found(index):
    ans = ask("How many hospitals are in Al Khor?", index, FakeLLM("The sources do not include hospital counts for Al Khor."))
    assert ans.status == "not_found" and "could not find" in ans.text


def test_reply_that_only_repeats_the_questions_numbers_is_not_an_answer(index):
    from arag.pipeline import declined

    assert declined("المنتخب القطري فاز بنهائي كأس العالم 2022.", "من فاز بنهائي كأس العالم 2022؟")
    ans = ask("من فاز بنهائي كأس العالم 2022؟", index, FakeLLM("المنتخب القطري فاز بنهائي كأس العالم 2022."))
    assert ans.status == "not_found"
