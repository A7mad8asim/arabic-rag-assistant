import numpy as np

from arag.index import Index, rrf

from conftest import DATASETS, PORTAL


def top_dataset(index, q):
    return index.search(q, 3)[0].chunk.dataset_id


def test_english_and_arabic_find_the_same_dataset(index):
    assert top_dataset(index, "How many gyms were there in Doha in 2023?") == "gyms-by-municipality"
    assert top_dataset(index, "كم عدد الصالات الرياضية في الدوحة عام 2023؟") == "gyms-by-municipality"
    assert top_dataset(index, "Qatari female population") == "population-by-nationality"
    assert top_dataset(index, "كم عدد السكان القطريين الإناث؟") == "population-by-nationality"
    assert top_dataset(index, "استهلاك الكهرباء في القطاع السكني") == "electricity-consumption"


def test_year_picks_the_right_rows(index):
    hit = index.search("gyms Doha 2023", 1)[0]
    assert hit.chunk.kind == "rows" and hit.chunk.years == ["2023"]


def test_rrf_rewards_agreement():
    assert rrf([[1, 2, 3], [3, 1, 2]])[0] == 1


def test_dense_failure_falls_back_to_bm25():
    def broken(texts):
        raise ConnectionError("no ollama")

    index = Index.build(DATASETS, PORTAL)
    index.vectors, index.embed = np.zeros((len(index.chunks), 4), dtype=np.float32), broken
    assert index.search("gyms in Doha", 2)[0].chunk.dataset_id == "gyms-by-municipality"


def test_save_and_load(index, tmp_path):
    index.save(tmp_path)
    loaded = Index.load(tmp_path)
    assert [c.id for c in loaded.chunks] == [c.id for c in index.chunks]
    assert top_dataset(loaded, "electricity residential") == "electricity-consumption"
