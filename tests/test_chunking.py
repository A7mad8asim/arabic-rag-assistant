from arag.chunking import columns, dataset_chunks

from conftest import GYMS, POPULATION, PORTAL


def test_arabic_twin_columns_are_paired():
    cols = {c.name: c for c in columns(GYMS["meta"]["fields"])}
    assert "lbldy" not in cols
    assert cols["municipality"].arabic_twin == "lbldy"


def test_pairing_survives_spelling_variants():
    cols = {c.name: c for c in columns(POPULATION["meta"]["fields"])}
    assert cols["nationality"].arabic_twin == "ljnsy"
    assert cols["sex"].arabic_twin == "ljns"


def test_card_and_rows():
    chunks = dataset_chunks(GYMS, PORTAL, rows_per_chunk=2)
    card, rows = chunks[0], chunks[1:]
    assert card.kind == "card" and "Gyms by Entity" in card.text and "الصالات الرياضية" in card.text
    assert "Years: 2022–2023" in card.text
    assert card.url == f"{PORTAL}/explore/dataset/gyms-by-municipality/"
    # one chunk per year here (2 rows each), every row bilingual
    assert [c.years for c in rows] == [["2022"], ["2023"]]
    assert "Municipality: Doha / الدوحة · Number: 104" in rows[1].text


def test_large_groups_are_split():
    chunks = dataset_chunks(POPULATION, PORTAL, rows_per_chunk=2)
    assert [c.kind for c in chunks] == ["card", "rows", "rows"]
