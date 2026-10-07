from arag.chunking import dataset_chunks
from arag.large import clean_view_rows, plan_views, view_fields
from arag.portal import download, fetch_mode

FIELDS = [
    {"name": "lsn_year", "label_en": "Year", "label_ar": "السنة", "type": "date"},
    {"name": "lshhr_month", "label_en": "Month", "label_ar": "الشهر", "type": "double"},
    {"name": "hs8", "label_en": "HS8", "label_ar": "", "type": "text"},
    {"name": "dwl_lmnsh", "label_en": "دولة المنشأ", "label_ar": "دولة المنشأ", "type": "text"},
    {"name": "country_of_origin", "label_en": "Country of Origin", "label_ar": "دولة المنشأ", "type": "text"},
    {"name": "quantity", "label_en": "Quantity", "label_ar": "الكمية", "type": "double"},
    {"name": "weight_kg", "label_en": "Weight (KG)", "label_ar": "الوزن", "type": "double"},
    {"name": "value_qr", "label_en": "Value (QR)", "label_ar": "القيمة", "type": "double"},
]
META = {"dataset_id": "imports", "fields": FIELDS, "metas": {"default": {
    "title_en": "Qatar Imports", "title_ar": "واردات قطر", "records_count": 900000}}}


def test_trade_tables_get_country_and_month_totals_of_additive_measures_only():
    views = {v.key: v for v in plan_views(FIELDS)}
    assert views["by-country"].group_by == ["lsn_year", "country_of_origin", "dwl_lmnsh"]
    assert views["by-month"].group_by == ["lsn_year", "lshhr_month"]
    assert views["by-country"].measures == ["weight_kg", "value_qr"]  # never quantity: its units differ by product


def test_tables_without_additive_measures_stay_card_only():
    fields = [f for f in FIELDS if f["name"] not in ("weight_kg", "value_qr")]
    assert plan_views(fields) == []
    assert fetch_mode({"fields": fields, "metas": {"default": {"records_count": 10**6}}}, 30000) == "card"
    assert fetch_mode(META, 30000) == "views" and fetch_mode(META, 10**7) == "rows"


def test_clean_rows_orders_months_and_names_them_in_both_languages():
    month = plan_views(FIELDS)[1]
    rows = [{"lsn_year": "2024-01-01T00:00:00+00:00", "lshhr_month": m, "weight_kg": 1.6, "value_qr": 10.4} for m in (10.0, 3.0)]
    out = clean_view_rows(rows, month, FIELDS)
    assert [r["lshhr_month"] for r in out] == ["March / مارس", "October / أكتوبر"]
    assert out[0]["lsn_year"] == "2024" and out[0]["value_qr"] == 10 and out[0]["weight_kg"] == 2


class FakePortal:
    def totals(self, ds, group_by, measures):
        if "country_of_origin" in group_by:
            return [{"lsn_year": "2023-01-01T00:00:00+00:00", "country_of_origin": "Japan", "dwl_lmnsh": "اليابان",
                     "weight_kg": 126029468.88, "value_qr": 3601310831.52}]
        return [{"lsn_year": "2023-01-01T00:00:00+00:00", "lshhr_month": 3.0, "weight_kg": 5.0, "value_qr": 7.0}]


def test_totals_become_bilingual_chunks_marked_as_computed():
    chunks = dataset_chunks(download(FakePortal(), META, "views"), "https://x")
    by_country = next(c for c in chunks if c.id.startswith("imports#by-country"))
    assert "computed from the dataset's rows" in by_country.text
    assert "Country of Origin: Japan / اليابان · Total Weight (KG): 126,029,469 · Total Value (QR): 3,601,310,832" in by_country.text
    assert by_country.kind == "rows" and by_country.years == ["2023"]
    assert "Indexed as totals" in chunks[0].text
    assert [f["label_en"] for f in view_fields(plan_views(FIELDS)[0], FIELDS)][-1] == "Total Value (QR)"
