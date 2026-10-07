"""A tiny fixture corpus in the portal's format. The values are made up for testing; they are not real statistics."""

from __future__ import annotations

import pytest

from arag.index import Index

PORTAL = "https://www.data.gov.qa"


def dataset(ds_id, title_en, title_ar, desc_en, desc_ar, theme, fields, records):
    return {
        "meta": {
            "dataset_id": ds_id,
            "fields": fields,
            "metas": {"default": {
                "title_en": title_en, "title_ar": title_ar,
                "description_en": f"<p>{desc_en}</p>", "description_ar": f"<p>{desc_ar}</p>",
                "theme_en": [theme[0]], "theme_ar": [theme[1]],
                "keyword_en": [], "keyword_ar": [],
                "publisher": "National Planning Council", "records_count": len(records),
            }},
        },
        "records": records,
    }


YEAR = {"name": "year", "label_en": "Year", "label_ar": "السنة", "type": "date"}

GYMS = dataset(
    "gyms-by-municipality", "Gyms by Entity and Municipality", "الصالات الرياضية حسب الجهة والبلدية",
    "Number of gyms by entity and municipality.", "عدد الصالات الرياضية حسب الجهة والبلدية.",
    ("Culture, Sports and Tourism", "الثقافة والرياضة والسياحة"),
    [YEAR,
     {"name": "municipality", "label_en": "Municipality", "label_ar": "البلدية", "type": "text"},
     {"name": "lbldy", "label_en": "البلدية", "label_ar": "البلدية", "type": "text"},
     {"name": "number", "label_en": "Number", "label_ar": "العدد", "type": "int"}],
    [{"year": "2022", "municipality": "Doha", "lbldy": "الدوحة", "number": 91},
     {"year": "2022", "municipality": "Al Rayyan", "lbldy": "الريان", "number": 37},
     {"year": "2023", "municipality": "Doha", "lbldy": "الدوحة", "number": 104},
     {"year": "2023", "municipality": "Al Rayyan", "lbldy": "الريان", "number": 41}],
)

POPULATION = dataset(
    "population-by-nationality", "Population by Nationality and Sex", "السكان حسب الجنسية والجنس",
    "Mid-year population estimates by nationality and sex.", "تقديرات السكان في منتصف العام حسب الجنسية والجنس.",
    ("Population and Demography", "السكان والديموغرافيا"),
    [YEAR,
     # The Arabic label is spelled differently from the English column's (إ vs ا): pairing must still work.
     {"name": "nationality", "label_en": "Nationality", "label_ar": "الجنسية", "type": "text"},
     {"name": "ljnsy", "label_en": "الجنسيه", "label_ar": "الجنسيه", "type": "text"},
     {"name": "sex", "label_en": "Sex", "label_ar": "الجنس", "type": "text"},
     {"name": "ljns", "label_en": "الجنس", "label_ar": "الجنس", "type": "text"},
     {"name": "population", "label_en": "Population", "label_ar": "السكان", "type": "int"}],
    [{"year": "2023", "nationality": "Qatari", "ljnsy": "قطري", "sex": "Male", "ljns": "ذكور", "population": 165432},
     {"year": "2023", "nationality": "Qatari", "ljnsy": "قطري", "sex": "Female", "ljns": "إناث", "population": 163210},
     {"year": "2023", "nationality": "Non-Qatari", "ljnsy": "غير قطري", "sex": "Male", "ljns": "ذكور", "population": 2050000}],
)

ELECTRICITY = dataset(
    "electricity-consumption", "Electricity Consumption by Sector", "استهلاك الكهرباء حسب القطاع",
    "Electricity consumption in gigawatt hours by sector.", "استهلاك الكهرباء بالجيجاواط ساعة حسب القطاع.",
    ("Energy and Environment", "الطاقة والبيئة"),
    [YEAR,
     {"name": "sector", "label_en": "Sector", "label_ar": "القطاع", "type": "text"},
     {"name": "lqt", "label_en": "القطاع", "label_ar": "القطاع", "type": "text"},
     {"name": "gwh", "label_en": "Consumption (GWh)", "label_ar": "الاستهلاك (جيجاواط ساعة)", "type": "double"}],
    [{"year": "2023", "sector": "Residential", "lqt": "السكني", "gwh": 18250.5},
     {"year": "2023", "sector": "Industrial", "lqt": "الصناعي", "gwh": 12011.25}],
)

DATASETS = [GYMS, POPULATION, ELECTRICITY]


@pytest.fixture(scope="session")
def index() -> Index:
    return Index.build(DATASETS, PORTAL, rows_per_chunk=2)


class FakeLLM:
    """Returns a fixed reply and records the prompt it was given."""

    name = "fake"

    def __init__(self, reply: str):
        self.reply = reply
        self.prompts: list[str] = []

    def complete(self, system: str, user: str) -> str:
        self.prompts.append(user)
        return self.reply
