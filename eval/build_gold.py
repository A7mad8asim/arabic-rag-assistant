"""Add the held-out test questions to eval/gold.jsonl, verifying every answer against the downloaded data.

Each lookup names its dataset, the text that identifies exactly one row (`where`) and the expected value.
The script refuses to write a question whose row is missing, ambiguous, or does not hold the value.
Exact duplicate datasets (same English title) are accepted as alternatives automatically.

    python eval/build_gold.py            # needs data/raw from `arag fetch`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from arag.chunking import columns, render_row
from arag.config import EVAL_DIR, Settings
from arag.portal import dataset_path
from arag.text import numbers_in

# id, dataset, where (all must appear in the row), answer, English, Arabic, Gulf dialect
LOOKUPS = [
    ("T01", "number-of-workers-and-farms-in-active-farms-by-municipality-nationality-and-occupational-category",
     ["Year: 2025", "Al Shahaniya", "Category: Workers"], "4,918",
     "How many workers were there in active farms in Al Shahaniya in 2025?",
     "كم عدد العمال في المزارع العاملة في الشيحانية سنة 2025؟", False),
    ("T02", "value-of-agricultural-and-fish-production",
     ["Year: 2023", "Product Group: Vegetables", "Product: Vegetables"], "302,392",
     "What was the value of vegetable production in 2023?",
     "كم قيمة إنتاج الخضار في 2023؟", True),
    ("T03", "quantities-of-plant-production",
     ["Year: 2018", "Crop: Green Pepper"], "1,796",
     "How much green pepper was produced in 2018?",
     "كم كان إنتاج الفلفل في 2018؟", False),
    ("T04", "hotel-gulf-guests-by-nationality-and-hotel-stays",
     ["Year: 2019", "Nationality: Qatar /", "Metric: Nights"], "868,940",
     "How many hotel nights did Qatari guests spend in 2019?",
     "كم عدد ليالي الإقامة للنزلاء القطريين في الفنادق عام 2019؟", False),
    ("T05", "cultural-events-at-the-cultural-village-foundation-katara-by-type-of-event",
     ["Seminars, Lectures", "2023: 76"], "76",
     "How many seminars, lectures and conferences did Katara hold in 2023?",
     "كم عدد الندوات والمحاضرات والمؤتمرات في كتارا عام 2023؟", False),
    ("T06", "sports-workers-in-hotel-gyms-and-private-gyms-by-occupation-and-gender",
     ["Year: 2021", "Occupation: Administrators", "Place of Work: Private Gyms", "Gender: Females"], "155",
     "How many female administrators worked in private gyms in 2021?",
     "كم عدد الإداريين الإناث في الصالات الرياضية الخاصة عام 2021؟", False),
    ("T07", "public-schools-teaching-and-administrative-staff-by-nationality-gender-and-level-of-education",
     ["Academic Year: 2018/2019", "Category: Pre-primary", "Nationality: Non-Qatari", "Gender: Females"], "408",
     "How many non-Qatari women were on the teaching and administrative staff of public pre-primary schools in 2018/2019?",
     "كم عدد الإناث غير القطريات في هيئة التدريس والإدارة في رياض الأطفال الحكومية في العام الدراسي 2018/2019؟", False),
    ("T08", "trainees-at-the-private-training-centers-by-field-of-training-gender-nationality-and-age-groups",
     ["Year: 2022", "Age Groups: 25-39", "Nationality: Qataris", "Field of Training: Administration", "Gender: Females"], "866",
     "How many Qatari women aged 25-39 trained in administration at private training centres in 2022?",
     "كم عدد القطريات من الفئة العمرية 25-39 اللاتي تدربن في مجال الإدارة في المراكز التدريبية الخاصة عام 2022؟", False),
    ("T09", "illiterate-persons-and-total-population-15-years-and-above-by-gender",
     ["Year: 2024", "Gender: Females", "Indicator: Illiterate Persons"], "2,564",
     "How many illiterate women aged 15 and above were there in 2024?",
     "كم عدد الأميات من الإناث (15 سنة فأكثر) في 2024؟", False),
    ("T10", "persons-attending-night-schools-and-illiteracy-eradication-centers-by-level-of-education-and-gender",
     ["Academic Year: 2012/2013", "Level of Education: Secondary", "Gender: Males"], "1,867",
     "How many men attended secondary level at night schools in 2012/2013?",
     "كم عدد الذكور الملتحقين بالمرحلة الثانوية في المدارس الليلية عام 2012/2013؟", False),
    ("T11", "completed-residential-buildings-by-municipality-and-their-connection-to-the-public-utilities",
     ["Census Year: 2020", "Municipality: Al Rayyan", "Service: Water", "Status: Connected"], "61,893",
     "How many completed residential buildings in Al Rayyan were connected to the water network in the 2020 census?",
     "كم عدد المباني السكنية المكتملة في الريان المتصلة بشبكة المياه في تعداد 2020؟", False),
    ("T12", "amount-of-energy-used-in-operational-district-cooling-plants-and-energy-savings-by-economic-activity",
     ["Year: 2019", "Economic Activity: Education", "Electricity Consumption for Cooling"], "191,908",
     "How much electricity did district cooling plants use for cooling in the education sector in 2019?",
     "كم كمية الطاقة الكهربائية المستخدمة في محطات تبريد المناطق لقطاع التعليم في 2019؟", False),
    ("T13", "type-of-animals-and-classification-at-the-zoo-in-al-khor-park",
     ["Item: Mammals", "2021 -  Species"], "27",
     "How many mammal species were at the Al Khor Park zoo in 2021?",
     "كم نوع ثدييات كان في حديقة حيوانات الخور سنة 2021؟", True),
    ("T14", "waste-and-scrap-exports-by-type-and-year",
     ["Year: 2020", "Groups of Goods: Rubber waste"], "75,812,244",
     "What weight of rubber waste did Qatar export in 2020, in kilograms?",
     "كم كان وزن صادرات نفايات المطاط في 2020 بالكيلوغرام؟", False),
    ("T15", "fog-dust-storm-and-haze-monthly-occurrence",
     ["Year: 2013", "Month: June", "Haze"], "17",
     "How many days of haze were there in June 2013?",
     "كم يوم كان فيه عجاج في يونيو 2013؟", True),
    ("T17", "percentage-distribution-of-gross-domestic-product-by-economic-activities-at-current-prices-quarterly",
     ["Year: 2022", "Q2: 10.82", "Manufacturing"], "10.82",
     "What share of GDP at current prices did manufacturing have in Q2 2022?",
     "ما نسبة مساهمة الصناعة التحويلية في الناتج المحلي الإجمالي بالأسعار الجارية في الربع الثاني 2022؟", False),
    ("T18", "value-of-paid-claims-by-type",
     ["Year: 2013", "Type: Cargo"], "568,265",
     "What was the value of paid cargo insurance claims in 2013?",
     "كم بلغت قيمة تعويضات تأمين النقل المدفوعة في 2013؟", False),
    ("T19", "health-statistics-number-and-rate-for-health-indicators-in-private-sector",
     ["Year: 2016", "Indicator: Dentists"], "1,653",
     "How many dentists worked in the private sector in 2016?",
     "كم دكتور أسنان في القطاع الخاص سنة 2016؟", True),
    ("T20", "number-of-reported-infectious-disease-cases-by-disease-and-gender",
     ["Year: 2014", "Disease: Influenza", "Gender: Male"], "3,680",
     "How many influenza cases were reported among males in 2014?",
     "كم عدد حالات الإنفلونزا المسجلة بين الذكور في 2014؟", False),
    ("T21", "people-with-disabilities-less-than-15-years-who-received-services-at-rumeilah-hospital-and-qatar",
     ["Year: 2020", "Type of Disability: Intellectual", "Gender: Female"], "160",
     "How many girls under 15 with intellectual disabilities received services at Rumeilah Hospital and Qatar Rehabilitation Institute in 2020?",
     "كم عدد الإناث أقل من 15 سنة ذوات الإعاقة الذهنية اللاتي تلقين خدمات في مستشفى الرميلة ومركز قطر لإعادة التأهيل عام 2020؟", False),
    ("T22", "special-needs-statistics-number-of-students-with-disabilities-integrated-into-public-schools-by",
     ["Year: 2021/2022", "Level of Education: Secondery", "Nationality: Non-Qatari", "Gender: Male"], "200",
     "How many non-Qatari male students with disabilities were integrated into public secondary schools in 2021/2022?",
     "كم عدد الطلاب الذكور غير القطريين من ذوي الإعاقة المدمجين في المدارس الثانوية الحكومية في 2021/2022؟", False),
    ("T23", "building-permits-issued-by-type-of-permit-and-month-al-wakrah-municipality",
     ["Year: 2022", "Month: OCTOBER", "Type of Permit: Additions"], "78",
     "How many building permits for additions were issued in Al Wakrah in October 2022?",
     "كم رخصة إضافات طلعت في الوكرة في أكتوبر 2022؟", True),
    ("T24", "building-completion-certificates-issued-by-type-of-certificate-and-month-al-daayen-municipality",
     ["Year: 2019", "Month: APRIL", "Type of Certificate: New Building"], "44",
     "How many completion certificates for new buildings were issued in Al Daayen in April 2019?",
     "كم شهادة إتمام لمباني جديدة صدرت في بلدية الظعاين في أبريل 2019؟", False),
    ("T25", "population-and-labour-force-by-municipality",
     ["Year: 2024", "Municipality: Al Wakra", "Indicator: Economically Active"], "195,414",
     "How many economically active people were there in Al Wakra in 2024?",
     "كم عدد النشيطين اقتصادياً في الوكرة عام 2024؟", False),
    ("T26", "population-15-by-relation-to-labour-force-and-age-groups-quarterly",
     ["Year: 2023", "Quarter: Q2", "Age Group: 45 - 54", "Activity status: Economically Active"], "301,184",
     "How many people aged 45-54 were economically active in Q2 2023?",
     "كم عدد النشيطين اقتصادياً من الفئة العمرية 45-54 في الربع الثاني من 2023؟", False),
    ("T27", "number-of-employees-by-nationality-gender-and-main-economic-activity-wholesale-and-retail-trade",
     ["Year: 2019", "Activity Code: 47", "Nationality: Non-Qatari", "Gender: Females"], "18,388",
     "How many non-Qatari women worked in retail trade in 2019?",
     "كم عدد العاملات غير القطريات في تجارة التجزئة عام 2019؟", False),
    ("T28", "non-registered-live-births-by-nationality-and-gender",
     ["Year: 2014", "Nationality: Qataris", "Gender: Males"], "40",
     "How many non-registered live births of Qatari boys were there in 2014?",
     "كم عدد المواليد الأحياء الذكور القطريين فاقدي القيد في 2014؟", False),
    ("T29", "registered-deaths-by-nationality-gender-and-municipality",
     ["Year: 2020", "Municipality: Al Wakra", "Nationality: Non Qataris", "Gender: Males"], "49",
     "How many deaths of non-Qatari men were registered in Al Wakra in 2020?",
     "كم عدد الوفيات المسجلة للذكور غير القطريين في الوكرة عام 2020؟", False),
    ("T30", "number-of-deaths-and-injuries-resulting-from-fires",
     ["Year: 2015", "Result of the Fire: Slight Injury"], "69",
     "How many people were slightly injured in fires in 2015?",
     "كم شخص أصيب إصابة خفيفة في الحرائق سنة 2015؟", True),
    ("T31", "lawsuits-filed-in-the-investment-and-trade-court-by-classification",
     ["Year: 2023", "Lawsuit Classification: Finance & investment", "Status: Number of Lawsuits"], "120",
     "How many lawsuits over finance and investment company disputes were filed in the Investment and Trade Court in 2023?",
     "كم عدد دعاوى منازعات شركات التمويل والاستثمار المرفوعة في محكمة الاستثمار والتجارة عام 2023؟", False),
    ("T32", "number-of-reports-by-prosecution",
     ["Year: 2016", "Prosecution: Cheques Cases"], "20,152",
     "How many reports did the cheques cases prosecution receive in 2016?",
     "كم بلاغ وصل نيابة الشيكات في 2016؟", True),
    ("T33", "sdg-17-9-1-dollar-value-of-financial-and-technical-assistance-including-through-north-south-south",
     ["Year: 2019", "(US Dollar)"], "153,934,194",
     "What was the value of Qatar's development assistance to developing countries in US dollars in 2019?",
     "كم كانت قيمة المساعدات الإنمائية للبلدان النامية بالدولار الأمريكي في 2019؟", False),
    ("T34", "main-manufacturing-products",
     ["Year: 2017", "Product: Urea"], "5,777",
     "How much urea did Qatar produce in 2017, in thousand metric tons?",
     "كم كان إنتاج اليوريا في قطر عام 2017 بالألف طن متري؟", False),
    ("T35", "aircraft-fleet-size-of-the-national-carrier",
     ["Year: 2019", "Month: March"], "258",
     "How many aircraft did the national carrier have in March 2019?",
     "كم طائرة كانت عند الناقل الوطني في مارس 2019؟", True),
    ("T37", "registered-new-vehicles-and-motor-cycles-by-type-of-license",
     ["Year: 2017", "Type of License: Taxis"], "876",
     "How many new taxis were registered in 2017?",
     "كم تاكسي جديد تسجل في 2017؟", True),
]

# Other tables that hold the same row (found by auditing answers that had the right value but cited another
# dataset). Each must contain the answer and the question's year in one row, or the script refuses to add it.
ALSO_ACCEPT = {
    "T03": ["production-area-and-average-yield-of-crops"],
    "T04": ["media-culture-and-tourism-statistics-number-of-hotel-gulf-guests-and-nights-of-stay-by-country"],
    "T15": ["fog-duststorm-and-haze-doha-international-airport"],
    "T19": ["health-statistics-number-of-medical-staff-by-occupation-in-private-sector"],
    "T28": ["non-registered-live-births-by-nationality-and-gender0"],
}

UNANSWERABLE = [
    ("U06", "How many hotel nights did Qatari guests spend in 2040?", "كم عدد ليالي الإقامة للنزلاء القطريين في الفنادق عام 2040؟"),
    ("U07", "What is the population of Tokyo?", "كم عدد سكان طوكيو؟"),
    ("U08", "Who is the coach of Qatar's national football team?", "مين مدرب منتخب قطر لكرة القدم؟"),
    ("U09", "How many aircraft did the national carrier have in March 1985?", "كم طائرة كانت عند الناقل الوطني في مارس 1985؟"),
    ("U10", "What will Qatar's GDP growth rate be in 2035?", "كم سيكون معدل نمو الناتج المحلي الإجمالي لقطر في 2035؟"),
]


def main() -> int:
    raw = Settings.from_env().raw_dir
    titles: dict[str, list[str]] = {}
    for p in raw.glob("*.json"):
        meta = json.loads(p.read_text(encoding="utf-8"))["meta"]
        titles.setdefault(meta["metas"]["default"].get("title_en", "").strip().casefold(), []).append(meta["dataset_id"])

    out, problems = [], []
    for tid, ds, where, answer, en, ar, dialect in LOOKUPS:
        path = dataset_path(raw, ds)
        if not path.exists():
            problems.append(f"{tid}: dataset {ds} not downloaded")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        cols = columns(data["meta"]["fields"])
        rows = [render_row(r, cols) for r in data["records"]]
        hits = [r for r in rows if all(w in r for w in where)]
        if len(hits) != 1:
            problems.append(f"{tid}: {len(hits)} rows match {where}")
            continue
        if not numbers_in(answer) <= numbers_in(hits[0]):
            problems.append(f"{tid}: {answer} not in row: {hits[0][:200]}")
            continue
        title = data["meta"]["metas"]["default"].get("title_en", "").strip().casefold()
        accepted = [ds] + sorted(d for d in titles.get(title, []) if d != ds)
        need = numbers_in(answer) | {n for n in numbers_in(en) if len(n) == 4 and n[:2] in ("19", "20")}
        for alt in ALSO_ACCEPT.get(tid, []):
            alt_data = json.loads(dataset_path(raw, alt).read_text(encoding="utf-8"))
            alt_cols = columns(alt_data["meta"]["fields"])
            if any(need <= numbers_in(render_row(r, alt_cols)) for r in alt_data["records"]):
                accepted.append(alt)
            else:
                problems.append(f"{tid}: alternative {alt} has no row with {sorted(need)}")
        for lang, q in (("en", en), ("ar", ar)):
            out.append({"id": tid, "lang": lang, "kind": "lookup", "question": q, "dataset_ids": accepted,
                        "answer": [answer], "dialect": dialect and lang == "ar", "split": "test"})
    for uid, en, ar in UNANSWERABLE:
        for lang, q in (("en", en), ("ar", ar)):
            out.append({"id": uid, "lang": lang, "kind": "unanswerable", "question": q, "dataset_ids": [],
                        "answer": [], "dialect": uid == "U08" and lang == "ar", "split": "test"})

    for p in problems:
        print("SKIPPED", p, file=sys.stderr)

    gold_path = EVAL_DIR / "gold.jsonl"
    existing = [json.loads(l) for l in gold_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    kept = [dict(r, split=r.get("split", "dev")) for r in existing if r.get("split", "dev") == "dev"]
    with open(gold_path, "w", encoding="utf-8") as f:
        for r in kept + out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(kept)} dev + {len(out)} test questions -> {gold_path}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
