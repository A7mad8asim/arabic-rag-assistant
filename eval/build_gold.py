"""Add the held-out test questions and the large-table questions to eval/gold.jsonl, verifying every answer
against the downloaded data.

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

# The "large" split: the 16 datasets above 5,000 rows. Trade tables are indexed as server-side totals,
# so their questions name a view ("dataset#by-country"). Kept apart from the 120-question benchmark.
LARGE = [
    ("L01", "qatar-imports-2019-2024-copy#by-country", ["Year: 2023", "Country of Origin: Japan /"], "3,601,310,832",
     "What was the value of Qatar's imports from Japan in 2023?",
     "كم بلغت قيمة واردات قطر من اليابان في 2023؟", False),
    ("L02", "qatar-export-statistics-2019-2024#by-country", ["Year: 2022", "Country of Destination: China ·"], "75,647,195,422",
     "What was the value of Qatar's exports to China in 2022?",
     "كم بلغت قيمة صادرات قطر إلى الصين في 2022؟", False),
    ("L03", "qatar-export-statistics-2019-2024-copy#by-month", ["Year: 2023", "Month: March"], "30,846,246,645",
     "What was the total value of Qatar's exports in March 2023?",
     "كم كانت القيمة الإجمالية لصادرات قطر في مارس 2023؟", False),
    # 2014-2018 appear in two import tables whose totals differ by a few riyals, so these use years only one covers.
    ("L04", "qatar-imports-2012-2018-copy#by-country", ["Year: 2013", "Country of Origin: Germany /"], "6,330,541,976",
     "What was the value of Qatar's imports from Germany in 2013?",
     "كم قيمة اللي استوردته قطر من ألمانيا سنة 2013؟", True),
    ("L05", "qatar-imports-2025-2026#by-month", ["Year: 2025", "Month: July"], "11,523,988,197",
     "What was the total value of Qatar's imports in July 2025?",
     "كم كانت القيمة الإجمالية لواردات قطر في يوليو 2025؟", False),
    ("L06", "qatar-export-statistics-2019-2024-copy#by-country", ["Year: 2022", "United States Of America"], "6,290,786,951",
     "What was the value of Qatar's exports to the United States in 2022?",
     "كم صدّرت قطر لأمريكا سنة 2022 بالريال؟", True),
    ("L07", "qatar-imports-2019-2024-copy#by-country", ["Year: 2021", "Country of Origin: Indonesia /"], "91,301,636",
     "What was the weight of Qatar's imports from Indonesia in 2021, in kilograms?",
     "كم كان وزن واردات قطر من إندونيسيا في 2021 بالكيلوغرام؟", False),
    ("L08", "total-registered-deaths-by-age-group-and-cause-of-death-icd",
     ["Year: 2022", "Gender: Males", "Age Group: 50-54", "Cause of Death: Other forms of heart disease"], "76",
     "How many men aged 50-54 died of other forms of heart disease in 2022?",
     "كم عدد وفيات الذكور من الفئة العمرية 50-54 بسبب أشكال أخرى من أمراض القلب في 2022؟", False),
    ("L09", "registered-qataris-deaths-by-age-group-and-cause-of-death-icd",
     ["Year: 2021", "Gender: Males", "Age Group: 15-19", "Cause of Death: Transport accidents"], "13",
     "How many Qatari males aged 15-19 died in transport accidents in 2021?",
     "كم شاب قطري بين 15 و19 سنة توفى بحوادث النقل في 2021؟", True),
    ("L10", "main-economic-indicators-by-main-economic-activity-energy-and-industry",
     ["Year: 2017", "Main Economic Activity: Manufacture of textiles /", "Indicator: Productivity of Employee"], "124,507",
     # Three tables publish this (all establishments, 10+ employees, fewer than 10): the question names which.
     "What was the productivity per employee in textile manufacturing in 2017, across all establishments?",
     "كم كانت إنتاجية المشتغل في صناعة المنسوجات عام 2017 لجميع المنشآت؟", False),
    ("L11", "estimates-of-value-of-intermediate-services-by-main-economic-activity-energy-and-industry-10",
     ["Year: 2017", "Main Economic Activity: Manufacture of food products /", "Machinery and Equipment Maintenance"], "15,484",
     "What was the value of machinery and equipment maintenance services used in food manufacturing in 2017?",
     "كم كانت قيمة خدمات صيانة الآلات والمعدات في صناعة المنتجات الغذائية عام 2017؟", False),
    ("L12", "registered-deaths-by-nationality-gender-and-age-annual",
     ["Year: 2020", "of Age: 72", "Nationality: Qataris", "Gender: Males"], "8",
     "How many deaths of 72-year-old Qatari men were registered in 2020?",
     "كم عدد وفيات الذكور القطريين بعمر 72 سنة المسجلة في 2020؟", False),
]

# The "test2" split: a fresh held-out set written on 8 Oct 2026 after the row-matching fixes were designed
# (from failures in the test split) and before they were evaluated. Datasets and rows were sampled at random
# (seed 2026) from tables not used anywhere else in the gold set.
HELD2 = [
    ("H01", "economically-active-qatari-females-15-years-and-above-by-educational-status-and-occupation",
     ["Year: 2020", "Technicians and Associate", "University and above"], "2,501",
     "How many economically active Qatari women with a university degree or higher worked as technicians and associate professionals in 2020?",
     "كم عدد القطريات النشيطات اقتصادياً من حملة الشهادة الجامعية فما فوق اللاتي عملن فنيات واختصاصيات مساعدات في 2020؟", False),
    ("H02", "number-of-employees-and-estimates-of-compensation-of-employees-by-nationality-and-main-economic-activity-activity-codes-53-5229-isic-rev4-10-or-more-employees",
     ["Year: 2021", "Other passenger land transport (without time schedule)", "Nationality: Qataris"], "18",
     "How many Qataris worked in other passenger land transport (without a time schedule) in establishments with 10 or more employees in 2021?",
     "كم عدد القطريين العاملين في النقل البري للركاب غير المحدد بمواعيد في المنشآت التي يعمل بها 10 موظفين أو أكثر عام 2021؟", False),
    ("H03", "sdg-11-7-2-number-of-victims-of-physical-or-sexual-harassment-by-type-of-person-and-gender",
     ["Year: 2016", "Without disability", "Gender: Males"], "673",
     "How many male victims of physical or sexual harassment without a disability were recorded in 2016?",
     "كم رجل سليم بدون إعاقة تعرض لتحرش جسدي أو جنسي سنة 2016؟", True),
    ("H04", "sdg-13-1-1-number-of-injured-persons-and-deaths-attributed-to-disasters-per-100-000-population-by",
     ["Year: 2021", "Coronavirus", "Impact Type: Deaths", "15-64"], "9.9",
     "What was the coronavirus death rate per 100,000 people aged 15-64 in 2021?",
     "كم كان معدل وفيات وباء كورونا لكل 100 ألف نسمة للفئة العمرية 15-64 سنة في 2021؟", False),
    ("H05", "trainees-at-the-private-training-centers-according-to-the-educational-status-gender-nationality-and",
     ["Year: 2022", "Field of Training: Languages", "Non Qataris", "Educational Status: University", "Gender: Females"], "338",
     "How many non-Qatari women with a university education trained in languages at private training centres in 2022?",
     "كم عدد غير القطريات الجامعيات اللاتي تدربن في مجال اللغات في المراكز التدريبية الخاصة عام 2022؟", False),
    ("H06", "economically-active-population-15-years-and-above-by-educational-status-and-occupation",
     ["Year: 2020", "Service Workers and Shop", "Educational Status: Read & Write"], "4,551",
     "How many economically active people who can read and write worked as service workers or shop and market sales workers in 2020?",
     "كم عدد النشيطين اقتصادياً ممن يقرأون ويكتبون والعاملين في الخدمات والباعة في المحلات التجارية والأسواق عام 2020؟", False),
    ("H07", "estimates-of-value-of-intermediate-services-by-main-economic-activity-wholesale-and-retail-trade",
     ["Year: 2021", "Activity Code: 46 ·", "Transportation"], "56,141",
     "What was the value of transportation services used by wholesale trade (excluding motor vehicles) in 2021?",
     "كم كانت قيمة خدمات النقل والانتقالات في تجارة الجملة باستثناء المركبات عام 2021؟", False),
    ("H08", "services-provided-to-cases-received-by-the-protection-and-social-rehabilitation-center-by",
     ["Year: 2014", "External Unit Service", "Nationality: Non-Qatari", "Age Group: Adult", "Gender: Females"], "28",
     "How many external unit services did the Protection and Social Rehabilitation Center provide to adult non-Qatari women in 2014?",
     "كم خدمة وحدة خارجية قدمها مركز الحماية والتأهيل الاجتماعي لحريم بالغات غير قطريات سنة 2014؟", True),
    ("H09", "staff-providing-services-for-disabled-at-rumeilah-hospital-by-occupation-and-gender",
     ["Year: 2019", "Prosthetics Technician", "Gender: Males"], "10",
     "How many male prosthetics technicians served people with disabilities at Rumeilah Hospital in 2019?",
     "كم عدد فنيي الأطراف الصناعية الذكور في مستشفى الرميلة عام 2019؟", False),
    ("H10", "sdg-16-7-1-number-of-positions-of-judges-and-court-clerks-by-court-level-and-disability-status",
     ["Year: 2022", "Primary Courts", "Court clerks", "Without disability"], "136",
     "How many court clerk positions held by people without a disability were there in the primary courts in 2022?",
     "كم عدد وظائف رؤساء الأقلام من غير ذوي الإعاقة في المحاكم الابتدائية عام 2022؟", False),
    ("H11", "quantities-of-pesticides-for-the-control-of-palm-pests-by-type",
     ["Year: 2012", "Betalarve"], "142",
     "How many litres of Betalarve 2.5% were used against palm pests in 2012?",
     "كم لتراً من مبيد بيتالارف 2.5% استخدم لمكافحة آفات النخيل في 2012؟", False),
    ("H12", "sdg-11-7-2-number-of-victims-of-physical-or-sexual-harassment-by-age-group-and-gender",
     ["Year: 2020", "Under 15", "Gender: Males"], "21",
     "How many boys under 15 were victims of physical or sexual harassment in 2020?",
     "كم ولد تحت 15 سنة تعرض لتحرش جسدي أو جنسي سنة 2020؟", True),
]

# test2 alternatives: identical rows in sibling tables (found when answers had the right value but cited the
# sibling; each is verified by the script like any other alternative).
ALSO_ACCEPT_HELD2 = {
    "H04": ["sdg-1-5-1-number-of-injured-persons-and-deaths-attributed-to-disasters-per-100-000-population-by-age"],
    "H09": ["special-needs-statistics-number-of-staff-providing-services-for-disabled-at-rumailah-hospital-and"],
}

UNANSWERABLE = [
    ("U06", "How many hotel nights did Qatari guests spend in 2040?", "كم عدد ليالي الإقامة للنزلاء القطريين في الفنادق عام 2040؟"),
    ("U07", "What is the population of Tokyo?", "كم عدد سكان طوكيو؟"),
    ("U08", "Who is the coach of Qatar's national football team?", "مين مدرب منتخب قطر لكرة القدم؟"),
    ("U09", "How many aircraft did the national carrier have in March 1985?", "كم طائرة كانت عند الناقل الوطني في مارس 1985؟"),
    ("U10", "What will Qatar's GDP growth rate be in 2035?", "كم سيكون معدل نمو الناتج المحلي الإجمالي لقطر في 2035؟"),
]


def rendered_rows(data: dict, view: str | None = None) -> list[str]:
    """The dataset's rows as the index renders them; with `view`, the rows of that totals view; with view="*", all."""
    parts = []
    if view in (None, "*"):
        parts.append((columns(data["meta"]["fields"]), data["records"]))
    for v in data.get("views", []):
        if view == "*" or v["view"]["key"] == view:
            parts.append((columns(v["fields"]), v["records"]))
    return [render_row(r, cols) for cols, records in parts for r in records]


def year_numbers(text: str) -> set[str]:
    return {n for n in numbers_in(text) if len(n) == 4 and n[:2] in ("19", "20")}


def main() -> int:
    raw = Settings.from_env().raw_dir
    titles: dict[str, list[str]] = {}
    large: dict[str, dict] = {}  # datasets above 5,000 rows, for finding identical rows in sibling tables
    for p in raw.glob("*.json"):
        data = json.loads(p.read_text(encoding="utf-8"))
        meta = data["meta"]
        titles.setdefault(meta["metas"]["default"].get("title_en", "").strip().casefold(), []).append(meta["dataset_id"])
        if (meta["metas"]["default"].get("records_count") or 0) > 5000:
            large[meta["dataset_id"]] = data

    out, problems = [], []

    def add_lookups(specs, split, also_accept, scan_large):
        for tid, target, where, answer, en, ar, dialect in specs:
            ds, _, view = target.partition("#")
            path = dataset_path(raw, ds)
            if not path.exists():
                problems.append(f"{tid}: dataset {ds} not downloaded")
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            hits = [r for r in rendered_rows(data, view or None) if all(w in r for w in where)]
            if len(hits) != 1:
                problems.append(f"{tid}: {len(hits)} rows match {where}")
                continue
            if not numbers_in(answer) <= numbers_in(hits[0]):
                problems.append(f"{tid}: {answer} not in row: {hits[0][:200]}")
                continue
            title = data["meta"]["metas"]["default"].get("title_en", "").strip().casefold()
            accepted = [ds] + sorted(d for d in titles.get(title, []) if d != ds)
            need = numbers_in(answer) | year_numbers(en)
            for alt in also_accept.get(tid, []):
                alt_data = json.loads(dataset_path(raw, alt).read_text(encoding="utf-8"))
                if any(need <= numbers_in(r) for r in rendered_rows(alt_data, "*")):
                    accepted.append(alt)
                else:
                    problems.append(f"{tid}: alternative {alt} has no row with {sorted(need)}")
            # Sibling trade tables share totals (an export table and its "for visualisation" copy). Only for answers
            # of 100,000 or more: a small number like 13 plus a year turns up in unrelated tables by coincidence.
            if scan_large and float(answer.replace(",", "")) >= 100_000:
                accepted += sorted(d for d, other in large.items() if d not in accepted
                                   and any(need <= numbers_in(r) for r in rendered_rows(other, "*")))
            for lang, q in (("en", en), ("ar", ar)):
                out.append({"id": tid, "lang": lang, "kind": "lookup", "question": q, "dataset_ids": accepted,
                            "answer": [answer], "dialect": dialect and lang == "ar", "split": split})

    add_lookups(LOOKUPS, "test", ALSO_ACCEPT, scan_large=False)
    add_lookups(LARGE, "large", {}, scan_large=True)
    add_lookups(HELD2, "test2", ALSO_ACCEPT_HELD2, scan_large=False)
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
    counts = {s: sum(r["split"] == s for r in out) for s in ("test", "large", "test2")}
    print(f"{len(kept)} dev + {counts['test']} test + {counts['large']} large + {counts['test2']} test2 questions -> {gold_path}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
