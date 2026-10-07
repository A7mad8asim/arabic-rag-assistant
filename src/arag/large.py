"""Very large datasets: index server-side totals instead of millions of rows.

Six foreign-trade tables (imports and exports by product code, country and month) hold 2.8 million
rows between them. Indexing every row would grow the index about sevenfold, and the questions people
ask of them are totals ("How much did Qatar import from Japan in 2023?"). For such tables the portal
computes the totals itself (`group_by` + `sum()` in the Explore API), and we index those *views*:

- totals by year and country (with the country's Arabic name), and
- totals by year and month.

Only additive measures are summed: value in riyals and weight in kilograms. Quantities are not, because
their units differ from product to product. Every view chunk says the figures are computed totals.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .chunking import columns
from .text import is_arabic

ADDITIVE = {"value (qr)": "value_qr", "weight (kg)": "weight_kg"}


@dataclass
class View:
    key: str  # short id used in chunk ids
    title_en: str
    title_ar: str
    group_by: list[str]  # field names
    measures: list[str]  # field names, summed

    def to_dict(self) -> dict:
        return asdict(self)


def _field(fields: list[dict], *labels: str) -> dict | None:
    for f in fields:
        label = (f.get("label_en") or f.get("label") or "").strip().lower()
        if any(label.startswith(l) for l in labels):
            return f
    return None


def plan_views(fields: list[dict]) -> list[View]:
    """The views worth indexing for a large dataset, or [] when it is not a table we know how to total."""
    year = _field(fields, "year")
    measures = [f["name"] for f in fields if (f.get("label_en") or "").strip().lower() in ADDITIVE
                and f.get("type") in ("int", "double")]
    if not year or not measures:
        return []
    views = []
    country = _field(fields, "country of")
    if country:
        twin = next((c.arabic_twin for c in columns(fields) if c.name == country["name"] and c.arabic_twin), None)
        group = [year["name"], country["name"]] + ([twin] if twin else [])
        views.append(View("by-country", "totals by year and country", "المجاميع حسب السنة والدولة", group, measures))
    month = _field(fields, "month")
    if month:
        views.append(View("by-month", "totals by year and month", "المجاميع حسب السنة والشهر",
                          [year["name"], month["name"]], measures))
    return views


MONTHS = [("January", "يناير"), ("February", "فبراير"), ("March", "مارس"), ("April", "أبريل"), ("May", "مايو"),
          ("June", "يونيو"), ("July", "يوليو"), ("August", "أغسطس"), ("September", "سبتمبر"), ("October", "أكتوبر"),
          ("November", "نوفمبر"), ("December", "ديسمبر")]


def _month_number(v) -> int | None:
    try:
        m = int(float(v))
    except (TypeError, ValueError):
        return None
    return m if 1 <= m <= 12 else None


def clean_view_rows(rows: list[dict], view: View, fields: list[dict]) -> list[dict]:
    """Make portal group-by results readable and ordered: years as YYYY, months by name in both languages
    ("March / مارس", so a question in either language matches), sums as whole numbers."""
    types = {f["name"]: f.get("type") for f in fields}
    labels = {f["name"]: (f.get("label_en") or "").strip().lower() for f in fields}
    out = []
    for r in rows:
        row, order = {}, []
        for name in view.group_by:
            v = r.get(name)
            if types.get(name) == "date" and isinstance(v, str):
                v = v[:4]
            elif labels.get(name, "").startswith("month") and _month_number(v):
                m = _month_number(v)
                order.append(m)
                v = f"{MONTHS[m - 1][0]} / {MONTHS[m - 1][1]}"
            elif v in (None, ""):
                v = "غير مذكور" if is_arabic(labels.get(name, "")) else "Not stated"
            row[name] = v
        for name in view.measures:
            v = r.get(name)
            row[name] = int(round(v)) if isinstance(v, (int, float)) else v
        out.append((str(row[view.group_by[0]]), order, row))
    return [row for _, _, row in sorted(out, key=lambda t: (t[0], t[1]))]


def view_fields(view: View, fields: list[dict]) -> list[dict]:
    """The dataset's field definitions for the view's columns (measures relabelled as totals)."""
    by_name = {f["name"]: f for f in fields}
    out = [by_name[n] for n in view.group_by]
    for n in view.measures:
        f = dict(by_name[n])
        f["label_en"] = "Total " + (f.get("label_en") or n)
        f["label_ar"] = "إجمالي " + (f.get("label_ar") or "")
        f["type"] = "int"
        out.append(f)
    return out
