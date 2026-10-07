"""Turn each dataset into retrievable chunks.

- One *card* chunk per dataset: titles, descriptions, keywords and column names in both languages.
  It answers "which dataset covers X?" and keeps very large datasets findable.
- *Row* chunks: the table rows rendered as text, grouped by year where the dataset has a year
  column, at most `rows_per_chunk` rows each.

The portal stores Arabic values in twin columns (`municipality` = "Doha", `lbldy` = "الدوحة"). The
twins are found by their shared Arabic label and rendered together ("Doha / الدوحة"), so every chunk
is bilingual and a question in either language can match it lexically.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import groupby

from .text import is_arabic, normalize, strip_html


@dataclass
class Chunk:
    id: str
    dataset_id: str
    kind: str  # card | rows
    title_en: str
    title_ar: str
    text: str
    url: str
    years: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Column:
    name: str
    label_en: str
    label_ar: str
    type: str
    arabic_twin: str | None = None  # name of the column holding the Arabic version of this one


def columns(fields: list[dict]) -> list[Column]:
    """The dataset's columns, with each Arabic twin attached to its English column and dropped from the list."""
    cols = [Column(f["name"], (f.get("label_en") or f.get("label") or f["name"]).strip(), (f.get("label_ar") or "").strip(),
                   f.get("type", "text")) for f in fields]
    # Labels are matched after normalization: the portal spells the same label "الإقتصادي" and "الاقتصادى".
    twins = {normalize(c.label_ar): c for c in cols if c.type == "text" and is_arabic(c.label_en)}
    out = []
    for c in cols:
        key = normalize(c.label_ar)
        if twins.get(key) is c:
            continue
        if c.type == "text" and key in twins and twins[key].name not in {o.arabic_twin for o in out}:
            c.arabic_twin = twins[key].name
        out.append(c)
    # An Arabic column whose English partner was not found is kept as a normal column.
    paired = {c.arabic_twin for c in out}
    out += [t for t in twins.values() if t.name not in paired and t not in out]
    return out


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:,.2f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return f"{value:,}"
    return str(value).strip()


def render_row(row: dict, cols: list[Column]) -> str:
    parts = []
    for c in cols:
        value = row.get(c.name)
        if value in (None, ""):
            continue
        text = _fmt(value)
        twin = row.get(c.arabic_twin) if c.arabic_twin else None
        if twin and str(twin).strip() != text:
            text = f"{text} / {str(twin).strip()}"
        parts.append(f"{c.label_en}: {text}")
    return " · ".join(parts)


def _year_column(cols: list[Column]) -> Column | None:
    for c in cols:
        if c.type == "date" or c.name.lower() in ("year", "census_year", "reference_year"):
            return c
    return None


def dataset_chunks(dataset: dict, portal_url: str, rows_per_chunk: int = 20) -> list[Chunk]:
    meta = dataset["meta"]
    m = meta["metas"]["default"]
    ds = meta["dataset_id"]
    cols = columns(meta.get("fields", []))
    title_en, title_ar = m.get("title_en") or m.get("title") or ds, m.get("title_ar") or ""
    url = f"{portal_url.rstrip('/')}/explore/dataset/{ds}/"
    header = f"{title_en} | {title_ar}"

    records = dataset.get("records") or []
    views = dataset.get("views") or []
    year_col = _year_column(cols)
    year_source = records or [r for v in views for r in v["records"]]
    years = sorted({str(r.get(year_col.name))[:4] for r in year_source if r.get(year_col.name)}) if year_col else []

    card_lines = [
        header,
        strip_html(m.get("description_en") or m.get("description") or ""),
        strip_html(m.get("description_ar") or ""),
        "Topics: " + ", ".join((m.get("theme_en") or []) + (m.get("theme_ar") or [])),
        "Keywords: " + ", ".join((m.get("keyword_en") or []) + (m.get("keyword_ar") or [])),
        "Columns: " + "; ".join(f"{c.label_en} / {c.label_ar}" if c.label_ar and c.label_ar != c.label_en else c.label_en for c in cols),
        f"Rows: {m.get('records_count') or len(records)}" + (f" · Years: {years[0]}–{years[-1]}" if years else ""),
        ("Indexed as totals computed from its rows: " + "; ".join(v["view"]["title_en"] for v in views)) if views else "",
    ]
    chunks = [Chunk(f"{ds}#card", ds, "card", title_en, title_ar, "\n".join(l for l in card_lines if l.strip()), url, years)]
    chunks += _row_chunks(ds, "rows", header, records, cols, rows_per_chunk, title_en, title_ar, url)
    for v in views:
        view = v["view"]
        view_header = (f"{title_en} — {view['title_en']} (computed from the dataset's rows) | "
                       f"{title_ar} — {view['title_ar']}")
        chunks += _row_chunks(ds, view["key"], view_header, v["records"], columns(v["fields"]), rows_per_chunk,
                              title_en, title_ar, url)
    return chunks


def _row_chunks(ds: str, key: str, header: str, records: list[dict], cols: list[Column], rows_per_chunk: int,
                title_en: str, title_ar: str, url: str) -> list[Chunk]:
    """Rows rendered as text under `header`, grouped by year, at most `rows_per_chunk` per chunk."""
    year_col = _year_column(cols)

    def year_of(r):
        return str(r.get(year_col.name))[:4] if year_col and r.get(year_col.name) else ""

    ordered = sorted(records, key=year_of) if year_col else records
    chunks, n = [], 0
    for year, group in groupby(ordered, key=year_of):
        group = list(group)
        for start in range(0, len(group), rows_per_chunk):
            lines = [render_row(r, cols) for r in group[start : start + rows_per_chunk]]
            n += 1
            chunks.append(Chunk(f"{ds}#{key}{n}", ds, "rows", title_en, title_ar,
                                header + "\n" + "\n".join(lines), url, [year] if year else []))
    return chunks
