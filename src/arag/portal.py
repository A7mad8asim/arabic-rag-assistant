"""Download datasets from the Qatar Open Data portal (an Opendatasoft site, Explore API v2.1).

Each dataset is cached as data/raw/<dataset_id>.json: {"meta": <catalog entry>, "records": [...]},
so the index can be rebuilt offline and a second fetch only downloads what changed.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Iterable

import requests

from .config import Settings

log = logging.getLogger(__name__)


class Portal:
    def __init__(self, base_url: str, timeout_s: float = 60, session: requests.Session | None = None):
        self.api = f"{base_url.rstrip('/')}/api/explore/v2.1"
        self.timeout_s = timeout_s
        self.http = session or requests.Session()

    def _get(self, path: str, **params) -> object:
        for attempt in range(3):
            try:
                r = self.http.get(f"{self.api}{path}", params=params, timeout=self.timeout_s)
                r.raise_for_status()
                return r.json()
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
                if attempt == 2:
                    raise
                log.warning("retrying %s after %s", path, e)
                time.sleep(2 * (attempt + 1))

    def catalog(self, publisher: str | None = None) -> list[dict]:
        entries = self._get("/catalog/exports/json")
        if publisher:
            entries = [e for e in entries if e["metas"]["default"].get("publisher") == publisher]
        return entries

    def records(self, dataset_id: str) -> list[dict]:
        return self._get(f"/catalog/datasets/{dataset_id}/exports/json")


def dataset_path(raw_dir: Path, dataset_id: str) -> Path:
    return raw_dir / f"{dataset_id}.json"


def fetch(settings: Settings, portal: Portal | None = None, limit: int | None = None) -> dict:
    """Download the catalog and every dataset's rows. Returns counts for the CLI to print."""
    portal = portal or Portal(settings.portal_url)
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    entries = portal.catalog(settings.publisher)[:limit]
    stats = {"datasets": len(entries), "downloaded": 0, "unchanged": 0, "card_only": 0, "failed": 0}
    for i, meta in enumerate(entries, 1):
        ds = meta["dataset_id"]
        path = dataset_path(settings.raw_dir, ds)
        modified = meta["metas"]["default"].get("data_processed")
        if path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached["meta"]["metas"]["default"].get("data_processed") == modified:
                stats["unchanged"] += 1
                continue
        n_rows = meta["metas"]["default"].get("records_count") or 0
        try:
            if n_rows > settings.max_rows_per_dataset:
                records, stats["card_only"] = [], stats["card_only"] + 1
            else:
                records = portal.records(ds)
        except requests.RequestException as e:
            log.error("could not download %s: %s", ds, e)
            stats["failed"] += 1
            continue
        path.write_text(json.dumps({"meta": meta, "records": records}, ensure_ascii=False), encoding="utf-8")
        stats["downloaded"] += 1
        if i % 50 == 0:
            log.info("%d / %d datasets", i, len(entries))
    return stats


def load_raw(raw_dir: Path) -> Iterable[dict]:
    for path in sorted(raw_dir.glob("*.json")):
        yield json.loads(path.read_text(encoding="utf-8"))
