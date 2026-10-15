# ─────────────────────────────────────────────
#  RAG-PoisonGuard  |  provenance.py
#  Checks that a paper really exists on arXiv and that its title
#  and abstract match what arXiv holds for that ID.
#
#  This verifies origin, not truth: a real arXiv paper that is itself
#  misleading still passes. Papers that are not on arXiv cannot be
#  verified, and lookups that fail are reported as unavailable, never
#  as fake.
# ─────────────────────────────────────────────

import difflib
import re
import time
import xml.etree.ElementTree as ET

import requests

ARXIV_API = "https://export.arxiv.org/api/query"
_ATOM = "{http://www.w3.org/2005/Atom}"
_VERSION = re.compile(r"v\d+$")


def base_id(arxiv_id: str) -> str:
    return _VERSION.sub("", arxiv_id.strip())


def _norm(text) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower()).split())


def _http_get(params: dict) -> str:  # separate function so tests can replace it
    r = requests.get(
        ARXIV_API, params=params, timeout=30,
        headers={"User-Agent": "rag-poisonguard-eval/1.0"},
    )
    r.raise_for_status()
    return r.text


def parse_feed(xml_text: str) -> dict:
    """Map base arXiv ID to {'title', 'abstract'} for every real entry in an Atom feed."""
    out = {}
    for entry in ET.fromstring(xml_text).findall(_ATOM + "entry"):
        id_text = entry.findtext(_ATOM + "id") or ""
        if "/abs/" not in id_text:
            continue
        title = " ".join((entry.findtext(_ATOM + "title") or "").split())
        if title.lower() == "error":
            continue
        abstract = " ".join((entry.findtext(_ATOM + "summary") or "").split())
        out[base_id(id_text.split("/abs/", 1)[1])] = {"title": title, "abstract": abstract}
    return out


def _fetch_batch(batch: list, delay: float, retries: int, records: dict, failed: set):
    for attempt in range(retries):
        try:
            feed = parse_feed(_http_get({"id_list": ",".join(batch), "max_results": len(batch)}))
        except (requests.RequestException, ET.ParseError):
            time.sleep(delay * (attempt + 1))
            continue
        for i in batch:
            records[i] = feed.get(i)  # None means arXiv has no such paper
        return
    if len(batch) == 1:
        failed.add(batch[0])
        return
    mid = len(batch) // 2  # one bad ID should not hide the rest
    _fetch_batch(batch[:mid], delay, retries, records, failed)
    _fetch_batch(batch[mid:], delay, retries, records, failed)


def lookup(ids, batch_size: int = 40, delay: float = 3.0, retries: int = 2):
    """
    Look IDs up on arXiv. Returns (records, failed):
      records[id] = {'title', 'abstract'} if the paper exists, None if it does not
      failed      = IDs whose lookup could not be completed
    """
    wanted = sorted({base_id(i) for i in ids})
    records, failed = {}, set()
    for start in range(0, len(wanted), batch_size):
        _fetch_batch(wanted[start:start + batch_size], delay, retries, records, failed)
        time.sleep(delay)  # arXiv asks for about one request every 3 seconds
    return records, failed


def check_paper(paper: dict, record, title_min: float = 0.9, abstract_min: float = 0.9) -> str:
    """'verified', 'not_found' or 'mismatch' for a paper against its arXiv record."""
    if record is None:
        return "not_found"
    title_ratio = difflib.SequenceMatcher(
        None, _norm(paper.get("title")), _norm(record.get("title"))
    ).ratio()
    a = set(_norm(paper.get("abstract")).split())
    b = set(_norm(record.get("abstract")).split())
    jaccard = len(a & b) / len(a | b) if (a | b) else 0.0
    return "verified" if title_ratio >= title_min and jaccard >= abstract_min else "mismatch"
