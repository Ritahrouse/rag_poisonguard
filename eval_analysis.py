# ─────────────────────────────────────────────
#  RAG-PoisonGuard  |  eval_analysis.py
#  Pure analysis for the batch evaluation: detector variants,
#  metrics with confidence intervals, and the markdown report.
#  Needs only numpy (via the auditor), no torch or FAISS.
# ─────────────────────────────────────────────

import copy
import io
import math
import re
from contextlib import redirect_stdout
from typing import Optional

import provenance as prov
from agents.auditor_agent import AuditorAgent, FlagAgent, MetadataInspector, SimilarityChecker

DEFAULT_THRESHOLD = -1.7
THRESHOLDS = [-1.0, -1.25, -1.5, -1.7, -2.0, -2.5, -3.0]
VARIANTS = ["similarity", "metadata", "id_date", "full", "full+id_date", "docstring", "docstring+id_date"]
PROVENANCE_VARIANTS = ["provenance", "provenance_title", "full+provenance"]
VARIANT_LABELS = {
    "similarity": "Similarity z-score only",
    "metadata": "Metadata checks only",
    "id_date": "ID/date consistency only (new)",
    "full": "Current AuditorAgent",
    "full+id_date": "Current AuditorAgent + ID/date (new)",
    "docstring": "FlagAgent as documented (similarity alone does not flag)",
    "docstring+id_date": "As documented + ID/date (new)",
    "provenance": "arXiv provenance check only (new)",
    "provenance_title": "arXiv check on ID and title only, no abstract (new)",
    "full+provenance": "Current AuditorAgent + provenance (new)",
}
FALLBACK_MARKER = "dense retrieval models are more susceptible than sparse retrievers"

_ID_RE = re.compile(r"^(\d{2})(\d{2})\.\d{4,5}(v\d+)?$")


def is_fallback(poison: dict) -> bool:
    """True if the poison abstract is the injector's canned fallback text."""
    return FALLBACK_MARKER in poison.get("abstract", "")


def id_date_inconsistent(paper: dict, max_lag_months: int = 1) -> bool:
    """
    arXiv IDs start with YYMM. The ID is assigned at announcement, so its month
    cannot be earlier than the submission month (the 'published' date) and is
    normally the same or the next month. Flag IDs that break this.
    Old-style IDs (before 2007) are not checked.
    """
    m = _ID_RE.match(paper.get("id", ""))
    published = paper.get("published", "")
    if not m or len(published) < 7:
        return False
    try:
        id_idx = (2000 + int(m.group(1))) * 12 + int(m.group(2))
        pub_idx = int(published[:4]) * 12 + int(published[5:7])
    except ValueError:
        return False
    lag = id_idx - pub_idx
    return lag < 0 or lag > max_lag_months


def wilson(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (centre - margin) / denom), min(1.0, (centre + margin) / denom))


def _silent(fn, *args):
    with redirect_stdout(io.StringIO()):
        return fn(*args)


def _provenance_flags(papers: list, provenance: dict, abstract_min: float = 0.9) -> set:
    flagged = set()
    for p in papers:
        record = provenance.get(prov.base_id(p["id"]), "unavailable")
        if isinstance(record, str):  # lookup failed: unverifiable, not proven fake
            continue
        if prov.check_paper(p, record, abstract_min=abstract_min) != "verified":
            flagged.add(p["id"])
    return flagged


def flagged_ids(variant: str, papers: list, threshold: float = DEFAULT_THRESHOLD,
                provenance: Optional[dict] = None) -> set:
    papers = copy.deepcopy(papers)
    if variant in PROVENANCE_VARIANTS:
        if provenance is None:
            raise ValueError("provenance data is required for this variant")
        flagged = _provenance_flags(papers, provenance, 0.0 if variant == "provenance_title" else 0.9)
        if variant == "full+provenance":
            flagged |= flagged_ids("full", papers, threshold)
        return flagged
    if variant == "similarity":
        out = SimilarityChecker(threshold).check(papers)
        return {p["id"] for p in out if p["similarity_suspicious"]}
    if variant == "metadata":
        out = MetadataInspector().check(papers)
        for p in out:
            p["similarity_suspicious"] = False
        out = FlagAgent().decide(out)
        return {p["id"] for p in out if p["flagged"]}
    if variant.startswith("id_date:"):
        lag = int(variant.split(":", 1)[1])
        return {p["id"] for p in papers if id_date_inconsistent(p, lag)}
    if variant == "id_date":
        return {p["id"] for p in papers if id_date_inconsistent(p)}
    if variant == "full":
        _, quarantined = _silent(AuditorAgent(threshold).audit, papers)
        return {p["id"] for p in quarantined}
    if variant == "full+id_date":
        return flagged_ids("full", papers, threshold) | flagged_ids("id_date", papers, threshold)
    if variant == "docstring":
        out = SimilarityChecker(threshold).check(papers)
        out = MetadataInspector().check(out)
        flagged = set()
        for p in out:
            critical = any(any(cf in f for cf in FlagAgent.CRITICAL_FLAGS) for f in p["metadata_flags"])
            if critical or (p["similarity_suspicious"] and p["metadata_flags"]):
                flagged.add(p["id"])
        return flagged
    if variant == "docstring+id_date":
        return flagged_ids("docstring", papers, threshold) | flagged_ids("id_date", papers, threshold)
    raise ValueError(f"unknown variant: {variant}")


def evaluate(runs: list, variant: str, threshold: float = DEFAULT_THRESHOLD,
             provenance: Optional[dict] = None) -> dict:
    """Detection metrics for one detector variant over all runs."""
    n_runs = len(runs)
    clean_papers = fp_papers = clean_runs_with_fp = 0
    for r in runs:
        flagged = flagged_ids(variant, r["clean"], threshold, provenance)
        clean_papers += len(r["clean"])
        fp_papers += len(flagged)
        clean_runs_with_fp += 1 if flagged else 0

    retrieved = detected = top3 = top5 = asr_top5 = 0
    for r in runs:
        p = r["poison"]
        if p["rank"] is None:
            continue
        retrieved += 1
        top3 += 1 if p["rank"] <= 3 else 0
        top5 += 1 if p["rank"] <= 5 else 0
        if p["poison_id"] in flagged_ids(variant, p["retrieved"], threshold, provenance):
            detected += 1
        elif p["rank"] <= 5:
            asr_top5 += 1

    return {
        "n_runs": n_runs,
        "retrieved": retrieved,
        "detected": detected,
        "asr_after": retrieved - detected,
        "top3": top3,
        "top5": top5,
        "asr_after_top5": asr_top5,
        "clean_papers": clean_papers,
        "fp_papers": fp_papers,
        "clean_runs_with_fp": clean_runs_with_fp,
    }


def false_positives(runs: list, variant: str, threshold: float = DEFAULT_THRESHOLD,
                    provenance: Optional[dict] = None) -> list:
    """Legitimate papers flagged by a detector, as (query, id, published, title)."""
    out = []
    for r in runs:
        flagged = flagged_ids(variant, r["clean"], threshold, provenance)
        flags = {p["id"]: p["metadata_flags"] for p in MetadataInspector().check(copy.deepcopy(r["clean"]))}
        for p in r["clean"]:
            if p["id"] in flagged:
                out.append((r["query"], p["id"], p.get("published", ""), p.get("title", ""),
                            "; ".join(flags.get(p["id"], []))))
    return out


def provenance_summary(runs: list, provenance: dict) -> dict:
    """Lookup outcomes for the legitimate papers and for the poison papers."""
    def status(paper):
        record = provenance.get(prov.base_id(paper["id"]), "unavailable")
        return "unavailable" if isinstance(record, str) else prov.check_paper(paper, record)

    legit = {}
    for r in runs:
        for p in r["clean"]:
            legit[p["id"]] = status(p)
    poison = [status({"id": r["poison"]["poison_id"], "title": r["poison"]["title"],
                      "abstract": r["poison"]["abstract"]}) for r in runs]
    keys = ("verified", "not_found", "mismatch", "unavailable")
    return {
        "legit": {k: sum(1 for v in legit.values() if v == k) for k in keys},
        "poison": {k: sum(1 for v in poison if v == k) for k in keys},
    }


def poison_z(run: dict) -> Optional[float]:
    p = run["poison"]
    if p["rank"] is None:
        return None
    papers = _silent(SimilarityChecker(DEFAULT_THRESHOLD).check, copy.deepcopy(p["retrieved"]))
    for paper in papers:
        if paper["id"] == p["poison_id"]:
            return paper["similarity_z_score"]
    return None


def _ci(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({100 * k / n:.0f}%, CI {100 * lo:.0f}-{100 * hi:.0f}%)" if n else "n/a"


def _rate(k: int, n: int) -> str:
    return f"{100 * k / n:.0f}%" if n else "n/a"


def render_report(runs: list, meta: dict, provenance: Optional[dict] = None) -> str:
    n = len(runs)
    variants = VARIANTS + (PROVENANCE_VARIANTS if provenance else [])
    fallback_n = sum(1 for r in runs if r["poison"].get("is_fallback"))
    base = evaluate(runs, "full", DEFAULT_THRESHOLD)

    lines = [
        "# RAG-PoisonGuard batch evaluation",
        "",
        "## Setup",
        f"- Queries: {n}, top-k: {meta.get('top_k')}, attacker: {meta.get('attacker')}",
        f"- Poison papers using the canned fallback abstract: {fallback_n}/{n}",
        f"- Default similarity threshold: z <= {DEFAULT_THRESHOLD}",
        f"- Run date: {meta.get('date')}",
        "- Every arXiv paper is treated as legitimate. One poison paper is injected per query.",
        "",
        "## Main result (current AuditorAgent, default threshold)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Poison retrieved in top-k (ASR before mitigation) | {_ci(base['retrieved'], n)} |",
        f"| Poison in top 3 | {_ci(base['top3'], n)} |",
        f"| Poison in top 5 | {_ci(base['top5'], n)} |",
        f"| Poison caught, among retrieved | {_ci(base['detected'], base['retrieved'])} |",
        f"| Attack still reaches the user (ASR after mitigation) | {_ci(base['asr_after'], n)} |",
        f"| Attack reaches the user from the top 5 | {_ci(base['asr_after_top5'], n)} |",
        f"| Legitimate papers wrongly flagged | {_ci(base['fp_papers'], base['clean_papers'])} |",
        f"| Clean queries with at least one false alarm | {_ci(base['clean_runs_with_fp'], n)} |",
        "",
        f"## Detector ablation (threshold {DEFAULT_THRESHOLD})",
        "",
        "| Detector | Caught (of retrieved) | ASR after (top-k) | ASR after (top 5) | Papers wrongly flagged | Clean queries with false alarm |",
        "|---|---|---|---|---|---|",
    ]
    for v in variants:
        e = evaluate(runs, v, DEFAULT_THRESHOLD, provenance)
        lines.append(
            f"| {VARIANT_LABELS[v]} | {_rate(e['detected'], e['retrieved'])} "
            f"({e['detected']}/{e['retrieved']}) | {_rate(e['asr_after'], n)} "
            f"| {_rate(e['asr_after_top5'], n)} "
            f"| {_rate(e['fp_papers'], e['clean_papers'])} ({e['fp_papers']}/{e['clean_papers']}) "
            f"| {_rate(e['clean_runs_with_fp'], n)} ({e['clean_runs_with_fp']}/{n}) |"
        )

    lines += [
        "",
        "## Threshold sweep (current AuditorAgent)",
        "",
        "| z threshold | Caught (of retrieved) | ASR after | Papers wrongly flagged | Clean queries with false alarm |",
        "|---|---|---|---|---|",
    ]
    for t in THRESHOLDS:
        e = evaluate(runs, "full", t)
        lines.append(
            f"| {t} | {_rate(e['detected'], e['retrieved'])} | {_rate(e['asr_after'], n)} "
            f"| {_rate(e['fp_papers'], e['clean_papers'])} | {_rate(e['clean_runs_with_fp'], n)} |"
        )

    lines += [
        "",
        "## ID/date lag sweep",
        "How many months after the submission date the arXiv ID month may fall before a paper is flagged.",
        "An ID month before the submission month is never allowed.",
        "",
        "| Max lag (months) | Caught (of retrieved) | ASR after (top 5) | Papers wrongly flagged | Clean queries with false alarm |",
        "|---|---|---|---|---|",
    ]
    for lag in (0, 1, 2, 3, 999):
        e = evaluate(runs, f"id_date:{lag}")
        label = "no upper limit" if lag == 999 else str(lag)
        lines.append(
            f"| {label} | {_rate(e['detected'], e['retrieved'])} ({e['detected']}/{e['retrieved']}) "
            f"| {_rate(e['asr_after_top5'], n)} "
            f"| {_rate(e['fp_papers'], e['clean_papers'])} ({e['fp_papers']}/{e['clean_papers']}) "
            f"| {_rate(e['clean_runs_with_fp'], n)} ({e['clean_runs_with_fp']}/{n}) |"
        )

    lines += [
        "",
        "## Per query",
        "",
        "| Query | Poison rank | Poison z | Caught (current) | Caught (+ID/date) | Clean false alarms (current) |",
        "|---|---|---|---|---|---|",
    ]
    for r in runs:
        p = r["poison"]
        z = poison_z(r)
        if p["rank"] is None:
            caught = caught_id = "not retrieved"
        else:
            caught = "yes" if p["poison_id"] in flagged_ids("full", p["retrieved"]) else "no"
            caught_id = "yes" if p["poison_id"] in flagged_ids("full+id_date", p["retrieved"]) else "no"
        fa = len(flagged_ids("full", r["clean"]))
        lines.append(
            f"| {r['query'][:50]} | {p['rank'] or '-'} | {'-' if z is None else z} "
            f"| {caught} | {caught_id} | {fa} |"
        )

    if provenance:
        summ = provenance_summary(runs, provenance)
        lines += [
            "", "## arXiv provenance lookup", "",
            "| Papers | Verified | Not on arXiv | Text mismatch | Lookup unavailable |",
            "|---|---|---|---|---|",
            "| Legitimate (unique) | " + " | ".join(str(summ["legit"][k]) for k in
                                                  ("verified", "not_found", "mismatch", "unavailable")) + " |",
            "| Poison | " + " | ".join(str(summ["poison"][k]) for k in
                                       ("verified", "not_found", "mismatch", "unavailable")) + " |",
        ]
    fp_variants = ["id_date", "metadata"] + (["provenance"] if provenance else [])
    for v in fp_variants:
        fps = false_positives(runs, v, DEFAULT_THRESHOLD, provenance)
        lines += ["", f"## Legitimate papers flagged by: {VARIANT_LABELS[v]}", ""]
        if not fps:
            lines.append("None.")
        for q, pid, pub, title, flags in fps[:15]:
            lines.append(f"- {pid} (published {pub}): {title[:70]}  [query: {q[:40]}]"
                         + (f"  [flags: {flags}]" if flags else ""))
        if len(fps) > 15:
            lines.append(f"- ... and {len(fps) - 15} more")

    lines += [
        "",
        "## Caveats",
        "- Confidence intervals are 95% Wilson intervals. With about 30 queries they are wide.",
        "- The z-score is computed inside each retrieved set, so it depends on that set's spread.",
        "- The ID/date check exploits a flaw in the attacker (ID and date chosen independently).",
        "  A consistent attacker would defeat it. Treat it as one iteration of a red/blue loop.",
        "- Queries were chosen by hand and one attacker goal is used throughout.",
        "- Legitimate-paper labels are assumed, not verified.",
        "- Provenance verifies origin, not truth. A real arXiv paper written to mislead still passes,",
        "  and papers that are not on arXiv cannot be verified at all.",
        "",
    ]
    return "\n".join(lines)
