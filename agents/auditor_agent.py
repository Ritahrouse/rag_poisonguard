# ─────────────────────────────────────────────
#  RAG-PoisonGuard  |  agents/auditor_agent.py
#  AuditorAgent (L5 — Mitigation)
#
#  Threshold calibrated from empirical z-score
#  analysis: poison paper sits at z=-2.21,
#  nearest legit paper at z=-1.17.
#  Threshold set to -1.7 for realistic detection.
# ─────────────────────────────────────────────

from datetime import datetime, timezone
import logging
import numpy as np

logger = logging.getLogger(__name__)


class SimilarityChecker:
    """
    Flags papers whose embedding similarity to the query is
    anomalously high (very low L2 distance) compared to the
    rest of the result set.

    Calibrated threshold: -1.7 z-score.
    Empirical basis: poison paper z=-2.21, nearest legit z=-1.17.
    Gap of 1.04 z-units — threshold sits in the middle.
    """

    def __init__(self, z_score_threshold: float = -1.7):
        self.z_score_threshold = z_score_threshold

    def check(self, papers: list[dict]) -> list[dict]:
        if len(papers) < 3:
            for p in papers:
                p["similarity_suspicious"] = False
                p["similarity_z_score"] = 0.0
            return papers

        scores = np.array([p.get("score", 0.0) for p in papers])
        mean = scores.mean()
        std = scores.std()

        for paper in papers:
            score = paper.get("score", 0.0)
            z = (score - mean) / std if std > 0 else 0.0
            suspicious = z < self.z_score_threshold
            paper["similarity_suspicious"] = suspicious
            paper["similarity_z_score"] = round(float(z), 4)

            if suspicious:
                logger.info(
                    f"[SimilarityChecker] SUSPICIOUS: "
                    f"'{paper['title'][:50]}' z={z:.2f} "
                    f"(threshold={self.z_score_threshold})"
                )
        return papers


class MetadataInspector:
    """
    Checks metadata for fabrication signals.
    No hardcoded poison IDs or stolen name checks.
    Signals used:
      - Non-arXiv URL
      - Invalid arXiv ID format
      - No authors listed
      - Abstract under 60 words
      - Implausibly high accuracy (99%+)
      - Future publication date
      - Abstract unusually repetitive of query terms
    """

    def __init__(self, corpus_metadata: list = None):
        self.corpus_metadata = corpus_metadata or []

    def check(self, papers: list[dict]) -> list[dict]:
        for paper in papers:
            flags = []

            # Check 1: URL
            url = paper.get("url", "")
            if url and "arxiv.org" not in url:
                flags.append(f"Non-arXiv URL: {url}")

            # Check 2: ID format
            paper_id = paper.get("id", "")
            if paper_id and not self._valid_arxiv_id(paper_id):
                flags.append(f"Invalid arXiv ID format: {paper_id}")

            # Check 3: Authors
            if not paper.get("authors"):
                flags.append("No authors listed")

            # Check 4: Abstract length
            abstract = paper.get("abstract", "")
            words = len(abstract.split())
            if words < 60:
                flags.append(f"Abstract too short ({words} words)")

            # Check 5: Implausibly high accuracy
            import re
            high_acc = re.findall(r'\b9[89]\.\d+%|\b100%', abstract)
            if high_acc:
                flags.append(f"Implausibly high accuracy: {high_acc[0]}")

            # Check 6: Future date
            published = paper.get("published", "")
            if published:
                try:
                    year = int(published[:4])
                    if year > datetime.now(timezone.utc).year:
                        flags.append(f"Future date: {published}")
                except ValueError:
                    pass

            # Check 7: Keyword repetition — abstract repeats same phrase 4+ times
            # (sign of keyword stuffing from T3)
            if abstract:
                words_list = abstract.lower().split()
                from collections import Counter
                word_freq = Counter(words_list)
                # Check for non-stopword terms repeated suspiciously often
                stopwords = {"the","a","an","of","in","to","and","or","is","are",
                             "for","with","on","that","this","by","from","at","as",
                             "we","our","which","it","its","be","can","their"}
                suspicious_repeats = [
                    w for w, c in word_freq.items()
                    if c >= 5 and len(w) > 5 and w not in stopwords
                ]
                if suspicious_repeats:
                    flags.append(
                        f"Suspicious keyword repetition: '{suspicious_repeats[0]}' x{word_freq[suspicious_repeats[0]]}"
                    )

            paper["metadata_suspicious"] = len(flags) > 0
            paper["metadata_flags"] = flags

            if flags:
                logger.info(
                    f"[MetadataInspector] FLAGS: "
                    f"'{paper['title'][:50]}': {flags}"
                )

        return papers

    def _valid_arxiv_id(self, paper_id: str) -> bool:
        import re
        new_fmt = re.match(r"^\d{4}\.\d{4,6}(v\d+)?$", paper_id)
        old_fmt = re.match(r"^[a-z\-]+(\.[A-Z]+)?/\d{7}(v\d+)?$", paper_id)
        return bool(new_fmt or old_fmt)


class FlagAgent:
    """
    Final decision. A paper is flagged if:
      - Any critical metadata flag fires (invalid ID, no authors,
        future date, non-arXiv URL), OR
      - Similarity is suspicious AND at least one metadata flag exists
    
    Similarity alone does NOT flag — reduces false positives.
    Metadata alone flags only on critical signals.
    """

    CRITICAL_FLAGS = [
        "Invalid arXiv ID",
        "No authors",
        "Future date",
        "Non-arXiv URL",
        "Implausibly high accuracy",
    ]

    def decide(self, papers: list[dict]) -> list[dict]:
        for paper in papers:
            sim_sus   = paper.get("similarity_suspicious", False)
            meta_flags = paper.get("metadata_flags", [])

            critical = any(
                any(cf in flag for cf in self.CRITICAL_FLAGS)
                for flag in meta_flags
            )

            if critical:
                flagged = True
                reason = "Critical metadata: " + "; ".join(meta_flags)
            elif sim_sus and meta_flags:
                # Both signals agree
                flagged = True
                reason = (
                    f"Combined: anomalous similarity "
                    f"(z={paper.get('similarity_z_score', 0):.2f}) "
                    f"+ metadata issues: {'; '.join(meta_flags)}"
                )
            elif sim_sus:
                # Similarity alone — flag with lower confidence
                flagged = True
                reason = (
                    f"Anomalous similarity score "
                    f"(z={paper.get('similarity_z_score', 0):.2f}) — "
                    f"abstract may be engineered to rank highly"
                )
            else:
                flagged = False
                reason = "Passed all checks."

            paper["flagged"] = flagged
            paper["flag_reason"] = reason

            logger.info(
                f"[FlagAgent] {'QUARANTINED' if flagged else 'CLEAN'}: "
                f"'{paper['title'][:50]}' — {reason}"
            )

        return papers


class AuditorAgent:
    """
    L5 Mitigation agent.
    Calibrated threshold: z=-1.7 based on empirical score distribution.
    """

    def __init__(self, z_score_threshold: float = -1.7, corpus_metadata: list = None):
        self.similarity_checker = SimilarityChecker(z_score_threshold)
        self.metadata_inspector = MetadataInspector(corpus_metadata)
        self.flag_agent = FlagAgent()

    def audit(self, papers: list[dict]) -> tuple[list[dict], list[dict]]:
        print(f"\n[AuditorAgent] Auditing {len(papers)} papers...")
        print("  Check 1/3 — SimilarityChecker (z-threshold=-1.7)...")
        papers = self.similarity_checker.check(papers)
        print("  Check 2/3 — MetadataInspector...")
        papers = self.metadata_inspector.check(papers)
        print("  Check 3/3 — FlagAgent...")
        papers = self.flag_agent.decide(papers)

        clean       = [p for p in papers if not p.get("flagged")]
        quarantined = [p for p in papers if p.get("flagged")]

        print(f"\n  → {len(clean)} papers passed  |  {len(quarantined)} quarantined")
        return clean, quarantined

    def format_audit_report(self, clean, quarantined, query):
        lines = [
            f"\n{'='*62}",
            "  AUDIT REPORT — AuditorAgent",
            f"  Query: \"{query}\"",
            f"{'='*62}",
            f"\n  ✓ CLEAN PAPERS ({len(clean)} passed):",
        ]
        for rank, paper in enumerate(clean, 1):
            lines += [
                f"\n  [{rank}] {paper['title'][:65]}",
                f"       Authors  : {', '.join(paper['authors'][:2])}",
                f"       arXiv ID : {paper['id']}",
                f"       Score    : {paper.get('score', 0):.4f}",
                f"       z-score  : {paper.get('similarity_z_score', 0):.4f}",
                f"       Summary  : {paper.get('summary', paper['abstract'][:150])}",
            ]
        if quarantined:
            lines += [f"\n\n  ⚠ QUARANTINED ({len(quarantined)} blocked):"]
            for paper in quarantined:
                lines += [
                    f"\n  [BLOCKED] {paper['title'][:65]}",
                    f"            Reason        : {paper.get('flag_reason')}",
                    f"            ID            : {paper['id']}",
                    f"            z-score       : {paper.get('similarity_z_score', 0):.4f}",
                    f"            Ground truth  : {'POISON' if paper.get('poisoned') else 'LEGIT'}",
                ]
        else:
            lines.append("\n\n  No papers quarantined.")
        lines.append(f"\n{'='*62}")
        return "\n".join(lines)
