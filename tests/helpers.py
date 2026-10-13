from datetime import datetime, timezone


def make_paper(i: int, score: float = 1.0, **overrides) -> dict:
    abstract = " ".join(f"term{i}x{j}" for j in range(80))  # 80 distinct words
    paper = {
        "id": f"2401.{10000 + i}",
        "title": f"Legitimate paper number {i}",
        "authors": ["A. Author", "B. Author"],
        "abstract": abstract,
        "published": f"{datetime.now(timezone.utc).year}-01-01",
        "url": f"https://arxiv.org/abs/2401.{10000 + i}",
        "score": score,
    }
    paper.update(overrides)
    return paper


def make_result_set(poison_score: float = 0.4) -> list[dict]:
    """19 legit papers with similar scores plus 1 poisoned outlier."""
    papers = [make_paper(i, score=1.0 + 0.01 * i) for i in range(19)]
    papers.append(make_paper(99, score=poison_score, poisoned=True))
    return papers
