"""
Writes the final ranked shortlist to CSV — the deliverable format requested
("a ranked output file of your recommended candidates").
"""

import csv
from pathlib import Path

from src.models import Candidate, CandidateScore


def write_ranked_csv(
    scores: list[CandidateScore],
    candidates_by_id: dict[str, Candidate],
    output_path: str | Path,
    top_n: int | None = None,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_to_write = scores[:top_n] if top_n else scores

    fieldnames = [
        "rank",
        "candidate_id",
        "name",
        "current_title",
        "current_company",
        "years_experience",
        "location",
        "composite_score",
        "semantic_fit",
        "skills_match",
        "experience_fit",
        "llm_judgment",
        "platform_signal",
        "strengths",
        "concerns",
        "rationale",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in rows_to_write:
            cand = candidates_by_id.get(s.candidate_id)
            writer.writerow({
                "rank": s.rank,
                "candidate_id": s.candidate_id,
                "name": s.name,
                "current_title": cand.current_title if cand else "",
                "current_company": cand.current_company if cand else "",
                "years_experience": cand.years_experience if cand else "",
                "location": cand.location if cand else "",
                "composite_score": round(s.composite_score * 100, 1),
                "semantic_fit": round(s.semantic_fit * 100, 1),
                "skills_match": round(s.skills_match * 100, 1),
                "experience_fit": round(s.experience_fit * 100, 1),
                "llm_judgment": round(s.llm_judgment * 100, 1),
                "platform_signal": round(s.platform_signal * 100, 1),
                "strengths": " | ".join(s.llm_strengths),
                "concerns": " | ".join(s.llm_concerns),
                "rationale": s.llm_rationale,
            })

    print(f"Wrote {len(rows_to_write)} ranked candidates -> {output_path}")
