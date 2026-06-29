"""
Stage 4: Composite Scoring & Ranking.

Combines semantic similarity, structured skills/experience fit, LLM holistic
judgment, and platform signal into one weighted score per candidate. Weights
live in src/config.py (SCORE_WEIGHTS) so they're easy to tune per-role without
touching logic.
"""

from src import config
from src.models import Candidate, CandidateScore, JobRequirements
from src.scoring.structured_scoring import (
    compute_skills_match,
    compute_experience_fit,
    compute_platform_signal,
)


def build_composite_scores(
    candidates: list[Candidate],
    job_req: JobRequirements,
    semantic_scores: dict[str, float],
    llm_results: dict[str, dict],
    weights: dict[str, float] = None,
) -> list[CandidateScore]:
    weights = weights or config.SCORE_WEIGHTS
    scored = []

    for c in candidates:
        skills_score, matched, missing = compute_skills_match(c, job_req)
        experience_score = compute_experience_fit(c, job_req)
        platform_score = compute_platform_signal(c)
        semantic_score = semantic_scores.get(c.candidate_id, 0.5)

        llm_entry = llm_results.get(c.candidate_id, {})
        llm_score_raw = llm_entry.get("fit_score", 50)
        llm_score = max(0.0, min(1.0, llm_score_raw / 100.0))

        composite = (
            weights["semantic_fit"] * semantic_score
            + weights["skills_match"] * skills_score
            + weights["experience_fit"] * experience_score
            + weights["llm_judgment"] * llm_score
            + weights["platform_signal"] * platform_score
        )

        rationale = llm_entry.get("rationale", "")
        if missing:
            rationale += f" [Missing must-have skills: {', '.join(missing)}]"

        scored.append(CandidateScore(
            candidate_id=c.candidate_id,
            name=c.name,
            semantic_fit=round(semantic_score, 3),
            skills_match=round(skills_score, 3),
            experience_fit=round(experience_score, 3),
            llm_judgment=round(llm_score, 3),
            platform_signal=round(platform_score, 3),
            composite_score=round(composite, 4),
            llm_rationale=rationale.strip(),
            llm_strengths=llm_entry.get("strengths", []),
            llm_concerns=llm_entry.get("concerns", []),
        ))

    scored.sort(key=lambda s: s.composite_score, reverse=True)
    for rank, s in enumerate(scored, start=1):
        s.rank = rank

    return scored
