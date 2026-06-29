"""
Deterministic structured scoring: skills overlap and experience fit.

These scores are intentionally NOT keyword-exact-match — we use fuzzy string
matching so "JS" matches "JavaScript", "k8s" matches "Kubernetes", etc., which
is the single biggest practical gap between naive keyword filters and how a
recruiter actually reads a resume.
"""

from difflib import SequenceMatcher

from src.models import Candidate, JobRequirements

# Common synonym/abbreviation pairs seen in resumes and JDs.
# This is a small seed list — extend freely as you see real-world mismatches.
SKILL_SYNONYMS = {
    "js": "javascript",
    "ts": "typescript",
    "k8s": "kubernetes",
    "py": "python",
    "ml": "machine learning",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "ci/cd": "continuous integration",
    "gcp": "google cloud platform",
    "aws": "amazon web services",
    "sql": "structured query language",
    "oop": "object oriented programming",
    "rest apis": "rest",
    "node.js": "nodejs",
}


def _normalize_skill(skill: str) -> str:
    s = skill.lower().strip()
    return SKILL_SYNONYMS.get(s, s)


# Known pairs where one string is a substring of the other but they are
# DIFFERENT skills — substring matching must not treat these as equal.
SUBSTRING_FALSE_POSITIVES = {
    ("java", "javascript"),
    ("python", "pandas"),
    ("c", "c++"),
    ("c", "c#"),
    ("go", "django"),
    ("r", "react"),
}


def _is_known_false_positive(a: str, b: str) -> bool:
    pair = (a, b) if a < b else (b, a)
    return pair in SUBSTRING_FALSE_POSITIVES


def _fuzzy_skill_match(skill_a: str, skill_b: str, threshold: float = 0.82) -> bool:
    a, b = _normalize_skill(skill_a), _normalize_skill(skill_b)
    if a == b:
        return True
    if _is_known_false_positive(a, b):
        return False
    # Substring match, but only when it's a real prefix/suffix relationship
    # with a trailing/leading separator (e.g. "react" in "react.js", "node" in
    # "node.js") — not an arbitrary substring like "java" inside "javascript".
    for shorter, longer in ((a, b), (b, a)):
        if shorter and shorter in longer:
            idx = longer.find(shorter)
            before = longer[idx - 1] if idx > 0 else ""
            after = longer[idx + len(shorter)] if idx + len(shorter) < len(longer) else ""
            boundary_chars = ("", ".", " ", "-", "_", "/")
            if before in boundary_chars and after in boundary_chars:
                return True
    return SequenceMatcher(None, a, b).ratio() >= threshold


def compute_skills_match(candidate: Candidate, job_req: JobRequirements) -> tuple[float, list[str], list[str]]:
    """Returns (score in [0,1], matched_must_haves, missing_must_haves)."""
    candidate_skills = candidate.skills

    def matched_against(required_list):
        matched, missing = [], []
        for req_skill in required_list:
            hit = any(_fuzzy_skill_match(req_skill, cand_skill) for cand_skill in candidate_skills)
            (matched if hit else missing).append(req_skill)
        return matched, missing

    must_matched, must_missing = matched_against(job_req.must_have_skills)
    nice_matched, _ = matched_against(job_req.nice_to_have_skills)

    must_have_ratio = len(must_matched) / len(job_req.must_have_skills) if job_req.must_have_skills else 1.0

    if not job_req.nice_to_have_skills:
        # No nice-to-haves to evaluate — score purely on must-haves, don't
        # artificially cap a perfect must-have match below 1.0.
        return min(1.0, must_have_ratio), must_matched, must_missing

    nice_have_ratio = len(nice_matched) / len(job_req.nice_to_have_skills)
    # must-haves dominate the score; nice-to-haves add a smaller bonus
    score = 0.8 * must_have_ratio + 0.2 * nice_have_ratio
    return min(1.0, score), must_matched, must_missing


def compute_experience_fit(candidate: Candidate, job_req: JobRequirements) -> float:
    """Score how well years_experience aligns with the JD's expected range.
    Penalizes under-qualification more than slight over-qualification, mirroring
    how recruiters actually think (a 12 YOE candidate for a "5+ years" role isn't
    a bad fit the way a 2 YOE candidate is)."""
    years = candidate.years_experience
    min_req = job_req.min_years_experience
    max_req = job_req.max_years_experience

    if years >= min_req:
        if max_req is not None and years > max_req:
            overflow = years - max_req
            # gentle penalty for being notably more senior than the band
            return max(0.6, 1.0 - 0.04 * overflow)
        return 1.0

    # under-qualified: steeper penalty, scaled by how far short they are
    shortfall = min_req - years
    if min_req <= 0:
        return 1.0
    penalty_ratio = shortfall / max(min_req, 1.0)
    return max(0.0, 1.0 - 1.3 * penalty_ratio)


def compute_platform_signal(candidate: Candidate) -> float:
    """Normalize GitHub/blog/behavioral signals into a single [0,1] score.
    This deliberately does NOT penalize candidates with low GitHub activity to
    zero — plenty of excellent engineers have no public OSS presence. It's a
    gentle positive signal, not a gate."""
    import math

    repo_score = min(1.0, math.log1p(candidate.github_public_repos) / math.log1p(40))
    stars_score = min(1.0, math.log1p(candidate.github_stars_received) / math.log1p(300))
    contrib_score = min(1.0, math.log1p(candidate.github_contributions_last_year) / math.log1p(600))
    blog_score = 1.0 if ("no public technical writing" not in candidate.blog_activity.lower()) else 0.0

    base = 0.35 * repo_score + 0.25 * stars_score + 0.25 * contrib_score + 0.15 * blog_score

    # behavioral flags pull the score down, but don't zero it out alone —
    # the LLM stage weighs in on how serious the flag actually is
    if candidate.behavioral_flags:
        base *= 0.7

    return round(min(1.0, base), 3)
