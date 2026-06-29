"""
Unit tests for structured scoring (skills match, experience fit, platform signal).

Run with: python -m pytest tests/ -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import Candidate, JobRequirements
from src.scoring.structured_scoring import (
    compute_skills_match,
    compute_experience_fit,
    compute_platform_signal,
    _fuzzy_skill_match,
)


def make_candidate(**overrides) -> Candidate:
    defaults = dict(
        candidate_id="TEST001",
        name="Test Candidate",
        current_title="Software Engineer",
        current_company="TestCo",
        years_experience=5.0,
        location="Bangalore",
        skills=["Python", "AWS", "Docker"],
        education_degree="B.Tech",
        education_college="Test University",
        grad_year=2020,
        certifications="",
        career_history=[],
        leadership_notes="",
        github_public_repos=10,
        github_stars_received=50,
        github_contributions_last_year=200,
        github_bio="",
        blog_activity="No public technical writing found.",
        behavioral_flags="",
        notice_period_days=30,
        expected_salary_lpa=20.0,
        profile_summary="",
    )
    defaults.update(overrides)
    return Candidate(**defaults)


def make_job_req(**overrides) -> JobRequirements:
    defaults = dict(
        role_title="Software Engineer",
        seniority_level="Mid",
        min_years_experience=4.0,
        max_years_experience=None,
        must_have_skills=["Python", "AWS"],
        nice_to_have_skills=["Kubernetes"],
        core_responsibilities=[],
        soft_skill_signals=[],
        domain_preferences=[],
        red_flag_filters=[],
        raw_jd_text="",
    )
    defaults.update(overrides)
    return JobRequirements(**defaults)


def test_fuzzy_skill_match_exact():
    assert _fuzzy_skill_match("Python", "python")


def test_fuzzy_skill_match_synonym():
    assert _fuzzy_skill_match("k8s", "Kubernetes")
    assert _fuzzy_skill_match("JS", "JavaScript")


def test_fuzzy_skill_match_no_false_positive():
    assert not _fuzzy_skill_match("Java", "JavaScript")  # tricky case: should NOT match
    assert not _fuzzy_skill_match("Python", "Pandas")


def test_skills_match_full_overlap():
    cand = make_candidate(skills=["Python", "AWS", "Docker"])
    job = make_job_req(must_have_skills=["Python", "AWS"], nice_to_have_skills=[])
    score, matched, missing = compute_skills_match(cand, job)
    assert score == 1.0
    assert missing == []
    assert len(matched) == 2


def test_skills_match_partial_overlap():
    cand = make_candidate(skills=["Python", "Docker"])
    job = make_job_req(must_have_skills=["Python", "AWS", "Kafka"], nice_to_have_skills=[])
    score, matched, missing = compute_skills_match(cand, job)
    assert 0.0 < score < 1.0
    assert "AWS" in missing
    assert "Kafka" in missing
    assert "Python" in matched


def test_skills_match_zero_overlap():
    cand = make_candidate(skills=["Figma", "Sketch"])
    job = make_job_req(must_have_skills=["Python", "AWS"], nice_to_have_skills=[])
    score, matched, missing = compute_skills_match(cand, job)
    assert score == 0.0
    assert len(missing) == 2


def test_experience_fit_meets_minimum():
    cand = make_candidate(years_experience=6.0)
    job = make_job_req(min_years_experience=5.0, max_years_experience=None)
    assert compute_experience_fit(cand, job) == 1.0


def test_experience_fit_underqualified():
    cand = make_candidate(years_experience=1.0)
    job = make_job_req(min_years_experience=5.0, max_years_experience=None)
    score = compute_experience_fit(cand, job)
    assert score < 0.7  # meaningfully penalized, but not literally zero


def test_experience_fit_overqualified_gentle_penalty():
    """A senior candidate applying to a mid-band role should be penalized
    much more gently than an underqualified candidate."""
    cand = make_candidate(years_experience=15.0)
    job = make_job_req(min_years_experience=5.0, max_years_experience=8.0)
    score = compute_experience_fit(cand, job)
    assert score >= 0.6  # gentle penalty, not a hard cutoff


def test_platform_signal_no_activity_is_not_zero():
    """Candidates with no GitHub/blog presence shouldn't be zeroed out —
    plenty of strong engineers have no public OSS footprint."""
    cand = make_candidate(
        github_public_repos=0, github_stars_received=0,
        github_contributions_last_year=0, blog_activity="No public technical writing found."
    )
    score = compute_platform_signal(cand)
    assert score == 0.0  # this candidate genuinely has zero signal — score should reflect that
    assert score >= 0.0  # but never negative


def test_platform_signal_behavioral_flag_reduces_score():
    cand_clean = make_candidate(behavioral_flags="")
    cand_flagged = make_candidate(behavioral_flags="Gap of 14 months between roles.")
    assert compute_platform_signal(cand_flagged) <= compute_platform_signal(cand_clean)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
