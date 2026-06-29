"""
Loads the candidate dataset (CSV) into Candidate model objects.

Designed to be tolerant of the real dataset having slightly different column
names — see COLUMN_ALIASES below. If the real dataset's schema differs more
than that, adjust the mapping here; nothing else in the pipeline needs to change.
"""

import csv
import json
from pathlib import Path

from src.models import Candidate

# Map our canonical field name -> list of acceptable source column names.
# Add aliases here when plugging in the real dataset.
COLUMN_ALIASES = {
    "candidate_id": ["candidate_id", "id", "candidate id", "applicant_id"],
    "name": ["name", "full_name", "candidate_name"],
    "current_title": ["current_title", "title", "current_role", "designation"],
    "current_company": ["current_company", "company", "employer"],
    "years_experience": ["years_experience", "total_experience", "experience_years", "yoe"],
    "location": ["location", "city"],
    "skills": ["skills", "skillset", "key_skills"],
    "education_degree": ["education_degree", "degree", "qualification"],
    "education_college": ["education_college", "college", "university", "institute"],
    "grad_year": ["grad_year", "graduation_year"],
    "certifications": ["certifications", "certs"],
    "career_history": ["career_history", "experience_history", "work_history"],
    "leadership_notes": ["leadership_notes", "leadership", "management_experience"],
    "github_public_repos": ["github_public_repos", "public_repos", "repos"],
    "github_stars_received": ["github_stars_received", "stars", "github_stars"],
    "github_contributions_last_year": ["github_contributions_last_year", "contributions_last_year"],
    "github_bio": ["github_bio", "github_summary"],
    "blog_activity": ["blog_activity", "writing_activity", "publications"],
    "behavioral_flags": ["behavioral_flags", "flags", "notes"],
    "notice_period_days": ["notice_period_days", "notice_period"],
    "expected_salary_lpa": ["expected_salary_lpa", "expected_ctc", "expected_salary"],
    "profile_summary": ["profile_summary", "summary", "bio"],
}


def _resolve_columns(fieldnames: list[str]) -> dict[str, str]:
    """Build canonical_field -> actual_csv_column mapping."""
    lower_map = {f.lower().strip(): f for f in fieldnames}
    resolved = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lower_map:
                resolved[canonical] = lower_map[alias]
                break
    return resolved


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _safe_int(value, default=0):
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


def _parse_skills(raw: str) -> list[str]:
    if not raw:
        return []
    # support "; " separated or ", " separated or JSON list
    raw = raw.strip()
    if raw.startswith("["):
        try:
            return [s.strip() for s in json.loads(raw)]
        except json.JSONDecodeError:
            pass
    sep = ";" if ";" in raw else ","
    return [s.strip() for s in raw.split(sep) if s.strip()]


def _parse_career_history(raw: str) -> list[dict]:
    if not raw:
        return []
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []


def load_candidates(csv_path: str | Path) -> list[Candidate]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Candidate dataset not found at {csv_path}")

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        col_map = _resolve_columns(reader.fieldnames or [])

        missing_required = [
            c for c in ["candidate_id", "name", "skills", "years_experience"]
            if c not in col_map
        ]
        if missing_required:
            raise ValueError(
                f"Dataset is missing required columns (after alias resolution): {missing_required}. "
                f"Found columns: {reader.fieldnames}. "
                "Update COLUMN_ALIASES in src/ingestion/data_loader.py to map them."
            )

        candidates = []
        for row in reader:
            def get(field, default=""):
                col = col_map.get(field)
                return row.get(col, default) if col else default

            candidates.append(Candidate(
                candidate_id=get("candidate_id"),
                name=get("name"),
                current_title=get("current_title"),
                current_company=get("current_company"),
                years_experience=_safe_float(get("years_experience")),
                location=get("location"),
                skills=_parse_skills(get("skills")),
                education_degree=get("education_degree"),
                education_college=get("education_college"),
                grad_year=_safe_int(get("grad_year")),
                certifications=get("certifications"),
                career_history=_parse_career_history(get("career_history")),
                leadership_notes=get("leadership_notes"),
                github_public_repos=_safe_int(get("github_public_repos")),
                github_stars_received=_safe_int(get("github_stars_received")),
                github_contributions_last_year=_safe_int(get("github_contributions_last_year")),
                github_bio=get("github_bio"),
                blog_activity=get("blog_activity"),
                behavioral_flags=get("behavioral_flags"),
                notice_period_days=_safe_int(get("notice_period_days")),
                expected_salary_lpa=_safe_float(get("expected_salary_lpa")),
                profile_summary=get("profile_summary"),
                raw_row=row,
            ))
    return candidates


def load_job_description(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Job description not found at {path}")
    return path.read_text(encoding="utf-8")
