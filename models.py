"""
Shared data models. Using plain dataclasses (not pydantic) to keep the
dependency footprint small and the code readable for reviewers.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Candidate:
    candidate_id: str
    name: str
    current_title: str
    current_company: str
    years_experience: float
    location: str
    skills: list[str]
    education_degree: str
    education_college: str
    grad_year: int
    certifications: str
    career_history: list[dict]
    leadership_notes: str
    github_public_repos: int
    github_stars_received: int
    github_contributions_last_year: int
    github_bio: str
    blog_activity: str
    behavioral_flags: str
    notice_period_days: int
    expected_salary_lpa: float
    profile_summary: str
    raw_row: dict = field(default_factory=dict)

    def to_text_block(self) -> str:
        """Flattened text representation used for embeddings and LLM prompts."""
        history_lines = []
        for job in self.career_history:
            end = job.get("end_year", "Present")
            history_lines.append(
                f"  - {job.get('title')} at {job.get('company')} "
                f"({job.get('start_year')}–{end}, {job.get('duration_years')} yrs)"
            )
        history_str = "\n".join(history_lines) if history_lines else "  (no structured history)"

        return f"""Candidate: {self.name} ({self.candidate_id})
Current Role: {self.current_title} at {self.current_company}
Total Experience: {self.years_experience} years | Location: {self.location}
Skills: {', '.join(self.skills)}
Education: {self.education_degree}, {self.education_college} (Class of {self.grad_year})
Certifications: {self.certifications or 'None listed'}
Career History:
{history_str}
Leadership / Scope: {self.leadership_notes}
GitHub Activity: {self.github_public_repos} public repos, {self.github_stars_received} stars received, \
{self.github_contributions_last_year} contributions in the last year. Bio: {self.github_bio}
Technical Writing / Blog: {self.blog_activity}
Behavioral Flags: {self.behavioral_flags or 'None noted'}
Notice Period: {self.notice_period_days} days | Expected CTC: {self.expected_salary_lpa} LPA
Summary: {self.profile_summary}
"""


@dataclass
class JobRequirements:
    """Structured extraction of a job description, produced by the LLM parser."""
    role_title: str
    seniority_level: str
    min_years_experience: float
    max_years_experience: Optional[float]
    must_have_skills: list[str]
    nice_to_have_skills: list[str]
    core_responsibilities: list[str]
    soft_skill_signals: list[str]      # e.g. "mentorship", "ambiguity tolerance"
    domain_preferences: list[str]      # e.g. "fintech", "payments"
    red_flag_filters: list[str]        # e.g. "frequent job hopping" if JD implies stability need
    raw_jd_text: str


@dataclass
class CandidateScore:
    candidate_id: str
    name: str
    semantic_fit: float
    skills_match: float
    experience_fit: float
    llm_judgment: float
    platform_signal: float
    composite_score: float
    llm_rationale: str
    llm_strengths: list[str]
    llm_concerns: list[str]
    rank: Optional[int] = None
