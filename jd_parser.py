"""
Stage 1: Job Description Understanding.

Instead of keyword-extracting a JD, we ask Claude to *read it like a recruiter
would* — distinguishing must-haves from nice-to-haves, inferring the seniority
bar from tone and responsibilities (not just an explicit "X years" line), and
picking up soft signals like "comfortable with ambiguity" or "mentors others"
that a keyword filter would never capture.
"""

from src.models import JobRequirements
from src.utils.gemini_client import ask_gemini_json
SYSTEM_PROMPT = """You are an expert technical recruiter with 15 years of experience hiring \
for engineering, data, product, and design roles. You read job descriptions the way a \
seasoned recruiter does: you separate what's truly required from what's aspirational \
wishlist language, you infer the real seniority bar from the responsibilities described \
(not just an explicit years-of-experience line), and you pick up on subtle signals about \
team culture, working style, and what would make someone a strong vs. weak fit beyond \
their resume keywords.

You always respond with valid JSON only — no markdown fences, no commentary."""

USER_PROMPT_TEMPLATE = """Read the following job description carefully and extract a structured \
understanding of the role, the way you would before starting a search.

JOB DESCRIPTION:
---
{jd_text}
---

Return a JSON object with exactly these fields:

{{
  "role_title": "concise normalized title, e.g. 'Senior Backend Engineer'",
  "seniority_level": "one of: Entry, Mid, Senior, Staff/Principal, Lead/Manager",
  "min_years_experience": <number, your best estimate even if not explicitly stated>,
  "max_years_experience": <number or null>,
  "must_have_skills": ["skills/technologies that are truly non-negotiable, inferred from context not just listed bullet points"],
  "nice_to_have_skills": ["skills explicitly framed as bonus, preferred, or 'nice to have'"],
  "core_responsibilities": ["3-6 short phrases capturing what this person will actually spend their time doing"],
  "soft_skill_signals": ["behavioral/soft traits implied by the JD's language, e.g. 'mentorship', 'comfort with ambiguity', 'cross-functional collaboration', 'ownership under pressure' — infer these from phrasing, don't just quote the JD"],
  "domain_preferences": ["specific industry/domain experience that would be a strong plus, e.g. 'fintech', 'payments', 'healthcare' — empty list if none implied"],
  "red_flag_filters": ["traits this role's context suggests recruiters should be cautious about, e.g. 'frequent job hopping' if the role needs continuity, 'pure IC with no leadership' if the role explicitly needs mentorship — be conservative, only include if reasonably implied"]
}}

Be a thoughtful recruiter, not a keyword extractor: if the JD says "5+ years" but the \
responsibilities described clearly need staff-level systems thinking, reflect that tension. \
If skills are listed but the prose says "or similar" or "comfortable with at least one of", \
treat that flexibility correctly in must_have_skills (don't force a single specific tool as \
mandatory when the JD itself says it's flexible)."""


def parse_job_description(jd_text: str) -> JobRequirements:
    result = ask_gemini_json(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=USER_PROMPT_TEMPLATE.format(jd_text=jd_text),
        temperature=0.0,
    )

    return JobRequirements(
        role_title=result.get("role_title", "Unknown Role"),
        seniority_level=result.get("seniority_level", "Mid"),
        min_years_experience=float(result.get("min_years_experience") or 0),
        max_years_experience=(
            float(result["max_years_experience"])
            if result.get("max_years_experience") is not None else None
        ),
        must_have_skills=result.get("must_have_skills", []),
        nice_to_have_skills=result.get("nice_to_have_skills", []),
        core_responsibilities=result.get("core_responsibilities", []),
        soft_skill_signals=result.get("soft_skill_signals", []),
        domain_preferences=result.get("domain_preferences", []),
        red_flag_filters=result.get("red_flag_filters", []),
        raw_jd_text=jd_text,
    )
