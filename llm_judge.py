"""
Stage 3: LLM Holistic Judgment (precision stage).

This is the heart of "ranks candidates the way a great recruiter would."
Embeddings and skills-overlap can find plausible candidates, but they can't
read a career history and notice that someone's growth trajectory stalled, or
that a "Senior" title at a small startup with 8 months tenure means something
different than the same title at a stable company for 4 years, or that a
candidate's leadership note actually answers the JD's "mentors others" need
even though it never says the word "mentor."

We batch candidates (after the semantic + structured pre-filter narrows the
field) and ask Claude to evaluate each one against the full JD context,
returning a score plus a short rationale, strengths, and concerns — so the
output is explainable, not a black-box number.
"""

import json

from src.models import Candidate, JobRequirements
from src.utils.gemini_client import ask_gemini_json

SYSTEM_PROMPT = """You are a senior technical recruiter known for finding the right person, \
not just the person with the most matching keywords. You read between the lines of a resume: \
you can tell the difference between someone who owned a hard problem and someone who was \
adjacent to one, you notice when someone's trajectory shows real growth versus lateral moves \
with inflated titles, and you weigh context (company stage, team size, tenure) rather than \
treating every line item the same.

You are evaluating a pre-filtered shortlist of candidates against one job description. For \
EACH candidate, give a fit score from 0-100 and a short, specific rationale grounded in their \
actual profile — not generic praise. Be honest about weaknesses; a trustworthy shortlist names \
real concerns, it doesn't just compliment everyone.

Respond with valid JSON only — no markdown fences, no commentary before or after."""

USER_PROMPT_TEMPLATE = """JOB CONTEXT:
Role: {role_title} ({seniority_level})
Required experience: {min_years}+ years
Must-have skills: {must_haves}
Nice-to-have skills: {nice_to_haves}
Core responsibilities: {responsibilities}
Desired behavioral signals: {soft_skills}
Preferred domain background: {domains}
Caution flags for this role: {red_flags}

Original JD excerpt (for tone/context):
{jd_excerpt}

---

CANDIDATES TO EVALUATE:

{candidates_block}

---

For EACH candidate above, return an entry in a JSON array with this exact shape:

{{
  "candidate_id": "...",
  "fit_score": <integer 0-100>,
  "rationale": "<2-3 sentences, specific to this candidate's actual background, explaining the score>",
  "strengths": ["<specific strength 1>", "<specific strength 2>", "..."],
  "concerns": ["<specific, honest concern, or empty list if genuinely none>"]
}}

Scoring guide:
- 85-100: Excellent fit. Meets must-haves, experience aligns well, strong evidence of the soft \
skills/behaviors this role needs, no significant red flags.
- 65-84: Strong fit with minor gaps — e.g. missing one nice-to-have, slightly under/over the \
experience band, or thin evidence (not absence) of a soft skill.
- 40-64: Partial fit — meets some must-haves but has a real gap (skills, seniority, or domain) \
that would need to be addressed in an interview.
- Below 40: Weak fit — significant mismatch in core requirements.

Judge holistically: a candidate with slightly less experience but a clearly demonstrated track \
record of the exact responsibilities this role needs should often outscore someone with more \
years but only tangential experience. Use the career history, leadership notes, and platform \
activity together — don't just match the skills list.

Return ONLY the JSON array, one entry per candidate, in the same order they were given."""


def _format_candidates_block(candidates: list[Candidate]) -> str:
    blocks = []
    for c in candidates:
        blocks.append(f"--- {c.candidate_id} ---\n{c.to_text_block()}")
    return "\n".join(blocks)


def evaluate_candidates_batch(
    candidates: list[Candidate],
    job_req: JobRequirements,
    batch_size: int = 8,
) -> dict[str, dict]:
    """Evaluates candidates in batches (to keep prompts manageable and reliable)
    and returns candidate_id -> {fit_score, rationale, strengths, concerns}."""
    results = {}

    jd_excerpt = job_req.raw_jd_text[:1200]

    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]
        prompt = USER_PROMPT_TEMPLATE.format(
            role_title=job_req.role_title,
            seniority_level=job_req.seniority_level,
            min_years=job_req.min_years_experience,
            must_haves=", ".join(job_req.must_have_skills) or "None specified",
            nice_to_haves=", ".join(job_req.nice_to_have_skills) or "None specified",
            responsibilities="; ".join(job_req.core_responsibilities) or "Not specified",
            soft_skills=", ".join(job_req.soft_skill_signals) or "None specified",
            domains=", ".join(job_req.domain_preferences) or "None specified",
            red_flags=", ".join(job_req.red_flag_filters) or "None specified",
            jd_excerpt=jd_excerpt,
            candidates_block=_format_candidates_block(batch),
        )

        print(f"  Evaluating batch {i // batch_size + 1} "
              f"({len(batch)} candidates: {', '.join(c.candidate_id for c in batch)})...")

        try:
            response = ask_gemini_json(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt,
                temperature=0.0,
            )
        except RuntimeError as e:
            print(f"  [WARNING] Batch failed after retries, assigning neutral fallback scores: {e}")
            for c in batch:
                results[c.candidate_id] = {
                    "fit_score": 50,
                    "rationale": "LLM evaluation failed for this candidate; neutral fallback score assigned.",
                    "strengths": [],
                    "concerns": ["LLM evaluation unavailable — review manually."],
                }
            continue

        if isinstance(response, dict) and "candidates" in response:
            response = response["candidates"]  # tolerate Claude wrapping the array

        for entry in response:
            cid = entry.get("candidate_id")
            if cid:
                results[cid] = {
                    "fit_score": entry.get("fit_score", 50),
                    "rationale": entry.get("rationale", ""),
                    "strengths": entry.get("strengths", []),
                    "concerns": entry.get("concerns", []),
                }

        # safety net: if Claude skipped someone in the batch, fill a fallback
        for c in batch:
            if c.candidate_id not in results:
                results[c.candidate_id] = {
                    "fit_score": 50,
                    "rationale": "Not returned by LLM batch evaluation; neutral fallback score assigned.",
                    "strengths": [],
                    "concerns": ["Missing from LLM response — review manually."],
                }

    return results
