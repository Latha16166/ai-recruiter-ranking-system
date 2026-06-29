"""
Main entry point. Orchestrates the full pipeline:

  1. Load job description + candidate dataset
  2. Parse JD into structured requirements (LLM)
  3. Semantic retrieval: embed candidates + JD, score similarity (recall stage)
  4. Pre-filter to top-K candidates by a blend of semantic + structured score
     (keeps LLM calls bounded and cheap, without dropping anyone plausible)
  5. LLM holistic judgment on the pre-filtered set (precision stage)
  6. Composite scoring + final ranking
  7. Write ranked CSV + print top shortlist

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python -m src.main --jd data/job_description.txt --candidates data/candidates.csv
"""

import argparse
import sys
import time

from src import config
from src.ingestion.data_loader import load_candidates, load_job_description
from src.ingestion.jd_parser import parse_job_description
from src.retrieval.semantic_search import compute_semantic_scores
from src.scoring.structured_scoring import compute_skills_match, compute_experience_fit
from src.scoring.llm_judge import evaluate_candidates_batch
from src.scoring.composite_ranker import build_composite_scores
from src.utils.output_writer import write_ranked_csv


def prefilter_candidates(candidates, job_req, semantic_scores, top_k):
    """Blend semantic score with a cheap structured pre-score to pick the
    top_k candidates that go on to the (more expensive) LLM judgment stage.
    This keeps cost/latency bounded on large datasets while ensuring we don't
    drop someone with a strong structured match but middling embedding score."""
    prescored = []
    for c in candidates:
        skills_score, _, _ = compute_skills_match(c, job_req)
        exp_score = compute_experience_fit(c, job_req)
        sem_score = semantic_scores.get(c.candidate_id, 0.5)
        blended = 0.4 * sem_score + 0.35 * skills_score + 0.25 * exp_score
        prescored.append((c, blended))

    prescored.sort(key=lambda pair: pair[1], reverse=True)
    return [c for c, _ in prescored[:top_k]]


def run_pipeline(jd_path: str, candidates_path: str, output_path: str, top_n: int):
    t0 = time.time()

    print(f"[1/6] Loading job description from {jd_path} ...")
    jd_text = load_job_description(jd_path)

    print(f"[2/6] Loading candidates from {candidates_path} ...")
    candidates = load_candidates(candidates_path)
    print(f"      Loaded {len(candidates)} candidate profiles.")

    print("[3/6] Parsing job description into structured requirements (Claude) ...")
    job_req = parse_job_description(jd_text)
    print(f"      Role: {job_req.role_title} | Seniority: {job_req.seniority_level} | "
          f"Min experience: {job_req.min_years_experience}y")
    print(f"      Must-haves: {job_req.must_have_skills}")
    print(f"      Soft signals: {job_req.soft_skill_signals}")

    print(f"[4/6] Computing semantic similarity for {len(candidates)} candidates ...")
    semantic_scores = compute_semantic_scores(candidates, job_req, model_name=config.EMBEDDING_MODEL_NAME)

    top_k = min(config.TOP_K_FOR_LLM_RERANK, len(candidates))
    shortlist_pool = prefilter_candidates(candidates, job_req, semantic_scores, top_k)
    print(f"      Pre-filtered to top {len(shortlist_pool)} candidates for LLM holistic review.")

    print(f"[5/6] Running LLM holistic evaluation on {len(shortlist_pool)} candidates ...")
    llm_results = evaluate_candidates_batch(shortlist_pool, job_req, batch_size=8)

    print("[6/6] Computing composite scores and final ranking ...")
    scores = build_composite_scores(shortlist_pool, job_req, semantic_scores, llm_results)

    candidates_by_id = {c.candidate_id: c for c in candidates}
    write_ranked_csv(scores, candidates_by_id, output_path, top_n=top_n)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s.")
    print(f"\nTop {min(5, len(scores))} candidates:")
    for s in scores[:5]:
        print(f"  #{s.rank}  {s.name} ({s.candidate_id})  composite={s.composite_score*100:.1f}")

    return scores, job_req


def main():
    parser = argparse.ArgumentParser(description="AI Recruiter Ranking System")
    parser.add_argument("--jd", default=str(config.DATA_DIR / "job_description.txt"),
                         help="Path to job description text file")
    parser.add_argument("--candidates", default=str(config.DATA_DIR / "candidates.csv"),
                         help="Path to candidates CSV")
    parser.add_argument("--output", default=str(config.OUTPUT_DIR / "ranked_candidates.csv"),
                         help="Path to write ranked output CSV")
    parser.add_argument("--top-n", type=int, default=config.FINAL_SHORTLIST_SIZE,
                         help="Number of top candidates to include in final output")
    args = parser.parse_args()

    try:
        run_pipeline(args.jd, args.candidates, args.output, args.top_n)
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
