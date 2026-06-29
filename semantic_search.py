"""
Stage 2: Semantic Retrieval (recall stage).

We embed every candidate's full profile text and the job's structured
requirement summary, then compute cosine similarity. This is intentionally
the "wide net" stage — its job is to make sure no plausible candidate gets
dropped before the more expensive, more discerning LLM re-ranking stage runs.

Uses local sentence-transformers by default (free, no API call, runs anywhere).
If unavailable, falls back to a TF-IDF cosine similarity so the pipeline still
works without any extra model download.
"""

import numpy as np

from src.models import Candidate, JobRequirements

_model = None
_backend = None


def _try_load_sentence_transformer(model_name: str):
    global _model, _backend
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(model_name)
        _backend = "sentence_transformers"
        return True
    except Exception as e:
        print(f"  [Falling back from sentence-transformers: {e}]")
        return False


def _embed_with_sentence_transformers(texts: list[str]) -> np.ndarray:
    embeddings = _model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return np.array(embeddings)


def _embed_with_tfidf(texts: list[str]) -> np.ndarray:
    """Fallback: TF-IDF vectors, L2-normalized so dot product == cosine similarity."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    global _model, _backend
    if _model is None or _backend != "tfidf":
        _model = TfidfVectorizer(max_features=4000, stop_words="english", ngram_range=(1, 2))
        matrix = _model.fit_transform(texts)
        _backend = "tfidf"
    else:
        matrix = _model.transform(texts)
    return normalize(matrix).toarray()


def embed_texts(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    global _backend
    if _backend is None:
        loaded = _try_load_sentence_transformer(model_name)
        if not loaded:
            _backend = "tfidf"

    if _backend == "sentence_transformers":
        return _embed_with_sentence_transformers(texts)
    return _embed_with_tfidf(texts)


def jd_query_text(job_req: JobRequirements) -> str:
    """Build the text we embed for the JD side — a recruiter-style summary,
    not the raw JD, so the comparison is apples-to-apples with candidate profiles."""
    return f"""Role: {job_req.role_title} ({job_req.seniority_level})
Required experience: {job_req.min_years_experience}+ years
Must-have skills: {', '.join(job_req.must_have_skills)}
Nice-to-have skills: {', '.join(job_req.nice_to_have_skills)}
Core responsibilities: {'; '.join(job_req.core_responsibilities)}
Desired traits: {', '.join(job_req.soft_skill_signals)}
Preferred domain background: {', '.join(job_req.domain_preferences) or 'None specified'}
"""


def compute_semantic_scores(
    candidates: list[Candidate],
    job_req: JobRequirements,
    model_name: str = "all-MiniLM-L6-v2",
) -> dict[str, float]:
    """Returns candidate_id -> semantic similarity score in [0, 1]."""
    candidate_texts = [c.to_text_block() for c in candidates]
    query_text = jd_query_text(job_req)

    all_texts = candidate_texts + [query_text]
    embeddings = embed_texts(all_texts, model_name=model_name)

    candidate_vecs = embeddings[:-1]
    query_vec = embeddings[-1]

    # cosine similarity (vectors are already normalized in both backends)
    sims = candidate_vecs @ query_vec

    # normalize to [0, 1] for blending with other scores later
    sims_min, sims_max = float(sims.min()), float(sims.max())
    if sims_max - sims_min < 1e-9:
        normalized = np.full_like(sims, 0.5)
    else:
        normalized = (sims - sims_min) / (sims_max - sims_min)

    return {c.candidate_id: float(score) for c, score in zip(candidates, normalized)}
