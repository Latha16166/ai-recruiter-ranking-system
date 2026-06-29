"""
Central configuration. Reads secrets from environment variables — never hardcode keys.

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."
    python -m src.main --jd data/job_description.txt --candidates data/candidates.csv
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Embedding backend: "local" uses sentence-transformers (no API cost),
# "none" disables semantic retrieval and falls back to a simpler keyword/LLM-only flow.
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "local")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# Retrieval / ranking knobs
TOP_K_FOR_LLM_RERANK = int(os.getenv("TOP_K_FOR_LLM_RERANK", "25"))
FINAL_SHORTLIST_SIZE = int(os.getenv("FINAL_SHORTLIST_SIZE", "15"))

# Composite score weights (must sum to 1.0). Tunable per-role if desired.
SCORE_WEIGHTS = {
    "semantic_fit": 0.20,      # embedding similarity between candidate profile and JD
    "skills_match": 0.25,      # structured overlap between required/preferred skills and candidate skills
    "experience_fit": 0.15,    # years of experience & seniority alignment vs JD requirement
    "llm_judgment": 0.30,      # Claude's holistic read of fit, growth signals, red flags
    "platform_signal": 0.10,   # GitHub/blog/behavioral activity, normalized
}

OUTPUT_DIR = PROJECT_ROOT / "output"
DATA_DIR = PROJECT_ROOT / "data"
