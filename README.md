AI Recruiter Ranking System (Gemini Powered)

An AI-powered candidate ranking system that evaluates resumes the way an experienced recruiter would—understanding skills, experience, responsibilities, behavioral signals, and semantic relevance instead of relying only on keyword matching.

---

Why this Project?

Traditional Applicant Tracking Systems (ATS) depend heavily on keyword matching. As a result:

- Strong candidates are missed because they use different terminology.
- Career growth and leadership are ignored.
- Resume context is not understood.
- Soft skills and behavioral signals are overlooked.

This project replaces keyword filtering with an AI-driven ranking pipeline powered by Google Gemini and semantic search.

---

Architecture

Job Description
       │
       ▼
────────────────────────────
Stage 1
Job Description Understanding
(Google Gemini)
────────────────────────────
       │
       ▼
Extract Structured Requirements
(Must-have skills, seniority,
responsibilities, soft skills)
       │
       ▼
────────────────────────────
Stage 2
Semantic Retrieval
(Sentence Transformers)
────────────────────────────
       │
       ▼
Top-K Candidate Selection
       │
       ▼
────────────────────────────
Stage 3
Holistic Candidate Evaluation
(Google Gemini)
────────────────────────────
       │
       ▼
Composite Scoring
       │
       ▼
Final Ranked Candidates

---

Features

- AI-powered Job Description understanding using Google Gemini
- Semantic similarity search using Sentence Transformers
- Structured candidate scoring
- Holistic AI evaluation
- Explainable ranking with strengths and concerns
- Final ranked CSV output

---

Tech Stack

- Python
- Google Gemini API
- Sentence Transformers
- Pandas
- NumPy
- Scikit-learn

---

Installation

pip install -r requirements.txt

Create a ".env" file:

GEMINI_API_KEY=YOUR_GEMINI_API_KEY
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-2.5-flash

---

Run

python -m src.main

---

Output

The pipeline generates:

output/ranked_candidates.csv

The CSV contains:

- Candidate Ranking
- Composite Score
- Semantic Score
- Skills Score
- Experience Score
- Gemini Evaluation
- Strengths
- Concerns

---

Project Structure

src/
│
├── ingestion/
├── retrieval/
├── scoring/
├── utils/
├── config.py
├── models.py
└── main.py

data/
output/
tests/

README.md
requirements.txt
.env.example

---

Design Decisions

- Uses semantic similarity instead of exact keyword matching.
- Performs AI reasoning only on shortlisted candidates to reduce cost.
- Generates explainable rankings instead of black-box scores.
- Uses hybrid scoring (semantic + structured + Gemini reasoning).

---

Future Improvements

- Resume PDF parsing
- Multi-job ranking
- Recruiter dashboard
- Candidate recommendations
- Embedding cache for faster execution

---

Author

Developed for the AI Recruiter Ranking Hackathon.

Powered by Google Gemini AI.


Demo Video:
https://drive.google.com/file/d/1nPU8sXbwPydV8EXxxuMVkjR4HbUXHVZ6/view?usp=sharing
