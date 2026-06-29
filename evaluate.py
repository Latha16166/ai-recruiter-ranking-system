"""
Sanity-check script: correlates the pipeline's composite ranking against the
hidden `_true_seniority_score` baked into the synthetic dataset (NOT used
anywhere in the ranking pipeline itself — it exists purely so we have *some*
ground truth to validate against, since we don't have human recruiter labels
for the real dataset).

This is a directional sanity check, not a rigorous benchmark: the synthetic
"true" score only captures seniority/quality, not actual JD-fit, so don't
expect a perfect correlation — just a reasonable positive one for candidates
who matched the JD's track.

Usage:
    python -m src.evaluate --ranked output/ranked_candidates.csv --candidates data/candidates.csv
"""

import argparse
import csv

from scipy.stats import spearmanr


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranked", default="output/ranked_candidates.csv")
    parser.add_argument("--candidates", default="data/candidates.csv")
    args = parser.parse_args()

    ranked_rows = load_csv(args.ranked)
    candidate_rows = {r["candidate_id"]: r for r in load_csv(args.candidates)}

    pairs = []
    for row in ranked_rows:
        cid = row["candidate_id"]
        cand = candidate_rows.get(cid)
        if not cand or "_true_seniority_score" not in cand:
            continue
        pairs.append((float(row["composite_score"]), float(cand["_true_seniority_score"])))

    if len(pairs) < 3:
        print("Not enough overlapping rows with ground-truth score to evaluate.")
        return

    composite_scores, true_scores = zip(*pairs)
    corr, p_value = spearmanr(composite_scores, true_scores)

    print(f"Evaluated {len(pairs)} ranked candidates against synthetic ground truth.")
    print(f"Spearman correlation (composite_score vs _true_seniority_score): {corr:.3f} (p={p_value:.4f})")
    print("\nNote: this ground-truth signal captures general seniority/quality, not JD-specific fit, "
          "so a moderate positive correlation (not near-perfect) is expected and healthy — a candidate "
          "with high general seniority but the wrong skill track should NOT necessarily rank #1.")


if __name__ == "__main__":
    main()
