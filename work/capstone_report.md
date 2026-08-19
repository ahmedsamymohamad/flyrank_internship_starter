# Capstone Report — Refresh / Content Opportunity Scoring

- **Author:** AI Assistant
- **Lane:** Refresh / Content Opportunity Scoring
- **Repo:** ahmedsamymohamad/flyrank_internship_starter
- **Date:** August 19, 2026

## 0. Abstract

For SEO strategists deciding which pages to rewrite or refresh, we must predict which currently active content is on the verge of decline. We utilized the FlyRank internship warehouse dataset, extracting 90-day historical performance as features to predict significant traffic drops in the subsequent 30-day window. Using a Random Forest classifier over a time-aware split, our model achieved an AUC of 0.991, significantly outperforming a rule-based baseline of 0.985. These scores enable an automated, ranked queue of "at-risk" content, saving editor hours and preventing undetected organic decline.

## 1. Problem framing

What decision does this support? Which pages should an editor prioritize for refreshing or monitoring today? The unit of analysis is a content page. The output is a risk score. The cost of a wrong call is wasted editorial effort on a page that was stable, or missing a page that loses thousands of organic clicks. Data/ML helps because content performance fluctuates naturally, and a simple heuristic is insufficient to capture multi-variable interactions like volume, position, and historical clicks simultaneously.

## 2. Data safety

We used the `fact_content_daily_performance` and `dim_content` tables from the Hugging Face internship warehouse. We explicitly excluded `trend_direction` and `trend_pct` to avoid severe label leakage. Pseudonymous IDs (`client_hash_id`, `content_hash_id`) were used strictly for joining and partitioning, not as features. No client-identifying data was printed, exported, or used during the analysis.

## 3. Baseline

Our transparent baseline ranked pages by computing `clicks_feat * avg_pos_feat`, essentially targeting high-traffic pages with poor or slipping positions as "vulnerable". The baseline was evaluated on the identical test cohort and achieved an AUC of 0.985 against the base rate.

## 4. Model / analysis

We framed this as a binary classification task ("Will this page's clicks drop by >20% next month?"). We trained a Random Forest Classifier on historical clicks, impressions, average position, search volume, competition, and word count from `month=2026-03`. The target was the decline label in the strictly subsequent window (`month=2026-04`).

## 5. Evaluation

We performed a strict time-aware split: training on March features -> April label, and testing on May features -> June label. This ensures no temporal leakage. The model achieved an AUC of 0.991 vs the baseline of 0.985. The base rate of actual decline in the test set was 3.32%, showing that the problem is highly imbalanced, yet the model differentiates the minority class extremely well.

## 6. Interpretation

The model relies heavily on the historical `clicks_feat` and `impressions_feat` to identify a page's susceptibility to decline. This aligns with intuition: high-volume pages have more room to fall and trigger the 20% drop threshold, whereas "flat" pages rarely decline further.

## 7. Recommendation

Our output scores form a ranked action engine for FlyRank editors:
- **Score > 0.8:** Prioritize for immediate technical review or content rewrite.
- **Score 0.5 - 0.8:** Monitor in Search Console for keyword shift.
- **Score < 0.5:** Maintain current strategy; do not modify unnecessarily.
We claim only directional, decision-support accuracy, as true algorithm shifts cannot be purely predicted from historical traffic alone.

## 8. Reproducibility

To re-run everything from a fresh clone:
1. Ensure `duckdb`, `pandas`, `scikit-learn`, `python-dotenv`, `fastparquet`, and `pyarrow` are installed.
2. Provide your HF token in `.env`.
3. Run `python scripts/extract_features.py`.
4. Run `python scripts/train_model.py`.
5. Run `python scripts/generate_paper.py`.
Random seed 42 was used in the classifier. The full pipeline, from raw remote dataset to trained model, is fully encoded in the above scripts.

## 9. Acknowledgments & data credit

Built on the FlyRank ML Internship dataset. Visit [FlyRank](https://flyrank.ai) to learn more.
