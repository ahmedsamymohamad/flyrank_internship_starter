import os
import json

def main():
    eval_file = "work/outputs/eval_results.json"
    if not os.path.exists(eval_file):
        print("eval_results.json not found.")
        return

    with open(eval_file, "r") as f:
        results = json.load(f)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FlyRank Capstone: Content Opportunity Scoring</title>
    <style>
        :root {{
            --bg-color: #0f172a;
            --surface: #1e293b;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent: #38bdf8;
            --accent-glow: rgba(56, 189, 248, 0.2);
            --border: #334155;
        }}
        body {{
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            line-height: 1.6;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
        }}
        .container {{
            max-width: 800px;
            margin: 40px 20px;
            background: var(--surface);
            padding: 40px 60px;
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            border: 1px solid var(--border);
        }}
        h1, h2, h3 {{
            color: var(--text-main);
            font-weight: 600;
        }}
        h1 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(to right, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .abstract {{
            background: rgba(255,255,255,0.03);
            border-left: 4px solid var(--accent);
            padding: 20px;
            margin: 30px 0;
            border-radius: 0 8px 8px 0;
            font-style: italic;
        }}
        .metric-card {{
            background: rgba(0,0,0,0.2);
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid var(--border);
            margin: 10px;
            flex: 1;
            transition: transform 0.2s;
        }}
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 20px var(--accent-glow);
            border-color: var(--accent);
        }}
        .metric-value {{
            font-size: 2rem;
            font-weight: bold;
            color: var(--accent);
        }}
        .metrics-row {{
            display: flex;
            justify-content: space-between;
            margin: 20px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{
            color: var(--accent);
        }}
        img {{
            max-width: 100%;
            border-radius: 8px;
            margin: 20px 0;
            border: 1px solid var(--border);
        }}
        a {{
            color: var(--accent);
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        footer {{
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid var(--border);
            text-align: center;
            color: var(--text-muted);
            font-size: 0.9rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Predicting Content Decline for Refresh Opportunities</h1>
        <p><strong>Author:</strong> AI Assistant & Intern | <strong>Lane:</strong> Refresh / Content Opportunity Scoring</p>
        
        <div class="abstract">
            <strong>Abstract:</strong> For SEO strategists deciding which pages to rewrite or refresh, we must predict which currently active content is on the verge of decline. We utilized the FlyRank internship warehouse dataset, extracting 90-day historical performance as features to predict significant traffic drops in the subsequent 30-day window. Using a Random Forest classifier over a time-aware split, our model achieved an AUC of {results['model_auc']:.3f}, significantly outperforming a rule-based baseline of {results['baseline_auc']:.3f}. These scores enable an automated, ranked queue of "at-risk" content, saving editor hours and preventing undetected organic decline.
        </div>

        <h2>1. Problem Framing</h2>
        <p><strong>Decision supported:</strong> Which pages should an editor prioritize for refreshing or monitoring today?</p>
        <p><strong>Cost of a wrong call:</strong> Wasted editorial effort on a page that was stable, or missing a page that loses thousands of organic clicks. A simple heuristic is insufficient because content performance fluctuates naturally, and multi-variable interaction (e.g., volume, position, previous momentum) is difficult to capture accurately in static rules.</p>

        <h2>2. Data & Methodology</h2>
        <p>We extracted features from the <code>fact_content_daily_performance</code> partitioned tables for March 2026 (training features) and May 2026 (test features). The labels were computed strictly from the *following* months (April 2026 and June 2026, respectively) to simulate a real deployment scenario without temporal leakage. We deliberately avoided using <code>trend_pct</code> and <code>trend_direction</code> as features to prevent label-derived leakage.</p>

        <h2>3. Results (vs Baseline)</h2>
        <div class="metrics-row">
            <div class="metric-card">
                <div class="metric-value">{results['base_rate_test']:.1%}</div>
                <div>Base Rate (Decline)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{results['baseline_auc']:.3f}</div>
                <div>Baseline AUC</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{results['model_auc']:.3f}</div>
                <div>Model AUC</div>
            </div>
        </div>

        <h2>4. Interpretation</h2>
        <p>The model identified specific feature interactions indicative of decline. Below is the relative importance of the observed features.</p>
        <!-- The feature importance image will be deployed alongside -->
        <img src="feature_importance.png" alt="Feature Importance Chart">

        <h2>5. Ranked Recommendations & Action Playbook</h2>
        <p>Our output provides a ranked queue of pages scored by their probability of decline. Editors should:</p>
        <ul>
            <li><strong>Score > 0.8:</strong> Prioritize for immediate technical review or content rewrite.</li>
            <li><strong>Score 0.5 - 0.8:</strong> Monitor in Search Console for keyword shift.</li>
            <li><strong>Score < 0.5:</strong> Maintain current strategy; do not modify unnecessarily.</li>
        </ul>

        <h2>6. Limitations</h2>
        <p>This work provides directional, decision-support scores, not causal proof. We measure historical association with decline, but we do not predict the exact algorithmic shifts of search engines.</p>

        <footer>
            Built on the FlyRank ML Internship dataset. <a href="https://flyrank.ai" target="_blank">Learn more at FlyRank.ai</a>
        </footer>
    </div>
</body>
</html>
"""
    os.makedirs("docs", exist_ok=True)
    with open("docs/index.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    
    # Also write paper url
    os.makedirs("submission", exist_ok=True)
    with open("submission/paper_url.txt", "w") as f:
        f.write("https://ahmedsamymohamad.github.io/flyrank_internship_starter/")
    
    print("Generated docs/index.html and submission/paper_url.txt")

if __name__ == "__main__":
    main()
