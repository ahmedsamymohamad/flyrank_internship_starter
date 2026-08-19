import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    features_path = "work/outputs/capstone_features.parquet"
    if not os.path.exists(features_path):
        print(f"Features file {features_path} not found.")
        return

    df = pd.read_parquet(features_path)
    
    # Check data
    print("Total rows:", len(df))
    
    # Drop NAs in features
    feature_cols = ['clicks_feat', 'impressions_feat', 'avg_pos_feat', 'search_volume', 'competition', 'word_count']
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Split
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']
    
    X_train = train_df[feature_cols]
    y_train = train_df['is_declining_label']
    
    X_test = test_df[feature_cols]
    y_test = test_df['is_declining_label']
    
    base_rate_train = y_train.mean()
    base_rate_test = y_test.mean()
    print(f"Base rate (train): {base_rate_train:.2%}")
    print(f"Base rate (test):  {base_rate_test:.2%}")
    
    # Baseline: Rank by negative of clicks drop in feature window (if available).
    # Since we don't have past-past clicks, our transparent rule baseline:
    # High clicks + High average position -> vulnerable. 
    # Let's say baseline score is just clicks_feat * avg_pos_feat
    baseline_score = test_df['clicks_feat'] * test_df['avg_pos_feat']
    baseline_auc = roc_auc_score(y_test, baseline_score)
    print(f"Baseline AUC: {baseline_auc:.3f}")
    
    # Train model
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    
    model_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"Model AUC: {model_auc:.3f}")
    
    # Feature Importance
    importances = clf.feature_importances_
    feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    
    # Plot feature importance
    plt.figure(figsize=(8,5))
    plt.barh([x[0] for x in feat_imp][::-1], [x[1] for x in feat_imp][::-1])
    plt.title("Feature Importance for Decline Prediction")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("work/outputs/feature_importance.png")
    
    # Save results to JSON
    results = {
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "base_rate_test": base_rate_test,
        "baseline_auc": baseline_auc,
        "model_auc": model_auc,
        "feature_importances": {k: float(v) for k, v in feat_imp}
    }
    
    with open("work/outputs/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("Saved evaluation results to work/outputs/eval_results.json")

if __name__ == "__main__":
    main()
