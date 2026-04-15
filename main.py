"""
main.py
-------
Entry point — orchestrates the full SMS Spam Classification pipeline.
"""

import os
import sys
import warnings
import pandas as pd

# ── 1. FORCE PATH FIX ──────────────────────────────────────────
# This ensures Python looks in the root directory for the 'src' folder
project_root = os.path.dirname(os.path.abspath(__file__))
# print(f"[INFO] Project Root: {project_root}")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

warnings.filterwarnings('ignore')

# ── 2. LOCAL IMPORTS ───────────────────────────────────────────
try:
    from src.preprocessing      import load_and_preprocess
    from src.feature_extraction import get_tfidf_features
    from src.model_training     import get_models, train_all_models
    from src.hyperparameter_tuning import tune_logistic_regression, tune_svm
    from src.evaluation         import (evaluate_all_models, plot_model_comparison,
                                       plot_f1_ranking, plot_accuracy_vs_f1)
    from src.visualizations     import run_all_eda_charts
    from src.model_saver        import save_model
    print("✅ All modules imported successfully!")
except ImportError as e:
    print(f"❌ Critical Import Error: {e}")
    print("\n[TROUBLESHOOTING]")
    print(f"1. Ensure 'src' is a FOLDER at: {os.path.join(project_root, 'src')}")
    print(f"2. Ensure an empty '__init__.py' exists inside the 'src' folder.")
    sys.exit(1)

# ── 3. CONFIGURATION ───────────────────────────────────────────
print(os.path.join(os.getcwd(), "data", "spam.csv"))
DATASET_PATH = os.path.join(os.getcwd(), "data", "spam.csv")  # Adjust if your dataset is in a different location
os.makedirs("outputs", exist_ok=True)
os.makedirs("models",  exist_ok=True)

def banner(title: str):
    print("\n" + "█" * 55)
    print(f"  {title}")
    print("█" * 55)

# ── 4. MAIN PIPELINE ───────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  SMS SPAM CLASSIFICATION SYSTEM")
    print("  End-to-End NLP + ML Pipeline with Visualizations")
    print("=" * 55)

    # Step 1: Load & Preprocess
    banner("STEP 1 — LOADING & PREPROCESSING DATASET")
    df = load_and_preprocess(DATASET_PATH)

    # Step 2: EDA Charts (8 charts)
    banner("STEP 2 — EXPLORATORY DATA ANALYSIS (8 CHARTS)")
    run_all_eda_charts(df)

    # Step 3: Feature Extraction
    banner("STEP 3 — FEATURE EXTRACTION (TF-IDF)")
    X_train, X_test, y_train, y_test, vectorizer = get_tfidf_features(df)

    # Step 4: Train Models
    banner("STEP 4 — TRAINING ML MODELS")
    trained_models = train_all_models(get_models(), X_train, y_train)

    # Step 5: Evaluate
    banner("STEP 5 — EVALUATING ALL MODELS")
    base_results = evaluate_all_models(trained_models, X_test, y_test)

    # Step 6: Hyperparameter Tuning
    banner("STEP 6 — HYPERPARAMETER TUNING")
    tuned_lr,  _ = tune_logistic_regression(X_train, y_train)
    tuned_svm, _ = tune_svm(X_train, y_train)
    
    tuned_models = {
        "Tuned Logistic Regression": tuned_lr,
        "Tuned SVM":                 tuned_svm
    }
    tuned_results = evaluate_all_models(tuned_models, X_test, y_test)

    # Step 7: Final Comparison Charts
    banner("STEP 7 — FINAL MODEL COMPARISON (3 CHARTS)")
    all_results = pd.concat([base_results, tuned_results], ignore_index=True)
    all_results.sort_values('F1-Score', ascending=False, inplace=True)
    all_results.reset_index(drop=True, inplace=True)

    print("\n[RESULTS] Complete Model Comparison Table:")
    print(all_results.to_string(index=False))

    plot_model_comparison(all_results)   
    plot_f1_ranking(all_results)          
    plot_accuracy_vs_f1(all_results)      

    # Step 8: Select & Save Best Model
    banner("STEP 8 — SAVING BEST MODEL")
    best_row  = all_results.iloc[0]
    best_name = best_row['Model']

    print(f"\n{'★' * 55}")
    print(f"  🏆 BEST MODEL : {best_name}")
    print(f"     Accuracy   : {best_row['Accuracy']:.4f}")
    print(f"     F1-Score   : {best_row['F1-Score']:.4f}")
    print(f"{'★' * 55}")

    all_models = {**trained_models, **tuned_models}
    save_model(all_models[best_name], vectorizer, best_name)

    print("\n[✅] Pipeline complete!")
    print("     Check the 'outputs/' folder for all 12 charts.")
# run_app.py

from main import main

if __name__ == "__main__":
    main()   
