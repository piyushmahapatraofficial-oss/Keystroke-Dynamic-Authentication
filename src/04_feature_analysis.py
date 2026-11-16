"""
04_feature_analysis.py — Feature Importance Analysis
Keystroke Dynamics Authentication (CMU Benchmark)

Identifies which timing features best distinguish users using:
  1. Random Forest feature importances (mean decrease in impurity)
  2. Permutation importances (more robust — measures actual drop in performance)
Saves graphs to outputs/graphs/.
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import joblib

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join("dataset", "DSL-StrongPasswordData.xls")
GRAPHS_DIR  = os.path.join("outputs", "graphs")
MODELS_DIR  = "models"
RESULTS_DIR = "outputs"

RANDOM_STATE = 42

# ── Load data and split ────────────────────────────────────────────────────────
print("Loading dataset …")
df = pd.read_excel(DATA_PATH, engine="xlrd")

META_COLS = ["subject", "sessionIndex", "rep"]
FEAT_COLS = [c for c in df.columns if c not in META_COLS]

train_df = df[df["sessionIndex"] <= 6].copy()
test_df  = df[df["sessionIndex"] >= 7].copy()

X_train = train_df[FEAT_COLS].values
X_test  = test_df[FEAT_COLS].values

le = LabelEncoder()
y_train = le.fit_transform(train_df["subject"].values)
y_test  = le.transform(test_df["subject"].values)

# ── Load or retrain Random Forest ─────────────────────────────────────────────
rf_path = os.path.join(MODELS_DIR, "random_forest.pkl")
if os.path.exists(rf_path):
    print("Loading trained Random Forest from disk …")
    rf = joblib.load(rf_path)
else:
    print("Training Random Forest …")
    rf = RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)

# ── 1. Random Forest built-in feature importances (MDI) ───────────────────────
print("Computing Random Forest feature importances (MDI) …")
mdi_importances = pd.Series(rf.feature_importances_, index=FEAT_COLS)
mdi_importances = mdi_importances.sort_values(ascending=False)

print("\nTop 15 features (MDI):")
print(mdi_importances.head(15).to_string())
mdi_importances.to_csv(os.path.join(RESULTS_DIR, "feature_importances_mdi.csv"), header=["importance"])

# Bar chart — top 15
fig, ax = plt.subplots(figsize=(10, 6))
mdi_importances.head(15).plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
ax.set_title("Top 15 Features — Random Forest MDI Importance", fontsize=12, fontweight="bold")
ax.set_xlabel("Feature")
ax.set_ylabel("Importance (Mean Decrease Impurity)")
ax.tick_params(axis="x", labelrotation=45, labelsize=9)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "12_rf_mdi_importance.png"), dpi=150)
plt.close()
print("Saved: outputs/graphs/12_rf_mdi_importance.png")

# Full ranked bar chart
fig, ax = plt.subplots(figsize=(14, 5))
mdi_importances.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white", alpha=0.8)
ax.set_title("All Features — Random Forest MDI Importance (Ranked)", fontsize=12)
ax.set_xlabel("Feature")
ax.set_ylabel("Importance")
ax.tick_params(axis="x", labelrotation=90, labelsize=7)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "13_rf_mdi_all_features.png"), dpi=150)
plt.close()
print("Saved: outputs/graphs/13_rf_mdi_all_features.png")

# ── 2. Permutation importance on test data ────────────────────────────────────
# This measures the actual drop in macro-F1 when each feature is randomly
# shuffled, so it reflects how much the model USES each feature for prediction.
print("\nComputing permutation importances on test data (this may take ~30s) …")
perm = permutation_importance(
    rf, X_test, y_test,
    n_repeats=10,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    scoring="f1_macro"
)

perm_df = pd.DataFrame({
    "feature":    FEAT_COLS,
    "importance": perm.importances_mean,
    "std":        perm.importances_std
}).sort_values("importance", ascending=False).reset_index(drop=True)

print("\nTop 15 features (Permutation Importance):")
print(perm_df.head(15)[["feature", "importance", "std"]].to_string(index=False))
perm_df.to_csv(os.path.join(RESULTS_DIR, "feature_importances_permutation.csv"), index=False)

# Bar chart with error bars — top 15
top15 = perm_df.head(15)
fig, ax = plt.subplots(figsize=(11, 6))
ax.bar(top15["feature"], top15["importance"], yerr=top15["std"],
       color="darkorange", edgecolor="white", alpha=0.85, capsize=4)
ax.set_title("Top 15 Features — Permutation Importance on Test Data", fontsize=12, fontweight="bold")
ax.set_xlabel("Feature")
ax.set_ylabel("Mean Decrease in Macro-F1")
ax.tick_params(axis="x", labelrotation=45, labelsize=9)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "14_permutation_importance.png"), dpi=150)
plt.close()
print("Saved: outputs/graphs/14_permutation_importance.png")

# ── 3. Compare MDI vs Permutation rankings ────────────────────────────────────
top_n = 15
mdi_top  = mdi_importances.head(top_n).reset_index()
mdi_top.columns = ["feature", "mdi_rank"]
mdi_top["mdi_rank"] = range(1, top_n + 1)

perm_top = perm_df.head(top_n)[["feature"]].copy()
perm_top["perm_rank"] = range(1, top_n + 1)

comparison = pd.merge(mdi_top, perm_top, on="feature", how="outer")
print("\nMDI vs Permutation — top 15 comparison:")
print(comparison.to_string(index=False))
comparison.to_csv(os.path.join(RESULTS_DIR, "feature_importance_comparison.csv"), index=False)

# ── 4. Violin plots for top 5 features across subjects ────────────────────────
# Shows how much the top features vary between subjects
top5_features = mdi_importances.head(5).index.tolist()
fig, axes = plt.subplots(1, 5, figsize=(20, 6))
for i, feat in enumerate(top5_features):
    # Sample a few subjects to keep the plot readable
    sample_subjs = sorted(df["subject"].unique())[:20]
    subset = df[df["subject"].isin(sample_subjs)][["subject", feat]]
    sns.violinplot(data=subset, x="subject", y=feat, ax=axes[i],
                   palette="tab20", inner="quartile", cut=0)
    axes[i].set_title(feat, fontsize=9)
    axes[i].set_xlabel("Subject", fontsize=7)
    axes[i].set_ylabel("Duration (s)", fontsize=7)
    axes[i].tick_params(axis="x", labelrotation=90, labelsize=5)
plt.suptitle("Top 5 Features — Distribution Across First 20 Subjects", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "15_top_features_violin.png"), dpi=150)
plt.close()
print("Saved: outputs/graphs/15_top_features_violin.png")

print("\nFeature analysis experiment complete.")
