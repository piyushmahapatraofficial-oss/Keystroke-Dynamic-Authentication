"""
02_identification.py — User Identification (Multi-class Classification)
Keystroke Dynamics Authentication (CMU Benchmark)

Trains and compares 5 classifiers to identify which of the 51 subjects
typed a given keystroke sample. Uses sessions 1-6 for training and
sessions 7-8 for testing (temporal split — no data leakage).
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix, classification_report)
from sklearn.model_selection import cross_val_score
import joblib

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join("dataset", "DSL-StrongPasswordData.xls")
GRAPHS_DIR  = os.path.join("outputs", "graphs")
MODELS_DIR  = "models"
RESULTS_DIR = "outputs"
os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_STATE = 42

# ── Load dataset ───────────────────────────────────────────────────────────────
print("Loading dataset …")
df = pd.read_excel(DATA_PATH, engine="xlrd")

META_COLS = ["subject", "sessionIndex", "rep"]
FEAT_COLS = [c for c in df.columns if c not in META_COLS]

# ── Temporal train / test split ────────────────────────────────────────────────
# Sessions 1-6 → train | Sessions 7-8 → test
# This mirrors real-world usage: the system learns from early typing samples
# and must authenticate from later ones. Random splitting would mix sessions
# and give artificially high accuracy because the model sees samples from
# the same session in both train and test.
train_df = df[df["sessionIndex"] <= 6].copy()
test_df  = df[df["sessionIndex"] >= 7].copy()

print(f"  Train samples : {len(train_df)}  (sessions 1-6)")
print(f"  Test  samples : {len(test_df)}   (sessions 7-8)")

X_train = train_df[FEAT_COLS].values
y_train_raw = train_df["subject"].values
X_test  = test_df[FEAT_COLS].values
y_test_raw  = test_df["subject"].values

# Encode subject labels to integers
le = LabelEncoder()
le.fit(y_train_raw)
y_train = le.transform(y_train_raw)
y_test  = le.transform(y_test_raw)

# Save label encoder for later use
joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.pkl"))

# Feature scaling (required for LR, SVM, KNN — harmless for tree-based models)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)   # fit only on training data
X_test_sc  = scaler.transform(X_test)        # apply same scale to test

joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

# ── Define models ──────────────────────────────────────────────────────────────
# Hyperparameters chosen via understanding of the dataset size (20k samples,
# 51 classes) and common practices — not tuned on the test set.
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, C=1.0, solver="lbfgs",
        random_state=RANDOM_STATE
    ),
    "K-Nearest Neighbors": KNeighborsClassifier(
        n_neighbors=5, metric="euclidean"
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=20, min_samples_leaf=2, random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_leaf=1,
        random_state=RANDOM_STATE, n_jobs=-1
    ),
    "Support Vector Machine": SVC(
        kernel="rbf", C=10, gamma="scale", decision_function_shape="ovr",
        random_state=RANDOM_STATE, probability=True
    ),
}

# Tree-based models don't need scaling
unscaled_models = {"Decision Tree", "Random Forest"}

# ── Train, evaluate, compare ───────────────────────────────────────────────────
results = []
print("\nTraining and evaluating models …")
print("-" * 60)

for name, model in models.items():
    print(f"\n  [{name}]")
    Xtr = X_train if name in unscaled_models else X_train_sc
    Xte = X_test  if name in unscaled_models else X_test_sc

    # Train
    model.fit(Xtr, y_train)

    # Predict
    y_pred = model.predict(Xte)

    # Metrics
    acc  = accuracy_score(y_test, y_pred)
    mac_f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)
    mac_pre = precision_score(y_test, y_pred, average="macro", zero_division=0)
    mac_rec = recall_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"    Accuracy        : {acc:.4f}")
    print(f"    Macro-F1        : {mac_f1:.4f}")
    print(f"    Macro-Precision : {mac_pre:.4f}")
    print(f"    Macro-Recall    : {mac_rec:.4f}")

    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Macro-F1": round(mac_f1, 4),
        "Macro-Precision": round(mac_pre, 4),
        "Macro-Recall": round(mac_rec, 4),
    })

    # Save model
    safe_name = name.lower().replace(" ", "_")
    joblib.dump(model, os.path.join(MODELS_DIR, f"{safe_name}.pkl"))
    print(f"    Model saved to models/{safe_name}.pkl")

# ── Results table ──────────────────────────────────────────────────────────────
results_df = pd.DataFrame(results).sort_values("Macro-F1", ascending=False).reset_index(drop=True)
print("\n\n===== MODEL COMPARISON (sorted by Macro-F1) =====")
print(results_df.to_string(index=False))
results_df.to_csv(os.path.join(RESULTS_DIR, "identification_results.csv"), index=False)
print(f"\nSaved: outputs/identification_results.csv")

# ── Bar chart — model comparison ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(results_df))
w = 0.2
metrics = ["Accuracy", "Macro-F1", "Macro-Precision", "Macro-Recall"]
colors  = ["steelblue", "darkorange", "green", "purple"]
for i, (metric, color) in enumerate(zip(metrics, colors)):
    ax.bar(x + i * w, results_df[metric], width=w, label=metric, color=color, alpha=0.85)
ax.set_xticks(x + 1.5 * w)
ax.set_xticklabels(results_df["Model"], rotation=15, ha="right", fontsize=9)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title("User Identification — Model Comparison", fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "08_model_comparison.png"), dpi=150)
plt.close()
print("Saved: outputs/graphs/08_model_comparison.png")

# ── Confusion matrix for best model ───────────────────────────────────────────
best_name = results_df.iloc[0]["Model"]
best_model = models[best_name]
Xte_best = X_test if best_name in unscaled_models else X_test_sc
y_pred_best = best_model.predict(Xte_best)

cm = confusion_matrix(y_test, y_pred_best)
fig, ax = plt.subplots(figsize=(16, 14))
sns.heatmap(cm, ax=ax, cmap="Blues", linewidths=0.2, cbar=True,
            xticklabels=le.classes_, yticklabels=le.classes_,
            annot=False)   # too many classes for text annotations
ax.set_title(f"Confusion Matrix — {best_name} (Best Model)", fontsize=12, fontweight="bold")
ax.set_xlabel("Predicted Subject")
ax.set_ylabel("True Subject")
ax.tick_params(axis="both", labelsize=5)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "09_confusion_matrix.png"), dpi=150)
plt.close()
print("Saved: outputs/graphs/09_confusion_matrix.png")

# Save detailed classification report for best model
report = classification_report(y_test, y_pred_best, target_names=le.classes_, zero_division=0)
with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
    f.write(f"Best model: {best_name}\n\n")
    f.write(report)
print("Saved: outputs/classification_report.txt")

print(f"\n\nBest model for identification: {best_name}")
print("Identification experiment complete.")
