"""
03_verification.py — User Verification (Binary Authentication)
Keystroke Dynamics Authentication (CMU Benchmark)

For each of the 51 subjects, trains a one-vs-all binary classifier.
Computes FAR (False Acceptance Rate) and FRR (False Rejection Rate).
Decision thresholds are chosen on the training/validation data only —
the final test set is never used for threshold selection.
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import joblib

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join("dataset", "DSL-StrongPasswordData.xls")
GRAPHS_DIR  = os.path.join("outputs", "graphs")
MODELS_DIR  = "models"
RESULTS_DIR = "outputs"
os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

RANDOM_STATE = 42

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading dataset …")
df = pd.read_excel(DATA_PATH, engine="xlrd")

META_COLS = ["subject", "sessionIndex", "rep"]
FEAT_COLS = [c for c in df.columns if c not in META_COLS]

# Temporal split: sessions 1-6 → train+validate, sessions 7-8 → test
# Within the training block, we use sessions 1-5 to fit the model and
# session 6 as the validation fold for threshold selection.
train_val_df = df[df["sessionIndex"] <= 6].copy()
test_df      = df[df["sessionIndex"] >= 7].copy()

# Within training block: sessions 1-5 = fit, session 6 = validation
fit_df  = train_val_df[train_val_df["sessionIndex"] <= 5].copy()
val_df  = train_val_df[train_val_df["sessionIndex"] == 6].copy()

subjects = sorted(df["subject"].unique())
print(f"  Number of subjects : {len(subjects)}")
print(f"  Fit samples        : {len(fit_df)}")
print(f"  Validation samples : {len(val_df)}")
print(f"  Test samples       : {len(test_df)}")
print()

# ── Per-subject verification ───────────────────────────────────────────────────
# Approach:
#   • Build a RandomForest binary classifier per subject.
#   • Label = 1 (genuine) for the target subject, 0 (impostor) for all others.
#   • Use the classifier's probability score (P(genuine)) as the decision score.
#   • Select the threshold that minimises |FAR - FRR| on the validation set.
#   • Apply the selected threshold on the test set to get final FAR and FRR.

results = []
all_thresholds = []

print("Running per-subject verification …")
print("-" * 60)

for subj in subjects:
    # Build binary labels (1 = genuine, 0 = impostor)
    fit_df["label"]  = (fit_df["subject"] == subj).astype(int)
    val_df["label"]  = (val_df["subject"] == subj).astype(int)
    test_df["label"] = (test_df["subject"] == subj).astype(int)

    X_fit  = fit_df[FEAT_COLS].values
    y_fit  = fit_df["label"].values
    X_val  = val_df[FEAT_COLS].values
    y_val  = val_df["label"].values
    X_test = test_df[FEAT_COLS].values
    y_test = test_df["label"].values

    # Scale features
    scaler = StandardScaler()
    X_fit_sc  = scaler.fit_transform(X_fit)
    X_val_sc  = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)

    # Train binary classifier
    clf = RandomForestClassifier(
        n_estimators=100, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1
    )
    clf.fit(X_fit_sc, y_fit)

    # ── Threshold selection on VALIDATION data only ──
    val_scores = clf.predict_proba(X_val_sc)[:, 1]  # P(genuine)

    # Try a range of thresholds; pick the one that minimises |FAR - FRR|
    thresholds = np.linspace(0, 1, 200)
    best_thresh = 0.5
    best_diff   = float("inf")

    for t in thresholds:
        y_pred_val = (val_scores >= t).astype(int)
        # Genuine samples: y_val == 1, Impostor samples: y_val == 0
        genuine_mask  = (y_val == 1)
        impostor_mask = (y_val == 0)

        if genuine_mask.sum() == 0 or impostor_mask.sum() == 0:
            continue

        # FRR = genuine samples rejected / total genuine
        frr_val = np.mean(y_pred_val[genuine_mask] == 0)
        # FAR = impostors accepted / total impostors
        far_val = np.mean(y_pred_val[impostor_mask] == 1)

        diff = abs(far_val - frr_val)
        if diff < best_diff:
            best_diff   = diff
            best_thresh = t

    all_thresholds.append(best_thresh)

    # ── Evaluate on TEST data with the chosen threshold ──
    test_scores = clf.predict_proba(X_test_sc)[:, 1]
    y_pred_test = (test_scores >= best_thresh).astype(int)

    genuine_mask  = (y_test == 1)
    impostor_mask = (y_test == 0)

    frr_test = np.mean(y_pred_test[genuine_mask] == 0) if genuine_mask.sum() > 0 else np.nan
    far_test = np.mean(y_pred_test[impostor_mask] == 1) if impostor_mask.sum() > 0 else np.nan

    eer_approx = (far_test + frr_test) / 2  # approximate EER

    results.append({
        "Subject": subj,
        "Threshold": round(best_thresh, 4),
        "FAR": round(far_test, 4),
        "FRR": round(frr_test, 4),
        "EER_approx": round(eer_approx, 4),
    })

    print(f"  {subj:20s}  threshold={best_thresh:.3f}  FAR={far_test:.4f}  FRR={frr_test:.4f}")

# ── Summary ───────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(RESULTS_DIR, "verification_results.csv"), index=False)

mean_far = results_df["FAR"].mean()
mean_frr = results_df["FRR"].mean()
mean_eer = results_df["EER_approx"].mean()

print("\n===== VERIFICATION SUMMARY =====")
print(f"  Mean FAR : {mean_far:.4f} ({mean_far*100:.2f}%)")
print(f"  Mean FRR : {mean_frr:.4f} ({mean_frr*100:.2f}%)")
print(f"  Mean EER : {mean_eer:.4f} ({mean_eer*100:.2f}%)")
print(f"\nSaved: outputs/verification_results.csv")

# ── Plot FAR and FRR per subject ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 5))
x = np.arange(len(results_df))
ax.bar(x - 0.2, results_df["FAR"], width=0.4, label="FAR", color="tomato",   alpha=0.85)
ax.bar(x + 0.2, results_df["FRR"], width=0.4, label="FRR", color="steelblue", alpha=0.85)
ax.axhline(mean_far, color="red",   linestyle="--", linewidth=1, label=f"Mean FAR={mean_far:.3f}")
ax.axhline(mean_frr, color="blue",  linestyle="--", linewidth=1, label=f"Mean FRR={mean_frr:.3f}")
ax.set_xticks(x)
ax.set_xticklabels(results_df["Subject"], rotation=90, fontsize=6)
ax.set_ylabel("Rate")
ax.set_title("FAR and FRR per Subject (Verification)", fontsize=12, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "10_far_frr_per_subject.png"), dpi=150)
plt.close()
print("Saved: outputs/graphs/10_far_frr_per_subject.png")

# ── Threshold distribution ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(all_thresholds, bins=20, color="mediumpurple", edgecolor="white", alpha=0.85)
ax.set_title("Distribution of Chosen Thresholds Across Subjects", fontsize=12)
ax.set_xlabel("Decision Threshold")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "11_threshold_distribution.png"), dpi=150)
plt.close()
print("Saved: outputs/graphs/11_threshold_distribution.png")

print("\nVerification experiment complete.")
