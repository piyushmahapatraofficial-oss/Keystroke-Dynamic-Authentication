"""
01_eda.py — Exploratory Data Analysis
Keystroke Dynamics Authentication (CMU Benchmark)

Analyses key-hold durations and inter-key timing, session-level variation,
outliers, and distributions. Saves all graphs to outputs/graphs/.
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — saves files without a display
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")   # suppress xlrd/pandas deprecation notices

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join("dataset", "DSL-StrongPasswordData.xls")
GRAPHS_DIR  = os.path.join("outputs", "graphs")
os.makedirs(GRAPHS_DIR, exist_ok=True)

# ── Load dataset ───────────────────────────────────────────────────────────────
print("Loading dataset …")
df = pd.read_excel(DATA_PATH, engine="xlrd")
print(f"  Shape      : {df.shape}")
print(f"  Subjects   : {df['subject'].nunique()}")
print(f"  Sessions   : {df['sessionIndex'].nunique()}")
print(f"  Repetitions per subject: {df.groupby('subject').size().mean():.0f}")
print(f"  Columns    : {list(df.columns)}")
print()

# Feature columns are everything except subject, sessionIndex, rep
META_COLS = ["subject", "sessionIndex", "rep"]
FEAT_COLS = [c for c in df.columns if c not in META_COLS]
print(f"  Feature columns ({len(FEAT_COLS)}): {FEAT_COLS}")

# ── 1. Basic statistics ────────────────────────────────────────────────────────
print("\n--- Descriptive statistics ---")
stats = df[FEAT_COLS].describe().T
stats.to_csv(os.path.join("outputs", "feature_stats.csv"))
print(stats[["mean", "std", "min", "max"]].to_string())

# Check for missing values
missing = df.isnull().sum().sum()
print(f"\n  Missing values: {missing}")

# ── 2. Histograms of all features ──────────────────────────────────────────────
print("\nPlotting feature histograms …")
n_cols = 4
n_rows = int(np.ceil(len(FEAT_COLS) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
axes = axes.flatten()
for i, col in enumerate(FEAT_COLS):
    axes[i].hist(df[col].dropna(), bins=40, color="steelblue", edgecolor="white", alpha=0.8)
    axes[i].set_title(col, fontsize=8)
    axes[i].set_xlabel("Time (s)", fontsize=7)
    axes[i].set_ylabel("Count", fontsize=7)
    axes[i].tick_params(labelsize=6)
# Hide empty subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Distribution of All Timing Features", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "01_feature_histograms.png"), dpi=150)
plt.close()
print("  Saved: 01_feature_histograms.png")

# ── 3. KDE plots for selected Hold-time (H.) and LatencyDown-Down (DD.) features ─
print("Plotting KDE plots …")
hold_cols = [c for c in FEAT_COLS if c.startswith("H.")]
dd_cols   = [c for c in FEAT_COLS if c.startswith("DD.")]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for col in hold_cols:
    axes[0].set_xlabel("Duration (s)")
    df[col].dropna().plot.kde(ax=axes[0], linewidth=1, label=col)
axes[0].set_title("KDE — Key Hold Times (H.*)", fontsize=11)
axes[0].legend(fontsize=6, ncol=2)
axes[0].set_xlim(-0.1, 0.8)

for col in dd_cols:
    df[col].dropna().plot.kde(ax=axes[1], linewidth=1, label=col)
axes[1].set_title("KDE — Latency DownDown (DD.*)", fontsize=11)
axes[1].legend(fontsize=6, ncol=2)
axes[1].set_xlim(-0.2, 1.0)
axes[1].set_xlabel("Duration (s)")

plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "02_kde_plots.png"), dpi=150)
plt.close()
print("  Saved: 02_kde_plots.png")

# ── 4. Boxplots across sessions for a few representative features ──────────────
print("Plotting session boxplots …")
# Pick the first hold key and first latency for illustration
sample_feats = hold_cols[:3] + dd_cols[:3]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, col in enumerate(sample_feats):
    sns.boxplot(data=df, x="sessionIndex", y=col, ax=axes[i],
                palette="Set2", flierprops=dict(marker=".", markersize=2))
    axes[i].set_title(col, fontsize=9)
    axes[i].set_xlabel("Session")
    axes[i].set_ylabel("Duration (s)")
plt.suptitle("Timing Features Across Sessions (sample features)", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "03_session_boxplots.png"), dpi=150)
plt.close()
print("  Saved: 03_session_boxplots.png")

# ── 5. Session-level mean for one hold feature ─────────────────────────────────
print("Plotting session mean trend …")
session_means = df.groupby("sessionIndex")[hold_cols].mean()
fig, ax = plt.subplots(figsize=(10, 5))
session_means.plot(ax=ax, marker="o", linewidth=1.5)
ax.set_title("Mean Key-Hold Duration per Session", fontsize=12)
ax.set_xlabel("Session Index")
ax.set_ylabel("Mean Duration (s)")
ax.legend(fontsize=7, ncol=4, bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "04_session_mean_hold.png"), dpi=150)
plt.close()
print("  Saved: 04_session_mean_hold.png")

# ── 6. Per-subject mean hold-time for one feature (variation across users) ─────
print("Plotting per-subject variation …")
first_hold = hold_cols[0]
subj_means = df.groupby("subject")[first_hold].mean().sort_values()
fig, ax = plt.subplots(figsize=(14, 5))
subj_means.plot(kind="bar", ax=ax, color="teal", edgecolor="white")
ax.set_title(f"Per-Subject Mean of '{first_hold}'", fontsize=12)
ax.set_xlabel("Subject")
ax.set_ylabel("Mean Duration (s)")
ax.tick_params(axis="x", labelrotation=90, labelsize=6)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "05_subject_variation.png"), dpi=150)
plt.close()
print("  Saved: 05_subject_variation.png")

# ── 7. Outlier detection — z-score ────────────────────────────────────────────
print("Identifying outliers (|z| > 4) …")
z_scores  = np.abs((df[FEAT_COLS] - df[FEAT_COLS].mean()) / df[FEAT_COLS].std())
outlier_counts = (z_scores > 4).sum().sort_values(ascending=False)
print("  Top outlier features:")
print(outlier_counts[outlier_counts > 0].to_string())

fig, ax = plt.subplots(figsize=(12, 4))
outlier_counts.plot(kind="bar", ax=ax, color="salmon", edgecolor="white")
ax.set_title("Number of Outliers per Feature (|z| > 4)", fontsize=12)
ax.set_xlabel("Feature")
ax.set_ylabel("Count")
ax.tick_params(axis="x", labelrotation=90, labelsize=7)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "06_outlier_counts.png"), dpi=150)
plt.close()
print("  Saved: 06_outlier_counts.png")

# ── 8. Correlation heatmap ─────────────────────────────────────────────────────
print("Plotting correlation heatmap …")
corr = df[FEAT_COLS].corr()
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(corr, ax=ax, cmap="coolwarm", center=0, linewidths=0.3,
            xticklabels=True, yticklabels=True, annot=False)
ax.set_title("Feature Correlation Matrix", fontsize=13)
ax.tick_params(labelsize=6)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "07_correlation_heatmap.png"), dpi=150)
plt.close()
print("  Saved: 07_correlation_heatmap.png")

print("\nEDA complete. All graphs saved to outputs/graphs/")
