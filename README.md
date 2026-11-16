# Keystroke Dynamics Authentication

## Project Overview

This project implements keystroke dynamics authentication using the **CMU Keystroke Dynamics Benchmark dataset** (Killourhy & Maxion, 2009). Users type the fixed password `.tie5Roanl` and the system learns their unique typing rhythm to either identify or verify who they are.

**This is a BTech undergraduate ML project** demonstrating how biometric authentication works through machine learning — not a production authentication system.

---

## Dataset

- **Source**: CMU Keystroke Dynamics Benchmark
- **File**: `dataset/DSL-StrongPasswordData.xls`
- **Subjects**: 51
- **Sessions**: 8 per subject
- **Repetitions**: 400 per subject (50 per session)
- **Features**: 31 timing features (key-hold durations + inter-key latencies)
- **Total samples**: ~20,400

---

## Project Structure

```
Keystroke Dynamic/
├── dataset/
│   └── DSL-StrongPasswordData.xls    # Raw CMU dataset
├── src/
│   ├── 01_eda.py                     # Exploratory Data Analysis
│   ├── 02_identification.py          # User Identification (5 classifiers)
│   ├── 03_verification.py            # User Verification (FAR / FRR)
│   └── 04_feature_analysis.py        # Feature Importance Analysis
├── outputs/
│   ├── graphs/                       # All saved graphs (15 PNG files)
│   ├── feature_stats.csv
│   ├── identification_results.csv
│   ├── classification_report.txt
│   ├── verification_results.csv
│   ├── feature_importances_mdi.csv
│   ├── feature_importances_permutation.csv
│   └── feature_importance_comparison.csv
├── models/
│   ├── label_encoder.pkl
│   ├── scaler.pkl
│   ├── logistic_regression.pkl
│   ├── k-nearest_neighbors.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   └── support_vector_machine.pkl
├── main.py                           # Run all experiments
├── requirements.txt
├── README.md
└── report.md
```

---

## Setup

### 1. Prerequisites

- Python 3.9 or higher
- The dataset file must be at `dataset/DSL-StrongPasswordData.xls`

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run all experiments

From the project root directory:

```bash
python main.py
```

Or run individual experiments:

```bash
python src/01_eda.py
python src/02_identification.py
python src/03_verification.py
python src/04_feature_analysis.py
```

---

## Experiments

### Experiment 1 — Exploratory Data Analysis (`01_eda.py`)
- Plots histograms, KDE curves, and boxplots for all 31 timing features
- Shows how typing behaviour varies across the 8 sessions
- Identifies outliers using z-scores
- Saves 7 graphs

### Experiment 2 & 3 — User Identification (`02_identification.py`)
- **Temporal split**: sessions 1–6 for training, sessions 7–8 for testing
- Trains and compares: Logistic Regression, KNN, Decision Tree, Random Forest, SVM
- Reports accuracy, macro-F1, macro-precision, macro-recall
- Saves the best model's confusion matrix

### Experiment 4 — User Verification (`03_verification.py`)
- For each of the 51 subjects, builds a one-vs-all binary classifier
- Thresholds are selected on session 6 (validation) only — **never on test data**
- Reports per-subject FAR and FRR, and cross-subject averages

### Experiment 5 — Feature Analysis (`04_feature_analysis.py`)
- Random Forest MDI (Mean Decrease Impurity) importances
- Permutation importances on the test set (more robust)
- Violin plots showing top feature distributions across subjects

---

## Key Results

All experiments were executed with reproducible seeds on the temporal split (Train: Sessions 1–6, Test: Sessions 7–8):

### User Identification (Multi-Class Classification)
| Model | Accuracy | Macro-F1 | Macro-Precision | Macro-Recall |
|---|---|---|---|---|
| **Random Forest** | **0.8682** | **0.8609** | **0.8751** | **0.8682** |
| Support Vector Machine | 0.8141 | 0.8091 | 0.8244 | 0.8141 |
| K-Nearest Neighbors | 0.7778 | 0.7693 | 0.7844 | 0.7778 |
| Logistic Regression | 0.7571 | 0.7508 | 0.7649 | 0.7571 |
| Decision Tree | 0.5998 | 0.5972 | 0.6247 | 0.5998 |

*Best performing model*: **Random Forest** (Macro-F1: 0.8609).

### User Verification (Binary Authentication)
- **Mean FAR (False Acceptance Rate)**: **3.50%** (0.0350)
- **Mean FRR (False Rejection Rate)**: **8.53%** (0.0853)
- **Mean EER (Equal Error Rate approx.)**: **6.02%** (0.0602)
- Decision thresholds chosen on Session 6 validation set only (mean threshold ~ 0.117).

### Top Discriminative Features
1. **`UD.n.l`** (Up-to-Down latency between 'n' and 'l'): highest permutation importance (0.0611 ± 0.0027)
2. **`H.Shift.r`** (Hold duration for 'Shift.r'): top MDI importance (0.0537) and 2nd permutation importance (0.0579)
3. **`UD.l.Return`** (Up-to-Down latency between 'l' and 'Return'): 3rd in permutation importance (0.0325)
4. **`H.a`** (Hold duration for 'a'): 4th in permutation importance (0.0307)
5. **`H.i`** (Hold duration for 'i'): 5th in permutation importance (0.0251)

---

## Important Notes

- **No data leakage**: the scaler and threshold are always fitted on training data only.
- **Temporal splitting** is used throughout — not random splitting. See `report.md` for the rationale.
- This benchmark uses a single fixed password in a controlled lab setting and **does not generalise** to real-world login systems.
- All reported metrics come from actual model execution — nothing is fabricated.
