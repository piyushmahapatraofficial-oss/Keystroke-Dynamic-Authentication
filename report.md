# Academic Report
# Keystroke Dynamics Authentication Using Machine Learning
## CMU Benchmark Dataset (Killourhy & Maxion, 2009)

---

## Abstract

This report presents a machine learning study of keystroke dynamics for user authentication using the CMU Keystroke Dynamics Benchmark dataset. We implement and evaluate two authentication tasks: **user identification** (who typed this?) and **user verification** (did the genuine user type this?). We compare five classification algorithms and analyse which timing features best discriminate between users. All experiments use a temporal train/test split to prevent data leakage.

---

## 1. Introduction

Keystroke dynamics is a behavioural biometric that captures the unique rhythm with which a person types. Unlike passwords, which can be stolen, the typing pattern is difficult to replicate. The CMU benchmark records 51 subjects typing the fixed password `.tie5Roanl` across 400 repetitions each, providing a controlled experimental environment for evaluating keystroke-based authentication.

This study addresses two tasks:
1. **Identification**: Given a typing sample, which of the 51 subjects produced it? (51-class classification)
2. **Verification**: Given a typing sample and a claimed identity, is the claim genuine? (binary classification)

---

## 2. Dataset

| Property | Value |
|---|---|
| Source | CMU Keystroke Dynamics Benchmark |
| Subjects | 51 |
| Repetitions per subject | 400 |
| Sessions per subject | 8 |
| Features | 31 timing features |
| Total samples | ~20,400 |
| Password | `.tie5Roanl` (fixed) |

### 2.1 Feature Types

The 31 features capture three types of timing events:
- **H.key** — Hold time: duration a key is held down
- **DD.key1.key2** — Down-Down latency: time from pressing key1 to pressing key2
- **UD.key1.key2** — Up-Down latency: time from releasing key1 to pressing key2

All timings are in seconds and reflect sub-second finger movements.

### 2.2 Data Quality

- No missing values were found in the dataset.
- Outliers exist (|z| > 4) in several features, most likely caused by typing hesitations or brief distractions. These were retained because they represent genuine typing behaviour.

---

## 3. Methodology

### 3.1 Temporal Train / Test Split

We use **sessions 1–6 for training** and **sessions 7–8 for testing**.

**Why not random splitting?**

Random splitting assigns samples from all sessions to both training and test sets. Because each session represents a continuous typing episode, samples in the same session are highly correlated. A random split allows the model to "see" the user's style from every session, including sessions 7 and 8, before testing — this artificially inflates performance estimates.

Temporal splitting respects the realistic deployment scenario: a system must authenticate users based on later sessions using knowledge learned from earlier sessions. This produces honest, deployable performance estimates.

### 3.2 Data Leakage Prevention

- Feature scalers (StandardScaler) are **fitted on training data only** and applied to test data.
- Verification decision thresholds are selected on a validation fold (session 6) **only**.
- The test set (sessions 7–8) is **never consulted** during model development or threshold selection.

### 3.3 Random Seeds

All models and train/test splits use `random_state=42` to ensure fully reproducible results.

---

## 4. Experiment 1: Exploratory Data Analysis

Key findings from the EDA:

1. **Typing times are approximately log-normal**: Most hold times cluster between 0.05 s and 0.2 s, with a right-skewed tail.
2. **Inter-key latencies are wider**: Down-Down latencies range from near 0 to over 0.5 s, reflecting variability in finger transitions.
3. **Session effects**: Mean timing values shift slightly across sessions, suggesting mild practice or fatigue effects. However, the within-subject pattern remains consistent, which is what allows authentication.
4. **Subject variability**: Different subjects have distinctly different mean hold times for the same key — this is the signal that ML models exploit.
5. **Outliers**: A small number of samples show extreme values (|z| > 4), likely caused by distraction or accidental key holds. These do not dominate any feature.
6. **Feature correlation**: Hold times for adjacent keys tend to be moderately correlated. Down-Down and Up-Down latencies for the same key pair are highly correlated.

---

## 5. Experiment 2 & 3: User Identification

Five models were trained and compared using macro-F1 as the primary metric. Macro-F1 is preferred over accuracy because it weights each user equally, regardless of class imbalance.

### 5.1 Models and Hyperparameters

| Model | Key Hyperparameters |
|---|---|
| Logistic Regression | C=1.0, max_iter=1000, solver=lbfgs |
| K-Nearest Neighbors | k=5, metric=euclidean |
| Decision Tree | max_depth=20, min_samples_leaf=2 |
| Random Forest | 200 trees, min_samples_leaf=1 |
| Support Vector Machine | RBF kernel, C=10, gamma=scale |

Feature scaling (StandardScaler) was applied to Logistic Regression, KNN, and SVM. Tree-based models do not require scaling.

### 5.2 Results

The models were evaluated on the held-out test sessions (Sessions 7–8, 5,100 samples) across all 51 subjects:

| Model | Accuracy | Macro-F1 | Macro-Precision | Macro-Recall |
|---|---|---|---|---|
| **Random Forest** | **0.8682** | **0.8609** | **0.8751** | **0.8682** |
| Support Vector Machine (RBF) | 0.8141 | 0.8091 | 0.8244 | 0.8141 |
| K-Nearest Neighbors (k=5) | 0.7778 | 0.7693 | 0.7844 | 0.7778 |
| Logistic Regression | 0.7571 | 0.7508 | 0.7649 | 0.7571 |
| Decision Tree | 0.5998 | 0.5972 | 0.6247 | 0.5998 |

**Observations**:
- **Random Forest** achieved the highest test macro-F1 (0.8609) and accuracy (86.82%), demonstrating that bagging non-linear decision trees effectively models the complex interactions between keystroke timing features.
- **SVM with RBF kernel** ranked second (macro-F1: 0.8091), confirming that user clusters in the scaled 31-dimensional space have smooth non-linear boundaries.
- **KNN** attained 0.7693 macro-F1, showing strong local clustering.
- **Logistic Regression** achieved 0.7508 macro-F1, providing a solid linear baseline.
- **Single Decision Tree** suffered from high variance and achieved 0.5972 macro-F1, demonstrating why ensemble methods are necessary.

### 5.3 Why Macro-F1?

Each subject represents one class. With 51 roughly balanced classes, accuracy and macro-F1 are similar. We use macro-F1 because:
- It treats each subject equally.
- It is more informative when some subjects are easier to classify than others.
- It is the standard metric in biometric benchmarking.

---

## 6. Experiment 4: User Verification

### 6.1 Approach

For each subject *s*:
1. Build a binary dataset: samples from *s* are labelled **genuine** (1); all others are labelled **impostor** (0).
2. Train a Random Forest binary classifier on sessions 1–5 (fit fold).
3. Evaluate candidate thresholds on session 6 (validation fold) by computing FAR and FRR at each threshold.
4. Select the threshold that minimises |FAR − FRR|, which approximates the Equal Error Rate (EER) point.
5. Apply the selected threshold to sessions 7–8 (test fold) to report final FAR and FRR.

This three-way split (fit / validation / test) guarantees that threshold selection uses no test-set information.

### 6.2 Metrics

- **FAR (False Acceptance Rate)**: Fraction of impostor samples wrongly accepted as genuine. High FAR = low security.
- **FRR (False Rejection Rate)**: Fraction of genuine samples wrongly rejected. High FRR = poor usability.
- **EER (Equal Error Rate)**: The operating point where FAR ≈ FRR. Lower is better.

### 6.3 Threshold Selection

The minimum-|FAR−FRR| criterion on the validation fold is a standard approach in biometric system design. It finds the operating point where the system is equally wrong about genuine users and impostors — a reasonable default when there is no application-specific cost asymmetry.

Strictly, the threshold is set without touching the test set, preventing optimistic inflation of EER estimates.

### 6.4 Results

The binary verification classifiers were evaluated across all 51 subjects on Sessions 7–8 with validation-tuned thresholds:

| Metric | Cross-Subject Mean | Notes |
|---|---|---|
| **FAR (False Acceptance Rate)** | **0.0350 (3.50%)** | Very low security breach rate |
| **FRR (False Rejection Rate)** | **0.0853 (8.53%)** | Legitimate user rejection rate |
| **EER (Equal Error Rate approx.)** | **0.0602 (6.02%)** | Balance between usability & security |

Individual subject performance (detailed in `outputs/verification_results.csv`) shows:
- Perfect or near-perfect verification (FAR < 0.5%, FRR = 0%) for distinctive users like `s036` (FAR=0.0000, FRR=0.0000), `s043` (FAR=0.0038, FRR=0.0000), `s053` (FAR=0.0046, FRR=0.0000), and `s055` (FAR=0.0086, FRR=0.0000).
- Higher rejection rates for subjects with significant cross-session fatigue or typing drift (e.g. `s016` FRR=0.7900), highlighting intra-subject behavioural variability over time.

---

## 7. Experiment 5: Feature Importance

### 7.1 Methods

**MDI (Mean Decrease Impurity)**: The standard importance measure exported by scikit-learn's RandomForestClassifier. It sums the reduction in Gini impurity that each feature contributes across all trees. Fast but can be biased towards high-cardinality features.

**Permutation Importance**: For each feature, randomly shuffle its values across the test set and measure the drop in macro-F1. A large drop means the model relies heavily on that feature for correct predictions. This method is more reliable than MDI because it measures generalisation on unseen data.

### 7.2 Key Findings

Based on actual experimental evaluation on the held-out test data:

**Top Features by Permutation Importance (Macro-F1 decrease on Test Set)**:
1. **`UD.n.l`** (importance: 0.0611 ± 0.0027): Latency from releasing 'n' to pressing 'l'.
2. **`H.Shift.r`** (importance: 0.0579 ± 0.0037): Duration of the capital 'R' modifier key.
3. **`UD.l.Return`** (importance: 0.0325 ± 0.0019): Hand transition from final letter 'l' to 'Return'.
4. **`H.a`** (importance: 0.0307 ± 0.0018): Hold duration of vowel 'a'.
5. **`H.i`** (importance: 0.0251 ± 0.0021): Hold duration of key 'i'.
6. **`H.t`** (importance: 0.0230 ± 0.0012): Hold duration of key 't'.
7. **`DD.e.five`** (importance: 0.0191 ± 0.0014): Transition between letters and numbers.

**Top Features by Random Forest MDI (Gini Impurity Decrease)**:
1. `H.Shift.r` (0.0537)
2. `UD.n.l` (0.0490)
3. `H.a` (0.0418)
4. `UD.l.Return` (0.0401)
5. `H.t` (0.0382)

Both ranking methods strongly agree that **hold durations (`H.*`)** and **specific transitions involving capital shift or trailing characters (`UD.n.l`, `UD.l.Return`, `DD.e.five`)** provide the most distinctive biometric signature.

**Plain-language interpretation**: Each person has a characteristic "dwell time" for each key and a characteristic "gap rhythm" between successive key presses. These patterns are largely unconscious and are difficult to deliberately mimic. The machine learning models learn these patterns from early typing sessions and use them to verify identity in later sessions.

---

## 8. Limitations

1. **Fixed password**: The benchmark records only one password. Real systems would need to authenticate users across different input texts.
2. **Controlled lab environment**: Subjects knew they were being recorded and typed at a desk. Performance on mobile devices, under time pressure, or in everyday contexts may differ.
3. **Stationarity assumption**: The model assumes that a user's typing style in sessions 1–6 is representative of sessions 7–8. Over months or years, typing patterns may drift significantly.
4. **Small cohort**: 51 subjects is insufficient to draw broad conclusions about population-level performance.
5. **Same-session test samples**: Within the test sessions (7–8), samples from the same session may still be correlated. A stricter evaluation would hold out entire subjects.
6. **Equal class balance assumption**: In a real deployment, the ratio of genuine-to-impostor attempts is very skewed, which would change the optimal operating threshold.

---

## 9. Ethical Considerations

- Keystroke dynamics can be collected without explicit consent by monitoring typing activity.
- Systems that fail to authenticate a legitimate user (high FRR) may unfairly lock out users with disabilities or non-standard typing patterns.
- Systems that accept impostors (high FAR) provide a false sense of security.
- This technique should supplement passwords, not replace them entirely.

---

## 10. References

Killourhy, K. S., & Maxion, R. A. (2009). *Comparing anomaly-detection algorithms for keystroke dynamics.* International Conference on Dependable Systems and Networks (DSN). IEEE.

---

## 11. Conclusion

This project demonstrates that keystroke dynamics can effectively authenticate users, achieving good identification accuracy across 51 subjects using a temporal train/test split. Random Forest consistently performs well because it can model non-linear, feature-interaction patterns in timing data. User verification achieves reasonable EER values, confirming the discriminative power of typing rhythm. The most important features are key hold times and specific inter-key latencies, which represent stable aspects of individual typing biomechanics.

The benchmark is a useful academic tool but represents an idealised setting. Real-world keystroke authentication systems must address password variability, device diversity, and temporal drift in typing patterns.
