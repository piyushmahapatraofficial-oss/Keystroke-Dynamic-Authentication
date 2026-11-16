# Viva Preparation Document
# Keystroke Dynamics Authentication — G10

> **At least 20 questions and accurate answers based on the actual project.**
> All answers reflect the methodology, code, and results from this project.

---

## Section A: Foundations

**Q1. What is keystroke dynamics?**

Keystroke dynamics is a behavioural biometric that captures the unique rhythm with which a person types. The system records timing events — such as how long each key is held down and the gaps between successive key presses — and uses these to build a statistical model of a user's typing pattern. Because typing rhythm is largely unconscious and stable, it can be used to verify or identify users.

---

**Q2. What is the CMU Keystroke Dynamics dataset, and why is it used?**

The CMU Keystroke Dynamics Benchmark (Killourhy & Maxion, 2009) is a publicly available benchmark dataset specifically designed for evaluating keystroke authentication algorithms. It records 51 subjects typing the fixed password `.tie5Roanl` a total of 400 times each, spread across 8 sessions. It is widely used because it provides a standardised, well-documented baseline for comparison across research papers, making it ideal for academic work.

---

**Q3. What features does the dataset contain?**

The dataset contains 31 timing features per password entry:
- **H.key** (Hold time): how long the key is pressed
- **DD.key1.key2** (Down-Down latency): time from pressing key1 to pressing key2
- **UD.key1.key2** (Up-Down latency): time from releasing key1 to pressing key2

All values are in seconds. The password has 10 characters, so there are 10 hold times plus various inter-key latencies.

---

**Q4. What is the difference between user identification and user verification?**

| Task | Question answered | Type of problem |
|---|---|---|
| Identification | Who typed this? | 51-class classification |
| Verification | Did the claimed user type this? | Binary classification (genuine / impostor) |

Identification requires the system to pick one of 51 known users from a sample. Verification requires only a yes/no decision about one specific user claim — a much more practical and common use case in real systems.

---

**Q5. Why do you use a temporal train/test split instead of random splitting?**

Random splitting assigns samples from all sessions to both training and test sets. Because each session is a coherent typing episode, samples from the same session are highly correlated. Including a session's data in both training and test sets allows the model to "see" examples of the user's typing from every session before testing, which inflates accuracy estimates.

Temporal splitting (sessions 1–6 → train, sessions 7–8 → test) mirrors the real deployment scenario: a system is trained on early typing samples and must authenticate the user in future sessions it has never seen. This gives honest, deployable performance figures.

---

## Section B: Identification Experiment

**Q6. Which five classifiers did you train for identification and why?**

1. **Logistic Regression** — a simple linear baseline; interpretable and fast.
2. **K-Nearest Neighbors (KNN)** — a lazy learner that uses proximity in feature space; intuitive but sensitive to distance metric choice.
3. **Decision Tree** — a non-linear rule-based model; easy to visualise but prone to overfitting.
4. **Random Forest** — an ensemble of decision trees that reduces overfitting; generally the strongest performer on structured data.
5. **SVM (RBF kernel)** — finds the maximum-margin hyperplane in a high-dimensional feature space; effective for compact, well-separated clusters.

---

**Q7. What is macro-F1 and why do you use it over accuracy?**

Macro-F1 computes the F1 score for each class independently and then averages them, giving equal weight to every class regardless of size. With 51 classes that are roughly balanced (400 samples each), accuracy and macro-F1 give similar results. We use macro-F1 because:
- It treats each user as equally important.
- If some users are harder to classify, macro-F1 reflects this explicitly.
- It is the standard metric in biometric and multi-class benchmarking literature.

---

**Q8. What is feature scaling and which models require it?**

Feature scaling transforms features to a common range (e.g., zero mean, unit variance). It is required for:
- **Logistic Regression** — the gradient descent optimiser is sensitive to feature scale.
- **KNN** — distances in Euclidean space are dominated by features with large numeric ranges.
- **SVM** — the margin calculation is affected by feature magnitude.

**Tree-based models (Decision Tree, Random Forest) do not require scaling** because they use threshold comparisons, not distance or gradient calculations.

We use `StandardScaler`, fitted only on the training data, then applied identically to the test data to prevent leakage.

---

**Q9. What is a confusion matrix and what does it tell you?**

A confusion matrix is a 51×51 table where row *i* and column *j* contains the number of samples belonging to class *i* that were predicted as class *j*. The diagonal elements are correct predictions. Off-diagonal elements are misclassifications. It reveals:
- Which users are frequently confused with each other (similar typing patterns)
- Which users are always correctly identified (very distinctive typing)

---

**Q10. What hyperparameters did you use for each model?**

| Model | Key Hyperparameters | Rationale |
|---|---|---|
| Logistic Regression | C=1.0, max_iter=1000 | Default regularisation; sufficient iterations for convergence |
| KNN | k=5, metric=euclidean | Standard starting point for 51-class problem |
| Decision Tree | max_depth=20, min_samples_leaf=2 | Limits overfitting |
| Random Forest | 200 trees | More trees = more stable predictions |
| SVM | RBF kernel, C=10, gamma=scale | RBF handles non-linear boundaries; C=10 allows tighter margin |

---

## Section C: Verification Experiment

**Q11. What is FAR and FRR?**

- **FAR (False Acceptance Rate)**: The fraction of impostor attempts that the system incorrectly accepts as genuine. Formula: `FAR = FP / (FP + TN)` where the positives are impostors and the negatives are genuine. In plain terms: how often does an imposter get let in?
- **FRR (False Rejection Rate)**: The fraction of genuine attempts that the system incorrectly rejects. Formula: `FRR = FN / (FN + TP)` where the positives are genuine users. In plain terms: how often does the real user get locked out?

Low FAR means high security; low FRR means high usability. These two goals are in tension.

---

**Q12. What is EER (Equal Error Rate)?**

The EER is the operating point where FAR = FRR. At this point, the system is equally likely to reject a genuine user as it is to accept an impostor. Lower EER means better overall system performance. We approximate EER by selecting the threshold that minimises |FAR − FRR| on the validation data.

---

**Q13. How did you select the decision threshold without using the test set?**

We used a three-way temporal split:
1. **Fit fold** (sessions 1–5): Train the binary classifier.
2. **Validation fold** (session 6): Try 200 candidate thresholds and select the one minimising |FAR − FRR|.
3. **Test fold** (sessions 7–8): Apply the pre-selected threshold and report final FAR/FRR.

The test set is never consulted during threshold selection. This prevents the common mistake of tuning the threshold on test data and then reporting optimistically low EER values.

---

**Q14. Why do you train a separate classifier for each of the 51 subjects?**

Verification is a one-vs-all problem: for subject *s*, we want to know whether a sample came from *s* or from one of the other 50 subjects. Training one classifier per subject allows each classifier to learn the boundary between that specific subject's typing pattern and all impostor patterns. A single multi-class model could also be adapted for verification by using the probability score for the claimed class, but the per-subject approach is conceptually clearer and standard in the literature.

---

**Q15. What if the threshold is too low or too high?**

- **Threshold too low** (always predict genuine): FAR → 1 (every impostor is accepted), FRR → 0 (every genuine user is accepted). The system is useless for security.
- **Threshold too high** (always predict impostor): FRR → 1 (every genuine user is rejected), FAR → 0 (no impostor enters). The system is unusable.
- The optimal threshold (EER point) balances security and usability for the specific deployment scenario.

---

## Section D: Feature Analysis

**Q16. What is MDI feature importance?**

MDI (Mean Decrease in Impurity) measures, for each feature, the total reduction in Gini impurity (a measure of class mixing at a node) that is achieved by splitting on that feature, averaged across all trees in the Random Forest. A higher MDI score means the feature is used more frequently and more effectively to separate classes.

**Limitation**: MDI can overestimate the importance of high-cardinality or continuous features because they offer more possible split points.

---

**Q17. What is permutation importance and why is it better than MDI?**

Permutation importance measures the actual drop in macro-F1 when a feature's values are randomly shuffled on the test set. Shuffling a feature destroys its information content without changing the other features, so the drop in performance directly measures how much the model relies on that feature for prediction.

It is considered more reliable than MDI because:
- It is computed on held-out test data, measuring generalisation not training fit.
- It is not biased towards continuous or high-cardinality features.

---

**Q18. Which types of features were most important and why?**

Hold times (H.* features) and certain inter-key latencies (DD.*, UD.* features) rank highest. This is consistent with the biomechanics literature: each person's finger speed, dwell time per key, and hand transition rhythms are stable biometric traits shaped by years of typing practice. Features with high discriminative power have high inter-subject variance relative to intra-subject variance.

---

**Q19. Can correlated features distort importance rankings?**

Yes. When two features carry similar information (e.g., H.t and a DD latency involving the same key), the model may split their importance score between them. Neither will rank as highly as a single uncorrelated feature with the same total predictive power. For this reason, we report both MDI and permutation importance and compare the rankings.

---

## Section E: Limitations and Real-World Context

**Q20. Does this project prove keystroke dynamics is production-ready?**

No. The benchmark uses:
- A single fixed password, not free-text or varying input
- A controlled lab setting with adult subjects
- 51 subjects — too few for population-level statistical conclusions
- Sessions recorded over a short period — does not capture long-term drift in typing patterns

A real-world system would need to handle typing variability due to device type, ambient conditions, fatigue, injury, and password changes. This project demonstrates the method is feasible for a constrained benchmark, not that it is ready for deployment.

---

**Q21. What is data leakage and how did you prevent it in this project?**

Data leakage occurs when information from the test set influences model training or evaluation, producing optimistic results that do not generalise. We prevented it by:
1. **Temporal splitting**: never including test-set sessions (7–8) in training.
2. **Scaler fitted on training data only**: the StandardScaler's mean and variance are computed only on sessions 1–6.
3. **Threshold selection on validation fold**: the decision threshold for verification is chosen on session 6 only, before evaluation on sessions 7–8.
4. **No test-set peeking**: model selection and hyperparameter choices are based on domain knowledge and cross-validation on the training set.

---

**Q22. Why is keystroke dynamics considered a soft biometric?**

Keystroke dynamics is classified as a *behavioural* biometric rather than a *physiological* biometric (like a fingerprint). It is considered "soft" because:
- It can change over time (learning, injury, different keyboard)
- It cannot uniquely identify every person with certainty
- It is not universal (some people use voice input or touchscreens)

It is best used as a **continuous authentication** layer that augments password-based authentication rather than replacing it entirely.

---

**Q23. What would happen if you trained on sessions 7–8 and tested on sessions 1–6?**

This would create a temporal anachronism: the model would have already "seen the future" of the user's typing and would be tested on earlier, more variable data. In practice, it would not inflate accuracy much in this dataset because the sessions are symmetric. However, it violates the deployment logic (a system trained today cannot incorporate data from future sessions that do not yet exist).

---

**Q24. What is the difference between precision, recall, and F1?**

For a single class:
- **Precision** = TP / (TP + FP): of all samples predicted to belong to this class, how many actually do?
- **Recall** = TP / (TP + FN): of all samples that actually belong to this class, how many were correctly predicted?
- **F1** = 2 × (Precision × Recall) / (Precision + Recall): harmonic mean of precision and recall. Useful when precision and recall trade off against each other.

Macro versions average these over all 51 classes with equal weight.
