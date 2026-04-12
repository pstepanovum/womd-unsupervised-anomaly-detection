# PRESENTATION.md — Slide Content

> 10 slides · 20 minutes · CSC 642 Spring 2026

---

## Slide 1 — Title

**[Full-bleed header slide]**

# Unsupervised Statistical Learning for Kinematic Anomaly Detection in Autonomous Driving Scenarios

**Pavel Stepanov · Ramses Loaces · Justin Cabrera**
Department of Computer Science — University of Miami
CSC 642 · Spring 2026

---## Slide 2 — Problem & Motivation

**[Header]** The Long-Tail Problem in Autonomous Driving

**[Left column — text]**

- Most driving is routine: lane-following, steady speeds, predictable stops
- Safety-critical events live in the **long tail** — rare but dangerous
  - Sudden braking / aggressive acceleration
  - Pedestrians jaywalking
  - Near-miss interactions
- Labeled anomaly data is **scarce by definition**
- "Dangerous" is context-dependent — no universal ground truth

**[Right column — key insight box]**

> Unsupervised learning lets the data define normal — and flags what falls outside.

**[Bottom — two research goals]**

| Goal 1                                                    | Goal 2                                                            |
| --------------------------------------------------------- | ----------------------------------------------------------------- |
| Detect anomalous agents from kinematics alone — no labels | Predict 5-second future displacement from current kinematic state |

---

## Slide 3 — Related Work

**[Header]** Where This Work Fits

**[Table — method comparison]**

| Method               | Type             | Strength                          | Limitation               |
| -------------------- | ---------------- | --------------------------------- | ------------------------ |
| Kalman Filter        | Classical        | Interpretable, fast               | Linear dynamics only     |
| LSTM / GRU           | Deep learning    | Captures sequences                | Needs large labeled data |
| Transformer          | Deep learning    | State-of-the-art accuracy         | Black box, expensive     |
| One-Class SVM        | Unsupervised     | No labels needed                  | Kernel tuning sensitive  |
| **Isolation Forest** | **Unsupervised** | **Scalable, minimal assumptions** | **No temporal context**  |

**[Callout box]**

> Prior work on WOMD focuses on deep learning prediction.
> **Classical statistical baselines over the full agent population remain unexplored.**
> This work fills that gap.

---

## Slide 4 — Dataset & Pipeline

**[Header]** Waymo Open Motion Dataset v1.3.1

**[Top — dataset facts, 3 stat boxes]**

|                                   |                           |                                 |
| --------------------------------- | ------------------------- | ------------------------------- |
| **18,311** raw agent observations | **20 Hz** object tracking | **20-second** scenario segments |

**[Center — data flow diagram as text]**

```
Waymo GCS Bucket (TFRecords)
        │
        ▼
Python Notebook (Google Colab)
  · GCS authentication
  · Parse Scenario protos
  · Extract kinematics at t = 0, t = -0.1s, t = +5s
        │
        ▼
waymo_final_dataset.csv
        │
        ▼
R — Statistical Modeling
  · Preprocessing → Anomaly Detection → Motion Prediction
        │
        ▼
Results (Tables / Plots)
```

**[Bottom — agent type breakdown]**

| Agent Type      | Count (raw) | After preprocessing |
| --------------- | ----------- | ------------------- |
| Vehicle         | majority    | included            |
| Pedestrian      | ~25%        | included            |
| Cyclist         | ~5%         | included            |
| SDC (AV itself) | removed     | excluded            |

**Features extracted:** Speed · Acceleration · Yaw Rate · Vel_X · Vel_Y · Length · Width · FutureDist_5s

---

## Slide 5 — Methods: Anomaly Detection

**[Header]** Three-Layer Anomaly Detection Pipeline

**[Layer 1 box]**

### Layer 1 — Isolation Forest

- Input features: **Speed, Acceleration, YawRate**
- 100 trees, unsupervised, no labels
- Assigns anomaly score ∈ [0, 1] to each agent
- Threshold: **99th percentile → top 1% flagged**
- Result: **46 anomalous agents** out of 4,538

> Key idea: anomalies are rare and different — they get isolated in fewer random tree splits.

**[Layer 2 box]**

### Layer 2 — Statistical Validation (Fisher's Exact Test)

|            | Normal | Anomalous |
| ---------- | ------ | --------- |
| Vehicle    | ...    | ...       |
| Cyclist    | ...    | ...       |
| Pedestrian | ...    | ...       |

- Test: are anomalies distributed uniformly across agent types?
- **p = 0.0005** (Monte Carlo, 2,000 replicates)
- Anomalies cluster in specific agent types — **not random noise**

**[Layer 3 box]**

### Layer 3 — Random Forest Classifier

- Uses IF scores as pseudo-labels
- Features: **Length, Width, YawRate**
- Down-sampled to handle 99:1 class imbalance
- 5-fold CV optimizing AUC-ROC
- Learns to characterize what anomalous agents look like

---

## Slide 6 — Methods: Motion Prediction

**[Header]** Linear Regression for 5-Second Displacement

**[Model formula box]**

$$\Delta d_{5s} = \beta_0 + \beta_1 \cdot \text{Speed} + \beta_2 \cdot \text{Acceleration} + \beta_3 \cdot \text{YawRate} + \beta_4 \cdot V_x + \beta_5 \cdot V_y$$

**[Two experimental conditions — side by side]**

|             | Condition A                   | Condition B                      |
| ----------- | ----------------------------- | -------------------------------- |
| **Dataset** | All agents                    | Moving agents only (Speed > 0)   |
| **N**       | 6,614                         | 3,196                            |
| **Purpose** | Baseline including easy cases | Honest evaluation under dynamics |

**[Evaluation setup]**

- 80 / 20 train / test split
- 5-fold cross-validation on training set
- Metrics: **RMSE** (meters) · **R²**

**[Callout]**

> Stationary agents have near-zero displacement — trivially easy to predict.
> The two conditions reveal whether R² reflects real predictive power or dataset composition.

---

## Slide 7 — Results

**[Header]** Results

**[Section A — Descriptive Statistics]**

### Mean Speed by Agent Type (m/s)

| Agent Type | All Agents | Moving Only |
| ---------- | ---------- | ----------- |
| Vehicle    | 2.47       | **6.74**    |
| Cyclist    | 4.20       | **3.92**    |
| Pedestrian | 1.01       | **1.07**    |

> Without filtering: vehicles appear _slower_ than cyclists due to stopped cars at intersections.
> Filtering restores expected ordering.

**[Section B — Anomaly Detection Results]**

### Random Forest Classifier Performance

| Metric            | CV (Train) | Test (Hold-out) |
| ----------------- | ---------- | --------------- |
| AUC-ROC           | **0.887**  | —               |
| Sensitivity       | 0.836      | 0.796           |
| Specificity       | 0.839      | 0.667           |
| Accuracy          | —          | **79.5%**       |
| Balanced Accuracy | —          | **73.1%**       |

Fisher's Exact Test: **p = 0.0005** — anomalies are non-uniformly distributed across agent types.

**[Section C — Motion Prediction Results]**

### Linear Regression — 5-Second Displacement

| Dataset            | Test RMSE  | Test R²   | CV RMSE | CV R² |
| ------------------ | ---------- | --------- | ------- | ----- |
| With stationary    | **7.44 m** | **0.880** | 7.65 m  | 0.886 |
| Without stationary | 13.15 m    | 0.728     | 10.05 m | 0.860 |

---

## Slide 8 — Discussion & Limitations

**[Header]** What the Results Tell Us

**[Two key takeaways — side by side boxes]**

**Takeaway 1: Unsupervised detection works**

- IF scores carry real discriminative signal
- p = 0.0005 confirms structure, not noise
- Anomalies concentrate in pedestrians / cyclists — consistent with irregular urban kinematics

**Takeaway 2: Dataset composition dominates metrics**

- Removing stationary agents: ΔR² = −0.15
- Not a model failure — a dataset artifact
- **Benchmarks on mixed populations are inflated**
- Moving-only evaluation is the honest number for dynamic AV deployment

**[Limitations — bulleted]**

- **Pseudo-labels:** 46 anomaly labels come from IF scores, not verified ground truth — test specificity of 0.667 reflects this noise
- **Single frame:** We use one snapshot per agent, ignoring the full 20-second trajectory
- **Linear model:** Pedestrian and cyclist motion may be nonlinear
- **No temporal context:** Isolation Forest operates on instantaneous kinematics only

---

## Slide 9 — Conclusion & Future Work

**[Header]** Summary & Next Steps

**[Left — what we built]**

### Two-Stage Pipeline

**Stage 1 — Anomaly Detection**

- Isolation Forest on Speed / Acceleration / Yaw Rate
- Fisher's Exact Test: p < 0.001
- Random Forest characterization: AUC = 0.887

**Stage 2 — Motion Prediction**

- Linear regression on 5 kinematic features
- R² = 0.88 with stationary agents included
- Strong classical baseline

**[Right — future work]**

### Next Steps

1. Temporal features over the full 20-second trajectory
2. Validate anomalies against Waymo's `IsInteractive` ground-truth flags
3. Mahalanobis distance scoring as IF alternative
4. Neural extensions benchmarked against these baselines

**[Bottom — takeaway banner]**

> Classical statistical methods deliver **interpretable, statistically validated safety insights** from large-scale real-world AV data — without deep learning.

**Code:** github.com/pstepanovum/womd-unsupervised-anomaly-detection

---

## Slide 10 — Q&A

**[Header]** Thank You — Questions?

**[Center]**

**Pavel Stepanov** — pas273@miami.edu
**Ramses Loaces** — rxl1238@miami.edu
**Justin Cabrera** — justin.cabrera2718@gmail.com

**[Bottom — quick reference for Q&A]**

| Key Result                  | Value                                |
| --------------------------- | ------------------------------------ |
| Agents analyzed             | 4,538 (anomaly) · 6,614 (regression) |
| Anomalies flagged           | 46 (top 1% by IF score)              |
| Fisher's p-value            | 0.0005                               |
| RF AUC-ROC                  | 0.887                                |
| Regression R² (all agents)  | 0.880                                |
| Regression R² (moving only) | 0.728                                |
