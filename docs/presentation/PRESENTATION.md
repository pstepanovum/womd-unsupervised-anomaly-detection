# PRESENTATION.md — Slide Content

> 10 slides · 20 minutes · CSC 642 Spring 2026

---

## Slide 1 — Title

**[Full-bleed header slide]**

# Unsupervised Statistical Learning for Kinematic Anomaly Detection in Autonomous Driving Scenarios

**Pavel Stepanov · Ramses Loaces · Justin Cabrera**
Department of Computer Science — University of Miami
CSC 642 · Spring 2026

## Slide 2 — Problem & Motivation

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

**Features extracted:** Speed · Acceleration · Yaw Rate · Vel_X · Vel_Y · Length · Width · FutureDist_5s

**Preprocessing:** Two analysis datasets derived from the same CSV — anomaly detection uses strict filter (Speed > 0, Valid_Current, no NAs) → **4,538 agents**; regression uses permissive filter (NA removal only) → **6,614 agents**

---

## Slide 5 — Waymo Dataset Statistics

**[Header]** What the Data Looks Like

**[Agent type breakdown]**

| Agent Type      | Share  | Notes                        |
| --------------- | ------ | ---------------------------- |
| Vehicle         | ~70%   | Dominant class               |
| Pedestrian      | ~25%   | More irregular motion        |
| Cyclist         | ~5%    | Smallest classOur anomaly detection pipeline has two layers — and as Justin just showed, the choice of yaw rate as a key feature comes directly from that correlation analysis.

Layer 1: Isolation Forest. We fit 100 trees on Speed, Acceleration, and YawRate. Each agent receives an anomaly score between 0 and 1. We threshold at the 99th percentile — flagging 46 agents out of 4,538 as anomalous.

Layer 2: Statistical Validation. We build a contingency table of agent type versus anomaly status and run Fisher's Exact Test with Monte Carlo simulation. Result: p = 0.0005. The anomalies are not uniformly distributed — certain agent types are disproportionately flagged. That's signal, not noise.               |
| SDC (AV itself) | —      | Excluded from all analysis   |

**[Show: speed_distribution.png — stacked histogram by agent type]**

- Vehicles appear **slower** than cyclists when stationary agents included — counterintuitive
- Filter to moving agents only → expected ordering restored: Vehicle > Cyclist > Pedestrian
- Speed distribution is right-skewed (skewness ≈ 1.04)

**[Show: correlation_heatmap.png]**

- **YawRate is orthogonal** to all other features — |r| ≤ 0.09 across the board
- Length–Width nearly redundant (r ≈ 0.97)
- Speed–Length weak positive (r ≈ 0.28)

**[Note]**

- `IsInteractive` = proximity-based AV planning flag — **not a safety incident label**

---

## Slide 6 — Methods: Anomaly Detection

**[Header]** Two-Layer Anomaly Detection Pipeline

**[Layer 1]**

### Layer 1 — Isolation Forest

- Features: **Speed, Acceleration, YawRate** — YawRate uncorrelated with all others (|r| ≤ 0.09)
- 100 trees, unsupervised, no labels
- Threshold: **99th percentile → 46 flagged** out of 4,538

**[Layer 2]**

### Layer 2 — Statistical Validation

- Fisher's Exact Test (Monte Carlo, 2,000 replicates)
- **p = 0.0005** — anomalies cluster by agent type, not random noise

---

## Slide 7 — Methods: Motion Prediction

**[Header]** Linear Regression for 5-Second Displacement

$$\Delta d_{5s} = \beta_0 + \beta_1 \cdot \text{Speed} + \beta_2 \cdot \text{Acceleration} + \beta_3 \cdot \text{YawRate} + \beta_4 \cdot V_x + \beta_5 \cdot V_y$$

**[Two conditions]**

|             | Condition A               | Condition B                      |
| ----------- | ------------------------- | -------------------------------- |
| **Dataset** | All agents (n = 6,614)    | Moving only — Speed > 0 (n = 3,196) |
| **Purpose** | Baseline with easy cases  | Honest dynamic evaluation        |

- 80 / 20 train / test split · 5-fold CV · Metrics: **RMSE** · **R²**

---

## Slide 8 — Results

**[Header]** Results

**[Anomaly Detection]**

- **46 flagged** agents out of 4,538 — two categories:
  - **Genuine extremes:** 3 agents at 25–50 m/s (up to 180 km/h, 355 m/s² acceleration)
  - **Sensor artifact:** 17 near-stationary agents with YawRate ≈ ±31 rad/s — physically impossible, numerical instability at near-zero speed
- **IsInteractive = False for all top 20** — kinematic outlier ≠ safety incident

**[Videos — three scenarios shown]**

| Video | Scenario | Speed | Accel | IF Score | Category |
| ----- | -------- | ----- | ----- | -------- | -------- |
| vehicle_01 | `909244b8…` | **49.75 m/s** (180 km/h) | 355.78 m/s² | 0.870 | Genuine extreme |
| anomaly_02 | `5baef16b…` | **24.92 m/s** | 230.99 m/s² | 0.853 | Genuine extreme |
| anomaly_10 | `262112a6…` | < 3 m/s | — | — | Sensor artifact (YawRate ≈ ±31 rad/s) |

**[Feature Importance]**

- Isolation Forest (permutation): **YawRate** dominant — anomaly signal lives in turning behavior
- Regression: **Speed β ≈ 0.95** — displacement is a direct function of velocity; YawRate β ≈ -0.007

**[Regression]**

| Dataset            | Test RMSE  | Test R²   | CV RMSE | CV R²  |
| ------------------ | ---------- | --------- | ------- | ------ |
| With stationary    | **7.44 m** | **0.880** | 7.65 m  | 0.886  |
| Without stationary | 13.15 m    | 0.728     | 10.05 m | 0.860  |

---

## Slide 9 — Discussion & Limitations

**[Header]** What the Results Tell Us

**[Key nuance]**

- **Kinematic outlier ≠ safety incident** — none of the top 20 anomalies were flagged by Waymo as planning-relevant
- Pipeline better described as a **data quality + behavioral characterization tool** than a safety-incident detector

**[Takeaways]**

- Unsupervised detection works and is statistically validatable — Fisher p = 0.0005
- YawRate drives anomaly detection · Speed drives displacement — models answer fundamentally different questions
- ΔR² = −0.15 when removing stationary agents — **benchmarks on mixed populations are inflated**

**[Limitations]**

- Single snapshot per agent — ignores full 20-second trajectory
- Linear regression — pedestrian/cyclist motion may be nonlinear
- Only 300 scenarios — high-speed anomaly tail is underrepresented
- YawRate sensor artifact (±31 rad/s) dominates flagged anomalies — filter needed before production use

---

## Slide 10 — Conclusion & Future Work

**[Header]** Summary & Next Steps

**[Stage 1 — Anomaly Detection]**

- Isolation Forest · Fisher p < 0.001
- Surfaced genuine extremes + systematic sensor artifact

**[Stage 2 — Motion Prediction]**

- Linear regression · R² = 0.88 (all agents) · R² = 0.73 (moving only)
- Speed β ≈ 0.95 — strong classical baseline

**[Future Work]**

1. Filter YawRate at |yaw| < 10 rad/s to remove sensor artifact before anomaly detection
2. Validate flagged agents against `IsInteractive` with appropriate caveats
3. Incorporate full 20-second trajectory features
4. Apply pipeline to full dataset — richer high-speed anomaly tail

> Classical statistical methods deliver **interpretable, statistically validated safety insights** from real-world AV data — without deep learning.

**Code:** github.com/pstepanovum/womd-unsupervised-anomaly-detection

---

## Slide 11 — Q&A

**[Header]** Thank You — Questions?

**Pavel Stepanov** — pas273@miami.edu
**Ramses Loaces** — rxl1238@miami.edu
**Justin Cabrera** — justin.cabrera2718@gmail.com

| Key Result                  | Value                                |
| --------------------------- | ------------------------------------ |
| Agents analyzed             | 4,538 (anomaly) · 6,614 (regression) |
| Anomalies flagged           | 46 (top 1% by IF score)              |
| Fisher's p-value            | 0.0005                               |
| Speed β (regression)        | ≈ 0.95                               |
| Regression R² (all agents)  | 0.880                                |
| Regression R² (moving only) | 0.728                                |
