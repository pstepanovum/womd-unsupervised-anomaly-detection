# Unsupervised Statistical Learning for Kinematic Anomaly Detection in Autonomous Driving

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![R 4.0+](https://img.shields.io/badge/R-4.0+-276DC3.svg)](https://www.r-project.org/)
[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](docs/paper/Stepanov_Loaces_Cabrera_KinematicAnomalyDetection.pdf)

## Abstract

This project applies an unsupervised statistical learning framework to the **Waymo Open Motion Dataset (WOMD v1.3.1)** to identify high-risk and anomalous driving behaviors. We extract kinematic features from over 18,000 traffic agent observations and benchmark three unsupervised anomaly detectors — **Isolation Forest**, **One-Class SVM**, and **Local Outlier Factor (LOF)** — on a filtered cohort of 4,538 moving agents. Detectors are evaluated via Fisher's Exact Test for agent-type association and precision against Waymo's `IsInteractive` ground-truth label. A complementary linear regression task predicts 5-second future displacement from the kinematic state vector.

## Authors

**University of Miami — Department of Computer Science**

- Pavel Stepanov (pas273@miami.edu)
- Ramses Loaces (rxl1238@miami.edu)
- Justin Cabrera (justin.cabrera2718@gmail.com)

## Key Findings

- **One-Class SVM is the strongest of the three baselines.** It ties Isolation Forest on agent-type association (Fisher *p* = 0.0005) and uniquely recovers a Waymo-flagged interactive agent (precision = 0.022 vs. 0.000).
- **Yaw rate is the most distinct kinematic signal** (*r* ≈ 0.02 with all other features), which is what drives the Isolation Forest scores.
- **Most top-ranked anomalies are sensor artifacts:** near-stationary agents register implausible ±31 rad/s yaw rates (≈5 rotations/sec), indicating numerical instability in heading derivatives at low speed. Only ~3 of the top 20 flagged agents are genuinely unusual events (e.g., one vehicle at 49.75 m/s with 355 m/s² acceleration).
- **Linear regression for 5-second displacement reaches *R²* = 0.880** (RMSE = 7.44 m) when stationary agents are included. Restricting to moving agents only — the more honest operational estimate — drops this to *R²* = 0.728. Speed dominates (*β* = 0.95); yaw rate contributes essentially nothing.

### Detector Comparison (k = 46 flagged, N = 4,538)

| Method            | Fisher *p* | Precision (`IsInteractive`) | % Pedestrian |
|-------------------|------------|-----------------------------|--------------|
| Isolation Forest  | **0.0005** | 0.000                       | 0.348        |
| One-Class SVM     | **0.0005** | **0.022**                   | **0.413**    |
| LOF (k = 20)      | 0.0015     | 0.000                       | 0.348        |

Pairwise Jaccard overlap: *J*(IF, OC-SVM) = 0.51, *J*(IF, LOF) = 0.46, *J*(OC-SVM, LOF) = 0.31.

## Dataset

The analysis uses the **Waymo Open Motion Dataset (v1.3.1)**. Raw TFRecord files (20 Hz object tracks over 20-second segments) are processed into a structured tabular dataset.

- **Raw size:** 18,311 agent observations across multiple scenarios.
- **Anomaly detection cohort:** *N* = 4,538 (moving agents, no missing predictors, SDC removed).
- **Regression cohorts:** *N* = 6,614 (with stationary), *N* = 3,196 (moving only).
- **Predictors:** Speed (m/s), Acceleration (m/s²), Yaw Rate (rad/s), Velocity components (V_x, V_y), Length, Width.
- **Validation labels:** `IsInteractive` (Waymo interaction flag), `FutureDist_5s` (5-second Euclidean displacement target).

## Methodology

1. **Feature extraction (Python).** Parse `scenario_pb2.Scenario` protos; extract kinematic state at *t* = 0, the previous frame for derivatives (Δ*t* = 0.1 s, π-wrapped for yaw), and *t* = 5 s for the regression target.
2. **Preprocessing (R).** Drop SDC tracks, require `Valid_Current = True`, drop NA across kinematic predictors. Encode `AgentType` as a factor. Standardize features. Split 80/20 train/test.
3. **Anomaly detection (R).** Fit Isolation Forest (100 trees), One-Class SVM (RBF, *ν* = 0.01), and LOF (*k* = 20) on the same standardized matrix with a fixed *k* = 46 flagged budget (≈1% contamination).
4. **Evaluation.** Fisher's Exact Test (Monte Carlo, 2,000 replicates) for agent-type association; precision against `IsInteractive`; pairwise Jaccard overlap of flagged sets; qualitative inspection of top-20 anomalies.
5. **Motion prediction (R).** Linear regression for 5-second displacement in two regimes (with / without stationary), 5-fold CV, evaluated by RMSE and *R²* on the held-out test set.

## Repository Structure

```
.
├── docs/
│   ├── paper/                  IEEE-format final paper (main.tex, named PDF, figures, anomaly_comparison.csv)
│   ├── presentation/           Slides, script, fonts
│   └── archive/                Earlier submissions (proposal, midterm)
├── src/python/                 Colab notebook for WOMD TFRecord extraction
├── src/r/                      R Markdown statistical pipeline
├── results/                    Top-anomaly listings, key model results, videos
└── data/                       Local data location (CSV not committed)
```

## Running the Code

### Python (Data Extraction — Google Colab)

`src/python/waymo.ipynb` is designed for **Google Colab**. It requires:

- Google Cloud authentication (`auth.authenticate_user()`)
- Access to `gs://waymo_open_dataset_motion_v_1_3_1/`
- `waymo-open-dataset-tf-2-12-0` and TensorFlow installed via pip

Output: `waymo_final_dataset.csv`, downloaded to local disk.

### R (Statistical Analysis)

Open `src/r/ProjectSubmission.Rmd` in RStudio and update the dataset path to point to your local `waymo_final_dataset.csv`. Required packages: `tidyverse`, `caret`, `isotree`, `e1071`, `dbscan` (for LOF), `randomForest`.

Render the report:

```r
rmarkdown::render("src/r/ProjectSubmission.Rmd")
```

### Paper

Build the IEEE-format paper:

```bash
cd docs/paper
latexmk -pdf main.tex
```

## Requirements

- Python 3.10+ — TensorFlow 2.12, `waymo-open-dataset-tf-2-12-0`, Matplotlib, Pandas
- R 4.0+ — `tidyverse`, `caret`, `isotree`, `e1071`, `dbscan`, `randomForest`

## License

MIT — see [LICENSE](LICENSE).
