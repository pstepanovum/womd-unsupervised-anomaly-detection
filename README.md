# Unsupervised Statistical Learning for Kinematic Anomaly Detection in Autonomous Driving

## Abstract

This project applies unsupervised statistical learning techniques to the Waymo Open Motion Dataset (WOMD v1.3.1) to identify high-risk and anomalous driving behaviors. By extracting kinematic features (velocity, acceleration, yaw rate) from over 10,000 traffic agents, we employ Principal Component Analysis (PCA) for dimensionality reduction and compare Mahalanobis Distance against One-Class Support Vector Machines (SVM) for outlier detection. The identified anomalies are validated against ground-truth interaction flags provided by Waymo to assess the correlation between statistical outliers and safety-critical events.

## Authors

**University of Miami - Department of Computer Science**

- Pavel Stepanov (pas273@miami.edu)
- Ramses Loaces (rxl1238@miami.edu)
- Justin Cabrera (justin.cabrera2718@gmail.com)

## Dataset

The analysis utilizes the **Waymo Open Motion Dataset (v1.3.1)**. Raw TFRecord files were processed to extract a structured tabular dataset.

- **Sample Size:** N ≈ 300 scenarios (approx. 10,000 agents).
- **Input Features (X):**
  - Kinematics: Speed (m/s), Acceleration (m/s²), Yaw Rate (rad/s).
  - Geometry: Length, Width, Height (m).
- **Validation Labels:**
  - `IsInteractive`: Binary flag indicating complex interaction with the autonomous vehicle.
  - `FutureDist_5s`: Euclidean displacement over a 5-second horizon.

## Methodology

The statistical pipeline consists of four stages:

1. **Preprocessing & Feature Engineering:**
   - Exclusion of autonomous vehicle tracks and invalid sensor data.
   - Derivation of Lateral G-Force and Jerk metrics.
   - Z-score standardization of continuous variables.

2. **Dimensionality Reduction:**
   - Principal Component Analysis (PCA) to project high-dimensional kinematic data into lower-dimensional latent space for visualization.

3. **Anomaly Detection:**
   - **Parametric:** Multivariate outlier detection using Mahalanobis Distance.
   - **Non-Parametric:** One-Class Support Vector Machine (SVM) with Radial Basis Function (RBF) kernel.

4. **Validation:**
   - Statistical correlation analysis between detected outliers and Waymo interaction flags.
   - Qualitative validation via scenario reconstruction (video generation).

## Repository Structure

- `data/`: Contains scripts for raw data extraction and processed CSV files.
- `src/`: Source code for statistical models (R) and data processing (Python).
- `results/`: Output figures, PCA biplots, and anomaly visualization videos.
- `docs/`: LaTeX source files for the final project report.

## Requirements

- Python 3.10+ (TensorFlow, Matplotlib, Pandas)
- R 4.0+ (stats, e1071, ggplot2)

## License

This project is licensed under the MIT License.
