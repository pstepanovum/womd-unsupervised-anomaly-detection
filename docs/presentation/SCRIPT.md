# Presentation Script — 20 Minutes
## Unsupervised Statistical Learning for Kinematic Anomaly Detection in Autonomous Driving Scenarios

**Authors:** Pavel Stepanov, Ramses Loaces, Justin Cabrera
**Course:** CSC 642 — Spring 2026, University of Miami

---

## Timing Overview

| Segment | Speaker | Duration | Cumulative |
|---|---|---|---|
| 1. Title & Hook | Pavel | 1 min | 1 min |
| 2. Problem & Motivation | Pavel | 2.5 min | 3.5 min |
| 3. Related Work | Ramses | 2 min | 5.5 min |
| 4. Dataset & Pipeline | Justin | 2 min | 7.5 min |
| 5. Methods: Anomaly Detection | Ramses | 3 min | 10.5 min |
| 6. Methods: Motion Prediction | Justin | 2 min | 12.5 min |
| 7. Results | Pavel | 3.5 min | 16 min |
| 8. Discussion & Limitations | Ramses | 2 min | 18 min |
| 9. Conclusion & Future Work | Pavel | 1 min | 19 min |
| 10. Questions (buffer) | All | 1 min | 20 min |

---

## Slide 1 — Title (1 min) | Speaker: Pavel

**[Show title slide with all three names]**

> Good [morning/afternoon], everyone. My name is Pavel Stepanov, and I'm here with my teammates Ramses Loaces and Justin Cabrera. Today we're presenting our research on unsupervised kinematic anomaly detection in autonomous driving.
>
> Here's the core question that drives our work: **How does an autonomous vehicle recognize that something unusual is happening on the road — without anyone ever labeling what "unusual" looks like?**
>
> That's the problem we tackle. Let's get into it.

---

## Slide 2 — Problem & Motivation (2.5 min) | Speaker: Pavel

**[Slide: image of urban traffic / AV sensor view, bullet points on the long tail problem]**

> Autonomous vehicles are getting very good at handling the everyday: steady highway cruising, predictable lane-following, routine intersections. But safety-critical incidents live in the **long tail** of driving distributions — sudden braking, aggressive lane changes, a pedestrian running into the street.
>
> The challenge is that these events are rare by definition, so **labeled examples are scarce**. You can't train a supervised classifier without labels. And the definition of "dangerous" is subjective — a 30 km/h speed is fine on a highway, alarming in a crosswalk.
>
> This is exactly where **unsupervised learning** shines: we don't need labels. We let the data tell us what's normal, and then flag what falls outside that.
>
> We had two concrete goals:
> 1. Detect anomalous motion patterns from kinematic signals alone — no human annotation.
> 2. Predict where an agent will be in 5 seconds using only its current kinematic state.
>
> Both are essential for an AV that needs to proactively plan around unexpected behavior.

---

## Slide 3 — Related Work (2 min) | Speaker: Ramses

**[Slide: brief comparison table — Kalman Filter / LSTM / Transformer / Isolation Forest / One-Class SVM]**

> Let me quickly situate this work in the literature.
>
> The trajectory forecasting space has evolved from classical Kalman Filters to deep sequence models like LSTMs and Transformer-based architectures. These achieve impressive accuracy, but they require massive labeled datasets and are difficult to interpret.
>
> On the anomaly detection side, **Isolation Forest** and **One-Class SVM** remain the go-to methods in practice. They're scalable, they make minimal distributional assumptions, and they're explainable — you can trace why a specific agent was flagged.
>
> Prior work using the Waymo Open Motion Dataset has focused almost entirely on deep learning prediction pipelines. **Classical statistical baselines over the full agent population have been largely ignored.**
>
> Our contribution is exactly that: interpretable, reproducible statistical methods applied at scale — establishing a baseline that future neural approaches should be measured against.

---

## Slide 4 — Dataset & Pipeline (2 min) | Speaker: Justin

**[Slide: data flow diagram — GCS → Python notebook → CSV → R → Results]**

> We work with the **Waymo Open Motion Dataset v1.3.1** — one of the largest real-world AV datasets available publicly. It provides 20 Hz object tracks over 20-second urban driving segments, with rich metadata: agent type (vehicle, pedestrian, cyclist), bounding boxes, interaction flags, and traffic signals.
>
> Our pipeline has two stages. First, Python running in Google Colab pulls raw TFRecord shards from Waymo's Google Cloud Storage bucket and extracts a tabular feature set for over **18,000 agents** across hundreds of scenarios. We extract the kinematic state at the prediction moment: speed, acceleration, yaw rate, velocity components, and the future 5-second displacement.
>
> That CSV then flows into R, where all statistical modeling happens.
>
> After preprocessing — removing the autonomous SDC vehicle, filtering for valid sensor frames, and dropping missing values — we end up with **4,538 agents** for anomaly detection and **6,614 agents** for regression.

---

## Slide 5 — Methods: Anomaly Detection (3 min) | Speaker: Ramses

**[Slide: diagram of Isolation Forest concept, then pipeline: IF → threshold → Fisher's Test → RF classifier]**

> Our anomaly detection pipeline has three layers.
>
> **Layer 1: Isolation Forest.**
> The intuition is elegant: anomalous points are rare and different, so they get isolated in fewer random tree splits than normal points. We fit a forest of 100 trees on three kinematic features — speed, acceleration, and yaw rate. Each agent gets an anomaly score between 0 and 1. We flag the top 1% — the 99th percentile — as anomalous. That gives us **46 flagged agents out of 4,538**.
>
> **Layer 2: Statistical Validation.**
> Are these 46 flags just random noise, or is there real structure? We build a contingency table: agent type (vehicle, pedestrian, cyclist) versus anomaly status (yes/no), and run **Fisher's Exact Test** with Monte Carlo simulation. The result is **p = 0.0005**. The anomalies are not uniformly distributed — certain agent types are disproportionately flagged. That's not noise, that's signal.
>
> **Layer 3: Supervised Characterization.**
> We use the Isolation Forest labels as pseudo-labels to train a **Random Forest classifier**, using bounding box dimensions (length, width) and yaw rate as features. Because we have a severe 99:1 class imbalance, we apply down-sampling to balance the training set. Five-fold cross-validation optimizing AUC-ROC. The classifier learns to distinguish anomalous from normal agents — it achieved a cross-validated AUC of **0.887**.

---

## Slide 6 — Methods: Motion Prediction (2 min) | Speaker: Justin

**[Slide: linear regression formula, two dataset conditions visualized]**

> For our second objective — short-term motion forecasting — we use a linear regression model.
>
> The target is **5-second future displacement**: how far in meters will this agent travel in the next five seconds? The predictors are the full kinematic state: speed, acceleration, yaw rate, and the signed velocity components Vx and Vy.
>
> We evaluate the model under **two conditions**:
> - Condition A includes all agents — moving and stationary.
> - Condition B restricts to moving agents only (speed > 0), which gives us 3,196 observations.
>
> This split is intentional. Stationary agents have near-zero displacement, so they're trivially easy to predict. Including them inflates R-squared. We want to see what the model actually does under dynamic conditions.
>
> Both models were trained with 5-fold cross-validation and evaluated on a held-out 20% test set.

---

## Slide 7 — Results (3.5 min) | Speaker: Pavel

**[Slide: three tables from the paper — speeds by agent type, RF performance, regression performance]**

> Let's talk about what we found.
>
> **Descriptive analysis first.**
> When you look at raw mean speeds including stationary agents, vehicles appear slower than cyclists — counterintuitive. That's because a huge proportion of vehicles are stopped at intersections. Once we restrict to moving agents, the expected ordering is restored: vehicles at 6.74 m/s, cyclists at 3.92 m/s, pedestrians at 1.07 m/s. This is an important preprocessing lesson — **stationary agents can systematically bias your statistics**.
>
> **Anomaly Detection.**
> Isolation Forest flagged 46 agents — exactly 1% by design. Fisher's Exact Test: p = 0.0005. The anomalies cluster in specific agent types. Our Random Forest classifier, trained on those pseudo-labels, achieved:
> - Cross-validated AUC-ROC of **0.887**
> - Sensitivity of 0.836, Specificity of 0.839 on training
> - On the held-out test set: accuracy of 79.5%, balanced accuracy of 73.1%
>
> That's strong performance given that we're training on noisy, unsupervised pseudo-labels rather than ground truth.
>
> **Motion Prediction.**
> Here's the most striking result. With stationary agents included: test RMSE of **7.44 meters**, R-squared of **0.880**. A linear model. Five features. That's a very strong fit.
>
> Remove stationary agents and the picture changes: RMSE jumps to **13.15 meters**, R-squared drops to 0.728. The model is still decent, but the drop of 15 points in R-squared makes clear that stationary agents were carrying a lot of the predictive weight. For a safety-critical deployment evaluating dynamic scenarios, the 0.728 is the honest number.

---

## Slide 8 — Discussion & Limitations (2 min) | Speaker: Ramses

**[Slide: two key takeaways + limitations bullet list]**

> Two key takeaways from the results.
>
> First, **unsupervised anomaly detection works** — and we can validate it statistically. The Fisher test tells us the Isolation Forest isn't just picking random points. The structure of anomalies correlates with agent type in a meaningful way. Pedestrians and cyclists exhibit more irregular kinematics, and the model captures that.
>
> Second, **dataset composition matters enormously**. The 15-point drop in R-squared when removing stationary agents is not a model failure — it's a dataset artifact. Any evaluation metric reported on a mixed moving-and-stationary population is likely inflated. This has direct implications for how AV research benchmarks should be constructed.
>
> On limitations: the 46 anomaly labels we use to train the Random Forest are pseudo-labels — they come from an unsupervised score, not verified ground truth. The test specificity of 0.667 on the anomalous class reflects that noise. We also work from a single snapshot frame per agent, ignoring the full 20-second trajectory — which is a significant loss of information. And our regression model is linear; the true dynamics of pedestrian motion especially may be nonlinear.

---

## Slide 9 — Conclusion & Future Work (1 min) | Speaker: Pavel

**[Slide: summary bullet points + future directions]**

> To summarize: we built a two-stage statistical pipeline for autonomous driving data.
>
> Stage one: Isolation Forest detects kinematic anomalies unsupervised. Fisher's Exact Test validates them at p < 0.001. A Random Forest characterizes them at 0.887 AUC.
>
> Stage two: Linear regression predicts 5-second displacement with R-squared of 0.88 — a strong classical baseline.
>
> The result: **interpretable, reproducible statistical methods can deliver meaningful safety-relevant insights from large-scale real-world AV data**, without any deep learning.
>
> For future work, we plan three extensions:
> 1. Incorporate temporal trajectory features over the full 20-second window.
> 2. Validate detected anomalies against Waymo's built-in `IsInteractive` ground-truth flags.
> 3. Explore Mahalanobis distance scoring as an alternative to Isolation Forest.
>
> Our full code is open-source on GitHub.
>
> Thank you — we're happy to take questions.

---

## Slide 10 — Q&A Buffer (1 min) | All

**Anticipated questions and prepared answers:**

**Q: Why Isolation Forest and not One-Class SVM?**
> Isolation Forest scales better to our dataset size (~4,500 agents) and requires no kernel tuning. One-Class SVM is sensitive to hyperparameter choice; IF is more robust out of the box and interpretable via feature importance.

**Q: The 99th percentile threshold — isn't that arbitrary?**
> It's a design choice, yes. We chose 1% to ensure only the most extreme kinematic outliers are flagged. The Fisher test validates that this threshold produces statistically meaningful, non-random groups. A lower threshold would give more recall at the cost of precision.

**Q: Why linear regression and not something more powerful for prediction?**
> Intentionally. We wanted to establish a classical baseline first. An R-squared of 0.88 from a linear model means there's strong linear signal in the kinematic features. Neural approaches should be benchmarked against this, not introduced without first knowing what simpler methods can already achieve.

**Q: How generalizable are these results beyond Waymo?**
> The kinematic features we use — speed, acceleration, yaw rate — are universal to any tracked agent. The pipeline is dataset-agnostic. The specific thresholds and model weights would need retraining, but the methodology transfers directly.

---

## Delivery Tips

- **Pace:** The script is calibrated for a natural speaking pace (~130 words/minute). Don't rush the results section — the tables need time to land.
- **Transitions:** Each speaker should briefly introduce the next: *"I'll hand it over to Ramses who'll cover our methods."*
- **Visuals:** Point at specific numbers when citing them (p = 0.0005, R² = 0.88). Don't just read the table — explain what the number means.
- **Practice handoffs:** The three handoff moments (Pavel → Ramses, Ramses → Justin, Justin → Ramses, Ramses → Pavel) should feel smooth. Rehearse them.
- **Time check:** At the 10-minute mark you should be finishing Slide 5 (anomaly detection methods). If behind, compress Slide 8 (discussion).
