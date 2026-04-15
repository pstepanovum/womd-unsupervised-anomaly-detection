# Presentation Script — 20 Minutes

## Unsupervised Statistical Learning for Kinematic Anomaly Detection in Autonomous Driving Scenarios

**Authors:** Pavel Stepanov, Ramses Loaces, Justin Cabrera
**Course:** CSC 642 — Spring 2026, University of Miami

---

**Speaker totals — Pavel: 6 min · Ramses: 6 min · Justin: 7 min · Q&A: 1 min**

---

## Slide 1 — Title (1 min) | Speaker: Pavel (Done)

**[Show title slide with all three names]**

> Good afternoon, everyone. My name is Pavel Stepanov, and I'm here with my teammates Ramses Loaces and Justin Cabrera. Today we're presenting our research on unsupervised kinematic anomaly detection in autonomous driving.
>
> Here's the core question that drives our work: **How does an autonomous vehicle recognize that something unusual is happening on the road — without anyone ever labeling what "unusual" looks like?**
>
> That's the problem we tackle. Let's get into it.

---

## Slide 2-3 — Problem & Motivation (2 min) | Speaker: Pavel (Done)

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

## Slide 4-5 — Related Work (2 min) | Speaker: Ramses (Done)

**[Slide: brief comparison table — Kalman Filter / LSTM / Transformer / Isolation Forest / One-Class SVM]**

> Let me quickly place this work in the literature — and get into some of the technical mechanics.
>
> Classical Kalman Filters model motion as a linear dynamical system — fast and interpretable, but they break down under nonlinear behavior. LSTMs and Transformer-based architectures handle sequences well and achieve state-of-the-art prediction accuracy, but they require large labeled datasets and produce black-box outputs.
>
> On the anomaly detection side, two methods dominate: **One-Class SVM** and **Isolation Forest**. OC-SVM maps data into a high-dimensional kernel space and finds the tightest boundary around the normal class — but it's sensitive to the choice of kernel and the nu hyperparameter, which controls the fraction of training points treated as outliers.
>
> **Isolation Forest** works differently. It partitions the feature space by randomly selecting a feature and a split value at each node. The key insight is that anomalous points — being rare and extreme — require _fewer_ splits to isolate. The anomaly score is derived from the average path length across all trees, normalized by the expected path length for a random point: s(x,n) = 2^(−E(h(x)) / c(n)). Scores near 1 signal anomalies; near 0.5 is indistinguishable from normal.
>
> Prior work on the Waymo Open Motion Dataset has focused almost entirely on deep learning prediction pipelines. **Classical statistical baselines applied to the full agent population have been largely ignored.** That's the gap we fill.

---

## Slide 6 — Dataset & Pipeline (2 min) | Speaker: Pavel (Done)

**[Slide: data flow diagram — GCS → Python notebook → CSV → R → Results]**

> We work with the **Waymo Open Motion Dataset v1.3.1** — one of the largest real-world AV datasets available publicly. It provides 20 Hz object tracks over 20-second urban driving segments, with rich metadata: agent type (vehicle, pedestrian, cyclist), bounding boxes, interaction flags, and traffic signals.
>
> Our pipeline has two stages. First, Python running in Google Colab pulls raw TFRecord shards from Waymo's Google Cloud Storage bucket and extracts a tabular feature set for over **18,000 agents** across hundreds of scenarios. We extract the kinematic state at the prediction moment: speed, acceleration, yaw rate, velocity components, and the future 5-second displacement.
>
> That CSV then flows into R, where all statistical modeling happens.
>
> The two analyses use different preprocessing. For anomaly detection, we apply the strictest filter: valid sensor frames only, stationary agents removed, and complete cases across all kinematic and geometric features — leaving **4,538 moving agents**. For regression, the goal is different — we actually want to compare behavior with and without stationary agents — so we start from a more permissive filter: NA removal only, no speed cutoff. That gives us **6,614 agents** including stationary ones, which we then split into two conditions for the comparison.

---

## Slide 7-9 — Dataset Statistics (2 min) | Speaker: Justin (Done)

**[Slide: agent type breakdown table, speed distribution histogram, correlation heatmap]**

> Before diving into the models, let me give you a feel for what this data actually looks like.
>
> In terms of composition: vehicles make up roughly 70% of the dataset, pedestrians around 25%, and cyclists about 5%. The Waymo self-driving car — the SDC — is excluded from all analysis. This class imbalance matters: it affects the anomaly distribution and how we interpret per-class statistics.
>
> **[Point to correlation heatmap]**
>
> The heatmap here shows Pearson correlations across all kinematic features. The key finding: **yaw rate is essentially uncorrelated with everything else** — |r| ≤ 0.09 across the board. Length and width are nearly redundant at r ≈ 0.97. Speed and length share a weak positive relationship.
>
> This independence of yaw rate is not a nuisance — it's exactly what makes it a strong discriminating feature for anomaly detection. It carries information no other variable does. You'll see why that matters in the next slide.
>
> One last note on Waymo's behavioral flags: IsInteractive marks agents relevant to the AV's immediate planning — nearby vehicles at intersections. It is **not** a safety-incident label. That distinction will matter when we look at the results.
>
> **[Point to speed distribution histogram]**
>
> Here's the speed distribution across all three agent types. One thing stands out immediately: vehicles appear _slower_ than cyclists in the raw data — which seems wrong. It's entirely because a large fraction of vehicles are stopped at intersections. Once you filter to moving agents only, the expected ordering snaps back: vehicles fastest, then cyclists, then pedestrians. That's an important preprocessing consideration we carry into both models.

> These results gave us some confidence in the intuition behind our starting data (vehicles moving faster than pedestrians)

---

## Slide 10-11 — Methods: Anomaly Detection (2 min) | Speaker: Ramses (Done)

**[Slide: three-layer pipeline diagram]**

> Our anomaly detection pipeline has two layers — and as Justin just showed, the choice of yaw rate as a key feature comes directly from that correlation analysis.
>
> **Layer 1: Isolation Forest.**
> We fit 100 trees on Speed, Acceleration, and YawRate. Each agent receives an anomaly score between 0 and 1. We threshold at the 99th percentile — flagging **46 agents** out of 4,538 as anomalous.
>
> **Layer 2: Statistical Validation.**
> We build a contingency table of agent type versus anomaly status and run **Fisher's Exact Test** with Monte Carlo simulation. Result: **p = 0.0005**. The anomalies are not uniformly distributed — certain agent types are disproportionately flagged. That's signal, not noise.

---

## Slide 12 — Methods: Motion Prediction (1.5 min) | Speaker: Justin

**[Slide: linear regression formula, two dataset conditions visualized]**

> For motion forecasting we use a linear regression model predicting **5-second future displacement** from the full kinematic state: speed, acceleration, yaw rate, and the signed velocity components Vx and Vy.
>
> We evaluate under **two conditions** — intentionally. Condition A includes all agents: moving and stationary. Condition B restricts to moving agents only, giving us 3,196 observations.
>
> Stationary agents have near-zero displacement — trivially easy to predict and likely to inflate R-squared. The two-condition design lets us directly measure that inflation.
>
> Both models were trained with 5-fold cross-validation and evaluated on a held-out 20% test set.

---

## Slide 13-14 — Results (3.5 min) | Speaker: Justin

**[Slide: three result tables + video clips of flagged anomalous agents]**

> Let's talk about what we found — and I want to start with the anomalies themselves, because the videos tell the story better than any table.

**[Show anomalous agent video clips — ~45 seconds]**

> These are three actual scenario clips from the Waymo dataset, centered on the agents our Isolation Forest flagged as most anomalous. They represent both categories of what we found.

**[Video 1 — vehicle_01, scenario 909244b8]**

> The first video: a vehicle reaching **49.75 m/s — roughly 180 km/h** — with an acceleration reading of 355 m/s². That is a genuine kinematic extreme. In an urban driving scenario, that speed is far outside normal operating range.

**[Video 2 — anomaly_02, scenario 5baef16b]**

> Second video: another vehicle flagged for high acceleration — **24.92 m/s, 230.99 m/s²**. Again a real outlier in speed and dynamics.

**[Video 3 — anomaly_10, scenario 262112a6]**

> Third video — and this one is different. This agent is nearly stationary, but its yaw rate is recorded at exactly **±31 rad/s**. A rotation rate of 31 radians per second is not physically possible for any ground vehicle. This is the sensor artifact — and it accounts for **17 of the top 20** flagged agents. Near-zero speed, always the same magnitude, appearing across vehicles, pedestrians, and cyclists. This is numerical instability in the heading derivative computation when speed approaches zero. The model flagged real structure in the data — it's just that the structure reflects a measurement issue, not a driving incident.
>
> There's one more detail worth highlighting: **none of the top 20 anomalies carry an IsInteractive flag from Waymo**. IsInteractive marks agents that are safety-relevant to the AV's immediate planning. Our most extreme kinematic outliers — the 180 km/h vehicle, the ±31 rad/s artifacts — were not flagged by Waymo as planning-relevant. We'll come back to what that means in discussion, but it's an important caveat to carry into the results.
>
> **[Transition to regression results]**

> For motion prediction: with stationary agents included, test RMSE **7.44 meters**, R-squared **0.880**. Remove stationary agents and RMSE jumps to **13.15 meters**, R-squared drops to **0.728**.
>
> The standardized beta coefficients give us the feature importance for the regression — and the picture could not be clearer. **Speed dominates with β ≈ 0.95**, essentially the entire predictive signal. Acceleration and the velocity components contribute marginally. YawRate is nearly zero at β ≈ -0.007 — it adds almost nothing to displacement prediction.
>
> That contrast is the key finding across both models: **YawRate is the anomaly signal, Speed is the displacement signal.** They're measuring completely different aspects of agent behavior, and the feature importance of each model confirms exactly that. Neither result is surprising given the physics — but having the data validate the physics is precisely the sanity check you want before trusting a pipeline on real-world safety data.

---

## Slide 15 — Discussion & Limitations (2 min) | Speaker: Ramses

**[Slide: two key takeaways + limitations bullet list]**

> Before we get to the takeaways, I want to address what Justin flagged at the end of the results section — because it's the most important nuance in this project.
>
> The pipeline produced statistically validated results: Fisher's p = 0.0005. The numbers look good. But when we actually inspect what was flagged, none of the top 20 anomalies were marked as safety-relevant by Waymo's own IsInteractive flag. That's not a failure of the model — it's a clarification of what the model is actually measuring. **A kinematic outlier is not the same thing as a safety incident.** IsInteractive reflects which agents are in close proximity to the AV during routine planning — adjacent lane vehicles, approaching intersections. Our flagged events — extreme speeds and sensor artifacts — exist outside that frame entirely.
>
> What this means in practice: the pipeline is more accurately described as a **data quality and behavioral characterization tool** than a safety-incident detector. It found genuine extremes in the data, and it surfaced a systematic measurement problem that would have gone unnoticed without it. Both are useful contributions — but they require domain knowledge to interpret correctly. The danger is treating model outputs as ground truth without that context.
>
> With that framing, two broader takeaways.
>
> First, **unsupervised detection works and is statistically validatable**. The Fisher test confirms structure in who gets flagged. Anomalies concentrate in pedestrians and cyclists — consistent with their more irregular urban kinematics.
>
> Second, **dataset composition dominates metrics**. The 15-point R-squared drop when removing stationary agents is not a model failure — it's a dataset artifact. Any evaluation reported on a mixed moving-and-stationary population is inflated. The 0.728 on moving agents is the honest number for dynamic AV deployment.
>
> On limitations: we work from a single snapshot frame per agent, ignoring the full 20-second trajectory. Our regression is linear; pedestrian and cyclist motion may be nonlinear. And we only analyzed 300 scenarios — the genuine high-speed anomaly tail would be richer with the full dataset.

---

## Slide 16 — Conclusion & Future Work (1 min) | Speaker: Pavel

**[Slide: summary bullet points + future directions]**

> To summarize: we built a two-stage statistical pipeline for autonomous driving data.
>
> Stage one: Isolation Forest detects kinematic anomalies unsupervised. Fisher's Exact Test validates them at p < 0.001. Manual inspection reveals both genuine extreme events and a systematic sensor artifact that only became visible through the anomaly detection step.
>
> Stage two: Linear regression predicts 5-second displacement with R-squared of 0.88 — a strong classical baseline.
>
> The result: **interpretable, reproducible statistical methods can deliver meaningful safety-relevant insights from large-scale real-world AV data**, without any deep learning — and with the important caveat that interpreting those insights requires understanding what each model is and isn't measuring.
>
> For future work, three extensions:

> 1. Filter yaw rate at |yaw| < 10 rad/s before anomaly detection, to remove the sensor artifact and surface more meaningful behavioral anomalies.
> 2. Validate flagged agents against Waymo's `IsInteractive` flag as a soft reference label, with appropriate caveats about what that flag actually represents.
> 3. Apply the pipeline to the full dataset — not just 300 scenarios — to get a richer tail of genuine high-speed events.
>
> Our full code is open-source on GitHub.
>
> Thank you — we're happy to take questions.

---

## Slide 17 — Q&A Buffer (1 min) | All

**Anticipated questions and prepared answers:**

**Q: Why Isolation Forest and not One-Class SVM?**

> Isolation Forest scales better to our dataset size (~4,500 agents) and requires no kernel tuning. One-Class SVM is sensitive to the nu hyperparameter and the kernel choice; IF is more robust out of the box and interpretable via feature importance.

**Q: The 99th percentile threshold — isn't that arbitrary?**

> It's a design choice, yes. We chose 1% to ensure only the most extreme kinematic outliers are flagged. The Fisher test validates that this threshold produces statistically meaningful, non-random groups. A lower threshold would give more recall at the cost of precision.

**Q: None of the top 20 were IsInteractive — does that mean the results are useless?**

> No — it means we're measuring something different from what IsInteractive tracks. IsInteractive is a proximity-based planning flag for routine AV decisions. Our pipeline finds kinematic extremes that fall outside normal operating ranges and surfaces data quality issues. Those are complementary, not competing, signals.

**Q: Why linear regression and not something more powerful for prediction?**

> Intentionally. We wanted to establish a classical baseline first. An R-squared of 0.88 from a linear model means there's strong linear signal in the kinematic features. Neural approaches should be benchmarked against this, not introduced without first knowing what simpler methods can already achieve.

**Q: How generalizable are these results beyond Waymo?**

> The kinematic features we use — speed, acceleration, yaw rate — are universal to any tracked agent. The pipeline is dataset-agnostic. The specific thresholds and model weights would need retraining, but the methodology transfers directly. The sensor artifact finding (±31 rad/s at near-zero speed) may also appear in other datasets using the same heading derivative computation.

---
