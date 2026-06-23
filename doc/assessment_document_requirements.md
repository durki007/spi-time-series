# Assessment Document Requirements (Milestone 6)

**Project:** Time-Series-Enhanced Predictive Process Mining
**Course:** Supervised Process Intelligence Praktikum SS 2026 — RWTH Aachen (Dr. A. Berti)
**Deliverable:** Milestone 6 Assessment Document
**Source:** *Testing & Assessment Document* (Berti, SS 2026)

---

## Conventions

These requirements describe what the **assessment document** must contain. They reuse the
project's existing template:

- **ID** — `REQ-AD-NN` (Assessment Document namespace).
- **Type** — Content / Evidence / Reproducibility.
- **Text** — "The assessment document shall …".
- **Acceptance criteria** — verifiable conditions that satisfy the requirement.
- **Traces** — links to existing project requirements (`REQ-DP/ME/IT/RE/DOC`) and report sections.

The assessment document is distinct from the README (how to install/run) and the architecture
documentation (how modules and data flow are organised). It answers a single central question:
**how well does the implemented approach solve the assigned problem, where does it work, and where
does it not?** Reporting only what was implemented is not sufficient — the document must report what
was *learned* from evaluating it.

The driving project question is: **does adding aligned time-series features improve predictive
performance over an event-log-only baseline?** The assessment is organised around answering it.

---

## 1. Project Goal and Prediction Task

### REQ-AD-01 — Main project question
- **Type:** Content
- **Text:** The assessment document shall state the main project question (log-only vs.
  log + aligned time-series features) and frame all results as evidence for or against it.
- **Acceptance criteria:**
  - The question is stated explicitly in the introduction.
  - Each results subsection is tied back to the question.
  - The document reports findings, not only implemented functionality.
- **Traces:** REQ-ME-01, REQ-RE-01

### REQ-AD-02 — Prediction unit and target
- **Type:** Content
- **Text:** The assessment document shall define what one row of the supervised dataset
  represents and how each target is constructed.
- **Acceptance criteria:**
  - Prediction unit defined: one case-prefix (single running-case prefix) per row.
  - Regression target defined with unit: remaining time in hours,
    `y_{c,k} = t_complete(c) − t_last(c,k)`.
  - Classification target defined: case outcome (e.g. accepted vs. not), with every prefix
    receiving the final case label while features use only information up to that prefix.
  - Filtering rules documented: incomplete cases, very short prefixes, outliers, missing labels.
- **Traces:** REQ-DP-03, REQ-ME-01

---

## 2. Data and Labels

### REQ-AD-03 — Data description (minimum information)
- **Type:** Content
- **Text:** The assessment document shall identify the source and version of every dataset and
  report minimum descriptive statistics.
- **Acceptance criteria:**
  - Dataset identified: BPI Challenge 2017 (XES, Dutch financial institution loan applications),
    with citation.
  - Reported: number of cases (~31,500), number of events (~1.2M), number of activities,
    time span, and case-length distribution.
  - Attributes used as features or as analysis groups are listed.
  - Time-series signals described: derivation, resampling frequency, missing-value handling,
    and alignment strategy to prefix timestamps.
- **Traces:** REQ-DP-01, REQ-DP-02, REQ-DP-05

### REQ-AD-04 — Prefix construction, schema, and example
- **Type:** Content
- **Text:** The assessment document shall describe prefix generation, the leakage-safe feature
  rule, the expected input schema, and point to a tiny valid example log.
- **Acceptance criteria:**
  - Prefix-generation strategy stated (growing prefix / fixed-length, max length, sampling rule).
  - Explicit statement that prefix features contain only information available up to the cutoff;
    time-series features are extracted from the full raw log first, then aligned to the cutoff.
  - Minimal schema listed: case id, activity, timestamp, target attributes, optional
    time-series keys; supported formats (.xes / .csv / .parquet).
  - A small committed example log (`tests/data/`) is referenced.
- **Traces:** REQ-DP-03, REQ-DP-06, REQ-IT-03

---

## 3. Implementation Summary

### REQ-AD-05 — Pipeline and configuration summary
- **Type:** Content
- **Text:** The assessment document shall summarise the implemented pipeline, modules,
  configuration mechanism, and prototype at a level sufficient to interpret the experiments.
- **Acceptance criteria:**
  - Modules summarised (`config`, `data`, `preprocessing`, `features`, `models`, `pipeline`,
    `evaluation`) and the fixed-width aggregation encoding with a single pooled model is described.
  - Configuration location identified (centralised config); model families listed
    (Ridge, RandomForest, HistGradientBoosting).
  - Clearly separated from README and architecture documentation.
- **Traces:** REQ-IT-01, REQ-ME-06, REQ-DOC-01, REQ-DOC-02

---

## 4. Experimental Design

### REQ-AD-06 — Leakage-safe splitting
- **Type:** Content
- **Text:** The assessment document shall describe a leakage-safe split reflecting the
  operational setting and state the exact split.
- **Acceptance criteria:**
  - Chronological (time-based) split by case time, or a group split keeping all prefixes of a
    case together; no random row split of prefixes.
  - Roles separated: train (fit), validation (tune hyperparameters/thresholds and feature
    selection), test (single final estimate).
  - Exact split dates / percentages / case counts stated.
  - The chronological nature is evidenced (e.g. dummy baseline yields negative R² because the
    train-period mean drifts from the test-period mean).
- **Traces:** REQ-ME-05

### REQ-AD-07 — Baselines and competing approaches
- **Type:** Content
- **Text:** The assessment document shall compare the main approach against meaningful baselines.
- **Acceptance criteria:**
  - Regression baselines: training-set mean/median (dummy) and a log-only model.
  - Classification baselines: majority class and/or logistic regression, plus a log-only model.
  - The central comparison reported: log-only vs. log + time-series features.
  - Any case where the enhanced model does not improve is reported and explained, not omitted.
- **Traces:** REQ-RE-01, REQ-ME-02, REQ-ME-03

### REQ-AD-08 — Parameters and configuration
- **Type:** Reproducibility
- **Text:** The assessment document shall make result-affecting parameters visible and state
  which were varied and which were fixed.
- **Acceptance criteria:**
  - Listed: prefix lengths/windows, thresholds, selected feature sets, model hyperparameters,
    random seeds.
  - Statement of which hyperparameter searches were run (e.g. RandomizedSearchCV) and which
    models used defaults (e.g. HGB with no search).
  - Parameters traceable to a single config source.
- **Traces:** REQ-IT-01

### REQ-AD-09 — Evaluation-set coverage
- **Type:** Content
- **Text:** The assessment document shall show that the test set covers the situations the system
  must handle.
- **Acceptance criteria:**
  - Variation covered where possible: short vs. long cases, frequent vs. rare variants,
    early vs. late prefixes, and low-load vs. high-load periods (relevant for time-series signals).
  - Overclaiming avoided where coverage is thin; qualitative analysis used as appropriate.

---

## 5. Results

### REQ-AD-10 — Prediction metrics
- **Type:** Evidence
- **Text:** The assessment document shall report task-appropriate metrics with units, on the
  held-out test set, consistent across compared models.
- **Acceptance criteria:**
  - Regression: weighted MAE and weighted RMSE (in hours) plus weighted R²; transformation
    handling documented (e.g. log1p with Duan's smearing for retransformation).
  - Classification: AUC (primary), F1, Brier score, balanced accuracy.
  - Units of target and error values stated explicitly.
- **Traces:** REQ-ME-04

### REQ-AD-11 — Evaluation by prefix length
- **Type:** Evidence
- **Text:** The assessment document shall report performance as a function of prefix length and
  define the prefix groups used.
- **Acceptance criteria:**
  - Early / middle / late prefix groups defined for this log.
  - The point at which the model becomes useful enough for an operational decision is discussed.
  - Long-prefix behaviour analysed, including end-of-log right-censoring contaminating both the
    time-series feature and the regression target in the test period.
- **Traces:** REQ-RE-01, REQ-ME-07

### REQ-AD-12 — Time-series-enhanced analysis
- **Type:** Evidence
- **Text:** The assessment document shall analyse under which conditions the time-series signal
  helps and report its overhead.
- **Acceptance criteria:**
  - Time-series preprocessing documented: resampling frequency, missing-value handling,
    lag/sliding-window statistics, synchronisation with prefix timestamps.
  - Conditions analysed: high-load periods, specific activities, early vs. late prefixes,
    prediction horizons.
  - Group-level ablation reported (log-based vs. time-series feature blocks) rather than
    per-column permutation importance on correlated groups.
  - Negative findings reported and explained (e.g. the active-case-count block degrading
    weighted R² on most prefix lengths due to concept drift / covariate shift on the
    chronological split).
  - Computational and implementation overhead reported.

### REQ-AD-13 — Plots
- **Type:** Evidence
- **Text:** The assessment document shall include interpreted plots with labelled axes and units.
- **Acceptance criteria:**
  - At least: performance vs. prefix length; prediction-error distribution; predicted-vs-actual
    (regression); ROC/PR curves (classification); feature-importance / SHAP summary; runtime plot.
  - Axes labelled with units; linear vs. log scales justified; every plot captioned and interpreted.

### REQ-AD-14 — Tables with takeaways
- **Type:** Evidence
- **Text:** Every important table shall be accompanied by a one-sentence takeaway, a plot where
  patterns matter, and the configuration that produced it.
- **Acceptance criteria:**
  - No standalone tables without interpretation.
  - Bad results explained, not hidden.

### REQ-AD-15 — Runtime and resource measurements
- **Type:** Evidence
- **Text:** The assessment document shall measure time-consuming stages separately and report the
  environment.
- **Acceptance criteria:**
  - Timed stages: loading/preprocessing, prefix generation, feature extraction,
    time-series alignment, model training, single-prefix prediction.
  - Machine/environment reported.
  - Statement on whether the prototype is interactive or offline-batch only.

### REQ-AD-16 — Feature attribution
- **Type:** Evidence
- **Text:** The assessment document shall quantify the relative contribution of log-based vs.
  time-series feature groups.
- **Acceptance criteria:**
  - TreeSHAP (`shap.TreeExplainer`) used for tree models; standardized coefficients for Ridge.
  - Mean absolute SHAP values summed per feature group to compare log vs. time-series contribution.
- **Traces:** REQ-RE-02

---

## 6. Testing Evidence

### REQ-AD-17 — Testing strategy
- **Type:** Evidence
- **Text:** The assessment document shall report the testing strategy and its outcomes.
- **Acceptance criteria:**
  - Combination of unit, integration, manual, and sanity-check tests described.
  - For each: what was tested, what data was used, how to run it, pass/fail status, known gaps.
  - Tests runnable via a single command (e.g. `pytest`).
- **Traces:** REQ-IT-03

### REQ-AD-18 — Unit-test targets
- **Type:** Evidence
- **Text:** The assessment document shall evidence unit tests covering the core logic.
- **Acceptance criteria:**
  - Covered: label construction on a tiny synthetic log; prefix generation
    (count, ordering, case identity); split (no case overlap, correct time order);
    feature matrix (columns, shapes, no unexpected missing values); model-training smoke test;
    metric computation; time-series feature-extraction smoke test.
  - Test data small enough to verify by inspection; meaningful checks preferred over
    file-existence checks.
- **Traces:** REQ-IT-03

### REQ-AD-19 — Expert/debug inspection
- **Type:** Evidence
- **Text:** The assessment document shall provide evidence the team inspected its own system's
  behaviour.
- **Acceptance criteria:**
  - Examples of generated prefixes and labels, feature vectors for small cases, and predictions
    on easy synthetic examples shown.
  - Behavioural sanity checked: similar cases give similar outputs; clearly different cases give
    meaningfully different outputs.
  - Known bugs documented (e.g. `WaitingStateFeatures` matching `"O_Sent"` while actual event
    names are `"O_Sent (mail and online)"` / `"O_Sent (online only)"`), with their effect on results.

---

## 7. Reproducibility, Prototype, and Deployment

### REQ-AD-20 — Reproducibility contract
- **Type:** Reproducibility
- **Text:** Another person shall be able to reproduce the main results from the repository.
- **Acceptance criteria:**
  - Dependency spec present (`pyproject.toml` / Poetry) with exact Python and library versions.
  - Data placement or download instructions provided; placeholder structure if data is not committed.
  - One command or one notebook path runs the full experiment; fixed seeds where meaningful.
  - Generated metrics and figures saved by the scripts; no hidden manual path edits.
- **Traces:** REQ-IT-01, REQ-IT-04

### REQ-AD-21 — Prototype evidence
- **Type:** Evidence
- **Text:** The assessment document shall demonstrate the pipeline on at least one realistic
  new-data scenario.
- **Acceptance criteria:**
  - Scenario shown: load trained artifact → build features for a new case/prefix →
    produce prediction → print/visualise result.
  - Acceptable form: CLI script, clean notebook demo section, or small service.

### REQ-AD-22 — Deployment sanity check
- **Type:** Reproducibility
- **Text:** The assessment document shall state what execution check was performed on an
  environment other than the author's laptop.
- **Acceptance criteria:**
  - At least one fresh-environment run described.
  - Exact commands documented if Docker / Compose / a web service is used.

---

## 8. Critical Evaluation and Retrospective

### REQ-AD-23 — Critical evaluation
- **Type:** Content
- **Text:** The assessment document shall interpret results, not only present them.
- **Acceptance criteria:**
  - Performance dependence explained for: prefix length, selected features, model type and
    hyperparameters, and dataset period.
  - Stated explicitly: strong points, criticalities and failure modes, and the improvements that
    would be tried next.
  - Model-choice analysis included (e.g. HGB/RF convergence given no HGB search and a low-R²,
    elapsed-time-dominated target; feature signal as the binding constraint, not model choice).

### REQ-AD-24 — Threats to validity
- **Type:** Content
- **Text:** The assessment document shall discuss threats to validity across the standard
  dimensions.
- **Acceptance criteria:**
  - Data, label, evaluation (leakage), external (generalisability), implementation (bugs), and
    interpretation validity each addressed.
  - Concrete project threats named: concept drift / covariate shift on the chronological split;
    end-of-log right-censoring; the activity-matching bug.

### REQ-AD-25 — Project retrospective
- **Type:** Content
- **Text:** The phase review shall reflect on the project as a software project.
- **Acceptance criteria:**
  - Covered: what worked vs. the initial requirements, what did not, which requirements changed
    and why, technical and organisational learnings, late-discovered risks, and what would be
    done differently.
  - Reflective and professional in tone (not attributing blame).
- **Traces:** REQ-DOC-03, REQ-DOC-04

---

## Traceability Summary

| Assessment requirement | Existing requirements evidenced |
|---|---|
| REQ-AD-01 | REQ-ME-01, REQ-RE-01 |
| REQ-AD-02 | REQ-DP-03, REQ-ME-01 |
| REQ-AD-03 | REQ-DP-01, REQ-DP-02, REQ-DP-05 |
| REQ-AD-04 | REQ-DP-03, REQ-DP-06, REQ-IT-03 |
| REQ-AD-05 | REQ-IT-01, REQ-ME-06, REQ-DOC-01/02 |
| REQ-AD-06 | REQ-ME-05 |
| REQ-AD-07 | REQ-RE-01, REQ-ME-02, REQ-ME-03 |
| REQ-AD-08 | REQ-IT-01 |
| REQ-AD-10 | REQ-ME-04 |
| REQ-AD-11 | REQ-RE-01, REQ-ME-07 |
| REQ-AD-16 | REQ-RE-02 |
| REQ-AD-17 / 18 | REQ-IT-03 |
| REQ-AD-20 | REQ-IT-01, REQ-IT-04 |
| REQ-AD-25 | REQ-DOC-03, REQ-DOC-04 |

---

## Final Submission Checklist

- [ ] Task, data, labels, split, metrics, baselines, and configurations are stated.
- [ ] Main results reproducible from a script or clearly documented notebook.
- [ ] Tables accompanied by plots and interpretation.
- [ ] Tests cover the most important logic and run with one command.
- [ ] Prototype loads a trained artifact and processes a new case/prefix.
- [ ] Limitations and threats to validity explicitly discussed.
- [ ] README, architecture documentation, and requirements up to date.
- [ ] Git repository contains final code, generated figures, and instructions.

## Common Mistakes to Avoid

- No baseline, or comparing only complex models with each other.
- Random row split leaking prefixes of one case across train and test.
- Reporting validation results as final test results.
- Tables/plots without interpretation.
- Unclear target definition or silently changed thresholds.
- Evaluation that runs only on one laptop.
- Tests that do not check important logic.
- Reporting only strong points and hiding limitations.
